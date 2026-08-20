/*
    Copyright 2017 Philippe Grandclement

    This file is part of Kadath.

    Kadath is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    Kadath is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with Kadath.  If not, see <http://www.gnu.org/licenses/>.
*/

/*
 * Modifications (Celephais):
 *   2026-06-16  Modified for the Celephais tree; see
 *               PATCHES-KADATH-UPSTREAM.md and LICENSE_SOURCE_AUDIT.tsv.
 *   2026-08-04  Added the ope_1d overload driving kernels that read and write
 *               their line in place at the traversal stride.
 */

#include "For_Kadath/Array/headcpp.hpp"
#include "For_Kadath/Array/exceptions.hpp"
#include "For_Kadath/Val_domain/val_domain.hpp"
#include "For_Kadath/Scalar/scalar.hpp"
#include "For_Kadath/Base_spectral/transform_line_offsets.hpp"
#include "For_Kadath/Utilities/runtime_env.hpp"
#include "der_1d_cheb_contract.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

namespace Kadath
{
    int der_1d(int base, Array<double>& tab);

    namespace
    {
        constexpr int derivative_lane_tile = 4;
        constexpr std::size_t derivative_workspace_default_byte_cap =
            1024 * 1024;

        struct OpeDer1dOuterOffset
        {
            std::size_t line_offset = 0;
            std::size_t basis_offset = 0;
        };

        struct OpeDer1dTraversalPlan
        {
            bool ready = false;
            int line_length = 0;
            std::size_t stride = 0;
            std::vector<std::size_t> low_offsets;
            std::vector<OpeDer1dOuterOffset> outer_offsets;
        };

        struct OpeDer1dWorkspaceState
        {
            int scope_depth = 0;
            bool in_use = false;
            std::vector<int> shape;
            std::vector<OpeDer1dTraversalPlan> plans;
            std::unique_ptr<Array<double>> line_scratch;
            int scratch_length = 0;
            OpeDer1dWorkspaceStats stats;
        };

        OpeDer1dWorkspaceState& derivative_workspace()
        {
            thread_local OpeDer1dWorkspaceState state;
            return state;
        }

        bool derivative_workspace_enabled()
        {
            static const bool enabled = env_flag_enabled("OPE1D_WORKSPACE", true);
            return enabled;
        }

        bool derivative_batch_enabled()
        {
            static const bool enabled = env_flag_enabled("OPE_DER_BATCH", true);
            return enabled;
        }

        bool derivative_aosoa_enabled()
        {
            static const bool enabled =
                env_flag_enabled("OPE_DER_AOSOA", false);
            return enabled;
        }

        bool derivative_workspace_scope_active()
        {
            return derivative_workspace_enabled() &&
                   derivative_workspace().scope_depth > 0;
        }

        // Latched on first use so the per-batch call sites stop paying a
        // getenv environ scan; 0 marks "not parsed yet" (env_positive_int
        // never returns <= 0). The workspace reset hooks clear the latch so
        // unit tests that swap the variable mid-process keep observing it.
        std::size_t cached_workspace_byte_cap = 0;

        std::size_t derivative_workspace_byte_cap()
        {
            if (cached_workspace_byte_cap == 0)
                cached_workspace_byte_cap = static_cast<std::size_t>(env_positive_int(
                    "OPE1D_WORKSPACE_MAX_BYTES",
                    static_cast<int>(derivative_workspace_default_byte_cap)));
            return cached_workspace_byte_cap;
        }

        std::size_t checked_product(std::size_t lhs, std::size_t rhs)
        {
            if (lhs != 0 && rhs > std::numeric_limits<std::size_t>::max() / lhs)
                KADATH_THROW("Ope_der_1d traversal size overflow.");
            return lhs * rhs;
        }

        std::size_t retained_workspace_bytes(const OpeDer1dWorkspaceState& state)
        {
            std::size_t bytes = state.shape.capacity() * sizeof(int);
            bytes += state.plans.capacity() * sizeof(OpeDer1dTraversalPlan);
            for (const OpeDer1dTraversalPlan& plan : state.plans) {
                bytes += plan.low_offsets.capacity() * sizeof(std::size_t);
                bytes += plan.outer_offsets.capacity() * sizeof(OpeDer1dOuterOffset);
            }
            if (state.line_scratch != nullptr) {
                bytes += sizeof(Array<double>);
                bytes += state.line_scratch->get_nbr() * sizeof(double);
            }
            return bytes;
        }

        void refresh_workspace_retained_stats(OpeDer1dWorkspaceState& state)
        {
            state.stats.retained_bytes = retained_workspace_bytes(state);
            state.stats.peak_retained_bytes =
                std::max(state.stats.peak_retained_bytes, state.stats.retained_bytes);
        }

        void release_workspace_scratch(OpeDer1dWorkspaceState& state)
        {
            state.line_scratch.reset();
            state.scratch_length = 0;
        }

        void release_workspace_capacity(OpeDer1dWorkspaceState& state)
        {
            release_workspace_scratch(state);
            for (OpeDer1dTraversalPlan& plan : state.plans) {
                std::vector<std::size_t>().swap(plan.low_offsets);
                std::vector<OpeDer1dOuterOffset>().swap(plan.outer_offsets);
            }
            std::vector<OpeDer1dTraversalPlan>().swap(state.plans);
            std::vector<int>().swap(state.shape);
            state.stats.retained_bytes = 0;
        }

        bool derivative_kernel_workspace_compatible(int base, int line_length)
        {
            switch (base) {
                case CHEB:
                case CHEB_EVEN:
                case CHEB_ODD:
                case COSSIN:
                case COS:
                case COS_EVEN:
                case COS_ODD:
                case SIN:
                case SIN_EVEN:
                case SIN_ODD:
                case LEG:
                case LEG_EVEN:
                case LEG_ODD:
                    return true;
                case COSSIN_EVEN:
                case COSSIN_ODD:
                    return (line_length % 2) == 0;
                default:
                    return false;
            }
        }

        bool derivative_kernel_aosoa_compatible(int base)
        {
            switch (base) {
                case CHEB:
                case CHEB_EVEN:
                case CHEB_ODD:
                case COSSIN:
                case COS:
                case COS_EVEN:
                case COS_ODD:
                case SIN:
                case SIN_EVEN:
                case SIN_ODD:
                    return true;
                default:
                    return false;
            }
        }

        int derivative_kernel_aosoa_output_base(int base)
        {
            switch (base) {
                case CHEB: return CHEB;
                case CHEB_EVEN: return CHEB_ODD;
                case CHEB_ODD: return CHEB_EVEN;
                case COSSIN: return COSSIN;
                case COS: return SIN;
                case COS_EVEN: return SIN_EVEN;
                case COS_ODD: return SIN_ODD;
                case SIN: return COS;
                case SIN_EVEN: return COS_EVEN;
                case SIN_ODD: return COS_ODD;
                default:
                    KADATH_THROW("Ope_der_1d AoSoA kernel received an unsupported basis.");
            }
        }

        void apply_derivative_kernel_aosoa(int base,
                                           int coefficient_count,
                                           int lane_count,
                                           const double* input,
                                           double* output)
        {
            const std::size_t value_count = checked_product(
                static_cast<std::size_t>(coefficient_count),
                static_cast<std::size_t>(lane_count));
            std::fill(output, output + value_count, 0.);

            const auto offset = [lane_count](int coefficient, int lane) {
                return static_cast<std::size_t>(coefficient) *
                           static_cast<std::size_t>(lane_count) +
                       static_cast<std::size_t>(lane);
            };

            switch (base) {
                case CHEB:
                    for (int coefficient = 0; coefficient < coefficient_count;
                         ++coefficient) {
                        for (int source = coefficient + 1;
                            source < coefficient_count; source += 2) {
                            for (int lane = 0; lane < lane_count; ++lane) {
                                output[offset(coefficient, lane)] = std::fma(
                                    static_cast<double>(source),
                                    input[offset(source, lane)],
                                    output[offset(coefficient, lane)]);
                            }
                        }
                    }
                    for (std::size_t value = 0; value < value_count; ++value)
                        output[value] *= 2;
                    for (int lane = 0; lane < lane_count; ++lane)
                        output[offset(0, lane)] /= 2;
                    return;

                case CHEB_EVEN:
                    for (int coefficient = 0; coefficient < coefficient_count;
                         ++coefficient) {
                        for (int source = coefficient + 1;
                            source < coefficient_count; ++source) {
                            for (int lane = 0; lane < lane_count; ++lane) {
                                output[offset(coefficient, lane)] = std::fma(
                                    static_cast<double>(2 * source),
                                    input[offset(source, lane)],
                                    output[offset(coefficient, lane)]);
                            }
                        }
                    }
                    for (int lane = 0; lane < lane_count; ++lane)
                        output[offset(coefficient_count - 1, lane)] = 0;
                    for (std::size_t value = 0; value < value_count; ++value)
                        output[value] *= 2;
                    return;

                case CHEB_ODD:
                    for (int coefficient = 0; coefficient < coefficient_count;
                         ++coefficient) {
                        int source = coefficient;
                        // The legacy contiguous-source loop is vectorized in
                        // eight-value blocks: products are rounded before its
                        // ordered adds, while the scalar remainder contracts
                        // multiply-add. Mirror that distinction across lanes
                        // so the AoSoA path remains byte-identical.
                        const int vectorized_end =
                            derivative_arithmetic::cheb_odd_rounded_end(
                                coefficient, coefficient_count);
                        for (; source < vectorized_end; ++source) {
                            for (int lane = 0; lane < lane_count; ++lane) {
                                const double weighted_source =
                                    (2 * source + 1) *
                                    input[offset(source, lane)];
                                output[offset(coefficient, lane)] +=
                                    weighted_source;
                            }
                        }
                        for (; source < coefficient_count; ++source) {
                            for (int lane = 0; lane < lane_count; ++lane) {
                                output[offset(coefficient, lane)] = std::fma(
                                    static_cast<double>(2 * source + 1),
                                    input[offset(source, lane)],
                                    output[offset(coefficient, lane)]);
                            }
                        }
                    }
                    for (std::size_t value = 0; value < value_count; ++value)
                        output[value] *= 2;
                    for (int lane = 0; lane < lane_count; ++lane)
                        output[offset(0, lane)] /= 2;
                    return;

                case COSSIN:
                    for (int coefficient = 0;
                         coefficient < coefficient_count - 1;
                         coefficient += 2) {
                        for (int lane = 0; lane < lane_count; ++lane) {
                            output[offset(coefficient + 1, lane)] =
                                -(coefficient / 2) *
                                input[offset(coefficient, lane)];
                        }
                    }
                    for (int coefficient = 1; coefficient < coefficient_count;
                         coefficient += 2) {
                        for (int lane = 0; lane < lane_count; ++lane) {
                            output[offset(coefficient - 1, lane)] =
                                ((coefficient - 1) / 2) *
                                input[offset(coefficient, lane)];
                        }
                    }
                    for (int lane = 0; lane < lane_count; ++lane)
                        output[offset(coefficient_count - 1, lane)] = 0;
                    return;

                case COS:
                    for (int coefficient = 0; coefficient < coefficient_count;
                         ++coefficient) {
                        for (int lane = 0; lane < lane_count; ++lane) {
                            output[offset(coefficient, lane)] =
                                -coefficient * input[offset(coefficient, lane)];
                        }
                    }
                    return;

                case COS_EVEN:
                    for (int coefficient = 0; coefficient < coefficient_count;
                         ++coefficient) {
                        for (int lane = 0; lane < lane_count; ++lane) {
                            output[offset(coefficient, lane)] =
                                -2 * coefficient *
                                input[offset(coefficient, lane)];
                        }
                    }
                    for (int lane = 0; lane < lane_count; ++lane)
                        output[offset(coefficient_count - 1, lane)] = 0;
                    return;

                case COS_ODD:
                    for (int coefficient = 0; coefficient < coefficient_count;
                         ++coefficient) {
                        for (int lane = 0; lane < lane_count; ++lane) {
                            output[offset(coefficient, lane)] =
                                -(2 * coefficient + 1) *
                                input[offset(coefficient, lane)];
                        }
                    }
                    return;

                case SIN:
                    for (int coefficient = 0; coefficient < coefficient_count;
                         ++coefficient) {
                        for (int lane = 0; lane < lane_count; ++lane) {
                            output[offset(coefficient, lane)] =
                                coefficient * input[offset(coefficient, lane)];
                        }
                    }
                    return;

                case SIN_EVEN:
                    for (int coefficient = 0; coefficient < coefficient_count;
                         ++coefficient) {
                        for (int lane = 0; lane < lane_count; ++lane) {
                            output[offset(coefficient, lane)] =
                                2 * coefficient *
                                input[offset(coefficient, lane)];
                        }
                    }
                    return;

                case SIN_ODD:
                    for (int coefficient = 0; coefficient < coefficient_count;
                         ++coefficient) {
                        for (int lane = 0; lane < lane_count; ++lane) {
                            output[offset(coefficient, lane)] =
                                (2 * coefficient + 1) *
                                input[offset(coefficient, lane)];
                        }
                    }
                    for (int lane = 0; lane < lane_count; ++lane)
                        output[offset(coefficient_count - 1, lane)] = 0;
                    return;

                default:
                    KADATH_THROW("Ope_der_1d AoSoA kernel received an unsupported basis.");
            }
        }

        bool same_workspace_shape(const OpeDer1dWorkspaceState& state,
                                  const Dim_array& dimensions)
        {
            const int ndim = dimensions.get_ndim();
            if (state.shape.size() != static_cast<std::size_t>(ndim + 1) ||
                state.shape[0] != ndim)
                return false;
            for (int axis = 0; axis < ndim; ++axis) {
                if (state.shape[static_cast<std::size_t>(axis + 1)] !=
                    dimensions(axis))
                    return false;
            }
            return true;
        }

        void install_workspace_shape(OpeDer1dWorkspaceState& state,
                                     const Dim_array& dimensions)
        {
            const int ndim = dimensions.get_ndim();
            state.shape.reserve(static_cast<std::size_t>(ndim + 1));
            state.shape.push_back(ndim);
            for (int axis = 0; axis < ndim; ++axis)
                state.shape.push_back(dimensions(axis));
            state.plans.resize(static_cast<std::size_t>(ndim));
            refresh_workspace_retained_stats(state);
        }

        OpeDer1dTraversalPlan build_traversal_plan(const Dim_array& dimensions,
                                                   int var)
        {
            const int ndim = dimensions.get_ndim();
            OpeDer1dTraversalPlan plan;
            plan.line_length = dimensions(var);

            std::size_t before = 1;
            for (int axis = var + 1; axis < ndim; ++axis)
                before = checked_product(before, static_cast<std::size_t>(dimensions(axis)));
            plan.stride = before;

            std::size_t after = 1;
            for (int axis = 0; axis < var; ++axis)
                after = checked_product(after, static_cast<std::size_t>(dimensions(axis)));

            std::vector<std::size_t> coordinates(static_cast<std::size_t>(ndim), 0);
            plan.outer_offsets.reserve(before);
            for (std::size_t ordinal = 0; ordinal < before; ++ordinal) {
                std::size_t quotient = ordinal;
                for (int axis = var + 1; axis < ndim; ++axis) {
                    const std::size_t extent = static_cast<std::size_t>(dimensions(axis));
                    coordinates[static_cast<std::size_t>(axis)] = quotient % extent;
                    quotient /= extent;
                }

                std::size_t high_flat = 0;
                for (int axis = var + 1; axis < ndim; ++axis) {
                    high_flat = checked_product(
                        high_flat, static_cast<std::size_t>(dimensions(axis)));
                    high_flat += coordinates[static_cast<std::size_t>(axis)];
                }
                plan.outer_offsets.push_back({high_flat, high_flat});
            }

            const std::size_t low_scale = checked_product(
                static_cast<std::size_t>(plan.line_length), before);
            plan.low_offsets.reserve(after);
            for (std::size_t ordinal = 0; ordinal < after; ++ordinal) {
                std::size_t quotient = ordinal;
                for (int axis = 0; axis < var; ++axis) {
                    const std::size_t extent = static_cast<std::size_t>(dimensions(axis));
                    coordinates[static_cast<std::size_t>(axis)] = quotient % extent;
                    quotient /= extent;
                }

                std::size_t low_flat = 0;
                for (int axis = 0; axis < var; ++axis) {
                    low_flat = checked_product(
                        low_flat, static_cast<std::size_t>(dimensions(axis)));
                    low_flat += coordinates[static_cast<std::size_t>(axis)];
                }
                plan.low_offsets.push_back(checked_product(low_flat, low_scale));
            }
            plan.ready = true;
            return plan;
        }

        std::size_t nested_plan_bytes(const OpeDer1dTraversalPlan& plan)
        {
            std::size_t bytes =
                plan.low_offsets.capacity() * sizeof(std::size_t);
            bytes += plan.outer_offsets.capacity() * sizeof(OpeDer1dOuterOffset);
            return bytes;
        }

        const OpeDer1dTraversalPlan* cached_traversal_plan(
            const Dim_array& dimensions, int var, OpeDer1dTraversalPlan& local)
        {
            OpeDer1dWorkspaceState& state = derivative_workspace();
            const std::size_t byte_cap = derivative_workspace_byte_cap();
            if (!same_workspace_shape(state, dimensions)) {
                // Bound retention to one coefficient shape. Drop every nested
                // vector before acquiring storage for the replacement shape.
                release_workspace_capacity(state);
                install_workspace_shape(state, dimensions);
            }

            OpeDer1dTraversalPlan& plan = state.plans[static_cast<std::size_t>(var)];
            if (plan.ready) {
                ++state.stats.plan_hits;
                return &plan;
            }

            ++state.stats.plan_misses;
            local = build_traversal_plan(dimensions, var);
            const std::size_t retained = retained_workspace_bytes(state);
            const std::size_t candidate_bytes = nested_plan_bytes(local);
            if (retained <= byte_cap && candidate_bytes <= byte_cap - retained) {
                plan = std::move(local);
                refresh_workspace_retained_stats(state);
                return &plan;
            }

            // Oversized plans remain lease-local and are released at the end
            // of this call instead of living for the whole assembly scope.
            ++state.stats.plan_fallbacks;
            if (retained > byte_cap)
                release_workspace_capacity(state);
            return &local;
        }

        class OpeDer1dTraversalLease
        {
          public:
            OpeDer1dTraversalLease(const Dim_array& dimensions, int var)
            {
                OpeDer1dWorkspaceState& state = derivative_workspace();
                if (derivative_workspace_enabled() && state.scope_depth > 0 &&
                    !state.in_use) {
                    state.in_use = true;
                    shared_ = true;
                    try {
                        if (state.scratch_length != 0 &&
                            state.scratch_length != dimensions(var)) {
                            release_workspace_scratch(state);
                            refresh_workspace_retained_stats(state);
                        }
                        plan_ = cached_traversal_plan(dimensions, var, local_);
                    } catch (...) {
                        state.in_use = false;
                        shared_ = false;
                        throw;
                    }
                } else {
                    ++state.stats.plan_misses;
                    ++state.stats.plan_fallbacks;
                    local_ = build_traversal_plan(dimensions, var);
                    plan_ = &local_;
                }
            }

            ~OpeDer1dTraversalLease()
            {
                if (shared_)
                    derivative_workspace().in_use = false;
            }

            const OpeDer1dTraversalPlan& get() const { return *plan_; }

          private:
            OpeDer1dTraversalPlan local_;
            const OpeDer1dTraversalPlan* plan_ = nullptr;
            bool shared_ = false;
        };

        class OpeDer1dScratchLease
        {
          public:
            explicit OpeDer1dScratchLease(int line_length)
            {
                OpeDer1dWorkspaceState& state = derivative_workspace();
                if (derivative_workspace_enabled() && state.scope_depth > 0 &&
                    state.in_use && try_shared(state, line_length))
                    return;

                ++state.stats.scratch_misses;
                local_ = std::make_unique<Array<double>>(line_length);
                scratch_ = local_.get();
            }

            Array<double>& get() const { return *scratch_; }

          private:
            bool try_shared(OpeDer1dWorkspaceState& state, int line_length)
            {
                if (state.scratch_length != 0 &&
                    state.scratch_length != line_length)
                    return false;

                const bool missing = state.line_scratch == nullptr;
                const std::size_t per_scratch =
                    sizeof(Array<double>) +
                    checked_product(static_cast<std::size_t>(line_length),
                                    sizeof(double));
                const std::size_t additional = missing ? per_scratch : 0;
                const std::size_t retained = retained_workspace_bytes(state);
                const std::size_t byte_cap = derivative_workspace_byte_cap();
                if (retained > byte_cap || additional > byte_cap - retained) {
                    ++state.stats.plan_fallbacks;
                    return false;
                }

                if (missing) {
                    ++state.stats.scratch_misses;
                    state.line_scratch =
                        std::make_unique<Array<double>>(line_length);
                } else {
                    ++state.stats.scratch_hits;
                }
                scratch_ = state.line_scratch.get();
                state.scratch_length = line_length;
                refresh_workspace_retained_stats(state);
                return true;
            }

            std::unique_ptr<Array<double>> local_;
            Array<double>* scratch_ = nullptr;
        };

        bool same_dimensions(const Dim_array& lhs, const Dim_array& rhs)
        {
            if (lhs.get_ndim() != rhs.get_ndim())
                return false;
            for (int axis = 0; axis < lhs.get_ndim(); ++axis) {
                if (lhs(axis) != rhs(axis))
                    return false;
            }
            return true;
        }

        bool derivative_lane_workspace_compatible(
            int var, const Base_spectral& base, const Dim_array& dimensions)
        {
            const Array<int>* axis_bases = base.get_base_1d(var);
            const int line_length = dimensions(var);
            for (std::size_t i = 0; i < axis_bases->get_nbr(); ++i) {
                if (!derivative_kernel_workspace_compatible(
                        axis_bases->get_data()[i], line_length))
                    return false;
            }
            return true;
        }
    } // namespace

    OpeDer1dAssemblyWorkspaceScope::OpeDer1dAssemblyWorkspaceScope()
    {
        ++derivative_workspace().scope_depth;
    }

    OpeDer1dAssemblyWorkspaceScope::~OpeDer1dAssemblyWorkspaceScope()
    {
        OpeDer1dWorkspaceState& state = derivative_workspace();
        assert(state.scope_depth > 0);
        --state.scope_depth;
        if (state.scope_depth == 0) {
            assert(!state.in_use);
            release_workspace_capacity(state);
        }
    }

    void reset_ope_der_1d_workspace()
    {
        OpeDer1dWorkspaceState& state = derivative_workspace();
        if (state.in_use)
            KADATH_THROW("Cannot reset an in-use Ope_der_1d traversal workspace.");
        cached_workspace_byte_cap = 0;
        release_workspace_capacity(state);
    }

    void reset_ope_der_1d_workspace_stats()
    {
        OpeDer1dWorkspaceState& state = derivative_workspace();
        const std::size_t retained = retained_workspace_bytes(state);
        cached_workspace_byte_cap = 0;
        state.stats = {};
        state.stats.retained_bytes = retained;
        state.stats.peak_retained_bytes = retained;
        state.stats.retained_byte_cap = derivative_workspace_byte_cap();
    }

    OpeDer1dWorkspaceStats ope_der_1d_workspace_stats()
    {
        OpeDer1dWorkspaceState& state = derivative_workspace();
        state.stats.retained_bytes = retained_workspace_bytes(state);
        state.stats.retained_byte_cap = derivative_workspace_byte_cap();
        return state.stats;
    }

    Array<double> Base_spectral::ope_1d(int (*func)(int, Array<double>&), int var, const Array<double>& in,
                                        Base_spectral& base_out) const
    {

        Array<double> res(in.get_dimensions());

        int after = 1;
        for (int i = 0; i < var; i++)
            after *= in.get_size(i);

        int before = 1;
        for (int i = var + 1; i < ndim; i++)
            before *= in.get_size(i);

        int nbr = in.get_size(var);

        // in and res share dimensions, so one carried flat offset addresses
        // both. Replaces the per-line Horner rebuild and the two Index walks;
        // the traversal order is the one Index::inc() produced (see
        // transform_line_offsets.hpp). Bit-identical (COO byte-hash gate).
        Transform_line_offsets outer_line(in.get_dimensions(), var + 1, ndim, 1, 1);
        Transform_line_offsets inner_line(in.get_dimensions(), 0, var, nbr * before, nbr * before);

        // The basis of axis `var` is stored over the axes above it, so its flat
        // slot is the outer offset (see Base_spectral::allocate) and is constant
        // across the whole inner loop.
        assert(bases_1d[var]->get_nbr() == static_cast<std::size_t>(before));
        const int* base_data = bases_1d[var]->get_data();
        int* base_out_data = base_out.bases_1d[var]->get_data();

        Array<double> tab_1d(nbr);
        const double* in_data = in.get_data();
        double* res_data = res.get_data();

        // Loop on dimensions before
        for (int i = 0; i < before; i++) {

            const int line_start = outer_line.in_offset();
            // On get la base

            int base = base_data[line_start];
            // Loop on dimensions after :
            for (int j = 0; j < after; j++) {

                // The 1D line advances by the constant stride `before` (= the
                // dim-`var` flat stride; the var digit never overflows, k < nbr).
                const int line_pos = line_start + inner_line.in_offset();

                int read_pos = line_pos;
                for (int k = 0; k < nbr; k++) {
                    tab_1d.set(k) = in_data[read_pos];
                    read_pos += before;
                }
                // Transformation
                base_out_data[line_start] = func(base, tab_1d);

                int write_pos = line_pos;
                for (int k = 0; k < nbr; k++) {
                    res_data[write_pos] = tab_1d(k);
                    write_pos += before;
                }

                inner_line.advance();
            }
            outer_line.advance();
        }

        return res;
    }

    Array<double> Base_spectral::ope_1d(int (*func)(int, const double*, double*, int, int), int var,
                                        const Array<double>& in, Base_spectral& base_out) const
    {

        Array<double> res(in.get_dimensions());

        int after = 1;
        for (int i = 0; i < var; i++)
            after *= in.get_size(i);

        int before = 1;
        for (int i = var + 1; i < ndim; i++)
            before *= in.get_size(i);

        int nbr = in.get_size(var);

        // Same traversal as the gathering overload above, line for line: the
        // kernel just reads `in` and writes `res` where the gather and the
        // scatter used to copy through a contiguous buffer.
        Transform_line_offsets outer_line(in.get_dimensions(), var + 1, ndim, 1, 1);
        Transform_line_offsets inner_line(in.get_dimensions(), 0, var, nbr * before, nbr * before);

        assert(bases_1d[var]->get_nbr() == static_cast<std::size_t>(before));
        const int* base_data = bases_1d[var]->get_data();
        int* base_out_data = base_out.bases_1d[var]->get_data();

        const double* in_data = in.get_data();
        double* res_data = res.get_data();

        // Loop on dimensions before
        for (int i = 0; i < before; i++) {

            const int line_start = outer_line.in_offset();
            // On get la base

            int base = base_data[line_start];
            // Loop on dimensions after :
            for (int j = 0; j < after; j++) {

                const int line_pos = line_start + inner_line.in_offset();

                // Transformation
                base_out_data[line_start] =
                    func(base, in_data + line_pos, res_data + line_pos, nbr, before);

                inner_line.advance();
            }
            outer_line.advance();
        }

        return res;
    }

    void Base_spectral::validate_ope_der_1d_batch(
        int var,
        const Base_spectral* const* bases_in,
        const Array<double>* const* inputs,
        Base_spectral* const* bases_out,
        Array<double>* const* outputs,
        int lane_count)
    {
        if (lane_count < 0)
            KADATH_THROW("Ope_der_1d batch lane count must not be negative.");
        if (lane_count == 0)
            return;
        if (bases_in == nullptr || inputs == nullptr || bases_out == nullptr ||
            outputs == nullptr)
            KADATH_THROW("Ope_der_1d batch pointer arrays must not be null.");

        for (int lane = 0; lane < lane_count; ++lane) {
            if (bases_in[lane] == nullptr || inputs[lane] == nullptr ||
                bases_out[lane] == nullptr || outputs[lane] == nullptr)
                KADATH_THROW("Ope_der_1d batch lane pointers must not be null.");
        }

        const Dim_array& reference_dimensions = inputs[0]->get_dimensions();
        const int reference_ndim = reference_dimensions.get_ndim();
        if (var < 0 || var >= reference_ndim)
            KADATH_THROW("Ope_der_1d variable is outside the coefficient dimensions.");
        for (int axis = 0; axis < reference_ndim; ++axis) {
            if (reference_dimensions(axis) <= 0)
                KADATH_THROW("Ope_der_1d requires positive coefficient dimensions.");
        }

        std::size_t expected_basis_count = 1;
        for (int axis = var + 1; axis < reference_ndim; ++axis) {
            expected_basis_count = checked_product(
                expected_basis_count,
                static_cast<std::size_t>(reference_dimensions(axis)));
        }

        for (int lane = 0; lane < lane_count; ++lane) {
            const Base_spectral& base_in = *bases_in[lane];
            Base_spectral& base_out = *bases_out[lane];
            const Array<double>& input = *inputs[lane];
            Array<double>& output = *outputs[lane];

            if (base_in.ndim != reference_ndim || base_out.ndim != reference_ndim)
                KADATH_THROW("Ope_der_1d bases and coefficients have incompatible dimensions.");
            if (!base_in.def || !base_out.def ||
                base_in.bases_1d[static_cast<std::size_t>(var)] == nullptr ||
                base_out.bases_1d[static_cast<std::size_t>(var)] == nullptr)
                KADATH_THROW("Ope_der_1d requires defined input and output bases.");
            if (!same_dimensions(input.get_dimensions(), reference_dimensions) ||
                !same_dimensions(output.get_dimensions(), reference_dimensions))
                KADATH_THROW("Ope_der_1d batch lanes must have identical coefficient shapes.");
            if (input.get_data() == nullptr || output.get_data() == nullptr)
                KADATH_THROW("Ope_der_1d input and output buffers must be allocated.");
            if (base_in.bases_1d[static_cast<std::size_t>(var)]->get_nbr() !=
                    expected_basis_count ||
                base_out.bases_1d[static_cast<std::size_t>(var)]->get_nbr() !=
                    expected_basis_count)
                KADATH_THROW("Ope_der_1d basis shape does not match coefficient traversal.");

            const int* basis_values =
                base_in.bases_1d[static_cast<std::size_t>(var)]->get_data();
            for (std::size_t i = 0; i < expected_basis_count; ++i) {
                if (basis_values[i] < 0 || basis_values[i] >= NBR_MAX_BASE)
                    KADATH_THROW("Ope_der_1d basis code is outside the legacy dispatch table.");
            }
        }

        // Result objects are persistent caller storage. Reject any alias that
        // could turn a direct write into an in-place transform or couple two
        // lane results. Input bases may be shared, but output bases may not.
        for (int lane = 0; lane < lane_count; ++lane) {
            for (int other = 0; other < lane_count; ++other) {
                if (outputs[lane]->get_data() == inputs[other]->get_data())
                    KADATH_THROW("Ope_der_1d input and output buffers must not alias.");
                if (lane != other && bases_out[lane] == bases_in[other])
                    KADATH_THROW("Ope_der_1d output bases must not alias another lane input base.");
            }
            for (int other = lane + 1; other < lane_count; ++other) {
                if (outputs[lane]->get_data() == outputs[other]->get_data())
                    KADATH_THROW("Ope_der_1d output buffers must be distinct per lane.");
                if (bases_out[lane] == bases_out[other])
                    KADATH_THROW("Ope_der_1d output bases must be distinct per lane.");
            }
        }
    }

    void Base_spectral::ope_der_1d_batch_workspace(
        int var,
        const Base_spectral* const* bases_in,
        const Array<double>* const* inputs,
        Base_spectral* const* bases_out,
        Array<double>* const* outputs,
        int lane_count)
    {
        if (lane_count == 0)
            return;

        OpeDer1dWorkspaceState& workspace = derivative_workspace();
        ++workspace.stats.calls;
        workspace.stats.lanes += static_cast<unsigned long long>(lane_count);

        OpeDer1dTraversalLease lease(inputs[0]->get_dimensions(), var);
        const OpeDer1dTraversalPlan& plan = lease.get();
        OpeDer1dScratchLease scratch_lease(plan.line_length);

        for (int tile_begin = 0; tile_begin < lane_count;
             tile_begin += derivative_lane_tile) {
            const int tile_end =
                std::min(tile_begin + derivative_lane_tile, lane_count);
            for (const OpeDer1dOuterOffset& outer : plan.outer_offsets) {
                for (std::size_t low_offset : plan.low_offsets) {
                    const std::size_t line_start = low_offset + outer.line_offset;
                    for (int lane = tile_begin; lane < tile_end; ++lane) {
                        Array<double>& line = scratch_lease.get();
                        const double* input = inputs[lane]->get_data();
                        for (int coefficient = 0;
                             coefficient < plan.line_length; ++coefficient) {
                            line.set(coefficient) =
                                input[line_start +
                                      static_cast<std::size_t>(coefficient) *
                                          plan.stride];
                        }

                        const int base =
                            bases_in[lane]
                                ->bases_1d[static_cast<std::size_t>(var)]
                                ->get_data()[outer.basis_offset];
                        const int output_base = der_1d(base, line);
                        bases_out[lane]
                            ->bases_1d[static_cast<std::size_t>(var)]
                            ->get_data()[outer.basis_offset] = output_base;

                        double* output = outputs[lane]->get_data();
                        for (int coefficient = 0;
                             coefficient < plan.line_length; ++coefficient) {
                            output[line_start +
                                   static_cast<std::size_t>(coefficient) *
                                       plan.stride] = line(coefficient);
                        }
                    }
                }
            }
        }
    }

    void Base_spectral::ope_der_1d_batch_aosoa(
        int var,
        const Base_spectral* const* bases_in,
        const Array<double>* const* inputs,
        Base_spectral* const* bases_out,
        Array<double>* const* outputs,
        int lane_count)
    {
        if (lane_count == 0)
            return;

        OpeDer1dWorkspaceState& workspace = derivative_workspace();
        ++workspace.stats.calls;
        workspace.stats.lanes += static_cast<unsigned long long>(lane_count);
        ++workspace.stats.aosoa_calls;
        workspace.stats.aosoa_lanes +=
            static_cast<unsigned long long>(lane_count);

        OpeDer1dTraversalLease lease(inputs[0]->get_dimensions(), var);
        const OpeDer1dTraversalPlan& plan = lease.get();
        const std::size_t scratch_size = checked_product(
            static_cast<std::size_t>(plan.line_length),
            static_cast<std::size_t>(lane_count));
        std::vector<double> gathered(scratch_size);
        std::vector<double> transformed(scratch_size);

        for (const OpeDer1dOuterOffset& outer : plan.outer_offsets) {
            const int base =
                bases_in[0]
                    ->bases_1d[static_cast<std::size_t>(var)]
                    ->get_data()[outer.basis_offset];
            const int output_base = derivative_kernel_aosoa_output_base(base);
            for (int lane = 0; lane < lane_count; ++lane) {
                bases_out[lane]
                    ->bases_1d[static_cast<std::size_t>(var)]
                    ->get_data()[outer.basis_offset] = output_base;
            }

            for (std::size_t low_offset : plan.low_offsets) {
                const std::size_t line_start = low_offset + outer.line_offset;
                for (int coefficient = 0; coefficient < plan.line_length;
                     ++coefficient) {
                    const std::size_t coefficient_offset =
                        static_cast<std::size_t>(coefficient) *
                        static_cast<std::size_t>(lane_count);
                    const std::size_t source_offset =
                        line_start + static_cast<std::size_t>(coefficient) *
                                         plan.stride;
                    for (int lane = 0; lane < lane_count; ++lane) {
                        gathered[coefficient_offset +
                                 static_cast<std::size_t>(lane)] =
                            inputs[lane]->get_data()[source_offset];
                    }
                }

                apply_derivative_kernel_aosoa(
                    base, plan.line_length, lane_count, gathered.data(),
                    transformed.data());

                for (int coefficient = 0; coefficient < plan.line_length;
                     ++coefficient) {
                    const std::size_t coefficient_offset =
                        static_cast<std::size_t>(coefficient) *
                        static_cast<std::size_t>(lane_count);
                    const std::size_t target_offset =
                        line_start + static_cast<std::size_t>(coefficient) *
                                         plan.stride;
                    for (int lane = 0; lane < lane_count; ++lane) {
                        outputs[lane]->get_data()[target_offset] =
                            transformed[coefficient_offset +
                                        static_cast<std::size_t>(lane)];
                    }
                }
            }
        }
    }

    Array<double> Base_spectral::ope_der_1d(int var,
                                             const Array<double>& in,
                                             Base_spectral& base_out) const
    {
        if (!derivative_workspace_scope_active()) {
            // Preserve the scalar wrapper's validation/refusal contract without
            // constructing the full-sized output used only by the workspace path.
            const Dim_array& dimensions = in.get_dimensions();
            const int input_ndim = dimensions.get_ndim();
            if (var < 0 || var >= input_ndim)
                KADATH_THROW("Ope_der_1d variable is outside the coefficient dimensions.");
            for (int axis = 0; axis < input_ndim; ++axis) {
                if (dimensions(axis) <= 0)
                    KADATH_THROW("Ope_der_1d requires positive coefficient dimensions.");
            }
            if (ndim != input_ndim || base_out.ndim != input_ndim)
                KADATH_THROW("Ope_der_1d bases and coefficients have incompatible dimensions.");
            if (!def || !base_out.def ||
                bases_1d[static_cast<std::size_t>(var)] == nullptr ||
                base_out.bases_1d[static_cast<std::size_t>(var)] == nullptr)
                KADATH_THROW("Ope_der_1d requires defined input and output bases.");
            if (in.get_data() == nullptr)
                KADATH_THROW("Ope_der_1d input and output buffers must be allocated.");

            std::size_t expected_basis_count = 1;
            for (int axis = var + 1; axis < input_ndim; ++axis) {
                expected_basis_count = checked_product(
                    expected_basis_count,
                    static_cast<std::size_t>(dimensions(axis)));
            }
            if (bases_1d[static_cast<std::size_t>(var)]->get_nbr() !=
                    expected_basis_count ||
                base_out.bases_1d[static_cast<std::size_t>(var)]->get_nbr() !=
                    expected_basis_count)
                KADATH_THROW("Ope_der_1d basis shape does not match coefficient traversal.");

            const int* basis_values =
                bases_1d[static_cast<std::size_t>(var)]->get_data();
            for (std::size_t i = 0; i < expected_basis_count; ++i) {
                if (basis_values[i] < 0 || basis_values[i] >= NBR_MAX_BASE)
                    KADATH_THROW("Ope_der_1d basis code is outside the legacy dispatch table.");
            }
            return ope_1d(der_1d, var, in, base_out);
        }

        Array<double> result(in.get_dimensions());
        const Base_spectral* bases_in[1] = {this};
        const Array<double>* inputs[1] = {&in};
        Base_spectral* bases_out[1] = {&base_out};
        Array<double>* outputs[1] = {&result};
        validate_ope_der_1d_batch(var, bases_in, inputs, bases_out, outputs, 1);

        if (!derivative_lane_workspace_compatible(var, *this,
                                                  in.get_dimensions()))
            return ope_1d(der_1d, var, in, base_out);

        ope_der_1d_batch_workspace(var, bases_in, inputs, bases_out, outputs, 1);
        return result;
    }

    void Base_spectral::ope_der_1d_batch(
        int var,
        const Base_spectral* const* bases_in,
        const Array<double>* const* inputs,
        Base_spectral* const* bases_out,
        Array<double>* const* outputs,
        int lane_count)
    {
        validate_ope_der_1d_batch(var, bases_in, inputs, bases_out, outputs,
                                  lane_count);
        if (lane_count == 0)
            return;

        if (derivative_workspace_scope_active() && derivative_batch_enabled()) {
            if (derivative_aosoa_enabled()) {
                bool aosoa_compatible = true;
                const Array<int>& reference_bases =
                    *bases_in[0]->bases_1d[static_cast<std::size_t>(var)];
                for (std::size_t basis_slot = 0;
                     basis_slot < reference_bases.get_nbr(); ++basis_slot) {
                    const int reference_basis =
                        reference_bases.get_data()[basis_slot];
                    if (!derivative_kernel_aosoa_compatible(reference_basis)) {
                        aosoa_compatible = false;
                        break;
                    }
                    for (int lane = 1; lane < lane_count; ++lane) {
                        const Array<int>& lane_bases =
                            *bases_in[lane]
                                 ->bases_1d[static_cast<std::size_t>(var)];
                        if (lane_bases.get_data()[basis_slot] !=
                            reference_basis) {
                            aosoa_compatible = false;
                            break;
                        }
                    }
                    if (!aosoa_compatible)
                        break;
                }

                if (aosoa_compatible) {
                    ope_der_1d_batch_aosoa(var, bases_in, inputs, bases_out,
                                           outputs, lane_count);
                    return;
                }
                ++derivative_workspace().stats.aosoa_fallbacks;
            }

            std::array<const Base_spectral*, derivative_lane_tile>
                compatible_bases_in{};
            std::array<const Array<double>*, derivative_lane_tile>
                compatible_inputs{};
            std::array<Base_spectral*, derivative_lane_tile>
                compatible_bases_out{};
            std::array<Array<double>*, derivative_lane_tile>
                compatible_outputs{};
            int compatible_count = 0;

            const auto flush_compatible = [&]() {
                if (compatible_count == 0)
                    return;
                ope_der_1d_batch_workspace(
                    var, compatible_bases_in.data(), compatible_inputs.data(),
                    compatible_bases_out.data(), compatible_outputs.data(),
                    compatible_count);
                compatible_count = 0;
            };

            for (int lane = 0; lane < lane_count; ++lane) {
                if (derivative_lane_workspace_compatible(
                        var, *bases_in[lane], inputs[lane]->get_dimensions())) {
                    const std::size_t slot =
                        static_cast<std::size_t>(compatible_count++);
                    compatible_bases_in[slot] = bases_in[lane];
                    compatible_inputs[slot] = inputs[lane];
                    compatible_bases_out[slot] = bases_out[lane];
                    compatible_outputs[slot] = outputs[lane];
                    if (compatible_count == derivative_lane_tile)
                        flush_compatible();
                } else {
                    flush_compatible();
                    Array<double> lane_result = bases_in[lane]->ope_1d(
                        der_1d, var, *inputs[lane], *bases_out[lane]);
                    *outputs[lane] = std::move(lane_result);
                }
            }
            flush_compatible();
            return;
        }

        // Diagnostic fallback: preserve caller-owned output objects while
        // exercising the independently gated scalar path lane by lane.
        for (int lane = 0; lane < lane_count; ++lane) {
            Array<double> lane_result =
                bases_in[lane]->ope_1d(der_1d, var, *inputs[lane],
                                       *bases_out[lane]);
            *outputs[lane] = std::move(lane_result);
        }
    }

} // namespace Kadath
