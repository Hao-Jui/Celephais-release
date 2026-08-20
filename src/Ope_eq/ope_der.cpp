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

#include "For_Kadath/Ope_eq/ope_eq.hpp"
#include "For_Kadath/Array/exceptions.hpp"
#include "For_Kadath/Scalar/scalar.hpp"
#include "For_Kadath/Tensor/tensor.hpp"
#include "For_Kadath/Metric/metric.hpp"
#include "For_Kadath/System_of_eqs/system_of_eqs.hpp"

#include "mpi.h"

#include <array>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <optional>
#include <unordered_map>

namespace Kadath
{
    namespace
    {
        bool read_env_flag(const char* name)
        {
            const char* env = std::getenv(name);
            return env != nullptr && env[0] != '\0' && env[0] != '0';
        }

        bool ope_der_cache_probe_env()
        {
            static const bool enabled = read_env_flag("OPE_DER_CACHE_PROBE");
            return enabled;
        }

        // OPE_DER_CACHE: cached-primal Ope_der lane (v1 + v2a).
        // BROKEN on JFNK-MUMPS solver — produces a J that is bit-equivalent
        // at iter-1 printed precision but FP-perturbed by ~1e-12, blowing
        // GMRES iter count 20x and adding +85% to BNS G0 full-convergence
        // wall vs baseline. Do not enable until the Val_domain coef/repr
        // deep-copy FP-symmetry issue is fixed. See
        // projects/matrix-free/evidence/wlane-cached-primal-ope-der-v2a-cart-con-nosum-20260525T023500Z/verdict.md
        bool ope_der_cache_env()
        {
            static const bool enabled = read_env_flag("OPE_DER_CACHE");
            return enabled;
        }

        using ProbeCache =
            std::unordered_map<const Ope_der*, std::unique_ptr<Term_eq>>;

        ProbeCache& probe_cache()
        {
            thread_local ProbeCache cache;
            return cache;
        }

        // Correct cache: stores result.val_t deep copy. On hit, dispatches to
        // Metric_flat::derive_with_cached_value which reuses the cached val
        // and rebuilds derivative lanes.
        using PrimalCache =
            std::unordered_map<const Ope_der*, std::unique_ptr<Tensor>>;

        PrimalCache& primal_cache()
        {
            thread_local PrimalCache cache;
            return cache;
        }
    }

    namespace
    {
        bool& jacobian_scope_flag()
        {
            thread_local bool active = false;
            return active;
        }
    }

    OpeDerCacheJacobianScope::OpeDerCacheJacobianScope()
    {
        jacobian_scope_flag() = true;
    }

    OpeDerCacheJacobianScope::~OpeDerCacheJacobianScope()
    {
        jacobian_scope_flag() = false;
        if (ope_der_cache_env())
            primal_cache().clear();
    }

    bool ope_der_cache_jacobian_scope_active() { return jacobian_scope_flag(); }

    namespace
    {
        bool ope_der_dispatch_census_env()
        {
            static const bool enabled = read_env_flag("OPE_DER_DISPATCH_CENSUS");
            return enabled;
        }

        struct DispatchCensusEntry
        {
            long long count = 0;
            double seconds = 0.0;
        };

        std::array<DispatchCensusEntry,
                   static_cast<std::size_t>(OpeDerDispatchClass::Count)>&
        dispatch_census_table()
        {
            thread_local std::array<DispatchCensusEntry,
                                    static_cast<std::size_t>(OpeDerDispatchClass::Count)> table{};
            return table;
        }

        const char* dispatch_class_label(OpeDerDispatchClass cls)
        {
            switch (cls) {
                case OpeDerDispatchClass::Cart_Cov_NoSum:     return "cart/COV/no_sum*";
                case OpeDerDispatchClass::Cart_Con_NoSum:     return "cart/CON/no_sum";
                case OpeDerDispatchClass::Cart_Cov_Sum:       return "cart/COV/sum";
                case OpeDerDispatchClass::Cart_Con_Sum:       return "cart/CON/sum";
                case OpeDerDispatchClass::Cart_AdaptedBypass: return "cart/adapted_bypass";
                case OpeDerDispatchClass::Spher_Cov_NoSum:    return "spher/COV/no_sum";
                case OpeDerDispatchClass::Spher_Con_NoSum:    return "spher/CON/no_sum";
                case OpeDerDispatchClass::Spher_Sum:          return "spher/sum";
                case OpeDerDispatchClass::Mtz_Cov_NoSum:      return "mtz/COV/no_sum";
                case OpeDerDispatchClass::Mtz_Con_NoSum:      return "mtz/CON/no_sum";
                case OpeDerDispatchClass::Mtz_Sum:            return "mtz/sum";
                case OpeDerDispatchClass::NonFlatMetric:      return "non_flat_metric";
                case OpeDerDispatchClass::UnknownBasis:       return "unknown_basis";
                case OpeDerDispatchClass::Count:              return "<count_sentinel>";
            }
            return "<invalid>";
        }
    }

    bool ope_der_dispatch_census_enabled() { return ope_der_dispatch_census_env(); }

    void ope_der_dispatch_census_record(OpeDerDispatchClass cls, double elapsed_seconds)
    {
        const auto index = static_cast<std::size_t>(cls);
        auto& table = dispatch_census_table();
        if (index >= table.size())
            return;
        ++table[index].count;
        table[index].seconds += elapsed_seconds;
    }

    void ope_der_dispatch_census_dump()
    {
        if (!ope_der_dispatch_census_env())
            return;
        int rank = 0;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        if (rank != 0)
            return;
        const auto& table = dispatch_census_table();
        long long total_count = 0;
        double total_seconds = 0.0;
        for (const auto& entry : table) {
            total_count += entry.count;
            total_seconds += entry.seconds;
        }
        std::cout << "Ope_der dispatch census (rank 0, * = cached by v1)\n";
        std::cout << "  total_calls=" << total_count
                  << "  total_seconds=" << total_seconds << "\n";
        for (std::size_t i = 0; i < table.size(); ++i) {
            if (table[i].count == 0 && table[i].seconds == 0.0)
                continue;
            const auto cls = static_cast<OpeDerDispatchClass>(i);
            const double frac = total_seconds > 0.0
                                    ? 100.0 * table[i].seconds / total_seconds
                                    : 0.0;
            std::cout << "  - " << dispatch_class_label(cls)
                      << "  calls=" << table[i].count
                      << "  seconds=" << table[i].seconds
                      << "  (" << frac << "%)\n";
        }
    }

    void ope_der_dispatch_census_reset()
    {
        for (auto& entry : dispatch_census_table())
            entry = DispatchCensusEntry{};
    }

    bool ope_der_cache_probe_enabled() { return ope_der_cache_probe_env(); }

    void ope_der_cache_probe_reset()
    {
        if (ope_der_cache_probe_env())
            probe_cache().clear();
        if (ope_der_cache_env())
            primal_cache().clear();
    }

    Ope_der::Ope_der(const System_of_eqs* zesys, int tt, char ind, Ope_eq* target)
        : Ope_eq(zesys, target->get_dom(), 1), type_der(tt), ind_der(ind)
    {

        assert((tt == COV) || (tt == CON));
        parts[0].reset(target);
    }

    Ope_der::~Ope_der() {}

    Term_eq Ope_der::action() const
    {
        ScopedOpeActionProfile ope_action_profile_scope(*this);

        if (ope_der_cache_probe_env()) {
            auto& cache = probe_cache();
            auto it = cache.find(this);
            if (it != cache.end())
                return Term_eq(*it->second);
            std::optional<Term_eq> target_storage;
            const Term_eq& target = parts[0]->action_operand(target_storage);
            if (target.type_data != TERM_T) {
                KADATH_THROW("Ope_der only defined with respect for a tensor");
            }
            Term_eq result(syst->get_met()->derive(type_der, ind_der, target));
            cache.emplace(this, std::make_unique<Term_eq>(result));
            return result;
        }

        // Cached-primal lane: cache the value side of derive() and reuse it
        // bit-identically across columns within the same Jacobian build.
        // v1 engaged only COV/no_sum. v2a extends to CON/no_sum: the cached
        // val_t already encodes the post-manipulate_ind (CON) index_type, and
        // the dispatch path in Metric_flat::derive_partial_*_with_cached_value
        // flips only the freshly-built derivative lanes via
        // flip_derivative_index_type_only. Inner summation is still gated
        // out by the expected-valence check below.
        if (ope_der_cache_env() && ope_der_cache_jacobian_scope_active()) {
            const auto* flat_metric = dynamic_cast<const Metric_flat*>(syst->get_met());
            if (flat_metric != nullptr) {
                std::optional<Term_eq> target_storage;
                const Term_eq& target = parts[0]->action_operand(target_storage);
                if (target.type_data != TERM_T) {
                    KADATH_THROW("Ope_der only defined with respect for a tensor");
                }
                auto& cache = primal_cache();
                auto it = cache.find(this);
                if (it != cache.end()) {
                    return flat_metric->derive_with_cached_value(type_der, ind_der, target, *it->second);
                }
                Term_eq result(flat_metric->derive(type_der, ind_der, target));
                const Tensor& fresh = result.get_val_t();
                // Cache only when valence indicates no inner-summation
                // post-processing (valence = source.valence + 1).
                const int expected_valence = target.get_val_t().get_valence() + 1;
                if (fresh.get_valence() == expected_valence) {
                    auto stored_owner = std::make_unique<Tensor>(fresh, false);
                    Tensor& stored = *stored_owner;
                    for (int component = 0; component < fresh.get_n_comp(); ++component) {
                        Array<int> index(fresh.indices(component));
                        stored.set(index).set_domain(dom) = fresh(index)(dom);
                    }
                    if (fresh.is_name_affected()) {
                        stored.set_name_affected();
                        for (int i = 0; i < stored.get_valence(); ++i)
                            stored.set_name_ind(i, fresh.get_name_ind()[i]);
                    }
                    cache.emplace(this, std::move(stored_owner));
                }
                return result;
            }
        }

        std::optional<Term_eq> target_storage;
        const Term_eq& target = parts[0]->action_operand(target_storage);
        // Check it is a tensor
        if (target.type_data != TERM_T) {
            KADATH_THROW("Ope_der only defined with respect for a tensor");
        }

        return (syst->get_met()->derive(type_der, ind_der, target));
    }
} // namespace Kadath
