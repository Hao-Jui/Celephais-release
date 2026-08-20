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

#include "For_Kadath/System_of_eqs/system_of_eqs.hpp"
#include "For_Kadath/Array/exceptions.hpp"
#include "For_Kadath/Domain/bispheric.hpp"
#include "For_Kadath/Domain/bispheric_nosym.hpp"
#include "For_Kadath/Scalar/scalar.hpp"
#include "For_Kadath/Metric/metric.hpp"
#include "For_Kadath/Utilities/runtime_env.hpp"
#include "mpi.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <ostream>
#include <set>
#include <string_view>
#include <sys/resource.h>
#include <utility>
#include <vector>
#if defined(__APPLE__)
#include <mach/mach.h>
#endif

namespace Kadath
{
    namespace
    {
        bool do_jx_mpi_enabled()
        {
            static const bool enabled = env_flag_enabled("DO_JX_MPI", true);
            return enabled;
        }

        bool sec_member_mpi_enabled()
        {
            static const bool enabled = env_flag_enabled("SEC_MEMBER_MPI", true);
            return enabled;
        }

        double elapsed_since(std::chrono::steady_clock::time_point start)
        {
            return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        }
    }

    void System_of_eqs::classify_columns(std::vector<ColumnMetadata>& out) const
    {
        std::vector<ColumnInfo> column_map;
        build_column_map(column_map, true);
        classify_columns(out, column_map);
    }

    void System_of_eqs::classify_columns(std::vector<ColumnMetadata>& out,
                                         const std::vector<ColumnInfo>& column_map) const
    {
        out.clear();
        out.reserve(column_map.size());
        for (std::size_t i = 0; i < column_map.size(); ++i) {
            const ColumnInfo& info = column_map[i];
            ColumnMetadata metadata;
            metadata.column = static_cast<int>(i);
            metadata.domain = info.domain;
            metadata.term_idx = info.term_idx;
            metadata.var_idx = info.var_idx;
            metadata.var_double_idx = info.var_double_idx;
            metadata.var_name = info.var_name;
            metadata.basis_mode = info.basis_mode;
            metadata.domain_type_id = info.domain_type_id;
            metadata.tensor_component = info.tensor_component;
            metadata.coefficient_i = info.coefficient_i;
            metadata.coefficient_j = info.coefficient_j;
            metadata.coefficient_k = info.coefficient_k;
            metadata.coefficient_nr = info.coefficient_nr;
            metadata.coefficient_nt = info.coefficient_nt;
            metadata.coefficient_np = info.coefficient_np;
            if (info.is_var_domain) {
                metadata.column_class = ColumnClass::VarDomain;
                metadata.vardom_param = info.basis_mode >= 0 ? info.basis_mode : static_cast<int>(i);
                metadata.reason = "variable_domain_coefficient";
            } else if (info.var_double_idx >= 0) {
                metadata.column_class = ColumnClass::ScalarGlobal;
                metadata.reason = "global_scalar_parameter";
            } else if (info.domain >= 0) {
                metadata.column_class = info.field_class;
                switch (info.field_class) {
                    case ColumnClass::FieldInteriorVol:
                        metadata.reason = "field_coefficient_selected_by_volume_equation";
                        break;
                    case ColumnClass::FieldBoundaryTau:
                        metadata.reason = "field_coefficient_constrained_by_boundary_tau";
                        break;
                    case ColumnClass::FieldMatching:
                        metadata.reason = "field_coefficient_constrained_by_matching";
                        break;
                    case ColumnClass::FieldGauge:
                        metadata.reason = "field_coefficient_attached_to_global_constraint";
                        break;
                    default:
                        metadata.reason = "field_equation_attachment_not_resolved";
                        break;
                }
            } else {
                metadata.column_class = ColumnClass::Unknown;
                metadata.reason = "unmapped_column";
            }
            out.push_back(metadata);
        }
    }

    Array<double> System_of_eqs::check_equations()
    {

        Array<double> sec(sec_member());
        Array<double> errors(neq_int + neq);
        errors = 0;

        int pos = 0;
        for (int i = 0; i < neq_int; i++) {
            errors.set(i) = fabs(sec(pos));
            pos++;
        }

        for (int i = 0; i < neq; i++) {
            double max = 0;
            for (int j = 0; j < eq[i]->get_n_cond_tot(); j++) {
                if (fabs(sec(pos)) > max)
                    max = fabs(sec(pos));
                pos++;
            }
            errors.set(neq_int + i) = max;
        }
        return errors;
    }

    Array<double> System_of_eqs::sec_member()
    {

        vars_to_terms();

        if (met != nullptr)
            for (int d = dom_min; d <= dom_max; d++)
                met->update(d);
        for (int i = 0; i < ndef; i++)
            def[i]->compute_res();
        // A full replicated value pass recomputes every definition: record
        // them all fresh in the value-staleness ledger when a do_JX partition
        // exists (sec_member_partitioned_tail narrows this to its closure).
        if (do_jx_mpi_.active)
            do_jx_mpi_.def_val_current.assign(static_cast<std::size_t>(ndef), 1);

        int conte = 0;
        Term_eq** results_raw = results_shadow_view();
        for (int i = 0; i < neq; i++)
            eq[i]->apply(conte, results_raw);
        results_shadow_sync();

        // Need to assert the size :
        if (nbr_conditions == -1) {
            nbr_conditions = 0;
            for (int i = 0; i < neq_int; i++)
                nbr_conditions++;
            for (int i = 0; i < neq; i++)
                nbr_conditions += eq[i]->get_n_cond_tot();
        }

        // Computation of the second member itself :
        Array<double> res(nbr_conditions);
        res = 0;
        conte = 0;
        int pos_res = 0;
        for (int i = 0; i < neq_int; i++) {
            res.set(pos_res) = eq_int[i]->get_val();
            pos_res++;
        }

        for (int i = 0; i < neq; i++)
            eq[i]->export_val(conte, results_raw, res, pos_res);
        return res;
    }

    const std::string& System_of_eqs::equation_owner_var_name(
        int equation_index) const
    {
        static const std::string unavailable;
        if (equation_index < 0 ||
            equation_index >= static_cast<int>(eq_column_attachments.size())) {
            return unavailable;
        }
        return eq_column_attachments[static_cast<std::size_t>(equation_index)]
            .owner_var_name;
    }

    bool System_of_eqs::describe_residual_rows(
        std::vector<ResidualRowDescriptor>& descriptors)
    {
        if (nbr_conditions == -1)
            (void)sec_member();

        descriptors.clear();
        descriptors.reserve(static_cast<std::size_t>(nbr_conditions));
        bool all_available = true;
        for (int integral = 0; integral < neq_int; ++integral) {
            ResidualRowDescriptor descriptor;
            const int sector = eq_int[integral]->get_reflection_sector();
            if (sector == -1 || sector == 1) {
                // A global scalar row has no honest unique coefficient-space
                // coordinate. Keep sides empty and carry the audited sector
                // explicitly instead of fabricating a phi index.
                descriptor.family = ResidualRowEquationFamily::Integral;
                descriptor.equation_index = integral;
                descriptor.available = true;
                descriptor.explicit_sector = static_cast<signed char>(sector);
            } else {
                all_available = false;
            }
            descriptors.push_back(std::move(descriptor));
        }
        int conte = 0;
        Term_eq** residuals = results_shadow_view();
        for (int equation_index = 0; equation_index < neq; ++equation_index) {
            const int expected_conte = conte + eq[equation_index]->n_ope;
            const int expected_rows = eq[equation_index]->get_n_cond_tot();
            std::vector<ResidualRowDescriptor> block;
            bool available = eq[equation_index]->describe_residual_rows(
                conte, residuals, equation_index, block);
            available = available && conte == expected_conte &&
                        block.size() == static_cast<std::size_t>(expected_rows);
            if (available) {
                for (const ResidualRowDescriptor& descriptor : block) {
                    if (!descriptor.available ||
                        descriptor.family != ResidualRowEquationFamily::Field ||
                        descriptor.equation_index != equation_index ||
                        descriptor.explicit_sector != 0 ||
                        descriptor.sides.empty()) {
                        available = false;
                        break;
                    }
                }
            }
            if (!available) {
                conte = expected_conte;
                block.assign(static_cast<std::size_t>(expected_rows),
                             ResidualRowDescriptor{});
            }
            descriptors.insert(descriptors.end(), block.begin(), block.end());
            all_available = all_available && available;
        }

        if (descriptors.size() != static_cast<std::size_t>(nbr_conditions)) {
            descriptors.assign(static_cast<std::size_t>(nbr_conditions),
                               ResidualRowDescriptor{});
            return false;
        }
        return all_available;
    }

    double System_of_eqs::sec_member_eq_int(int eq_int_index)
    {
        if (eq_int_index < 0 || eq_int_index >= neq_int)
            KADATH_THROW("Integral equation index out of range in sec_member_eq_int");

        vars_to_terms();

        if (met != nullptr)
            for (int d = dom_min; d <= dom_max; d++)
                met->update(d);

        // Ope_id leaves retain pointers to the persistent definition result
        // slots. Map those slots back to their registration indices, then walk
        // only the selected Eq_int's transitive definition references. Parser
        // registration requires dependencies to precede their users, so the
        // final ascending sweep preserves the same order as sec_member().
        std::map<const Term_eq*, int> def_index_of_term;
        for (int i = 0; i < ndef; ++i)
            def_index_of_term[def[i]->get_res()] = i;

        std::set<const Term_eq*> targets;
        for (int k = 0; k < eq_int[eq_int_index]->n_ope; ++k)
            eq_int[eq_int_index]->parts[k]->collect_def_targets(targets);

        std::vector<char> def_needed(static_cast<std::size_t>(ndef), 0);
        std::vector<int> stack;
        for (const Term_eq* target : targets) {
            const auto it = def_index_of_term.find(target);
            if (it != def_index_of_term.end())
                stack.push_back(it->second);
        }
        while (!stack.empty()) {
            const int i = stack.back();
            stack.pop_back();
            if (def_needed[i])
                continue;
            def_needed[i] = 1;

            targets.clear();
            def[i]->collect_def_targets(targets);
            for (const Term_eq* target : targets) {
                const auto it = def_index_of_term.find(target);
                if (it != def_index_of_term.end() && it->second != i)
                    stack.push_back(it->second);
            }
        }

        for (int i = 0; i < ndef; ++i)
            if (def_needed[i])
                def[i]->compute_res();

        // If a do_JX value ledger already exists, be conservative: the
        // unknown preamble made every old definition value stale and this pass
        // refreshed only the selected closure.
        if (do_jx_mpi_.active) {
            do_jx_mpi_.def_val_current.assign(static_cast<std::size_t>(ndef), 0);
            for (int i = 0; i < ndef; ++i)
                if (def_needed[i])
                    do_jx_mpi_.def_val_current[i] = 1;
        }

        return eq_int[eq_int_index]->get_val();
    }

    // MPI row-partitioned value tail for the Newton residual
    // (sec_member_partitioned): the exact mirror of do_JX's partitioned tail
    // with derivatives replaced by values — same partial definition closure
    // (def_needed), same owned-equation apply loop, export_val instead of
    // export_der, Eq_int parts via get_val_d instead of get_der_d, and the
    // identical pair of MPI_Iallgatherv exchanges with the same counts,
    // displacements, and order (any divergence in collective order against
    // the matvec tail would deadlock; each call issues its own matched
    // pair on every rank). Bit-exact with the replicated value path: every
    // row and every Eq_int part value is produced by exactly one rank
    // through the unchanged serial code, exchanged verbatim, and Eq_int
    // parts are re-summed in the same canonical (i, k) order Eq_int::get_val
    // uses — no floating-point reduction occurs.
    Array<double> System_of_eqs::sec_member_partitioned_tail()
    {
        auto& part = do_jx_mpi_;

        for (int i = 0; i < ndef; i++)
            if (part.def_needed[i])
                def[i]->compute_res();
        // Value-staleness ledger: after a partial pass only the defs in this
        // rank's closure hold current values (out-of-range defs are already
        // flagged in def_needed on every rank by construction, so they stay
        // marked fresh here too).
        part.def_val_current.assign(part.def_needed.begin(), part.def_needed.end());

        Term_eq** results_raw = results_shadow_view();
        int conte = part.eq_operator_offset[part.eq_begin];
        for (int i = part.eq_begin; i < part.eq_end; i++)
            eq[i]->apply(conte, results_raw);
        results_shadow_sync();

        Array<double> res(nbr_conditions);
        res = 0;

        // Owned Eq_int parts, evaluated in canonical (eq_int, part) order.
        const int total_parts = part.part_offset[neq_int];
        int part_send_cursor = 0;
        for (int i = 0; i < neq_int; i++)
            for (int k = 0; k < eq_int[i]->n_ope; k++)
                if (part.part_owner[part.part_offset[i] + k] == part.rank)
                    part.part_send_workspace[part_send_cursor++] =
                        eq_int[i]->parts[k]->action().get_val_d();
        assert(part_send_cursor == part.part_counts[part.rank]);

        // Owned field rows, exported at their canonical positions.
        conte = part.eq_operator_offset[part.eq_begin];
        int pos_res = part.eq_row_begin[part.eq_begin];
        for (int i = part.eq_begin; i < part.eq_end; i++)
            eq[i]->export_val(conte, results_raw, res, pos_res);

        // Post both collectives in the same order on every rank. Once the
        // part exchange completes, reconstruct the disjoint Eq_int prefix
        // while the field-row exchange may continue into the result suffix.
        MPI_Request part_request = MPI_REQUEST_NULL;
        MPI_Request row_request = MPI_REQUEST_NULL;
        if (total_parts > 0)
            MPI_Iallgatherv(part.part_send_workspace.data(), part_send_cursor, MPI_DOUBLE,
                            part.part_recv_workspace.data(), part.part_counts.data(),
                            part.part_displs.data(), MPI_DOUBLE, MPI_COMM_WORLD,
                            &part_request);
        if (nbr_conditions > neq_int)
            MPI_Iallgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
                            res.set_data() + neq_int, part.row_counts.data(),
                            part.row_displs.data(), MPI_DOUBLE, MPI_COMM_WORLD,
                            &row_request);
        if (total_parts > 0)
            MPI_Wait(&part_request, MPI_STATUS_IGNORE);

        // Reconstruct the Eq_int rows: read each exchanged part value from
        // its owner's slot and sum in the same order Eq_int::get_val uses.
        std::copy(part.part_displs.begin(), part.part_displs.end(),
                  part.owner_cursor_workspace.begin());
        for (int i = 0; i < neq_int; i++) {
            double value = 0.;
            for (int k = 0; k < eq_int[i]->n_ope; k++) {
                const int owner = part.part_owner[part.part_offset[i] + k];
                value += part.part_recv_workspace[part.owner_cursor_workspace[owner]++];
            }
            res.set(i) = value;
        }
        if (nbr_conditions > neq_int)
            MPI_Wait(&row_request, MPI_STATUS_IGNORE);
        return res;
    }

    Array<double> System_of_eqs::sec_member_partitioned()
    {
        // Reuse the do_JX row partition for the value pass only when do_JX
        // has already built and validated it for the current topology; the
        // value path never builds the partition itself, so the first Newton
        // residual (and any call before the first partitioned matvec) stays
        // on the replicated path below.
        bool partition_ready = false;
        if (sec_member_mpi_enabled() && nbr_conditions != -1) {
            int mpi_initialized = 0;
            MPI_Initialized(&mpi_initialized);
            if (mpi_initialized != 0) {
                int nproc = 1;
                MPI_Comm_size(MPI_COMM_WORLD, &nproc);
                if (nproc > 1 && do_jx_mpi_.neq == neq && do_jx_mpi_.neq_int == neq_int &&
                    do_jx_mpi_.nbr_conditions == nbr_conditions && do_jx_mpi_.nproc == nproc)
                    partition_ready = do_jx_mpi_.active && do_jx_mpi_.val_active;
            }
        }
        if (!partition_ready)
            return sec_member();

        vars_to_terms();

        if (met != nullptr)
            for (int d = dom_min; d <= dom_max; d++)
                met->update(d);

        Array<double> res = sec_member_partitioned_tail();

        // First partitioned value call per topology (and the first call
        // after the measured-cost repartition): verify bitwise against the
        // replicated value path. Collective verdict (no rank-divergent
        // throw); on mismatch fall back permanently and return the
        // replicated result. The reference is the bare sec_member() — its
        // head re-runs vars_to_terms + metric update on unchanged unknowns,
        // which is deterministic and value-identical.
        if (do_jx_mpi_.val_selftest_pending) {
            do_jx_mpi_.val_selftest_pending = false;
            Array<double> reference = sec_member();
            const int local_ok =
                (std::memcmp(res.get_data(), reference.get_data(),
                             sizeof(double) * static_cast<std::size_t>(nbr_conditions)) == 0)
                    ? 1
                    : 0;
            int global_ok = 0;
            MPI_Allreduce(&local_ok, &global_ok, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
            if (do_jx_mpi_.rank == 0 && global_ok == 0) {
                std::cout << "sec_member MPI partition self-test"
                          << (do_jx_mpi_.rebalanced ? " (rebalanced)" : "") << ": "
                          << "FAIL — falling back to the replicated residual"
                          << " (n=" << nbr_conditions << ", nproc=" << do_jx_mpi_.nproc << ")"
                          << '\n';
            }
            if (global_ok == 0) {
                do_jx_mpi_.val_active = false;
                res = reference;
            }
        }
        return res;
    }

    Array<double> System_of_eqs::take_forwarded_residual_or_compute(
        const char* producer)
    {
        // Always consume the single-shot slot. In particular, disabling
        // forwarding must not leave a residual around that could become stale
        // and later be consumed if the runtime flag changes.
        std::unique_ptr<Array<double>> forwarded = take_forwarded_residual();
        if (!env_flag_enabled("RESIDUAL_FORWARD", true))
            return sec_member_partitioned();

        if (!env_flag_enabled("RESIDUAL_FORWARD_SELFTEST", false)) {
            if (forwarded)
                return std::move(*forwarded);
            return sec_member_partitioned();
        }

        // The self-test deliberately computes the fresh vector on every rank
        // before deciding whether the forwarded candidate is admissible. This
        // preserves the collective ordering of sec_member_partitioned even if
        // a lifecycle bug made slot availability differ between ranks.
        Array<double> fresh(sec_member_partitioned());
        const bool local_bytes_equal = forwarded &&
            forwarded->get_nbr() == fresh.get_nbr() &&
            std::memcmp(forwarded->get_data(), fresh.get_data(),
                        sizeof(double) * static_cast<std::size_t>(fresh.get_nbr())) == 0;
        const int local[2] = {forwarded ? 1 : 0, local_bytes_equal ? 1 : 0};
        int global[2] = {0, 0};
        MPI_Allreduce(local, global, 2, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

        int rank = 0;
        int nproc = 1;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &nproc);
        const bool globally_exact = global[0] == nproc && global[1] == nproc;
        if (rank == 0 && global[0] != 0) {
            std::cout << "  [self-test] forwarded residual (" << producer
                      << ") -> fresh residual bytes: "
                      << (globally_exact ? "exact" : "DIFFERS")
                      << " (n=" << fresh.get_nbr() << ", matching_ranks="
                      << global[1] << '/' << nproc << ')' << std::endl;
        }
        return globally_exact ? std::move(*forwarded) : std::move(fresh);
    }

    bool System_of_eqs::forwarded_residual_infinity_norm(double& norm) const
    {
        if (!forwarded_residual_)
            return false;

        norm = 0.;
        const Array<double>& residual = *forwarded_residual_;
        const int nres = static_cast<int>(residual.get_nbr());
        for (int i = 0; i < nres; ++i) {
            const double magnitude = std::abs(residual(i));
            // Fail closed: std::max silently discards NaN when its finite
            // accumulator is the first operand, which could report a false
            // refreshed-system convergence. Infinity remains infinity.
            if (!std::isfinite(magnitude)) {
                norm = std::numeric_limits<double>::infinity();
                return true;
            }
            norm = std::max(norm, magnitude);
        }
        return true;
    }

    Array<double> System_of_eqs::do_JX(const Array<double>& xx)
    {
        if (nbr_conditions == -1) {
            KADATH_THROW("Number of conditions unknown ; call sec_member first");
        }
        Array<double> result(nbr_conditions);
        do_JX(xx, result);
        return result;
    }

    void System_of_eqs::do_JX(const Array<double>& xx, Array<double>& result)
    {
        if (nbr_conditions == -1) {
            KADATH_THROW("Number of conditions unknown ; call sec_member first");
        }
        if (result.get_ndim() != 1 ||
            result.get_nbr() != static_cast<std::size_t>(nbr_conditions)) {
            KADATH_THROW("do_JX output buffer has the wrong size");
        }
        if (!variable_domains_cache_valid_) {
            std::set<int> doms;
            const int n_var_domain = espace.nbr_unknowns_from_variable_domains();
            for (int cc = 0; cc < n_var_domain; cc++) {
                int conte = 0;
                Array<int> zedoms(2);
                zedoms = -1;
                espace.affecte_coef_to_variable_domains(conte, cc, zedoms);
                for (int d = 0; d < zedoms.get_size(0); d++)
                    if (zedoms(d) != -1)
                        doms.insert(zedoms(d));
            }
            variable_domains_cache_.assign(doms.begin(), doms.end());
            variable_domains_cache_valid_ = true;
        }
        const std::vector<int>& variable_domains = variable_domains_cache_;

        // Partition readiness is keyed before the preamble: the per-rank
        // term/cst closure (DO_JX_TERM_CLOSURE) filters the preamble
        // itself, so it must be known here. The build is deterministic and
        // communication-free, so hoisting it above xx_to_ders changes
        // nothing else.
        bool partition_ready = false;
        if (do_jx_mpi_enabled() && nbr_conditions != -1) {
            int mpi_initialized = 0;
            MPI_Initialized(&mpi_initialized);
            if (mpi_initialized != 0) {
                int nproc = 1;
                int rank = 0;
                MPI_Comm_size(MPI_COMM_WORLD, &nproc);
                MPI_Comm_rank(MPI_COMM_WORLD, &rank);
                if (nproc > 1) {
                    if (do_jx_mpi_.neq != neq || do_jx_mpi_.neq_int != neq_int ||
                        do_jx_mpi_.nbr_conditions != nbr_conditions || do_jx_mpi_.nproc != nproc)
                        build_do_jx_mpi_partition(nproc, rank);
                    partition_ready = do_jx_mpi_.active;
                }
            }
        }

        // Filtered preamble: seed (xx_to_ders) and shape-update
        // (update_terms_from_variable_domains) only the terms / TERM_T
        // constants whose derivatives this rank's owned rows read. Never
        // filter while the matvec self-test is pending — its replicated
        // reference tail reads every derivative off this preamble; the
        // value self-test is held out too (conservatively: its reference is
        // value-only, but keeping the filter off until both one-shot
        // verifications resolve keeps the state machine simple).
        // Rank-uniform by construction
        // (env vars are launcher-forwarded; every flag below follows the
        // same collective state machine on all ranks), so the collectives
        // in the tails and self-tests stay matched. The filter has its own
        // one-shot bitwise self-test below (term_filter_selftest_pending).
        const bool use_term_filter = partition_ready && do_jx_mpi_.term_filter_active &&
                                     !do_jx_mpi_.selftest_pending &&
                                     !do_jx_mpi_.val_selftest_pending;

        const auto seed_and_update_preamble = [&](bool filtered) {
            const std::vector<char>* term_filter =
                filtered ? &do_jx_mpi_.term_needed : nullptr;
            xx_to_ders(xx, term_filter);
            if (!variable_domains.empty()) {
                Array<int> zedoms(static_cast<int>(variable_domains.size()));
                for (std::size_t i = 0; i < variable_domains.size(); i++)
                    zedoms.set(static_cast<int>(i)) = variable_domains[i];

                // Terms already contain their field-coefficient tangents, so
                // adapted domains add the mapping contribution directly.
                // Constants were never seeded by xx_to_ders and deliberately
                // retain replacement semantics inside this call.
                accumulate_terms_from_variable_domains(
                    zedoms, term_filter, filtered ? &do_jx_mpi_.cst_needed : nullptr);
            }
            if (met != nullptr)
                for (int d = dom_min; d <= dom_max; d++)
                    met->update(d);
        };
        seed_and_update_preamble(use_term_filter);
        // Definition recompute + equation apply/export tail. Runs either
        // replicated (every rank computes the full vector — historical path)
        // or MPI row-partitioned (DO_JX_MPI): each rank computes a
        // contiguous block of field-equation rows plus the Eq_int integral
        // parts whose domain it owns, and the full vector is reassembled with
        // MPI_Iallgatherv. Bit-exact with the replicated path: every row and
        // every Eq_int part value is produced by exactly one rank through the
        // same serial code, exchanged verbatim, and Eq_int parts are re-summed
        // in their canonical order — no floating-point reduction occurs.
        // measure_costs: record per-def, per-eq, and per-Eq_int-part wall
        // costs into the partition struct (used by the self-test reference
        // pass to drive the measured-cost repartition; the timers do not
        // change any value).
        const auto replicated_tail = [&](bool measure_costs, Array<double>& res) {
            // Per-part timers need the flattened part layout; the partition
            // (and so part_offset) is built before the self-test reference
            // pass runs, but guard anyway and skip them if it is not.
            const bool measure_part_costs =
                measure_costs &&
                do_jx_mpi_.part_offset.size() == static_cast<std::size_t>(neq_int) + 1;
            if (measure_costs) {
                do_jx_mpi_.def_cost.assign(static_cast<std::size_t>(ndef), 0.0);
                do_jx_mpi_.eq_cost.assign(static_cast<std::size_t>(neq), 0.0);
                do_jx_mpi_.part_cost.assign(
                    measure_part_costs
                        ? static_cast<std::size_t>(do_jx_mpi_.part_offset[neq_int])
                        : 0,
                    0.0);
            }
            // Run fn(); when measure_costs is set, accumulate its wall time
            // into cost_slot[idx]. Collapses the timed/untimed branch that the
            // def-compute, eq-apply, and eq-export loops all share.
            const auto timed = [&](std::vector<double>& cost_slot, int idx, auto&& fn) {
                if (measure_costs) {
                    const auto t0 = std::chrono::steady_clock::now();
                    fn();
                    cost_slot[idx] += elapsed_since(t0);
                } else {
                    fn();
                }
            };

            for (int i = 0; i < ndef; i++) {
                timed(do_jx_mpi_.def_cost, i, [&] { def[i]->compute_res(); });
            }

            int conte = 0;
            Term_eq** results_raw = results_shadow_view();
            for (int i = 0; i < neq; i++) {
                timed(do_jx_mpi_.eq_cost, i, [&] { eq[i]->apply(conte, results_raw); });
            }
            results_shadow_sync();

            conte = 0;
            int pos_res = 0;
            res = 0;
            for (int i = 0; i < neq_int; i++) {
                if (measure_part_costs) {
                    // Reproduces Eq_int::get_der bit-exactly (parts summed in
                    // canonical k order) while timing each integral part for
                    // the measured-cost repartition.
                    double value = 0.;
                    for (int k = 0; k < eq_int[i]->n_ope; k++) {
                        const auto t0 = std::chrono::steady_clock::now();
                        value += eq_int[i]->parts[k]->action().get_der_d();
                        do_jx_mpi_.part_cost[do_jx_mpi_.part_offset[i] + k] +=
                            elapsed_since(t0);
                    }
                    res.set(pos_res) = value;
                } else {
                    res.set(pos_res) = eq_int[i]->get_der();
                }
                pos_res++;
            }

            for (int i = 0; i < neq; i++) {
                timed(do_jx_mpi_.eq_cost, i,
                      [&] { eq[i]->export_der(conte, results_raw, res, pos_res); });
            }
        };

        const auto partitioned_tail = [&](Array<double>& res) {
            auto& part = do_jx_mpi_;

            for (int i = 0; i < ndef; i++)
                if (part.def_needed[i])
                    def[i]->compute_res();

            Term_eq** results_raw = results_shadow_view();
            int conte = part.eq_operator_offset[part.eq_begin];
            for (int i = part.eq_begin; i < part.eq_end; i++)
                eq[i]->apply(conte, results_raw);
            results_shadow_sync();

            // Owned Eq_int parts, evaluated in canonical (eq_int, part) order.
            const int total_parts = part.part_offset[neq_int];
            int part_send_cursor = 0;
            for (int i = 0; i < neq_int; i++)
                for (int k = 0; k < eq_int[i]->n_ope; k++)
                    if (part.part_owner[part.part_offset[i] + k] == part.rank)
                        part.part_send_workspace[part_send_cursor++] =
                            eq_int[i]->parts[k]->action().get_der_d();
            assert(part_send_cursor == part.part_counts[part.rank]);

            // Owned field rows, exported at their canonical positions.
            conte = part.eq_operator_offset[part.eq_begin];
            int pos_res = part.eq_row_begin[part.eq_begin];
            // Some exporters advance past mathematically-zero coefficients
            // without storing them. Only this rank's in-place send slice needs
            // initialization; remote slices are fully overwritten by the row
            // exchange, and the Eq_int prefix is reconstructed explicitly below.
            if (part.row_counts[part.rank] > 0)
                std::fill_n(res.set_data() + pos_res,
                            static_cast<std::size_t>(part.row_counts[part.rank]), 0.0);
            for (int i = part.eq_begin; i < part.eq_end; i++)
                eq[i]->export_der(conte, results_raw, res, pos_res);

            // Post both collectives in the same order on every rank. Once the
            // part exchange completes, reconstruct the disjoint Eq_int prefix
            // while the field-row exchange may continue into the result suffix.
            MPI_Request part_request = MPI_REQUEST_NULL;
            MPI_Request row_request = MPI_REQUEST_NULL;
            if (total_parts > 0)
                MPI_Iallgatherv(part.part_send_workspace.data(), part_send_cursor, MPI_DOUBLE,
                                part.part_recv_workspace.data(), part.part_counts.data(),
                                part.part_displs.data(), MPI_DOUBLE, MPI_COMM_WORLD,
                                &part_request);
            if (nbr_conditions > neq_int)
                MPI_Iallgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
                                res.set_data() + neq_int, part.row_counts.data(),
                                part.row_displs.data(), MPI_DOUBLE, MPI_COMM_WORLD,
                                &row_request);
            if (total_parts > 0)
                MPI_Wait(&part_request, MPI_STATUS_IGNORE);

            // Reconstruct the Eq_int rows: read each exchanged part value from
            // its owner's slot and sum in the same order Eq_int::get_der uses.
            std::copy(part.part_displs.begin(), part.part_displs.end(),
                      part.owner_cursor_workspace.begin());
            for (int i = 0; i < neq_int; i++) {
                double value = 0.;
                for (int k = 0; k < eq_int[i]->n_ope; k++) {
                    const int owner = part.part_owner[part.part_offset[i] + k];
                    value += part.part_recv_workspace[part.owner_cursor_workspace[owner]++];
                }
                res.set(i) = value;
            }
            if (nbr_conditions > neq_int)
                MPI_Wait(&row_request, MPI_STATUS_IGNORE);
        };

        if (partition_ready)
            partitioned_tail(result);
        else
            replicated_tail(false, result);

        // First partitioned call per topology (and the first call after the
        // measured-cost repartition): verify bitwise against the replicated
        // path. Collective verdict (no rank-divergent throw); on mismatch fall
        // back permanently and return the replicated result. The reference
        // pass doubles as the cost-measurement pass for the repartition.
        if (partition_ready && do_jx_mpi_.selftest_pending) {
            do_jx_mpi_.selftest_pending = false;
            const bool measure_costs = !do_jx_mpi_.rebalanced;
            Array<double> reference(nbr_conditions);
            replicated_tail(measure_costs, reference);
            const int local_ok =
                (std::memcmp(result.get_data(), reference.get_data(),
                             sizeof(double) * static_cast<std::size_t>(nbr_conditions)) == 0)
                    ? 1
                    : 0;
            int global_ok = 0;
            MPI_Allreduce(&local_ok, &global_ok, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
            if (do_jx_mpi_.rank == 0 && global_ok == 0) {
                std::cout << "do_JX MPI partition self-test"
                          << (do_jx_mpi_.rebalanced ? " (rebalanced)" : "") << ": "
                          << "FAIL — falling back to the replicated matvec"
                          << " (n=" << nbr_conditions << ", nproc=" << do_jx_mpi_.nproc << ")"
                          << '\n';
            }
            if (global_ok == 0) {
                do_jx_mpi_.active = false;
                result = std::move(reference);
            } else if (!do_jx_mpi_.rebalanced) {
                rebalance_do_jx_partition_from_measured_costs();
            }
        }

        // First filtered-preamble call after each partition (re)derivation:
        // verify the per-rank term/cst closure bitwise by re-running the FULL
        // preamble on the same xx (xx_to_ders is deterministic and overwrites
        // every slot it seeds; the variable-domain update and metric update
        // are redone identically) through the identical partitioned tail.
        // Mutually exclusive with the partition self-test above:
        // use_term_filter is false while selftest_pending is set.
        // Collective-symmetric — use_term_filter is rank-uniform, every rank
        // runs the second tail (same Allgatherv pair) and the verdict is an
        // Allreduce. On mismatch the filter is disabled permanently and the
        // full-preamble result is returned. Cost: one extra matvec, one-shot
        // per split.
        if (use_term_filter && do_jx_mpi_.term_filter_selftest_pending) {
            do_jx_mpi_.term_filter_selftest_pending = false;
            seed_and_update_preamble(false);
            Array<double> reference(nbr_conditions);
            partitioned_tail(reference);
            const int local_ok =
                (std::memcmp(result.get_data(), reference.get_data(),
                             sizeof(double) * static_cast<std::size_t>(nbr_conditions)) == 0)
                    ? 1
                    : 0;
            int global_ok = 0;
            MPI_Allreduce(&local_ok, &global_ok, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
            if (do_jx_mpi_.rank == 0 && global_ok == 0) {
                std::cout << "do_JX term-closure self-test: "
                          << "FAIL — falling back to the full preamble"
                          << " (n=" << nbr_conditions << ", nproc=" << do_jx_mpi_.nproc << ")"
                          << '\n';
            }
            if (global_ok == 0) {
                do_jx_mpi_.term_filter_active = false;
                result = std::move(reference);
            }
        }
    }

    bool System_of_eqs::do_col_J_def_filter_enabled() const
    {
        return solver_runtime_config.diagnostics.def_filter;
    }

    // Jacobian column computation, sparse-emit, profile dump, and cache reset
    // bodies were migrated to JacobianColumnEngine in stream S2; the
    // System_of_eqs public surface below is preserved as one-line shims so
    // external call sites (apps/, src_par/, Jacobian coloring units) continue to work
    // unchanged.
    void System_of_eqs::reset_do_col_J_cache() { jac_col_engine_.reset_cache(); }
    void System_of_eqs::reset_do_col_J_cache(
        JacobianColumnEngine::Workspace& workspace)
    {
        jac_col_engine_.reset_cache(workspace);
    }
    void System_of_eqs::release_do_col_J_assembly_scratch()
    {
        jac_col_engine_.release_assembly_scratch();
    }
    void System_of_eqs::release_do_col_J_assembly_scratch(
        JacobianColumnEngine::Workspace& workspace)
    {
        jac_col_engine_.release_assembly_scratch(workspace);
    }

    Array<double> System_of_eqs::do_col_J(int cc)
    {
        if (nbr_conditions == -1) {
            KADATH_THROW("Number of conditions unknown ; call sec_member first");
        }
        Array<double> res(nbr_conditions);
        res = 0;
        jac_col_engine_.compute_column(cc, res);
        return res;
    }

    void System_of_eqs::do_col_J(int cc, double* out)
    {
        if (nbr_conditions == -1) {
            KADATH_THROW("Number of conditions unknown ; call sec_member first");
        }
        Array<double> buf(nbr_conditions);
        buf = 0.0;
        jac_col_engine_.compute_column(cc, buf);
        std::memcpy(out, buf.get_data(), static_cast<size_t>(nbr_conditions) * sizeof(double));
    }

    void System_of_eqs::do_col_J_sparse(int cc, double drop_tol,
                                        SparseColumnEmitter emit,
                                        JacobianSelectedRows selected_rows)
    {
        jac_col_engine_.compute_column_sparse(cc, drop_tol, emit, selected_rows);
    }

    void System_of_eqs::do_col_J_sparse(
        JacobianColumnEngine::Workspace& workspace, int cc, double drop_tol,
        SparseColumnEmitter emit, JacobianSelectedRows selected_rows)
    {
        jac_col_engine_.compute_column_sparse(
            workspace, cc, drop_tol, emit, selected_rows);
    }

    bool System_of_eqs::do_cols_J_wlane2_sparse(int first_column, int second_column,
                                                double drop_tol,
                                                SparseColumnEmitter emit_first,
                                                SparseColumnEmitter emit_second,
                                                std::string& failure_reason,
                                                JacobianSelectedRows selected_rows)
    {
        // Packed W-lane assembly drops tangent lanes 1..W-1 through the metric
        // curvature builders (Metric/Metric_general compute_christo/ricci/...),
        // so it is only correct for metrics that report it lane-safe. Curved
        // metrics fall back to the scalar do_col_J path (lane 0, correct).
        if (met != nullptr && !met->supports_packed_lane_jacobian()) {
            failure_reason = "packed W-lane path unsupported for curved (non-flat) metric";
            return false;
        }
        return jac_col_engine_.compute_packed_wlane2_columns_sparse(
            first_column, second_column, drop_tol, emit_first, emit_second,
            failure_reason, selected_rows);
    }

    bool System_of_eqs::do_cols_J_wlane4_sparse(const std::array<int, 4>& columns,
                                                double drop_tol,
                                                std::array<SparseColumnEmitter, 4>& emitters,
                                                std::string& failure_reason,
                                                JacobianSelectedRows selected_rows)
    {
        if (met != nullptr && !met->supports_packed_lane_jacobian()) {
            failure_reason = "packed W-lane path unsupported for curved (non-flat) metric";
            return false;
        }
        return jac_col_engine_.compute_packed_wlane4_columns_sparse(
            columns, drop_tol, emitters, failure_reason, selected_rows);
    }

    bool System_of_eqs::do_cols_J_wlane8_sparse(const std::array<int, 8>& columns,
                                                double drop_tol,
                                                std::array<SparseColumnEmitter, 8>& emitters,
                                                std::string& failure_reason,
                                                JacobianSelectedRows selected_rows)
    {
        if (met != nullptr && !met->supports_packed_lane_jacobian()) {
            failure_reason = "packed W-lane path unsupported for curved (non-flat) metric";
            return false;
        }
        return jac_col_engine_.compute_packed_wlane8_columns_sparse(
            columns, drop_tol, emitters, failure_reason, selected_rows);
    }

    bool System_of_eqs::do_cols_J_wlane16_sparse(const std::array<int, 16>& columns,
                                                 double drop_tol,
                                                 std::array<SparseColumnEmitter, 16>& emitters,
                                                 std::string& failure_reason,
                                                 JacobianSelectedRows selected_rows)
    {
        if (met != nullptr && !met->supports_packed_lane_jacobian()) {
            failure_reason = "packed W-lane path unsupported for curved (non-flat) metric";
            return false;
        }
        return jac_col_engine_.compute_packed_wlane16_columns_sparse(
            columns, drop_tol, emitters, failure_reason, selected_rows);
    }

    bool System_of_eqs::do_cols_J_wlane32_sparse(const std::array<int, 32>& columns,
                                                 double drop_tol,
                                                 std::array<SparseColumnEmitter, 32>& emitters,
                                                 std::string& failure_reason,
                                                 JacobianSelectedRows selected_rows)
    {
        if (met != nullptr && !met->supports_packed_lane_jacobian()) {
            failure_reason = "packed W-lane path unsupported for curved (non-flat) metric";
            return false;
        }
        return jac_col_engine_.compute_packed_wlane32_columns_sparse(
            columns, drop_tol, emitters, failure_reason, selected_rows);
    }

    bool System_of_eqs::do_cols_J_wlane2_sparse(
        JacobianColumnEngine::Workspace& workspace,
        PackedJacobianColumns<2> columns, double drop_tol,
        PackedSparseColumnEmitters<2> emitters,
        std::string& failure_reason, JacobianSelectedRows selected_rows)
    {
        if (met != nullptr && !met->supports_packed_lane_jacobian()) {
            failure_reason =
                "packed W-lane path unsupported for curved (non-flat) metric";
            return false;
        }
        return jac_col_engine_.compute_packed_wlane2_columns_sparse(
            workspace, columns, drop_tol, emitters, failure_reason,
            selected_rows);
    }

    bool System_of_eqs::do_cols_J_wlane4_sparse(
        JacobianColumnEngine::Workspace& workspace,
        PackedJacobianColumns<4> columns, double drop_tol,
        PackedSparseColumnEmitters<4> emitters,
        std::string& failure_reason, JacobianSelectedRows selected_rows)
    {
        if (met != nullptr && !met->supports_packed_lane_jacobian()) {
            failure_reason =
                "packed W-lane path unsupported for curved (non-flat) metric";
            return false;
        }
        return jac_col_engine_.compute_packed_wlane4_columns_sparse(
            workspace, columns, drop_tol, emitters, failure_reason,
            selected_rows);
    }

    bool System_of_eqs::do_cols_J_wlane8_sparse(
        JacobianColumnEngine::Workspace& workspace,
        PackedJacobianColumns<8> columns, double drop_tol,
        PackedSparseColumnEmitters<8> emitters,
        std::string& failure_reason, JacobianSelectedRows selected_rows)
    {
        if (met != nullptr && !met->supports_packed_lane_jacobian()) {
            failure_reason =
                "packed W-lane path unsupported for curved (non-flat) metric";
            return false;
        }
        return jac_col_engine_.compute_packed_wlane8_columns_sparse(
            workspace, columns, drop_tol, emitters, failure_reason,
            selected_rows);
    }

    bool System_of_eqs::do_cols_J_wlane16_sparse(
        JacobianColumnEngine::Workspace& workspace,
        PackedJacobianColumns<16> columns, double drop_tol,
        PackedSparseColumnEmitters<16> emitters,
        std::string& failure_reason, JacobianSelectedRows selected_rows)
    {
        if (met != nullptr && !met->supports_packed_lane_jacobian()) {
            failure_reason =
                "packed W-lane path unsupported for curved (non-flat) metric";
            return false;
        }
        return jac_col_engine_.compute_packed_wlane16_columns_sparse(
            workspace, columns, drop_tol, emitters, failure_reason,
            selected_rows);
    }

    bool System_of_eqs::do_cols_J_wlane32_sparse(
        JacobianColumnEngine::Workspace& workspace,
        PackedJacobianColumns<32> columns, double drop_tol,
        PackedSparseColumnEmitters<32> emitters,
        std::string& failure_reason, JacobianSelectedRows selected_rows)
    {
        if (met != nullptr && !met->supports_packed_lane_jacobian()) {
            failure_reason =
                "packed W-lane path unsupported for curved (non-flat) metric";
            return false;
        }
        return jac_col_engine_.compute_packed_wlane32_columns_sparse(
            workspace, columns, drop_tol, emitters, failure_reason,
            selected_rows);
    }

    void System_of_eqs::dump_do_col_J_profile() const { jac_col_engine_.dump_profile(); }
    void System_of_eqs::dump_do_col_J_profile(
        JacobianColumnEngine::Workspace& workspace) const
    {
        jac_col_engine_.dump_profile(workspace);
    }

    void System_of_eqs::update_terms_from_variable_domains(const Array<int>& zedoms,
                                                           const std::vector<char>* term_filter,
                                                           const std::vector<char>* cst_filter)
    {
        update_terms_from_variable_domains_impl(
            zedoms, term_filter, cst_filter, VariableDomainTermUpdate::replace);
    }

    void System_of_eqs::accumulate_terms_from_variable_domains(
        const Array<int>& zedoms, const std::vector<char>* term_filter,
        const std::vector<char>* cst_filter)
    {
        update_terms_from_variable_domains_impl(
            zedoms, term_filter, cst_filter, VariableDomainTermUpdate::accumulate);
    }

    void System_of_eqs::update_terms_from_variable_domains_impl(
        const Array<int>& zedoms, const std::vector<char>* term_filter,
        const std::vector<char>* cst_filter, VariableDomainTermUpdate term_update)
    {
        assert(term_filter == nullptr ||
               static_cast<int>(term_filter->size()) == nterm);
        assert(cst_filter == nullptr ||
               static_cast<int>(cst_filter->size()) == nterm_cst);
        // Replacement callers pre-zero all terms before invoking this function;
        // the accumulation caller has already seeded field tangents. In both
        // modes, non-matching terms need no action here.
        // The optional filters come from the do_JX per-rank term closure
        // (DO_JX_TERM_CLOSURE): a masked-out term/cst keeps its previous
        // derivative, which no row owned by this rank reads.
        for (int i = 0; i < nterm; i++) {
            if (term_filter != nullptr && !(*term_filter)[i])
                continue;
            int dom = term[i]->get_dom();
            for (int d = 0; d < zedoms.get_size(0); d++)
                if (zedoms(d) == dom) {
                    if (term_update == VariableDomainTermUpdate::accumulate) {
                        espace.get_domain(dom)->accumulate_term_eq_mapping_derivative(
                            term[i].get());
                    } else {
                        espace.get_domain(dom)->update_term_eq(term[i].get());
                    }
                    break;
                }
        }

        for (int i = 0; i < nterm_cst; i++) {
            if (cst_filter != nullptr && !(*cst_filter)[i])
                continue;
            if (cst[i]->get_type_data() == TERM_T) {
                int dom = cst[i]->get_dom();
                for (int d = 0; d < zedoms.get_size(0); d++)
                    if (zedoms(d) == dom) {
                        espace.get_domain(dom)->update_term_eq(cst[i].get());
                        break;
                    }
            }
        }
    }
} // namespace Kadath
