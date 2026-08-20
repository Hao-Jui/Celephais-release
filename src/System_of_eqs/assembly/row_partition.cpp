// MPI row partition for the matrix-free Jv product (System_of_eqs::do_JX,
// gated by DO_JX_MPI) and, through the same split, for the nonlinear
// residual value pass (System_of_eqs::sec_member_partitioned, gated by
// SEC_MEMBER_MPI). Builds the contiguous field-equation split, the
// Allgatherv layout, the Eq_int part ownership, the exact per-equation
// definition closure, and the per-rank term/cst preamble closure
// (DO_JX_TERM_CLOSURE); the partitioned/replicated tails themselves
// stay in assembly/solver.cpp (they are entangled with the do_JX profiling
// scope).
#include "For_Kadath/System_of_eqs/system_of_eqs.hpp"
#include "For_Kadath/Utilities/runtime_env.hpp"
#include "mpi.h"

#include <algorithm>
#include <cstdlib>
#include <map>
#include <set>
#include <string>
#include <string_view>
#include <vector>

namespace Kadath
{
    namespace
    {
        bool do_jx_mpi_selftest_enabled()
        {
            static const bool enabled = env_flag_enabled("DO_JX_MPI_SELFTEST", true);
            return enabled;
        }

        bool do_jx_term_closure_enabled()
        {
            static const bool enabled = env_flag_enabled("DO_JX_TERM_CLOSURE", true);
            return enabled;
        }
    }

    // Derive everything that depends on the equation split (ownership,
    // Allgatherv layout, Eq_int part owners, definition closure) from a
    // given contiguous split. The split itself must be identical on every
    // rank; everything here is deterministic given the split.
    void System_of_eqs::derive_do_jx_partition_from_split(const std::vector<int>& eq_split)
    {
        auto& part = do_jx_mpi_;
        const int nproc = part.nproc;
        const int rank = part.rank;

        part.eq_begin = eq_split[rank];
        part.eq_end = eq_split[rank + 1];
        part.row_counts.assign(static_cast<std::size_t>(nproc), 0);
        part.row_displs.assign(static_cast<std::size_t>(nproc), 0);
        for (int r = 0; r < nproc; r++) {
            part.row_counts[r] =
                part.eq_row_begin[eq_split[r + 1]] - part.eq_row_begin[eq_split[r]];
            part.row_displs[r] = part.eq_row_begin[eq_split[r]] - neq_int;
        }

        // Domain owner map: the rank owning a domain's first field equation
        // owns the domain (Eq_int parts and the definition closure follow
        // it); domains with no field equation fall back to round-robin.
        const auto rank_of_eq = [&](int i) {
            return static_cast<int>(std::upper_bound(eq_split.begin() + 1, eq_split.end(), i) -
                                    (eq_split.begin() + 1));
        };
        const int ndoms = (dom_max >= dom_min) ? (dom_max - dom_min + 1) : 0;
        std::vector<int> domain_owner(static_cast<std::size_t>(ndoms), -1);
        for (int i = 0; i < neq; i++) {
            const int d = eq[i]->get_ndom();
            if (d >= dom_min && d <= dom_max && domain_owner[d - dom_min] < 0)
                domain_owner[d - dom_min] = rank_of_eq(i);
        }
        for (int d = 0; d < ndoms; d++)
            if (domain_owner[d] < 0)
                domain_owner[d] = d % nproc;

        // Eq_int integral parts: each part is domain-local, so its owner is
        // the domain owner (out-of-range domains go to rank 0).
        const int total_parts = part.part_offset[neq_int];
        part.part_owner.assign(static_cast<std::size_t>(total_parts), 0);
        part.part_counts.assign(static_cast<std::size_t>(nproc), 0);
        part.part_displs.assign(static_cast<std::size_t>(nproc), 0);
        for (int i = 0; i < neq_int; i++) {
            for (int k = 0; k < eq_int[i]->n_ope; k++) {
                const int d = eq_int[i]->parts[k]->get_dom();
                const bool in_range = (d >= dom_min && d <= dom_max);
                const int owner = in_range ? domain_owner[d - dom_min] : 0;
                part.part_owner[part.part_offset[i] + k] = owner;
                part.part_counts[owner]++;
            }
        }
        for (int r = 1; r < nproc; r++)
            part.part_displs[r] = part.part_displs[r - 1] + part.part_counts[r - 1];

        // Stable scratch shared by the partitioned residual and Jv tails.
        // Resize here, where the split-dependent counts are established,
        // rather than on every collective assembly.
        part.part_send_workspace.resize(
            static_cast<std::size_t>(part.part_counts[rank]));
        part.part_recv_workspace.resize(static_cast<std::size_t>(total_parts));
        part.owner_cursor_workspace.resize(static_cast<std::size_t>(nproc));

        // Exact definition closure: only the defs the owned equation rows
        // and owned Eq_int parts actually reference (def-to-def chains
        // already expanded in eq_defs/part_defs at build time). Defs
        // registered outside the domain range stay replicated everywhere
        // (mirrors the column engine's defs_global handling).
        part.def_needed.assign(static_cast<std::size_t>(ndef), 0);
        for (int i = part.eq_begin; i < part.eq_end; i++)
            for (const int j : part.eq_defs[i])
                part.def_needed[j] = 1;
        for (int p = 0; p < total_parts; p++)
            if (part.part_owner[p] == rank)
                for (const int j : part.part_defs[p])
                    part.def_needed[j] = 1;
        for (int i = 0; i < ndef; i++) {
            const int d = def[i]->get_dom();
            if (d < dom_min || d > dom_max)
                part.def_needed[i] = 1;
        }

        // Per-rank term/cst closure for the do_JX preamble
        // (DO_JX_TERM_CLOSURE): classify the Ope_id targets of the
        // owned field equations, the owned Eq_int parts, and every def in
        // def_needed (def-to-def chains are already expanded there, so each
        // needed def's DIRECT targets cover its whole chain) against the
        // term[] / cst[] Term_eq pointers the parser bound at registration
        // time. Targets of any other kind — def/def_glob results (handled by
        // def_needed / frozen at construction), metric internals and domain
        // normals (replicated update, untouched by the filter), term_double
        // entries (always seeded — trivial scalars), cst_hard literals
        // (der-zero constants) — are external to the filtered preamble and
        // intentionally ignored. The default-1 fallback keeps any accidental
        // read of an inactive filter harmless.
        part.term_needed.assign(static_cast<std::size_t>(nterm), 1);
        part.cst_needed.assign(static_cast<std::size_t>(nterm_cst), 1);
        if (part.term_filter_active) {
            part.term_needed.assign(static_cast<std::size_t>(nterm), 0);
            part.cst_needed.assign(static_cast<std::size_t>(nterm_cst), 0);
            std::map<const Term_eq*, int> term_index_of;
            std::map<const Term_eq*, int> cst_index_of;
            for (int i = 0; i < nterm; i++)
                term_index_of[term[i].get()] = i;
            for (int i = 0; i < nterm_cst; i++)
                cst_index_of[cst[i].get()] = i;

            std::set<const Term_eq*> targets;
            for (int i = part.eq_begin; i < part.eq_end; i++)
                for (int p = 0; p < eq[i]->n_ope; p++)
                    if (eq[i]->parts[p] != nullptr)
                        eq[i]->parts[p]->collect_def_targets(targets);
            for (int i = 0; i < neq_int; i++)
                for (int k = 0; k < eq_int[i]->n_ope; k++)
                    if (part.part_owner[part.part_offset[i] + k] == rank)
                        eq_int[i]->parts[k]->collect_def_targets(targets);
            for (int i = 0; i < ndef; i++)
                if (part.def_needed[i])
                    def[i]->collect_def_targets(targets);

            for (const Term_eq* t : targets) {
                const auto term_it = term_index_of.find(t);
                if (term_it != term_index_of.end()) {
                    part.term_needed[term_it->second] = 1;
                    continue;
                }
                const auto cst_it = cst_index_of.find(t);
                if (cst_it != cst_index_of.end())
                    part.cst_needed[cst_it->second] = 1;
            }
        }
        // Re-verify the filtered preamble on the first filtered call after
        // every (re)derivation — the closure sets above depend on the split.
        part.term_filter_selftest_pending = part.term_filter_active;
    }

    // Build (or rebuild) the row partition when the equation topology or
    // the communicator size changed. The initial split balances row counts
    // (the only structural cost proxy); the self-test reference pass then
    // measures real per-def/per-eq costs and triggers a one-shot
    // repartition. Deterministic for a given topology, so every rank
    // derives the identical initial partition without communication.
    void System_of_eqs::build_do_jx_mpi_partition(int nproc, int rank)
    {
        auto& part = do_jx_mpi_;
        part = DoJxMpiPartition{};
        part.nproc = nproc;
        part.rank = rank;
        part.neq = neq;
        part.neq_int = neq_int;
        part.nbr_conditions = nbr_conditions;

        // Row layout: eq_int rows first, then one contiguous row block per
        // field equation — the same layout export_der walks.
        part.eq_operator_offset.assign(static_cast<std::size_t>(neq) + 1, 0);
        part.eq_row_begin.assign(static_cast<std::size_t>(neq) + 1, neq_int);
        for (int i = 0; i < neq; i++) {
            part.eq_operator_offset[i + 1] = part.eq_operator_offset[i] + eq[i]->n_ope;
            part.eq_row_begin[i + 1] = part.eq_row_begin[i] + eq[i]->get_n_cond_tot();
        }
        if (part.eq_row_begin[neq] != nbr_conditions)
            return; // unexpected layout — stay on the replicated path

        part.part_offset.assign(static_cast<std::size_t>(neq_int) + 1, 0);
        for (int i = 0; i < neq_int; i++)
            part.part_offset[i + 1] = part.part_offset[i] + eq_int[i]->n_ope;

        // Exact per-equation definition references. Ope_id leaves hold the
        // Term_eq pointers the parser bound at registration time; a def
        // reference is a target equal to some def[i]->get_res() (compute_res
        // assigns *res in place, so the pointers are stable). Expand
        // def-to-def chains once here; a referenced def always has a lower
        // index than its referrer (the parser requires it to exist when the
        // referring expression is parsed), so computing flagged defs in
        // index order preserves chain order.
        {
            std::map<const Term_eq*, int> def_index_of_term;
            for (int i = 0; i < ndef; i++)
                def_index_of_term[def[i]->get_res()] = i;

            std::set<const Term_eq*> targets;
            const auto referenced_defs = [&](const std::set<const Term_eq*>& seen) {
                std::vector<int> refs;
                for (const Term_eq* t : seen) {
                    const auto it = def_index_of_term.find(t);
                    if (it != def_index_of_term.end())
                        refs.push_back(it->second);
                }
                return refs;
            };

            std::vector<std::vector<int>> def_direct(static_cast<std::size_t>(ndef));
            for (int i = 0; i < ndef; i++) {
                targets.clear();
                def[i]->collect_def_targets(targets);
                for (const int j : referenced_defs(targets))
                    if (j != i)
                        def_direct[i].push_back(j);
            }
            std::vector<std::vector<int>> def_closed(static_cast<std::size_t>(ndef));
            std::vector<char> visited(static_cast<std::size_t>(ndef), 0);
            std::vector<int> stack;
            for (int i = 0; i < ndef; i++) {
                std::fill(visited.begin(), visited.end(), 0);
                stack.assign(def_direct[i].begin(), def_direct[i].end());
                while (!stack.empty()) {
                    const int j = stack.back();
                    stack.pop_back();
                    if (visited[j])
                        continue;
                    visited[j] = 1;
                    def_closed[i].push_back(j);
                    stack.insert(stack.end(), def_direct[j].begin(), def_direct[j].end());
                }
                std::sort(def_closed[i].begin(), def_closed[i].end());
            }

            const auto closed_refs = [&]() {
                std::vector<int> refs = referenced_defs(targets);
                std::vector<char> in_set(static_cast<std::size_t>(ndef), 0);
                for (const int j : refs)
                    in_set[j] = 1;
                const std::size_t direct_count = refs.size();
                for (std::size_t k = 0; k < direct_count; k++)
                    for (const int j : def_closed[refs[k]])
                        if (!in_set[j]) {
                            in_set[j] = 1;
                            refs.push_back(j);
                        }
                std::sort(refs.begin(), refs.end());
                return refs;
            };

            part.eq_defs.assign(static_cast<std::size_t>(neq), {});
            for (int i = 0; i < neq; i++) {
                targets.clear();
                for (int p = 0; p < eq[i]->n_ope; p++)
                    if (eq[i]->parts[p] != nullptr)
                        eq[i]->parts[p]->collect_def_targets(targets);
                part.eq_defs[i] = closed_refs();
            }
            part.part_defs.assign(static_cast<std::size_t>(part.part_offset[neq_int]), {});
            for (int i = 0; i < neq_int; i++)
                for (int k = 0; k < eq_int[i]->n_ope; k++) {
                    targets.clear();
                    eq_int[i]->parts[k]->collect_def_targets(targets);
                    part.part_defs[part.part_offset[i] + k] = closed_refs();
                }
            // Persist the transitive def-to-def chains: give_val_def uses them
            // to lazily repair rank-local stale defs after a partitioned value
            // pass (ensure_def_values_current).
            part.def_closed = std::move(def_closed);
        }

        // Per-rank term/cst preamble closure (DO_JX_TERM_CLOSURE):
        // precompute the xx-block stride of every term so the filtered
        // xx_to_ders can jump skipped blocks. The stride is the same
        // Domain::nbr_unknowns(var, dom) count add_var used to size the
        // block affecte_tau consumes (keep-sets verified case-for-case for
        // the live domain classes); the sum cross-check below catches any
        // count/consumption drift at the structural level and the one-shot
        // bitwise self-test (term_filter_selftest_pending) catches the rest.
        // The filter must stay off when term derivatives are read OUTSIDE
        // the equation/def operator trees:
        //   - curved metrics: Metric::update rebuilds metric internals from
        //     the metric unknown's term val_t AND der_t, and that term never
        //     appears as an equation-tree target (Metric_flat::update is a
        //     no-op and its internals are der-zero constants — the same
        //     lane-safety predicate the packed W-lane path uses);
        //   - user operators (add_ope): the opaque Param* may capture fields
        //     whose reads are invisible to the Ope_id target collection.
        {
            part.term_xx_stride.assign(static_cast<std::size_t>(nterm), 0);
            long long consumed_unknowns =
                espace.nbr_unknowns_from_variable_domains() + nvar_double;
            for (int i = 0; i < nterm; i++) {
                const int d = term[i]->get_dom();
                part.term_xx_stride[i] =
                    espace.get_domain(d)->nbr_unknowns(*var[assoc_var[i]], d);
                consumed_unknowns += part.term_xx_stride[i];
            }
            const bool metric_reads_visible =
                (met == nullptr) || met->supports_packed_lane_jacobian();
            const bool user_ope_free = ((nopeuser == 0) && (nopeuser_bin == 0)) || user_opes_target_only_;
            part.term_filter_active = do_jx_term_closure_enabled() &&
                                      metric_reads_visible && user_ope_free &&
                                      (consumed_unknowns == nbr_unknowns);
        }

        const long long total_rows = nbr_conditions - neq_int;
        std::vector<int> eq_split(static_cast<std::size_t>(nproc) + 1, neq);
        eq_split[0] = 0;
        {
            int cursor = 0;
            for (int r = 1; r < nproc; r++) {
                const long long target = (total_rows * r) / nproc;
                while (cursor < neq &&
                       static_cast<long long>(part.eq_row_begin[cursor]) - neq_int < target)
                    ++cursor;
                eq_split[r] = cursor;
            }
        }
        derive_do_jx_partition_from_split(eq_split);

        part.active = true;
        part.selftest_pending = do_jx_mpi_selftest_enabled();
        part.val_active = true;
        part.val_selftest_pending = do_jx_mpi_selftest_enabled();
    }

    // One-shot measured-cost repartition: balance the makespan of
    // (owned equation apply+export) + (owned Eq_int integral parts) +
    // (definition closure compute) using the per-def/per-eq/per-part
    // costs measured by the self-test reference pass.
    // Rank 0's measurements drive the split; the split is broadcast so all
    // ranks derive the identical partition. Any contiguous split is
    // bit-exact, so a measured (machine-dependent) split is safe.
    void System_of_eqs::rebalance_do_jx_partition_from_measured_costs()
    {
        auto& part = do_jx_mpi_;
        const int nproc = part.nproc;

        std::vector<int> eq_split(static_cast<std::size_t>(nproc) + 1, neq);
        eq_split[0] = 0;
        if (part.rank == 0) {
            // Out-of-range defs replicate everywhere, so they do not
            // influence the split; in-range def costs are charged to a
            // block when one of its equations' exact closures pulls the
            // def in for the first time.
            std::vector<char> def_in_range(static_cast<std::size_t>(ndef), 0);
            double total_cost = 0.0;
            for (int i = 0; i < ndef; i++) {
                const int d = def[i]->get_dom();
                if (d >= dom_min && d <= dom_max) {
                    def_in_range[i] = 1;
                    total_cost += part.def_cost[i];
                }
            }

            // Eq_int parts follow their domain's owner, i.e. the block
            // containing the domain's first field equation (the same rule
            // derive_do_jx_partition_from_split applies), so each candidate
            // block is charged the parts it would own. Domains with no field
            // equation (round-robin fallback) are excluded from the model.
            const bool parts_measured =
                static_cast<int>(part.part_cost.size()) == part.part_offset[neq_int];
            const int ndoms = (dom_max >= dom_min) ? (dom_max - dom_min + 1) : 0;
            std::vector<int> first_eq_of_domain(static_cast<std::size_t>(ndoms), -1);
            for (int i = 0; i < neq; i++) {
                const int d = eq[i]->get_ndom();
                if (d >= dom_min && d <= dom_max && first_eq_of_domain[d - dom_min] < 0)
                    first_eq_of_domain[d - dom_min] = i;
            }
            std::vector<std::vector<int>> parts_of_eq(static_cast<std::size_t>(neq));
            if (parts_measured && neq > 0)
                for (int i = 0; i < neq_int; i++)
                    for (int k = 0; k < eq_int[i]->n_ope; k++) {
                        const int d = eq_int[i]->parts[k]->get_dom();
                        const bool in_range = (d >= dom_min && d <= dom_max);
                        // Out-of-range parts always live on rank 0, whose
                        // block starts at equation 0.
                        const int home_eq = in_range ? first_eq_of_domain[d - dom_min] : 0;
                        if (home_eq >= 0)
                            parts_of_eq[home_eq].push_back(part.part_offset[i] + k);
                    }

            double max_single = 0.0;
            for (int i = 0; i < neq; i++) {
                double pinned_cost = part.eq_cost[i];
                for (const int p : parts_of_eq[i])
                    pinned_cost += part.part_cost[p];
                total_cost += pinned_cost;
                max_single = std::max(max_single, pinned_cost);
            }

            // Greedy feasibility check for a makespan bound: walk the
            // equations in order, charging each block its equations, the
            // Eq_int parts that follow them, and the cost of every def
            // newly added to its exact closure.
            std::vector<char> def_in_block(static_cast<std::size_t>(ndef), 0);
            std::vector<int> candidate_split;
            const auto charge_equation = [&](int i) {
                // Marks every def the equation (or its Eq_int parts) newly
                // pulls into the block and charges it once, mirroring the
                // def_needed dedup in derive_do_jx_partition_from_split.
                double marginal = part.eq_cost[i];
                const auto charge_def = [&](int j) {
                    if (!def_in_block[j]) {
                        def_in_block[j] = 1;
                        if (def_in_range[j])
                            marginal += part.def_cost[j];
                    }
                };
                for (const int j : part.eq_defs[i])
                    charge_def(j);
                for (const int p : parts_of_eq[i]) {
                    marginal += part.part_cost[p];
                    for (const int j : part.part_defs[p])
                        charge_def(j);
                }
                return marginal;
            };
            const auto fits_in_blocks = [&](double bound) {
                candidate_split.assign(1, 0);
                std::fill(def_in_block.begin(), def_in_block.end(), 0);
                double block_cost = 0.0;
                for (int i = 0; i < neq; i++) {
                    const bool block_empty = (candidate_split.back() == i);
                    double marginal = charge_equation(i);
                    if (!block_empty && block_cost + marginal > bound) {
                        // charge_equation marked equation i's defs against
                        // the old block; the wholesale reset discards them
                        // before recharging against the fresh block.
                        candidate_split.push_back(i);
                        std::fill(def_in_block.begin(), def_in_block.end(), 0);
                        block_cost = 0.0;
                        marginal = charge_equation(i);
                    }
                    block_cost += marginal;
                    if (static_cast<int>(candidate_split.size()) > nproc)
                        return false;
                }
                // The final block accumulates without triggering a cut, so
                // its cost must be checked against the bound as well.
                return static_cast<int>(candidate_split.size()) <= nproc &&
                       block_cost <= bound;
            };

            if (total_cost > 0.0) {
                double lo = max_single;
                double hi = total_cost;
                for (int iter = 0; iter < 50; iter++) {
                    const double mid = 0.5 * (lo + hi);
                    if (fits_in_blocks(mid))
                        hi = mid;
                    else
                        lo = mid;
                }
                if (fits_in_blocks(hi)) {
                    for (std::size_t r = 1; r < candidate_split.size(); r++)
                        eq_split[r] = candidate_split[r];
                    for (std::size_t r = candidate_split.size();
                         r < eq_split.size() - 1; r++)
                        eq_split[r] = neq;
                }
            }
        }
        MPI_Bcast(eq_split.data(), nproc + 1, MPI_INT, 0, MPI_COMM_WORLD);
        derive_do_jx_partition_from_split(eq_split);
        part.rebalanced = true;
        // Re-verify the rebalanced partition on the next call (matvec and
        // value paths each re-verify independently).
        part.selftest_pending = do_jx_mpi_selftest_enabled();
        part.val_selftest_pending = do_jx_mpi_selftest_enabled();
    }
} // namespace Kadath
