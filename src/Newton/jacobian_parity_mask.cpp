/*
    Added 2026.

    Column and row grading for the y = 0 reflection-parity mask; see
    Linear_algebra/jacobian_parity_mask.hpp for the layout argument.  The
    assembler owns the entry sweeps (the rank-local COO storage is private to
    it); this file owns the grading and the engage / disable policy.
*/

#include "Linear_algebra/jacobian_parity_mask.hpp"

#include "For_Kadath/Array/exceptions.hpp"
#include "For_Kadath/Space/space.hpp"
#include "For_Kadath/System_of_eqs/Jacobian/column_types.hpp"
#include "For_Kadath/System_of_eqs/system_of_eqs.hpp"
#include "For_Kadath/Tensor/tensor.hpp"
#include "For_Kadath/Term_eq/term_eq.hpp"
#include "For_Kadath/Utilities/runtime_env.hpp"
#include "For_Kadath/Val_domain/val_domain.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <utility>

namespace Kadath
{
    namespace
    {
        // Unique azimuthal index of one coefficient array position.  The
        // coefficient array is stored with phi fastest (affecte_tau builds the
        // offset as i*nt*np + j*np + k), so k is simply offset % np.
        int phi_index_of_offset(std::size_t coefficient_offset, int phi_size)
        {
            return static_cast<int>(coefficient_offset %
                                    static_cast<std::size_t>(phi_size));
        }

        // Unique azimuthal index carried by a probed Val_domain, or -1 when the
        // probe wrote nothing.  Returns -2 when the writes straddle more than
        // one azimuthal index, which would invalidate the whole grading.
        int phi_index_of_val_domain(const Val_domain& value, int phi_size)
        {
            if (value.check_if_zero())
                return -1;
            const Array<double>& coefficients = value.get_coef_ref();
            int found = -1;
            const std::size_t count = static_cast<std::size_t>(coefficients.get_nbr());
            for (std::size_t offset = 0; offset < count; ++offset) {
                if (coefficients.get_data()[offset] == 0.0)
                    continue;
                const int phi = phi_index_of_offset(offset, phi_size);
                if (found < 0)
                    found = phi;
                else if (found != phi)
                    return -2;
            }
            return found;
        }

        // The phi basis is normally one enum shared by the whole component.
        // Return zero for an undefined, empty, or nonuniform basis so grading
        // can refuse the mask instead of guessing a convention.
        int uniform_phi_basis_of_val_domain(const Val_domain& value)
        {
            const Base_spectral& base = value.get_base();
            if (!base.is_def())
                return 0;
            const Array<int>* phi_bases = base.get_base_1d(2);
            if (phi_bases == nullptr || phi_bases->get_nbr() == 0)
                return 0;
            const int basis = phi_bases->get_data()[0];
            for (std::size_t index = 1;
                 index < static_cast<std::size_t>(phi_bases->get_nbr());
                 ++index) {
                if (phi_bases->get_data()[index] != basis)
                    return 0;
            }
            return basis;
        }

        // Variable names reach the column map padded (add_var normalizes to a
        // fixed-width token), so every comparison has to trim first.
        std::string trim_ascii_space(const std::string& text)
        {
            const std::size_t first = text.find_first_not_of(" \t\n\r");
            if (first == std::string::npos)
                return {};
            const std::size_t last = text.find_last_not_of(" \t\n\r");
            return text.substr(first, last - first + 1);
        }

        bool output_path_active(const char* path)
        {
            return path != nullptr && path[0] != '\0' &&
                   std::string(path) != "0";
        }

        JacobianEmissionBlockFingerprint fingerprint_selection_plan(
            const std::shared_ptr<const JacobianSelectionPlan>& plan)
        {
            JacobianEmissionBlockFingerprint fingerprint;
            if (!plan)
                return fingerprint;
            fingerprint.parity_label = plan->selected_block();
            fingerprint.selected_rows = plan->selected_rows();
            fingerprint.selected_columns = plan->selected_columns();
            return fingerprint;
        }

        void refresh_jacobian_emission_fingerprint(
            JacobianEmissionPlan& plan, int dimension,
            const std::shared_ptr<JacobianParityMaskState>& parity_state,
            bool parity_split_requested, bool parity_split_ready)
        {
            JacobianEmissionFingerprint fingerprint;
            fingerprint.kind = plan.kind;
            fingerprint.full_dimension = dimension;
            fingerprint.assembled_dimension = plan.assembly_selection_plan
                ? static_cast<int>(
                      plan.assembly_selection_plan->selected_columns().size())
                : dimension;
            fingerprint.selection_plan_requested =
                plan.selection_plan_requested;
            fingerprint.parity_mask_requested = plan.parity_mask_requested;
            fingerprint.fused_parity_mask_requested =
                plan.fused_parity_mask_requested;
            fingerprint.fused_emission_active = plan.fused_emission_active;
            fingerprint.fused_verify_active = plan.fused_verify_active;
            fingerprint.speculative_j1_fusion = plan.speculative_j1_fusion;
            fingerprint.local_coo_blocks_requested =
                plan.local_coo_blocks_requested;
            fingerprint.physical_block_emission_requested =
                plan.physical_block_emission_requested;
            fingerprint.parity_split_requested = parity_split_requested;
            fingerprint.parity_split_ready = parity_split_ready;
            fingerprint.assembly_block =
                fingerprint_selection_plan(plan.assembly_selection_plan);
            for (std::size_t block = 0; block < plan.block_plans.size(); ++block) {
                fingerprint.payload_blocks[block] =
                    fingerprint_selection_plan(plan.block_plans[block]);
            }
            if (parity_split_requested && parity_state &&
                parity_state->n == dimension &&
                parity_state->row_sector.size() ==
                    static_cast<std::size_t>(dimension) &&
                parity_state->column_sector.size() ==
                    static_cast<std::size_t>(dimension)) {
                fingerprint.row_sector = parity_state->row_sector;
                fingerprint.column_sector = parity_state->column_sector;
            }
            plan.fingerprint = std::move(fingerprint);
        }

    } // namespace

    void JacobianEmissionPlan::route_payload_to_combined(std::string reason)
    {
        kind = Kind::Combined;
        block_plans = {};
        refusal_reason = std::move(reason);
        fingerprint.kind = kind;
        fingerprint.payload_blocks = {};
    }

    void JacobianEmissionPlan::demote_fused_emission(std::string reason)
    {
        route_payload_to_combined(std::move(reason));
        fused_emission_active = false;
        fused_verify_active = false;
        speculative_j1_fusion = false;
        fingerprint.fused_emission_active = false;
        fingerprint.fused_verify_active = false;
        fingerprint.speculative_j1_fusion = false;
    }

    void JacobianEmissionPlan::accept_speculative_fused_emission() noexcept
    {
        speculative_j1_fusion = false;
        fingerprint.speculative_j1_fusion = false;
    }

    JacobianEmissionPlan plan_jacobian_emission(
        System_of_eqs& system, int dimension,
        const std::shared_ptr<const JacobianSelectionPlan>& step_selection_plan,
        const JacobianEmissionCaps& caps)
    {
        JacobianEmissionPlan plan;
        const SolverRuntimeConfig& config = system.get_solver_runtime_config();
        plan.selection_plan_requested = caps.selected_block_supported &&
            config.sparse_sector_reduce && !caps.parity_mass_probe_requested;
        plan.parity_mask_requested = config.sparse_parity_mask;
        plan.local_coo_blocks_requested =
            env_flag_enabled("JACOBIAN_LOCAL_COO_BLOCKS");
        plan.fused_parity_mask_requested =
            env_flag_enabled("JACOBIAN_FUSED_PARITY_MASK", true);
        const bool fused_verify_requested =
            env_flag_enabled("JACOBIAN_FUSED_PARITY_VERIFY", false);

        const std::shared_ptr<JacobianParityMaskState>& parity_state_at_entry =
            system.jacobian_parity_mask_state();
        const bool selection_plan_current = plan.selection_plan_requested &&
            step_selection_plan && parity_state_at_entry &&
            parity_state_at_entry->n == dimension &&
            parity_state_at_entry->reduction_decision ==
                JacobianParityMaskState::ReductionDecision::Eligible &&
            parity_state_at_entry->selection_plan.get() ==
                step_selection_plan.get();
        if (selection_plan_current)
            plan.assembly_selection_plan = step_selection_plan;

        if (plan.fused_parity_mask_requested && plan.parity_mask_requested &&
            !plan.assembly_selection_plan && !plan.selection_plan_requested &&
            !plan.local_coo_blocks_requested &&
            (!parity_state_at_entry || parity_state_at_entry->n != dimension)) {
            JacobianParityColumnGrading grading =
                grade_jacobian_parity_columns(system);
            if (jacobian_parity_column_grading_disable_reason(grading).empty()) {
                JacobianParityRowPrediction row_prediction =
                    predict_jacobian_parity_rows(system);
                const bool labels_complete =
                    row_prediction.all_rows_available &&
                    jacobian_parity_sector_labels_complete(grading.sector,
                                                           dimension) &&
                    jacobian_parity_sector_labels_complete(row_prediction.sector,
                                                           dimension);
                if (labels_complete) {
                    const auto symmetric_columns = std::count(
                        grading.sector.begin(), grading.sector.end(),
                        static_cast<signed char>(1));
                    const auto symmetric_rows = std::count(
                        row_prediction.sector.begin(),
                        row_prediction.sector.end(),
                        static_cast<signed char>(1));
                    if (symmetric_rows == symmetric_columns) {
                        auto state = std::make_shared<JacobianParityMaskState>();
                        state->n = dimension;
                        state->column_sector = std::move(grading.sector);
                        state->row_sector = std::move(row_prediction.sector);
                        state->row_grading_source_label = "structural";
                        state->structural_labels_available = true;
                        system.jacobian_parity_mask_state() = std::move(state);
                        plan.speculative_j1_fusion = true;
                    }
                }
            }
        }

        const std::shared_ptr<JacobianParityMaskState>& parity_state =
            system.jacobian_parity_mask_state();
        plan.fused_emission_active = plan.speculative_j1_fusion ||
            (plan.fused_parity_mask_requested && plan.parity_mask_requested &&
             !plan.assembly_selection_plan &&
             !plan.local_coo_blocks_requested && parity_state &&
             jacobian_fused_parity_mask_ready(*parity_state, dimension));
        plan.fused_verify_active =
            plan.fused_emission_active && fused_verify_requested;

        const bool physical_payload_allowed =
            caps.physical_payload_supported &&
            !caps.analyze_reuse_requested &&
            !caps.replay_capture_requested &&
            !caps.parity_mass_probe_requested;
        const bool fused_parity_blocks_requested =
            plan.fused_parity_mask_requested && !fused_verify_requested;
        plan.physical_block_emission_requested = physical_payload_allowed &&
            (static_cast<bool>(step_selection_plan) ||
             fused_parity_blocks_requested);
        plan.centralized_coo_diagnostic_requested =
            output_path_active(std::getenv("JACOBIAN_SELECTED_ROW_ENTRIES_CSV")) ||
            output_path_active(std::getenv("JACOBIAN_COO_BYTE_HASH")) ||
            output_path_active(
                std::getenv("JACOBIAN_CANONICAL_COO_BYTE_HASH")) ||
            output_path_active(std::getenv("JACOBIAN_PARITY_MASS"));

        if (plan.physical_block_emission_requested &&
            plan.assembly_selection_plan) {
            if (plan.assembly_selection_plan->selected_block() != +1) {
                KADATH_THROW(
                    "physical selected Jacobian block must carry parity label +1");
            }
            plan.kind = JacobianEmissionPlan::Kind::ReducedPlus;
            plan.block_plans[0] = plan.assembly_selection_plan;
        } else if (plan.physical_block_emission_requested &&
                   plan.fused_emission_active &&
                   !plan.fused_verify_active) {
            plan.kind = JacobianEmissionPlan::Kind::FusedPair;
            const std::vector<JacobianSelectionPlan::BlockLabel> row_labels(
                parity_state->row_sector.begin(), parity_state->row_sector.end());
            const std::vector<JacobianSelectionPlan::BlockLabel> column_labels(
                parity_state->column_sector.begin(),
                parity_state->column_sector.end());
            for (std::size_t block = 0; block < plan.block_plans.size(); ++block) {
                const int label = block == 0 ? +1 : -1;
                JacobianSelectionPlanBuild built = make_jacobian_selection_plan(
                    row_labels, column_labels, label, -label);
                if (!built.plan) {
                    KADATH_THROW(
                        "physical fused parity-block plan is invalid: " +
                        built.fallback_reason);
                }
                plan.block_plans[block] = std::move(built.plan);
            }
        } else {
            plan.kind = JacobianEmissionPlan::Kind::Combined;
            if (!physical_payload_allowed) {
                plan.refusal_reason = !caps.physical_payload_supported
                    ? "caller requires centralized COO"
                    : (caps.analyze_reuse_requested
                           ? "MUMPS analyze reuse requires centralized COO"
                           : (caps.replay_capture_requested
                                  ? "replay capture requires centralized COO"
                                  : "parity-mass probe requires centralized COO"));
            } else if (fused_verify_requested) {
                plan.refusal_reason =
                    "fused verification requires centralized COO";
            } else {
                plan.refusal_reason = "physical parity payload is not ready";
            }
        }
        if (plan.physical_block_emission_requested &&
            plan.centralized_coo_diagnostic_requested) {
            plan.route_payload_to_combined(
                "centralized COO diagnostic requires Combined payload");
        }

        const bool parity_split_requested = !step_selection_plan &&
            (config.sparse_parity_split_solve ||
             plan.physical_block_emission_requested);
        const bool parity_structure_valid = parity_split_requested &&
            parity_state && parity_state->n == dimension &&
            parity_state->row_sector.size() ==
                static_cast<std::size_t>(dimension) &&
            parity_state->column_sector.size() ==
                static_cast<std::size_t>(dimension);
        const bool parity_split_ready = parity_structure_valid &&
            jacobian_parity_split_ready_for_next_emission(
                *parity_state, dimension, plan.parity_mask_requested,
                plan.fused_parity_mask_requested, fused_verify_requested);
        refresh_jacobian_emission_fingerprint(
            plan, dimension, parity_state, parity_split_requested,
            parity_split_ready);
        return plan;
    }

    int jacobian_parity_field_grading(const std::string& variable_name,
                                      int component)
    {
        const std::string name = trim_ascii_space(variable_name);
        if (name == "P" || name == "N" || name == "H" || name == "xiScal" ||
            name == "varscal" || name == "cstB")
            return +1;
        if (name == "phi")
            return -1;
        if (name == "bet")
            return (component == 1) ? +1 : -1; // components 0,1,2 = x,y,z
        return 0;
    }

    int jacobian_parity_global_grading(const std::string& variable_name)
    {
        const std::string name = trim_ascii_space(variable_name);
        if (name == "ome" || name == "xaxis" || name == "Hc" ||
            name == "Hc1" || name == "Hc2" || name == "Hscale" ||
            name == "Hscale1" || name == "Hscale2" || name == "Mb" ||
            name == "Madm" || name == "qlMadm" || name == "qlMadm1" ||
            name == "qlMadm2" || name == "omes1" || name == "omes2" ||
            name == "muz1" || name == "muz2" || name == "cstB")
            return +1;
        if (name == "yaxis" || name == "zvel" || name == "adot")
            return -1;
        return 0;
    }

    bool System_of_eqs::build_column_phi_and_component_indices(
        std::vector<int>& phi_index, std::vector<int>& phi_basis,
        std::vector<int>& phi_domain, std::vector<int>& component_index,
        int& unsupported_tau_seed_domain,
        bool& unsupported_variable_domain_layout) const
    {
        unsupported_tau_seed_domain = -1;
        unsupported_variable_domain_layout = false;
        phi_index.assign(static_cast<std::size_t>(nbr_unknowns), -1);
        phi_basis.assign(static_cast<std::size_t>(nbr_unknowns), 0);
        phi_domain.assign(static_cast<std::size_t>(nbr_unknowns), -1);
        component_index.assign(static_cast<std::size_t>(nbr_unknowns), -1);

        // Surface / variable-domain unknowns occupy the leading columns.  Probe
        // them one-hot through the production xx_to_vars_from_adapted walk.
        const int variable_domain_columns =
            espace.nbr_unknowns_from_variable_domains();
        if (variable_domain_columns > 0) {
            std::vector<VariableDomainBlock> blocks;
            if (!espace.describe_variable_domain_blocks(blocks)) {
                unsupported_variable_domain_layout = true;
                return false;
            }
            int next_column = 0;
            Array<double> one_hot(nbr_unknowns);
            for (const VariableDomainBlock& block : blocks) {
                if (block.domain < 0 ||
                    block.domain >= espace.get_nbr_domains() ||
                    block.first_column != next_column || block.column_count <= 0 ||
                    block.first_column + block.column_count >
                        variable_domain_columns) {
                    unsupported_variable_domain_layout = true;
                    return false;
                }
                const Domain* domain = espace.get_domain(block.domain);
                if (domain->get_ndim() < 3 ||
                    domain->nbr_unknowns_from_adapted() != block.column_count) {
                    unsupported_variable_domain_layout = true;
                    return false;
                }
                const int phi_size = domain->get_nbr_coefs()(2);
                const int end_column =
                    block.first_column + block.column_count;
                for (int column = block.first_column; column < end_column;
                     ++column) {
                    one_hot = 0.0;
                    one_hot.set(column) = 1.0;
                    Val_domain shape(domain);
                    shape.std_base();
                    int position = block.first_column;
                    domain->xx_to_vars_from_adapted(shape, one_hot, position);
                    if (position != end_column) {
                        unsupported_variable_domain_layout = true;
                        return false;
                    }
                    phi_index[static_cast<std::size_t>(column)] =
                        phi_index_of_val_domain(shape, phi_size);
                    phi_basis[static_cast<std::size_t>(column)] =
                        uniform_phi_basis_of_val_domain(shape);
                    phi_domain[static_cast<std::size_t>(column)] = block.domain;
                }
                next_column = end_column;
            }
            if (next_column != variable_domain_columns) {
                unsupported_variable_domain_layout = true;
                return false;
            }
        }

        // The numeric globals follow; they carry no azimuthal index and keep -1.

        // Field unknowns: one descriptor block per Term_eq, emitted in
        // affecte_tau_one_coef order, i.e. exactly the assembler's column order.
        std::vector<ColumnInfo> column_map;
        build_column_map(column_map, false);
        std::vector<int> term_start_column(static_cast<std::size_t>(nterm), -1);
        for (int column = 0; column < static_cast<int>(column_map.size()); ++column) {
            const ColumnInfo& info = column_map[static_cast<std::size_t>(column)];
            if (info.term_idx >= 0 && info.term_idx < nterm &&
                term_start_column[static_cast<std::size_t>(info.term_idx)] < 0) {
                term_start_column[static_cast<std::size_t>(info.term_idx)] = column;
            }
        }

        std::vector<TauSeedDescriptor> descriptors;
        for (int term_index = 0; term_index < nterm; ++term_index) {
            const int start = term_start_column[static_cast<std::size_t>(term_index)];
            if (start < 0)
                continue;
            const int domain_index = term[static_cast<std::size_t>(term_index)]->get_dom();
            const Domain* domain = espace.get_domain(domain_index);
            const int phi_size = domain->get_nbr_coefs()(2);
            const Tensor& model = term[static_cast<std::size_t>(term_index)]->get_val_t();
            if (!domain->describe_tau_seed_block(model, domain_index, descriptors)) {
                unsupported_tau_seed_domain = domain_index;
                return false;
            }
            for (std::size_t local = 0; local < descriptors.size(); ++local) {
                const TauSeedDescriptor& descriptor = descriptors[local];
                const int column = start + static_cast<int>(local);
                if (column >= nbr_unknowns)
                    KADATH_THROW("build_column_phi_and_component_indices: column overflow");
                int phi = -1;
                for (int write = 0; write < descriptor.write_count; ++write) {
                    const int candidate = phi_index_of_offset(
                        descriptor.writes[static_cast<std::size_t>(write)]
                            .coefficient_offset,
                        phi_size);
                    if (phi < 0)
                        phi = candidate;
                    else if (phi != candidate)
                        phi = -2; // Galerkin partners straddle two phi indices.
                }
                phi_index[static_cast<std::size_t>(column)] = phi;
                component_index[static_cast<std::size_t>(column)] = descriptor.component;
                phi_domain[static_cast<std::size_t>(column)] = domain_index;
                if (descriptor.component >= 0 &&
                    descriptor.component < model.get_n_comp()) {
                    const Array<int> component(
                        model.indices(descriptor.component));
                    phi_basis[static_cast<std::size_t>(column)] =
                        uniform_phi_basis_of_val_domain(
                            model(component)(domain_index));
                }
            }
        }
        return true;
    }

    bool System_of_eqs::read_inactive_jacobian_state(
        const JacobianSelectionPlan& plan, std::vector<double>& values,
        std::string& failure_reason) const
    {
        values.clear();
        failure_reason.clear();
        const std::vector<int>& selected = plan.selected_columns();
        std::vector<unsigned char> active(
            static_cast<std::size_t>(nbr_unknowns), 0);
        for (std::size_t position = 0; position < selected.size(); ++position) {
            const int column = selected[position];
            if (column < 0 || column >= nbr_unknowns ||
                (position > 0 && column <= selected[position - 1])) {
                failure_reason = "selection columns are malformed";
                return false;
            }
            active[static_cast<std::size_t>(column)] = 1;
        }

        std::vector<ColumnInfo> column_map;
        build_column_map(column_map, false);
        if (column_map.size() != static_cast<std::size_t>(nbr_unknowns)) {
            failure_reason = "column map size does not match the unknown count";
            return false;
        }

        // Variable-domain coordinates do not have a general inverse map.  The
        // production adapted-nosym layout does expose one live radius and a
        // forward one-hot map.  An exclusive coefficient pivot recovers each
        // independent coordinate without reading a shared Galerkin anchor.
        const int variable_domain_count =
            espace.nbr_unknowns_from_variable_domains();
        if (variable_domain_count > 0) {
            std::vector<VariableDomainBlock> blocks;
            if (!espace.describe_variable_domain_blocks(blocks)) {
                failure_reason =
                    "space does not describe variable-domain blocks";
                return false;
            }
            int next_column = 0;
            Array<double> one_hot(nbr_unknowns);
            for (const VariableDomainBlock& block : blocks) {
                if (block.domain < 0 ||
                    block.domain >= espace.get_nbr_domains() ||
                    block.first_column != next_column ||
                    block.column_count <= 0 ||
                    block.first_column + block.column_count >
                        variable_domain_count) {
                    failure_reason =
                        "variable-domain block layout is malformed";
                    return false;
                }
                const int end = block.first_column + block.column_count;
                bool has_inactive = false;
                for (int column = block.first_column; column < end; ++column)
                    has_inactive = has_inactive ||
                        active[static_cast<std::size_t>(column)] == 0;
                if (!has_inactive) {
                    next_column = end;
                    continue;
                }

                const Domain* const domain = espace.get_domain(block.domain);
                if (domain->nbr_unknowns_from_adapted() !=
                    block.column_count) {
                    failure_reason =
                        "variable-domain block size disagrees with its domain";
                    return false;
                }
                std::vector<Val_domain> live_mapping;
                domain->snapshot_mapping(live_mapping);
                if (live_mapping.size() != 1) {
                    failure_reason =
                        "variable-domain block does not expose exactly one live mapping component";
                    return false;
                }
                const Array<double>& live_coefficients =
                    live_mapping.front().get_coef_ref();
                const std::size_t coefficient_count =
                    static_cast<std::size_t>(live_coefficients.get_nbr());
                std::vector<std::vector<std::pair<std::size_t, double>>> seeds(
                    static_cast<std::size_t>(block.column_count));
                std::vector<int> support_count(coefficient_count, 0);

                for (int column = block.first_column; column < end; ++column) {
                    one_hot = 0.0;
                    one_hot.set(column) = 1.0;
                    Val_domain shape(domain);
                    shape.std_base();
                    int position = block.first_column;
                    domain->xx_to_vars_from_adapted(
                        shape, one_hot, position);
                    if (position != end) {
                        failure_reason =
                            "variable-domain one-hot walk consumed the wrong number of columns";
                        return false;
                    }
                    const Array<double>& coefficients = shape.get_coef_ref();
                    if (static_cast<std::size_t>(coefficients.get_nbr()) !=
                        coefficient_count) {
                        failure_reason =
                            "variable-domain one-hot and live mapping shapes differ";
                        return false;
                    }
                    auto& writes = seeds[static_cast<std::size_t>(
                        column - block.first_column)];
                    for (std::size_t offset = 0; offset < coefficient_count;
                         ++offset) {
                        const double value = coefficients.get_data()[offset];
                        if (!std::isfinite(value)) {
                            failure_reason =
                                "variable-domain one-hot coefficient is non-finite";
                            return false;
                        }
                        if (value == 0.0)
                            continue;
                        writes.emplace_back(offset, value);
                        ++support_count[offset];
                    }
                    if (writes.empty()) {
                        failure_reason =
                            "variable-domain one-hot column has no coefficient support";
                        return false;
                    }
                }

                for (int column = block.first_column; column < end; ++column) {
                    if (active[static_cast<std::size_t>(column)] != 0)
                        continue;
                    const auto& writes = seeds[static_cast<std::size_t>(
                        column - block.first_column)];
                    const auto pivot = std::find_if(
                        writes.begin(), writes.end(),
                        [&](const auto& write) {
                            return support_count[write.first] == 1;
                        });
                    if (pivot == writes.end()) {
                        failure_reason =
                            "variable-domain column has no exclusive coefficient pivot";
                        return false;
                    }
                    const double live =
                        live_coefficients.get_data()[pivot->first];
                    const double coordinate = live / pivot->second;
                    if (!std::isfinite(coordinate)) {
                        failure_reason =
                            "variable-domain coordinate is non-finite";
                        return false;
                    }
                    values.push_back(coordinate);
                }
                next_column = end;
            }
            if (next_column != variable_domain_count) {
                failure_reason =
                    "variable-domain blocks do not cover their columns";
                return false;
            }
        }

        std::vector<std::vector<TauSeedDescriptor>> term_descriptors(
            static_cast<std::size_t>(nterm));
        std::vector<unsigned char> term_descriptor_ready(
            static_cast<std::size_t>(nterm), 0);
        for (int column = variable_domain_count; column < nbr_unknowns;
             ++column) {
            if (active[static_cast<std::size_t>(column)] != 0)
                continue;
            const ColumnInfo& info =
                column_map[static_cast<std::size_t>(column)];
            double coordinate = 0.0;
            if (info.var_double_idx >= 0) {
                if (info.var_double_idx >= nvar_double) {
                    failure_reason =
                        "numeric-global column index is malformed";
                    return false;
                }
                coordinate = *var_double[static_cast<std::size_t>(
                    info.var_double_idx)];
            } else {
                if (info.var_idx < 0 || info.var_idx >= nvar ||
                    info.term_idx < 0 || info.term_idx >= nterm ||
                    info.domain < dom_min || info.domain > dom_max ||
                    info.basis_mode < 0) {
                    failure_reason = "field column metadata is malformed";
                    return false;
                }
                auto& descriptors = term_descriptors[static_cast<std::size_t>(
                    info.term_idx)];
                if (term_descriptor_ready[static_cast<std::size_t>(
                        info.term_idx)] == 0) {
                    const Domain* const domain =
                        espace.get_domain(info.domain);
                    const Tensor& model =
                        term[static_cast<std::size_t>(info.term_idx)]->get_val_t();
                    if (!domain->describe_tau_seed_block(
                            model, info.domain, descriptors)) {
                        failure_reason =
                            "field domain does not describe its tau seed block";
                        return false;
                    }
                    term_descriptor_ready[static_cast<std::size_t>(
                        info.term_idx)] = 1;
                }
                if (info.basis_mode >=
                    static_cast<int>(descriptors.size())) {
                    failure_reason =
                        "field tau-seed index is outside its descriptor block";
                    return false;
                }
                const TauSeedDescriptor& descriptor = descriptors[
                    static_cast<std::size_t>(info.basis_mode)];
                if (descriptor.component < 0 ||
                    descriptor.component >=
                        var[static_cast<std::size_t>(info.var_idx)]->get_n_comp() ||
                    descriptor.write_count <= 0 ||
                    descriptor.writes[0].value == 0.0) {
                    failure_reason =
                        "field tau-seed primary write is malformed";
                    return false;
                }
                const Tensor& field =
                    *var[static_cast<std::size_t>(info.var_idx)];
                const Array<int> component(
                    field.indices(descriptor.component));
                const Val_domain& value = field(component)(info.domain);
                double live = 0.0;
                if (!value.check_if_zero()) {
                    const Array<double>& coefficients = value.get_coef_ref();
                    if (descriptor.writes[0].coefficient_offset >=
                        static_cast<std::size_t>(coefficients.get_nbr())) {
                        failure_reason =
                            "field tau-seed primary write is outside coefficient storage";
                        return false;
                    }
                    live = coefficients.get_data()[
                        descriptor.writes[0].coefficient_offset];
                }
                coordinate = live / descriptor.writes[0].value;
            }
            if (!std::isfinite(coordinate)) {
                failure_reason = "inactive state coefficient is non-finite";
                return false;
            }
            values.push_back(coordinate);
        }
        return true;
    }

    JacobianParityColumnGrading
    grade_jacobian_parity_columns(const System_of_eqs& system)
    {
        std::vector<ColumnInfo> column_map;
        system.build_column_map(column_map, false);

        JacobianParityColumnGrading grading;
        if (!system.build_column_phi_and_component_indices(
                grading.phi_index, grading.phi_basis, grading.phi_domain,
                grading.component_index,
                grading.unsupported_tau_seed_domain,
                grading.unsupported_variable_domain_layout)) {
            return grading;
        }
        const std::size_t n = grading.phi_index.size();
        if (column_map.size() != n)
            KADATH_THROW("grade_jacobian_parity_columns: column map size != n");
        grading.sector.assign(n, 0);

        for (std::size_t column = 0; column < n; ++column) {
            const ColumnInfo& info = column_map[column];
            const int phi = grading.phi_index[column];
            if (phi == -2)
                ++grading.mixed_phi_columns;
            int phi_parity = +1;
            if (phi >= 0) {
                const int domain_index = grading.phi_domain[column];
                const int basis = grading.phi_basis[column];
                if (domain_index >= 0 &&
                    domain_index < system.get_space().get_nbr_domains()) {
                    phi_parity = system.get_space()
                                     .get_domain(domain_index)
                                     ->phi_coefficient_parity(phi, basis);
                } else {
                    phi_parity = 0;
                }
                if (phi_parity == 0) {
                    ++grading.unsupported_phi_basis_columns;
                    phi_parity = +1;
                }
            }

            // Surface shapes are even.  Numeric globals need a name-keyed grading:
            // y-axis translation and time-odd velocity/infall parameters are
            // antisymmetric under the reflection-composed-with-time-reversal
            // involution, while ordinary scalar amplitudes remain even.
            int field = info.is_var_domain
                            ? +1
                            : (info.var_double_idx >= 0
                                   ? jacobian_parity_global_grading(info.var_name)
                                   : jacobian_parity_field_grading(
                                         info.var_name,
                                         grading.component_index[column]));
            if (field == 0) {
                ++grading.ungraded_columns;
                const std::string name = trim_ascii_space(info.var_name);
                if (std::find(grading.ungraded_names.begin(),
                              grading.ungraded_names.end(),
                              name) == grading.ungraded_names.end()) {
                    grading.ungraded_names.push_back(name);
                }
                field = +1;
            }
            grading.sector[column] = static_cast<signed char>(field * phi_parity);
        }
        if (grading.unsupported_phi_basis_columns > 0)
            grading.sector.clear();
        return grading;
    }

    std::string jacobian_parity_column_grading_disable_reason(
        const JacobianParityColumnGrading& grading)
    {
        if (grading.unsupported_variable_domain_layout)
            return "space does not describe its variable-domain column blocks";
        if (grading.unsupported_tau_seed_domain >= 0) {
            return "domain " +
                   std::to_string(grading.unsupported_tau_seed_domain) +
                   " does not describe its tau seed block";
        }
        if (grading.mixed_phi_columns > 0) {
            return std::to_string(grading.mixed_phi_columns) +
                   (grading.mixed_phi_columns == 1
                        ? " column has mixed phi indices"
                        : " columns have mixed phi indices");
        }
        if (grading.unsupported_phi_basis_columns > 0) {
            return std::to_string(grading.unsupported_phi_basis_columns) +
                   (grading.unsupported_phi_basis_columns == 1
                        ? " column has an unsupported or ambiguous phi basis"
                        : " columns have unsupported or ambiguous phi bases");
        }
        if (grading.ungraded_columns > 0) {
            std::string reason = std::to_string(grading.ungraded_columns) +
                                 " columns have no y parity:";
            for (const std::string& name : grading.ungraded_names)
                reason += " " + name;
            return reason;
        }
        return {};
    }

    JacobianParityRowPrediction
    predict_jacobian_parity_rows(System_of_eqs& system)
    {
        JacobianParityRowPrediction prediction;
        std::vector<ResidualRowDescriptor> descriptors;
        const bool structurally_complete =
            system.describe_residual_rows(descriptors);
        prediction.sector.assign(descriptors.size(), 0);

        for (std::size_t row = 0; row < descriptors.size(); ++row) {
            const ResidualRowDescriptor& descriptor = descriptors[row];
            if (!descriptor.available) {
                ++prediction.unavailable_rows;
                continue;
            }
            if (descriptor.family == ResidualRowEquationFamily::Integral) {
                if ((descriptor.explicit_sector != -1 &&
                     descriptor.explicit_sector != 1) ||
                    descriptor.equation_index != static_cast<int>(row) ||
                    !descriptor.sides.empty()) {
                    ++prediction.unavailable_rows;
                    continue;
                }
                prediction.sector[row] = descriptor.explicit_sector;
                continue;
            }
            if (descriptor.family != ResidualRowEquationFamily::Field ||
                descriptor.explicit_sector != 0 ||
                descriptor.equation_index < 0 || descriptor.sides.empty()) {
                ++prediction.unavailable_rows;
                continue;
            }

            const std::string& owner = system.equation_owner_var_name(
                descriptor.equation_index);
            signed char row_sector = 0;
            bool gradable = !owner.empty();
            for (const ResidualRowCoordinate& coordinate : descriptor.sides) {
                if (coordinate.domain < 0 ||
                    coordinate.domain >= system.get_space().get_nbr_domains() ||
                    coordinate.component < 0 || coordinate.phi_index < 0) {
                    gradable = false;
                    break;
                }
                const Domain* coordinate_domain =
                    system.get_space().get_domain(coordinate.domain);
                if (coordinate.phi_index >=
                    coordinate_domain->get_nbr_coefs()(2)) {
                    gradable = false;
                    break;
                }
                int field_sector = jacobian_parity_field_grading(
                    owner, coordinate.component);
                if (field_sector == 0)
                    field_sector = jacobian_parity_global_grading(owner);
                if (field_sector == 0) {
                    gradable = false;
                    break;
                }
                const int phi_sector =
                    coordinate_domain->phi_coefficient_parity(
                        coordinate.phi_index, coordinate.phi_basis);
                if (phi_sector == 0) {
                    ++prediction.unsupported_phi_basis_rows;
                    gradable = false;
                    break;
                }
                const signed char side_sector = static_cast<signed char>(
                    field_sector * phi_sector);
                if (row_sector == 0)
                    row_sector = side_sector;
                else if (row_sector != side_sector) {
                    gradable = false;
                    break;
                }
            }
            if (!gradable || row_sector == 0) {
                ++prediction.ungraded_rows;
                continue;
            }
            prediction.sector[row] = row_sector;
        }

        prediction.all_rows_available =
            structurally_complete && prediction.unavailable_rows == 0 &&
            prediction.ungraded_rows == 0 &&
            prediction.unsupported_phi_basis_rows == 0;
        return prediction;
    }

    JacobianParityRowGradingSelection select_jacobian_parity_row_grading(
        const JacobianParityRowPrediction& structural_prediction,
        const std::vector<signed char>& column_sector,
        const std::vector<double>& row_mass_symmetric,
        const std::vector<double>& row_mass_antisymmetric)
    {
        JacobianParityRowGradingSelection selection;
        const std::size_t row_count = row_mass_symmetric.size();
        if (row_mass_antisymmetric.size() != row_count) {
            selection.fallback_reason =
                "matrix row-mass tables have different sizes";
            return selection;
        }

        if (column_sector.size() != row_count) {
            selection.fallback_reason =
                "Jacobian row and column counts do not match";
        } else if (std::any_of(
                       column_sector.begin(), column_sector.end(),
                       [](signed char sector) {
                           return sector != 1 && sector != -1;
                       })) {
            selection.fallback_reason =
                "Jacobian columns are not all gradable";
        } else if (structural_prediction.sector.size() != row_count) {
            selection.fallback_reason =
                "structural descriptor row count does not match the Jacobian";
        } else if (!structural_prediction.all_rows_available) {
            selection.fallback_reason =
                "structural residual-row descriptors are incomplete"
                " (unavailable=" +
                std::to_string(structural_prediction.unavailable_rows) +
                ", ungraded=" +
                std::to_string(structural_prediction.ungraded_rows) +
                ", unsupported_phi_basis=" +
                std::to_string(
                    structural_prediction.unsupported_phi_basis_rows) +
                ")";
        } else if (std::any_of(
                       structural_prediction.sector.begin(),
                       structural_prediction.sector.end(),
                       [](signed char sector) {
                           return sector != 1 && sector != -1;
                       })) {
            selection.fallback_reason =
                "structural row grading contains a label outside "
                "plus-or-minus one";
        } else {
            selection.source =
                JacobianParityRowGradingSelection::Source::Structural;
            selection.sector = structural_prediction.sector;
            return selection;
        }

        JacobianParityMaskState matrix_grading;
        derive_jacobian_parity_row_sectors(
            matrix_grading, row_mass_symmetric, row_mass_antisymmetric);
        selection.sector = std::move(matrix_grading.row_sector);
        return selection;
    }

    bool regrade_jacobian_parity_rows_after_structural_measurement(
        JacobianParityRowGradingSelection& selection,
        double structural_maximum_cross, double maximum_entry,
        const std::vector<double>& row_mass_symmetric,
        const std::vector<double>& row_mass_antisymmetric)
    {
        if (selection.source !=
            JacobianParityRowGradingSelection::Source::Structural)
            return false;
        const double ratio = maximum_entry > 0.0
                                 ? structural_maximum_cross / maximum_entry
                                 : 0.0;
        if (ratio < jacobian_parity_cross_tolerance)
            return false;

        JacobianParityMaskState matrix_grading;
        derive_jacobian_parity_row_sectors(
            matrix_grading, row_mass_symmetric, row_mass_antisymmetric);
        selection.source =
            JacobianParityRowGradingSelection::Source::MatrixDerivedSecondPass;
        selection.sector = std::move(matrix_grading.row_sector);
        return true;
    }

    JacobianParityRowOracleComparison compare_jacobian_parity_row_prediction(
        const JacobianParityRowPrediction& prediction,
        const std::vector<signed char>& matrix_row_sector)
    {
        JacobianParityRowOracleComparison comparison;
        if (prediction.sector.size() != matrix_row_sector.size()) {
            comparison.failure_reason =
                "descriptor and matrix row-sector sizes differ";
            return comparison;
        }

        for (std::size_t row = 0; row < prediction.sector.size(); ++row) {
            const signed char predicted = prediction.sector[row];
            if (predicted == 0) {
                ++comparison.unavailable_rows;
                continue;
            }
            if ((predicted != 1 && predicted != -1) ||
                (matrix_row_sector[row] != 1 &&
                 matrix_row_sector[row] != -1)) {
                comparison.failure_reason =
                    "row-sector tables contain a label outside plus-or-minus one";
                return comparison;
            }
            ++comparison.compared_rows;
            if (predicted != matrix_row_sector[row]) {
                if (comparison.first_mismatch < 0)
                    comparison.first_mismatch = static_cast<int>(row);
                ++comparison.mismatched_rows;
            }
        }
        comparison.exact_on_covered_rows =
            comparison.compared_rows > 0 && comparison.mismatched_rows == 0;
        comparison.whole_fixture_covered =
            prediction.all_rows_available && comparison.unavailable_rows == 0;
        return comparison;
    }

    JacobianSelectionPlan::JacobianSelectionPlan(
        BlockLabel selected_block, std::vector<int> selected_rows,
        std::vector<int> selected_columns)
        : selected_block_(selected_block),
          selected_rows_(std::move(selected_rows)),
          selected_columns_(std::move(selected_columns))
    {
    }

    JacobianSelectionPlanBuild make_jacobian_selection_plan(
        const std::vector<JacobianSelectionPlan::BlockLabel>& row_block_labels,
        const std::vector<JacobianSelectionPlan::BlockLabel>& column_block_labels,
        JacobianSelectionPlan::BlockLabel selected_block,
        JacobianSelectionPlan::BlockLabel excluded_block)
    {
        JacobianSelectionPlanBuild result;
        if (selected_block == excluded_block) {
            result.fallback_reason = "selected and excluded block labels are identical";
            return result;
        }
        if (row_block_labels.size() != column_block_labels.size()) {
            result.fallback_reason =
                "row and column block-label tables have different sizes";
            return result;
        }

        std::vector<int> selected_rows;
        selected_rows.reserve(row_block_labels.size());
        for (std::size_t row = 0; row < row_block_labels.size(); ++row) {
            const auto label = row_block_labels[row];
            if (label != selected_block && label != excluded_block) {
                result.fallback_reason =
                    "row block label at index " + std::to_string(row) +
                    " is outside the declared two-block partition";
                return result;
            }
            if (label == selected_block)
                selected_rows.push_back(static_cast<int>(row));
        }

        std::vector<int> selected_columns;
        selected_columns.reserve(column_block_labels.size());
        for (std::size_t column = 0; column < column_block_labels.size();
             ++column) {
            const auto label = column_block_labels[column];
            if (label != selected_block && label != excluded_block) {
                result.fallback_reason =
                    "column block label at index " + std::to_string(column) +
                    " is outside the declared two-block partition";
                return result;
            }
            if (label == selected_block)
                selected_columns.push_back(static_cast<int>(column));
        }

        if (selected_rows.empty()) {
            result.fallback_reason = "selected block is empty";
            return result;
        }
        if (selected_rows.size() != selected_columns.size()) {
            result.fallback_reason =
                "selected block is not square (rows " +
                std::to_string(selected_rows.size()) + " vs columns " +
                std::to_string(selected_columns.size()) + ")";
            return result;
        }

        result.plan = std::make_shared<JacobianSelectionPlan>(
            selected_block, std::move(selected_rows),
            std::move(selected_columns));
        return result;
    }

    namespace
    {
        std::string validate_selected_indices(
            std::size_t full_size, const std::vector<int>& selected_indices)
        {
            if (selected_indices.empty())
                return "selected index set is empty";
            for (std::size_t position = 0; position < selected_indices.size();
                 ++position) {
                const int index = selected_indices[position];
                if (index < 0 ||
                    static_cast<std::size_t>(index) >= full_size)
                    return "selected index is outside the full vector";
                if (position > 0 &&
                    index <= selected_indices[position - 1]) {
                    return "selected indices are not strictly increasing";
                }
            }
            return {};
        }
    } // namespace

    JacobianSelectedValues gather_jacobian_selected_values(
        std::span<const double> full_values,
        const std::vector<int>& selected_indices)
    {
        JacobianSelectedValues result;
        result.failure_reason =
            validate_selected_indices(full_values.size(), selected_indices);
        if (!result.failure_reason.empty())
            return result;
        result.values.reserve(selected_indices.size());
        for (const int index : selected_indices) {
            const double value = full_values[index];
            if (!std::isfinite(value)) {
                result.values.clear();
                result.failure_reason = "selected vector value is non-finite";
                return result;
            }
            result.values.push_back(value);
        }
        return result;
    }

    JacobianSelectedValues scatter_jacobian_selected_values(
        std::span<const double> selected_values, int full_size,
        const std::vector<int>& selected_indices)
    {
        JacobianSelectedValues result;
        if (full_size < 0) {
            result.failure_reason = "full vector size is negative";
            return result;
        }
        result.failure_reason =
            validate_selected_indices(static_cast<std::size_t>(full_size),
                                      selected_indices);
        if (!result.failure_reason.empty())
            return result;
        if (selected_values.size() != selected_indices.size()) {
            result.failure_reason =
                "selected value count does not match selected index count";
            return result;
        }
        result.values.assign(static_cast<std::size_t>(full_size), 0.0);
        for (std::size_t position = 0; position < selected_values.size();
             ++position) {
            const double value = selected_values[position];
            if (!std::isfinite(value)) {
                result.values.clear();
                result.failure_reason = "selected vector value is non-finite";
                return result;
            }
            result.values[static_cast<std::size_t>(
                selected_indices[position])] = value;
        }
        return result;
    }

    JacobianSelectionNorms measure_jacobian_selection_norms(
        std::span<const double> full_values,
        const std::vector<int>& selected_indices)
    {
        JacobianSelectionNorms result;
        result.failure_reason =
            validate_selected_indices(full_values.size(), selected_indices);
        if (!result.failure_reason.empty())
            return result;

        std::size_t next_selected = 0;
        for (std::size_t index = 0; index < full_values.size(); ++index) {
            const double magnitude = std::abs(full_values[index]);
            const bool selected =
                next_selected < selected_indices.size() &&
                static_cast<std::size_t>(selected_indices[next_selected]) ==
                    index;
            if (!std::isfinite(magnitude)) {
                if (selected)
                    result.active_linf = magnitude;
                else
                    result.forbidden_linf = magnitude;
                result.failure_reason = "full vector value is non-finite";
                return result;
            }
            if (selected) {
                result.active_linf = std::max(result.active_linf, magnitude);
                ++next_selected;
            } else {
                result.forbidden_linf =
                    std::max(result.forbidden_linf, magnitude);
            }
        }
        return result;
    }

    JacobianPreJ1SelectionPlanBuild make_jacobian_pre_j1_selection_plan(
        const JacobianParityRowPrediction& row_prediction,
        const JacobianParityColumnGrading& column_grading,
        const double* full_entry_residual, int full_size)
    {
        JacobianPreJ1SelectionPlanBuild result;
        if (!row_prediction.all_rows_available) {
            result.fallback_reason =
                "structural residual-row descriptors are unavailable";
            return result;
        }

        const std::string column_reason =
            jacobian_parity_column_grading_disable_reason(column_grading);
        if (!column_reason.empty()) {
            result.fallback_reason =
                "Jacobian columns are not all gradable: " + column_reason;
            return result;
        }
        if (full_size < 0 ||
            row_prediction.sector.size() !=
                static_cast<std::size_t>(full_size) ||
            column_grading.sector.size() !=
                static_cast<std::size_t>(full_size)) {
            result.fallback_reason =
                "structural row, column, and residual counts do not match";
            return result;
        }

        std::vector<JacobianSelectionPlan::BlockLabel> row_labels(
            row_prediction.sector.begin(), row_prediction.sector.end());
        std::vector<JacobianSelectionPlan::BlockLabel> column_labels(
            column_grading.sector.begin(), column_grading.sector.end());
        JacobianSelectionPlanBuild built = make_jacobian_selection_plan(
            row_labels, column_labels, +1, -1);
        if (!built.plan) {
            result.fallback_reason = built.fallback_reason;
            return result;
        }

        if (full_entry_residual == nullptr) {
            result.entry_norms.failure_reason =
                "full vector storage is null";
        } else {
            result.entry_norms = measure_jacobian_selection_norms(
                std::span<const double>{
                    full_entry_residual,
                    static_cast<std::size_t>(full_size)},
                built.plan->selected_rows());
        }
        if (!result.entry_norms) {
            result.fallback_reason =
                "entry-residual guard cannot be evaluated: " +
                result.entry_norms.failure_reason;
            return result;
        }
        result.entry_limit = std::max(
            jacobian_pre_j1_forbidden_relative_tolerance *
                result.entry_norms.active_linf,
            jacobian_pre_j1_forbidden_absolute_floor);
        if (result.entry_norms.forbidden_linf > result.entry_limit) {
            result.fallback_reason =
                "entry-residual guard refused: forbidden Linf exceeds "
                "max(1e-10 * active Linf, 1e-12)";
            return result;
        }

        result.plan = std::move(built.plan);
        return result;
    }

    JacobianForbiddenResidualCheck check_jacobian_forbidden_residual(
        const JacobianSelectionNorms& norms, double& forbidden_baseline,
        bool& forbidden_baseline_installed) noexcept
    {
        JacobianForbiddenResidualCheck result;
        if (!norms.failure_reason.empty() ||
            !std::isfinite(norms.active_linf) ||
            !std::isfinite(norms.forbidden_linf) || norms.active_linf < 0.0 ||
            norms.forbidden_linf < 0.0) {
            result.limit = std::numeric_limits<double>::quiet_NaN();
            return result;
        }
        if (!forbidden_baseline_installed) {
            forbidden_baseline = norms.forbidden_linf;
            forbidden_baseline_installed = true;
            result.allowed = true;
        } else if (!std::isfinite(forbidden_baseline) ||
                   forbidden_baseline < 0.0) {
            result.limit = std::numeric_limits<double>::quiet_NaN();
            return result;
        }
        const double maximum = std::numeric_limits<double>::max();
        result.limit =
            forbidden_baseline >
                    maximum / jacobian_forbidden_residual_baseline_multiplier
                ? maximum
                : jacobian_forbidden_residual_baseline_multiplier *
                      forbidden_baseline;
        if (!result.allowed)
            result.allowed = norms.forbidden_linf <= result.limit;
        return result;
    }

    double install_or_measure_jacobian_inactive_state_drift(
        const std::vector<double>& current, std::vector<double>& baseline,
        bool& baseline_installed)
    {
        if (!baseline_installed) {
            baseline = current;
            baseline_installed = true;
            return 0.0;
        }
        if (current.size() != baseline.size())
            return std::numeric_limits<double>::infinity();
        double linf = 0.0;
        for (std::size_t index = 0; index < current.size(); ++index) {
            if (!std::isfinite(current[index]) ||
                !std::isfinite(baseline[index])) {
                return std::numeric_limits<double>::infinity();
            }
            linf = std::max(
                linf, std::abs(current[index] - baseline[index]));
        }
        return linf;
    }

    bool jacobian_inactive_state_drift_allowed(double drift_linf) noexcept
    {
        return std::isfinite(drift_linf) && drift_linf >= 0.0 &&
            drift_linf <= jacobian_inactive_state_drift_tolerance;
    }

    bool jacobian_selection_factor_compatible(
        const std::shared_ptr<const JacobianSelectionPlan>& retained_plan,
        int retained_dimension,
        const std::shared_ptr<const JacobianSelectionPlan>& current_plan,
        int current_dimension) noexcept
    {
        return retained_dimension > 0 &&
            retained_dimension == current_dimension &&
            retained_plan.get() == current_plan.get();
    }

    void abandon_jacobian_parity_reduction(
        JacobianParityMaskState& state, const std::string& reason, int rank)
    {
        if (state.reduction_decision ==
            JacobianParityMaskState::ReductionDecision::MaskedFullFallback) {
            return;
        }
        state.reduction_decision =
            JacobianParityMaskState::ReductionDecision::MaskedFullFallback;
        state.selection_plan.reset();
        state.inactive_state_baseline.clear();
        state.inactive_state_baseline_installed = false;
        state.forbidden_baseline = 0.0;
        state.forbidden_baseline_installed = false;
        state.reduction_fallback_reason = reason;
        if (rank == 0) {
            std::cout << "Jacobian sector reduction: masked-full fallback, "
                      << reason << std::endl;
        }
    }

    void disable_jacobian_parity_mask(JacobianParityMaskState& state,
                                      const std::string& reason, int rank)
    {
        (void)reason;
        (void)rank;
        state.decision = JacobianParityMaskState::Decision::Disabled;
        state.column_sector.clear();
        state.row_sector.clear();
        state.structural_labels_available = false;
        state.unmasked_full_j_emitted = false;
    }

    void derive_jacobian_parity_row_sectors(
        JacobianParityMaskState& state,
        const std::vector<double>& row_mass_symmetric,
        const std::vector<double>& row_mass_antisymmetric)
    {
        const std::size_t n = row_mass_symmetric.size();
        state.row_sector.assign(n, 0);
        for (std::size_t row = 0; row < n; ++row) {
            state.row_sector[row] =
                (row_mass_symmetric[row] >= row_mass_antisymmetric[row])
                    ? static_cast<signed char>(1)
                    : static_cast<signed char>(-1);
        }
    }

    void decide_jacobian_parity_mask(JacobianParityMaskState& state,
                                     double maximum_cross, double maximum_entry,
                                     int rank)
    {
        long long symmetric_columns = 0;
        for (signed char sector : state.column_sector)
            if (sector > 0)
                ++symmetric_columns;
        long long symmetric_rows = 0;
        for (signed char sector : state.row_sector)
            if (sector > 0)
                ++symmetric_rows;

        // Each sector must be square: a rectangular block is singular whatever
        // the entries say.
        if (symmetric_rows != symmetric_columns) {
            disable_jacobian_parity_mask(
                state,
                "sector blocks are not square (rows S " +
                    std::to_string(symmetric_rows) + " vs columns S " +
                    std::to_string(symmetric_columns) + ")",
                rank);
            return;
        }

        const double ratio =
            maximum_entry > 0.0 ? maximum_cross / maximum_entry : 0.0;
        const bool approximate_engagement =
            ratio >= jacobian_parity_cross_tolerance &&
            ratio <= jacobian_parity_approximate_cross_tolerance;
        if (!(ratio < jacobian_parity_cross_tolerance) &&
            !approximate_engagement) {
            disable_jacobian_parity_mask(
                state, "sectors are coupled (max cross / max |J| = " +
                           std::to_string(ratio) + ")",
                rank);
            return;
        }

        state.decision = JacobianParityMaskState::Decision::Engaged;
        // The engage summary is printed by the assembler as one combined line
        // once the drop statistics exist; only record the measurement here.
        state.engaged_cross_ratio = ratio;
        state.approximate_engagement = approximate_engagement;
    }

    void decide_jacobian_parity_reduction(JacobianParityMaskState& state,
                                          double maximum_cross,
                                          double maximum_entry, int rank)
    {
        if (state.reduction_decision !=
            JacobianParityMaskState::ReductionDecision::Undecided) {
            return;
        }
        if (state.decision != JacobianParityMaskState::Decision::Engaged) {
            abandon_jacobian_parity_reduction(
                state, "first-J parity mask did not engage", rank);
            return;
        }
        if (!std::isfinite(maximum_cross) || !std::isfinite(maximum_entry) ||
            maximum_cross < 0.0 || maximum_entry <= 0.0) {
            abandon_jacobian_parity_reduction(
                state, "first-J parity measurement is invalid", rank);
            return;
        }

        const double ratio =
            maximum_entry > 0.0 ? maximum_cross / maximum_entry : 0.0;
        if (!(ratio < jacobian_parity_cross_tolerance)) {
            abandon_jacobian_parity_reduction(
                state,
                "first-J parity is approximate (max cross / max |J| >= 1e-12)",
                rank);
            return;
        }
        if (state.n <= 0 || state.row_sector.size() != static_cast<std::size_t>(state.n) ||
            state.column_sector.size() != static_cast<std::size_t>(state.n)) {
            abandon_jacobian_parity_reduction(
                state, "first-J parity block-label tables are malformed", rank);
            return;
        }

        std::vector<JacobianSelectionPlan::BlockLabel> row_labels(
            state.row_sector.begin(), state.row_sector.end());
        std::vector<JacobianSelectionPlan::BlockLabel> column_labels(
            state.column_sector.begin(), state.column_sector.end());
        JacobianSelectionPlanBuild built = make_jacobian_selection_plan(
            row_labels, column_labels, +1, -1);
        if (!built.plan) {
            abandon_jacobian_parity_reduction(
                state, built.fallback_reason, rank);
            return;
        }

        state.selection_plan = std::move(built.plan);
        state.reduction_fallback_reason.clear();
        state.reduction_decision =
            JacobianParityMaskState::ReductionDecision::Eligible;
    }
} // namespace Kadath
