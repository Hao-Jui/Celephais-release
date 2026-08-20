#include "coloring_internals.hpp"

#include "For_Kadath/Domain/adapted.hpp"
#include "For_Kadath/Domain/spheric_adapted_nosym.hpp"
#include "For_Kadath/Domain/spheric_nosym.hpp"

#include <chrono>
#include <cstdint>

namespace Kadath
{
    using namespace coloring_internals;

    namespace
    {
        struct JacobianStructuralCounts
        {
            int dom_min = -1;
            int dom_max = -1;
            int ndom = 0;
            int nbr_unknowns = 0;
            int nbr_conditions = 0;
            int nvar = 0;
            int nvar_double = 0;
            int nterm = 0;
            int nterm_double = 0;
            int neq = 0;
            int neq_int = 0;
            int variable_domain_unknowns = 0;
            int space_ndim = 0;
            int space_domains = 0;

            std::size_t var_size = 0;
            std::size_t var_double_size = 0;
            std::size_t names_var_size = 0;
            std::size_t names_var_double_size = 0;
            std::size_t term_size = 0;
            std::size_t term_double_size = 0;
            std::size_t assoc_var_size = 0;
            std::size_t assoc_var_double_size = 0;
            std::size_t eq_size = 0;
            std::size_t eq_int_size = 0;
            std::size_t results_size = 0;
            std::size_t eq_list_size = 0;
            std::size_t eq_int_list_size = 0;
            std::size_t eq_column_attachments_size = 0;

            bool operator==(const JacobianStructuralCounts&) const = default;
        };

        struct DomainTopologySnapshot
        {
            const Domain* identity = nullptr;
            int number = -1;
            int ndim = 0;
            int basis_type = 0;
            std::vector<int> points;
            std::vector<int> coefficients;

            bool operator==(const DomainTopologySnapshot&) const = default;
        };

        struct TermTopologySnapshot
        {
            const Term_eq* identity = nullptr;
            int domain = -1;
            std::vector<std::int64_t> tensor_layout_and_bases;

            bool operator==(const TermTopologySnapshot&) const = default;
        };

        struct EquationTopologySnapshot
        {
            const Equation* identity = nullptr;
            int row_count = 0;
            std::vector<unsigned char> domain_incidence;

            bool operator==(const EquationTopologySnapshot&) const = default;
        };

        struct AttachmentTopologySnapshot
        {
            ColumnClass column_class = ColumnClass::Unknown;
            int domain = -1;
            int boundary = -1;
            std::string owner_var_name;

            bool operator==(const AttachmentTopologySnapshot&) const = default;
        };

        struct JacobianAssemblerStructuralPlanKey
        {
            const Space* space_identity = nullptr;
            JacobianStructuralCounts counts;
            std::vector<DomainTopologySnapshot> domains;
            std::vector<const double*> var_double_identities;
            std::vector<const Tensor*> var_identities;
            std::vector<const Term_eq*> term_double_identities;
            std::vector<TermTopologySnapshot> terms;
            std::vector<const Eq_int*> integral_equation_identities;
            std::vector<EquationTopologySnapshot> equations;
            std::vector<int> assoc_var;
            std::vector<int> assoc_var_double;
            std::vector<std::string> names_var;
            std::vector<std::string> names_var_double;
            std::vector<std::tuple<std::string, int, int>> eq_list;
            std::vector<std::tuple<std::string, int, int>> eq_int_list;
            std::vector<AttachmentTopologySnapshot> attachments;

            bool operator==(const JacobianAssemblerStructuralPlanKey&) const = default;
        };

        void append_dimensions(std::vector<int>& out, const Dim_array& dimensions)
        {
            out.push_back(dimensions.get_ndim());
            for (int axis = 0; axis < dimensions.get_ndim(); ++axis)
                out.push_back(dimensions(axis));
        }

        void append_int_array(std::vector<std::int64_t>& out,
                              const Array<int>& values)
        {
            out.push_back(values.get_ndim());
            for (int axis = 0; axis < values.get_ndim(); ++axis)
                out.push_back(values.get_size(axis));
            out.push_back(static_cast<std::int64_t>(values.get_nbr()));
            for (std::size_t i = 0; i < values.get_nbr(); ++i)
                out.push_back(values.get_data()[i]);
        }

        void append_spectral_base(std::vector<std::int64_t>& out,
                                  const Base_spectral& base, int ndim)
        {
            out.push_back(base.is_def() ? 1 : 0);
            out.push_back(ndim);
            if (!base.is_def())
                return;

            for (int axis = 0; axis < ndim; ++axis) {
                const Array<int>* one_dimensional_base = base.get_base_1d(axis);
                if (one_dimensional_base == nullptr) {
                    out.push_back(-1);
                    continue;
                }
                out.push_back(1);
                append_int_array(out, *one_dimensional_base);
            }
        }

        double elapsed_seconds(std::chrono::steady_clock::time_point start)
        {
            return std::chrono::duration<double>(
                       std::chrono::steady_clock::now() - start)
                .count();
        }

        std::string trim_ascii_space(const std::string& text)
        {
            const std::size_t first = text.find_first_not_of(" \t\n\r");
            if (first == std::string::npos)
                return "";
            const std::size_t last = text.find_last_not_of(" \t\n\r");
            return text.substr(first, last - first + 1);
        }

        bool eq_full_is_bare_owner_zero(const std::string& expression,
                                        const std::string& owner_var_name)
        {
            const std::string owner = trim_ascii_space(owner_var_name);
            if (owner.empty())
                return false;

            const std::size_t equals_pos = expression.find('=');
            if (equals_pos == std::string::npos)
                return false;

            const std::string lhs = trim_ascii_space(expression.substr(0, equals_pos));
            const std::string rhs = trim_ascii_space(expression.substr(equals_pos + 1));
            return rhs == "0" &&
                   (lhs == owner || lhs == owner + "^i");
        }

        void assign_column_probe_base(const Tensor& reference, Tensor& probe, int dom)
        {
            const int component_count = probe.get_n_comp();
            for (int comp = 0; comp < component_count; ++comp) {
                const Array<int> idx(probe.indices(comp));
                if (!probe(idx)(dom).check_if_zero() &&
                    !probe(idx)(dom).get_base().is_def()) {
                    probe.set(idx).set_domain(dom).set_base() =
                        reference(idx)(dom).get_base();
                }
            }
        }

        int semantic_domain_type_id(const Domain& domain)
        {
            if (dynamic_cast<const Domain_nucleus_nosym*>(&domain) != nullptr)
                return static_cast<int>(ColumnDomainType::SphericNucleusNoSym);
            if (dynamic_cast<const Domain_shell_nosym*>(&domain) != nullptr)
                return static_cast<int>(ColumnDomainType::SphericShellNoSym);
            if (dynamic_cast<const Domain_compact_nosym*>(&domain) != nullptr)
                return static_cast<int>(ColumnDomainType::SphericCompactNoSym);
            if (dynamic_cast<const Domain_shell_outer_adapted_nosym*>(&domain) != nullptr)
                return static_cast<int>(ColumnDomainType::SphericShellOuterAdaptedNoSym);
            if (dynamic_cast<const Domain_shell_inner_adapted_nosym*>(&domain) != nullptr)
                return static_cast<int>(ColumnDomainType::SphericShellInnerAdaptedNoSym);
            if (dynamic_cast<const Domain_bispheric_chi_first_nosym*>(&domain) != nullptr)
                return static_cast<int>(ColumnDomainType::BisphericChiFirstNoSym);
            if (dynamic_cast<const Domain_bispheric_rect_nosym*>(&domain) != nullptr)
                return static_cast<int>(ColumnDomainType::BisphericRectNoSym);
            if (dynamic_cast<const Domain_bispheric_eta_first_nosym*>(&domain) != nullptr)
                return static_cast<int>(ColumnDomainType::BisphericEtaFirstNoSym);
            if (dynamic_cast<const Domain_nucleus*>(&domain) != nullptr)
                return static_cast<int>(ColumnDomainType::SphericNucleus);
            if (dynamic_cast<const Domain_shell*>(&domain) != nullptr)
                return static_cast<int>(ColumnDomainType::SphericShell);
            if (dynamic_cast<const Domain_compact*>(&domain) != nullptr)
                return static_cast<int>(ColumnDomainType::SphericCompact);
            if (dynamic_cast<const Domain_shell_outer_adapted*>(&domain) != nullptr)
                return static_cast<int>(ColumnDomainType::SphericShellOuterAdapted);
            if (dynamic_cast<const Domain_shell_inner_adapted*>(&domain) != nullptr)
                return static_cast<int>(ColumnDomainType::SphericShellInnerAdapted);
            if (dynamic_cast<const Domain_bispheric_chi_first*>(&domain) != nullptr)
                return static_cast<int>(ColumnDomainType::BisphericChiFirst);
            if (dynamic_cast<const Domain_bispheric_rect*>(&domain) != nullptr)
                return static_cast<int>(ColumnDomainType::BisphericRect);
            if (dynamic_cast<const Domain_bispheric_eta_first*>(&domain) != nullptr)
                return static_cast<int>(ColumnDomainType::BisphericEtaFirst);
            return static_cast<int>(ColumnDomainType::Unknown);
        }

        void assign_column_semantics(ColumnInfo& info, const Domain& domain,
                                     const TauSeedDescriptor& descriptor)
        {
            const Dim_array& dimensions = domain.get_nbr_coefs();
            if (dimensions.get_ndim() != 3 || dimensions(0) <= 0 ||
                dimensions(1) <= 0 || dimensions(2) <= 0 ||
                descriptor.component < 0 || descriptor.write_count <= 0) {
                return;
            }

            const std::size_t nr = static_cast<std::size_t>(dimensions(0));
            const std::size_t nt = static_cast<std::size_t>(dimensions(1));
            const std::size_t np = static_cast<std::size_t>(dimensions(2));
            const std::size_t offset = descriptor.writes[0].coefficient_offset;
            if (offset >= nr * nt * np)
                return;

            info.domain_type_id = semantic_domain_type_id(domain);
            info.tensor_component = descriptor.component;
            info.coefficient_i = static_cast<int>(offset / (nt * np));
            const std::size_t angular_offset = offset % (nt * np);
            info.coefficient_j = static_cast<int>(angular_offset / np);
            info.coefficient_k = static_cast<int>(angular_offset % np);
            info.coefficient_nr = dimensions(0);
            info.coefficient_nt = dimensions(1);
            info.coefficient_np = dimensions(2);
        }
    } // namespace

    struct JacobianAssemblerStructuralPlanCache
    {
        JacobianAssemblerStructuralPlanKey key;
        JacobianAssemblerStructuralPlan plan;
        bool valid = false;
        bool metadata_ready = false;
    };

    // PRODUCTION HOT PATH: Newton and JacobianColumnEngine depend on this column-row incidence.
    void System_of_eqs::build_column_map(std::vector<ColumnInfo>& column_map,
                                         bool classify_field_columns) const
    {
        column_map.clear();
        column_map.reserve(nbr_unknowns);

        // 1. Variable domains (from Space)
        int n_var_dom = espace.nbr_unknowns_from_variable_domains();
        for (int i = 0; i < n_var_dom; ++i) {
            ColumnInfo info;
            info.var_idx = -1;
            info.var_double_idx = -1;
            info.domain = -1;
            info.term_idx = -1;
            info.basis_mode = i;
            info.var_name = "__var_domain__";
            info.is_var_domain = true;
            column_map.push_back(info);
        }

        // 2. Scalar variables (var_double)
        for (int i = 0; i < nvar_double; ++i) {
            ColumnInfo info;
            info.var_idx = -1;
            info.var_double_idx = i;
            info.domain = -1;
            info.term_idx = -1;
            info.basis_mode = -1;
            info.var_name = !names_var_double[i].empty() ? names_var_double[i].c_str() : "__unnamed_double__";
            info.is_var_domain = false;
            column_map.push_back(info);
        }

        // 3. Field variables - iterate through terms
        for (int i = 0; i < nterm; ++i) {
            int dom = term[i]->get_dom();
            int var_idx = assoc_var[i];
            std::string vname =
                (var_idx >= 0 && !names_var[var_idx].empty()) ? names_var[var_idx].c_str() : "__unnamed_field__";

            const Domain* const domain = espace.get_domain(dom);
            const Tensor& term_value = term[i]->get_val_t();
            int n_coefs = domain->nbr_unknowns(term_value, dom);
            std::vector<TauSeedDescriptor> semantic_descriptors;
            const bool semantics_available =
                domain->describe_tau_seed_block(term_value, dom,
                                                semantic_descriptors);
            if (semantics_available &&
                semantic_descriptors.size() != static_cast<std::size_t>(n_coefs)) {
                KADATH_THROW("Column semantic descriptor count does not match "
                             "Domain::nbr_unknowns");
            }

            const int term_start_col = static_cast<int>(column_map.size());
            for (int c = 0; c < n_coefs; ++c) {
                ColumnInfo info;
                info.var_idx = var_idx;
                info.var_double_idx = -1;
                info.domain = dom;
                info.term_idx = i;
                info.basis_mode = c;
                if (semantics_available) {
                    assign_column_semantics(
                        info, *domain,
                        semantic_descriptors[static_cast<std::size_t>(c)]);
                }
                if (classify_field_columns) {
                    info.field_class = classify_field_column_from_equations(
                        i, term_start_col + c, term_start_col);
                }
                info.var_name = vname;
                info.is_var_domain = false;
                column_map.push_back(info);
            }
        }
    }

    void System_of_eqs::build_direct_singleton_jacobian_columns(
        DirectJacobianColumnPlan& direct_plan) const
    {
        std::vector<ColumnInfo> column_map;
        build_column_map(column_map, true);
        build_direct_singleton_jacobian_columns(direct_plan, column_map);
    }

    void System_of_eqs::build_direct_singleton_jacobian_columns(
        DirectJacobianColumnPlan& direct_plan,
        const std::vector<ColumnInfo>& column_map) const
    {
        if (nbr_conditions < 0) {
            KADATH_THROW(kColoringNeedsSecMember);
        }

        direct_plan.columns.assign(static_cast<std::size_t>(nbr_unknowns),
                                   DirectJacobianColumn{});
        direct_plan.entries.clear();
        direct_plan.entries.reserve(static_cast<std::size_t>(nbr_unknowns));

        std::vector<int> term_start_column(static_cast<std::size_t>(nterm), -1);
        for (int col = 0; col < static_cast<int>(column_map.size()); ++col) {
            const ColumnInfo& info = column_map[static_cast<std::size_t>(col)];
            if (info.term_idx >= 0 &&
                info.term_idx < nterm &&
                term_start_column[static_cast<std::size_t>(info.term_idx)] < 0) {
                term_start_column[static_cast<std::size_t>(info.term_idx)] = col;
            }
        }

        int row_offset = neq_int;
        const int attached_count =
            std::min<int>(neq, static_cast<int>(eq_column_attachments.size()));
        for (int eq_idx = 0; eq_idx < neq; ++eq_idx) {
            const Equation* equation = eq[static_cast<std::size_t>(eq_idx)].get();
            const int row_count = (equation != nullptr) ? equation->get_n_cond_tot() : 0;
            const auto* full_equation = dynamic_cast<const Eq_full*>(equation);
            if (full_equation == nullptr || eq_idx >= attached_count ||
                eq_idx >= static_cast<int>(eq_list.size())) {
                row_offset += row_count;
                continue;
            }

            const EquationColumnAttachment& attachment =
                eq_column_attachments[static_cast<std::size_t>(eq_idx)];
            const std::string owner_var_name =
                trim_ascii_space(attachment.owner_var_name);
            if (attachment.column_class != ColumnClass::FieldInteriorVol ||
                attachment.domain < 0 ||
                owner_var_name.empty()) {
                row_offset += row_count;
                continue;
            }

            const std::string& expression =
                std::get<0>(eq_list[static_cast<std::size_t>(eq_idx)]);
            if (!eq_full_is_bare_owner_zero(expression, owner_var_name)) {
                row_offset += row_count;
                continue;
            }

            for (int col = 0; col < static_cast<int>(column_map.size()); ++col) {
                const ColumnInfo& info = column_map[static_cast<std::size_t>(col)];
                const std::string column_var_name =
                    trim_ascii_space(info.var_name);
                if (info.is_var_domain ||
                    info.var_double_idx >= 0 ||
                    info.term_idx < 0 ||
                    info.term_idx >= nterm ||
                    info.domain != attachment.domain ||
                    column_var_name != owner_var_name ||
                    info.field_class != ColumnClass::FieldInteriorVol) {
                    continue;
                }

                const int term_start =
                    term_start_column[static_cast<std::size_t>(info.term_idx)];
                if (term_start < 0)
                    continue;

                Tensor probe(term[static_cast<std::size_t>(info.term_idx)]->get_val_t(),
                             false);
                probe.annule_hard();
                int column_counter = term_start;
                espace.get_domain(info.domain)->affecte_tau_one_coef(
                    probe, info.domain, col, column_counter);
                assign_column_probe_base(
                    term[static_cast<std::size_t>(info.term_idx)]->get_val_t(),
                    probe, info.domain);

                Term_eq residual_term(
                    info.domain,
                    term[static_cast<std::size_t>(info.term_idx)]->get_val_t(),
                    probe);
                Term_eq* residual_terms[] = {&residual_term};
                Array<double> exported(row_count);
                exported = 0.0;
                int residual_index = 0;
                int exported_row_start = 0;
                full_equation->export_der(residual_index, residual_terms,
                                          exported, exported_row_start);

                const std::size_t first_entry_index = direct_plan.entries.size();
                for (int local_row = 0; local_row < row_count; ++local_row) {
                    const double value = exported(local_row);
                    if (std::fabs(value) > 0.0) {
                        direct_plan.entries.push_back(
                            DirectJacobianEntry{row_offset + local_row, value});
                    }
                }
                const std::size_t entry_count =
                    direct_plan.entries.size() - first_entry_index;
                if (entry_count > 0) {
                    DirectJacobianColumn& direct_column =
                        direct_plan.columns[static_cast<std::size_t>(col)];
                    direct_column.first_entry_index = first_entry_index;
                    direct_column.entry_count = entry_count;
                }
            }

            row_offset += row_count;
        }
    }

    const JacobianAssemblerStructuralPlan&
    System_of_eqs::get_jacobian_assembler_structural_plan(
        bool include_column_metadata,
        bool cache_enabled,
        JacobianAssemblerStructuralPlanAccess& access) const
    {
        access = JacobianAssemblerStructuralPlanAccess{};
        if (jacobian_assembler_structural_plan_cache_ == nullptr) {
            jacobian_assembler_structural_plan_cache_ =
                std::make_shared<JacobianAssemblerStructuralPlanCache>();
        }
        JacobianAssemblerStructuralPlanCache& cache =
            *jacobian_assembler_structural_plan_cache_;

        JacobianAssemblerStructuralPlanKey current_key;
        if (cache_enabled) {
            const auto check_start = std::chrono::steady_clock::now();
            current_key.space_identity = &espace;
            current_key.counts = JacobianStructuralCounts{
                dom_min,
                dom_max,
                ndom,
                nbr_unknowns,
                nbr_conditions,
                nvar,
                nvar_double,
                nterm,
                nterm_double,
                neq,
                neq_int,
                espace.nbr_unknowns_from_variable_domains(),
                espace.get_ndim(),
                espace.get_nbr_domains(),
                var.size(),
                var_double.size(),
                names_var.size(),
                names_var_double.size(),
                term.size(),
                term_double.size(),
                assoc_var.size(),
                assoc_var_double.size(),
                eq.size(),
                eq_int.size(),
                results.size(),
                eq_list.size(),
                eq_int_list.size(),
                eq_column_attachments.size()};

            current_key.domains.reserve(
                static_cast<std::size_t>(espace.get_nbr_domains()));
            for (int domain_index = 0;
                 domain_index < espace.get_nbr_domains(); ++domain_index) {
                const Domain* domain = espace.get_domain(domain_index);
                DomainTopologySnapshot snapshot;
                snapshot.identity = domain;
                snapshot.number = domain->get_num();
                snapshot.ndim = domain->get_ndim();
                snapshot.basis_type = domain->get_type_base();
                append_dimensions(snapshot.points, domain->get_nbr_points());
                append_dimensions(snapshot.coefficients,
                                  domain->get_nbr_coefs());
                current_key.domains.push_back(std::move(snapshot));
            }

            current_key.var_double_identities.reserve(var_double.size());
            for (const double* item : var_double)
                current_key.var_double_identities.push_back(item);
            current_key.var_identities.reserve(var.size());
            for (const Tensor* item : var)
                current_key.var_identities.push_back(item);
            current_key.term_double_identities.reserve(term_double.size());
            for (const auto& item : term_double)
                current_key.term_double_identities.push_back(item.get());

            current_key.terms.reserve(term.size());
            for (const auto& item : term) {
                TermTopologySnapshot snapshot;
                snapshot.identity = item.get();
                snapshot.domain = item->get_dom();
                const Tensor& tensor = item->get_val_t();
                std::vector<std::int64_t>& layout =
                    snapshot.tensor_layout_and_bases;
                layout.push_back(tensor.get_valence());
                layout.push_back(tensor.get_ndim());
                layout.push_back(tensor.get_n_comp());
                layout.push_back(espace.get_nbr_domains());
                for (int index = 0; index < tensor.get_valence(); ++index)
                    layout.push_back(tensor.get_index_type(index));
                for (int domain_index = 0;
                     domain_index < espace.get_nbr_domains(); ++domain_index) {
                    layout.push_back(
                        tensor.get_basis().get_basis(domain_index));
                }
                for (int component = 0;
                     component < tensor.get_n_comp(); ++component) {
                    const Array<int> indices = tensor.indices(component);
                    append_int_array(layout, indices);
                    for (int domain_index = 0;
                         domain_index < espace.get_nbr_domains(); ++domain_index) {
                        const Base_spectral& base =
                            tensor(indices)(domain_index).get_base();
                        append_spectral_base(layout, base,
                                             espace.get_domain(domain_index)
                                                 ->get_ndim());
                    }
                }
                current_key.terms.push_back(std::move(snapshot));
            }

            current_key.integral_equation_identities.reserve(eq_int.size());
            for (const auto& item : eq_int)
                current_key.integral_equation_identities.push_back(item.get());
            current_key.equations.reserve(eq.size());
            for (const auto& item : eq) {
                EquationTopologySnapshot snapshot;
                snapshot.identity = item.get();
                if (item != nullptr) {
                    snapshot.row_count = item->get_n_cond_tot();
                    snapshot.domain_incidence.reserve(
                        static_cast<std::size_t>(espace.get_nbr_domains()));
                    for (int domain_index = 0;
                         domain_index < espace.get_nbr_domains(); ++domain_index) {
                        snapshot.domain_incidence.push_back(
                            item->take_into_account(domain_index) ? 1 : 0);
                    }
                }
                current_key.equations.push_back(std::move(snapshot));
            }

            current_key.assoc_var = assoc_var;
            current_key.assoc_var_double = assoc_var_double;
            current_key.names_var = names_var;
            current_key.names_var_double = names_var_double;
            current_key.eq_list = eq_list;
            current_key.eq_int_list = eq_int_list;
            current_key.attachments.reserve(eq_column_attachments.size());
            for (const EquationColumnAttachment& attachment :
                 eq_column_attachments) {
                current_key.attachments.push_back(AttachmentTopologySnapshot{
                    attachment.column_class,
                    attachment.domain,
                    attachment.boundary,
                    attachment.owner_var_name});
            }

            const bool hit =
                cache.valid && cache.key == current_key &&
                (!include_column_metadata || cache.metadata_ready);
            access.cache_check_seconds = elapsed_seconds(check_start);
            if (hit) {
                access.cache_hit = true;
                return cache.plan;
            }
        }

        const auto build_start = std::chrono::steady_clock::now();
        std::vector<ColumnInfo> column_map;
        build_column_map(column_map, /*classify_field_columns=*/true);
        JacobianAssemblerStructuralPlan rebuilt;
        build_direct_singleton_jacobian_columns(
            rebuilt.direct_singleton_plan, column_map);
        if (include_column_metadata)
            classify_columns(rebuilt.column_metadata, column_map);
        access.cache_miss_build_seconds = elapsed_seconds(build_start);

        cache.plan = std::move(rebuilt);
        cache.metadata_ready = include_column_metadata;
        cache.valid = cache_enabled;
        if (cache_enabled)
            cache.key = std::move(current_key);
        return cache.plan;
    }

    std::set<std::tuple<ColumnClass, int, std::string>>
    System_of_eqs::build_bare_owner_zero_attachment_set() const
    {
        std::set<std::tuple<ColumnClass, int, std::string>> result;
        const int attached = std::min<int>(neq, static_cast<int>(eq_column_attachments.size()));
        for (int eq_idx = 0; eq_idx < attached; ++eq_idx) {
            const EquationColumnAttachment& attachment =
                eq_column_attachments[static_cast<std::size_t>(eq_idx)];
            if (attachment.column_class != ColumnClass::FieldBoundaryTau &&
                attachment.column_class != ColumnClass::FieldMatching)
                continue;
            if (attachment.owner_var_name.empty() || attachment.domain < 0)
                continue;
            if (eq_idx >= static_cast<int>(eq_list.size()))
                continue;
            const std::string& expression = std::get<0>(eq_list[static_cast<std::size_t>(eq_idx)]);
            if (eq_full_is_bare_owner_zero(expression, attachment.owner_var_name)) {
                result.emplace(attachment.column_class,
                               attachment.domain,
                               trim_ascii_space(attachment.owner_var_name));
            }
        }
        return result;
    }

    // =============================================================================
    // Row incidence construction (Optimization 1 - cached in coloring)
    // =============================================================================

    /*
     * Build row incidence for each column
     * Uses the same take_into_account() logic as do_col_J
     *
     * NOTE: Integral equations are EXCLUDED from the coloring graph because they
     * connect all columns (global constraints). They must be handled separately
     * in the seeded Jacobian computation.
     */
    void System_of_eqs::build_column_row_incidence(const std::vector<ColumnInfo>& column_map,
                                                   std::vector<std::set<int>>& rows_per_column) const
    {
        if (nbr_conditions < 0) {
            KADATH_THROW(kColoringNeedsSecMember);
        }

        rows_per_column.clear();
        rows_per_column.resize(nbr_unknowns);

        for (int cc = 0; cc < nbr_unknowns; ++cc) {
            const auto& info = column_map[cc];
            std::set<int>& affected_rows = rows_per_column[cc];

            // Determine which domain(s) this column affects
            int zedom = -1;
            Array<int> zedoms(2);
            zedoms = -1;
            bool is_var_double = false;

            if (info.is_var_domain) {
                int temp_conte = 0;
                espace.affecte_coef_to_variable_domains(temp_conte, cc, zedoms);
            } else if (info.var_double_idx >= 0) {
                is_var_double = true;
            } else if (info.domain >= 0) {
                zedom = info.domain;
            }

            // Skip integral equations in row numbering - they're excluded from coloring
            // (they connect all columns, destroying sparsity)
            int pos_res = neq_int; // Start after integral equation rows

            // Field equations only
            for (int i = 0; i < neq; ++i) {
                bool affects_eq = is_var_double || eq[i]->take_into_account(zedom) ||
                                  eq[i]->take_into_account(zedoms(0)) || eq[i]->take_into_account(zedoms(1));

                if (affects_eq) {
                    for (int r = 0; r < eq[i]->get_n_cond_tot(); ++r) {
                        affected_rows.insert(pos_res + r);
                    }
                }
                pos_res += eq[i]->get_n_cond_tot();
            }
        }
    }
} // namespace Kadath
