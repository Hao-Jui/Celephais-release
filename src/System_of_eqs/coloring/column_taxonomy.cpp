#include "coloring_internals.hpp"

namespace Kadath
{
    namespace
    {
        bool lexicographically_later_mode(const Index& lhs, const Index& rhs, int ndim)
        {
            for (int dim = ndim - 1; dim >= 0; --dim) {
                if (lhs(dim) != rhs(dim))
                    return lhs(dim) > rhs(dim);
            }
            return false;
        }

        bool primary_nonzero_mode(const Val_domain& value, const Dim_array& nbr_coefs,
                                  Index& primary)
        {
            if (value.check_if_zero())
                return false;

            const Array<double> coeffs = value.get_coef();
            Index pos(nbr_coefs);
            bool found = false;
            do {
                if (std::fabs(coeffs(pos)) > 0.0) {
                    if (!found ||
                        lexicographically_later_mode(pos, primary, nbr_coefs.get_ndim())) {
                        primary = pos;
                    }
                    found = true;
                }
            } while (pos.inc());
            return found;
        }

        bool nucleus_scalar_condition_selected(const Val_domain& value, const Index& pos,
                                               const Dim_array& nbr_coefs, int mlim,
                                               int llim, int order)
        {
            bool indic = true;
            const int kmin = 2 * mlim + 2;
            if ((pos(2) == 1) || (pos(2) == nbr_coefs(2) - 1))
                indic = false;

            int lquant = 0;
            const int baset = (*value.get_base().get_base_1d(1))(pos(2));
            switch (baset) {
                case COS_EVEN:
                    if ((pos(1) == 0) && (pos(2) >= kmin))
                        indic = false;
                    lquant = 2 * pos(1);
                    break;
                case COS_ODD:
                    if ((pos(1) == nbr_coefs(1) - 1) ||
                        ((pos(1) == 0) && (pos(2) >= kmin)))
                        indic = false;
                    lquant = 2 * pos(1) + 1;
                    break;
                case SIN_EVEN:
                    if (((pos(1) == 1) && (pos(2) >= kmin + 2)) ||
                        (pos(1) == 0) || (pos(1) == nbr_coefs(1) - 1))
                        indic = false;
                    lquant = 2 * pos(1);
                    break;
                case SIN_ODD:
                    if (((pos(1) == 0) && (pos(2) >= kmin + 2)) ||
                        (pos(1) == nbr_coefs(1) - 1))
                        indic = false;
                    lquant = 2 * pos(1) + 1;
                    break;
                default:
                    KADATH_THROW("Unknown theta basis in nucleus scalar volume classification");
            }

            int max = 0;
            if (indic) {
                const int baser = (*value.get_base().get_base_1d(0))(pos(1), pos(2));
                switch (baser) {
                    case CHEB_EVEN:
                    case LEG_EVEN:
                        if ((lquant > llim) && (pos(0) == 0))
                            indic = false;
                        max = nbr_coefs(0);
                        break;
                    case CHEB_ODD:
                    case LEG_ODD:
                        if ((lquant > llim + 1) && (pos(0) == 0))
                            indic = false;
                        max = nbr_coefs(0) - 1;
                        break;
                    default:
                        KADATH_THROW("Unknown radial basis in nucleus scalar volume classification");
                }
            }

            int lim = 0;
            switch (order) {
                case 2:
                    lim = max - 2;
                    break;
                case 1:
                    lim = ((pos(1) == 0) && (pos(2) == 0)) ? max - 2 : max - 1;
                    break;
                case 0:
                    lim = max - 1;
                    break;
                default:
                    KADATH_THROW("Unknown order in nucleus scalar volume classification");
            }

            return indic && pos(0) <= lim;
        }

        bool nucleus_vr_condition_selected(const Val_domain& value, const Index& pos,
                                           const Dim_array& nbr_coefs, int order)
        {
            bool indic = true;
            if ((pos(2) == 1) || (pos(2) == nbr_coefs(2) - 1))
                indic = false;

            const int baset = (*value.get_base().get_base_1d(1))(pos(2));
            switch (baset) {
                case COS_EVEN:
                    if ((pos(1) == 0) && (pos(2) != 0))
                        indic = false;
                    break;
                case SIN_ODD:
                    if (pos(1) == nbr_coefs(1) - 1)
                        indic = false;
                    break;
                default:
                    KADATH_THROW("Unknown theta basis in nucleus vr volume classification");
            }

            int max = 0;
            if (indic) {
                const int baser = (*value.get_base().get_base_1d(0))(pos(1), pos(2));
                switch (baser) {
                    case CHEB_EVEN:
                    case LEG_EVEN:
                        if (pos(0) == 0)
                            indic = false;
                        max = nbr_coefs(0);
                        break;
                    case CHEB_ODD:
                    case LEG_ODD:
                        if (pos(0) == nbr_coefs(0) - 1)
                            indic = false;
                        if ((pos(2) == 0) && (pos(1) > 1) && (pos(0) == 0))
                            indic = false;
                        if ((pos(2) > 0) && (pos(0) == 0))
                            indic = false;
                        max = nbr_coefs(0) - 1;
                        break;
                    default:
                        KADATH_THROW("Unknown radial basis in nucleus vr volume classification");
                }
            }

            const int lim = (order == 2) ? max - 2 : max - 1;
            if (order != 0 && order != 2)
                KADATH_THROW("Unknown order in nucleus vr volume classification");
            return indic && pos(0) <= lim;
        }

        bool nucleus_vt_condition_selected(const Val_domain& value, const Index& pos,
                                           const Dim_array& nbr_coefs, int order)
        {
            bool indic = true;
            if ((pos(2) == 1) || (pos(2) == nbr_coefs(2) - 1))
                indic = false;

            const int baset = (*value.get_base().get_base_1d(1))(pos(2));
            switch (baset) {
                case SIN_EVEN:
                    if ((pos(1) == nbr_coefs(1) - 1) || (pos(1) == 0))
                        indic = false;
                    break;
                case COS_ODD:
                    if (pos(1) == nbr_coefs(1) - 1)
                        indic = false;
                    if ((pos(1) == 0) && (pos(2) > 3))
                        indic = false;
                    break;
                default:
                    KADATH_THROW("Unknown theta basis in nucleus vt volume classification");
            }

            int max = 0;
            if (indic) {
                const int baser = (*value.get_base().get_base_1d(0))(pos(1), pos(2));
                switch (baser) {
                    case CHEB_EVEN:
                    case LEG_EVEN:
                        if (((pos(2) == 2) || (pos(2) == 3)) &&
                            (pos(1) > 0) && (pos(0) == 0))
                            indic = false;
                        if ((pos(2) > 3) && (pos(0) == 0))
                            indic = false;
                        max = nbr_coefs(0);
                        break;
                    case CHEB_ODD:
                    case LEG_ODD:
                        if (pos(0) == nbr_coefs(0) - 1)
                            indic = false;
                        if (pos(0) == 0)
                            indic = false;
                        max = nbr_coefs(0) - 1;
                        break;
                    default:
                        KADATH_THROW("Unknown radial basis in nucleus vt volume classification");
                }
            }

            const int lim = (order == 2) ? max - 2 : max - 1;
            if (order != 0 && order != 2)
                KADATH_THROW("Unknown order in nucleus vt volume classification");
            return indic && pos(0) <= lim;
        }

        bool nucleus_vp_condition_selected(const Val_domain& value, const Index& pos,
                                           const Dim_array& nbr_coefs, int order)
        {
            bool indic = true;
            if ((pos(2) == 1) || (pos(2) == nbr_coefs(2) - 1))
                indic = false;

            const int baset = (*value.get_base().get_base_1d(1))(pos(2));
            switch (baset) {
                case COS_EVEN:
                    if ((pos(2) > 3) && (pos(1) == 0))
                        indic = false;
                    break;
                case SIN_ODD:
                    if (pos(1) == nbr_coefs(1) - 1)
                        indic = false;
                    break;
                default:
                    KADATH_THROW("Unknown theta basis in nucleus vp volume classification");
            }

            int max = 0;
            if (indic) {
                const int baser = (*value.get_base().get_base_1d(0))(pos(1), pos(2));
                switch (baser) {
                    case CHEB_EVEN:
                    case LEG_EVEN:
                        if (((pos(2) == 2) || (pos(2) == 3)) &&
                            (pos(1) > 0) && (pos(0) == 0))
                            indic = false;
                        if ((pos(2) > 3) && (pos(0) == 0))
                            indic = false;
                        max = nbr_coefs(0);
                        break;
                    case CHEB_ODD:
                    case LEG_ODD:
                        if (pos(0) == nbr_coefs(0) - 1)
                            indic = false;
                        if ((pos(2) == 0) && (pos(1) > 0) && (pos(0) == 0))
                            indic = false;
                        if (((pos(2) == 4) || (pos(2) == 5)) &&
                            (pos(1) > 0) && (pos(0) == 0))
                            indic = false;
                        if ((pos(2) > 5) && (pos(0) == 0))
                            indic = false;
                        max = nbr_coefs(0) - 1;
                        break;
                    default:
                        KADATH_THROW("Unknown radial basis in nucleus vp volume classification");
                }
            }

            const int lim = (order == 2) ? max - 2 : max - 1;
            if (order != 0 && order != 2)
                KADATH_THROW("Unknown order in nucleus vp volume classification");
            return indic && pos(0) <= lim;
        }

        bool nucleus_primary_mode_selected_by_volume(const Tensor& probe, int dom, int order)
        {
            const Dim_array nbr_coefs = probe.get_space().get_domain(dom)->get_nbr_coefs();
            const int valence = probe.get_valence();
            const int tensor_basis = probe.get_basis().get_basis(dom);
            const int mlim = probe.is_m_order_affected()
                ? probe.get_parameters()->get_m_order()
                : 0;

            for (int comp = 0; comp < probe.get_n_comp(); ++comp) {
                const Array<int> component(probe.indices(comp));
                const Val_domain& value = probe(component)(dom);
                Index primary(nbr_coefs);
                if (!primary_nonzero_mode(value, nbr_coefs, primary))
                    continue;

                if (valence == 1 && tensor_basis == SPHERICAL_BASIS) {
                    const int which = component(0);
                    if (which == 1)
                        return nucleus_vr_condition_selected(value, primary, nbr_coefs, order);
                    if (which == 2)
                        return nucleus_vt_condition_selected(value, primary, nbr_coefs, order);
                    if (which == 3)
                        return nucleus_vp_condition_selected(value, primary, nbr_coefs, order);
                }

                return nucleus_scalar_condition_selected(value, primary, nbr_coefs,
                                                         mlim, 0, order);
            }
            return false;
        }

        bool bispheric_phi_axis_mode_selected(const Val_domain& value, const Index& pos,
                                              const Dim_array& nbr_coefs, int forgot_axis,
                                              int axis_dim)
        {
            const int k = pos(2);
            const int axis = pos(axis_dim);
            const int axis_count = nbr_coefs(axis_dim);
            const int basep = (*value.get_base().get_base_1d(2))(0);

            switch (basep) {
                case COS:
                    if ((k % 2 == 1) && (axis == axis_count - 1 - forgot_axis))
                        return false;
                    if ((k == 0) || (k % 2 == 1))
                        return true;
                    return axis != 0;
                case SIN:
                    if ((k == 0) || (k == nbr_coefs(2) - 1))
                        return false;
                    if ((k % 2 == 1) && (axis == axis_count - 1 - forgot_axis))
                        return false;
                    if (k % 2 == 1)
                        return true;
                    return axis != 0;
                case COSSIN: {
                    if ((k == 1) || (k == nbr_coefs(2) - 1))
                        return false;
                    const int mm = (k % 2 == 0) ? k / 2 : (k - 1) / 2;
                    const bool odd_mode = (mm % 2 == 1);
                    if (odd_mode && (axis == axis_count - 1 - forgot_axis))
                        return false;
                    if ((k == 0) || odd_mode)
                        return true;
                    return axis != 0;
                }
                default:
                    KADATH_THROW("Unknown phi basis in bispheric volume classification");
            }
        }

        bool bispheric_condition_selected(const Val_domain& value, const Index& pos,
                                          const Dim_array& nbr_coefs, int order,
                                          const Domain* domain)
        {
            int forgot_chi = 0;
            int forgot_eta = 0;
            switch (order) {
                case 0:
                    forgot_chi = 0;
                    forgot_eta = 0;
                    break;
                case 1:
                    forgot_chi = 1;
                    forgot_eta = (dynamic_cast<const Domain_bispheric_eta_first*>(domain) ||
                                  dynamic_cast<const Domain_bispheric_eta_first_nosym*>(domain)) ? 2 : 1;
                    break;
                case 2:
                    forgot_chi = 1;
                    forgot_eta = 2;
                    break;
                default:
                    KADATH_THROW("Unknown order in bispheric volume classification");
            }

            if (dynamic_cast<const Domain_bispheric_chi_first*>(domain) != nullptr ||
                dynamic_cast<const Domain_bispheric_chi_first_nosym*>(domain) != nullptr) {
                if (pos(1) >= nbr_coefs(1) - forgot_chi || pos(0) >= nbr_coefs(0) - order)
                    return false;
                return bispheric_phi_axis_mode_selected(value, pos, nbr_coefs, forgot_chi, 1);
            }

            if (dynamic_cast<const Domain_bispheric_rect*>(domain) != nullptr ||
                dynamic_cast<const Domain_bispheric_rect_nosym*>(domain) != nullptr) {
                if (pos(1) >= nbr_coefs(1) - forgot_chi || pos(0) >= nbr_coefs(0) - forgot_eta)
                    return false;
                return bispheric_phi_axis_mode_selected(value, pos, nbr_coefs, forgot_chi, 1);
            }

            if (dynamic_cast<const Domain_bispheric_eta_first*>(domain) != nullptr ||
                dynamic_cast<const Domain_bispheric_eta_first_nosym*>(domain) != nullptr) {
                if (pos(1) >= nbr_coefs(1) - forgot_eta || pos(0) >= nbr_coefs(0) - forgot_chi)
                    return false;
                return bispheric_phi_axis_mode_selected(value, pos, nbr_coefs, forgot_chi, 0);
            }

            return false;
        }

        bool bispheric_primary_mode_selected_by_volume(const Tensor& probe, int dom, int order,
                                                       const Domain* domain)
        {
            const Dim_array nbr_coefs = probe.get_space().get_domain(dom)->get_nbr_coefs();
            for (int comp = 0; comp < probe.get_n_comp(); ++comp) {
                const Array<int> component(probe.indices(comp));
                const Val_domain& value = probe(component)(dom);
                Index primary(nbr_coefs);
                if (!primary_nonzero_mode(value, nbr_coefs, primary))
                    continue;
                return bispheric_condition_selected(value, primary, nbr_coefs, order, domain);
            }
            return false;
        }

        int scalar_volume_order(const Equation* equation)
        {
            if (dynamic_cast<const Eq_inside*>(equation) != nullptr)
                return 2;
            if (dynamic_cast<const Eq_full*>(equation) != nullptr)
                return 0;
            if (dynamic_cast<const Eq_one_side*>(equation) != nullptr)
                return 1;
            if (const auto* ordered = dynamic_cast<const Eq_order*>(equation))
                return ordered->order;
            if (const auto* vel_pot = dynamic_cast<const Eq_vel_pot*>(equation))
                return vel_pot->order;
            return -1;
        }

        bool equation_local_row_is_volume(const Equation* equation, int row)
        {
            if (equation == nullptr || row < 0)
                return false;
            if (dynamic_cast<const Eq_vel_pot*>(equation) != nullptr)
                return row > 0;
            return dynamic_cast<const Eq_inside*>(equation) != nullptr ||
                   dynamic_cast<const Eq_order*>(equation) != nullptr ||
                   dynamic_cast<const Eq_order_array*>(equation) != nullptr ||
                   dynamic_cast<const Eq_full*>(equation) != nullptr ||
                   dynamic_cast<const Eq_one_side*>(equation) != nullptr;
        }

    } // namespace

// =============================================================================
    // Column mapping construction
    // =============================================================================

    /*
     * Build column-to-variable mapping
     * Iterates through unknowns in the same order as do_col_J
     */
    bool System_of_eqs::field_coefficient_exported_by_equation(int term_idx, int global_col,
                                                               int term_start_col, int eq_idx,
                                                               bool volume_rows_only) const
    {
        if (term_idx < 0 || term_idx >= nterm || eq_idx < 0 || eq_idx >= neq)
            return false;
        const Equation* equation = eq[static_cast<std::size_t>(eq_idx)].get();
        if (equation == nullptr || equation->get_n_cond_tot() <= 0)
            return false;

        const int target_dom = term[static_cast<std::size_t>(term_idx)]->get_dom();
        const int target_var = assoc_var[static_cast<std::size_t>(term_idx)];

        std::vector<std::unique_ptr<Term_eq>> storage;
        std::vector<Term_eq*> raw;
        storage.reserve(static_cast<std::size_t>(equation->n_ope));
        raw.reserve(static_cast<std::size_t>(equation->n_ope));

        for (int part = 0; part < equation->n_ope; ++part) {
            const int part_dom = equation->parts[part]->get_dom();
            int source_term = term_idx;
            for (int candidate = 0; candidate < nterm; ++candidate) {
                if (assoc_var[static_cast<std::size_t>(candidate)] == target_var &&
                    term[static_cast<std::size_t>(candidate)]->get_dom() == part_dom) {
                    source_term = candidate;
                    break;
                }
            }

            Tensor probe(term[static_cast<std::size_t>(source_term)]->get_val_t(), false);
            probe.annule_hard();
            if (part_dom == target_dom) {
                int conte = term_start_col;
                espace.get_domain(target_dom)->affecte_tau_one_coef(
                    probe, target_dom, global_col, conte);
                const int ncomp = probe.get_n_comp();
                for (int comp = 0; comp < ncomp; ++comp) {
                    const Array<int> idx(probe.indices(comp));
                    if (!probe(idx)(target_dom).check_if_zero() &&
                        !probe(idx)(target_dom).get_base().is_def()) {
                        probe.set(idx).set_domain(target_dom).set_base() =
                            term[static_cast<std::size_t>(term_idx)]->get_val_t()(idx)(target_dom).get_base();
                    }
                }
            }

            storage.push_back(std::make_unique<Term_eq>(part_dom, probe));
            raw.push_back(storage.back().get());
        }

        Array<double> exported(equation->get_n_cond_tot());
        exported = 0.0;
        int conte = 0;
        int pos_res = 0;
        equation->export_val(conte, raw.data(), exported, pos_res);
        for (int row = 0; row < exported.get_size(0); ++row) {
            if (volume_rows_only && !equation_local_row_is_volume(equation, row))
                continue;
            if (std::fabs(exported(row)) > 0.0)
                return true;
        }
        return false;
    }

    bool System_of_eqs::field_coefficient_selected_by_volume_basis(int term_idx, int global_col,
                                                                   int term_start_col, int eq_idx) const
    {
        if (term_idx < 0 || term_idx >= nterm || eq_idx < 0 || eq_idx >= neq)
            return false;
        const Equation* equation = eq[static_cast<std::size_t>(eq_idx)].get();
        if (equation == nullptr)
            return false;
        const int dom = term[static_cast<std::size_t>(term_idx)]->get_dom();
        const Domain* domain = espace.get_domain(dom);
        const int order = scalar_volume_order(equation);
        const bool vel_pot = dynamic_cast<const Eq_vel_pot*>(equation) != nullptr;
        const bool bispheric =
            dynamic_cast<const Domain_bispheric_chi_first*>(domain) != nullptr ||
            dynamic_cast<const Domain_bispheric_rect*>(domain) != nullptr ||
            dynamic_cast<const Domain_bispheric_eta_first*>(domain) != nullptr ||
            dynamic_cast<const Domain_bispheric_chi_first_nosym*>(domain) != nullptr ||
            dynamic_cast<const Domain_bispheric_rect_nosym*>(domain) != nullptr ||
            dynamic_cast<const Domain_bispheric_eta_first_nosym*>(domain) != nullptr;

        auto basis_candidate = [&](int candidate_col) -> bool {
            if (dynamic_cast<const Domain_nucleus*>(domain) != nullptr) {
                if (order == 2) {
                    const int local_mode = candidate_col - term_start_col;
                    if (local_mode > 0) {
                        Tensor previous(term[static_cast<std::size_t>(term_idx)]->get_val_t(), false);
                        previous.annule_hard();
                        int previous_conte = term_start_col;
                        domain->affecte_tau_one_coef(previous, dom, candidate_col - 1,
                                                     previous_conte);
                        return nucleus_primary_mode_selected_by_volume(previous, dom,
                                                                       order);
                    }
                    return false;
                } else {
                    Tensor probe(term[static_cast<std::size_t>(term_idx)]->get_val_t(), false);
                    probe.annule_hard();
                    int conte = term_start_col;
                    domain->affecte_tau_one_coef(probe, dom, candidate_col, conte);
                    return nucleus_primary_mode_selected_by_volume(probe, dom, order);
                }
            } else if (bispheric) {
                Tensor probe(term[static_cast<std::size_t>(term_idx)]->get_val_t(), false);
                probe.annule_hard();
                int conte = term_start_col;
                domain->affecte_tau_one_coef(probe, dom, candidate_col, conte);
                return bispheric_primary_mode_selected_by_volume(probe, dom, order,
                                                                 domain);
            } else {
                const Dim_array nbr_coefs = domain->get_nbr_coefs();
                if (nbr_coefs.get_ndim() > 0 && nbr_coefs(0) > 0) {
                    const int local_mode = candidate_col - term_start_col;
                    const int radial_slot = local_mode % nbr_coefs(0);
                    return radial_slot >= order;
                }
                return false;
            }
        };

        bool candidate = false;
        if (order >= 0) {
            candidate = basis_candidate(global_col);
        } else {
            return field_coefficient_exported_by_equation(term_idx, global_col,
                                                          term_start_col, eq_idx);
        }
        if (!candidate)
            return false;
        if (vel_pot) {
            for (int candidate_col = term_start_col; candidate_col < global_col; ++candidate_col) {
                if (basis_candidate(candidate_col))
                    return true;
            }
            return false;
        }
        if (!bispheric)
            return true;
        return field_coefficient_exported_by_equation(term_idx, global_col,
                                                      term_start_col, eq_idx,
                                                      true);
    }

    ColumnClass System_of_eqs::classify_field_column_from_equations(int term_idx, int global_col,
                                                                    int term_start_col) const
    {
        if (term_idx < 0 || term_idx >= nterm)
            return ColumnClass::FieldUnknown;

        const int dom = term[static_cast<std::size_t>(term_idx)]->get_dom();
        const int var_idx = assoc_var[static_cast<std::size_t>(term_idx)];
        const std::string var_name =
            (var_idx >= 0 && !names_var[static_cast<std::size_t>(var_idx)].empty())
                ? names_var[static_cast<std::size_t>(var_idx)]
                : "";

        bool has_boundary_tau_owner = false;
        bool has_matching_owner = false;
        bool has_gauge_owner = false;
        const int attached = std::min<int>(neq, static_cast<int>(eq_column_attachments.size()));
        for (int eq_idx = 0; eq_idx < attached; ++eq_idx) {
            const EquationColumnAttachment& attachment =
                eq_column_attachments[static_cast<std::size_t>(eq_idx)];
            if (!var_name.empty() && attachment.owner_var_name != var_name)
                continue;

            const Equation* equation = eq[static_cast<std::size_t>(eq_idx)].get();
            const bool domain_relevant =
                attachment.domain == dom || (equation != nullptr && equation->take_into_account(dom));
            if (!domain_relevant)
                continue;

            switch (attachment.column_class) {
                case ColumnClass::FieldInteriorVol:
                    if (attachment.domain == dom &&
                        field_coefficient_selected_by_volume_basis(term_idx, global_col,
                                                                   term_start_col, eq_idx)) {
                        return ColumnClass::FieldInteriorVol;
                    }
                    break;
                case ColumnClass::FieldBoundaryTau:
                    has_boundary_tau_owner = true;
                    break;
                case ColumnClass::FieldMatching:
                    has_matching_owner = true;
                    break;
                case ColumnClass::FieldGauge:
                    has_gauge_owner = true;
                    break;
                default:
                    break;
            }
        }

        if (has_boundary_tau_owner)
            return ColumnClass::FieldBoundaryTau;
        if (has_matching_owner)
            return ColumnClass::FieldMatching;
        if (has_gauge_owner)
            return ColumnClass::FieldGauge;
        return ColumnClass::FieldUnknown;
    }
} // namespace Kadath
