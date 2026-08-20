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
 */

#include "For_Kadath/Array/headcpp.hpp"
#include "For_Kadath/Array/exceptions.hpp"

#include "For_Kadath/Domain/spheric.hpp"
#include "For_Kadath/Scalar/scalar.hpp"
#include "For_Kadath/Tensor/tensor.hpp"
namespace Kadath
{
    int Domain_nucleus::nbr_conditions_val_domain_vr(const Val_domain& so, int order) const
    {

        int res = 0;

        Index pos(nbr_coefs);
        do {
            bool indic = true;
            // True coef in phi ?
            if ((pos(2) == 1) || (pos(2) == nbr_coefs(2) - 1))
                indic = false;
            // Get base in theta :
            int baset = (*so.get_base().bases_1d[1])(pos(2));
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
                    KADATH_THROW("Unknow theta basis in Domain_nucleus::nbr_conditions_val_domain_vr");
            }

            int max = 0;
            if (indic) {
                // Base in r :
                int baser = (*so.get_base().bases_1d[0])(pos(1), pos(2));
                switch (baser) {
                    case CHEB_EVEN:
                        if (pos(0) == 0)
                            indic = false;
                        max = nbr_coefs(0);
                        break;
                    case LEG_EVEN:
                        if (pos(0) == 0)
                            indic = false;
                        max = nbr_coefs(0);
                        break;
                    case CHEB_ODD:
                        if (pos(0) == nbr_coefs(0) - 1)
                            indic = false;
                        if ((pos(2) == 0) && (pos(1) > 1) && (pos(0) == 0))
                            indic = false;
                        if ((pos(2) > 0) && (pos(0) == 0))
                            indic = false;
                        max = nbr_coefs(0) - 1;
                        break;
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
                        KADATH_THROW("Unknow radial basis in Domain_nucleus::nbr_conditions_val_domain_vr");
                }
            }
            // Order with respect to r :
            int lim = 0;
            switch (order) {
                case 2:
                    lim = max - 2;
                    break;
                case 0:
                    lim = max - 1;
                    break;
                default:
                    KADATH_THROW("Unknown case in Domain_nucleus_nbr_conditions");
            }
            if (pos(0) > lim)
                indic = false;
            if (indic)
                res++;
        } while (pos.inc());

        return res;
    }

    int Domain_nucleus::nbr_conditions_val_domain_vt(const Val_domain& so, int order) const
    {

        int res = 0;

        Index pos(nbr_coefs);
        do {
            bool indic = true;
            // True coef in phi ?
            if ((pos(2) == 1) || (pos(2) == nbr_coefs(2) - 1))
                indic = false;
            // Get base in theta :
            int baset = (*so.get_base().bases_1d[1])(pos(2));
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
                    KADATH_THROW("Unknow theta basis in Domain_nucleus::nbr_conditions_val_domain_vt");
            }

            int max = 0;
            if (indic) {
                // Base in r :
                int baser = (*so.get_base().bases_1d[0])(pos(1), pos(2));
                switch (baser) {
                    case CHEB_EVEN:
                        if (((pos(2) == 2) || (pos(2) == 3)) && (pos(1) > 0) && (pos(0) == 0))
                            indic = false;
                        if ((pos(2) > 3) && (pos(0) == 0))
                            indic = false;
                        max = nbr_coefs(0);
                        break;
                    case LEG_EVEN:
                        if (((pos(2) == 2) || (pos(2) == 3)) && (pos(1) > 0) && (pos(0) == 0))
                            indic = false;
                        if ((pos(2) > 3) && (pos(0) == 0))
                            indic = false;
                        max = nbr_coefs(0);
                        break;
                    case CHEB_ODD:
                        if (pos(0) == nbr_coefs(0) - 1)
                            indic = false;
                        if (pos(0) == 0)
                            indic = false;
                        max = nbr_coefs(0) - 1;
                        break;
                    case LEG_ODD:
                        if (pos(0) == nbr_coefs(0) - 1)
                            indic = false;
                        if (pos(0) == 0)
                            indic = false;
                        max = nbr_coefs(0) - 1;
                        break;
                    default:
                        KADATH_THROW("Unknow radial basis in Domain_nucleus::nbr_conditions_val_domain_vt");
                }
            }
            // Order with respect to r :
            int lim = 0;
            switch (order) {
                case 2:
                    lim = max - 2;
                    break;
                case 0:
                    lim = max - 1;
                    break;
                default:
                    KADATH_THROW("Unknown case in Domain_nucleus_nbr_conditions");
            }
            if (pos(0) > lim)
                indic = false;
            if (indic)
                res++;
        } while (pos.inc());

        return res;
    }

    int Domain_nucleus::nbr_conditions_val_domain_vp(const Val_domain& so, int order) const
    {

        int res = 0;

        Index pos(nbr_coefs);
        do {
            bool indic = true;
            // True coef in phi ?
            if ((pos(2) == 1) || (pos(2) == nbr_coefs(2) - 1))
                indic = false;
            // Get base in theta :
            int baset = (*so.get_base().bases_1d[1])(pos(2));
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
                    KADATH_THROW("Unknow theta basis in Domain_nucleus::nbr_conditions_val_domain_vp");
            }

            int max = 0;
            if (indic) {
                // Base in r :
                int baser = (*so.get_base().bases_1d[0])(pos(1), pos(2));
                switch (baser) {
                    case CHEB_EVEN:
                        if (((pos(2) == 2) || (pos(2) == 3)) && (pos(1) > 0) && (pos(0) == 0))
                            indic = false;
                        if ((pos(2) > 3) && (pos(0) == 0))
                            indic = false;
                        max = nbr_coefs(0);
                        break;
                    case LEG_EVEN:
                        if (((pos(2) == 2) || (pos(2) == 3)) && (pos(1) > 0) && (pos(0) == 0))
                            indic = false;
                        if ((pos(2) > 3) && (pos(0) == 0))
                            indic = false;
                        max = nbr_coefs(0);
                        break;
                    case CHEB_ODD:
                        if (pos(0) == nbr_coefs(0) - 1)
                            indic = false;
                        if ((pos(2) == 0) && (pos(1) > 0) && (pos(0) == 0))
                            indic = false;
                        if (((pos(2) == 4) || (pos(2) == 5)) && (pos(1) > 0) && (pos(0) == 0))
                            indic = false;
                        if ((pos(2) > 5) && (pos(0) == 0))
                            indic = false;
                        max = nbr_coefs(0) - 1;
                        break;
                    case LEG_ODD:
                        if (pos(0) == nbr_coefs(0) - 1)
                            indic = false;
                        if ((pos(2) == 0) && (pos(1) > 0) && (pos(0) == 0))
                            indic = false;
                        if (((pos(2) == 4) || (pos(2) == 5)) && (pos(1) > 0) && (pos(0) == 0))
                            indic = false;
                        if ((pos(2) > 5) && (pos(0) == 0))
                            indic = false;
                        max = nbr_coefs(0) - 1;
                        break;
                    default:
                        KADATH_THROW("Unknow radial basis in Domain_nucleus::nbr_conditions_val_domain_vp");
                }
            }
            // Order with respect to r :
            int lim = 0;
            switch (order) {
                case 2:
                    lim = max - 2;
                    break;
                case 0:
                    lim = max - 1;
                    break;
                default:
                    KADATH_THROW("Unknown case in Domain_nucleus_nbr_conditions");
            }
            if (pos(0) > lim)
                indic = false;
            if (indic)
                res++;
        } while (pos.inc());

        return res;
    }

    int Domain_nucleus::nbr_conditions_val_domain(const Val_domain& so, int mlim, int llim, int order) const
    {

        int res = 0;
        int kmin = 2 * mlim + 2;

        Index pos(nbr_coefs);
        do {
            bool indic = true;
            // True coef in phi ?
            if ((pos(2) == 1) || (pos(2) == nbr_coefs(2) - 1))
                indic = false;
            // Get base in theta :
            int baset = (*so.get_base().bases_1d[1])(pos(2));
            int lquant;
            switch (baset) {
                case COS_EVEN:
                    if ((pos(1) == 0) && (pos(2) >= kmin))
                        indic = false;
                    lquant = 2 * pos(1);
                    break;
                case COS_ODD:
                    if ((pos(1) == nbr_coefs(1) - 1) || ((pos(1) == 0) && (pos(2) >= kmin)))
                        indic = false;
                    lquant = 2 * pos(1) + 1;
                    break;
                case SIN_EVEN:
                    if (((pos(1) == 1) && (pos(2) >= kmin + 2)) || (pos(1) == 0) || (pos(1) == nbr_coefs(1) - 1))
                        indic = false;
                    lquant = 2 * pos(1);
                    break;
                case SIN_ODD:
                    if (((pos(1) == 0) && (pos(2) >= kmin + 2)) || (pos(1) == nbr_coefs(1) - 1))
                        indic = false;
                    lquant = 2 * pos(1) + 1;
                    break;
                default:
                    KADATH_THROW("Unknow theta basis in Domain_nucleus::nbr_unknowns_val_domain");
            }

            int max = 0;
            if (indic) {
                // Base in r :
                int baser = (*so.get_base().bases_1d[0])(pos(1), pos(2));

                switch (baser) {
                    case CHEB_EVEN:
                        if ((lquant > llim) && (pos(0) == 0))
                            indic = false;
                        max = nbr_coefs(0);
                        break;
                    case LEG_EVEN:
                        if ((lquant > llim) && (pos(0) == 0))
                            indic = false;
                        max = nbr_coefs(0);
                        break;
                    case CHEB_ODD:
                        if ((lquant > llim + 1) && (pos(0) == 0))
                            indic = false;
                        max = nbr_coefs(0) - 1;
                        break;
                    case LEG_ODD:
                        if ((lquant > llim + 1) && (pos(0) == 0))
                            indic = false;
                        max = nbr_coefs(0) - 1;
                        break;
                    default:
                        KADATH_THROW("Unknow radial basis in Domain_nucleus::nbr_unknowns_val_domain");
                }
            }

            // Order with respect to r :
            int lim = 0;
            switch (order) {
                case 2:
                    lim = max - 2;
                    break;
                case 1: {
                    if ((pos(1) == 0) && (pos(2) == 0))
                        lim = max - 2;
                    else
                        lim = max - 1;
                } break;
                case 0:
                    lim = max - 1;
                    break;
                default:
                    KADATH_THROW("Unknown case in Domain_nucleus_nbr_conditions");
            }

            if (pos(0) > lim)
                indic = false;

            if (indic)
                res++;
        } while (pos.inc());
        return res;
    }

    Array<int> Domain_nucleus::nbr_conditions(const Tensor& tt, int dom, int order, int n_cmp, Array<int>** p_cmp) const
    {

        int size = (n_cmp == -1) ? tt.get_n_comp() : n_cmp;
        Array<int> res(size);
        int val = tt.get_valence();
        switch (val) {
            case 0:
                if (!tt.is_m_order_affected())
                    res.set(0) = nbr_conditions_val_domain(tt()(dom), 0, 0, order);
                else
                    res.set(0) = nbr_conditions_val_domain(tt()(dom), tt.get_parameters()->get_m_order(), 0, order);
                break;
            case 1: {
                bool found = false;
                // Cartesian basis
                if (tt.get_basis().get_basis(dom) == CARTESIAN_BASIS) {
                    if (n_cmp == -1) {
                        res.set(0) = nbr_conditions_val_domain(tt(1)(dom), 0, 0, order);
                        res.set(1) = nbr_conditions_val_domain(tt(2)(dom), 0, 0, order);
                        res.set(2) = nbr_conditions_val_domain(tt(3)(dom), 0, 0, order);
                    } else
                        for (int i = 0; i < n_cmp; i++) {
                            if ((*p_cmp[i])(0) == 1)
                                res.set(i) = nbr_conditions_val_domain(tt(1)(dom), 0, 0, order);
                            if ((*p_cmp[i])(0) == 2)
                                res.set(i) = nbr_conditions_val_domain(tt(2)(dom), 0, 0, order);
                            if ((*p_cmp[i])(0) == 3)
                                res.set(i) = nbr_conditions_val_domain(tt(3)(dom), 0, 0, order);
                        }
                    found = true;
                }
                // Spherical coordinates
                if (tt.get_basis().get_basis(dom) == SPHERICAL_BASIS) {
                    if (n_cmp == -1) {
                        res.set(0) = nbr_conditions_val_domain_vr(tt(1)(dom), order);
                        res.set(1) = nbr_conditions_val_domain_vt(tt(2)(dom), order);
                        res.set(2) = nbr_conditions_val_domain_vp(tt(3)(dom), order);
                    } else
                        for (int i = 0; i < n_cmp; i++) {
                            if ((*p_cmp[i])(0) == 1)
                                res.set(i) = nbr_conditions_val_domain_vr(tt(1)(dom), order);
                            if ((*p_cmp[i])(0) == 2)
                                res.set(i) = nbr_conditions_val_domain_vt(tt(2)(dom), order);
                            if ((*p_cmp[i])(0) == 3)
                                res.set(i) = nbr_conditions_val_domain_vp(tt(3)(dom), order);
                        }
                    found = true;
                }
                if (!found) {
                    KADATH_THROW("Unknown type of vector Domain_nucleus::nbr_conditions");
                }
            } break;
            case 2: {
                bool found = false;
                // Cartesian basis and symetric
                if ((tt.get_basis().get_basis(dom) == CARTESIAN_BASIS) && (tt.get_n_comp() == 6)) {
                    if (n_cmp == -1) {
                        res.set(0) = nbr_conditions_val_domain(tt(1, 1)(dom), 0, 0, order);
                        res.set(1) = nbr_conditions_val_domain(tt(1, 2)(dom), 0, 0, order);
                        res.set(2) = nbr_conditions_val_domain(tt(1, 3)(dom), 0, 0, order);
                        res.set(3) = nbr_conditions_val_domain(tt(2, 2)(dom), 0, 0, order);
                        res.set(4) = nbr_conditions_val_domain(tt(2, 3)(dom), 0, 0, order);
                        res.set(5) = nbr_conditions_val_domain(tt(3, 3)(dom), 0, 0, order);
                    } else
                        for (int i = 0; i < n_cmp; i++) {
                            if (((*p_cmp[i])(0) == 1) && ((*p_cmp[i])(1) == 1))
                                res.set(i) = nbr_conditions_val_domain(tt(1, 1)(dom), 0, 0, order);
                            if (((*p_cmp[i])(0) == 1) && ((*p_cmp[i])(1) == 2))
                                res.set(i) = nbr_conditions_val_domain(tt(1, 2)(dom), 0, 0, order);
                            if (((*p_cmp[i])(0) == 1) && ((*p_cmp[i])(1) == 3))
                                res.set(i) = nbr_conditions_val_domain(tt(1, 3)(dom), 0, 0, order);
                            if (((*p_cmp[i])(0) == 2) && ((*p_cmp[i])(1) == 2))
                                res.set(i) = nbr_conditions_val_domain(tt(2, 2)(dom), 0, 0, order);
                            if (((*p_cmp[i])(0) == 2) && ((*p_cmp[i])(1) == 3))
                                res.set(i) = nbr_conditions_val_domain(tt(2, 3)(dom), 0, 0, order);
                            if (((*p_cmp[i])(0) == 3) && ((*p_cmp[i])(1) == 3))
                                res.set(i) = nbr_conditions_val_domain(tt(3, 3)(dom), 0, 0, order);
                        }
                    found = true;
                }
                // Cartesian basis and not symetric
                if ((tt.get_basis().get_basis(dom) == CARTESIAN_BASIS) && (tt.get_n_comp() == 9)) {
                    if (n_cmp == -1) {
                        res.set(0) = nbr_conditions_val_domain(tt(1, 1)(dom), 0, 0, order);
                        res.set(1) = nbr_conditions_val_domain(tt(1, 2)(dom), 0, 0, order);
                        res.set(2) = nbr_conditions_val_domain(tt(1, 3)(dom), 0, 0, order);
                        res.set(3) = nbr_conditions_val_domain(tt(2, 1)(dom), 0, 0, order);
                        res.set(4) = nbr_conditions_val_domain(tt(2, 2)(dom), 0, 0, order);
                        res.set(5) = nbr_conditions_val_domain(tt(2, 3)(dom), 0, 0, order);
                        res.set(6) = nbr_conditions_val_domain(tt(3, 1)(dom), 0, 0, order);
                        res.set(7) = nbr_conditions_val_domain(tt(3, 2)(dom), 0, 0, order);
                        res.set(8) = nbr_conditions_val_domain(tt(3, 3)(dom), 0, 0, order);
                    } else
                        for (int i = 0; i < n_cmp; i++) {
                            if (((*p_cmp[i])(0) == 1) && ((*p_cmp[i])(1) == 1))
                                res.set(i) = nbr_conditions_val_domain(tt(1, 1)(dom), 0, 0, order);
                            if (((*p_cmp[i])(0) == 1) && ((*p_cmp[i])(1) == 2))
                                res.set(i) = nbr_conditions_val_domain(tt(1, 2)(dom), 0, 0, order);
                            if (((*p_cmp[i])(0) == 1) && ((*p_cmp[i])(1) == 3))
                                res.set(i) = nbr_conditions_val_domain(tt(1, 3)(dom), 0, 0, order);
                            if (((*p_cmp[i])(0) == 2) && ((*p_cmp[i])(1) == 1))
                                res.set(i) = nbr_conditions_val_domain(tt(2, 1)(dom), 0, 0, order);
                            if (((*p_cmp[i])(0) == 2) && ((*p_cmp[i])(1) == 2))
                                res.set(i) = nbr_conditions_val_domain(tt(2, 2)(dom), 0, 0, order);
                            if (((*p_cmp[i])(0) == 2) && ((*p_cmp[i])(1) == 3))
                                res.set(i) = nbr_conditions_val_domain(tt(2, 3)(dom), 0, 0, order);
                            if (((*p_cmp[i])(0) == 3) && ((*p_cmp[i])(1) == 1))
                                res.set(i) = nbr_conditions_val_domain(tt(3, 1)(dom), 0, 0, order);
                            if (((*p_cmp[i])(0) == 3) && ((*p_cmp[i])(1) == 2))
                                res.set(i) = nbr_conditions_val_domain(tt(3, 2)(dom), 0, 0, order);
                            if (((*p_cmp[i])(0) == 3) && ((*p_cmp[i])(1) == 3))
                                res.set(i) = nbr_conditions_val_domain(tt(3, 3)(dom), 0, 0, order);
                        }
                    found = true;
                }
                if (!found) {
                    KADATH_THROW("Unknown type of 2-tensor Domain_nucleus::nbr_conditions");
                }
            } break;
            default:
                cerr << "Valence " << val << " not implemented in Domain_nucleus::nbr_conditions" << endl;
                break;
        }
        return res;
    }

    bool Domain_nucleus::describe_volume_residual_rows(
        const Tensor& tt, int dom, int order, const Array<int>& ncond,
        int n_cmp, Array<int>** p_cmp,
        std::vector<ResidualRowDescriptor>& descriptors) const
    {
        descriptors.clear();
        std::vector<int> components;
        if (tt.get_valence() > 2 ||
            (tt.get_valence() == 1 && tt.get_n_comp() != 3) ||
            (tt.get_valence() == 2 && tt.get_n_comp() != 6 &&
             tt.get_n_comp() != 9) ||
            order < 0 || order > 2 ||
            !residual_tensor_components_in_tau_order(
                tt, dom, n_cmp, p_cmp, components) ||
            ncond.get_nbr() != components.size() ||
            (tt.get_valence() > 0 &&
             tt.get_basis().get_basis(dom) != CARTESIAN_BASIS)) {
            return false;
        }

        for (std::size_t slot = 0; slot < components.size(); ++slot) {
            const int component = components[slot];
            const Array<int> indices = tt.indices(component);
            const Val_domain& field = tt.get_valence() == 0
                                          ? tt()(dom)
                                          : tt(indices)(dom);
            if (!field.get_base().is_def() ||
                field.get_base().get_base_1d(0) == nullptr ||
                field.get_base().get_base_1d(1) == nullptr) {
                descriptors.clear();
                return false;
            }
            const int mlim = tt.get_valence() == 0 &&
                                     tt.is_m_order_affected()
                                 ? tt.get_parameters()->get_m_order()
                                 : 0;
            const int kmin = 2 * mlim + 2;
            const std::size_t before = descriptors.size();
            for (int k = 0; k < nbr_coefs(2) - 1; ++k) {
                if (k == 1)
                    continue;
                const int theta_basis =
                    (*field.get_base().get_base_1d(1))(k);
                for (int j = 0; j < nbr_coefs(1); ++j) {
                    bool true_theta = true;
                    int lquant = 0;
                    switch (theta_basis) {
                        case COS_EVEN:
                            true_theta = k < kmin || j != 0;
                            lquant = 2 * j;
                            break;
                        case COS_ODD:
                            true_theta = j != nbr_coefs(1) - 1 &&
                                         (k < kmin || j != 0);
                            lquant = 2 * j + 1;
                            break;
                        case SIN_EVEN:
                            true_theta = j != 0 &&
                                         j != nbr_coefs(1) - 1 &&
                                         (k < kmin + 2 || j != 1);
                            lquant = 2 * j;
                            break;
                        case SIN_ODD:
                            true_theta = j != nbr_coefs(1) - 1 &&
                                         (k < kmin + 2 || j != 0);
                            lquant = 2 * j + 1;
                            break;
                        default:
                            descriptors.clear();
                            return false;
                    }
                    if (!true_theta)
                        continue;

                    const int radial_basis =
                        (*field.get_base().get_base_1d(0))(j, k);
                    int radial_max = 0;
                    int first_regular_l = 0;
                    switch (radial_basis) {
                        case CHEB_EVEN:
                        case LEG_EVEN:
                            radial_max = nbr_coefs(0);
                            first_regular_l = 0;
                            break;
                        case CHEB_ODD:
                        case LEG_ODD:
                            radial_max = nbr_coefs(0) - 1;
                            first_regular_l = 1;
                            break;
                        default:
                            descriptors.clear();
                            return false;
                    }

                    int radial_limit = radial_max - 1;
                    if (order == 2 ||
                        (order == 1 && j == 0 && k == 0)) {
                        radial_limit = radial_max - 2;
                    }
                    for (int i = 0; i < nbr_coefs(0); ++i) {
                        if ((lquant > first_regular_l && i == 0) ||
                            i > radial_limit) {
                            continue;
                        }
                        ResidualRowDescriptor descriptor;
                        if (!append_volume_residual_row(
                                field, dom, component, k, descriptor)) {
                            descriptors.clear();
                            return false;
                        }
                        descriptors.push_back(std::move(descriptor));
                    }
                }
            }
            if (descriptors.size() - before !=
                static_cast<std::size_t>(ncond(static_cast<int>(slot)))) {
                descriptors.clear();
                return false;
            }
        }
        return true;
    }
} // namespace Kadath
