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
#include "For_Kadath/Domain/polar_periodic.hpp"
#include "For_Kadath/Array/point.hpp"
#include "For_Kadath/Array/array.hpp"
#include "For_Kadath/Scalar/scalar.hpp"
#include "For_Kadath/Tensor/tensor.hpp"

namespace Kadath
{
    int Domain_polar_periodic_nucleus::nbr_conditions_val_domain(const Val_domain& so, int llim, int order) const
    {
        int res = 0;

        Index pos(nbr_coefs);
        do {
            bool indic = true;

            // Base in time
            int basetime = (*so.get_base().bases_1d[2])(0);
            switch (basetime) {
                case COS:
                    break;
                case SIN:
                    if ((pos(2) == 0) || (pos(2) == nbr_coefs(2) - 1))
                        indic = false;
                    break;
                default:
                    KADATH_THROW("Unknow time basis in Domain_polar_polar_nucleus::nbr_conditions_val_domain");
            }

            // Get base in theta :
            int baset = (*so.get_base().bases_1d[1])(pos(2));
            int lquant;
            switch (baset) {
                case COS_EVEN:
                    lquant = 2 * pos(1);
                    break;
                case COS_ODD:
                    if (pos(1) == nbr_coefs(1) - 1)
                        indic = false;
                    lquant = 2 * pos(1) + 1;
                    break;
                case SIN_EVEN:
                    if ((pos(1) == 0) || (pos(1) == nbr_coefs(1) - 1))
                        indic = false;
                    lquant = 2 * pos(1);
                    break;
                case SIN_ODD:
                    if (pos(1) == nbr_coefs(1) - 1)
                        indic = false;
                    lquant = 2 * pos(1) + 1;
                    break;
                default:
                    KADATH_THROW("Unknow theta basis in Domain_polar_periodic_nucleus::nbr_conditions_val_domain");
            }
            int max = 0;
            if (indic) {
                // Base in r :
                int baser = (*so.get_base().bases_1d[0])(pos(1), pos(2));
                switch (baser) {
                    case CHEB_EVEN:
                        if ((pos(0) == 0) && (lquant > llim))
                            indic = false;
                        max = nbr_coefs(0);
                        break;
                    case LEG_EVEN:
                        if ((pos(0) == 0) && (lquant > llim))
                            indic = false;
                        max = nbr_coefs(0);
                        break;
                    case CHEB_ODD:
                        if (((pos(0) == 0) && (lquant > llim)) || (pos(0) == nbr_coefs(0) - 1))
                            indic = false;
                        max = nbr_coefs(0) - 1;
                        break;
                    case LEG_ODD:
                        if (((pos(0) == 0) && (lquant > llim)) || (pos(0) == nbr_coefs(0) - 1))
                            indic = false;
                        max = nbr_coefs(0) - 1;
                        break;
                    default:
                        KADATH_THROW("Unknown base in Domain_polar_periodic_nucleus::nbr_conditions_val_domain");
                }
            }

            // Order with respect to r :
            int lim = 0;
            switch (order) {
                case 2:
                    lim = max - 1;
                    break;
                case 0:
                    lim = max;
                    break;
                default:
                    KADATH_THROW("Unknown case in Domain_polar_periodic_nucleus_nbr_conditions");
            }

            if (pos(0) >= lim)
                indic = false;

            if (indic)
                res++;

        } while (pos.inc());

        return res;
    }

    Array<int> Domain_polar_periodic_nucleus::nbr_conditions(const Tensor& tt, int dom, int order, int n_cmp,
                                                             Array<int>** p_cmp) const
    {

        int size = (n_cmp == -1) ? tt.get_n_comp() : n_cmp;
        Array<int> res(size);
        int val = tt.get_valence();
        switch (val) {
            case 0:
                res.set(0) = nbr_conditions_val_domain(tt()(dom), 0, order);
                break;
            case 1:
                if (n_cmp == -1) {
                    res.set(0) = nbr_conditions_val_domain(tt(1)(dom), 2, order);
                    res.set(1) = nbr_conditions_val_domain(tt(2)(dom), 2, order);
                    res.set(2) = nbr_conditions_val_domain(tt(3)(dom), 2, order);
                } else
                    for (int i = 0; i < n_cmp; i++) {
                        if ((*p_cmp[i])(0) == 1)
                            res.set(i) = nbr_conditions_val_domain(tt(1)(dom), 2, order);
                        if ((*p_cmp[i])(0) == 2)
                            res.set(i) = nbr_conditions_val_domain(tt(2)(dom), 2, order);
                        if ((*p_cmp[i])(0) == 3)
                            res.set(i) = nbr_conditions_val_domain(tt(3)(dom), 2, order);
                    }
                break;
            case 2:
                if (tt.get_n_comp() == 6) {
                    if (n_cmp == -1) {
                        res.set(0) = nbr_conditions_val_domain(tt(1, 1)(dom), 2, order);
                        res.set(1) = nbr_conditions_val_domain(tt(1, 2)(dom), 2, order);
                        res.set(2) = nbr_conditions_val_domain(tt(1, 3)(dom), 2, order);
                        res.set(3) = nbr_conditions_val_domain(tt(2, 2)(dom), 2, order);
                        res.set(4) = nbr_conditions_val_domain(tt(2, 3)(dom), 2, order);
                        res.set(5) = nbr_conditions_val_domain(tt(3, 3)(dom), 2, order);
                    } else
                        for (int i = 0; i < n_cmp; i++) {
                            if (((*p_cmp[i])(0) == 1) && ((*p_cmp[i])(1) == 1))
                                res.set(i) = nbr_conditions_val_domain(tt(1, 1)(dom), 2, order);
                            if (((*p_cmp[i])(0) == 1) && ((*p_cmp[i])(1) == 2))
                                res.set(i) = nbr_conditions_val_domain(tt(1, 2)(dom), 2, order);
                            if (((*p_cmp[i])(0) == 1) && ((*p_cmp[i])(1) == 3))
                                res.set(i) = nbr_conditions_val_domain(tt(1, 3)(dom), 2, order);
                            if (((*p_cmp[i])(0) == 2) && ((*p_cmp[i])(1) == 2))
                                res.set(i) = nbr_conditions_val_domain(tt(2, 2)(dom), 2, order);
                            if (((*p_cmp[i])(0) == 2) && ((*p_cmp[i])(1) == 3))
                                res.set(i) = nbr_conditions_val_domain(tt(2, 3)(dom), 2, order);
                            if (((*p_cmp[i])(0) == 3) && ((*p_cmp[i])(1) == 3))
                                res.set(i) = nbr_conditions_val_domain(tt(3, 3)(dom), 2, order);
                        }
                }
                // not symetric
                if (tt.get_n_comp() == 9) {
                    if (n_cmp == -1) {
                        res.set(0) = nbr_conditions_val_domain(tt(1, 1)(dom), 2, order);
                        res.set(1) = nbr_conditions_val_domain(tt(1, 2)(dom), 2, order);
                        res.set(2) = nbr_conditions_val_domain(tt(1, 3)(dom), 2, order);
                        res.set(3) = nbr_conditions_val_domain(tt(2, 1)(dom), 2, order);
                        res.set(4) = nbr_conditions_val_domain(tt(2, 2)(dom), 2, order);
                        res.set(5) = nbr_conditions_val_domain(tt(2, 3)(dom), 2, order);
                        res.set(6) = nbr_conditions_val_domain(tt(3, 1)(dom), 2, order);
                        res.set(7) = nbr_conditions_val_domain(tt(3, 2)(dom), 2, order);
                        res.set(8) = nbr_conditions_val_domain(tt(3, 3)(dom), 2, order);
                    } else
                        for (int i = 0; i < n_cmp; i++) {
                            if (((*p_cmp[i])(0) == 1) && ((*p_cmp[i])(1) == 1))
                                res.set(i) = nbr_conditions_val_domain(tt(1, 1)(dom), 2, order);
                            if (((*p_cmp[i])(0) == 1) && ((*p_cmp[i])(1) == 2))
                                res.set(i) = nbr_conditions_val_domain(tt(1, 2)(dom), 2, order);
                            if (((*p_cmp[i])(0) == 1) && ((*p_cmp[i])(1) == 3))
                                res.set(i) = nbr_conditions_val_domain(tt(1, 3)(dom), 2, order);
                            if (((*p_cmp[i])(0) == 2) && ((*p_cmp[i])(1) == 1))
                                res.set(i) = nbr_conditions_val_domain(tt(2, 1)(dom), 2, order);
                            if (((*p_cmp[i])(0) == 2) && ((*p_cmp[i])(1) == 2))
                                res.set(i) = nbr_conditions_val_domain(tt(2, 2)(dom), 2, order);
                            if (((*p_cmp[i])(0) == 2) && ((*p_cmp[i])(1) == 3))
                                res.set(i) = nbr_conditions_val_domain(tt(2, 3)(dom), 2, order);
                            if (((*p_cmp[i])(0) == 3) && ((*p_cmp[i])(1) == 1))
                                res.set(i) = nbr_conditions_val_domain(tt(3, 1)(dom), 2, order);
                            if (((*p_cmp[i])(0) == 3) && ((*p_cmp[i])(1) == 2))
                                res.set(i) = nbr_conditions_val_domain(tt(3, 2)(dom), 2, order);
                            if (((*p_cmp[i])(0) == 3) && ((*p_cmp[i])(1) == 3))
                                res.set(i) = nbr_conditions_val_domain(tt(3, 3)(dom), 2, order);
                        }
                }
                break;
            default:
                cerr << "Valence " << val << " not implemented in Domain_polar_periodic_nucleus::nbr_conditions"
                     << endl;
                break;
        }
        return res;
    }
} // namespace Kadath
