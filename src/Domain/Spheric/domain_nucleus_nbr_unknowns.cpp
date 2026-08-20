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
    int Domain_nucleus::nbr_unknowns_val_domain_vr(const Val_domain& so) const
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
                    KADATH_THROW("Unknow theta basis in Domain::nbr_unknowns_val_domain_vr");
            }

            if (indic) {
                // Base in r :
                int baser = (*so.get_base().bases_1d[0])(pos(1), pos(2));
                switch (baser) {
                    case CHEB_EVEN:
                        if (pos(0) == 0)
                            indic = false;
                        break;
                    case LEG_EVEN:
                        if (pos(0) == 0)
                            indic = false;
                        break;
                    case CHEB_ODD:
                        if (pos(0) == nbr_coefs(0) - 1)
                            indic = false;
                        if ((pos(2) == 0) && (pos(1) > 1) && (pos(0) == 0))
                            indic = false;
                        if ((pos(2) > 0) && (pos(0) == 0))
                            indic = false;
                        break;
                    case LEG_ODD:
                        if (pos(0) == nbr_coefs(0) - 1)
                            indic = false;
                        if ((pos(2) == 0) && (pos(1) > 1) && (pos(0) == 0))
                            indic = false;
                        if ((pos(2) > 0) && (pos(0) == 0))
                            indic = false;
                        break;
                    default:
                        KADATH_THROW("Unknow radial basis in Domain::nbr_unknowns_val_domain_vr");
                }
            }

            if (indic)
                res++;
        } while (pos.inc());
        return res;
    }

    int Domain_nucleus::nbr_unknowns_val_domain_vt(const Val_domain& so) const
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
                    KADATH_THROW("Unknow theta basis in Domain_nucleus::nbr_unknowns_val_domain_vt");
            }

            if (indic) {
                // Base in r :
                int baser = (*so.get_base().bases_1d[0])(pos(1), pos(2));
                switch (baser) {
                    case CHEB_EVEN:
                        if (((pos(2) == 2) || (pos(2) == 3)) && (pos(1) > 0) && (pos(0) == 0))
                            indic = false;
                        if ((pos(2) > 3) && (pos(0) == 0))
                            indic = false;
                        break;
                    case LEG_EVEN:
                        if (((pos(2) == 2) || (pos(2) == 3)) && (pos(1) > 0) && (pos(0) == 0))
                            indic = false;
                        if ((pos(2) > 3) && (pos(0) == 0))
                            indic = false;
                        break;
                    case CHEB_ODD:
                        if (pos(0) == nbr_coefs(0) - 1)
                            indic = false;
                        if (pos(0) == 0)
                            indic = false;
                        break;
                    case LEG_ODD:
                        if (pos(0) == nbr_coefs(0) - 1)
                            indic = false;
                        if (pos(0) == 0)
                            indic = false;
                        break;
                    default:
                        KADATH_THROW("Unknow radial basis in Domain_nucleus::nbr_unknowns_val_domain_vt");
                }
            }

            if (indic)
                res++;
        } while (pos.inc());
        return res;
    }

    int Domain_nucleus::nbr_unknowns_val_domain_vp(const Val_domain& so) const
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
                    KADATH_THROW("Unknow theta basis in Domain_nucleus::nbr_unknowns_val_domain_vp");
            }

            if (indic) {
                // Base in r :
                int baser = (*so.get_base().bases_1d[0])(pos(1), pos(2));
                switch (baser) {
                    case CHEB_EVEN:
                        if (((pos(2) == 2) || (pos(2) == 3)) && (pos(1) > 0) && (pos(0) == 0))
                            indic = false;
                        if ((pos(2) > 3) && (pos(0) == 0))
                            indic = false;
                        break;
                    case LEG_EVEN:
                        if (((pos(2) == 2) || (pos(2) == 3)) && (pos(1) > 0) && (pos(0) == 0))
                            indic = false;
                        if ((pos(2) > 3) && (pos(0) == 0))
                            indic = false;
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
                        break;
                    default:
                        KADATH_THROW("Unknow radial basis in Domain_nucleus::nbr_unknowns_val_domain_vp");
                }
            }

            if (indic)
                res++;
        } while (pos.inc());
        return res;
    }

    int Domain_nucleus::nbr_unknowns_val_domain(const Val_domain& so, int mlim, int llim) const
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

            if (indic) {
                // Base in r :
                int baser = (*so.get_base().bases_1d[0])(pos(1), pos(2));
                switch (baser) {
                    case CHEB_EVEN:
                        if ((lquant > llim) && (pos(0) == 0))
                            indic = false;
                        break;
                    case LEG_EVEN:
                        if ((lquant > llim) && (pos(0) == 0))
                            indic = false;
                        break;
                    case CHEB_ODD:
                        if (((lquant > llim + 1) && (pos(0) == 0)) || (pos(0) == nbr_coefs(0) - 1))
                            indic = false;
                        break;
                    case LEG_ODD:
                        if (((lquant > llim + 1) && (pos(0) == 0)) || (pos(0) == nbr_coefs(0) - 1))
                            indic = false;
                        break;
                    default:
                        KADATH_THROW("Unknow radial basis in Domain_nucleus::nbr_unknowns_val_domain");
                }
            }

            if (indic)
                res++;
        } while (pos.inc());

        return res;
    }

    int Domain_nucleus::nbr_unknowns(const Tensor& tt, int dom) const
    {

        // Check right domain
        assert(tt.get_space().get_domain(dom) == this);

        int res = 0;
        int val = tt.get_valence();
        switch (val) {
            case 0:
                res += nbr_unknowns_val_domain(tt()(dom), 0, 0);
                break;
            case 1: {
                bool found = false;
                // Cartesian basis
                if (tt.get_basis().get_basis(dom) == CARTESIAN_BASIS) {
                    res += nbr_unknowns_val_domain(tt(1)(dom), 0, 0);
                    res += nbr_unknowns_val_domain(tt(2)(dom), 0, 0);
                    res += nbr_unknowns_val_domain(tt(3)(dom), 0, 0);
                    found = true;
                }
                // Spherical coordinates
                if (tt.get_basis().get_basis(dom) == SPHERICAL_BASIS) {
                    res += nbr_unknowns_val_domain_vr(tt(1)(dom));
                    res += nbr_unknowns_val_domain_vt(tt(2)(dom));
                    res += nbr_unknowns_val_domain_vp(tt(3)(dom));
                    found = true;
                }
                if (!found) {
                    KADATH_THROW("Unknown type of vector Domain_nucleus::nbr_unknowns");
                }
            } break;
            case 2: {
                bool found = false;
                // Cartesian basis and symetric
                if ((tt.get_basis().get_basis(dom) == CARTESIAN_BASIS) && (tt.get_n_comp() == 6)) {
                    res += nbr_unknowns_val_domain(tt(1, 1)(dom), 0, 0);
                    res += nbr_unknowns_val_domain(tt(1, 2)(dom), 0, 0);
                    res += nbr_unknowns_val_domain(tt(1, 3)(dom), 0, 0);
                    res += nbr_unknowns_val_domain(tt(2, 2)(dom), 0, 0);
                    res += nbr_unknowns_val_domain(tt(2, 3)(dom), 0, 0);
                    res += nbr_unknowns_val_domain(tt(3, 3)(dom), 0, 0);
                    found = true;
                }
                // Cartesian basis and not symetric
                if ((tt.get_basis().get_basis(dom) == CARTESIAN_BASIS) && (tt.get_n_comp() == 9)) {
                    for (int i = 1; i <= 3; i++)
                        for (int j = 1; j <= 3; j++)
                            res += nbr_unknowns_val_domain(tt(i, j)(dom), 0, 0);
                    found = true;
                }
                if ((tt.get_basis().get_basis(dom) == SPHERICAL_BASIS) && (tt.get_n_comp() == 6)) {
                    res = 0;
                    found = true;
                }

                if (!found) {
                    KADATH_THROW("Unknown type of 2-tensor Domain_nucleus::nbr_unknowns");
                }
            } break;
            default:
                cerr << "Valence " << val << " not implemented in Domain_nucleus::nbr_unknowns" << endl;
                break;
        }
        return res;
    }
} // namespace Kadath
