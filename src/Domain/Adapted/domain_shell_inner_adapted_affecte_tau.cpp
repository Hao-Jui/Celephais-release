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
#include "For_Kadath/Domain/adapted.hpp"
#include "For_Kadath/Array/point.hpp"
#include "For_Kadath/Array/array.hpp"
#include "For_Kadath/Scalar/scalar.hpp"
#include "For_Kadath/Tensor/tensor.hpp"

namespace Kadath
{
    void Domain_shell_inner_adapted::affecte_tau_val_domain(Val_domain& so, int mlim, const Array<double>& values,
                                                            int& conte) const
    {

        int kmin = 2 * mlim + 2;

        so.allocate_coef();
        *so.cf = 0.;
        Index pos_cf(nbr_coefs);

        // True values
        // Loop on phi :
        for (int k = 0; k < nbr_coefs(2) - 1; k++)
            if (k != 1) {
                pos_cf.set(2) = k;
                // Loop on theta
                int baset = (*so.get_base().bases_1d[1])(k);
                for (int j = 0; j < nbr_coefs(1); j++) {
                    pos_cf.set(1) = j;
                    bool true_tet = true;
                    switch (baset) {
                        case COS_EVEN:
                            if ((j == 0) && (k >= kmin))
                                true_tet = false;
                            break;
                        case COS_ODD:
                            if ((j == nbr_coefs(1) - 1) || ((j == 0) && (k >= kmin)))
                                true_tet = false;
                            break;
                        case SIN_EVEN:
                            if (((j == 1) && (k >= kmin + 2)) || (j == 0) || (j == nbr_coefs(1) - 1))
                                true_tet = false;
                            break;
                        case SIN_ODD:
                            if (((j == 0) && (k > kmin + 2)) || (j == nbr_coefs(1) - 1))
                                true_tet = false;
                            break;
                        default:
                            KADATH_THROW("Unknow theta basis in Domain_shell_inner_adapted::affecte_tau_val_domain");
                    }

                    if (true_tet)
                        for (int i = 0; i < nbr_coefs(0); i++) {
                            pos_cf.set(0) = i;
                            so.cf->set(pos_cf) += values(conte);
                            conte++;
                        }
                }
            }

        // Appropriate regularisation
        // Loop on phi :
        for (int k = 0; k < nbr_coefs(2) - 1; k++) {
            pos_cf.set(2) = k;
            int baset = (*so.get_base().bases_1d[1])(k);
            // Loop on r :
            for (int i = 0; i < nbr_coefs(0); i++) {
                pos_cf.set(0) = i;
                double sum = 0.;
                switch (baset) {
                    case COS_EVEN:
                        if (k >= kmin) {
                            for (int j = 1; j < nbr_coefs(1); j++) {
                                pos_cf.set(1) = j;
                                sum += (*so.cf)(pos_cf);
                            }
                            pos_cf.set(1) = 0;
                            so.cf->set(pos_cf) = -sum;
                        }
                        break;
                    case COS_ODD:
                        if (k >= kmin) {
                            for (int j = 1; j < nbr_coefs(1); j++) {
                                pos_cf.set(1) = j;
                                sum += (*so.cf)(pos_cf);
                            }
                            pos_cf.set(1) = 0;
                            so.cf->set(pos_cf) = -sum;
                        }
                        break;
                    case SIN_EVEN:
                        if (k >= kmin + 2) {
                            for (int j = 2; j < nbr_coefs(1); j++) {
                                pos_cf.set(1) = j;
                                sum += j * (*so.cf)(pos_cf);
                            }
                            pos_cf.set(1) = 1;
                            so.cf->set(pos_cf) = -sum;
                        }
                        break;
                    case SIN_ODD:
                        if (k >= kmin + 2) {
                            for (int j = 1; j < nbr_coefs(1); j++) {
                                pos_cf.set(1) = j;
                                sum += (2 * j + 1) * (*so.cf)(pos_cf);
                            }
                            pos_cf.set(1) = 0;
                            so.cf->set(pos_cf) = -sum;
                        }
                        break;
                    default:
                        KADATH_THROW("Unknow theta basis in Domain_shell_inner_adapted::affecte_tau_val_domain");
                }
            }
        }
    }

    void Domain_shell_inner_adapted::affecte_tau(Tensor& tt, int dom, const Array<double>& cf, int& pos_cf) const
    {

        // Check right domain
        assert(tt.get_space().get_domain(dom) == this);

        int val = tt.get_valence();
        switch (val) {
            case 0:
                affecte_tau_val_domain(tt.set().set_domain(dom), 0, cf, pos_cf);
                break;
            case 1: {
                bool found = false;
                // Cartesian basis
                if (tt.get_basis().get_basis(dom) == CARTESIAN_BASIS) {
                    affecte_tau_val_domain(tt.set(1).set_domain(dom), 0, cf, pos_cf);
                    affecte_tau_val_domain(tt.set(2).set_domain(dom), 0, cf, pos_cf);
                    affecte_tau_val_domain(tt.set(3).set_domain(dom), 0, cf, pos_cf);
                    found = true;
                }
                // Spherical coordinates
                if (tt.get_basis().get_basis(dom) == SPHERICAL_BASIS) {
                    affecte_tau_val_domain(tt.set(1).set_domain(dom), 0, cf, pos_cf);
                    affecte_tau_val_domain(tt.set(2).set_domain(dom), 1, cf, pos_cf);
                    affecte_tau_val_domain(tt.set(3).set_domain(dom), 1, cf, pos_cf);
                    found = true;
                }
                if (!found) {
                    KADATH_THROW("Unknown type of vector Domain_shell_inner_adapted::affecte_tau");
                }
            } break;
            case 2: {
                bool found = false;
                // Cartesian basis and symetric
                if ((tt.get_basis().get_basis(dom) == CARTESIAN_BASIS) && (tt.get_n_comp() == 6)) {
                    affecte_tau_val_domain(tt.set(1, 1).set_domain(dom), 0, cf, pos_cf);
                    affecte_tau_val_domain(tt.set(1, 2).set_domain(dom), 0, cf, pos_cf);
                    affecte_tau_val_domain(tt.set(1, 3).set_domain(dom), 0, cf, pos_cf);
                    affecte_tau_val_domain(tt.set(2, 2).set_domain(dom), 0, cf, pos_cf);
                    affecte_tau_val_domain(tt.set(2, 3).set_domain(dom), 0, cf, pos_cf);
                    affecte_tau_val_domain(tt.set(3, 3).set_domain(dom), 0, cf, pos_cf);
                    found = true;
                }
                // Cartesian basis and not symetric
                if ((tt.get_basis().get_basis(dom) == CARTESIAN_BASIS) && (tt.get_n_comp() == 9)) {
                    affecte_tau_val_domain(tt.set(1, 1).set_domain(dom), 0, cf, pos_cf);
                    affecte_tau_val_domain(tt.set(1, 2).set_domain(dom), 0, cf, pos_cf);
                    affecte_tau_val_domain(tt.set(1, 3).set_domain(dom), 0, cf, pos_cf);
                    affecte_tau_val_domain(tt.set(2, 1).set_domain(dom), 0, cf, pos_cf);
                    affecte_tau_val_domain(tt.set(2, 2).set_domain(dom), 0, cf, pos_cf);
                    affecte_tau_val_domain(tt.set(2, 3).set_domain(dom), 0, cf, pos_cf);
                    affecte_tau_val_domain(tt.set(3, 1).set_domain(dom), 0, cf, pos_cf);
                    affecte_tau_val_domain(tt.set(3, 2).set_domain(dom), 0, cf, pos_cf);
                    affecte_tau_val_domain(tt.set(3, 3).set_domain(dom), 0, cf, pos_cf);
                    found = true;
                }
                // Spherical coordinates and symetric
                if ((tt.get_basis().get_basis(dom) == SPHERICAL_BASIS) && (tt.get_n_comp() == 6)) {
                    affecte_tau_val_domain(tt.set(1, 1).set_domain(dom), 0, cf, pos_cf);
                    affecte_tau_val_domain(tt.set(1, 2).set_domain(dom), 1, cf, pos_cf);
                    affecte_tau_val_domain(tt.set(1, 3).set_domain(dom), 1, cf, pos_cf);
                    affecte_tau_val_domain(tt.set(2, 2).set_domain(dom), 2, cf, pos_cf);
                    affecte_tau_val_domain(tt.set(2, 3).set_domain(dom), 2, cf, pos_cf);
                    affecte_tau_val_domain(tt.set(3, 3).set_domain(dom), 2, cf, pos_cf);
                    found = true;
                }
                // Spherical coordinates and not symetric
                if ((tt.get_basis().get_basis(dom) == SPHERICAL_BASIS) && (tt.get_n_comp() == 9)) {
                    affecte_tau_val_domain(tt.set(1, 1).set_domain(dom), 0, cf, pos_cf);
                    affecte_tau_val_domain(tt.set(1, 2).set_domain(dom), 1, cf, pos_cf);
                    affecte_tau_val_domain(tt.set(1, 3).set_domain(dom), 1, cf, pos_cf);
                    affecte_tau_val_domain(tt.set(2, 1).set_domain(dom), 1, cf, pos_cf);
                    affecte_tau_val_domain(tt.set(2, 2).set_domain(dom), 2, cf, pos_cf);
                    affecte_tau_val_domain(tt.set(2, 3).set_domain(dom), 2, cf, pos_cf);
                    affecte_tau_val_domain(tt.set(3, 1).set_domain(dom), 1, cf, pos_cf);
                    affecte_tau_val_domain(tt.set(3, 2).set_domain(dom), 2, cf, pos_cf);
                    affecte_tau_val_domain(tt.set(3, 3).set_domain(dom), 2, cf, pos_cf);
                    found = true;
                }
                if (!found) {
                    KADATH_THROW("Unknown type of 2-tensor Domain_shell_inner_adapted::affecte_tau");
                }
            } break;
            default:
                cerr << "Valence " << val << " not implemented in Domain_shell_inner_adapted::affecte_tau" << endl;
                break;
        }
    }
} // namespace Kadath
