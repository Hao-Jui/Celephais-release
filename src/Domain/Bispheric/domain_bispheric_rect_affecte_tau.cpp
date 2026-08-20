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

#include "For_Kadath/Domain/bispheric.hpp"
#include "For_Kadath/Scalar/scalar.hpp"
#include "For_Kadath/Tensor/tensor.hpp"
namespace Kadath
{
    void Domain_bispheric_rect::affecte_tau_val_domain(Val_domain& so, const Array<double>& values, int& conte) const
    {

        int basep = (*so.get_base().bases_1d[2])(0);

        so.allocate_coef();
        *so.cf = 0.;
        Index pos(nbr_coefs);
        do {

            bool indic = true;
            switch (basep) {
                case COS:
                    // Last odd ones
                    if ((pos(2) % 2 == 1) && (pos(1) == nbr_coefs(1) - 1))
                        indic = false;
                    // Regularity for even ones :
                    if ((pos(2) != 0) && (pos(2) % 2 == 0) && (pos(1) == 0))
                        indic = false;
                    break;
                case SIN:
                    // sin(0)
                    if ((pos(2) == 0) || (pos(2) == nbr_coefs(2) - 1))
                        indic = false;
                    // Last odd ones :
                    if ((pos(2) % 2 == 1) && (pos(1) == nbr_coefs(1) - 1))
                        indic = false;
                    // Regularity for even ones :
                    if ((pos(2) % 2 == 0) && (pos(1) == 0))
                        indic = false;
                    break;
                default:
                    KADATH_THROW("Unknown phi basis in Domain_bispheric_rect::affecte_tau_val_domain");
            }

            if (indic) {
                so.cf->set(pos) = values(conte);
                conte++;
            }
        } while (pos.inc());

        // Regularity on the axis :
        // Loop on phi :
        for (int k = 1; k < nbr_coefs(2); k++) {
            pos.set(2) = k;
            int basechi = (*so.get_base().bases_1d[1])(k);
            if ((basechi == CHEB_EVEN) || (basechi == LEG_EVEN)) {
                // Regularity :
                for (int i = 0; i < nbr_coefs(0); i++) {
                    pos.set(0) = i;
                    double summ = 0;
                    double val = 1;
                    for (int j = 1; j < nbr_coefs(1); j++) {
                        pos.set(1) = j;
                        switch (basechi) {
                            case CHEB_EVEN:
                                val *= -1.;
                                summ += val * (*so.cf)(pos);
                                break;
                            case LEG_EVEN:
                                val *= -double(2 * j - 1) / double(2 * j);
                                summ += val * (*so.cf)(pos);
                                break;
                            default:
                                KADATH_THROW("Unknown base in Domain_bispheric_rect::affecte_tau");
                        }
                    }
                    pos.set(1) = 0;
                    so.cf->set(pos) = -summ;
                }
            }
        }
    }

    void Domain_bispheric_rect::affecte_tau(Tensor& tt, int dom, const Array<double>& cf, int& pos_cf) const
    {

        // Check right domain
        assert(tt.get_space().get_domain(dom) == this);

        int val = tt.get_valence();
        switch (val) {
            case 0:
                affecte_tau_val_domain(tt.set().set_domain(dom), cf, pos_cf);
                break;
            case 1: {
                bool found = false;
                // Cartesian basis
                if (tt.get_basis().get_basis(dom) == CARTESIAN_BASIS) {
                    affecte_tau_val_domain(tt.set(1).set_domain(dom), cf, pos_cf);
                    affecte_tau_val_domain(tt.set(2).set_domain(dom), cf, pos_cf);
                    affecte_tau_val_domain(tt.set(3).set_domain(dom), cf, pos_cf);
                    found = true;
                }
                if (!found) {
                    KADATH_THROW("Unknown type of vector Domain_bispheric_rect::affecte_tau");
                }
            } break;
            case 2: {
                bool found = false;
                // Cartesian basis and symetric
                if ((tt.get_basis().get_basis(dom) == CARTESIAN_BASIS) && (tt.get_n_comp() == 6)) {
                    affecte_tau_val_domain(tt.set(1, 1).set_domain(dom), cf, pos_cf);
                    affecte_tau_val_domain(tt.set(1, 2).set_domain(dom), cf, pos_cf);
                    affecte_tau_val_domain(tt.set(1, 3).set_domain(dom), cf, pos_cf);
                    affecte_tau_val_domain(tt.set(2, 2).set_domain(dom), cf, pos_cf);
                    affecte_tau_val_domain(tt.set(2, 3).set_domain(dom), cf, pos_cf);
                    affecte_tau_val_domain(tt.set(3, 3).set_domain(dom), cf, pos_cf);
                    found = true;
                }
                // Cartesian basis and not symetric
                if ((tt.get_basis().get_basis(dom) == CARTESIAN_BASIS) && (tt.get_n_comp() == 9)) {
                    affecte_tau_val_domain(tt.set(1, 1).set_domain(dom), cf, pos_cf);
                    affecte_tau_val_domain(tt.set(1, 2).set_domain(dom), cf, pos_cf);
                    affecte_tau_val_domain(tt.set(1, 3).set_domain(dom), cf, pos_cf);
                    affecte_tau_val_domain(tt.set(2, 1).set_domain(dom), cf, pos_cf);
                    affecte_tau_val_domain(tt.set(2, 2).set_domain(dom), cf, pos_cf);
                    affecte_tau_val_domain(tt.set(2, 3).set_domain(dom), cf, pos_cf);
                    affecte_tau_val_domain(tt.set(3, 1).set_domain(dom), cf, pos_cf);
                    affecte_tau_val_domain(tt.set(3, 2).set_domain(dom), cf, pos_cf);
                    affecte_tau_val_domain(tt.set(3, 3).set_domain(dom), cf, pos_cf);
                    found = true;
                }
                if (!found) {
                    KADATH_THROW("Unknown type of 2-tensor Domain_bispheric_rect::affecte_tau");
                }
            } break;
            default:
                cerr << "Valence " << val << " not implemented in Domain_bispheric_rect::affecte_tau" << endl;
                break;
        }
    }
} // namespace Kadath
