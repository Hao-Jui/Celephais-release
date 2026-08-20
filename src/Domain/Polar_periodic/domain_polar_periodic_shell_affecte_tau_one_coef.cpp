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

#include <sstream>
#include "For_Kadath/Domain/polar_periodic.hpp"
#include "For_Kadath/Array/point.hpp"
#include "For_Kadath/Array/array.hpp"
#include "For_Kadath/Scalar/scalar.hpp"
#include "For_Kadath/Tensor/tensor.hpp"
namespace Kadath
{
    void Domain_polar_periodic_shell::affecte_tau_one_coef_val_domain(Val_domain& so, int cc, int& conte) const
    {

        so.is_zero = false;
        so.allocate_coef();
        *so.cf = 0.;
        Index pos_cf(nbr_coefs);

        bool found = false;

        int basetime = (*so.get_base().bases_1d[2])(0);
        int mink, maxk;
        switch (basetime) {
            case COS:
                mink = 0;
                maxk = nbr_coefs(2);
                break;
            case SIN:
                mink = 1;
                maxk = nbr_coefs(2) - 1;
                break;
            default:
                KADATH_THROW("Unknown basis in Domain_polar_periodic_shell_affecte_tau_val_domain");
        }
        // Loop on time
        for (int k = mink; k < maxk; k++) {

            pos_cf.set(2) = k;
            // Loop on theta
            int baset = (*so.get_base().bases_1d[1])(k);
            int minj, maxj;
            switch (baset) {
                case COS_EVEN:
                    minj = 0;
                    maxj = nbr_coefs(1);
                    break;
                case COS_ODD:
                    minj = 0;
                    maxj = nbr_coefs(1) - 1;
                    break;
                case SIN_EVEN:
                    minj = 1;
                    maxj = nbr_coefs(1) - 1;
                    break;
                case SIN_ODD:
                    minj = 0;
                    maxj = nbr_coefs(1) - 1;
                    break;

                default:
                    KADATH_THROW("Unknown theta basis in Domain_polar_periodic_shell_affecte_tau_val_domain");
            }

            for (int j = minj; j < maxj; j++) {
                pos_cf.set(1) = j;
                // Loop on r :
                for (int i = 0; i < nbr_coefs(0); i++) {
                    pos_cf.set(0) = i;
                    // No garlekin
                    if (conte == cc) {
                        found = true;
                        so.cf->set(pos_cf) = 1.;
                    }
                    conte++;
                } // end loop i
            } // end loop j
        } // end loop k
        // If not found put to zero :
        if (!found)
            so.set_zero();
    }

    void Domain_polar_periodic_shell::affecte_tau_one_coef(Tensor& tt, int dom, int cc, int& pos_cf) const
    {

        // Check right domain
        assert(tt.get_space().get_domain(dom) == this);

        int val = tt.get_valence();
        switch (val) {
            case 0:
                affecte_tau_one_coef_val_domain(tt.set().set_domain(dom), cc, pos_cf);
                break;
            case 1:
                affecte_tau_one_coef_val_domain(tt.set(1).set_domain(dom), cc, pos_cf);
                affecte_tau_one_coef_val_domain(tt.set(2).set_domain(dom), cc, pos_cf);
                affecte_tau_one_coef_val_domain(tt.set(3).set_domain(dom), cc, pos_cf);
                break;
            case 2:
                // symetric
                if (tt.get_n_comp() == 6) {
                    affecte_tau_one_coef_val_domain(tt.set(1, 1).set_domain(dom), cc, pos_cf);
                    affecte_tau_one_coef_val_domain(tt.set(1, 2).set_domain(dom), cc, pos_cf);
                    affecte_tau_one_coef_val_domain(tt.set(1, 3).set_domain(dom), cc, pos_cf);
                    affecte_tau_one_coef_val_domain(tt.set(2, 2).set_domain(dom), cc, pos_cf);
                    affecte_tau_one_coef_val_domain(tt.set(2, 3).set_domain(dom), cc, pos_cf);
                    affecte_tau_one_coef_val_domain(tt.set(3, 3).set_domain(dom), cc, pos_cf);
                }
                //  not symetric
                if (tt.get_n_comp() == 9) {
                    affecte_tau_one_coef_val_domain(tt.set(1, 1).set_domain(dom), cc, pos_cf);
                    affecte_tau_one_coef_val_domain(tt.set(1, 2).set_domain(dom), cc, pos_cf);
                    affecte_tau_one_coef_val_domain(tt.set(1, 3).set_domain(dom), cc, pos_cf);
                    affecte_tau_one_coef_val_domain(tt.set(2, 1).set_domain(dom), cc, pos_cf);
                    affecte_tau_one_coef_val_domain(tt.set(2, 2).set_domain(dom), cc, pos_cf);
                    affecte_tau_one_coef_val_domain(tt.set(2, 3).set_domain(dom), cc, pos_cf);
                    affecte_tau_one_coef_val_domain(tt.set(3, 1).set_domain(dom), cc, pos_cf);
                    affecte_tau_one_coef_val_domain(tt.set(3, 2).set_domain(dom), cc, pos_cf);
                    affecte_tau_one_coef_val_domain(tt.set(3, 3).set_domain(dom), cc, pos_cf);
                }
                break;
            default:
                std::ostringstream oss;
                oss << "Valence " << val << " not implemented in Domain_polar_shell::affecte_tau" << endl;
                KADATH_THROW(oss.str());
                break;
        }
    }
} // namespace Kadath
