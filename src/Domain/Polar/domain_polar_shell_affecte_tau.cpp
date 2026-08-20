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
#include "For_Kadath/Domain/polar.hpp"
#include "For_Kadath/Array/point.hpp"
#include "For_Kadath/Array/array.hpp"
#include "For_Kadath/Scalar/scalar.hpp"
#include "For_Kadath/Tensor/tensor.hpp"
namespace Kadath
{
    void Domain_polar_shell::affecte_tau_val_domain(Val_domain& so, int mquant, const Array<double>& values,
                                                    int& conte) const
    {

        so.allocate_coef();
        *so.cf = 0.;

        Index pos_cf(nbr_coefs);
        // True values
        // Loop on theta
        int baset = (*so.get_base().bases_1d[1])(0);
        for (int j = 0; j < nbr_coefs(1); j++) {
            pos_cf.set(1) = j;
            bool true_tet = true;
            switch (baset) {
                case COS_EVEN:
                    if ((j == 0) && (mquant != 0))
                        true_tet = false;
                    break;
                case COS_ODD:
                    if ((j == nbr_coefs(1) - 1) || ((j == 0) && (mquant != 0)))
                        true_tet = false;
                    break;
                case SIN_EVEN:
                    if (((j == 1) && (mquant > 1)) || (j == 0) || (j == nbr_coefs(1) - 1))
                        true_tet = false;
                    break;
                case SIN_ODD:
                    if (((j == 0) && (mquant > 1)) || (j == nbr_coefs(1) - 1))
                        true_tet = false;
                    break;
                default:
                    KADATH_THROW("Unknow theta basis in Domain_polar_shell::affecte_tau_val_domain");
            }

            if (true_tet)
                for (int i = 0; i < nbr_coefs(0); i++) {
                    pos_cf.set(0) = i;
                    so.cf->set(pos_cf) += values(conte);
                    conte++;
                }
        }

        // Regularity on the axis :
        if (mquant > 1) {
            // Loop on r :
            for (int i = 0; i < nbr_coefs(0); i++) {
                pos_cf.set(0) = i;
                double sum = 0.;
                switch (baset) {
                    case COS_EVEN:
                        for (int j = 1; j < nbr_coefs(1); j++) {
                            pos_cf.set(1) = j;
                            sum += (*so.cf)(pos_cf);
                        }
                        pos_cf.set(1) = 0;
                        so.cf->set(pos_cf) = -sum;
                        break;
                    case COS_ODD:
                        for (int j = 1; j < nbr_coefs(1); j++) {
                            pos_cf.set(1) = j;
                            sum += (*so.cf)(pos_cf);
                        }
                        pos_cf.set(1) = 0;
                        so.cf->set(pos_cf) = -sum;
                        break;
                    case SIN_EVEN:
                        for (int j = 2; j < nbr_coefs(1); j++) {
                            pos_cf.set(1) = j;
                            sum += j * (*so.cf)(pos_cf);
                        }
                        pos_cf.set(1) = 1;
                        so.cf->set(pos_cf) = -sum;
                        break;
                    case SIN_ODD:
                        for (int j = 1; j < nbr_coefs(1); j++) {
                            pos_cf.set(1) = j;
                            sum += (2 * j + 1) * (*so.cf)(pos_cf);
                        }
                        pos_cf.set(1) = 0;
                        so.cf->set(pos_cf) = -sum;
                        break;
                    default:
                        KADATH_THROW("Unknow theta basis in Domain_polar_shell::affecte_tau_val_domain");
                }
            }
        }
    }

    void Domain_polar_shell::affecte_tau(Tensor& tt, int dom, const Array<double>& cf, int& pos_cf) const
    {

        // Check right domain
        assert(tt.get_space().get_domain(dom) == this);

        int val = tt.get_valence();
        switch (val) {
            case 0:
                if (!tt.is_m_quant_affected())
                    affecte_tau_val_domain(tt.set().set_domain(dom), 0, cf, pos_cf);
                else
                    affecte_tau_val_domain(tt.set().set_domain(dom), tt.get_parameters()->get_m_quant(), cf, pos_cf);
                break;
            default:
                cerr << "Valence " << val << " not implemented in Domain_polar_shell::affecte_tau" << endl;
                break;
        }
    }
} // namespace Kadath
