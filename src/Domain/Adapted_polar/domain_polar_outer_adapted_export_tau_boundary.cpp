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
#include "For_Kadath/Domain/adapted_polar.hpp"
#include "For_Kadath/Array/point.hpp"
#include "For_Kadath/Array/array.hpp"
#include "For_Kadath/Scalar/scalar.hpp"
#include "For_Kadath/Tensor/tensor.hpp"

namespace Kadath
{
    void Domain_polar_shell_outer_adapted::export_tau_val_domain_boundary(const Val_domain& so, int mquant, int bound,
                                                                          Array<double>& sec, int& pos_sec,
                                                                          int ncond) const
    {

        if (so.check_if_zero())
            pos_sec += ncond;
        else {
            so.coef();
            Index pos_cf(nbr_coefs);
            Index pos_galerkin(nbr_coefs);

            // Loop on theta
            int baset = (*so.get_base().bases_1d[1])(0);
            for (int j = 0; j < nbr_coefs(1); j++) {
                pos_cf.set(1) = j;
                switch (baset) {
                    case COS_EVEN:
                        if (mquant == 0) {
                            sec.set(pos_sec) = val_boundary(bound, so, pos_cf);
                            pos_sec++;
                        } else if (j != 0) {
                            // Galerkin base
                            pos_galerkin = pos_cf;
                            pos_galerkin.set(1) = 0;
                            sec.set(pos_sec) =
                                val_boundary(bound, so, pos_cf) - 2. * val_boundary(bound, so, pos_galerkin);
                            pos_sec++;
                        }
                        break;
                    case COS_ODD:
                        if (j != nbr_coefs(1) - 1) {
                            if (mquant == 0) {
                                sec.set(pos_sec) = val_boundary(bound, so, pos_cf);
                                pos_sec++;
                            } else if (j != 0) {
                                // Galerkin base
                                pos_galerkin = pos_cf;
                                pos_galerkin.set(1) = 0;
                                sec.set(pos_sec) =
                                    val_boundary(bound, so, pos_cf) - val_boundary(bound, so, pos_galerkin);
                                pos_sec++;
                            }
                        }
                        break;
                    case SIN_EVEN:
                        if ((j != 0) && (j != nbr_coefs(1) - 1)) {
                            if (mquant <= 1) {
                                sec.set(pos_sec) = val_boundary(bound, so, pos_cf);
                                pos_sec++;
                            } else if (j != 1) {
                                pos_galerkin = pos_cf;
                                pos_galerkin.set(1) = 1;
                                sec.set(pos_sec) =
                                    val_boundary(bound, so, pos_cf) - j * val_boundary(bound, so, pos_galerkin);
                                pos_sec++;
                            }
                        }
                        break;
                    case SIN_ODD:
                        if (j != nbr_coefs(1) - 1) {
                            if (mquant <= 1) {
                                sec.set(pos_sec) = val_boundary(bound, so, pos_cf);
                                pos_sec++;
                            } else if (j != 0) {
                                pos_galerkin = pos_cf;
                                pos_galerkin.set(1) = 0;
                                sec.set(pos_sec) = val_boundary(bound, so, pos_cf) -
                                                   (2 * j + 1) * val_boundary(bound, so, pos_galerkin);
                                pos_sec++;
                            }
                        }

                        break;
                    default:
                        KADATH_THROW("Unknow theta basis in Domain_polar_shell_outer_adapted::export_tau_val_domain_boundary");
                }
            }
        }
    }

    void Domain_polar_shell_outer_adapted::export_tau_boundary(const Tensor& tt, int dom, int bound, Array<double>& res,
                                                               int& pos_res, const Array<int>& ncond, int,
                                                               Array<int>**) const
    {

        // Check boundary
        if ((bound != INNER_BC) && (bound != OUTER_BC)) {
            KADATH_THROW("Unknown boundary in Domain_polar_shell_outer_adapted::export_tau_boundary");
        }

        int val = tt.get_valence();
        switch (val) {
            case 0:
                if (!tt.is_m_quant_affected())
                    export_tau_val_domain_boundary(tt()(dom), 0, bound, res, pos_res, ncond(0));
                else
                    export_tau_val_domain_boundary(tt()(dom), tt.get_parameters()->get_m_quant(), bound, res, pos_res,
                                                   ncond(0));
                break;
            default:
                cerr << "Valence " << val << " not implemented in Domain_polar_shell_outer_adapted::export_tau_boundary"
                     << endl;
                break;
        }
    }
} // namespace Kadath
