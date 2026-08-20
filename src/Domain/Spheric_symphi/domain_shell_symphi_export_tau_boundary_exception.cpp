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
#include "For_Kadath/Domain/spheric_symphi.hpp"
#include "For_Kadath/Array/array.hpp"
#include "For_Kadath/Scalar/scalar.hpp"
#include "For_Kadath/Tensor/tensor.hpp"
#include "For_Kadath/Param/param.hpp"

namespace Kadath
{
    void Domain_shell_symphi::export_tau_val_domain_boundary_exception(const Val_domain& so, int mlim, int bound,
                                                                       Array<double>& sec, int& pos_sec, int ncond,
                                                                       const Param& par, int type_exception,
                                                                       const Val_domain& exception) const
    {

        // Place for the exceptionnal part
        int ktarget = par.get_int(0);
        int jtarget = par.get_int(1);
        double value = par.get_double(0);

        assert((type_exception >= 1) && (type_exception <= 4));

        if (so.check_if_zero())
            pos_sec += ncond;

        else {
            so.coef();

            Index pos_cf(nbr_coefs);
            Index pos_galerkin(nbr_coefs);

            int kmin, kmax;
            // Base in phi
            int basep = (*so.get_base().bases_1d[2])(0);
            switch (basep) {
                case COS_EVEN:
                    kmin = 0;
                    kmax = nbr_coefs(2) - 1;
                    break;
                case COS_ODD:
                    kmin = 0;
                    kmax = nbr_coefs(2) - 2;
                    break;
                case SIN_EVEN:
                    kmin = 1;
                    kmax = nbr_coefs(2) - 2;
                    break;
                case SIN_ODD:
                    kmin = 0;
                    kmax = nbr_coefs(2) - 2;
                    break;
                default:
                    KADATH_THROW("Unknow phi basis in Domain_shell_symphi::export_tau_val_domain_exception");
            }

            // Loop on phi :
            for (int k = kmin; k <= kmax; k++) {
                pos_cf.set(2) = k;
                // Loop on theta
                int baset = (*so.get_base().bases_1d[1])(k);
                for (int j = 0; j < nbr_coefs(1); j++) {

                    int mquant;

                    switch (basep) {
                        case COS_EVEN:
                            mquant = 2 * k;
                            break;
                        case COS_ODD:
                            mquant = 2 * k + 1;
                            break;
                        case SIN_EVEN:
                            mquant = 2 * k;
                            break;
                        case SIN_ODD:
                            mquant = 2 * k + 1;
                            break;
                        default:
                            KADATH_THROW("Unknow phi basis in Domain_shell_symphi::export_tau_val_domain_exception");
                    }

                    pos_cf.set(1) = j;
                    switch (baset) {
                        case COS_EVEN:
                            if (mquant == 0) {
                                if ((k == ktarget) && (j == jtarget)) {
                                    switch (type_exception) {
                                        case 1:
                                            sec.set(pos_sec) = val_boundary(bound, exception, pos_cf) - value;
                                            break;
                                        case 2:
                                            sec.set(pos_sec) = 0;
                                            break;
                                        case 3:
                                            sec.set(pos_sec) = val_boundary(bound, exception, pos_cf);
                                            break;
                                        case 4:
                                            sec.set(pos_sec) = 0;
                                            break;
                                        default:
                                            KADATH_THROW("bad value for type_exception");
                                    }
                                } else
                                    sec.set(pos_sec) = val_boundary(bound, so, pos_cf);
                                pos_sec++;
                            } else if (j != 0) {
                                // Galerkin base
                                pos_galerkin = pos_cf;
                                pos_galerkin.set(1) = 0;
                                if ((k == ktarget) && (j == jtarget)) {
                                    switch (type_exception) {
                                        case 1:
                                            sec.set(pos_sec) = val_boundary(bound, exception, pos_cf) - value;
                                            break;
                                        case 2:
                                            sec.set(pos_sec) = 0;
                                            break;
                                        case 3:
                                            sec.set(pos_sec) = val_boundary(bound, exception, pos_cf);
                                            break;
                                        case 4:
                                            sec.set(pos_sec) = 0;
                                            break;
                                        default:
                                            KADATH_THROW("bad value for type_exception");
                                    }
                                } else
                                    sec.set(pos_sec) =
                                        val_boundary(bound, so, pos_cf) - 2. * val_boundary(bound, so, pos_galerkin);
                                pos_sec++;
                            }
                            break;
                        case COS_ODD:
                            if (j != nbr_coefs(1) - 1) {
                                if (mquant == 0) {

                                    if ((k == ktarget) && (j == jtarget)) {
                                        switch (type_exception) {
                                            case 1:
                                                sec.set(pos_sec) = val_boundary(bound, exception, pos_cf) - value;
                                                break;
                                            case 2:
                                                sec.set(pos_sec) = 0;
                                                break;
                                            case 3:
                                                sec.set(pos_sec) = val_boundary(bound, exception, pos_cf);
                                                break;
                                            case 4:
                                                sec.set(pos_sec) = 0;
                                                break;
                                            default:
                                                KADATH_THROW("bad value for type_exception");
                                        }
                                    } else
                                        sec.set(pos_sec) = val_boundary(bound, so, pos_cf);
                                    pos_sec++;
                                } else if (j != 0) {
                                    // Galerkin base
                                    pos_galerkin = pos_cf;
                                    pos_galerkin.set(1) = 0;
                                    if ((k == ktarget) && (j == jtarget)) {
                                        switch (type_exception) {
                                            case 1:
                                                sec.set(pos_sec) = val_boundary(bound, exception, pos_cf) - value;
                                                break;
                                            case 2:
                                                sec.set(pos_sec) = 0;
                                                break;
                                            case 3:
                                                sec.set(pos_sec) = val_boundary(bound, exception, pos_cf);
                                                break;
                                            case 4:
                                                sec.set(pos_sec) = 0;
                                                break;
                                            default:
                                                KADATH_THROW("bad value for type_exception");
                                        }
                                    } else
                                        sec.set(pos_sec) =
                                            val_boundary(bound, so, pos_cf) - val_boundary(bound, so, pos_galerkin);
                                    pos_sec++;
                                }
                            }
                            break;
                        case SIN_EVEN:
                            if ((j != 0) && (j != nbr_coefs(1) - 1)) {
                                if (mquant <= 1) {
                                    if ((k == ktarget) && (j == jtarget)) {
                                        switch (type_exception) {
                                            case 1:
                                                sec.set(pos_sec) = val_boundary(bound, exception, pos_cf) - value;
                                                break;
                                            case 2:
                                                sec.set(pos_sec) = 0;
                                                break;
                                            case 3:
                                                sec.set(pos_sec) = val_boundary(bound, exception, pos_cf);
                                                break;
                                            case 4:
                                                sec.set(pos_sec) = 0;
                                                break;
                                            default:
                                                KADATH_THROW("bad value for type_exception");
                                        }
                                    } else
                                        sec.set(pos_sec) = val_boundary(bound, so, pos_cf);
                                    pos_sec++;
                                } else if (j != 1) {
                                    // Galerkin base
                                    pos_galerkin = pos_cf;
                                    pos_galerkin.set(1) = 1;
                                    if ((k == ktarget) && (j == jtarget)) {
                                        switch (type_exception) {
                                            case 1:
                                                sec.set(pos_sec) = val_boundary(bound, exception, pos_cf) - value;
                                                break;
                                            case 2:
                                                sec.set(pos_sec) = 0;
                                                break;
                                            case 3:
                                                sec.set(pos_sec) = val_boundary(bound, exception, pos_cf);
                                                break;
                                            case 4:
                                                sec.set(pos_sec) = 0;
                                                break;
                                            default:
                                                KADATH_THROW("bad value for type_exception");
                                        }
                                    } else
                                        sec.set(pos_sec) =
                                            val_boundary(bound, so, pos_cf) - j * val_boundary(bound, so, pos_galerkin);
                                    pos_sec++;
                                }
                            }
                            break;
                        case SIN_ODD:
                            if (j != nbr_coefs(1) - 1) {
                                if (mquant <= 1) {
                                    if ((k == ktarget) && (j == jtarget)) {
                                        switch (type_exception) {
                                            case 1:
                                                sec.set(pos_sec) = val_boundary(bound, exception, pos_cf) - value;
                                                break;
                                            case 2:
                                                sec.set(pos_sec) = 0;
                                                break;
                                            case 3:
                                                sec.set(pos_sec) = val_boundary(bound, exception, pos_cf);
                                                break;
                                            case 4:
                                                sec.set(pos_sec) = 0;
                                                break;
                                            default:
                                                KADATH_THROW("bad value for type_exception");
                                        }
                                    } else
                                        sec.set(pos_sec) = val_boundary(bound, so, pos_cf);
                                    pos_sec++;
                                } else if (j != 0) {
                                    // Galerkin base
                                    pos_galerkin = pos_cf;
                                    pos_galerkin.set(1) = 0;
                                    if ((k == ktarget) && (j == jtarget)) {
                                        switch (type_exception) {
                                            case 1:
                                                sec.set(pos_sec) = val_boundary(bound, exception, pos_cf) - value;
                                                break;
                                            case 2:
                                                sec.set(pos_sec) = 0;
                                                break;
                                            case 3:
                                                sec.set(pos_sec) = val_boundary(bound, exception, pos_cf);
                                                break;
                                            case 4:
                                                sec.set(pos_sec) = 0;
                                                break;
                                            default:
                                                KADATH_THROW("bad value for type_exception");
                                        }
                                    } else
                                        sec.set(pos_sec) = val_boundary(bound, so, pos_cf) -
                                                           (2 * j + 1) * val_boundary(bound, so, pos_galerkin);
                                    pos_sec++;
                                }
                            }
                            break;
                        default:
                            std::ostringstream oss;
                            oss
                                << "Unknow theta basis in Domain_shell_symphi::export_tau_val_domain_boundary_exception"
                                << endl;
                            KADATH_THROW(oss.str());
                    }
                }
            }
        }
    }

    void Domain_shell_symphi::export_tau_boundary_exception(const Tensor& tt, int dom, int bound, Array<double>& res,
                                                            int& pos_res, const Array<int>& ncond, const Param& par,
                                                            int type_exception, const Tensor& exception, int n_cmp,
                                                            Array<int>** p_cmp) const
    {
        // Check boundary_exception
        if ((bound != INNER_BC) && (bound != OUTER_BC)) {
            KADATH_THROW("Unknown boundary_exception in Domain_shell_symphi::export_tau_boundary_exception");
        }

        int val = tt.get_valence();
        switch (val) {
            case 0:
                export_tau_val_domain_boundary_exception(tt()(dom), 0, bound, res, pos_res, ncond(0), par,
                                                         type_exception, exception()(dom));
                break;
            default:
                cerr << "Valence " << val << " not implemented in Domain_shell_symphi::export_tau_boundary_exception"
                     << endl;
                break;
        }
    }
} // namespace Kadath
