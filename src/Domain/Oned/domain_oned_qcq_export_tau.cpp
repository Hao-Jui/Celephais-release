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
#include "For_Kadath/Domain/oned.hpp"
#include "For_Kadath/Array/array.hpp"
#include "For_Kadath/Scalar/scalar.hpp"
#include "For_Kadath/Tensor/tensor.hpp"
namespace Kadath
{
    void Domain_oned_qcq::export_tau_val_domain(const Val_domain& so, int order, Array<double>& sec, int& pos_sec,
                                                int ncond) const
    {

        if (so.check_if_zero())
            pos_sec += ncond;
        else {
            so.coef();
            Index pos_cf(nbr_coefs);
            for (int i = 0; i < nbr_coefs(0) - order; i++) {
                pos_cf.set(0) = i;
                sec.set(pos_sec) = (*so.cf)(pos_cf);
                pos_sec++;
            }
        }
    }

    void Domain_oned_qcq::export_tau(const Tensor& tt, int dom, int order, Array<double>& res, int& pos_res,
                                     const Array<int>& ncond, int, Array<int>**) const
    {
        int val = tt.get_valence();
        switch (val) {
            case 0:
                export_tau_val_domain(tt()(dom), order, res, pos_res, ncond(0));
                break;
            default:
                cerr << "Valence " << val << " not implemented in Domain_oned_qcq::export_tau" << endl;
                break;
        }
    }
} // namespace Kadath
