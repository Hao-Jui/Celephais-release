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
#include "For_Kadath/Domain/spheric_time.hpp"
#include "For_Kadath/Array/array.hpp"
#include "For_Kadath/Scalar/scalar.hpp"
#include "For_Kadath/Tensor/tensor.hpp"
namespace Kadath
{
    void Domain_spheric_time_nucleus::affecte_tau_one_coef_val_domain(Val_domain& so, int cc, int& conte) const
    {

        so.is_zero = false;
        so.allocate_coef();
        *so.cf = 0.;
        Index pos_cf(nbr_coefs);

        bool found = false;
        for (int j = 0; j < nbr_coefs(1); j++) {
            pos_cf.set(1) = j;
            for (int i = 0; i < nbr_coefs(0); i++) {
                pos_cf.set(0) = i;
                if (conte == cc) {
                    so.cf->set(pos_cf) = 1;
                    found = true;
                } else {
                    so.cf->set(pos_cf) = 0.;
                }
                conte++;
            }
        }
        // If not found put to zero :
        if (!found)
            so.set_zero();
    }

    void Domain_spheric_time_nucleus::affecte_tau_one_coef(Tensor& tt, int dom, int cc, int& pos_cf) const
    {

        // Check right domain
        assert(tt.get_space().get_domain(dom) == this);

        int val = tt.get_valence();
        switch (val) {
            case 0:
                affecte_tau_one_coef_val_domain(tt.set().set_domain(dom), cc, pos_cf);
                break;
            default:
                cerr << "Valence " << val << " not implemented in Domain_spheric_time_nucleus::affecte_tau" << endl;
                break;
        }
    }
} // namespace Kadath
