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
#include "For_Kadath/Domain/spheric_periodic.hpp"
#include "For_Kadath/Term_eq/term_eq.hpp"
#include "For_Kadath/Scalar/scalar.hpp"
#include "For_Kadath/Tensor/tensor.hpp"
#include "For_Kadath/Array/array.hpp"
namespace Kadath
{
    // Val_domain versions
    Val_domain Domain_spheric_periodic_shell::fitschwarz(const Val_domain& so, int dim) const
    {

        so.coef();
        Val_domain dso(der_normal(so, OUTER_BC));

        Val_domain res(this);
        res.base = so.base;
        res.allocate_coef();

        double rmax = alpha + beta;

        Index pos(nbr_coefs);

        // Loop on harmonic
        for (int nn = 0; nn < nbr_coefs(1); nn++) {

            pos.set(1) = nn;

            double sh = (nn == 0) ? 1. / pow(rmax, dim - 2) : 0;
            double dsh = (nn == 0) ? -double(dim - 2) / pow(rmax, dim - 1) : 1;

            // Affecte to res
            for (int i = 0; i < nbr_coefs(0); i++) {
                pos.set(0) = i;
                res.set_coef(pos) = (nn == 0) ? so.get_coef(pos) * dsh - dso.get_coef(pos) * sh : so.get_coef(pos);
            }
        }
        return res;
    }

    // Term_eq version
    Term_eq Domain_spheric_periodic_shell::fitschwarz(const Term_eq& so, int dim) const
    {

        // Check it is a tensor
        if (so.get_type_data() != TERM_T) {
            KADATH_THROW("fitschwarz only defined with respect for a tensor");
        }

        // Right domain ?
        int dom = so.get_dom();
        if (this != so.get_val_t().get_space().get_domain(dom)) {
            KADATH_THROW("Domain mismatch in Domain_spheric_periodic_shell::fitschwarz (Term_eq version)");
        }

        Tensor resval(so.get_val_t(), false);
        for (int i = 0; i < so.get_val_t().get_n_comp(); i++) {
            Array<int> ind(so.get_val_t().indices(i));
            Val_domain value(so.get_val_t()(ind)(dom));
            if (value.check_if_zero())
                resval.set(ind).set_domain(dom).set_zero();
            else {
                resval.set(ind).set_domain(dom) = fitschwarz(value, dim);
            }
        }

        if (so.get_p_der_t() != nullptr) {
            Tensor resder(so.get_val_t(), false);
            for (int i = 0; i < so.get_der_t().get_n_comp(); i++) {
                Array<int> ind(so.get_der_t().indices(i));
                Val_domain valder(so.get_der_t()(ind)(dom));
                if (valder.check_if_zero())
                    resder.set(ind).set_domain(dom).set_zero();
                else {
                    resder.set(ind).set_domain(dom) = fitschwarz(valder, dim);
                }
            }
            return Term_eq(dom, resval, resder);
        } else
            return Term_eq(dom, resval);
    }
} // namespace Kadath
