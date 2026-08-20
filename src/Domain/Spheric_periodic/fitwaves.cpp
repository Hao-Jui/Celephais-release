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
    void coef_1d(int, Array<double>&);

    double Domain_spheric_periodic_shell::give_harmonique(const Val_domain& so, int nn, double phase, int dim,
                                                          double factor) const
    {
        so.coef();
        Index pos(nbr_coefs);
        pos.set(1) = nn;
        int expo = 0;
        switch ((*so.base.bases_1d[1])(0)) {
            case COS:
                expo = nn;
                break;
            case COS_EVEN:
                expo = 2 * nn;
                break;
            case COS_ODD:
                expo = 2 * nn + 1;
                break;
            default:
                cerr << "Unknown case in Domain_spheric_periodic_shell::fitwaves" << endl;
                break;
        }

        // Get the value at rmax of this harmonic
        double AA = val_boundary(OUTER_BC, so, pos);
        double lambda2 = factor - expo * expo * ome * ome;
        double res;
        double rmax = alpha + beta;

        if (lambda2 > 0)
            res = AA / exp(-sqrt(lambda2) * rmax) * pow(rmax, double(dim - 1) / 2.);
        else
            res = AA / sin(sqrt(-lambda2) * rmax + phase) * pow(rmax, double(dim - 1) / 2.);
        return res;
    }

    // Val_domain version
    Val_domain Domain_spheric_periodic_shell::fitwaves(const Val_domain& so, const Array<double>& phases, int dim,
                                                       double factor) const
    {

        so.coef();

        Val_domain dso(der_normal(so, OUTER_BC));

        Val_domain res(this);
        res.base = so.base;
        res.allocate_coef();

        double rmax = alpha + beta;

        Index pos(nbr_coefs);

        int max = 0;
        switch ((*so.base.bases_1d[1])(0)) {
            case COS:
                max = nbr_coefs(1);
                break;
            case COS_EVEN:
                max = nbr_coefs(1);
                break;
            case COS_ODD:
                max = nbr_coefs(1) - 1;
                break;
            default:
                cerr << "Unknown case in Domain_spheric_periodic_shell::fitwaves" << endl;
                break;
        }

        // Loop on harmonic
        for (int nn = 0; nn < max; nn++) {

            pos.set(1) = nn;

            // Correspond to what ?
            int expo = 0;
            switch ((*so.base.bases_1d[1])(0)) {
                case COS:
                    expo = nn;
                    break;
                case COS_EVEN:
                    expo = 2 * nn;
                    break;
                case COS_ODD:
                    expo = 2 * nn + 1;
                    break;
                default:
                    cerr << "Unknown case in Domain_spheric_periodic_shell::fitwaves" << endl;
                    break;
            }

            double lambda2 = factor - expo * expo * ome * ome;
            // Various cases :
            double sh = (lambda2 > 0) ? exp(-sqrt(lambda2) * rmax) / pow(rmax, double(dim - 1) / 2.)
                                      : sin(sqrt(-lambda2) * rmax + phases(nn)) / pow(rmax, double(dim - 1) / 2.);

            double dsh = (lambda2 > 0) ? exp(-sqrt(lambda2) * rmax) / pow(rmax, double(dim - 1) / 2.) *
                                             (-sqrt(lambda2) - double(dim - 1) / 2. / rmax)
                                       : sqrt(-lambda2) * cos(sqrt(-lambda2) * rmax + phases(nn)) /
                                                 pow(rmax, double(dim - 1) / 2.) -
                                             double(dim - 1) / 2. * sin(sqrt(-lambda2) * rmax + phases(nn)) /
                                                 pow(rmax, double(dim + 1) / 2.);

            // Affecte to res
            for (int i = 0; i < nbr_coefs(0); i++) {
                pos.set(0) = i;
                res.set_coef(pos) = rmax * rmax * (so.get_coef(pos) * dsh - dso.get_coef(pos) * sh);
            }
        }
        return res;
    }

    // Term_eq version
    Term_eq Domain_spheric_periodic_shell::fitwaves(const Term_eq& so, const Array<double>& phases, int dim,
                                                    double factor) const
    {

        // Check it is a tensor
        if (so.get_type_data() != TERM_T) {
            KADATH_THROW("fitwaves only defined with respect for a tensor");
        }

        // Right domain ?
        int dom = so.get_dom();
        if (this != so.get_val_t().get_space().get_domain(dom)) {
            KADATH_THROW("Domain mismatch in Domain_spheric_periodic_shell::fitwaves (Term_eq version)");
        }

        Tensor resval(so.get_val_t(), false);
        for (int i = 0; i < so.get_val_t().get_n_comp(); i++) {
            Array<int> ind(so.get_val_t().indices(i));
            Val_domain value(so.get_val_t()(ind)(dom));
            if (value.check_if_zero())
                resval.set(ind).set_domain(dom).set_zero();
            else {
                resval.set(ind).set_domain(dom) = fitwaves(value, phases, dim, factor);
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
                    resder.set(ind).set_domain(dom) = fitwaves(valder, phases, dim, factor);
                }
            }
            return Term_eq(dom, resval, resder);
        } else
            return Term_eq(dom, resval);
    }

} // namespace Kadath
