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

    double Domain_spheric_periodic_shell::give_harmonique_nonflat(const Val_domain& so, double masse, int nn,
                                                                  double phase, int dim, double factor) const
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
        double cor = 2 * expo * expo * ome * ome - 1;
        double res;
        double rmax = alpha + beta;

        if (lambda2 > 0)
            res = AA / exp(-sqrt(lambda2) * rmax) * pow(rmax, double(dim - 1) / 2.);
        else
            res = AA / cos(sqrt(-lambda2) * rmax + masse / sqrt(-lambda2) * cor * log(rmax) + phase) *
                  pow(rmax, double(dim - 1) / 2.);
        return res;
    }

    // Term_eq version
    Term_eq Domain_spheric_periodic_shell::fitwaves_nonflat(const Term_eq& so, const Term_eq& field,
                                                            const Array<double>& phases, int dim, double factor) const
    {

        // Verifications
        if ((field.get_type_data() != TERM_T) || (so.get_type_data() != TERM_T)) {
            KADATH_THROW("Domain_spheric_periodic_shell::fitwaves_nonflat only defined with respect to tensors");
        }

        if ((field.get_val_t().get_valence() != 0) || (so.get_val_t().get_valence() != 0)) {
            KADATH_THROW("Domain_spheric_periodic_shell::fitwaves_nonflat only defined with respect to scalars");
        }

        bool doder = ((so.get_p_der_t() == nullptr) || (field.get_p_der_t() == nullptr)) ? false : true;

        int dom = so.get_dom();

        Index ppp(nbr_coefs);
        double masse_val = val_boundary(OUTER_BC, mult_r(field.get_val_t()()(dom)-1) / 2., ppp);
        double masse_der = (doder) ? val_boundary(OUTER_BC, mult_r(field.get_der_t()()(dom)-1) / 2., ppp) : 0;

        double rmax = alpha + beta;

        int max = 0;
        int baset = (*so.get_val_t()()(dom).get_base().bases_1d[1])(0);
        switch (baset) {
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

        Index pos(nbr_coefs);

        Val_domain resval(this);
        resval.allocate_coef();
        Val_domain resder(this);
        resder.allocate_coef();
        do {
            resval.set_coef(pos) = 0;
            resder.set_coef(pos) = 0;
        } while (pos.inc());
        pos.set_start();

        for (int nn = 0; nn < max; nn++) {

            pos.set(1) = nn;

            // Correspond to what ?
            int expo = 0;
            switch (baset) {
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
            double cor = 2 * expo * expo * ome * ome - 1;

            Term_eq* sh;
            Term_eq* dsh;
            double la = sqrt(fabs(lambda2));
            double arg = (lambda2 > 0) ? la * rmax : la * rmax + masse_val * cor * log(rmax) / la + phases(nn);
            double shval = (lambda2 > 0) ? exp(-arg) / rmax : cos(arg) / rmax;
            double dshval = (lambda2 > 0)
                                ? -la * exp(-arg) / rmax - exp(-arg) / rmax / rmax
                                : -sin(arg) / rmax * (la + masse_val * cor / la / rmax) - cos(arg) / rmax / rmax;

            if (doder) {
                double shder = (lambda2 > 0) ? 0 : -sin(arg) / rmax * log(rmax) / la * cor * masse_der;
                sh = new Term_eq(dom, shval, shder);
                double dshder = (lambda2 > 0)
                                    ? 0
                                    : (-cos(arg) / rmax * (la + masse_val * cor / la / rmax) * cor * log(rmax) / la +
                                       sin(arg) / rmax / rmax * (cor * log(rmax) / la - cor / la)) *
                                          masse_der;
                dsh = new Term_eq(dom, dshval, dshder);
            } else {
                sh = new Term_eq(dom, shval);
                dsh = new Term_eq(dom, dshval);
            }

            Term_eq* ff;
            Term_eq* dff;

            double ffval = val_boundary(OUTER_BC, so.get_val_t()()(dom), pos);
            double dffval = val_boundary(OUTER_BC, so.get_val_t()()(dom).der_r(), pos);
            if (doder) {
                double ffder = val_boundary(OUTER_BC, so.get_der_t()()(dom), pos);
                double dffder = val_boundary(OUTER_BC, so.get_der_t()()(dom).der_r(), pos);
                ff = new Term_eq(dom, ffval, ffder);
                dff = new Term_eq(dom, dffval, dffder);
            } else {
                ff = new Term_eq(dom, ffval);
                dff = new Term_eq(dom, dffval);
            }

            Term_eq auxi(rmax * rmax * ((*sh) * (*dff) - (*dsh) * (*ff)));
            resval.set_coef(pos) = auxi.get_val_d();
            if (doder)
                resder.set_coef(pos) = auxi.get_der_d();

            delete sh;
            delete dsh;
            delete ff;
            delete dff;
        }

        // Put the basis and the result in a term_eq
        if (so.get_val_t()()(dom).check_if_zero()) {
            resval.set_zero();
        } else {
            resval.set_base() = so.get_val_t()()(dom).get_base();
        }
        if (doder) {
            if (so.get_der_t()()(dom).check_if_zero()) {
                resder.set_zero();
            } else {
                resder.set_base() = so.get_der_t()()(dom).get_base();
            }
        }
        Scalar res(so.get_val_t()(), false);
        res.set_domain(dom) = resval;

        if (doder) {
            Scalar der(so.get_der_t()(), false);
            der.set_domain(dom) = resder;
            return Term_eq(dom, res, der);
        } else
            return Term_eq(dom, res);
    }

} // namespace Kadath
