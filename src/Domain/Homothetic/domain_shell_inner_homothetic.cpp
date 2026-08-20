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
#include "For_Kadath/Utilities/utilities.hpp"
#include "For_Kadath/Domain/homothetic.hpp"
#include "For_Kadath/Array/point.hpp"
#include "For_Kadath/Array/array.hpp"
#include "For_Kadath/Val_domain/val_domain.hpp"
#include "For_Kadath/Scalar/scalar.hpp"

namespace Kadath
{
    void coef_1d(int, Array<double>&);
    void coef_i_1d(int, Array<double>&);
    int der_1d(int, Array<double>&);

    // Standard constructor
    Domain_shell_inner_homothetic::Domain_shell_inner_homothetic(const Space& sss, int num, int ttype, double rin,
                                                                 double rout, const Point& cr, const Dim_array& nbr)
        : Domain_shell_inner_adapted(sss, num, ttype, rin, rout, cr, nbr)
    {
    }

    // Constructor by copy
    Domain_shell_inner_homothetic::Domain_shell_inner_homothetic(const Domain_shell_inner_homothetic& so)
        : Domain_shell_inner_adapted(so)
    {
    }

    Domain_shell_inner_homothetic::Domain_shell_inner_homothetic(const Space& sss, int num, BinarySource& source)
        : Domain_shell_inner_adapted(sss, num, source)
    {
    }

    // Destructor
    Domain_shell_inner_homothetic::~Domain_shell_inner_homothetic() {}

    int Domain_shell_inner_homothetic::nbr_unknowns_from_adapted() const
    {

        int res = 1;
        return res;
    }

    void Domain_shell_inner_homothetic::affecte_coef(int& conte, int cc, bool& found) const
    {
        Val_domain auxi(this);
        auxi.std_base();
        auxi.set_in_coef();
        auxi.allocate_coef();
        *auxi.cf = 0;

        found = false;

        if (conte == cc) {
            Index pos_cf(nbr_coefs);
            auxi.cf->set(pos_cf) = 1;
            found = true;
        }

        conte++;

        if (found) {
            Scalar auxi_scal(sp);
            auxi_scal.set_domain(num_dom) = auxi;
            inner_radius_term_eq->set_der_t(auxi_scal);
        } else {
            inner_radius_term_eq->set_der_zero();
        }
        update();
    }

    void Domain_shell_inner_homothetic::xx_to_vars_from_adapted(Val_domain& new_inner_radius, const Array<double>& xx,
                                                                int& pos) const
    {

        new_inner_radius.allocate_coef();
        *new_inner_radius.cf = 0;

        Index pos_cf(nbr_coefs);
        pos_cf.set(0) = 0;

        new_inner_radius.cf->set(pos_cf) -= xx(pos);
        pos++;
        new_inner_radius.set_base() = inner_radius->get_base();
    }

    void Domain_shell_inner_homothetic::xx_to_ders_from_adapted(const Array<double>& xx, int& pos) const
    {

        Val_domain auxi(this);
        auxi.std_base();
        auxi.set_in_coef();
        auxi.allocate_coef();
        *auxi.cf = 0;

        Index pos_cf(nbr_coefs);
        pos_cf.set(0) = 0;

        auxi.cf->set(pos_cf) = xx(pos);
        pos++;

        Scalar auxi_scal(sp);
        auxi_scal.set_domain(num_dom) = auxi;
        inner_radius_term_eq->set_der_t(auxi_scal);
        update();
    }

    ostream& Domain_shell_inner_homothetic::print(ostream& o) const
    {
        o << "Adapted homothetic shell on the inside boundary" << endl;
        o << "Center  = " << center << endl;
        o << "Nbr pts = " << nbr_points << endl;
        o << "Outer radius " << outer_radius << endl;
        Index pos(nbr_points);
        o << "Inner radius " << (*inner_radius)(pos) << endl;
        o << endl;
        return o;
    }

    void Domain_shell_inner_homothetic::update_constante(const Val_domain& cor_inner_radius, const Scalar& old,
                                                         Scalar& res) const
    {
        update_variable(cor_inner_radius, old, res);
    }

    double Domain_shell_inner_homothetic::integ(const Val_domain& so, int bound) const
    {
        Val_domain rrso(mult_sin_theta(so * get_radius() * get_radius()));

        double res = 0;
        if (!so.check_if_zero()) {

            int baset = (*rrso.get_base().bases_1d[1])(0);
            Index pcf(nbr_coefs);
            switch (baset) {
                case COS_ODD:
                    break;
                case SIN_EVEN:
                    break;
                case COS_EVEN: {
                    res += M_PI * val_boundary(bound, rrso, pcf);
                    break;
                }
                case SIN_ODD: {
                    for (int j = 0; j < nbr_coefs(1); j++) {
                        pcf.set(1) = j;
                        res += 2. / (2 * double(j) + 1) * val_boundary(bound, rrso, pcf);
                    }
                    break;
                }

                default:
                    KADATH_THROW("Case not yet implemented in Domain_shell_inner_homothetic::integ");
            }
            res *= 2 * M_PI;
        }
        return res;
    }

} // namespace Kadath
