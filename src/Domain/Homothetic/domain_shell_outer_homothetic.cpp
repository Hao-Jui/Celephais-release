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
    Domain_shell_outer_homothetic::Domain_shell_outer_homothetic(const Space& sss, int num, int ttype, double rin,
                                                                 double rout, const Point& cr, const Dim_array& nbr)
        : Domain_shell_outer_adapted(sss, num, ttype, rin, rout, cr, nbr)
    {
    }

    // Constructor by copy
    Domain_shell_outer_homothetic::Domain_shell_outer_homothetic(const Domain_shell_outer_homothetic& so)
        : Domain_shell_outer_adapted(so)
    {
    }

    Domain_shell_outer_homothetic::Domain_shell_outer_homothetic(const Space& sss, int num, BinarySource& source)
        : Domain_shell_outer_adapted(sss, num, source)
    {
    }

    // Destructor
    Domain_shell_outer_homothetic::~Domain_shell_outer_homothetic() {}

    int Domain_shell_outer_homothetic::nbr_unknowns_from_adapted() const
    {

        int res = 1;
        return res;
    }

    void Domain_shell_outer_homothetic::affecte_coef(int& conte, int cc, bool& found) const
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
            outer_radius_term_eq->set_der_t(auxi_scal);
        } else {
            outer_radius_term_eq->set_der_zero();
        }
        update();
    }

    void Domain_shell_outer_homothetic::xx_to_vars_from_adapted(Val_domain& new_outer_radius, const Array<double>& xx,
                                                                int& pos) const
    {

        new_outer_radius.allocate_coef();
        *new_outer_radius.cf = 0;

        Index pos_cf(nbr_coefs);
        pos_cf.set(0) = 0;

        new_outer_radius.cf->set(pos_cf) -= xx(pos);
        pos++;
        new_outer_radius.set_base() = outer_radius->get_base();
    }

    void Domain_shell_outer_homothetic::xx_to_ders_from_adapted(const Array<double>& xx, int& pos) const
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
        outer_radius_term_eq->set_der_t(auxi_scal);
        update();
    }

    ostream& Domain_shell_outer_homothetic::print(ostream& o) const
    {
        o << "Adapted homothetic shell on the outside boundary" << endl;
        o << "Center  = " << center << endl;
        o << "Nbr pts = " << nbr_points << endl;
        Index pos(nbr_points);
        o << "Outer radius " << (*outer_radius)(pos) << endl;
        o << "Inner radius " << inner_radius << endl;
        o << endl;
        return o;
    }

    void Domain_shell_outer_homothetic::update_constante(const Val_domain& cor_outer_radius, const Scalar& old,
                                                         Scalar& res) const
    {
        update_variable(cor_outer_radius, old, res);
    }

} // namespace Kadath
