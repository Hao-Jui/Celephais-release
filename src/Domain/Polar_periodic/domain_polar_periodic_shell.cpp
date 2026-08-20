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
#include "For_Kadath/Domain/polar_periodic.hpp"
#include "For_Kadath/Array/point.hpp"
#include "For_Kadath/Array/array.hpp"
#include "For_Kadath/Term_eq/term_eq.hpp"
#include "For_Kadath/Val_domain/val_domain.hpp"
#include "For_Kadath/Scalar/scalar.hpp"

namespace Kadath
{
    void coef_1d(int, Array<double>&);
    void coef_i_1d(int, Array<double>&);
    int der_1d(int, Array<double>&);

    // Standard constructor
    Domain_polar_periodic_shell::Domain_polar_periodic_shell(int num, int ttype, double rin, double rout, double oome,
                                                             const Dim_array& nbr)
        : Domain(num, ttype, nbr), alpha((rout - rin) / 2.), beta((rout + rin) / 2.), ome(oome), type_time(TO_PI)
    {
        assert(nbr.get_ndim() == 3);
        switch (type_time) {
            case TO_PI:
                maxt = M_PI;
                break;
            default:
                KADATH_THROW("Unknown case for type_time");
        }
        do_coloc();
        // ome_term_eq = nullptr ;
        ome_term_eq = new Term_eq(num_dom, ome);
        ome_term_eq->set_der_d(0.);
    }

    // Constructor by copy
    Domain_polar_periodic_shell::Domain_polar_periodic_shell(const Domain_polar_periodic_shell& so)
        : Domain(so), alpha(so.alpha), beta(so.beta), ome(so.ome), type_time(so.type_time), maxt(so.maxt)
    {
        // ome_term_eq = nullptr ;
        ome_term_eq = new Term_eq(num_dom, ome);
        ome_term_eq->set_der_d(0.);
    }

    Domain_polar_periodic_shell::Domain_polar_periodic_shell(int num, BinarySource& source)
        : Domain(num, source)
    {
        alpha = source.read<double>();
        beta = source.read<double>();
        ome = source.read<double>();
        type_time = source.read<int>();
        switch (type_time) {
            case TO_PI:
                maxt = M_PI;
                break;
            default:
                KADATH_THROW("Unknown case for type_time");
        }
        do_coloc();
        ome_term_eq = new Term_eq(num_dom, ome);
        ome_term_eq->set_der_d(0.);
    }

    // Destructor
    Domain_polar_periodic_shell::~Domain_polar_periodic_shell()
    {
        if (ome_term_eq != nullptr)
            delete ome_term_eq;
    }

    void Domain_polar_periodic_shell::save(BinarySink& sink) const
    {
        nbr_points.save(sink);
        nbr_coefs.save(sink);
        sink.write<int>(ndim);
        sink.write<int>(type_base);
        sink.write<double>(alpha);
        sink.write<double>(beta);
        sink.write<double>(ome);
        sink.write<int>(type_time);
    }

    ostream& Domain_polar_periodic_shell::print(ostream& o) const
    {
        o << "Polar_periodic shell" << endl;
        o << "time goes to " << maxt << endl;
        o << beta - alpha << " < r < " << beta + alpha << endl;
        o << "Omega   = " << ome << endl;
        o << "Nbr pts = " << nbr_points << endl;
        o << endl;
        return o;
    }

    /*
    int Domain_polar_periodic_shell::nbr_unknowns_from_adapted() const {
      return 1 ;
    }


    void Domain_polar_periodic_shell::vars_to_terms() const {

      if (ome_term_eq != nullptr)
          delete ome_term_eq ;

      ome_term_eq = new Term_eq (num_dom, ome) ;
      update() ;
    }

    void Domain_polar_periodic_shell::affecte_coef(int& conte, int cc, bool& found) const {
        if (conte==cc) {
        ome_term_eq->set_der_d(1.) ;
        found = true ;
        }
        else {
        ome_term_eq->set_der_d(0.) ;
        found = false ;
        }
        conte ++ ;
        update() ;
    }

    void Domain_polar_periodic_shell::xx_to_vars_from_adapted(double new_ome, const Array<double>& xx, int& pos) const {

        new_ome -= xx(pos) ;
        pos ++ ;
    }

    void Domain_polar_periodic_shell::update_mapping (double cor) {

      ome += cor ;
      clear_import_plan_cache() ;
      for (int l=0 ; l<ndim ; l++) {
            if (absol[l] !=nullptr) delete absol[l] ;
            if (cart[l] !=nullptr) delete cart[l] ;
            if (cart_surr[l] !=nullptr) delete cart_surr[l] ;
            absol[l] = nullptr ;
            cart_surr[l] = nullptr ;
            cart[l]=  nullptr ;
        }
        if (radius !=nullptr)
            delete radius ;
        radius = nullptr ;
      update() ;
    }

    void Domain_polar_periodic_shell::update_variable (double cor_ome, const Scalar& old, Scalar& res) const {
          Val_domain cor (-cor_ome * old(num_dom).der_abs(3) / ome) ;
          res.set_domain(num_dom) = cor + old(num_dom) ;
          res.set_domain(num_dom).set_base() = old(num_dom).get_base() ;
    }


    void Domain_polar_periodic_shell::update_constante (double cor_ome, const Scalar& old, Scalar& res) const {

          Point MM(3) ;
          Index pos(nbr_points) ;
          res.set_domain(num_dom).allocate_conf() ;
          do {

          MM.set(1) = (*absol[0])(pos) ;
          MM.set(2) = (*absol[1])(pos) ;
          MM.set(3) = (*coloc[2])(pos)/(ome+cor_ome) ;

          res.set_domain(num_dom).set(pos) = old.val_point(MM, +1) ;

          }
          while (pos.inc()) ;

          res.set_domain(num_dom).set_base() = old(num_dom).get_base() ;
    }
    void Domain_polar_periodic_shell::xx_to_ders_from_adapted(const Array<double>& xx, int& pos) const {
         ome_term_eq->set_der_d(xx(pos)) ;
         pos++ ;
         update() ;
    }


    void Domain_polar_periodic_shell::update_term_eq (Term_eq* so) const {

      Tensor der (*so->val_t, false) ;
      for (int cmp=0 ; cmp<so->val_t->get_n_comp() ; cmp ++) {



        Val_domain dert ((*(*so->val_t).cmp[cmp])(num_dom).der_abs(3)) ;

        if (!dert.check_if_zero()) {
          Val_domain res(this) ;
          res.allocate_conf() ;
          Index pos (nbr_points) ;
          do {
        res.set(pos) =   (*ome_term_eq->der_d) * (*coloc[2])(pos(2)) * dert(pos) ;
          }
        while (pos.inc()) ;
      res.set_base() = (*(*so->val_t).cmp[cmp])(num_dom).get_base() ;
      der.cmp[cmp]->set_domain(num_dom) = res ;
        }
        else
          der.cmp[cmp]->set_domain(num_dom).set_zero() ;
      }

      so->set_der_t (der) ;
    }


    void Domain_polar_periodic_shell::update() const {
      ome_term_eq->set_val_d(ome) ;
    }
    */
    Val_domain Domain_polar_periodic_shell::der_normal(const Val_domain& so, int bound) const
    {

        Val_domain res(so.der_var(1));
        switch (bound) {
            case OUTER_BC:
                res /= alpha;
                break;
            case INNER_BC:
                res /= alpha;
                break;
            default:
                KADATH_THROW("Unknown boundary case in Domain_polar_periodic_shell::der_normal");
        }
        return res;
    }

    // Computes the cartesian coordinates
    void Domain_polar_periodic_shell::do_absol() const
    {
        for (int i = 0; i < 3; i++)
            assert(coloc[i] != nullptr);
        for (int i = 0; i < 3; i++)
            assert(absol[i] == nullptr);
        for (int i = 0; i < 3; i++) {
            absol[i] = new Val_domain(this);
            absol[i]->allocate_conf();
        }
        Index index(nbr_points);
        do {
            absol[0]->set(index) = (alpha * ((*coloc[0])(index(0))) + beta) * sin((*coloc[1])(index(1))); // rho
            absol[1]->set(index) = (alpha * ((*coloc[0])(index(0))) + beta) * cos((*coloc[1])(index(1))); // z
            absol[2]->set(index) = (*coloc[2])(index(2)) / ome;                                           // time
        } while (index.inc());
    }

    // Computes the radius
    void Domain_polar_periodic_shell::do_radius() const
    {

        for (int i = 0; i < 2; i++)
            assert(coloc[i] != nullptr);
        assert(radius == nullptr);
        radius = new Val_domain(this);
        radius->allocate_conf();
        Index index(nbr_points);
        do
            radius->set(index) = alpha * ((*coloc[0])(index(0))) + beta;
        while (index.inc());
    }

    // Is a point inside this domain ?
    bool Domain_polar_periodic_shell::is_in(const Point& xx, double prec) const
    {

        assert(xx.get_ndim() == 3);

        double rho_loc = xx(1);
        double z_loc = xx(2);
        double air_loc = sqrt(rho_loc * rho_loc + z_loc * z_loc);

        bool res = ((air_loc <= alpha + beta + prec) && (air_loc >= beta - alpha - prec)) ? true : false;
        if ((xx(3) < 0 - prec) || (xx(3) > maxt / ome + prec))
            res = false;

        return res;
    }

    // Convert absolute coordinates to numerical ones
    const Point Domain_polar_periodic_shell::absol_to_num(const Point& abs) const
    {

        assert(is_in(abs));
        Point num(3);
        double rho_loc = fabs(abs(1));
        double z_loc = abs(2);
        double air = sqrt(rho_loc * rho_loc + z_loc * z_loc);
        num.set(1) = (air - beta) / alpha;

        if (rho_loc == 0) {
            // Sur l'axe
            num.set(2) = (z_loc >= 0) ? 0 : M_PI;
        } else {
            num.set(2) = atan(rho_loc / z_loc);
        }

        if (num(2) < 0)
            num.set(2) = M_PI + num(2);

        num.set(3) = abs(3) * ome;

        return num;
    }

    double coloc_leg(int, int);
    void Domain_polar_periodic_shell::do_coloc()
    {

        switch (type_base) {
            case CHEB_TYPE:
                nbr_coefs = nbr_points;
                del_deriv();
                for (int i = 0; i < ndim; i++)
                    coloc[i] = new Array<double>(nbr_points(i));
                for (int i = 0; i < nbr_points(0); i++)
                    coloc[0]->set(i) = -cos(M_PI * i / (nbr_points(0) - 1));
                for (int j = 0; j < nbr_points(1); j++)
                    coloc[1]->set(j) = M_PI / 2. * j / (nbr_points(1) - 1);
                for (int k = 0; k < nbr_points(2); k++)
                    coloc[2]->set(k) = maxt * k / (nbr_points(2) - 1);
                break;
            case LEG_TYPE:
                nbr_coefs = nbr_points;
                del_deriv();
                for (int i = 0; i < ndim; i++)
                    coloc[i] = new Array<double>(nbr_points(i));
                for (int i = 0; i < nbr_points(0); i++)
                    coloc[0]->set(i) = coloc_leg(i, nbr_points(0));
                for (int j = 0; j < nbr_points(1); j++)
                    coloc[1]->set(j) = M_PI / 2. * j / (nbr_points(1) - 1);
                for (int k = 0; k < nbr_points(2); k++)
                    coloc[2]->set(k) = maxt * k / (nbr_points(2) - 1);
                break;
            default:
                KADATH_THROW("Unknown type of basis in Domain_polar_periodic_shell::do_coloc");
        }
    }
    // Base for a function symetric in z, using Chebyshev
    void Domain_polar_periodic_shell::set_cheb_base(Base_spectral& base) const
    {
        set_cheb_base_with_m(base, 0);
    }

    // Base for a function anti-symetric in z, using Chebyshev
    void Domain_polar_periodic_shell::set_anti_cheb_base(Base_spectral& base) const
    {
        set_anti_cheb_base_with_m(base, 0);
    }

    // Base for a function symetric in z, using Legendre
    void Domain_polar_periodic_shell::set_legendre_base(Base_spectral& base) const
    {
        set_legendre_base_with_m(base, 0);
    }

    // Base for a function anti-symetric in z, using Legendre
    void Domain_polar_periodic_shell::set_anti_legendre_base(Base_spectral& base) const
    {
        set_anti_legendre_base_with_m(base, 0);
    }

    // Base for a function symetric in z, using Chebyshev
    void Domain_polar_periodic_shell::set_cheb_base_with_m(Base_spectral& base, int m) const
    {

        assert(type_base == CHEB_TYPE);
        base.allocate(nbr_coefs);
        base.def = true;
        Index index(base.bases_1d[0]->get_dimensions());

        base.bases_1d[2]->set(0) = COS; // Base in time
        for (int k = 0; k < nbr_coefs(2); k++) {
            base.bases_1d[1]->set(k) = (m % 2 == 0) ? COS_EVEN : SIN_ODD;
            for (int j = 0; j < nbr_coefs(1); j++) {
                index.set(0) = j;
                index.set(1) = k;
                base.bases_1d[0]->set(index) = CHEB;
            }
        }
    }

    // Base for a function anti-symetric in z, using Chebyshev
    void Domain_polar_periodic_shell::set_anti_cheb_base_with_m(Base_spectral& base, int m) const
    {
        assert(type_base == CHEB_TYPE);
        base.allocate(nbr_coefs);

        Index index(base.bases_1d[0]->get_dimensions());

        base.def = true;

        base.bases_1d[2]->set(0) = COS; // Base in time
        for (int k = 0; k < nbr_coefs(2); k++) {
            base.bases_1d[1]->set(k) = (m % 2 == 0) ? COS_ODD : SIN_EVEN;
            for (int j = 0; j < nbr_coefs(1); j++) {
                index.set(0) = j;
                index.set(1) = k;
                base.bases_1d[0]->set(index) = CHEB;
            }
        }
    }

    // Base for a function symetric in z, using Legendre
    void Domain_polar_periodic_shell::set_legendre_base_with_m(Base_spectral& base, int m) const
    {

        assert(type_base == LEG_TYPE);
        base.allocate(nbr_coefs);
        base.def = true;
        Index index(base.bases_1d[0]->get_dimensions());

        base.bases_1d[2]->set(0) = COS; // Base in time
        for (int k = 0; k < nbr_coefs(2); k++) {
            base.bases_1d[1]->set(k) = (m % 2 == 0) ? COS_EVEN : SIN_ODD;
            for (int j = 0; j < nbr_coefs(1); j++) {
                index.set(0) = j;
                index.set(1) = k;
                base.bases_1d[0]->set(index) = LEG;
            }
        }
    }

    // Base for a function anti-symetric in z, using Chebyshev
    void Domain_polar_periodic_shell::set_anti_legendre_base_with_m(Base_spectral& base, int m) const
    {

        assert(type_base == LEG_TYPE);
        base.allocate(nbr_coefs);

        Index index(base.bases_1d[0]->get_dimensions());

        base.def = true;

        base.bases_1d[2]->set(0) = COS; // Base in time
        for (int k = 0; k < nbr_coefs(2); k++) {
            base.bases_1d[1]->set(k) = (m % 2 == 0) ? COS_ODD : SIN_EVEN;
            for (int j = 0; j < nbr_coefs(1); j++) {
                index.set(0) = j;
                index.set(1) = k;
                base.bases_1d[0]->set(index) = LEG;
            }
        }
    }

    // Computes the derivatives with respect to rho,Z as a function of the numerical ones.
    void Domain_polar_periodic_shell::do_der_abs_from_der_var(Val_domain** der_var, Val_domain** der_abs) const
    {
        Val_domain dr(*der_var[0] / alpha);
        Val_domain dtsr(der_var[1]->div_x() / alpha);

        // D / drho
        der_abs[0] = new Val_domain(dr.mult_sin_theta() + dtsr.mult_cos_theta());
        // d/dz :
        der_abs[1] = new Val_domain(dr.mult_cos_theta() - dtsr.mult_sin_theta());
        // d/dt :
        der_abs[2] = new Val_domain(*der_var[2] * ome);
    }

    void Domain_polar_periodic_shell::set_cheb_base_r_spher(Base_spectral& base) const
    {

        assert(type_base == CHEB_TYPE);
        base.allocate(nbr_coefs);

        Index index(base.bases_1d[0]->get_dimensions());

        base.def = true;
        base.bases_1d[2]->set(0) = SIN;
        for (int k = 0; k < nbr_coefs(2); k++) {
            base.bases_1d[1]->set(k) = COS_EVEN;
            for (int j = 0; j < nbr_coefs(1); j++) {
                index.set(0) = j;
                index.set(1) = k;
                base.bases_1d[0]->set(index) = CHEB;
            }
        }
    }

    void Domain_polar_periodic_shell::set_cheb_base_t_spher(Base_spectral& base) const
    {

        assert(type_base == CHEB_TYPE);
        base.allocate(nbr_coefs);

        Index index(base.bases_1d[0]->get_dimensions());

        base.def = true;
        base.bases_1d[2]->set(0) = SIN;
        for (int k = 0; k < nbr_coefs(2); k++) {
            base.bases_1d[1]->set(k) = SIN_EVEN;
            for (int j = 0; j < nbr_coefs(1); j++) {
                index.set(0) = j;
                index.set(1) = k;
                base.bases_1d[0]->set(index) = CHEB;
            }
        }
    }

    void Domain_polar_periodic_shell::set_cheb_base_p_spher(Base_spectral& base) const
    {

        assert(type_base == CHEB_TYPE);
        base.allocate(nbr_coefs);

        Index index(base.bases_1d[0]->get_dimensions());

        base.def = true;
        base.bases_1d[2]->set(0) = SIN;
        for (int k = 0; k < nbr_coefs(2); k++) {
            base.bases_1d[1]->set(k) = SIN_ODD;
            for (int j = 0; j < nbr_coefs(1); j++) {
                index.set(0) = j;
                index.set(1) = k;
                base.bases_1d[0]->set(index) = CHEB;
            }
        }
    }

    void Domain_polar_periodic_shell::set_legendre_base_r_spher(Base_spectral& base) const
    {

        assert(type_base == LEG_TYPE);
        base.allocate(nbr_coefs);

        Index index(base.bases_1d[0]->get_dimensions());

        base.def = true;
        base.bases_1d[2]->set(0) = SIN;
        for (int k = 0; k < nbr_coefs(2); k++) {
            base.bases_1d[1]->set(k) = COS_EVEN;
            for (int j = 0; j < nbr_coefs(1); j++) {
                index.set(0) = j;
                index.set(1) = k;
                base.bases_1d[0]->set(index) = LEG;
            }
        }
    }

    void Domain_polar_periodic_shell::set_legendre_base_t_spher(Base_spectral& base) const
    {

        assert(type_base == LEG_TYPE);
        base.allocate(nbr_coefs);

        Index index(base.bases_1d[0]->get_dimensions());

        base.def = true;
        base.bases_1d[2]->set(0) = SIN;
        for (int k = 0; k < nbr_coefs(2); k++) {
            base.bases_1d[1]->set(k) = SIN_EVEN;
            for (int j = 0; j < nbr_coefs(1); j++) {
                index.set(0) = j;
                index.set(1) = k;
                base.bases_1d[0]->set(index) = LEG;
            }
        }
    }

    void Domain_polar_periodic_shell::set_legendre_base_p_spher(Base_spectral& base) const
    {

        assert(type_base == CHEB_TYPE);
        base.allocate(nbr_coefs);

        Index index(base.bases_1d[0]->get_dimensions());

        base.def = true;
        base.bases_1d[2]->set(0) = SIN;
        for (int k = 0; k < nbr_coefs(2); k++) {
            base.bases_1d[1]->set(k) = SIN_ODD;
            for (int j = 0; j < nbr_coefs(1); j++) {
                index.set(0) = j;
                index.set(1) = k;
                base.bases_1d[0]->set(index) = LEG;
            }
        }
    }

    // Rules for the multiplication of two basis.
    Base_spectral Domain_polar_periodic_shell::mult(const Base_spectral& a, const Base_spectral& b) const
    {

        assert(a.ndim == 3);
        assert(b.ndim == 3);

        Base_spectral res(3);
        bool res_def = true;

        if (!a.def)
            res_def = false;
        if (!b.def)
            res_def = false;

        if (res_def) {

            // Bases in time :
            res.bases_1d[2] = std::make_unique<Array<int>>(a.bases_1d[2]->get_dimensions());
            switch ((*a.bases_1d[2])(0)) {
                case COS:
                    switch ((*b.bases_1d[2])(0)) {
                        case COS:
                            res.bases_1d[2]->set(0) = COS;
                            break;
                        case SIN:
                            res.bases_1d[2]->set(0) = SIN;
                            break;
                        default:
                            res_def = false;
                            break;
                    }
                    break;
                case SIN:
                    switch ((*b.bases_1d[2])(0)) {
                        case COS:
                            res.bases_1d[2]->set(0) = SIN;
                            break;
                        case SIN:
                            res.bases_1d[2]->set(0) = COS;
                            break;
                        default:
                            res_def = false;
                            break;
                    }
                    break;
            }

            // Bases in theta :
            Index index_1(a.bases_1d[1]->get_dimensions());
            res.bases_1d[1] = std::make_unique<Array<int>>(a.bases_1d[1]->get_dimensions());
            do {
                switch ((*a.bases_1d[1])(index_1)) {
                    case COS_EVEN:
                        switch ((*b.bases_1d[1])(index_1)) {
                            case COS_EVEN:
                                res.bases_1d[1]->set(index_1) = COS_EVEN;
                                break;
                            case COS_ODD:
                                res.bases_1d[1]->set(index_1) = COS_ODD;
                                break;
                            case SIN_EVEN:
                                res.bases_1d[1]->set(index_1) = SIN_EVEN;
                                break;
                            case SIN_ODD:
                                res.bases_1d[1]->set(index_1) = SIN_ODD;
                                break;
                            default:
                                res_def = false;
                                break;
                        }
                        break;
                    case COS_ODD:
                        switch ((*b.bases_1d[1])(index_1)) {
                            case COS_EVEN:
                                res.bases_1d[1]->set(index_1) = COS_ODD;
                                break;
                            case COS_ODD:
                                res.bases_1d[1]->set(index_1) = COS_EVEN;
                                break;
                            case SIN_EVEN:
                                res.bases_1d[1]->set(index_1) = SIN_ODD;
                                break;
                            case SIN_ODD:
                                res.bases_1d[1]->set(index_1) = SIN_EVEN;
                                break;
                            default:
                                res_def = false;
                                break;
                        }
                        break;
                    case SIN_EVEN:
                        switch ((*b.bases_1d[1])(index_1)) {
                            case COS_EVEN:
                                res.bases_1d[1]->set(index_1) = SIN_EVEN;
                                break;
                            case COS_ODD:
                                res.bases_1d[1]->set(index_1) = SIN_ODD;
                                break;
                            case SIN_EVEN:
                                res.bases_1d[1]->set(index_1) = COS_EVEN;
                                break;
                            case SIN_ODD:
                                res.bases_1d[1]->set(index_1) = COS_ODD;
                                break;
                            default:
                                res_def = false;
                                break;
                        }
                        break;
                    case SIN_ODD:
                        switch ((*b.bases_1d[1])(index_1)) {
                            case COS_EVEN:
                                res.bases_1d[1]->set(index_1) = SIN_ODD;
                                break;
                            case COS_ODD:
                                res.bases_1d[1]->set(index_1) = SIN_EVEN;
                                break;
                            case SIN_EVEN:
                                res.bases_1d[1]->set(index_1) = COS_ODD;
                                break;
                            case SIN_ODD:
                                res.bases_1d[1]->set(index_1) = COS_EVEN;
                                break;
                            default:
                                res_def = false;
                                break;
                        }
                        break;
                    default:
                        res_def = false;
                        break;
                }
            } while (index_1.inc());

            // Base in r :
            Index index_0(a.bases_1d[0]->get_dimensions());
            res.bases_1d[0] = std::make_unique<Array<int>>(a.bases_1d[0]->get_dimensions());
            do {
                switch ((*a.bases_1d[0])(index_0)) {
                    case CHEB:
                        switch ((*b.bases_1d[0])(index_0)) {
                            case CHEB:
                                res.bases_1d[0]->set(index_0) = CHEB;
                                break;
                            default:
                                res_def = false;
                                break;
                        }
                        break;
                    case LEG:
                        switch ((*b.bases_1d[0])(index_0)) {
                            case LEG:
                                res.bases_1d[0]->set(index_0) = LEG;
                                break;
                            default:
                                res_def = false;
                                break;
                        }
                        break;
                    default:
                        res_def = false;
                        break;
                }
            } while (index_0.inc());
        }

        if (!res_def)
            for (int dim = 0; dim < a.ndim; dim++)
                if (res.bases_1d[dim] != nullptr) {
                    res.bases_1d[dim].reset();
                    res.bases_1d[dim] = nullptr;
                }
        res.def = res_def;
        return res;
    }
} // namespace Kadath
