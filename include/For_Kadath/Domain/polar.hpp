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

#pragma once

#include "For_Kadath/Space/space.hpp"
namespace Kadath
{

    class Domain_polar_nucleus : public Domain
    {

      private:
        double alpha;
        Point center;

      public:
        Domain_polar_nucleus(int nim, int ttype, double radius, const Point& cr, const Dim_array& nbr);
        Domain_polar_nucleus(const Domain_polar_nucleus& so);
        Domain_polar_nucleus(const Domain_polar_nucleus& so, bool import);
        Domain_polar_nucleus(int num, BinarySource& source); ///< Modern API.

        ~Domain_polar_nucleus() override;
        void save(BinarySink&) const override; ///< Modern API.

      private:
        void do_absol() const override;
        void do_radius() const override;
        void do_cart() const override;
        void do_cart_surr() const override;

      private:
        void set_cheb_base(Base_spectral&) const override;
        void set_legendre_base(Base_spectral&) const override;
        void set_anti_cheb_base(Base_spectral&) const override;
        void set_anti_legendre_base(Base_spectral&) const override;
        void set_cheb_base_with_m(Base_spectral&, int m) const override;
        void set_legendre_base_with_m(Base_spectral&, int m) const override;
        void set_anti_cheb_base_with_m(Base_spectral&, int m) const override;
        void set_anti_legendre_base_with_m(Base_spectral&, int m) const override;
        void do_coloc() override;
        int give_place_var(char*) const override;

      public:
        Point get_center() const override { return center; };
        bool is_in(const Point& xx, double prec = 1e-13) const override;
        const Point absol_to_num(const Point&) const override;
        void do_der_abs_from_der_var(Val_domain** der_var, Val_domain** der_abs) const override;
        Base_spectral mult(const Base_spectral&, const Base_spectral&) const override;

      public:
        Val_domain mult_cos_theta(const Val_domain&) const override;
        Val_domain mult_sin_theta(const Val_domain&) const override;
        Val_domain div_sin_theta(const Val_domain&) const override;
        Val_domain div_x(const Val_domain&) const override;
        Val_domain mult_r(const Val_domain&) const override;
        Val_domain div_r(const Val_domain&) const override;
        Val_domain laplacian(const Val_domain&, int) const override;
        Val_domain laplacian2(const Val_domain&, int) const override;
        Val_domain der_r(const Val_domain&) const override;
        Val_domain dt(const Val_domain&) const override;
        Val_domain srdr(const Val_domain&) const override;
        double integrale(const Val_domain&) const override;
        double integ_volume(const Val_domain&) const override;

        double val_boundary(int, const Val_domain&, const Index&) const override;
        void find_other_dom(int, int, int&, int&) const override;
        Val_domain der_normal(const Val_domain&, int) const override;

        int nbr_unknowns(const Tensor&, int) const override;
        int nbr_unknowns_val_domain(const Val_domain& so, int mquant, int llim) const;
        Array<int> nbr_conditions(const Tensor&, int, int, int n_cmp = -1, Array<int>** p_cmp = nullptr) const override;
        int nbr_conditions_val_domain(const Val_domain& so, int mquant, int llim, int order) const;
        Array<int> nbr_conditions_boundary(const Tensor&, int, int, int n_cmp = -1,
                                                   Array<int>** p_cmp = nullptr) const override;
        int nbr_conditions_val_domain_boundary(const Val_domain& eq, int mquant) const;
        void export_tau(const Tensor&, int, int, Array<double>&, int&, const Array<int>&, int n_cmp = -1,
                                Array<int>** p_cmp = nullptr) const override;
        void export_tau_val_domain(const Val_domain& eq, int mquant, int llim, int order, Array<double>& res,
                                   int& pos_res, int ncond) const;
        void export_tau_boundary(const Tensor&, int, int, Array<double>&, int&, const Array<int>&,
                                         int n_cmp = -1, Array<int>** p_cmp = nullptr) const override;
        void export_tau_val_domain_boundary(const Val_domain& eq, int mquant, int bound, Array<double>& res,
                                            int& pos_res, int ncond) const;
        void affecte_tau(Tensor&, int, const Array<double>&, int&) const override;
        void affecte_tau_val_domain(Val_domain& so, int mquant, int llim, const Array<double>& cf, int& pos_cf) const;
        void affecte_tau_one_coef(Tensor&, int, int, int&) const override;
        void affecte_tau_one_coef_val_domain(Val_domain& so, int mquant, int llim, int cc, int& pos_cf) const;

      public:
        ostream& print(ostream& o) const override;
    };

    class Domain_polar_shell : public Domain
    {

      private:
        double alpha;
        double beta;
        Point center;

      public:
        Domain_polar_shell(int num, int ttype, double r_int, double r_ext, const Point& cr, const Dim_array& nbr);
        Domain_polar_shell(const Domain_polar_shell& so);
        Domain_polar_shell(const Domain_polar_shell& so, bool import);

        Domain_polar_shell(int num, BinarySource& source); ///< Modern API.

        ~Domain_polar_shell() override;
        void save(BinarySink&) const override; ///< Modern API.

      private:
        void do_absol() const override;
        void do_radius() const override;
        void do_cart() const override;
        void do_cart_surr() const override;

      private:
        void set_cheb_base(Base_spectral&) const override;
        void set_legendre_base(Base_spectral&) const override;
        void set_anti_cheb_base(Base_spectral&) const override;
        void set_anti_legendre_base(Base_spectral&) const override;
        void set_cheb_base_with_m(Base_spectral&, int m) const override;
        void set_legendre_base_with_m(Base_spectral&, int m) const override;
        void set_anti_cheb_base_with_m(Base_spectral&, int m) const override;
        void set_anti_legendre_base_with_m(Base_spectral&, int m) const override;

        void do_coloc() override;
        int give_place_var(char*) const override;

      public:
        Point get_center() const override { return center; };
        bool is_in(const Point& xx, double prec = 1e-13) const override;
        const Point absol_to_num(const Point&) const override;
        void do_der_abs_from_der_var(Val_domain** der_var, Val_domain** der_abs) const override;

        Base_spectral mult(const Base_spectral&, const Base_spectral&) const override;

      public:
        Val_domain mult_cos_theta(const Val_domain&) const override;
        Val_domain mult_sin_theta(const Val_domain&) const override;
        Val_domain div_sin_theta(const Val_domain&) const override;
        Val_domain mult_r(const Val_domain&) const override;
        Val_domain div_r(const Val_domain&) const override;
        Val_domain laplacian(const Val_domain&, int) const override;
        Val_domain laplacian2(const Val_domain&, int) const override;
        Val_domain der_r(const Val_domain&) const override;
        Val_domain dt(const Val_domain&) const override;
        Val_domain div_xp1(const Val_domain&) const override;
        double integrale(const Val_domain&) const override;
        double integ_volume(const Val_domain&) const override;

        void find_other_dom(int, int, int&, int&) const override;
        Val_domain der_normal(const Val_domain&, int) const override;
        double integ(const Val_domain&, int) const override;
        double integmoment(const Val_domain& so, int n, int bound) const override;
        double val_boundary(int, const Val_domain&, const Index&) const override;

        int nbr_unknowns(const Tensor&, int) const override;
        int nbr_unknowns_val_domain(const Val_domain& so, int mquant) const;
        Array<int> nbr_conditions(const Tensor&, int, int, int n_cmp = -1, Array<int>** p_cmp = nullptr) const override;
        int nbr_conditions_val_domain(const Val_domain& so, int mquant, int order) const;
        Array<int> nbr_conditions_boundary(const Tensor&, int, int, int n_cmp = -1,
                                                   Array<int>** p_cmp = nullptr) const override;
        int nbr_conditions_val_domain_boundary(const Val_domain& eq, int mquant) const;
        void export_tau(const Tensor&, int, int, Array<double>&, int&, const Array<int>&, int n_cmp = -1,
                                Array<int>** p_cmp = nullptr) const override;
        void export_tau_val_domain(const Val_domain& eq, int mquant, int order, Array<double>& res, int& pos_res,
                                   int ncond) const;
        void export_tau_boundary(const Tensor&, int, int, Array<double>&, int&, const Array<int>&,
                                         int n_cmp = -1, Array<int>** p_cmp = nullptr) const override;
        void export_tau_val_domain_boundary(const Val_domain& eq, int mquant, int bound, Array<double>& res,
                                            int& pos_res, int ncond) const;
        void affecte_tau(Tensor&, int, const Array<double>&, int&) const override;
        void affecte_tau_val_domain(Val_domain& so, int mquant, const Array<double>& cf, int& pos_cf) const;
        void affecte_tau_one_coef(Tensor&, int, int, int&) const override;
        void affecte_tau_one_coef_val_domain(Val_domain& so, int mquant, int cc, int& pos_cf) const;

      public:
        ostream& print(ostream& o) const override;
    };

    class Domain_polar_compact : public Domain
    {

      private:
        double alpha;
        Point center;

      public:
        Domain_polar_compact(int num, int ttype, double r_int, const Point& cr, const Dim_array& nbr);
        Domain_polar_compact(const Domain_polar_compact& so);
        Domain_polar_compact(const Domain_polar_compact& so, bool import);
        Domain_polar_compact(int num, BinarySource& source); ///< Modern API.

        ~Domain_polar_compact() override;
        void save(BinarySink&) const override; ///< Modern API.

      private:
        void do_radius() const override;

      private:
        void set_cheb_base(Base_spectral&) const override;
        void set_legendre_base(Base_spectral&) const override;
        void set_anti_cheb_base(Base_spectral&) const override;
        void set_anti_legendre_base(Base_spectral&) const override;
        void set_cheb_base_with_m(Base_spectral&, int m) const override;
        void set_legendre_base_with_m(Base_spectral&, int m) const override;
        void set_anti_cheb_base_with_m(Base_spectral&, int m) const override;
        void set_anti_legendre_base_with_m(Base_spectral&, int m) const override;
        void do_coloc() override;

        void do_absol() const override;
        void do_cart() const override;
        void do_cart_surr() const override;
        int give_place_var(char*) const override;

      public:
        Point get_center() const override { return center; };
        double get_alpha() const { return alpha; };

        bool is_in(const Point& xx, double prec = 1e-13) const override;
        const Point absol_to_num(const Point&) const override;
        void do_der_abs_from_der_var(Val_domain** der_var, Val_domain** der_abs) const override;

        Base_spectral mult(const Base_spectral&, const Base_spectral&) const override;

      public:
        Val_domain mult_cos_theta(const Val_domain&) const override;
        Val_domain mult_sin_theta(const Val_domain&) const override;
        Val_domain div_sin_theta(const Val_domain&) const override;
        Val_domain div_xm1(const Val_domain&) const override;

        Val_domain mult_xm1(const Val_domain&) const override;
        Val_domain mult_r(const Val_domain&) const override;
        Val_domain div_r(const Val_domain&) const override;
        Val_domain laplacian(const Val_domain&, int) const override;
        Val_domain laplacian2(const Val_domain&, int) const override;
        Val_domain der_r(const Val_domain&) const override;
        Val_domain der_r_rtwo(const Val_domain&) const override;
        Val_domain dt(const Val_domain&) const override;
        Val_domain div_xp1(const Val_domain&) const override;
        double integrale(const Val_domain&) const override;
        double integ_volume(const Val_domain&) const override;
        void set_val_inf(Val_domain& so, double xx) const override;

        void find_other_dom(int, int, int&, int&) const override;
        Val_domain der_normal(const Val_domain&, int) const override;
        double integ(const Val_domain&, int) const override;
        double val_boundary(int, const Val_domain&, const Index&) const override;

        int nbr_unknowns(const Tensor&, int) const override;
        int nbr_unknowns_val_domain(const Val_domain& so, int mquant) const;
        Array<int> nbr_conditions(const Tensor&, int, int, int n_cmp = -1, Array<int>** p_cmp = nullptr) const override;
        int nbr_conditions_val_domain(const Val_domain& eq, int mquant, int order) const;
        Array<int> nbr_conditions_boundary(const Tensor&, int, int, int n_cmp = -1,
                                                   Array<int>** p_cmp = nullptr) const override;
        int nbr_conditions_val_domain_boundary(const Val_domain& eq, int mquant) const;
        void export_tau(const Tensor&, int, int, Array<double>&, int&, const Array<int>&, int n_cmp = -1,
                                Array<int>** p_cmp = nullptr) const override;
        void export_tau_val_domain(const Val_domain& eq, int mquant, int order, Array<double>& res, int& pos_res,
                                   int ncond) const;
        void export_tau_boundary(const Tensor&, int, int, Array<double>&, int&, const Array<int>&,
                                         int n_cmp = -1, Array<int>** p_cmp = nullptr) const override;
        void export_tau_val_domain_boundary(const Val_domain& eq, int mquant, int bound, Array<double>& res,
                                            int& pos_res, int ncond) const;
        void affecte_tau(Tensor&, int, const Array<double>&, int&) const override;
        void affecte_tau_val_domain(Val_domain& so, int mquant, const Array<double>& cf, int& pos_cf) const;
        void affecte_tau_one_coef(Tensor&, int, int, int&) const override;
        void affecte_tau_one_coef_val_domain(Val_domain& so, int mquant, int cc, int& pos_cf) const;

      public:
        ostream& print(ostream& o) const override;
    };

    class Space_polar : public Space
    {
      public:
        Space_polar(int ttype, const Point& cr, const Dim_array& nbr, const Array<double>& bounds);
        Space_polar(BinarySource&); ///< Modern API.
        ~Space_polar() override;
        void save(BinarySink&) const override; ///< Modern API.

        void add_inner_bc(System_of_eqs& syst, const char* eq, int nused = -1, Array<int>** pused = nullptr) const;

        void add_outer_bc(System_of_eqs& syst, const char* eq, int nused = -1, Array<int>** pused = nullptr) const;

        void add_eq(System_of_eqs& syst, const char* eq, int nused = -1, Array<int>** pused = nullptr) const;

        void add_eq_full(System_of_eqs& syst, const char* eq, int nused = -1, Array<int>** pused = nullptr) const;

        void add_matching(System_of_eqs& syst, const char* eq, int nused = -1, Array<int>** pused = nullptr) const;

        void add_eq(System_of_eqs& syst, const char* eq, const char* rac, const char* rac_der, int nused = -1,
                    Array<int>** pused = nullptr) const;

        void add_eq_int_volume(System_of_eqs& syst, const char* eq);

        void add_eq_int_inf(System_of_eqs& syst, const char* eq);

        void add_eq_int_inner(System_of_eqs& syst, const char* eq);

        void add_eq_mode(System_of_eqs& syst, const char* f, int domtarget, int itarget, int jtarget);

        void add_eq_ori(System_of_eqs& syst, const char* eq);

        void add_eq_point(System_of_eqs& syst, const Point& pp, const char* eq);

        double int_inf(const Scalar& so) const;
    };
} // namespace Kadath
