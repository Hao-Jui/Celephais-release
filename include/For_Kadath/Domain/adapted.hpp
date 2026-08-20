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
#include "For_Kadath/Term_eq/term_eq.hpp"
#include "spheric.hpp"
#include "For_Kadath/Metric/metric.hpp"
#include <vector>

namespace Kadath
{

    class Domain_shell_inner_adapted : public Domain
    {

      protected:
        const Space& sp;
        Val_domain* inner_radius;
        mutable Term_eq* inner_radius_term_eq;
        double outer_radius;

        mutable Term_eq* normal_spher;

        mutable Term_eq* normal_cart;

        mutable Term_eq* rad_term_eq;

        mutable Term_eq* der_rad_term_eq;

        mutable Term_eq* dt_rad_term_eq;

        mutable Term_eq* dp_rad_term_eq;

        Point center;

      public:
        Domain_shell_inner_adapted(const Space& sp, int num, int ttype, double rin, double rout, const Point& cr,
                                   const Dim_array& nbr);

        Domain_shell_inner_adapted(const Space& sp, int num, int ttype, const Val_domain& rin, double rout,
                                   const Point& cr, const Dim_array& nbr);
        Domain_shell_inner_adapted(const Domain_shell_inner_adapted& so);

        Domain_shell_inner_adapted(const Space& sp, int num, BinarySource& source); ///< Modern API.

        ~Domain_shell_inner_adapted() override;
        void del_deriv() const override;
        void save(BinarySink&) const override; ///< Modern API.
        double integ(const Val_domain&, int) const override;

        Val_domain get_inner_radius() const { return *inner_radius; };
        void snapshot_mapping(std::vector<Val_domain>& out) const override { out.push_back(*inner_radius); }
        bool fingerprint_import_shape(std::uint64_t& fingerprint) const override {
            return fingerprint_import_shape_component(*inner_radius, fingerprint);
        }
      private:
        void snapshot_mapping_into(std::vector<Val_domain>& out, std::size_t& idx) const override {
            snapshot_mapping_component(*inner_radius, out, idx);
        }
      public:
        void restore_mapping(const std::vector<Val_domain>& in, std::size_t& idx) const override { set_mapping(in[idx++]); }

      private:
        void do_absol() const override;
        void do_radius() const override;
        void do_cart() const override;
        void do_cart_surr() const override;

      protected:
        void set_cheb_base(Base_spectral&) const override;
        void set_legendre_base(Base_spectral&) const override;

        void set_anti_cheb_base(Base_spectral&) const override;
        void set_anti_legendre_base(Base_spectral&) const override;

        Tensor change_basis_cart_to_spher(int, const Tensor&) const override;
        Tensor change_basis_spher_to_cart(int, const Tensor&) const override;

        void set_cheb_base_r_spher(Base_spectral&) const override;
        void set_cheb_base_t_spher(Base_spectral&) const override;
        void set_cheb_base_p_spher(Base_spectral&) const override;
        void set_legendre_base_r_spher(Base_spectral&) const override;
        void set_legendre_base_t_spher(Base_spectral&) const override;
        void set_legendre_base_p_spher(Base_spectral&) const override;

        void set_cheb_r_base(Base_spectral&) const override;
        void set_legendre_r_base(Base_spectral&) const override;

        void do_coloc() override;
        int give_place_var(char*) const override;

        int nbr_unknowns_from_adapted() const override;
        void vars_to_terms() const override;
        void affecte_coef(int&, int, bool&) const override;
        // un-hide the base double-bound overloads hidden by the const Val_domain& overrides below (-Woverloaded-virtual)
        using Domain::xx_to_vars_from_adapted;
        using Domain::update_variable;
        using Domain::update_constante;
        using Domain::update_mapping;
        void xx_to_vars_from_adapted(Val_domain&, const Array<double>&, int&) const override;
        void xx_to_ders_from_adapted(const Array<double>&, int&) const override;
        void update_term_eq(Term_eq*) const override;
        void update_variable(const Val_domain&, const Scalar&, Scalar&) const override;
        void update_constante(const Val_domain&, const Scalar&, Scalar&) const override;
        void update_mapping(const Val_domain&) override;

      public:
        void set_mapping(const Val_domain& so) const;

        void update() const;

      public:
        Point get_center() const override { return center; };
        bool is_in(const Point& xx, double prec = 1e-13) const override;
        const Point absol_to_num(const Point&) const override;

        const Point absol_to_num_bound(const Point&, int) const override;

        void do_der_abs_from_der_var(Val_domain** der_var, Val_domain** der_abs) const override;
        Base_spectral mult(const Base_spectral&, const Base_spectral&) const override;

      public:
        Val_domain mult_cos_phi(const Val_domain&) const override;
        Val_domain mult_sin_phi(const Val_domain&) const override;
        Val_domain mult_cos_theta(const Val_domain&) const override;
        Val_domain mult_sin_theta(const Val_domain&) const override;
        Val_domain div_sin_theta(const Val_domain&) const override;
        Val_domain div_cos_theta(const Val_domain&) const override;
        Val_domain ddp(const Val_domain&) const override;
        Val_domain der_r(const Val_domain&) const override;
        Val_domain div_r(const Val_domain&) const override;
        Val_domain laplacian(const Val_domain&, int) const override;
        Val_domain laplacian2(const Val_domain&, int) const override;

        double val_boundary(int, const Val_domain&, const Index&) const override;
        void find_other_dom(int, int, int&, int&) const override;
        Val_domain der_normal(const Val_domain&, int) const override;

        int nbr_unknowns(const Tensor&, int) const override;

        int nbr_unknowns_val_domain(const Val_domain& so, int mlim) const;
        Array<int> nbr_conditions(const Tensor&, int, int, int n_cmp = -1, Array<int>** p_cmp = nullptr) const override;
        bool describe_volume_residual_rows(
            const Tensor&, int, int, const Array<int>&, int, Array<int>**,
            std::vector<ResidualRowDescriptor>&) const override;

        int nbr_conditions_val_domain(const Val_domain& so, int mlim, int order) const;
        Array<int> nbr_conditions_boundary(const Tensor&, int, int, int n_cmp = -1,
                                                   Array<int>** p_cmp = nullptr) const override;
        bool describe_boundary_residual_rows(
            const Tensor&, int, int, const Array<int>&, int, Array<int>**,
            std::vector<ResidualRowDescriptor>&) const override;

        int nbr_conditions_val_domain_boundary(const Val_domain& eq, int mlim) const;
        void export_tau(const Tensor&, int, int, Array<double>&, int&, const Array<int>&, int n_cmp = -1,
                                Array<int>** p_cmp = nullptr) const override;

        void export_tau_val_domain(const Val_domain& eq, int mlim, int order, Array<double>& res, int& pos_res,
                                   int ncond) const;
        void export_tau_boundary(const Tensor&, int, int, Array<double>&, int&, const Array<int>&,
                                         int n_cmp = -1, Array<int>** p_cmp = nullptr) const override;

        void export_tau_val_domain_boundary(const Val_domain& eq, int mlim, int bound, Array<double>& res, int& pos_res,
                                            int ncond) const;
        void affecte_tau(Tensor&, int, const Array<double>&, int&) const override;

        void affecte_tau_val_domain(Val_domain& so, int mlim, const Array<double>& cf, int& pos_cf) const;
        void affecte_tau_one_coef(Tensor&, int, int, int&) const override;
        bool describe_tau_seed_block(const Tensor&, int,
                                     std::vector<TauSeedDescriptor>&) const override;

        void affecte_tau_one_coef_val_domain(Val_domain& so, int mlim, int cc, int& pos_cf) const;

        int nbr_points_boundary(int, const Base_spectral&) const override;
        void do_which_points_boundary(int, const Base_spectral&, Index**, int) const override;

        Term_eq flat_grad_spher(const Term_eq&) const;

        Term_eq partial_spher(const Term_eq&) const override;
        Term_eq partial_cart(const Term_eq&) const override;
        Term_eq connection_spher(const Term_eq&) const override;
        const Term_eq* give_normal(int, int) const override;

        Term_eq derive_r(const Term_eq& so) const;

        Term_eq derive_t(const Term_eq& so) const;

        Term_eq derive_p(const Term_eq& so) const;

        void do_normal_spher() const;

        void do_normal_cart() const;

        Term_eq der_normal_term_eq(const Term_eq&, int) const override;
        Term_eq dr_term_eq(const Term_eq&) const override;
        Term_eq lap_term_eq(const Term_eq&, int) const override;
        Term_eq mult_r_term_eq(const Term_eq&) const override;
        Term_eq integ_volume_term_eq(const Term_eq&) const override;

        Term_eq derive_flat_spher(int, char, const Term_eq&, const Metric*) const override;
        Term_eq derive_flat_cart(int, char, const Term_eq&, const Metric*) const override;

        Tensor import(int, int, int, const Array<int>&, Tensor**) const override;

        double integ_volume(const Val_domain&) const override;

      public:
        ostream& print(ostream& o) const override;

        friend class Space_spheric_adapted;
        friend class Space_spheric_homothetic;
        friend class Space_bin_ns;
        friend class Space_bhns;
        friend class Space_bin_bh;
        friend class Space_adapted_bh;
        friend class Space_KerrSchild_bh;
        friend class Space_bbh;
        friend class Space_Kerr_bbh;
    };

    class Domain_shell_outer_adapted : public Domain
    {

      protected:
        const Space& sp;
        Val_domain* outer_radius;
        mutable Term_eq* outer_radius_term_eq;
        double inner_radius;

        mutable Term_eq* normal_spher;

        mutable Term_eq* normal_cart;

        mutable Term_eq* rad_term_eq;

        mutable Term_eq* der_rad_term_eq;

        mutable Term_eq* dt_rad_term_eq;

        mutable Term_eq* dp_rad_term_eq;

        Point center;

      public:
        Domain_shell_outer_adapted(const Space& sp, int num, int ttype, double rin, double rout, const Point& cr,
                                   const Dim_array& nbr);

        Domain_shell_outer_adapted(const Space& sp, int num, int ttype, double rin, const Val_domain& rout,
                                   const Point& cr, const Dim_array& nbr);
        Domain_shell_outer_adapted(const Domain_shell_outer_adapted& so);

        Domain_shell_outer_adapted(const Space& sp, int num, BinarySource& source); ///< Modern API.

        ~Domain_shell_outer_adapted() override;
        void del_deriv() const override;
        void save(BinarySink&) const override; ///< Modern API.

        Val_domain get_outer_radius() const { return *outer_radius; };
        void snapshot_mapping(std::vector<Val_domain>& out) const override { out.push_back(*outer_radius); }
        bool fingerprint_import_shape(std::uint64_t& fingerprint) const override {
            return fingerprint_import_shape_component(*outer_radius, fingerprint);
        }
      private:
        void snapshot_mapping_into(std::vector<Val_domain>& out, std::size_t& idx) const override {
            snapshot_mapping_component(*outer_radius, out, idx);
        }
      public:
        void restore_mapping(const std::vector<Val_domain>& in, std::size_t& idx) const override { set_mapping(in[idx++]); }

      private:
        void do_absol() const override;
        void do_radius() const override;
        void do_cart() const override;
        void do_cart_surr() const override;

      protected:
        void set_cheb_base(Base_spectral&) const override;
        void set_legendre_base(Base_spectral&) const override;

        void set_anti_cheb_base(Base_spectral&) const override;
        void set_anti_legendre_base(Base_spectral&) const override;

        void set_cheb_base_r_spher(Base_spectral&) const override;
        void set_cheb_base_t_spher(Base_spectral&) const override;
        void set_cheb_base_p_spher(Base_spectral&) const override;
        void set_legendre_base_r_spher(Base_spectral&) const override;
        void set_legendre_base_t_spher(Base_spectral&) const override;
        void set_legendre_base_p_spher(Base_spectral&) const override;

        void set_cheb_r_base(Base_spectral&) const override;
        void set_legendre_r_base(Base_spectral&) const override;

        void do_coloc() override;
        int give_place_var(char*) const override;

        int nbr_unknowns_from_adapted() const override;
        void vars_to_terms() const override;
        void affecte_coef(int&, int, bool&) const override;
        // un-hide the base double-bound overloads hidden by the const Val_domain& overrides below (-Woverloaded-virtual)
        using Domain::xx_to_vars_from_adapted;
        using Domain::update_variable;
        using Domain::update_constante;
        using Domain::update_mapping;
        void xx_to_vars_from_adapted(Val_domain&, const Array<double>&, int&) const override;
        void xx_to_ders_from_adapted(const Array<double>&, int&) const override;
        void update_term_eq(Term_eq*) const override;
        void update_variable(const Val_domain&, const Scalar&, Scalar&) const override;
        void update_constante(const Val_domain&, const Scalar&, Scalar&) const override;
        void update_mapping(const Val_domain&) override;

      public:
        void set_mapping(const Val_domain& so) const;

        void update() const;

      public:
        Point get_center() const override { return center; };

        bool is_in(const Point& xx, double prec = 1e-13) const override;
        const Point absol_to_num(const Point&) const override;
        const Point absol_to_num_bound(const Point&, int) const override;
        void do_der_abs_from_der_var(Val_domain** der_var, Val_domain** der_abs) const override;
        Base_spectral mult(const Base_spectral&, const Base_spectral&) const override;

      public:
        Val_domain mult_cos_phi(const Val_domain&) const override;
        Val_domain mult_sin_phi(const Val_domain&) const override;
        Val_domain mult_cos_theta(const Val_domain&) const override;
        Val_domain mult_sin_theta(const Val_domain&) const override;
        Val_domain div_sin_theta(const Val_domain&) const override;
        Val_domain div_cos_theta(const Val_domain&) const override;
        Val_domain laplacian(const Val_domain&, int) const override;
        Val_domain laplacian2(const Val_domain&, int) const override;

        Tensor change_basis_cart_to_spher(int dd, const Tensor&) const override;
        Tensor change_basis_spher_to_cart(int dd, const Tensor&) const override;

        Val_domain ddp(const Val_domain&) const override;
        Val_domain der_r(const Val_domain&) const override;
        Val_domain div_r(const Val_domain&) const override;

        double val_boundary(int, const Val_domain&, const Index&) const override;
        void find_other_dom(int, int, int&, int&) const override;
        Val_domain der_normal(const Val_domain&, int) const override;

        int nbr_unknowns(const Tensor&, int) const override;

        int nbr_unknowns_val_domain(const Val_domain& so, int mlim) const;
        Array<int> nbr_conditions(const Tensor&, int, int, int n_cmp = -1, Array<int>** p_cmp = nullptr) const override;
        bool describe_volume_residual_rows(
            const Tensor&, int, int, const Array<int>&, int, Array<int>**,
            std::vector<ResidualRowDescriptor>&) const override;

        int nbr_conditions_val_domain(const Val_domain& so, int mlim, int order) const;
        Array<int> nbr_conditions_boundary(const Tensor&, int, int, int n_cmp = -1,
                                                   Array<int>** p_cmp = nullptr) const override;
        bool describe_boundary_residual_rows(
            const Tensor&, int, int, const Array<int>&, int, Array<int>**,
            std::vector<ResidualRowDescriptor>&) const override;

        int nbr_conditions_val_domain_boundary(const Val_domain& eq, int mlim) const;
        void export_tau(const Tensor&, int, int, Array<double>&, int&, const Array<int>&, int n_cmp = -1,
                                Array<int>** p_cmp = nullptr) const override;

        void export_tau_val_domain(const Val_domain& eq, int mlim, int order, Array<double>& res, int& pos_res,
                                   int ncond) const;
        void export_tau_boundary(const Tensor&, int, int, Array<double>&, int&, const Array<int>&,
                                         int n_cmp = -1, Array<int>** p_cmp = nullptr) const override;

        void export_tau_val_domain_boundary(const Val_domain& eq, int mlim, int bound, Array<double>& res, int& pos_res,
                                            int ncond) const;
        void affecte_tau(Tensor&, int, const Array<double>&, int&) const override;

        void affecte_tau_val_domain(Val_domain& so, int mlim, const Array<double>& cf, int& pos_cf) const;
        void affecte_tau_one_coef(Tensor&, int, int, int&) const override;
        bool describe_tau_seed_block(const Tensor&, int,
                                     std::vector<TauSeedDescriptor>&) const override;

        void affecte_tau_one_coef_val_domain(Val_domain& so, int mlim, int cf, int& pos_cf) const;

        int nbr_points_boundary(int, const Base_spectral&) const override;
        void do_which_points_boundary(int, const Base_spectral&, Index**, int) const override;

        Term_eq flat_grad_spher(const Term_eq&) const;
        Term_eq partial_spher(const Term_eq&) const override;
        Term_eq partial_cart(const Term_eq&) const override;
        Term_eq connection_spher(const Term_eq&) const override;
        const Term_eq* give_normal(int, int) const override;

        Term_eq derive_r(const Term_eq& so) const;

        Term_eq derive_t(const Term_eq& so) const;

        Term_eq derive_p(const Term_eq& so) const;

        void do_normal_spher() const;

        void do_normal_cart() const;

        Term_eq der_normal_term_eq(const Term_eq&, int) const override;
        Term_eq dr_term_eq(const Term_eq&) const override;
        Term_eq lap_term_eq(const Term_eq&, int) const override;
        Term_eq mult_r_term_eq(const Term_eq&) const override;
        Term_eq integ_volume_term_eq(const Term_eq&) const override;
        Term_eq integ_term_eq(const Term_eq&, int) const override;
        Term_eq derive_flat_spher(int, char, const Term_eq&, const Metric*) const override;
        Term_eq derive_flat_cart(int, char, const Term_eq&, const Metric*) const override;

        double integ_volume(const Val_domain&) const override;
        Tensor import(int, int, int, const Array<int>&, Tensor**) const override;

      public:
        ostream& print(ostream& o) const override;

        friend class Space_spheric_adapted;
        friend class Space_spheric_homothetic;
        friend class Space_bin_ns;
        friend class Space_bhns;
        friend class Space_bin_bh;
        friend class Space_adapted_bh;
        friend class Space_bbh;
        friend class Space_KerrSchild_bh;
        friend class Space_Kerr_bbh;
    };

    class Space_spheric_adapted : public Space
    {
      public:
        Space_spheric_adapted(int ttype, const Point& cr, const Dim_array& nbr, const Array<double>& bounds);
        Space_spheric_adapted(int ttype, const Point& cr, const Dim_array& nbr, const std::vector<double>& bounds);
        Space_spheric_adapted(BinarySource&); ///< Modern API.
        ~Space_spheric_adapted() override;
        void save(BinarySink&) const override; ///< Modern API.

        int nbr_unknowns_from_variable_domains() const override;
        void affecte_coef_to_variable_domains(int&, int, Array<int>&) const override;
        void xx_to_ders_variable_domains(const Array<double>&, int&) const override;
        void xx_to_vars_variable_domains(System_of_eqs*, const Array<double>&, int&) const override;

        void add_eq_ori(System_of_eqs& syst, const char* eq);

        void add_eq(System_of_eqs& syst, const char* eq, const char* rac, const char* rac_der, int nused = -1,
                    Array<int>** pused = nullptr);

        void add_eq_int_inf(System_of_eqs& syst, const char* eq);

        void add_eq_int(System_of_eqs& sys, const int dom, const int bc, const char* eq);

        void add_eq_int_volume(System_of_eqs& syst, int nz, const char* eq);

        Array<int> get_indices_matching_non_std(int, int) const override;
    };
} // namespace Kadath
