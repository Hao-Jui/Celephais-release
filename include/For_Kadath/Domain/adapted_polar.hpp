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
 *   2026-06-17  Added snapshot_mapping / restore_mapping overrides to the polar
 *               adapted domains (inner, outer, bilateral) so the Newton
 *               line-search snapshot_state/restore_state primitive is bit-exact
 *               on Space_polar_adapted / Space_polar_bilateral_adapted. See
 *               LICENSE_SOURCE_AUDIT.tsv and PATCHES-KADATH-UPSTREAM.md.
 */

#pragma once

#include <vector>
#include "For_Kadath/Space/space.hpp"
#include "For_Kadath/Term_eq/term_eq.hpp"
#include "polar.hpp"
#include "For_Kadath/Val_domain/val_domain.hpp"

namespace Kadath
{

    class Domain_polar_shell_inner_adapted : public Domain
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

        Point center;

      public:
        Domain_polar_shell_inner_adapted(const Space& sp, int num, int ttype, double rin, double rout, const Point& cr,
                                         const Dim_array& nbr);
        Domain_polar_shell_inner_adapted(const Space& sp, int num, int ttype, const Val_domain& rin, double rout,
                                         const Point& cr, const Dim_array& nbr);
        Domain_polar_shell_inner_adapted(const Domain_polar_shell_inner_adapted& so);
        Domain_polar_shell_inner_adapted(const Space& sp, const Domain_polar_shell_inner_adapted& so);
        Domain_polar_shell_inner_adapted(const Space& sp, int num, BinarySource& source); ///< Modern API.

        ~Domain_polar_shell_inner_adapted() override;
        void save(BinarySink&) const override; ///< Modern API.
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
        void del_deriv() const override;

      protected:
        void do_absol() const override;
        void do_radius() const override;
        void do_cart() const override;
        void do_cart_surr() const override;

      protected:
        void set_cheb_base(Base_spectral&) const override;
        void set_legendre_base(Base_spectral&) const override;
        void set_anti_cheb_base(Base_spectral&) const override;
        void set_anti_legendre_base(Base_spectral&) const override;
        void set_cheb_base_with_m(Base_spectral&, int m) const override;
        void set_legendre_base_with_m(Base_spectral&, int m) const override;
        void set_anti_cheb_base_with_m(Base_spectral&, int m) const override;
        void set_anti_legendre_base_with_m(Base_spectral&, int m) const override;
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
        double integrale(const Val_domain&) const override;
        double integ_volume(const Val_domain&) const override;
        double integ(const Val_domain& so, int bound) const override;

        double val_boundary(int, const Val_domain&, const Index&) const override;
        void find_other_dom(int, int, int&, int&) const override;
        Val_domain der_normal(const Val_domain&, int) const override;

        Term_eq flat_grad_spher(const Term_eq& so) const;

        Term_eq partial_spher(const Term_eq&) const override;
        Term_eq partial_cart(const Term_eq&) const override;
        const Term_eq* give_normal(int, int) const override;
        Term_eq grad_term_eq(const Term_eq&) const override;

        Term_eq derive_r(const Term_eq& so) const;
        Term_eq derive_t(const Term_eq& so) const;
        void do_normal_spher() const;
        void do_normal_cart() const;

        Term_eq der_normal_term_eq(const Term_eq&, int) const override;
        Term_eq dr_term_eq(const Term_eq&) const override;
        Term_eq lap_term_eq(const Term_eq&, int) const override;
        Term_eq lap2_term_eq(const Term_eq&, int) const override;
        Term_eq mult_r_term_eq(const Term_eq&) const override;
        Term_eq div_r_term_eq(const Term_eq&) const override;
        Term_eq integ_term_eq(const Term_eq&, int) const override;

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
        friend class Space_polar_adapted;
        friend class Space_adapted_bh_polar;
    };

    class Domain_polar_shell_outer_adapted : public Domain
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

        Point center;

      public:
        Domain_polar_shell_outer_adapted(const Space& sp, int num, int ttype, double rin, double rout, const Point& cr,
                                         const Dim_array& nbr);
        Domain_polar_shell_outer_adapted(const Space& sp, int num, int ttype, double rin, const Val_domain& rout,
                                         const Point& cr, const Dim_array& nbr);
        Domain_polar_shell_outer_adapted(const Domain_polar_shell_outer_adapted& so);
        Domain_polar_shell_outer_adapted(const Space& sp, const Domain_polar_shell_outer_adapted& so);
        Domain_polar_shell_outer_adapted(const Space& sp, int num, BinarySource& source); ///< Modern API.

        ~Domain_polar_shell_outer_adapted() override;

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

      protected:
        void do_absol() const override;
        void do_radius() const override;
        void do_cart() const override;
        void do_cart_surr() const override;

      protected:
        void set_cheb_base(Base_spectral&) const override;
        void set_legendre_base(Base_spectral&) const override;
        void set_anti_cheb_base(Base_spectral&) const override;
        void set_anti_legendre_base(Base_spectral&) const override;
        void set_cheb_base_with_m(Base_spectral&, int m) const override;
        void set_legendre_base_with_m(Base_spectral&, int m) const override;
        void set_anti_cheb_base_with_m(Base_spectral&, int m) const override;
        void set_anti_legendre_base_with_m(Base_spectral&, int m) const override;
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
        double integrale(const Val_domain&) const override;
        double integ_volume(const Val_domain&) const override;

        Term_eq flat_grad_spher(const Term_eq&) const;

        Term_eq partial_spher(const Term_eq&) const override;
        Term_eq partial_cart(const Term_eq&) const override;
        const Term_eq* give_normal(int, int) const override;
        Term_eq grad_term_eq(const Term_eq&) const override;
        Term_eq integ_term_eq(const Term_eq&, int) const override;
        Term_eq derive_r(const Term_eq& so) const;
        Term_eq derive_t(const Term_eq& so) const;
        void do_normal_spher() const;
        void do_normal_cart() const;

        Term_eq der_normal_term_eq(const Term_eq&, int) const override;
        Term_eq dr_term_eq(const Term_eq&) const override;
        Term_eq lap_term_eq(const Term_eq&, int) const override;
        Term_eq lap2_term_eq(const Term_eq&, int) const override;
        Term_eq mult_r_term_eq(const Term_eq&) const override;
        Term_eq div_r_term_eq(const Term_eq&) const override;

        double val_boundary(int, const Val_domain&, const Index&) const override;
        void find_other_dom(int, int, int&, int&) const override;
        Val_domain der_normal(const Val_domain&, int) const override;

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
        void affecte_tau_one_coef_val_domain(Val_domain& so, int mquant, int cf, int& pos_cf) const;

      public:
        ostream& print(ostream& o) const override;

        friend class Space_polar_adapted;
        friend class Space_adapted_bh_polar;
    };

    class Domain_polar_shell_bilateral_adapted : public Domain
    {

      private:
        const Space& sp;
        Val_domain* inner_radius;
        Val_domain* outer_radius;
        mutable Term_eq* inner_radius_term_eq;
        mutable Term_eq* outer_radius_term_eq;
        mutable Term_eq* normal_spher;
        mutable Term_eq* normal_cart;
        mutable Term_eq* rad_term_eq;
        mutable Term_eq* der_rad_term_eq;
        mutable Term_eq* dt_rad_term_eq;

        Point center;

      public:
        Domain_polar_shell_bilateral_adapted(const Space& sp, int num, int ttype, double rin, double rout,
                                             const Point& cr, const Dim_array& nbr);
        Domain_polar_shell_bilateral_adapted(const Space& sp, int num, int ttype, const Val_domain& rin,
                                             const Val_domain& rout, const Point& cr, const Dim_array& nbr);
        Domain_polar_shell_bilateral_adapted(const Domain_polar_shell_bilateral_adapted& so);
        Domain_polar_shell_bilateral_adapted(const Space& sp, const Domain_polar_shell_bilateral_adapted& so);
        Domain_polar_shell_bilateral_adapted(const Space& sp, int num, BinarySource& source); ///< Modern API.

        ~Domain_polar_shell_bilateral_adapted() override;

        void del_deriv() const override;

        void save(BinarySink&) const override; ///< Modern API.
        Val_domain get_inner_radius() const { return *inner_radius; };
        Val_domain get_outer_radius() const { return *outer_radius; };
        void snapshot_mapping(std::vector<Val_domain>& out) const override {
            out.push_back(*inner_radius);
            out.push_back(*outer_radius);
        }
        bool fingerprint_import_shape(std::uint64_t& fingerprint) const override {
            return fingerprint_import_shape_component(*inner_radius, fingerprint) &&
                   fingerprint_import_shape_component(*outer_radius, fingerprint);
        }
      private:
        void snapshot_mapping_into(std::vector<Val_domain>& out, std::size_t& idx) const override {
            snapshot_mapping_component(*inner_radius, out, idx);
            snapshot_mapping_component(*outer_radius, out, idx);
        }
      public:
        // Bilateral cannot delegate to the 1-arg set_mapping (it assigns only
        // *outer_radius). Restore both radii absolutely, then reproduce
        // set_mapping's cache-invalidation + vars_to_terms (bit-identical body).
        void restore_mapping(const std::vector<Val_domain>& in, std::size_t& idx) const override {
            *inner_radius = in[idx++];
            *outer_radius = in[idx++];
            clear_import_plan_cache();
            for (int l = 0; l < ndim; l++) {
                if (absol[l] != nullptr)
                    delete absol[l];
                if (cart[l] != nullptr)
                    delete cart[l];
                if (cart_surr[l] != nullptr)
                    delete cart_surr[l];
                absol[l] = nullptr;
                cart_surr[l] = nullptr;
                cart[l] = nullptr;
            }
            if (radius != nullptr)
                delete radius;
            radius = nullptr;
            vars_to_terms();
        }

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

        // Bilateral domain intentionally exposes only the 2-arg public interface;
        // 1-arg base virtuals remain accessible via base-class polymorphism.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Woverloaded-virtual"
        void xx_to_vars_from_adapted(Val_domain& cor_inner, Val_domain& cor_outer, const Array<double>& xx,
                                     int& pos) const;
        void update_variable(const Val_domain& cor_inner, const Val_domain& cor_outer, const Scalar& old,
                             Scalar& res) const;
        void update_constante(const Val_domain& cor_inner, const Val_domain& cor_outer, const Scalar& old,
                              Scalar& res) const;
        void update_mapping(const Val_domain& cor_inner, const Val_domain& cor_outer);
#pragma GCC diagnostic pop

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
        double integrale(const Val_domain&) const override;
        double integ_volume(const Val_domain&) const override;

        Term_eq flat_grad_spher(const Term_eq&) const;

        Term_eq partial_spher(const Term_eq&) const override;
        Term_eq partial_cart(const Term_eq&) const override;
        const Term_eq* give_normal(int, int) const override;
        Term_eq grad_term_eq(const Term_eq&) const override;
        Term_eq integ_term_eq(const Term_eq&, int) const override;

        Term_eq derive_r(const Term_eq& so) const;
        Term_eq derive_t(const Term_eq& so) const;
        void do_normal_spher() const;
        void do_normal_cart() const;

        Term_eq der_normal_term_eq(const Term_eq&, int) const override;
        Term_eq dr_term_eq(const Term_eq&) const override;
        Term_eq lap_term_eq(const Term_eq&, int) const override;
        Term_eq lap2_term_eq(const Term_eq&, int) const override;
        Term_eq mult_r_term_eq(const Term_eq&) const override;
        Term_eq div_r_term_eq(const Term_eq&) const override;

        double val_boundary(int, const Val_domain&, const Index&) const override;
        void find_other_dom(int, int, int&, int&) const override;
        Val_domain der_normal(const Val_domain&, int) const override;

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
        void affecte_tau_one_coef_val_domain(Val_domain& so, int mquant, int cf, int& pos_cf) const;

      public:
        ostream& print(ostream& o) const override;

        friend class Space_polar_bilateral_adapted;
    };

    class Space_polar_adapted : public Space
    {
      private:
        int n_shells;

      public:
        int ADAPTED_OUTER;
        int ADAPTED_INNER;

        Space_polar_adapted(int ttype, const Point& cr, const Dim_array& nbr, const Array<double>& bounds);
        Space_polar_adapted(int ttype, const Point& cr, const Dim_array& nbr, const std::vector<double>& bounds);
        Space_polar_adapted(const Space_polar_adapted& sp);
        Space_polar_adapted(BinarySource&); ///< Modern API.
        ~Space_polar_adapted() override;
        void save(BinarySink&) const override; ///< Modern API.

        int nbr_unknowns_from_variable_domains() const override;
        void affecte_coef_to_variable_domains(int&, int, Array<int>&) const override;
        void xx_to_ders_variable_domains(const Array<double>&, int&) const override;
        void xx_to_vars_variable_domains(System_of_eqs*, const Array<double>&, int&) const override;

        void add_eq_ori(System_of_eqs& syst, const char* eq);
        void add_eq(System_of_eqs& syst, const char* eq, const char* rac, const char* rac_der, int nused = -1,
                    Array<int>** pused = nullptr) const;

        void add_eq_matter(System_of_eqs& syst, const char* eq, const char* rac, const char* rac_der, int nused = -1,
                           Array<int>** pused = nullptr) const;
        void add_eq_int_inf(System_of_eqs& sys, const char* nom);
        void add_eq_int_volume(System_of_eqs& syst, int nz, const char* eq);
        void add_eq_int_inner(System_of_eqs& sys, const char* nom);
    };

    class Space_polar_bilateral_adapted : public Space
    {
      private:
        int n_shells;

      public:
        int ADAPTED_BILATERAL;

        Space_polar_bilateral_adapted(int ttype, const Point& cr, const Dim_array& nbr, const Array<double>& bounds);
        Space_polar_bilateral_adapted(int ttype, const Point& cr, const Dim_array& nbr,
                                      const std::vector<double>& bounds);
        Space_polar_bilateral_adapted(const Space_polar_bilateral_adapted& sp);
        Space_polar_bilateral_adapted(BinarySource&); ///< Modern API.
        ~Space_polar_bilateral_adapted() override;
        void save(BinarySink&) const override; ///< Modern API.

        int nbr_unknowns_from_variable_domains() const override;
        void affecte_coef_to_variable_domains(int&, int, Array<int>&) const override;
        void xx_to_ders_variable_domains(const Array<double>&, int&) const override;
        void xx_to_vars_variable_domains(System_of_eqs*, const Array<double>&, int&) const override;

        void add_eq_ori(System_of_eqs& syst, const char* eq);
        void add_eq(System_of_eqs& syst, const char* eq, const char* rac, const char* rac_der, int nused = -1,
                    Array<int>** pused = nullptr) const;

        void add_eq_matter(System_of_eqs& syst, const char* eq, const char* rac, const char* rac_der, int nused = -1,
                           Array<int>** pused = nullptr) const;
        void add_eq_int_inf(System_of_eqs& sys, const char* nom);
        void add_eq_int_volume(System_of_eqs& syst, int nz, const char* eq);
    };
} // namespace Kadath
