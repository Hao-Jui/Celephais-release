/*
    Copyright 2024 Philippe Grandclement

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

#include "For_Kadath/Array/point.hpp"
#include "For_Kadath/Space/space.hpp"

namespace Kadath
{
    class Term_eq;
    class Metric;

    Array<double> mat_leg_even(int, int);
    Array<double> mat_inv_leg_even(int, int);
    Array<double> mat_leg_odd(int, int);
    Array<double> mat_inv_leg_odd(int, int);

    /**
     * Spherical nucleus domain with no plane symmetry (full 4π solid angle).
     * \li 3 dimensions, centered on \c center (X_c, Y_c, Z_c).
     * \li Numerical coordinates: 0 ≤ x ≤ 1, 0 ≤ θ★ ≤ π, 0 ≤ φ★ < 2π.
     * \li r = α x, θ = θ★, φ = φ★.
     * \ingroup domain
     */
    class Domain_nucleus_nosym : public Domain
    {
      private:
        double alpha; ///< Relates the numerical to the physical radii.
        Point center; ///< Absolute coordinates of the center.

      public:
        Domain_nucleus_nosym(int num, int ttype, double radius, const Point& cr, const Dim_array& nbr);
        Domain_nucleus_nosym(const Domain_nucleus_nosym& so);
        Domain_nucleus_nosym(int num, BinarySource& source); ///< Modern API.

        ~Domain_nucleus_nosym() override;
        void save(BinarySink&) const override; ///< Modern API.

      private:
        void do_absol() const override;
        void do_cart() const override;
        void do_cart_surr() const override;
        void do_radius() const override;

        void set_cheb_base(Base_spectral& so) const override;
        void set_legendre_base(Base_spectral& so) const override;
        void set_anti_cheb_base(Base_spectral& so) const override;
        void set_anti_legendre_base(Base_spectral& so) const override;

        void set_cheb_base_x_cart(Base_spectral&) const override;
        void set_cheb_base_y_cart(Base_spectral&) const override;
        void set_cheb_base_z_cart(Base_spectral&) const override;
        void set_legendre_base_x_cart(Base_spectral&) const override;
        void set_legendre_base_y_cart(Base_spectral&) const override;
        void set_legendre_base_z_cart(Base_spectral&) const override;

        void do_coloc() override;
        int give_place_var(char*) const override;

      public:
        double get_rmax() const override { return alpha; }
        Point get_center() const override { return center; }
        bool fingerprint_import_shape(std::uint64_t&) const override { return true; }

        bool is_in(const Point& xx, double prec = 1e-13) const override;
        const Point absol_to_num(const Point& xxx) const override;
        const Point absol_to_num_bound(const Point&, int) const override;

        void do_der_abs_from_der_var_lanes(DerAbsLaneBatch& batch) const override;
        void do_der_abs_from_der_var(Val_domain** der_var, Val_domain** der_abs) const override;
        Base_spectral mult(const Base_spectral&, const Base_spectral&) const override;

      public:
        Val_domain mult_cos_phi(const Val_domain&) const override;
        Val_domain mult_sin_phi(const Val_domain&) const override;
        Val_domain mult_cos_theta(const Val_domain&) const override;
        Val_domain mult_sin_theta(const Val_domain&) const override;
        Val_domain div_sin_theta(const Val_domain&) const override;

        Val_domain div_x(const Val_domain&) const override;
        Val_domain mult_r(const Val_domain&) const override;
        Val_domain div_r(const Val_domain&) const override;
        Val_domain der_r(const Val_domain&) const override;
        Val_domain ddp(const Val_domain&) const override;
        Val_domain srdr(const Val_domain&) const override;
        Val_domain div_1mx2(const Val_domain&) const override;
        Val_domain laplacian2(const Val_domain&, int) const override;
        double integ_volume(const Val_domain&) const override;

        double val_boundary(int, const Val_domain&, const Index&) const override;
        void find_other_dom(int, int, int&, int&) const override;
        Val_domain der_normal(const Val_domain&, int) const override;

        int nbr_unknowns(const Tensor&, int) const override;
        int nbr_unknowns_val_domain(const Val_domain& so) const;

        Array<int> nbr_conditions(const Tensor&, int, int, int n_cmp = -1,
                                  Array<int>** p_cmp = nullptr) const override;
        int nbr_conditions_val_domain(const Val_domain& so, int order) const;

        Array<int> nbr_conditions_boundary(const Tensor&, int, int, int n_cmp = -1,
                                           Array<int>** p_cmp = nullptr) const override;
        int nbr_conditions_val_domain_boundary(const Val_domain& eq, int mlim) const;

        void export_tau(const Tensor&, int, int, Array<double>&, int&, const Array<int>&, int n_cmp = -1,
                        Array<int>** p_cmp = nullptr) const override;
        bool describe_volume_residual_rows(
            const Tensor&, int, int, const Array<int>&, int, Array<int>**,
            std::vector<ResidualRowDescriptor>&) const override;
        void export_tau_val_domain(const Val_domain& eq, int order, Array<double>& res, int& pos_res, int ncond) const;

        bool describe_boundary_residual_rows(
            const Tensor&, int, int, const Array<int>&, int, Array<int>**,
            std::vector<ResidualRowDescriptor>&) const override;
        void export_tau_boundary(const Tensor&, int, int, Array<double>&, int&, const Array<int>&, int n_cmp = -1,
                                 Array<int>** p_cmp = nullptr) const override;
        void export_tau_val_domain_boundary(const Val_domain& eq, int mlim, int bound, Array<double>& res, int& pos_res,
                                            int ncond) const;

        void affecte_tau(Tensor&, int, const Array<double>&, int&) const override;
        void affecte_tau_val_domain(Val_domain& so, const Array<double>& cf, int& pos_cf) const;

        void affecte_tau_one_coef(Tensor&, int, int, int&) const override;
        bool describe_tau_seed_block(const Tensor&, int,
                                     std::vector<TauSeedDescriptor>&) const override;
        void affecte_tau_one_coef_val_domain(Val_domain& so, int cc, int& pos_cf) const;

        int nbr_points_boundary(int, const Base_spectral&) const override;
        void do_which_points_boundary(int, const Base_spectral&, Index**, int) const override;

        Term_eq derive_flat_cart(int, char, const Term_eq&, const Metric*) const override;
        Tensor import(int, int, int, const Array<int>&, Tensor**) const override;
        bool import_lanes_native(int, int, int, const Array<int>&, int, Tensor* const*, Tensor**) const override;

        Tensor change_basis_cart_to_spher(int dd, const Tensor&) const override;
        Tensor change_basis_spher_to_cart(int dd, const Tensor&) const override;

      public:
        ostream& print(ostream& o) const override;
    };

    /**
     * Spherical shell domain with no plane symmetry.
     * \li 3 dimensions, centered on \c center.
     * \li Numerical coordinates: -1 ≤ x ≤ 1, 0 ≤ θ★ ≤ π, 0 ≤ φ★ < 2π.
     * \li r = α x + β, θ = θ★, φ = φ★.
     * \ingroup domain
     */
    class Domain_shell_nosym : public Domain
    {
      private:
        double alpha; ///< Relates the numerical to the physical radii.
        double beta;  ///< Relates the numerical to the physical radii.
        Point center; ///< Absolute coordinates of the center.

      public:
        Domain_shell_nosym(int num, int ttype, double r_int, double r_ext, const Point& cr, const Dim_array& nbr);
        Domain_shell_nosym(const Domain_shell_nosym& so);
        Domain_shell_nosym(int num, BinarySource& source); ///< Modern API.

        ~Domain_shell_nosym() override;
        void save(BinarySink&) const override; ///< Modern API.

      private:
        void do_absol() const override;
        void do_radius() const override;
        void do_cart() const override;
        void do_cart_surr() const override;

      private:
        void set_cheb_base(Base_spectral& so) const override;
        void set_legendre_base(Base_spectral& so) const override;
        void set_anti_cheb_base(Base_spectral& so) const override;
        void set_anti_legendre_base(Base_spectral& so) const override;

        void set_cheb_base_x_cart(Base_spectral&) const override;
        void set_cheb_base_y_cart(Base_spectral&) const override;
        void set_cheb_base_z_cart(Base_spectral&) const override;
        void set_legendre_base_x_cart(Base_spectral&) const override;
        void set_legendre_base_y_cart(Base_spectral&) const override;
        void set_legendre_base_z_cart(Base_spectral&) const override;

        void do_coloc() override;
        int give_place_var(char*) const override;

      public:
        double get_rmin() const override { return beta - alpha; }
        double get_rmax() const override { return beta + alpha; }
        Point get_center() const override { return center; }

        bool is_in(const Point& xx, double prec = 1e-13) const override;
        const Point absol_to_num(const Point& xxx) const override;
        const Point absol_to_num_bound(const Point&, int) const override;

        void do_der_abs_from_der_var(Val_domain** der_var, Val_domain** der_abs) const override;
        Base_spectral mult(const Base_spectral&, const Base_spectral&) const override;

      public:
        bool fingerprint_import_shape(std::uint64_t&) const override { return true; }

        Val_domain mult_cos_phi(const Val_domain&) const override;
        Val_domain mult_sin_phi(const Val_domain&) const override;
        Val_domain mult_cos_theta(const Val_domain&) const override;
        Val_domain mult_sin_theta(const Val_domain&) const override;
        Val_domain div_sin_theta(const Val_domain&) const override;

        Val_domain div_xp1(const Val_domain&) const override;
        Val_domain div_1mrsL(const Val_domain&) const override;
        Val_domain mult_1mrsL(const Val_domain&) const override;

        Val_domain mult_r(const Val_domain&) const override;
        Val_domain div_r(const Val_domain&) const override;
        Val_domain der_r(const Val_domain&) const override;
        Val_domain der_partial_var(const Val_domain&, int) const override;
        Val_domain ddp(const Val_domain&) const override;
        Val_domain div_xm1(const Val_domain&) const override;
        Val_domain mult_xm1(const Val_domain&) const override;
        Val_domain div_1mx2(const Val_domain&) const override;
        Val_domain laplacian2(const Val_domain&, int) const override;
        Val_domain dt(const Val_domain&) const override;
        double integ_volume(const Val_domain&) const override;

        void find_other_dom(int, int, int&, int&) const override;
        Val_domain der_normal(const Val_domain&, int) const override;
        double val_boundary(int, const Val_domain&, const Index&) const override;

        int nbr_unknowns(const Tensor&, int) const override;
        int nbr_unknowns_val_domain(const Val_domain& so, int mlim) const;

        Array<int> nbr_conditions(const Tensor&, int, int, int n_cmp = -1,
                                  Array<int>** p_cmp = nullptr) const override;
        int nbr_conditions_val_domain(const Val_domain& so, int mlim, int order) const;

        Array<int> nbr_conditions_boundary(const Tensor&, int, int, int n_cmp = -1,
                                           Array<int>** p_cmp = nullptr) const override;
        int nbr_conditions_val_domain_boundary(const Val_domain& so, int mlim) const;

        void export_tau(const Tensor&, int, int, Array<double>&, int&, const Array<int>&, int n_cmp = -1,
                        Array<int>** p_cmp = nullptr) const override;
        bool describe_volume_residual_rows(
            const Tensor&, int, int, const Array<int>&, int, Array<int>**,
            std::vector<ResidualRowDescriptor>&) const override;
        void export_tau_val_domain(const Val_domain& eq, int mlim, int order, Array<double>& res, int& pos_res,
                                   int ncond) const;

        bool describe_boundary_residual_rows(
            const Tensor&, int, int, const Array<int>&, int, Array<int>**,
            std::vector<ResidualRowDescriptor>&) const override;
        void export_tau_boundary(const Tensor&, int, int, Array<double>&, int&, const Array<int>&, int n_cmp = -1,
                                 Array<int>** p_cmp = nullptr) const override;
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
        Tensor import(int, int, int, const Array<int>&, Tensor**) const override;
        bool import_lanes_native(int, int, int, const Array<int>&, int, Tensor* const*, Tensor**) const override;
        Tensor change_basis_cart_to_spher(int dd, const Tensor&) const override;
        Tensor change_basis_spher_to_cart(int dd, const Tensor&) const override;

        Term_eq derive_flat_cart(int, char, const Term_eq&, const Metric*) const override;

      public:
        ostream& print(ostream& o) const override;
    };

    /**
     * Spherical compactified domain with no plane symmetry.
     * \li 3 dimensions, centered on \c center.
     * \li Numerical coordinates: -1 ≤ x ≤ 1, 0 ≤ θ★ ≤ π, 0 ≤ φ★ < 2π.
     * \li r = 1 / (α x - 1).
     * \ingroup domain
     */
    class Domain_compact_nosym : public Domain
    {
      private:
        double alpha; ///< Relates the numerical to the physical radii.
        Point center; ///< Absolute coordinates of the center.

      public:
        Domain_compact_nosym(int num, int ttype, double r_int, const Point& cr, const Dim_array& nbr);
        Domain_compact_nosym(const Domain_compact_nosym& so);
        Domain_compact_nosym(int num, BinarySource& source); ///< Modern API.

        ~Domain_compact_nosym() override;
        void save(BinarySink&) const override; ///< Modern API.

      private:
        void do_absol() const override;
        void do_radius() const override;
        void do_cart() const override;
        void do_cart_surr() const override;

      private:
        void set_cheb_base(Base_spectral& so) const override;
        void set_legendre_base(Base_spectral& so) const override;
        void set_anti_cheb_base(Base_spectral& so) const override;
        void set_anti_legendre_base(Base_spectral& so) const override;

        void set_cheb_base_x_cart(Base_spectral&) const override;
        void set_cheb_base_y_cart(Base_spectral&) const override;
        void set_cheb_base_z_cart(Base_spectral&) const override;
        void set_legendre_base_x_cart(Base_spectral&) const override;
        void set_legendre_base_y_cart(Base_spectral&) const override;
        void set_legendre_base_z_cart(Base_spectral&) const override;

        void do_coloc() override;
        int give_place_var(char*) const override;

      public:
        Point get_center() const override { return center; }
        double get_alpha() const { return alpha; }
        bool fingerprint_import_shape(std::uint64_t&) const override { return true; }

        bool is_in(const Point& xx, double prec = 1e-13) const override;
        const Point absol_to_num(const Point& xxx) const override;
        const Point absol_to_num_bound(const Point&, int) const override;

        void do_der_abs_from_der_var_lanes(DerAbsLaneBatch& batch) const override;
        void do_der_abs_from_der_var(Val_domain** der_var, Val_domain** der_abs) const override;
        Base_spectral mult(const Base_spectral&, const Base_spectral&) const override;

      public:
        Val_domain mult_cos_phi(const Val_domain&) const override;
        Val_domain mult_sin_phi(const Val_domain&) const override;
        Val_domain mult_cos_theta(const Val_domain&) const override;
        Val_domain mult_sin_theta(const Val_domain&) const override;
        Val_domain div_sin_theta(const Val_domain&) const override;
        Val_domain div_cos_theta(const Val_domain&) const override;
        Val_domain div_xm1(const Val_domain&) const override;
        Val_domain mult_xm1(const Val_domain&) const override;
        Val_domain mult_r(const Val_domain&) const override;
        Val_domain div_r(const Val_domain&) const override;
        Val_domain der_r(const Val_domain&) const override;
        Val_domain der_r_rtwo(const Val_domain&) const override;
        Val_domain der_partial_var(const Val_domain&, int) const override;
        Val_domain laplacian2(const Val_domain&, int) const override;
        Val_domain dt(const Val_domain&) const override;
        Val_domain ddp(const Val_domain&) const override;

        void set_val_inf(Val_domain&, double) const override;

        void find_other_dom(int, int, int&, int&) const override;
        Val_domain der_normal(const Val_domain&, int) const override;
        double integ(const Val_domain&, int) const override;
        double integ_volume(const Val_domain&) const override;

        double val_boundary(int, const Val_domain&, const Index&) const override;
        int nbr_points_boundary(int, const Base_spectral&) const override;
        void do_which_points_boundary(int, const Base_spectral&, Index**, int) const override;

        int nbr_unknowns(const Tensor&, int) const override;
        int nbr_unknowns_val_domain(const Val_domain& so, int mlim) const;

        Array<int> nbr_conditions(const Tensor&, int, int, int n_cmp = -1,
                                  Array<int>** p_cmp = nullptr) const override;
        int nbr_conditions_val_domain(const Val_domain& eq, int mlim, int order) const;

        Array<int> nbr_conditions_boundary(const Tensor&, int, int, int n_cmp = -1,
                                           Array<int>** p_cmp = nullptr) const override;
        int nbr_conditions_val_domain_boundary(const Val_domain& eq, int mlim) const;

        void export_tau(const Tensor&, int, int, Array<double>&, int&, const Array<int>&, int n_cmp = -1,
                        Array<int>** p_cmp = nullptr) const override;
        bool describe_volume_residual_rows(
            const Tensor&, int, int, const Array<int>&, int, Array<int>**,
            std::vector<ResidualRowDescriptor>&) const override;
        void export_tau_val_domain(const Val_domain& eq, int mlim, int order, Array<double>& res, int& pos_res,
                                   int ncond) const;

        bool describe_boundary_residual_rows(
            const Tensor&, int, int, const Array<int>&, int, Array<int>**,
            std::vector<ResidualRowDescriptor>&) const override;
        void export_tau_boundary(const Tensor&, int, int, Array<double>&, int&, const Array<int>&, int n_cmp = -1,
                                 Array<int>** p_cmp = nullptr) const override;
        void export_tau_val_domain_boundary(const Val_domain& eq, int mlim, int bound, Array<double>& res, int& pos_res,
                                            int ncond) const;

        void affecte_tau(Tensor&, int, const Array<double>&, int&) const override;
        void affecte_tau_val_domain(Val_domain& so, int mlim, const Array<double>& cf, int& pos_cf) const;

        void affecte_tau_one_coef(Tensor&, int, int, int&) const override;
        bool describe_tau_seed_block(const Tensor&, int,
                                     std::vector<TauSeedDescriptor>&) const override;
        void affecte_tau_one_coef_val_domain(Val_domain& so, int mlim, int cc, int& pos_cf) const;

        Tensor import(int, int, int, const Array<int>&, Tensor**) const override;
        bool import_lanes_native(int, int, int, const Array<int>&, int, Tensor* const*, Tensor**) const override;
        Tensor change_basis_cart_to_spher(int dd, const Tensor&) const override;
        Tensor change_basis_spher_to_cart(int dd, const Tensor&) const override;

        Term_eq derive_flat_cart(int, char, const Term_eq&, const Metric*) const override;

      public:
        ostream& print(ostream& o) const override;
    };

    /**
     * The \c Space_spheric_nosym class fills the space with one nucleus, several shells and a compactified domain,
     * all centered on the same point. No plane symmetry is imposed (full 4π solid angle).
     * \ingroup domain
     */
    class Space_spheric_nosym : public Space
    {
      public:
        /**
         * Standard constructor.
         * @param ttype [input] : the type of basis.
         * @param cr [input] : absolute coordinates of the center.
         * @param nbr [input] : number of points in each domain.
         * @param bounds [input] : radii of the various shells (and also determines the total number of domains).
         * @param withzec [input] : use a compactified domain or not.
         */
        Space_spheric_nosym(int ttype, const Point& cr, const Dim_array& nbr, const Array<double>& bounds,
                            bool withzec = true);

        Space_spheric_nosym(BinarySource& source, bool withzec = true); ///< Modern API.
        ~Space_spheric_nosym() override;
        void save(BinarySink&) const override; ///< Modern API.

        Array<int> get_indices_matching_non_std(int, int) const override;
    };

} // namespace Kadath
