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

#pragma once

#define STD_TYPE 0
#define LOG_TYPE 1
#define SURR_TYPE 2

#include "For_Kadath/Space/space.hpp"
#include "spheric.hpp"
#include "For_Kadath/List_comp/list_comp.hpp"

namespace Kadath
{

#ifndef KADATH_SAFE_DELETE_TEMPLATE_DEFINED
#define KADATH_SAFE_DELETE_TEMPLATE_DEFINED
    template <typename T> inline void safe_delete(T*& p)
    {
        if (p) {
            delete p;
            p = nullptr;
        }
    }
#endif

    /**
     * Class for bispherical coordinates with a symmetry with respect to the plane \f$ z=0 \f$.
     * This class implements a domain where \f$ \chi \f$ and \f$ \eta \f$ are bounded on a rectange (see below).
     * \li 3 dimensions.
     * \li The numerical coordinates are :
     *
     * \f$ -1 \leq \eta^\star \leq 1 \f$
     *
     * \f$ 0 \leq \chi^\star \leq 1 \f$
     *
     * \f$ 0 \leq \varphi^\star < 2\pi \f$
     *
     * \li Standard bispherical coordinates :
     *
     * \f$ \eta = \displaystyle\frac{\eta_{\rm max}-\eta_{\rm min}}{2} \eta^\star
     * +  \displaystyle\frac{\eta_{\rm max}+\eta_{\rm min}}{2} \f$
     *
     * \f$ \chi = (\chi_{\rm min}- \pi) \chi^\star + \pi \f$
     *
     * \f$ \varphi = \varphi^\star \f$
     *
     * \li Standard associated Cartesian coordinates are :
     *
     * \f$ x = a \displaystyle\frac{\sinh \eta}{\cosh\eta - \cos\chi} \f$
     *
     * \f$ y = a \displaystyle\frac{\cosh \eta \cos\varphi}{\cosh\eta - \cos\chi} \f$
     *
     * \f$ z = a \displaystyle\frac{\cosh \eta \sin\varphi}{\cosh\eta - \cos\chi} \f$
     * \ingroup domain
     */
    class Domain_bispheric_rect : public Domain
    {

      private:
        double aa;        ///< Distance scale \f$ a \f$.
        double r_ext;     ///< Radius \f$ R \f$ of the outer boundary.
        double eta_minus; ///< \f$ \eta \f$ associated with \f$ x=-1 \f$ .
        double eta_plus;  ///<  \f$ \eta \f$ associated  with \f$ x=+1 \f$ .
        double chi_min;   ///< Lower bound for \f$ \chi \f$.

        mutable Val_domain* p_eta; ///< Pointer on a \c Val_domain containing \f$ \eta \f$.
        mutable Val_domain* p_chi; ///< Pointer on a \c Val_domain containing \f$ \chi \f$.
        mutable Val_domain* p_phi; ///< Pointer on a \c Val_domain containing \f$ \varphi \f$.

        // For derivatives
        /**
         * Pointer on a \c Val_domain containing \f$ \displaystyle\frac{\partial \eta^\star}{\partial x} \f$
         * The explicit expression are :
         * \f$ \displaystyle\frac{\partial \eta}{\partial x} =
         * \displaystyle\frac{2 a (a^2 + r^2 - 2x^2)}{(a^2+r^2)^2 - 4a^2x^2} \f$
         *
         * and
         * \f$ \displaystyle\frac{\partial \eta^\star}{\partial \eta } =
         * \displaystyle\frac{2}{\eta_{\rm max} - \eta_{\rm min}} \f$
         */
        mutable Val_domain* p_detadx;
        /**
         * Pointer on a \c Val_domain containing \f$ \displaystyle\frac{\partial \eta^\star}{\partial y} \f$
         * The explicit expression are :
         * \f$ \displaystyle\frac{\partial \eta}{\partial y} =
         * \displaystyle\frac{-4axy}{(a^2+r^2)^2 - 4a^2x^2} \f$
         *
         * and
         * \f$ \displaystyle\frac{\partial \eta^\star}{\partial \eta } =
         * \displaystyle\frac{2}{\eta_{\rm max} - \eta_{\rm min}} \f$
         */
        mutable Val_domain* p_detady;
        /**
         * Pointer on a \c Val_domain containing \f$ \displaystyle\frac{\partial \eta^\star}{\partial z} \f$
         * The explicit expression are :
         * \f$ \displaystyle\frac{\partial \eta}{\partial z} =
         * \displaystyle\frac{-4axz}{(a^2+r^2)^2 - 4a^2x^2} \f$
         *
         * and
         * \f$ \displaystyle\frac{\partial \eta^\star}{\partial \eta } =
         * \displaystyle\frac{2}{\eta_{\rm max} - \eta_{\rm min}} \f$
         */
        mutable Val_domain* p_detadz;
        /**
         * Pointer on a \c Val_domain containing \f$ \displaystyle\frac{\partial \chi^\star}{\partial x} \f$
         * The explicit expression are :
         * \f$ \displaystyle\frac{\partial \chi}{\partial x} =
         * \displaystyle\frac{-4ax\sqrt{y^2+z^2}}{(r^2-a^2)^2 + 4 a^2 (y^2+z^2)} \f$
         *
         * and
         * \f$ \displaystyle\frac{\partial \chi^\star}{\partial \chi } =
         * \displaystyle\frac{1}{\chi_{\rm min} - \pi} \f$
         */
        mutable Val_domain* p_dchidx;
        /**
         * Pointer on a \c Val_domain containing \f$ \displaystyle\frac{\partial \chi^\star}{\partial y} \f$
         * The explicit expression are :
         * \f$ \displaystyle\frac{\partial \chi}{\partial y} = \cos\varphi
         * \displaystyle\frac{2ax(r^2-a^2-a^2(y^2+z^2)}{(r^2-a^2)^2 + 4 a^2 (y^2+z^2)} \f$
         *
         * and
         * \f$ \displaystyle\frac{\partial \chi^\star}{\partial \chi } =
         * \displaystyle\frac{1}{\chi_{\rm min} - \pi} \f$
         */
        mutable Val_domain* p_dchidy;
        /**
         * Pointer on a \c Val_domain containing \f$ \displaystyle\frac{\partial \chi^\star}{\partial y} \f$
         * The explicit expression are :
         * \f$ \displaystyle\frac{\partial \chi}{\partial z} = \sin\varphi
         * \displaystyle\frac{2ax(r^2-a^2-a^2(y^2+z^2)}{(r^2-a^2)^2 + 4 a^2 (y^2+z^2)} \f$
         *
         * and
         * \f$ \displaystyle\frac{\partial \chi^\star}{\partial \chi } =
         * \displaystyle\frac{1}{\chi_{\rm min} - \pi} \f$
         */
        mutable Val_domain* p_dchidz;
        /**
         * Pointer on a \c Val_domain containing \f$ \sin\chi \displaystyle\frac{\partial \varphi^\star}{\partial y} \f$
         * The explicit expression is :
         * \f$ \sin\chi \displaystyle\frac{\partial \varphi^\star}{\partial y} =
         * - \displaystyle\frac{(\cosh\eta - \cos\chi) \sin\varphi}{a} \f$
         */
        mutable Val_domain* p_dphidy;
        /**
         * Pointer on a \c Val_domain containing \f$ \sin\chi \displaystyle\frac{\partial \varphi^\star}{\partial z} \f$
         * The explicit expression is :
         * \f$ \sin\chi \displaystyle\frac{\partial \varphi^\star}{\partial z} =
         * \displaystyle\frac{(\cosh\eta - \cos\chi) \cos\varphi}{a} \f$
         */
        mutable Val_domain* p_dphidz;
        /**
         * Pointer on a \c Val_domain containing the surface element on the inner boundary of the domain (being
         * spherical)
         */
        mutable Val_domain* p_dsint;

      public:
        /**
         * Standard constructor :
         * @param num [input] : number of the domain (used by the \c Space).
         * @param ttype [input] : Chebyshev or Legendre type of spectral expansion.
         * @param aa [input] : scale factor \f$ a \f$.
         * @param eta_minus [input] : \f$ \eta_{\rm min} \f$ associated with \f$ x = -1 \f$.
         * @param eta_plus [input] : \f$ \eta_{\rm max} \f$ associated with \f$ x = +1 \f$.
         * @param chi_min [input] : lower bound \f$ \chi_{\rm min} \f$.
         * @param nbr [input] : number of points in each dimension.
         */

        Domain_bispheric_rect(int nd, int ttype, double aa, double rext, double eta_minus, double eta_plus,
                              double chi_min, const Dim_array& nbr);
        Domain_bispheric_rect(const Domain_bispheric_rect& so); ///< Copy constructor.
                                                                /**
                                                                 * Constructor from a file
                                                                 * @param num : number of the domain (used by the \c Space).
                                                                 * @param ff: file containd the domain, generated by the save function.
                                                                 */
        Domain_bispheric_rect(int num, BinarySource& source); ///< Modern API.

        ~Domain_bispheric_rect() override;
        void save(BinarySink&) const override; ///< Modern API.

      private:
        void do_absol() const override;
        void do_radius() const override;
        void do_cart() const override;
        void del_deriv() const override;
        void do_eta() const;   ///< Computes \f$ \eta \f$ in \c *p_eta
        void do_chi() const;   ///< Computes \f$ \chi \f$ in \c *p_chi
        void do_phi() const;   ///< Computes \f$ \varphi \f$ in \c *p_phi
        void do_dsint() const; ///< Computes the surface element and stores it in \c *p_dsint

        /**
         * Computes the partial derivatives of the numerical coordinates with respect to the Cartesian ones like
         * \f$ \displaystyle\frac{\partial \eta^\star}{\partial X} \f$, stored in \c *p_detadx, for instance.
         */
        void do_for_der() const;

      public:
        const Val_domain& get_chi() const override;
        const Val_domain& get_eta() const override;

      private:
        /**
         * Sets the base to the standard one for Chebyshev polynomials.
         * The bases are :
         * \li \f$ cos(m\varphi^\star)\f$ .
         * \li \f$ T_(2j) (\chi^\star)\f$ for \f$ m \f$ even
         * and \f$ T_(2j+1) (\chi^\star)\f$ for \f$ m \f$ odd .
         * \li \f$ T_{i} (\eta^\star) \f$.
         *
         * If \c type_coloc changes, \c coloc is uupdated and the derivative members destroyed.
         * @param so [output] : the returned base.
         */
        void set_cheb_base(Base_spectral&) const override;
        /**
         * Sets the base to the standard one for Legendre polynomials.
         * The bases are :
         * \li \f$ cos(m\varphi^\star)\f$ .
         * \li \f$ P_(2j) (\chi^\star)\f$ for \f$ m \f$ even
         * and \f$ P_(2j+1) (\chi^\star)\f$ for \f$ m \f$ odd .
         * \li \f$ P_{i} (\eta^\star) \f$.
         *
         * If \c type_coloc changes, \c coloc is uupdated and the derivative members destroyed.
         * @param so [output] : the returned base.
         */
        void set_legendre_base(Base_spectral&) const override;
        /**
         * Sets the base to the standard one for Chebyshev polynomials and an astisymetric function with respect to
         * \f$ z=0 \f$.
         * The bases are :
         * \li \f$ sin(m\varphi^\star)\f$ .
         * \li \f$ T_(2j) (\chi^\star)\f$ for \f$ m \f$ even
         * and \f$ T_(2j+1) (\chi^\star)\f$ for \f$ m \f$ odd .
         * \li \f$ T_{i} (\eta^\star) \f$.
         *
         * If \c type_coloc changes, \c coloc is uupdated and the derivative members destroyed.
         * @param so [output] : the returned base.
         */
        void set_anti_cheb_base(Base_spectral&) const override;
        /**
         * Sets the base to the standard one for Legendre polynomials and an astisymetric function with respect to
         * \f$ z=0 \f$.
         * The bases are :
         * \li \f$ sin(m\varphi^\star)\f$ .
         * \li \f$ P_(2j) (\chi^\star)\f$ for \f$ m \f$ even
         * and \f$ P_(2j+1) (\chi^\star)\f$ for \f$ m \f$ odd .
         * \li \f$ P_{i} (\eta^\star) \f$.
         *
         * If \c type_coloc changes, \c coloc is uupdated and the derivative members destroyed.
         * @param so [output] : the returned base.
         */
        void set_anti_legendre_base(Base_spectral&) const override;

        void do_coloc() override;
        int give_place_var(char*) const override;

      public:
        /**
         * Check whether a point lies inside \c Domain.
         * @param xx [input] : the point.
         * @param prec [input] : precision of the computation (used when comparing doubles).
         * @returns a \c true if the point is in the domain and \c false otherwise.
         */
        bool is_in(const Point& xx, double prec = 1e-13) const override;
        /**
         * Computes the numerical coordinates from the physical ones.
         * \li \f$ \eta^\star =
         * \displaystyle\frac{2}{\eta_{\rm max}-\eta_{\rm min}} ({\rm atanh}(\displaystyle\frac{2ax}{a^2+r^2})
         * - \displaystyle\frac{\eta_{\rm max}+\eta_{\rm min}}{2}) \f$
         * \li \f$ \chi^\star = \displaystyle\frac{({\rm atan} (\displaystyle\frac{2a\sqrt{y^2+z^2}}{r^2-a^2} - \pi)}
         * {\chi_{\rm min} - \pi} \f$
         * \li \f$ \varphi^\star = {\rm atan} \displaystyle\frac{z}{y} \f$.
         * @param xxx [input] : the absolute Cartesian \f$ (x, y, z) \f$ coordinates of the point.
         * @returns the numerical coordinates \f$ (x, \theta^\star, \varphi^\star) \f$.
         */
        const Point absol_to_num(const Point&) const override;

        const Point absol_to_num_bound(const Point&, int) const override;

        /**
         * Computes the derivative with respect to the absolute Cartesian coordinates from the
         * derivative with respect to the numerical coordinates.
         * @param der_var [input] : the \c ndim derivatives with respect to the numerical coordinates.
         * @param der_abs [output] : the \c ndim derivatives with respect to the absolute Cartesian coordinates.
         */
        void do_der_abs_from_der_var(Val_domain** der_var, Val_domain** der_abs) const override;

        Base_spectral mult(const Base_spectral&, const Base_spectral&) const override;

      public:
        Val_domain mult_r(const Val_domain&) const override;
        Val_domain mult_cos_phi(const Val_domain&) const override;
        Val_domain mult_sin_phi(const Val_domain&) const override;
        Val_domain div_chi(const Val_domain&) const override;
        Val_domain div_sin_chi(const Val_domain&) const override;

        double val_boundary(int, const Val_domain&, const Index&) const override;
        Val_domain der_normal(const Val_domain&, int) const override;
        double integ(const Val_domain&, int) const override;
        double integ_volume(const Val_domain&) const override;
        void find_other_dom(int, int, int&, int&) const override;

        int nbr_points_boundary(int, const Base_spectral&) const override;
        void do_which_points_boundary(int, const Base_spectral&, Index**, int) const override;

        int nbr_unknowns(const Tensor&, int) const override;
        /**
         * Computes the number of true unknowns of a \c Val_domain.
         * It takes into account the various symmetries and regularity conditions to determine the precise number of
         * degrees of freedom.
         * @param so : the field.
         * @returns the number of true unknowns.
         */
        int nbr_unknowns_val_domain(const Val_domain& so) const;
        Array<int> nbr_conditions(const Tensor&, int, int, int n_cmp = -1, Array<int>** p_cmp = nullptr) const override;
        /**
         * Computes number of discretized equations associated with a given tensorial equation in the bulk.
         * It takes into account the various Galerkin basis used.
         * @param eq : the residual of the equation.
         * @param order : order of the equation (i.e. 2 for a Laplacian for instance)
         * @returns the number of true unknowns.
         */
        int nbr_conditions_val_domain(const Val_domain& eq, int order) const;
        Array<int> nbr_conditions_boundary(const Tensor&, int, int, int n_cmp = -1,
                                                   Array<int>** p_cmp = nullptr) const override;
        /**
         * Computes number of discretized equations associated with a given equation on a boundary.
         * It takes into account the various Galerkin basis used.
         * It is used for implementing boundary conditions and matching ones.
         * @param eq : the residual of the equation.
         * @param bound : which boundary.
         * @returns the number of true conditions.
         */
        int nbr_conditions_val_domain_boundary(const Val_domain& eq, int bound) const;
        void export_tau(const Tensor&, int, int, Array<double>&, int&, const Array<int>&, int n_cmp = -1,
                                Array<int>** p_cmp = nullptr) const override;
        bool describe_volume_residual_rows(
            const Tensor&, int, int, const Array<int>&, int, Array<int>**,
            std::vector<ResidualRowDescriptor>&) const override;
        /**
         * Exports a residual equation in the bulk.
         * It makes use of the various Galerkin basis used.
         * @param eq : the residual of the equation.
         * @param order : describes the order of the equation (2 for a Laplacian for instance).
         * @param res : The \c Array where the discretized equations are stored.
         * @param pos_res : current position in res.
         * @param ncond : the corresponding number of equations. It is used when the equation is null.
         */
        void export_tau_val_domain(const Val_domain& eq, int order, Array<double>& res, int& pos_res, int ncond) const;
        void export_tau_boundary(const Tensor&, int, int, Array<double>&, int&, const Array<int>&,
                                         int n_cmp = -1, Array<int>** p_cmp = nullptr) const override;
        bool describe_boundary_residual_rows(
            const Tensor&, int, int, const Array<int>&, int, Array<int>**,
            std::vector<ResidualRowDescriptor>&) const override;
        /**
         * Exports all the residual equations corresponding to a tensorial one on a given boundary
         * It makes use of the various Galerkin basis used.
         * @param eq : the residual of the equation.
         * @param bound : the boundary at which the equation is enforced.
         * @param res : The \c Array where the discretized equations are stored.
         * @param pos_res : current position in res.
         * @param ncond : the corresponding number of equations. It is used when the equation is null.
         */
        void export_tau_val_domain_boundary(const Val_domain& eq, int bound, Array<double>& res, int& pos_res,
                                            int ncond) const;
        void affecte_tau(Tensor&, int, const Array<double>&, int&) const override;
        /**
         * Affects some coefficients to a \c Val_domain.
         * It takes into account the various symmetries and regularity conditions (by means of Garlekin basis).
         * @param so : the field to be affected.
         * @param cf : \c Array of the coefficients used.
         * @param pos_cf : current position in the array of coefficients.
         */
        void affecte_tau_val_domain(Val_domain& so, const Array<double>& cf, int& pos_cf) const;
        void affecte_tau_one_coef(Tensor&, int, int, int&) const override;
        int phi_coefficient_parity(int k, int phi_basis) const override;
        bool describe_tau_seed_block(const Tensor&, int,
                                     std::vector<TauSeedDescriptor>&) const override;

        /**
         * Sets at most one coefficient of a \c Val_domain to 1.
         * It takes into account the various symmetries and regularity conditions (by means of Garlekin basis).
         * @param so : the \c Val_domain to be affected. It is set to zero if cc does not corresponds to another field.
         * @param cc : location, in the overall system, of the coefficient to be set to 1.
         * @param pos_cf : current position.
         */
        void affecte_tau_one_coef_val_domain(Val_domain& so, int cc, int& pos_cf) const;

        Array<int> nbr_conditions_boundary_one_side(const Tensor&, int, int, int n_cmp = -1,
                                                            Array<int>** p_cmp = nullptr) const override;

        /**
         * Computes number of discretized equations associated with a given equation on a boundary, for a first order
         * equation It takes into account the various Galerkin basis used. It is used for implementing boundary
         * conditions and matching ones.
         * @param eq : the residual of the equation.
         * @param bound : which boundary.
         * @returns the number of true conditions.
         */
        int nbr_conditions_val_domain_boundary_one_side(const Val_domain& eq, int bound) const;

        void export_tau_boundary_one_side(const Tensor&, int, int, Array<double>&, int&, const Array<int>&,
                                                  int n_cmp = -1, Array<int>** p_cmp = nullptr) const override;
        /**
         * Exports all the residual equations corresponding to a tensorial one on a given boundary, for a first order
         * equation. It makes use of the various Galerkin basis used.
         * @param eq : the residual of the equation.
         * @param bound : the boundary at which the equation is enforced.
         * @param res : The \c Array where the discretized equations are stored.
         * @param pos_res : current position in res.
         * @param ncond : the corresponding number of equations. It is used when the equation is null.
         */
        void export_tau_val_domain_boundary_one_side(const Val_domain& eq, int bound, Array<double>& res, int& pos_res,
                                                     int ncond) const;
        /**
         * Returns a fit in \f$ 1/r^2\f$ with respect to the inner spherical boundary.
         * The value of the fit is the same as the input field on the inner boundary and a decrease in \f$ 1/r^2\f$ is
         * enforced elsewhere.
         * @param so : the field to be fitted.
         * @returns : the fit.
         */
        Term_eq fithor(const Term_eq& so) const;
        /**
         * Return a fit in \f$ 1/r^2\f$ with respect to the inner spherical boundary.
         * The value of the fit is the same as the input field on the inner boundary and a decrease in \f$ 1/r^2\f$ is
         * enforced elsewhere.
         * @param so : the field to be fitted.
         * @returns : the fit.
         */
        Val_domain fithor(const Val_domain&) const;

        Tensor import(int, int, int, const Array<int>&, Tensor**) const override;
        Term_eq derive_flat_cart(int, char, const Term_eq&, const Metric*) const override;

      public:
        ostream& print(ostream& o) const override;
    };

    /**
     * Class for bispherical coordinates with a symmetry with respect to the plane \f$ z=0 \f$.
     * This class implements a domain where \f$ \eta \f$ is given as a function of \f$ \chi \f$.
     * \li 3 dimensions.
     * \li The numerical coordinates are :
     *
     * \f$ -1 \leq \eta^\star \leq 1 \f$
     *
     * \f$ 0 \leq \chi^\star \leq 1 \f$
     *
     * \f$ 0 \leq \varphi^\star < 2\pi \f$
     *
     * \li Standard bispherical coordinates :
     *
     * \f$ \eta = \displaystyle\frac{f(\chi)-\eta_{\rm lim}}{2} \eta^\star
     * +  \displaystyle\frac{f(\chi)+\eta_{\rm lim}}{2} \f$
     *
     * \f$ \chi = \chi_{\rm max} \chi^\star \f$
     *
     * \f$ \varphi = \varphi^\star \f$
     *
     * The function \f$ f (\chi) \f$ is chosen to ensure that the outer boundary of the domain lies on a sphere
     * of radius \f$ R \f$. Explicitely, \f$ f \f$ must fulfill the following condition :
     *
     * \f$ \displaystyle\frac{\sinh ^2 f(\chi) + \sin ^2\chi} {(\cosh f(\chi) - \cos \chi)^2} =
     * \displaystyle\frac{R^2}{a^2} \f$.
     *
     * \li Standard associated Cartesian coordinates are :
     *
     * \f$ x = a \displaystyle\frac{\sinh \eta}{\cosh\eta - \cos\chi} \f$
     *
     * \f$ y = a \displaystyle\frac{\cosh \eta \cos\varphi}{\cosh\eta - \cos\chi} \f$
     *
     * \f$ z = a \displaystyle\frac{\cosh \eta \sin\varphi}{\cosh\eta - \cos\chi} \f$
     * \ingroup domain
     */
    class Domain_bispheric_chi_first : public Domain
    {

      private:
        double aa;      ///< Distance scale \f$ a \f$.
        double eta_lim; ///< Lower bound for \f$ \eta \f$.
        double r_ext;   ///< Radius \f$ R \f$ of the outer boundary.
        double chi_max; ///< Upper bound for \f$ \eta \f$.

        /**
         * Value of \f$ f (\chi=0) = \ln (\frac{R}{a}+1) - \ln (\frac{R}{a}-1) \f$.
         */
        double eta_c;

        /**
         * Pointer on a \c Val_domain containing the values of \f$ f (\chi) \f$.
         */
        mutable Val_domain* bound_eta;
        /**
         * Pointer on a \c Val_domain containing the values of the derivative \f$ f (\chi) \f$
         * with respect to \f$ \chi^\star \f$.
         */
        mutable Val_domain* bound_eta_der;
        mutable Val_domain* p_eta; ///< Pointer on a \c Val_domain containing \f$ \eta \f$.
        mutable Val_domain* p_chi; ///< Pointer on a \c Val_domain containing \f$ \chi \f$.
        mutable Val_domain* p_phi; ///< Pointer on a \c Val_domain containing \f$ \varphi \f$.

        // For derivatives
        /**
         * Pointer on a \c Val_domain containing \f$ \displaystyle\frac{\partial \eta^\star}{\partial x} \f$
         *
         * The explicit expression are :
         * \f$ \displaystyle\frac{\partial \eta^\star}{\partial x } =
         * (\displaystyle\frac{\partial \eta}{\partial x} - \displaystyle\frac{\partial f}{2\partial x})
         * (\displaystyle\frac{2}{f - \eta_{\rm lim}}) + (\eta - \displaystyle\frac{f+\eta_{\rm lim}}{2})
         * (-\displaystyle\frac{2}{(f-\eta_{\rm lim})^2} \displaystyle\frac{\partial f}{\partial x}) \f$
         *
         * \f$ \displaystyle\frac{\partial \eta}{\partial x} =
         * \displaystyle\frac{2 a (a^2 + r^2 - 2x^2)}{(a^2+r^2)^2 - 4a^2x^2} \f$
         * and \f$ \displaystyle\frac{\partial f}{\partial x} =
         * f'(\chi) \displaystyle\frac{\partial \chi}{\partial x}\f$.
         */
        mutable Val_domain* p_detadx;
        /**
         * Pointer on a \c Val_domain containing \f$ \displaystyle\frac{\partial \eta^\star}{\partial y} \f$
         *
         * The explicit expression are :
         * \f$ \displaystyle\frac{\partial \eta^\star}{\partial y } =
         * (\displaystyle\frac{\partial \eta}{\partial y} - \displaystyle\frac{\partial f}{2\partial y})
         * (\displaystyle\frac{2}{f - \eta_{\rm lim}}) + (\eta - \displaystyle\frac{f+\eta_{\rm lim}}{2})
         * (-\displaystyle\frac{2}{(f-\eta_{\rm lim})^2} \displaystyle\frac{\partial f}{\partial y}) \f$
         *
         *\f$ \displaystyle\frac{\partial \eta}{\partial y} =
         * \displaystyle\frac{-4axy}{(a^2+r^2)^2 - 4a^2x^2} \f$
         * and \f$ \displaystyle\frac{\partial f}{\partial y} =
         * f'(\chi) \displaystyle\frac{\partial \chi}{\partial y}\f$.
         */
        mutable Val_domain* p_detady;
        /**
         * Pointer on a \c Val_domain containing \f$ \displaystyle\frac{\partial \eta^\star}{\partial z} \f$
         *
         * The explicit expression are :
         * \f$ \displaystyle\frac{\partial \eta^\star}{\partial z } =
         * (\displaystyle\frac{\partial \eta}{\partial z} - \displaystyle\frac{\partial f}{2\partial z})
         * (\displaystyle\frac{2}{f - \eta_{\rm lim}}) + (\eta - \displaystyle\frac{f+\eta_{\rm lim}}{2})
         * (-\displaystyle\frac{2}{(f-\eta_{\rm lim})^2} \displaystyle\frac{\partial f}{\partial z}) \f$
         *
         *\f$ \displaystyle\frac{\partial \eta}{\partial z} =
         * \displaystyle\frac{-4axz}{(a^2+r^2)^2 - 4a^2x^2} \f$
         * and \f$ \displaystyle\frac{\partial f}{\partial z} =
         * f'(\chi) \displaystyle\frac{\partial \chi}{\partial z}\f$.
         */
        mutable Val_domain* p_detadz;
        /**
         * Pointer on a \c Val_domain containing \f$ \displaystyle\frac{\partial \chi^\star}{\partial x} \f$
         * The explicit expression are :
         * \f$ \displaystyle\frac{\partial \chi}{\partial x} =
         * \displaystyle\frac{-4ax\sqrt{y^2+z^2}}{(r^2-a^2)^2 + 4 a^2 (y^2+z^2)} \f$
         *
         * and
         * \f$ \displaystyle\frac{\partial \chi^\star}{\partial \chi } =
         * \displaystyle\frac{1}{\chi_{\rm max}} \f$
         */
        mutable Val_domain* p_dchidx;
        /**
         * Pointer on a \c Val_domain containing \f$ \displaystyle\frac{\partial \chi^\star}{\partial y} \f$
         * The explicit expression are :
         * \f$ \displaystyle\frac{\partial \chi}{\partial y} = \cos\varphi
         * \displaystyle\frac{2ax(r^2-a^2-a^2(y^2+z^2)}{(r^2-a^2)^2 + 4 a^2 (y^2+z^2)} \f$
         *
         * and
         * \f$ \displaystyle\frac{\partial \chi^\star}{\partial \chi } =
         * \displaystyle\frac{1}{\chi_{\rm max}} \f$
         */
        mutable Val_domain* p_dchidy;
        /**
         * Pointer on a \c Val_domain containing \f$ \displaystyle\frac{\partial \chi^\star}{\partial z} \f$
         * The explicit expression are :
         * \f$ \displaystyle\frac{\partial \chi}{\partial z} = \sin\varphi
         * \displaystyle\frac{2ax(r^2-a^2-a^2(y^2+z^2)}{(r^2-a^2)^2 + 4 a^2 (y^2+z^2)} \f$
         *
         * and
         * \f$ \displaystyle\frac{\partial \chi^\star}{\partial \chi } =
         * \displaystyle\frac{1}{\chi_{\rm max}} \f$
         */
        mutable Val_domain* p_dchidz;
        /**
         * Pointer on a \c Val_domain containing \f$ \sin\chi \displaystyle\frac{\partial \varphi^\star}{\partial y} \f$
         * The explicit expression is :
         * \f$ \sin\chi \displaystyle\frac{\partial \varphi^\star}{\partial y} =
         * - \displaystyle\frac{(\cosh\eta - \cos\chi) \sin\varphi}{a} \f$
         */
        mutable Val_domain* p_dphidy;
        /**
         * Pointer on a \c Val_domain containing \f$ \sin\chi \displaystyle\frac{\partial \varphi^\star}{\partial z} \f$
         * The explicit expression is :
         * \f$ \sin\chi \displaystyle\frac{\partial \varphi^\star}{\partial z} =
         * \displaystyle\frac{(\cosh\eta - \cos\chi) \cos\varphi}{a} \f$
         */
        mutable Val_domain* p_dphidz;
        /**
         * Pointer on a \c Val_domain containing the surface element on the inner boundary of the domain (being
         * spherical)
         */
        mutable Val_domain* p_dsint;

      public:
        /**
         * Standard constructor :
         * @param num [input] : number of the domain (used by the \c Space).
         * @param ttype [input] : Chebyshev or Legendre type of spectral expansion.
         * @param aa [input] : scale factor \f$ a \f$.
         * @param etalim [input] : lower bound \f$ \eta_{\rm lim} \f$.
         * @param rr [input] : radius of the outer boundary \f$ R \f$.
         * @param chi_max [input] : upper bound \f$ \chi_{\rm max} \f$.
         * @param nbr [input] : number of points in each dimension.
         */
        Domain_bispheric_chi_first(int num, int ttype, double aa, double etalim, double rr, double chi_max,
                                   const Dim_array& nbr);
        Domain_bispheric_chi_first(const Domain_bispheric_chi_first& so); ///< Constructor by copy.
                                                                          /**
                                                                           * Constructor from a file
                                                                           * @param num : number of the domain (used by the \c Space).
                                                                           * @param ff: file containd the domain, generated by the save function.
                                                                           */
        Domain_bispheric_chi_first(int num, BinarySource& source); ///< Modern API.

        ~Domain_bispheric_chi_first() override;
        void save(BinarySink&) const override; ///< Modern API.

      private:
        void do_absol() const override;
        void do_radius() const override;
        void do_cart() const override;
        void del_deriv() const override;
        /**
         * Computes \f$ f(\chi) \f$ and its first derivative stored respectively in
         * \f$ *p_bound_eta \f$ and \f$ *_bound_eta_der \f$.
         */
        void do_bound_eta() const;
        void do_eta() const;   ///< Computes \f$ \eta \f$ in \c *p_eta
        void do_chi() const;   ///< Computes \f$ \chi \f$ in \c *p_chi
        void do_phi() const;   ///< Computes \f$ \varphi \f$ in \c *p_phi
        void do_dsint() const; ///< Computes the surface element and stores it in \c *p_dsint
                               /**
                                * Computes the partial derivatives of the numerical coordinates with respect to the Cartesian ones like
                                * \f$ \displaystyle\frac{\partial \eta^\star}{\partial X} \f$, stored in \c *p_detadx, for instance.
                                */
        void do_for_der() const;

      public:
        const Val_domain& get_chi() const override;
        const Val_domain& get_eta() const override;

      private:
        /**
         * Sets the base to the standard one for Chebyshev polynomials.
         * The bases are :
         * \li \f$ cos(m\varphi^\star)\f$ .
         * \li \f$ T_(2j) (\chi^\star)\f$ for \f$ m \f$ even
         * and \f$ T_(2j+1) (\chi^\star)\f$ for \f$ m \f$ odd .
         * \li \f$ T_{i} (\eta^\star) \f$.
         *
         * If \c type_coloc changes, \c coloc is uupdated and the derivative members destroyed.
         * @param so [output] : the returned base.
         */
        void set_cheb_base(Base_spectral&) const override;
        /**
         * Sets the base to the standard one for Legendre polynomials.
         * The bases are :
         * \li \f$ cos(m\varphi^\star)\f$ .
         * \li \f$ P_(2j) (\chi^\star)\f$ for \f$ m \f$ even
         * and \f$ P_(2j+1) (\chi^\star)\f$ for \f$ m \f$ odd .
         * \li \f$ P_{i} (\eta^\star) \f$.
         *
         * If \c type_coloc changes, \c coloc is uupdated and the derivative members destroyed.
         * @param so [output] : the returned base.
         */
        void set_legendre_base(Base_spectral&) const override;
        /**
         * Sets the base to the standard one for Chebyshev polynomials and an astisymetric function with respect to
         * \f$ z=0 \f$.
         * The bases are :
         * \li \f$ sin(m\varphi^\star)\f$ .
         * \li \f$ T_(2j) (\chi^\star)\f$ for \f$ m \f$ even
         * and \f$ T_(2j+1) (\chi^\star)\f$ for \f$ m \f$ odd .
         * \li \f$ T_{i} (\eta^\star) \f$.
         *
         * If \c type_coloc changes, \c coloc is uupdated and the derivative members destroyed.
         * @param so [output] : the returned base.
         */
        void set_anti_cheb_base(Base_spectral&) const override;
        /**
         * Sets the base to the standard one for Legendre polynomials and an astisymetric function with respect to
         * \f$ z=0 \f$.
         * The bases are :
         * \li \f$ sin(m\varphi^\star)\f$ .
         * \li \f$ P_(2j) (\chi^\star)\f$ for \f$ m \f$ even
         * and \f$ P_(2j+1) (\chi^\star)\f$ for \f$ m \f$ odd .
         * \li \f$ P_{i} (\eta^\star) \f$.
         *
         * If \c type_coloc changes, \c coloc is uupdated and the derivative members destroyed.
         * @param so [output] : the returned base.
         */
        void set_anti_legendre_base(Base_spectral&) const override;

        void do_coloc() override;
        int give_place_var(char*) const override;

      public:
        /**
         * Check whether a point lies inside \c Domain.
         * @param xx [input] : the point.
         * @param prec [input] : precision of the computation (used when comparing doubles).
         * @returns a \c true if the point is in the domain and \c false otherwise.
         */
        bool is_in(const Point& xx, double prec = 1e-13) const override;
        /**
         * Computes the numerical coordinates from the physical ones.
         * \li \f$ \eta^\star =
         * \displaystyle\frac{2}{f(\chi)-\eta_{\rm lim}} ({\rm atanh}(\displaystyle\frac{2ax}{a^2+r^2})
         * - \displaystyle\frac{f(\chi)+\eta_{\rm lim}}{2}) \f$
         * \li \f$ \chi^\star = \displaystyle\frac{({\rm atan} \displaystyle\frac{2a\sqrt{y^2+z^2}}{r^2-a^2}}
         * {\chi_{\rm max}} \f$
         * \li \f$ \varphi^\star = {\rm atan} \displaystyle\frac{z}{y} \f$.
         * @param xxx [input] : the absolute Cartesian \f$ (x, y, z) \f$ coordinates of the point.
         * @returns the numerical coordinates \f$ (x, \theta^\star, \varphi^\star) \f$.
         */
        const Point absol_to_num(const Point&) const override;

        const Point absol_to_num_bound(const Point&, int) const override;

        /**
         * Computes the derivative with respect to the absolute Cartesian coordinates from the
         * derivative with respect to the numerical coordinates.
         * @param der_var [input] : the \c ndim derivatives with respect to the numerical coordinates.
         * @param der_abs [output] : the \c ndim derivatives with respect to the absolute Cartesian coordinates.
         */
        void do_der_abs_from_der_var(Val_domain** der_var, Val_domain** der_abs) const override;
        Base_spectral mult(const Base_spectral&, const Base_spectral&) const override;
        double integ(const Val_domain&, int) const override;
        double integ_volume(const Val_domain&) const override;

      public:
        Val_domain mult_r(const Val_domain&) const override;
        Val_domain mult_cos_phi(const Val_domain&) const override;
        Val_domain mult_sin_phi(const Val_domain&) const override;
        Val_domain div_chi(const Val_domain&) const override;
        Val_domain div_sin_chi(const Val_domain&) const override;

        double val_boundary(int, const Val_domain&, const Index&) const override;
        void find_other_dom(int, int, int&, int&) const override;
        Val_domain der_normal(const Val_domain&, int) const override;
        int nbr_points_boundary(int, const Base_spectral&) const override;
        void do_which_points_boundary(int, const Base_spectral&, Index**, int) const override;

        int nbr_unknowns(const Tensor&, int) const override;
        /**
         * Computes the number of true unknowns of a \c Val_domain.
         * It takes into account the various symmetries and regularity conditions to determine the precise number of
         * degrees of freedom.
         * @param so : the field.
         * @returns the number of true unknowns.
         */
        int nbr_unknowns_val_domain(const Val_domain& so) const;
        Array<int> nbr_conditions(const Tensor&, int, int, int n_cmp = -1, Array<int>** p_cmp = nullptr) const override;
        /**
         * Computes number of discretized equations associated with a given tensorial equation in the bulk.
         * It takes into account the various Galerkin basis used.
         * @param eq : the residual of the equation.
         * @param order : order of the equation (i.e. 2 for a Laplacian for instance)
         * @returns the number of true unknowns.
         */
        int nbr_conditions_val_domain(const Val_domain& eq, int order) const;
        Array<int> nbr_conditions_boundary(const Tensor&, int, int, int n_cmp = -1,
                                                   Array<int>** p_cmp = nullptr) const override;
        /**
         * Computes number of discretized equations associated with a given equation on a boundary.
         * It takes into account the various Galerkin basis used.
         * It is used for implementing boundary conditions and matching ones.
         * @param eq : the residual of the equation.
         * @param bound : which boundary.
         * @returns the number of true conditions.
         */
        int nbr_conditions_val_domain_boundary(const Val_domain& eq, int bound) const;
        void export_tau(const Tensor&, int, int, Array<double>&, int&, const Array<int>&, int n_cmp = -1,
                                Array<int>** p_cmp = nullptr) const override;
        bool describe_volume_residual_rows(
            const Tensor&, int, int, const Array<int>&, int, Array<int>**,
            std::vector<ResidualRowDescriptor>&) const override;
        /**
         * Exports a residual equation in the bulk.
         * It makes use of the various Galerkin basis used.
         * @param eq : the residual of the equation.
         * @param order : describes the order of the equation (2 for a Laplacian for instance).
         * @param res : The \c Array where the discretized equations are stored.
         * @param pos_res : current position in res.
         * @param ncond : the corresponding number of equations. It is used when the equation is null.
         */
        void export_tau_val_domain(const Val_domain& eq, int order, Array<double>& res, int& pos_res, int ncond) const;
        void export_tau_boundary(const Tensor&, int, int, Array<double>&, int&, const Array<int>&,
                                         int n_cmp = -1, Array<int>** p_cmp = nullptr) const override;
        bool describe_boundary_residual_rows(
            const Tensor&, int, int, const Array<int>&, int, Array<int>**,
            std::vector<ResidualRowDescriptor>&) const override;
        /**
         * Exports all the residual equations corresponding to a tensorial one on a given boundary
         * It makes use of the various Galerkin basis used.
         * @param eq : the residual of the equation.
         * @param bound : the boundary at which the equation is enforced.
         * @param res : The \c Array where the discretized equations are stored.
         * @param pos_res : current position in res.
         * @param ncond : the corresponding number of equations. It is used when the equation is null.
         */
        void export_tau_val_domain_boundary(const Val_domain& eq, int bound, Array<double>& res, int& pos_res,
                                            int ncond) const;
        void affecte_tau(Tensor&, int, const Array<double>&, int&) const override;
        /**
         * Affects some coefficients to a \c Val_domain.
         * It takes into account the various symmetries and regularity conditions (by means of Garlekin basis).
         * @param so : the field to be affected.
         * @param cf : \c Array of the coefficients used.
         * @param pos_cf : current position in the array of coefficients.
         */
        void affecte_tau_val_domain(Val_domain& so, const Array<double>& cf, int& pos_cf) const;
        void affecte_tau_one_coef(Tensor&, int, int, int&) const override;
        int phi_coefficient_parity(int k, int phi_basis) const override;
        bool describe_tau_seed_block(const Tensor&, int,
                                     std::vector<TauSeedDescriptor>&) const override;
        /**
         * Sets at most one coefficient of a \c Val_domain to 1.
         * It takes into account the various symmetries and regularity conditions (by means of Garlekin basis).
         * @param so : the \c Val_domain to be affected. It is set to zero if cc does not corresponds to another field.
         * @param cc : location, in the overall system, of the coefficient to be set to 1.
         * @param pos_cf : current position.
         */
        void affecte_tau_one_coef_val_domain(Val_domain& so, int cc, int& pos_cf) const;

        Array<int> nbr_conditions_boundary_one_side(const Tensor&, int, int, int n_cmp = -1,
                                                            Array<int>** p_cmp = nullptr) const override;

        /**
         * Computes number of discretized equations associated with a given equation on a boundary, for a first order
         * equation It takes into account the various Galerkin basis used. It is used for implementing boundary
         * conditions and matching ones.
         * @param eq : the residual of the equation.
         * @param bound : which boundary.
         * @returns the number of true conditions.
         */
        int nbr_conditions_val_domain_boundary_one_side(const Val_domain& eq, int bound) const;
        void export_tau_boundary_one_side(const Tensor&, int, int, Array<double>&, int&, const Array<int>&,
                                                  int n_cmp = -1, Array<int>** p_cmp = nullptr) const override;
        /**
         * Exports all the residual equations corresponding to a tensorial one on a given boundary, for a first order
         * equation. It makes use of the various Galerkin basis used.
         * @param eq : the residual of the equation.
         * @param bound : the boundary at which the equation is enforced.
         * @param res : The \c Array where the discretized equations are stored.
         * @param pos_res : current position in res.
         * @param ncond : the corresponding number of equations. It is used when the equation is null.
         */
        void export_tau_val_domain_boundary_one_side(const Val_domain& eq, int bound, Array<double>& res, int& pos_res,
                                                     int ncond) const;
        /**
         * Returns a fit in \f$ 1/r^2\f$ with respect to the inner spherical boundary.
         * The value of the fit is the same as the input field on the inner boundary and a decrease in \f$ 1/r^2\f$ is
         * enforced elsewhere.
         * @param so : the field to be fitted.
         * @returns : the fit.
         */
        Term_eq fithor(const Term_eq& so) const;
        /**
         * Return a fit in \f$ 1/r^2\f$ with respect to the inner spherical boundary.
         * The value of the fit is the same as the input field on the inner boundary and a decrease in \f$ 1/r^2\f$ is
         * enforced elsewhere.
         * @param so : the field to be fitted.
         * @returns : the fit.
         */
        Val_domain fithor(const Val_domain& so) const;

        Tensor import(int, int, int, const Array<int>&, Tensor**) const override;
        Term_eq derive_flat_cart(int, char, const Term_eq&, const Metric*) const override;

      public:
        ostream& print(ostream& o) const override;
    };

    /**
     * Class for bispherical coordinates with a symmetry with respect to the plane \f$ z=0 \f$.
     * This class implements a domain where \f$ \chi \f$ is given as a function of \f$ \eta \f$.
     * \li 3 dimensions.
     * \li The numerical coordinates are :
     *
     * \f$ 0 \leq \chi^\star \leq 1 \f$
     *
     * \f$ -1 \leq \eta^\star \leq 1 \f$
     *
     * \f$ 0 \leq \varphi^\star < 2\pi \f$
     *
     * \li Standard bispherical coordinates :
     *
     * \f$ \chi = (g(\eta)-\pi) \chi^\star + \pi \f$
     *
     * \f$ \eta = \displaystyle\frac{\eta_{\rm max}-\eta_{\rm min}}{2} \eta^\star
     * +  \displaystyle\frac{\eta_{\rm max}+\eta_{\rm min}}{2} \f$
     *
     * \f$ \varphi = \varphi^\star \f$
     *
     * The function \f$ g (\eta) \f$ is chosen to ensure that the outer boundary of the domain lies on a sphere
     * of radius \f$ R \f$. Explicitely, \f$ g \f$ must fulfill the following condition :
     *
     * \f$ \displaystyle\frac{\sinh ^2 \eta + \sin ^2 g(\eta)} {(\cosh \eta - \cos g(\eta))^2} =
     * \displaystyle\frac{R^2}{a^2} \f$.
     *
     * \li Standard associated Cartesian coordinates are :
     *
     * \f$ x = a \displaystyle\frac{\sinh \eta}{\cosh\eta - \cos\chi} \f$
     *
     * \f$ y = a \displaystyle\frac{\cosh \eta \cos\varphi}{\cosh\eta - \cos\chi} \f$
     *
     * \f$ z = a \displaystyle\frac{\cosh \eta \sin\varphi}{\cosh\eta - \cos\chi} \f$
     * \ingroup domain
     */
    class Domain_bispheric_eta_first : public Domain
    {

      private:
        double aa;      ///< Distance scale \f$ a \f$.
        double r_ext;   ///< Radius \f$ R \f$ of the outer boundary.
        double eta_min; ///< Lower bound for \f$ \eta \f$.
        double eta_max; ///< Upper bound for \f$ \eta \f$.

        /**
         * Value of \f$ g (\eta=0) = 2*{\rm atan}(\displaystyle\frac{a}{R}) \f$.
         */
        double chi_c;

        /**
         * Pointer on a \c Val_domain containing the values of \f$ g (\eta) \f$.
         */
        mutable Val_domain* bound_chi;
        /**
         * Pointer on a \c Val_domain containing the values of the derivative \f$ g (\eta) \f$
         * with respect to \f$ \eta^\star \f$.
         */
        mutable Val_domain* bound_chi_der;
        mutable Val_domain* p_eta; ///< Pointer on a \c Val_domain containing \f$ \eta \f$.
        mutable Val_domain* p_chi; ///< Pointer on a \c Val_domain containing \f$ \chi \f$.
        mutable Val_domain* p_phi; ///< Pointer on a \c Val_domain containing \f$ \varphi \f$.

        // For derivatives
        /**
         * Pointer on a \c Val_domain containing \f$ \displaystyle\frac{\partial \eta^\star}{\partial x} \f$
         * The explicit expression are :
         * \f$ \displaystyle\frac{\partial \eta}{\partial x} =
         * \displaystyle\frac{2 a (a^2 + r^2 - 2x^2)}{(a^2+r^2)^2 - 4a^2x^2} \f$
         *
         * and
         * \f$ \displaystyle\frac{\partial \eta^\star}{\partial \eta } =
         * \displaystyle\frac{2}{\eta_{\rm max} - \eta_{\rm min}} \f$
         */
        mutable Val_domain* p_detadx;
        /**
         * Pointer on a \c Val_domain containing \f$ \displaystyle\frac{\partial \eta^\star}{\partial y} \f$
         * The explicit expression are :
         * \f$ \displaystyle\frac{\partial \eta}{\partial y} =
         * \displaystyle\frac{-4axy}{(a^2+r^2)^2 - 4a^2x^2} \f$
         *
         * and
         * \f$ \displaystyle\frac{\partial \eta^\star}{\partial \eta } =
         * \displaystyle\frac{2}{\eta_{\rm max} - \eta_{\rm min}} \f$
         */
        mutable Val_domain* p_detady;
        /**
         * Pointer on a \c Val_domain containing \f$ \displaystyle\frac{\partial \eta^\star}{\partial z} \f$
         * The explicit expression are :
         * \f$ \displaystyle\frac{\partial \eta}{\partial z} =
         * \displaystyle\frac{-4axz}{(a^2+r^2)^2 - 4a^2x^2} \f$
         *
         * and
         * \f$ \displaystyle\frac{\partial \eta^\star}{\partial \eta } =
         * \displaystyle\frac{2}{\eta_{\rm max} - \eta_{\rm min}} \f$
         */
        mutable Val_domain* p_detadz;
        /**
         * Pointer on a \c Val_domain containing \f$ \displaystyle\frac{\partial \chi^\star}{\partial x} \f$
         *
         * The explicit expression are :
         * \f$ \displaystyle\frac{\partial \chi^\star}{\partial x } =
         * \displaystyle\frac{\displaystyle\frac{\partial \chi}{\partial x}(g(\eta) - \pi)
         * - \displaystyle\frac{\partial g}{\partial x}(\chi - \pi)}{(g(\eta)-\pi)^2} \f$
         *
         * \f$ \displaystyle\frac{\partial \chi}{\partial x} =
         * \displaystyle\frac{-4ax\sqrt{y^2+z^2}}{(r^2-a^2)^2 + 4 a^2 (y^2+z^2)} \f$
         * and \f$ \displaystyle\frac{\partial g}{\partial x} =
         * g'(\eta) \displaystyle\frac{\partial \eta}{\partial x}\f$.
         */
        mutable Val_domain* p_dchidx;
        /**
         * Pointer on a \c Val_domain containing \f$ \displaystyle\frac{\partial \chi^\star}{\partial y} \f$
         *
         * The explicit expression are :
         * \f$ \displaystyle\frac{\partial \chi^\star}{\partial y} =
         * \displaystyle\frac{\displaystyle\frac{\partial \chi}{\partial y}(g(\eta) - \pi)
         * - \displaystyle\frac{\partial g}{\partial y}(\chi - \pi)}{(g(\eta)-\pi)^2} \f$
         *
         * \f$ \displaystyle\frac{\partial \chi}{\partial y} =
         * \cos\varphi \displaystyle\frac{2ax(r^2-a^2-a^2(y^2+z^2)}{(r^2-a^2)^2 + 4 a^2 (y^2+z^2)} \f$
         * and \f$ \displaystyle\frac{\partial g}{\partial y} =
         * g'(\eta) \displaystyle\frac{\partial \eta}{\partial y}\f$.
         */
        mutable Val_domain* p_dchidy;
        /**
         * Pointer on a \c Val_domain containing \f$ \displaystyle\frac{\partial \chi^\star}{\partial z} \f$
         *
         * The explicit expression are :
         * \f$ \displaystyle\frac{\partial \chi^\star}{\partial z} =
         * \displaystyle\frac{\displaystyle\frac{\partial \chi}{\partial z}(g(\eta) - \pi)
         * - \displaystyle\frac{\partial g}{\partial z}(\chi - \pi)}{(g(\eta)-\pi)^2} \f$
         *
         * \f$ \displaystyle\frac{\partial \chi}{\partial z} =
         * \sin\varphi \displaystyle\frac{2ax(r^2-a^2-a^2(y^2+z^2)}{(r^2-a^2)^2 + 4 a^2 (y^2+z^2)} \f$
         * and \f$ \displaystyle\frac{\partial g}{\partial z} =
         * g'(\eta) \displaystyle\frac{\partial \eta}{\partial z}\f$.
         */
        mutable Val_domain* p_dchidz;
        /**
         * Pointer on a \c Val_domain containing \f$ \sin\chi \displaystyle\frac{\partial \varphi^\star}{\partial y} \f$
         * The explicit expression is :
         * \f$ \sin\chi \displaystyle\frac{\partial \varphi^\star}{\partial y} =
         * - \displaystyle\frac{(\cosh\eta - \cos\chi) \sin\varphi}{a} \f$
         */
        mutable Val_domain* p_dphidy;
        /**
         * Pointer on a \c Val_domain containing \f$ \sin\chi \displaystyle\frac{\partial \varphi^\star}{\partial z} \f$
         * The explicit expression is :
         * \f$ \sin\chi \displaystyle\frac{\partial \varphi^\star}{\partial z} =
         * \displaystyle\frac{(\cosh\eta - \cos\chi) \cos\varphi}{a} \f$
         */
        mutable Val_domain* p_dphidz;

      public:
        /**
         * Standard constructor :
         * @param num [input] : number of the domain (used by the \c Space).
         * @param ttype [input] : Chebyshev or Legendre type of spectral expansion.
         * @param aa [input] : scale factor \f$ a \f$.
         * @param rr [input] : radius of the outer boundary \f$ R \f$.
         * @param eta_min [input] : lower bound \f$ \eta_{\rm min} \f$.
         * @param eta_max [input] : upper bound \f$ \eta_{\rm max} \f$.
         * @param nbr [input] : number of points in each dimension.
         */
        Domain_bispheric_eta_first(int num, int ttype, double aa, double rr, double eta_min, double eta_max,
                                   const Dim_array& nbr);
        Domain_bispheric_eta_first(const Domain_bispheric_eta_first& so); ///< Copy constructor.
                                                                          /**
                                                                           * Constructor from a file
                                                                           * @param num : number of the domain (used by the \c Space).
                                                                           * @param ff: file containd the domain, generated by the save function.
                                                                           */
        Domain_bispheric_eta_first(int num, BinarySource& source); ///< Modern API.

        ~Domain_bispheric_eta_first() override;
        void save(BinarySink&) const override; ///< Modern API.

      private:
        void do_absol() const override;
        void do_radius() const override;
        void do_cart() const override;
        void del_deriv() const override;
        /**
         * Computes \f$ g(\eta) \f$ and its first derivative stored respectively in
         * \f$ *p_bound_chi \f$ and \f$ *_bound_chi_der \f$.
         */
        void do_bound_chi() const;
        void do_eta() const; ///< Computes \f$ \eta \f$ in \c *p_eta
        void do_chi() const; ///< Computes \f$ \chi \f$ in \c *p_chi
        void do_phi() const; ///< Computes \f$ \varphi \f$ in \c *p_phi
                             /**
                              * Computes the partial derivatives of the numerical coordinates with respect to the Cartesian ones like
                              * \f$ \displaystyle\frac{\partial \eta^\star}{\partial X} \f$, stored in \c *p_detadx, for instance.
                              */
        void do_for_der() const;

      public:
        const Val_domain& get_chi() const override;
        const Val_domain& get_eta() const override;

      private:
        /**
         * Sets the base to the standard one for Chebyshev polynomials.
         * The bases are :
         * \li \f$ cos(m\varphi^\star)\f$ .
         * \li \f$ T_{i} (\eta^\star) \f$.
         * \li \f$ T_(2j) (\chi^\star)\f$ for \f$ m \f$ even
         * and \f$ T_(2j+1) (\chi^\star)\f$ for \f$ m \f$ odd .
         *
         * If \c type_coloc changes, \c coloc is uupdated and the derivative members destroyed.
         * @param so [output] : the returned base.
         */
        void set_cheb_base(Base_spectral&) const override;
        /**
         * Sets the base to the standard one for Chebyshev polynomials.
         * The bases are :
         * \li \f$ cos(m\varphi^\star)\f$ .
         * \li \f$ P_{i} (\eta^\star) \f$.
         * \li \f$ P_(2j) (\chi^\star)\f$ for \f$ m \f$ even
         * and \f$ P_(2j+1) (\chi^\star)\f$ for \f$ m \f$ odd .
         *
         * If \c type_coloc changes, \c coloc is uupdated and the derivative members destroyed.
         * @param so [output] : the returned base.
         */
        void set_legendre_base(Base_spectral&) const override;

        /**
         * Sets the base to the standard one for Chebyshev polynomials and an astisymetric function with respect to
         * \f$ z=0 \f$.
         * The bases are :
         * \li \f$ sin(m\varphi^\star)\f$ .
         * \li \f$ T_{i} (\eta^\star) \f$.
         * \li \f$ T_(2j) (\chi^\star)\f$ for \f$ m \f$ even
         * and \f$ T_(2j+1) (\chi^\star)\f$ for \f$ m \f$ odd .
         *
         * If \c type_coloc changes, \c coloc is uupdated and the derivative members destroyed.
         * @param so [output] : the returned base.
         */
        void set_anti_cheb_base(Base_spectral&) const override;
        /**
         * Sets the base to the standard one for Legendre polynomials and an astisymetric function with respect to
         * \f$ z=0 \f$.
         * The bases are :
         * \li \f$ sin(m\varphi^\star)\f$ .
         * \li \f$ T_{i} (\eta^\star) \f$.
         * \li \f$ T_(2j) (\chi^\star)\f$ for \f$ m \f$ even
         * and \f$ T_(2j+1) (\chi^\star)\f$ for \f$ m \f$ odd .
         *
         * If \c type_coloc changes, \c coloc is uupdated and the derivative members destroyed.
         * @param so [output] : the returned base.
         */
        void set_anti_legendre_base(Base_spectral&) const override;
        void do_coloc() override;
        int give_place_var(char*) const override;

      public:
        /**
         * Check whether a point lies inside \c Domain.
         * @param xx [input] : the point.
         * @param prec [input] : precision of the computation (used when comparing doubles).
         * @returns a \c true if the point is in the domain and \c false otherwise.
         */
        bool is_in(const Point& xx, double prec = 1e-13) const override;
        /**
         * Computes the numerical coordinates from the physical ones.
         * \li \f$ \eta^\star =
         * \displaystyle\frac{2}{\eta_{\rm max}-\eta_{\rm min}} ({\rm atanh}(\displaystyle\frac{2ax}{a^2+r^2})
         * - \displaystyle\frac{\eta_{\rm max}+\eta_{\rm min}}{2}) \f$
         * \li \f$ \chi^\star = \displaystyle\frac{({\rm atan} \displaystyle\frac{2a\sqrt{y^2+z^2}}{r^2-a^2}-\pi)}
         * {(g(\eta) - \pi)} \f$
         * \li \f$ \varphi^\star = {\rm atan} \displaystyle\frac{z}{y} \f$.
         * @param xxx [input] : the absolute Cartesian \f$ (x, y, z) \f$ coordinates of the point.
         * @returns the numerical coordinates \f$ (x, \theta^\star, \varphi^\star) \f$.
         */
        const Point absol_to_num(const Point&) const override;

        const Point absol_to_num_bound(const Point&, int) const override;

        /**
         * Computes the derivative with respect to the absolute Cartesian coordinates from the
         * derivative with respect to the numerical coordinates.
         * @param der_var [input] : the \c ndim derivatives with respect to the numerical coordinates.
         * @param der_abs [output] : the \c ndim derivatives with respect to the absolute Cartesian coordinates.
         */
        void do_der_abs_from_der_var(Val_domain** der_var, Val_domain** der_abs) const override;
        Base_spectral mult(const Base_spectral&, const Base_spectral&) const override;

      public:
        Val_domain mult_r(const Val_domain&) const override;
        Val_domain mult_cos_phi(const Val_domain&) const override;
        Val_domain mult_sin_phi(const Val_domain&) const override;
        Val_domain div_chi(const Val_domain&) const override;
        Val_domain div_sin_chi(const Val_domain&) const override;

        double val_boundary(int, const Val_domain&, const Index&) const override;
        Val_domain der_normal(const Val_domain&, int) const override;
        double integ_volume(const Val_domain&) const override;
        void find_other_dom(int, int, int&, int&) const override;
        int nbr_points_boundary(int, const Base_spectral&) const override;
        void do_which_points_boundary(int, const Base_spectral&, Index**, int) const override;

        int nbr_unknowns(const Tensor&, int) const override;
        /**
         * Computes the number of true unknowns of a \c Val_domain.
         * It takes into account the various symmetries and regularity conditions to determine the precise number of
         * degrees of freedom.
         * @param so : the field.
         * @returns the number of true unknowns.
         */
        int nbr_unknowns_val_domain(const Val_domain& so) const;
        Array<int> nbr_conditions(const Tensor&, int, int, int n_cmp = -1, Array<int>** p_cmp = nullptr) const override;
        /**
         * Computes number of discretized equations associated with a given tensorial equation in the bulk.
         * It takes into account the various Galerkin basis used.
         * @param eq : the residual of the equation.
         * @param order : order of the equation (i.e. 2 for a Laplacian for instance)
         * @returns the number of true unknowns.
         */
        int nbr_conditions_val_domain(const Val_domain& eq, int order) const;
        Array<int> nbr_conditions_boundary(const Tensor&, int, int, int n_cmp = -1,
                                                   Array<int>** p_cmp = nullptr) const override;
        /**
         * Computes number of discretized equations associated with a given equation on a boundary.
         * It takes into account the various Galerkin basis used.
         * It is used for implementing boundary conditions and matching ones.
         * @param eq : the residual of the equation.
         * @param bound : which boundary.
         * @returns the number of true conditions.
         */
        int nbr_conditions_val_domain_boundary(const Val_domain& eq, int bound) const;
        void export_tau(const Tensor&, int, int, Array<double>&, int&, const Array<int>&, int n_cmp = -1,
                                Array<int>** p_cmp = nullptr) const override;
        bool describe_volume_residual_rows(
            const Tensor&, int, int, const Array<int>&, int, Array<int>**,
            std::vector<ResidualRowDescriptor>&) const override;
        /**
         * Exports a residual equation in the bulk.
         * It makes use of the various Galerkin basis used.
         * @param eq : the residual of the equation.
         * @param order : describes the order of the equation (2 for a Laplacian for instance).
         * @param res : The \c Array where the discretized equations are stored.
         * @param pos_res : current position in res.
         * @param ncond : the corresponding number of equations. It is used when the equation is null.
         */
        void export_tau_val_domain(const Val_domain& eq, int order, Array<double>& res, int& pos_res, int ncond) const;
        void export_tau_boundary(const Tensor&, int, int, Array<double>&, int&, const Array<int>&,
                                         int n_cmp = -1, Array<int>** p_cmp = nullptr) const override;
        bool describe_boundary_residual_rows(
            const Tensor&, int, int, const Array<int>&, int, Array<int>**,
            std::vector<ResidualRowDescriptor>&) const override;
        /**
         * Exports all the residual equations corresponding to a tensorial one on a given boundary
         * It makes use of the various Galerkin basis used.
         * @param eq : the residual of the equation.
         * @param bound : the boundary at which the equation is enforced.
         * @param res : The \c Array where the discretized equations are stored.
         * @param pos_res : current position in res.
         * @param ncond : the corresponding number of equations. It is used when the equation is null.
         */
        void export_tau_val_domain_boundary(const Val_domain& eq, int bound, Array<double>& res, int& pos_res,
                                            int ncond) const;
        void affecte_tau(Tensor&, int, const Array<double>&, int&) const override;
        /**
         * Affects some coefficients to a \c Val_domain.
         * It takes into account the various symmetries and regularity conditions (by means of Garlekin basis).
         * @param so : the field to be affected.
         * @param cf : \c Array of the coefficients used.
         * @param pos_cf : current position in the array of coefficients.
         */
        void affecte_tau_val_domain(Val_domain& so, const Array<double>& cf, int& pos_cf) const;
        void affecte_tau_one_coef(Tensor&, int, int, int&) const override;
        int phi_coefficient_parity(int k, int phi_basis) const override;
        bool describe_tau_seed_block(const Tensor&, int,
                                     std::vector<TauSeedDescriptor>&) const override;
        /**
         * Sets at most one coefficient of a \c Val_domain to 1.
         * It takes into account the various symmetries and regularity conditions (by means of Garlekin basis).
         * @param so : the \c Val_domain to be affected. It is set to zero if cc does not corresponds to another field.
         * @param cc : location, in the overall system, of the coefficient to be set to 1.
         * @param pos_cf : current position.
         */
        void affecte_tau_one_coef_val_domain(Val_domain& so, int cc, int& pos_cf) const;

        Array<int> nbr_conditions_boundary_one_side(const Tensor&, int, int, int n_cmp = -1,
                                                            Array<int>** p_cmp = nullptr) const override;
        /**
         * Computes number of discretized equations associated with a given equation on a boundary, for a first order
         * equation It takes into account the various Galerkin basis used. It is used for implementing boundary
         * conditions and matching ones.
         * @param eq : the residual of the equation.
         * @param bound : which boundary.
         * @returns the number of true conditions.
         */
        int nbr_conditions_val_domain_boundary_one_side(const Val_domain& eq, int bound) const;
        void export_tau_boundary_one_side(const Tensor&, int, int, Array<double>&, int&, const Array<int>&,
                                                  int n_cmp = -1, Array<int>** p_cmp = nullptr) const override;
        /**
         * Exports all the residual equations corresponding to a tensorial one on a given boundary, for a first order
         * equation. It makes use of the various Galerkin basis used.
         * @param eq : the residual of the equation.
         * @param bound : the boundary at which the equation is enforced.
         * @param res : The \c Array where the discretized equations are stored.
         * @param pos_res : current position in res.
         * @param ncond : the corresponding number of equations. It is used when the equation is null.
         */
        void export_tau_val_domain_boundary_one_side(const Val_domain& eq, int bound, Array<double>& res, int& pos_res,
                                                     int ncond) const;

        Tensor import(int, int, int, const Array<int>&, Tensor**) const override;
        Term_eq derive_flat_cart(int, char, const Term_eq&, const Metric*) const override;

      public:
        ostream& print(ostream& o) const override;
    };

    /**
     * The \c Space_bispheric class fills the space with bispherical coordinates (see the constructor for more details).
     * \ingroup domain
     */
    class Space_bispheric : public Space
    {
      protected:
        double a_minus; ///< X-absolute coordinate of the center of the first sphere.
        double a_plus;  ///< X-absolute coordinate of the center of the second sphere.
        int ndom_minus; ///< Number of spherical domains inside the first sphere.
        int ndom_plus;  ///< Number of spherical domains inside the second sphere.
        int nshells;    ///< Number of shells outside the bispheric region.

      public:
        int get_ndom_minus() const { return ndom_minus; }; /// Accessor ndom_minus
        int get_ndom_plus() const { return ndom_plus; };   /// Accessor ndom_minus
        int get_nshells() const { return nshells; };       /// Accessor ndom_minus

      public:
        /**
         * Standard constructor
         * @param ttype [input] : the type of basis.
         * @param dist [input] : distance \f$ d \f$ between the centers of the two spheres.
         * @param r1 [input] : radius \f$ r_1 \f$ of the first sphere.
         * @param r2 [input] : radius \f$ r_2 \f$ of the second sphere.
         * @param rext [input] : radius \f$ R \f$ of the outer boundary of the bispherical part.
         * @param nr [input] : number of points in each dimension
         *(\f$ nr-1 \f$ for coordinates of the type \f$ \varphi \f$)
         *
         *
         * The scale \f$ a \f$ for the bispherical coordinates is determined such that :
         * \f$ \sqrt{a^2 + r_1^2} + \sqrt{a^2 + r_2^2} = d \f$
         *
         * The first sphere is described by \f$ \eta_{\rm minus} = -{\rm asinh}\displaystyle\frac{a}{r_1}\f$, centered
         * at
         * \f$ x_1 = a \displaystyle\frac{\cosh \eta_{\rm minus}}{\sinh \eta_{\rm minus}} \f$.
         *
         * The second sphere is described by \f$ \eta_{\rm plus} = {\rm asinh}\displaystyle\frac{a}{r_2}\f$,  centered
         * at
         * \f$ x_2 = a \displaystyle\frac{\cosh \eta_{\rm plus}}{\sinh \eta_{\rm plus}} \f$.
         *
         * The cutoffs for the bispherical coordinates are \f$ \chi_c = 2 {\rm atan}\displaystyle\frac{a}{R} \f$ and
         * \f$ \eta_c = \ln (\displaystyle\frac{R}{a} + 1) - \ln (\displaystyle\frac{R}{a} - 1) \f$
         *
         * The various domains are then :
         * \li One \c Domain_nucleus, of radius \f$ r_1 \f$, centered at \f$ x_1 \f$.
         * \li One \c Domain_nucleus, of radius \f$ r_2 \f$, centered at \f$ x_2 \f$.
         * \li One \c Domain_bispheric_chi_first near the first sphere.
         * \li One \c Domain_bispheric_rect near the first sphere.
         * \li One \c Domain_bispheric_eta_first inbetween the two spheres.
         * \li One \c Domain_bispheric_rect near the second sphere.
         * \li One \c Domain_bispheric_chi_first near the second sphere.
         * \li One \c Domain_compact centered on the origin, with inner radius \f$ R \f$.
         */

        Space_bispheric(int ttype, double dist, double r1, double r2, double rext, int nr);

        /**
         * Constructor with several outer shells.
         * @param ttype [input] : the type of basis.
         * @param dist [input] : distance \f$ d \f$ between the centers of the two spheres.
         * @param r1 [input] : radius \f$ r_1 \f$ of the first sphere.
         * @param r2 [input] : radius \f$ r_2 \f$ of the second sphere.
         * @param nshells [input] : number of outer shells.
         * @param rr [input] : radiii of the outer shells.
         * @param nr [input] : number of points in each dimension
         *(\f$ nr-1 \f$ for coordinates of the type \f$ \varphi \f$)
         *
         *
         * The various domains are then :
         * \li One \c Domain_nucleus, of radius \f$ r_1 \f$, centered at \f$ x_1 \f$.
         * \li One \c Domain_nucleus, of radius \f$ r_2 \f$, centered at \f$ x_2 \f$.
         * \li One \c Domain_bispheric_chi_first near the first sphere.
         * \li One \c Domain_bispheric_rect near the first sphere.
         * \li One \c Domain_bispheric_eta_first inbetween the two spheres.
         * \li One \c Domain_bispheric_rect near the second sphere.
         * \li One \c Domain_bispheric_chi_first near the second sphere.
         * \li nshells \c Domain_shells outside the bispheric region, centered on the origin.
         * \li One \c Domain_compact centered on the origin.
         */
        Space_bispheric(int ttype, double dist, double r1, double r2, int nshells, const Array<double>& rr, int nr);

        /**
         * Constructor with several outer shells possibly of various types (log or \f$ 1/r\f$ mappings).
         * @param ttype [input] : the type of basis.
         * @param dist [input] : distance \f$ d \f$ between the centers of the two spheres.
         * @param r1 [input] : radius \f$ r_1 \f$ of the first sphere.
         * @param r2 [input] : radius \f$ r_2 \f$ of the second sphere.
         * @param nshells [input] : number of outer shells.
         * @param rr [input] : radiii of the outer shells.
         * @param type_r [input] : types of the various outer shells.
         * @param nr [input] : number of points in each dimension
         *(\f$ nr-1 \f$ for coordinates of the type \f$ \varphi \f$)
         *
         *
         * The various domains are then :
         * \li One \c Domain_nucleus, of radius \f$ r_1 \f$, centered at \f$ x_1 \f$.
         * \li One \c Domain_nucleus, of radius \f$ r_2 \f$, centered at \f$ x_2 \f$.
         * \li One \c Domain_bispheric_chi_first near the first sphere.
         * \li One \c Domain_bispheric_rect near the first sphere.
         * \li One \c Domain_bispheric_eta_first inbetween the two spheres.
         * \li One \c Domain_bispheric_rect near the second sphere.
         * \li One \c Domain_bispheric_chi_first near the second sphere.
         * \li nshells shells outside the bispheric region, centered on the origin. They can be \c Domain_shell, \c
         * Domain_shell_log or \c Domain_shell_surr
         * \li One \c Domain_compact centered on the origin.
         */
        Space_bispheric(int ttype, double dist, double r1, double r2, int nshells, const Array<double>& rr,
                        const Array<int>& type_r, int nr);

        /**
         * Constructor with several outer shells possibly of various types (log or \f$ 1/r\f$ mappings).
         * Each hole can be described by several shells.
         * @param ttype [input] : the type of basis.
         * @param dist [input] : distance \f$ d \f$ between the centers of the two spheres.
         * @param nminus [input] : number of domains inside the first sphere.
         * @param rminus [input] : radii  of the domains inside the first sphere.
         * @param nplus [input] : number of domains inside the second sphere.
         * @param rplus [input] : radii  of the domains inside the second sphere.
         * @param nshells [input] : number of outer shells.
         * @param rr [input] : radiii of the outer shells.
         * @param type_r [input] : types of the various outer shells.
         * @param nr [input] : number of points in each dimension
         * @param withnuc [input] : states wheather a nucleus is present or not
         *(\f$ nr-1 \f$ for coordinates of the type \f$ \varphi \f$)
         *
         *
         * The various domains are then :
         * \li One \c Domain_nucleus centered at \f$ x_1 \f$. (if present)
         * \li nminus-1 \c Domain_shell centered at \f$ x_1 \f$.
         * \li One \c Domain_nucleus centered at \f$ x_2 \f$. (if present)
         * \li nplus-2 \c Domain_shell centered at \f$ x_2 \f$.
         * \li One \c Domain_bispheric_chi_first near the first sphere.
         * \li One \c Domain_bispheric_rect near the first sphere.
         * \li One \c Domain_bispheric_eta_first inbetween the two spheres.
         * \li One \c Domain_bispheric_rect near the second sphere.
         * \li One \c Domain_bispheric_chi_first near the second sphere.
         * \li nshells shells outside the bispheric region, centered on the origin They can be \c Domain_shell, \c
         * Domain_shell_log or \c Domain_shell_surr
         * \li One \c Domain_compact centered on the origin.
         */
        Space_bispheric(int ttype, double dist, int nminus, const Array<double>& rminus, int nplus,
                        const Array<double>& rplus, int nshells, const Array<double>& rr, const Array<int>& type_r,
                        int nr, bool withnuc = true);

        /**
         * Constructor with several outer shells possibly of various types (log or \f$ 1/r\f$ mappings).
         * Each hole can be described by several shells also possibly with various types of mappings
         * @param ttype [input] : the type of basis.
         * @param dist [input] : distance \f$ d \f$ between the centers of the two spheres.
         * @param nminus [input] : number of domains inside the first sphere.
         * @param rminus [input] : radii  of the domains inside the first sphere.
         * @param type_r_minus [input] : types of the various shells around the first hole.
         * @param nplus [input] : number of domains inside the second sphere.
         * @param rplus [input] : radii  of the domains inside the second sphere.
         * @param type_r_plus [input] : types of the various shells around the second hole.
         * @param nshells [input] : number of outer shells.
         * @param rr [input] : radiii of the outer shells.
         * @param type_r [input] : types of the various outer shells.
         * @param nr [input] : number of points in each dimension
         * @param withnuc [input] : states wheather a nucleus is present or not
         *(\f$ nr-1 \f$ for coordinates of the type \f$ \varphi \f$)
         *
         *
         * The various domains are then :
         * \li One \c Domain_nucleus centered at \f$ x_1 \f$. (if present)
         * \li nminus-1 \c Shells centered at \f$ x_1 \f$. They can be \c Domain_shell, \c Domain_shell_log or \c
         * Domain_shell_surr
         * \li One \c Domain_nucleus centered at \f$ x_2 \f$ (if present)
         * \li nplus-2 \c Shells centered at \f$ x_2 \f$. They can be \c Domain_shell, \c Domain_shell_log or \c
         * Domain_shell_surr
         * \li One \c Domain_bispheric_chi_first near the first sphere.
         * \li One \c Domain_bispheric_rect near the first sphere.
         * \li One \c Domain_bispheric_eta_first inbetween the two spheres.
         * \li One \c Domain_bispheric_rect near the second sphere.
         * \li One \c Domain_bispheric_chi_first near the second sphere.
         * \li nshells shells outside the bispheric region, centered on the origin. They can be \c Domain_shell, \c
         * Domain_shell_log or \c Domain_shell_surr
         * \li One \c Domain_compact centered on the origin.
         */
        Space_bispheric(int ttype, double dist, int nminus, const Array<double>& rminus, const Array<int>& type_r_minus,
                        int nplus, const Array<double>& rplus, const Array<int>& type_r_plus, int nshells,
                        const Array<double>& rr, const Array<int>& type_r, int nr, bool withnuc = true);

        /**
         * Constructor without nucleus and one shell around each holes ; a compactified outer domain
         * @param ttype [input] : the type of basis.
         * @param dist [input] : distance \f$ d \f$ between the centers of the two spheres.
         * @param rhor1 [input] : radius  of the first horizon.
         * @param rshell1 [input] : radius  of the first shell.
         * @param rhor2 [input] : radius  of the second horizon.
         * @param rshell2 [input] : radius  of the second shell.
         * @param rext [input] : radius  of the bispherical domains.
         * @param nr [input] : number of points in each dimension
         *(\f$ nr-1 \f$ for coordinates of the type \f$ \varphi \f$)
         *
         */
        Space_bispheric(int ttype, double dist, double rhor1, double rshell1, double rhor2, double rshell2, double rext,
                        int nr);

        /**
         * Constructor without nucleus and one shell around each holes ; a compactified outer domain.
         * The resolution of each domain is speceified by hand
         * @param ttype [input] : the type of basis.
         * @param dist [input] : distance \f$ d \f$ between the centers of the two spheres.
         * @param rhor1 [input] : radius  of the first horizon.
         * @param rshell1 [input] : radius  of the first shell.
         * @param rhor2 [input] : radius  of the second horizon.
         * @param rshell2 [input] : radius  of the second shell.
         * @param rext [input] : radius  of the bispherical domains.
         * @param resol [input] : resolution in each domains
         *
         */
        Space_bispheric(int ttype, double dist, double rhor1, double rshell1, double rhor2, double rshell2, double rext,
                        Dim_array** resol);

        /**
         * Constructor without nucleus and one shell around each holes ; a compactified outer domain
         * @param ttype [input] : the type of basis.
         * @param dist [input] : distance \f$ d \f$ between the centers of the two spheres.
         * @param rhor1 [input] : radius  of the first horizon.
         * @param rshell1 [input] : radius  of the first shell.
         * @param rhor2 [input] : radius  of the second horizon.
         * @param rshell2 [input] : radius  of the second shell.
         * @param nshells [input] : number of outer shells.
         * @param rr [input] : radiii of the outer shells.
         * @param nr [input] : number of points in each dimension
         *(\f$ nr-1 \f$ for coordinates of the type \f$ \varphi \f$)
         *
         */
        Space_bispheric(int ttype, double dist, double rhor1, double rshell1, double rhor2, double rshell2, int nshells,
                        const Array<double>& rshells, int nr);

        Space_bispheric(BinarySource&, int shell_type = STD_TYPE, bool old = false); ///< Modern API.

        Space_bispheric(BinarySource&, int type_minus, int type_plus, int shell_type); ///< Modern API.

        /**
         * Sets a boundary condition at the inner radius of the first sphere.
         * @param syst : the \c System_of_eqs.
         * @param eq : the string describing the boundary condition.
         * @param nused : number of components of \c eq to be considered. All the components are used of it is -1.
         * @param pused : pointer on the indexes of the components to be considered. Not used of nused = -1 .
         */
        void add_bc_sphere_one(System_of_eqs& syst, const char* eq, int nused = -1, Array<int>** pused = nullptr);
        /**
         * Sets a boundary condition at the inner radius of the second sphere.
         * @param syst : the \c System_of_eqs.
         * @param eq : the string describing the boundary condition.
         * @param nused : number of components of \c eq to be considered. All the components are used of it is -1.
         * @param pused : pointer on the indexes of the components to be considered. Not used of nused = -1 .
         */
        void add_bc_sphere_two(System_of_eqs& syst, const char* eq, int nused = -1, Array<int>** pused = nullptr);

        /**
         * Sets a boundary condition at the outer sphere of the bispheric coordinates.
         * @param syst : the \c System_of_eqs.
         * @param eq : the string describing the boundary condition.
         * @param nused : number of components of \c eq to be considered. All the components are used of it is -1.
         * @param pused : pointer on the indexes of the components to be considered. Not used of nused = -1 .
         */
        void add_bc_outer_sphere(System_of_eqs& syst, const char* eq, int nused = -1, Array<int>** pused = nullptr);
        /**
         * Sets a boundary condition at the outer boundary.
         * @param syst : the \c System_of_eqs.
         * @param eq : the string describing the boundary condition.
         * @param nused : number of components of \c eq to be considered. All the components are used of it is -1.
         * @param pused : pointer on the indexes of the components to be considered. Not used of nused = -1 .
         */
        void add_bc_outer(System_of_eqs& syst, const char* eq, int nused = -1, Array<int>** pused = nullptr);

        /**
         * Sets an equation inside every domains (assumed to be second order).
         * @param syst : the \c System_of_eqs.
         * @param eq : the string describing the boundary condition.
         * @param nused : number of components of \c eq to be considered. All the components are used of it is -1.
         * @param pused : pointer on the indexes of the components to be considered. Not used of nused = -1 .
         */
        void add_eq(System_of_eqs& syst, const char* eq, int nused = -1, Array<int>** pused = nullptr);

        /**
         * Sets an equation inside every domains (assumed to be zeroth order).
         * @param syst : the \c System_of_eqs.
         * @param eq : the string describing the boundary condition.
         * @param nused : number of components of \c eq to be considered. All the components are used of it is -1.
         * @param pused : pointer on the indexes of the components to be considered. Not used of nused = -1 .
         */
        void add_eq_full(System_of_eqs& syst, const char* eq, int nused = -1, Array<int>** pused = nullptr);

        /**
         * Sets an equation inside every domains (assumed to be first order).
         * @param syst : the \c System_of_eqs.
         * @param eq : the string describing the boundary condition.
         * @param nused : number of components of \c eq to be considered. All the components are used of it is -1.
         * @param pused : pointer on the indexes of the components to be considered. Not used of nused = -1 .
         */
        void add_eq_one_side(System_of_eqs& syst, const char* eq, int nused = -1, Array<int>** pused = nullptr);

        /**
         * Sets a matching condition accross all the bispheric domain (intended for a second order equation).
         * @param syst : the \c System_of_eqs.
         * @param rac : the string describing the boundary condition.
         * @param list : list of the components to be considered.
         */
        void add_matching(System_of_eqs& syst, const char* rac, const List_comp& list);

        /**
         * Sets a matching condition accross all the bispheric domain (intended for a second order equation).
         * @param syst : the \c System_of_eqs.
         * @param rac : the string describing the boundary condition.
         * @param nused : number of components of \c eq to be considered. All the components are used of it is -1.
         * @param pused : pointer on the indexes of the components to be considered. Not used of nused = -1 .
         */
        void add_matching(System_of_eqs& syst, const char* rac, int nused = -1, Array<int>** pused = nullptr);

        /**
         * Sets a matching condition accross all the bispheric domain (intended for a first order equation).
         * @param syst : the \c System_of_eqs.
         * @param rac : the string describing the boundary condition.
         * @param nused : number of components of \c eq to be considered. All the components are used of it is -1.
         * @param pused : pointer on the indexes of the components to be considered. Not used of nused = -1 .
         */
        void add_matching_one_side(System_of_eqs& syst, const char* rec, int nused = -1, Array<int>** pused = nullptr);

        /**
         * Adds a bulk equation and two matching conditions.
         * @param syst : the \c System_of_eqs.
         * @param eq : the string describing the bulk equation.
         * @param rac : the string describing the first matching condition.
         * @param rac_der : the string describing the second matching condition.
         * @param nused : number of components of \c eq to be considered. All the components are used of it is -1.
         * @param pused : pointer on the indexes of the components to be considered. Not used of nused = -1 .
         */
        void add_eq(System_of_eqs& syst, const char* eq, const char* rac, const char* rac_der, int nused = -1,
                    Array<int>** pused = nullptr);

        /**
         * Adds a bulk equation and two matching conditions.
         * @param syst : the \c System_of_eqs.
         * @param eq : the string describing the bulk equation.
         * @param rac : the string describing the first matching condition.
         * @param rac_der : the string describing the second matching condition.
         * @param list : list of the components to be considered.
         */
        void add_eq(System_of_eqs& syst, const char* eq, const char* rac, const char* rac_der, const List_comp& list);

        /**
         * Adds a bulk equation and two matching conditions for a space without nucleii
         * @param syst : the \c System_of_eqs.
         * @param eq : the string describing the bulk equation.
         * @param rac : the string describing the first matching condition.
         * @param rac_der : the string describing the second matching condition.
         * @param nused : number of components of \c eq to be considered. All the components are used of it is -1.
         * @param pused : pointer on the indexes of the components to be considered. Not used of nused = -1 .
         */
        void add_eq_no_nucleus(System_of_eqs& syst, const char* eq, const char* rac, const char* rac_der,
                               int nused = -1, Array<int>** pused = nullptr);

        /**
         * Adds a bulk equation and two matching conditions for a space without nucleii
         * @param syst : the \c System_of_eqs.
         * @param eq : the string describing the bulk equation.
         * @param rac : the string describing the first matching condition.
         * @param rac_der : the string describing the second matching condition.
         * @param list : list of the components to be considered.
         */
        void add_eq_no_nucleus(System_of_eqs& syst, const char* eq, const char* rac, const char* rac_der,
                               const List_comp& list);
        /**
         * Adds an equation being a surface integral at infinity.
         * @param syst : the \c System_of_eqs.
         * @param eq : the string describing the equation (should contain something like integ(f)=b)
         */
        void add_eq_int_inf(System_of_eqs& syst, const char* eq);
        /**
         * Adds an equation being a surface integral on the first sphere.
         * @param syst : the \c System_of_eqs.
         * @param eq : the string describing the equation (should contain something like integ(f)=b)
         */
        void add_eq_int_sphere_one(System_of_eqs& syst, const char* eq);
        /**
         * Adds an equation being a surface integral on the second sphere.
         * @param syst : the \c System_of_eqs.
         * @param eq : the string describing the equation (should contain something like integ(f)=b)
         */
        void add_eq_int_sphere_two(System_of_eqs& syst, const char* eq);

        /**
         * Adds an equation saying that one coefficient of a field is zero (at infinity)
         * @param syst : the \c System_of_eqs.
         * @param f : the field
         * @param jtarget : the index \f$\theta\f$ of the mode that must vanished.
         * @param ktarget : the index \f$\varphi\f$ of the mode that must vanished.
         */
        void add_eq_zero_mode_inf(System_of_eqs& syst, const char* f, int jtarget, int ktarget);

        /**
         * Returns the location of the center of the first sphere.
         */
        double get_a_minus() const { return a_minus; };
        /**
         * Returns the location of the center of the second sphere.
         */
        double get_a_plus() const { return a_plus; };

        /**
         * Computes the surface integral on the first sphere.
         * @param so : the field to be integrated.
         * @returns the surface integral.
         */
        double int_sphere_one(const Scalar& so) const;
        /**
         * Computes the surface integral on the second sphere.
         * @param so : the field to be integrated.
         * @returns the surface integral.
         */
        double int_sphere_two(const Scalar& so) const;
        /**
         * Computes the surface integral at infinity.
         * @param so : the field to be integrated.
         * @returns the surface integral.
         */
        double int_inf(const Scalar& so) const;

        ~Space_bispheric() override; ///< Destructor
        void save(BinarySink&) const override; ///< Modern API.

        Array<int> get_indices_matching_non_std(int, int) const override;
    };
} // namespace Kadath
