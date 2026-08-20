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

#include "For_Kadath/IO/binary_sink.hpp"
#include "For_Kadath/IO/binary_source.hpp"
#include "space.hpp"
#include "For_Kadath/Domain/spheric_nosym.hpp"
#include "For_Kadath/Domain/spheric_adapted_nosym.hpp"
#include "For_Kadath/List_comp/list_comp.hpp"
#include "For_Kadath/Domain/bispheric_nosym.hpp"

namespace Kadath
{
    /**
     * Spacetime intended for binary neutron stars configurations (see constructor for details about the domains used)
     * \ingroup domain
     */

    class Space_bin_ns_nosym : public Space
    {

      protected:
        friend struct BinaryCoSpaceEquations; // shared add_eq* bodies (src/Space/Shared)
        int n_shells_outer{0}; ///< Number of outer shells.
        int n_shells1{0};      ///< Number of shells around the first nucleus.
        int n_shells2{0};      ///< Number of shells around the second nucleus.

      public:
        int NS1{0};      ///< Starting index of the first spheres.
        int NS2{3};      ///< Starting index of the second spheres.
        int ADAPTED1{1}; ///< Starting index of the first adapted domains.
        int ADAPTED2{4}; ///< Starting index of the second adapted domains.
        int OUTER{6};    ///< Starting index of the outer domains.

        Space_bin_ns_nosym(int ttype, double dist, const std::vector<double>& NS1_bounds,
                     const std::vector<double>& NS2_bounds, const std::vector<double>& outer_bounds, int nr);

        /**
         * Per-domain-resolution variant of the vector-bounds constructor, used by
         * per-domain p-refinement. \c nr_per_domain is indexed by absolute domain
         * index and must have exactly one entry per domain. Radial-only path:
         * angular counts are pinned to the shared base \f$(n_\theta = n_{base},
         * n_\varphi = n_{base}-1)\f$ and the call delegates to the full per-domain
         * \c Dim_array constructor below.
         * @param nr_per_domain [input] : per-domain radial resolution.
         */
        Space_bin_ns_nosym(int ttype, double dist, const std::vector<double>& NS1_bounds,
                     const std::vector<double>& NS2_bounds, const std::vector<double>& outer_bounds,
                     const std::vector<int>& nr_per_domain);

        /**
         * Full per-domain-resolution variant of the vector-bounds constructor,
         * carrying an independent \c Dim_array (radial, theta and phi point
         * counts) for every domain. Used by per-domain angular (theta/phi)
         * p-refinement coupled with split non-conforming seam matching.
         * \c res_per_domain is indexed by absolute domain index (the layout
         * documented above) and must have exactly one entry per domain.
         *
         * Radial counts may differ everywhere. Angular counts may also differ
         * across domains, subject to two constraints enforced here:
         * \li the five bispheric domains (\c OUTER .. \c OUTER+4) must share an
         *     identical \c Dim_array (their mutual seams couple every dimension);
         * \li each adapted pair (\c ADAPTED1/\c ADAPTED1+1 and
         *     \c ADAPTED2/\c ADAPTED2+1) must agree in theta AND phi, because the
         *     pair shares the one deformable star-surface object — that surface
         *     seam stays conforming.
         * Every other fixed-radius star-internal seam is coupled with split
         * \c Eq_matching_non_std, so a theta/phi jump across it stays well-posed.
         * @param res_per_domain [input] : per-domain (nr, ntheta, nphi).
         */
        Space_bin_ns_nosym(int ttype, double dist, const std::vector<double>& NS1_bounds,
                     const std::vector<double>& NS2_bounds, const std::vector<double>& outer_bounds,
                     const std::vector<Dim_array>& res_per_domain);

        Space_bin_ns_nosym(BinarySource& source, bool old = false); ///< Modern API.

        ~Space_bin_ns_nosym() override; ///< Destructor
        void save(BinarySink& sink) const override; ///< Modern API.

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
         * Adds a bulk equation and two matching conditions between nuclii and adapted domains.
         * @param syst : the \c System_of_eqs.
         * @param eq : the string describing the bulk equation.
         * @param rac : the string describing the first matching condition.
         * @param rac_der : the string describing the second matching condition.
         * @param nused : number of components of \c eq to be considered. All the components are used of it is -1.
         * @param pused : pointer on the indexes of the components to be considered. Not used of nused = -1 .
         */
        void add_eq_matter(System_of_eqs& syst, const char* eq, const char* rac, const char* rac_der, int nused = -1,
                           Array<int>** pused = nullptr);

        /**
         * Adds a bulk equation and two matching conditions. The nuclei and inner adapted domains are excluded from the
         * computational space.
         * @param syst : the \c System_of_eqs.
         * @param eq : the string describing the bulk equation.
         * @param rac : the string describing the first matching condition.
         * @param rac_der : the string describing the second matching condition.
         * @param nused : number of components of \c eq to be considered. All the components are used of it is -1.
         * @param pused : pointer on the indexes of the components to be considered. Not used of nused = -1 .
         */
        void add_eq_excised(System_of_eqs& syst, const char* eq, const char* rac, const char* rac_der, int nused = -1,
                            Array<int>** pused = nullptr);

        /**
         * Sets a boundary condition at the inner radius of the first outer shell.
         * @param syst : the \c System_of_eqs.
         * @param eq : the string describing the boundary condition.
         * @param nused : number of components of \c eq to be considered. All the components are used of it is -1.
         * @param pused : pointer on the indexes of the components to be considered. Not used of nused = -1 .
         */
        void add_bc_sphere_one(System_of_eqs& syst, const char* eq, int nused = -1, Array<int>** pused = nullptr);
        /**
         * Sets a boundary condition at the inner radius of the second outer shell.
         * @param syst : the \c System_of_eqs.
         * @param eq : the string describing the boundary condition.
         * @param nused : number of components of \c eq to be considered. All the components are used of it is -1.
         * @param pused : pointer on the indexes of the components to be considered. Not used of nused = -1 .
         */
        void add_bc_sphere_two(System_of_eqs& syst, const char* eq, int nused = -1, Array<int>** pused = nullptr);
        /**
         * Sets a boundary condition at the outer boundary.
         * @param syst : the \c System_of_eqs.
         * @param eq : the string describing the boundary condition.
         * @param nused : number of components of \c eq to be considered. All the components are used of it is -1.
         * @param pused : pointer on the indexes of the components to be considered. Not used of nused = -1 .
         */
        void add_bc_outer(System_of_eqs& syst, const char* eq, int nused = -1, Array<int>** pused = nullptr);

        /**
         * Adds a bulk equation and two matching conditions. The compactified domain is excluded from the computational
         * space.
         * @param syst : the \c System_of_eqs.
         * @param eq : the string describing the bulk equation.
         * @param rac : the string describing the first matching condition.
         * @param rac_der : the string describing the second matching condition.
         * @param nused : number of components of \c eq to be considered. All the components are used of it is -1.
         * @param pused : pointer on the indexes of the components to be considered. Not used of nused = -1 .
         */
        void add_eq_nozec(System_of_eqs& syst, const char* eq, const char* rac, const char* rac_der, int nused = -1,
                          Array<int>** pused = nullptr);

        /**
         * Adds a bulk equation and two matching conditions. The compactified domain is excluded from the computational
         * space.
         * @param syst : the \c System_of_eqs.
         * @param eq : the string describing the bulk equation.
         * @param rac : the string describing the first matching condition.
         * @param rac_der : the string describing the second matching condition.
         * @param used : list of components used
         */
        void add_eq_nozec(System_of_eqs& syst, const char* eq, const char* rac, const char* rac_der,
                          const List_comp& used);

        /**
         * Adds a bulk equation and two matching conditions. The outer shells are excluded from the computational space.
         * @param syst : the \c System_of_eqs.
         * @param eq : the string describing the bulk equation.
         * @param rac : the string describing the first matching condition.
         * @param rac_der : the string describing the second matching condition.
         * @param nused : number of components of \c eq to be considered. All the components are used of it is -1.
         * @param pused : pointer on the indexes of the components to be considered. Not used of nused = -1 .
         */
        void add_eq_noshell(System_of_eqs& syst, const char* eq, const char* rac, const char* rac_der, int nused = -1,
                            Array<int>** pused = nullptr);

        /**
         * Adds a bulk equation and two matching conditions. The outer shells are excluded from the computational space.
         * @param syst : the \c System_of_eqs.
         * @param eq : the string describing the bulk equation.
         * @param rac : the string describing the first matching condition.
         * @param rac_der : the string describing the second matching condition.
         * @param used : list of components used
         */
        void add_eq_noshell(System_of_eqs& syst, const char* eq, const char* rac, const char* rac_der,
                            const List_comp& used);

        int nbr_unknowns_from_variable_domains() const override;
        bool describe_variable_domain_blocks(
            std::vector<VariableDomainBlock>& blocks) const override;
        void affecte_coef_to_variable_domains(int&, int, Array<int>&) const override;
        bool supports_packed_variable_domain_jacobian() const override { return true; }
        bool affecte_coef_to_variable_domains_lane(int&, int, int, int, Array<int>&) const override;
        void restore_scalar_variable_domain_derivatives() const override;
        void xx_to_ders_variable_domains(const Array<double>&, int&) const override;
        void xx_to_vars_variable_domains(System_of_eqs*, const Array<double>&, int&) const override;
        Array<int> get_indices_matching_non_std(int dom, int bound) const override;

        /**
         * Adds an equation being a surface integral at infinity.
         * @param syst : the \c System_of_eqs.
         * @param eq : the string describing the equation (should contain something like integ(f)=b)
         */
        void add_eq_int_inf(System_of_eqs& syst, const char* eq);
        /**
         * Adds an equation being the value of some field at the origin of the first nucleus
         * @param syst : the \c System_of_eqs.
         * @param eq : the string describing the quantity that must be zero at the origin
         */
        void add_eq_ori_one(System_of_eqs& syst, const char* eq);
        /**
         * Adds an equation being the value of some field at the origin of the first nucleus
         * @param syst : the \c System_of_eqs.
         * @param eq : the string describing the quantity that must be zero at the origin
         */
        void add_eq_ori_two(System_of_eqs& syst, const char* eq);
        /**
         * Adds an equation being a volume integral over contiguous domains.
         * @param syst : the \c System_of_eqs.
         * @param dmin : the integral is taken beginning at this domain.
         * @param dmax : the integral is taken up to this domain.
         * @param eq : the string describing the equation (should contain something like integvolume(f)=b)
         */
        void add_eq_int_volume(System_of_eqs& sys, int dmin, int dmax, const char* nom);
        /**
         * Adds an equation being a surface integral over the outer boundary of the first inner adapted domain
         * @param syst : the \c System_of_eqs.
         * @param dmin : the integral is taken beginning at this domain.
         * @param dmax : the integral is taken up to this domain.
         * @param eq : the string describing the equation (should contain something like integvolume(f)=b)
         */
        void add_eq_int_outer_sphere_one(System_of_eqs& sys, const char* nom);
        /**
         * Adds an equation being a surface integral over the outer boundary of the second inner adapted domain
         * @param syst : the \c System_of_eqs.
         * @param dmin : the integral is taken beginning at this domain.
         * @param dmax : the integral is taken up to this domain.
         * @param eq : the string describing the equation (should contain something like integvolume(f)=b)
         */
        void add_eq_int_outer_sphere_two(System_of_eqs& sys, const char* nom);

        int get_n_shells_outer() const { return n_shells_outer; }
    };
} // namespace Kadath
