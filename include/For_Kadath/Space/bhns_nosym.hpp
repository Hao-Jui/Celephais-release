/*
    Copyright 2020 Samuel Tootle

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
#include "For_Kadath/Domain/homothetic_nosym.hpp"
#include "For_Kadath/List_comp/list_comp.hpp"
#include "For_Kadath/Domain/bispheric_nosym.hpp"

namespace Kadath
{
    /**
     * Spacetime intended for binary neutron stars configurations (see constructor for details about the domains used)
     * \ingroup domain
     */

    class Space_bhns_nosym : public Space
    {

      protected:
        friend struct BinaryCoSpaceEquations; // shared add_eq* bodies (src/Space/Shared)
        int n_shells_outer{0};  ///< Number of outer shells.
        int n_shells1{0};       ///< Number of shells outside the NS bulk, between ROUT and r_bisph.
        int n_shells2{0};       ///< Number of shells outside the BH bulk, between ROUT and r_bisph.

      public:
        int BH{3};        ///< Starting index of the first spheres.
        int NS{0};        ///< Starting index of the second spheres.
        int ADAPTEDNS{1}; ///< Starting index of the NS adapted domains.
        int ADAPTEDBH{4}; ///< Starting index of the BH adapted domains.
        int OUTER{6};     ///< Starting index of the outer domains.

        /**
         * Constructor with an outer shell surrounding the bispheric region; stars are initial spherical.
         * @param ttype [input] : the type of basis.
         * @param dist [input] : distance \f$ d \f$ between the centers of the two spheres.
         * @param NS_bounds [input] : the boundaries of the various NS domains from Nucleus to r_bisph
         * @param BH_bounds [input] : the boundaries of the various BH domains from Nucleus to r_bisph
         * @param outer_bounds [input] : the boundaries of the various outer shell domains from OUTER+5 to ndom-1
         * @param nr [input] : number of points in each dimension
         *
         * The various domains are then :
         * \li One \c Domain_nucleus_nosym, of radius  rinstar1, centered at \f$ x_1 \f$.
         * \li One \c Domain_shell_outer_adapted_nosym  centered at \f$ x_1 \f$.
         * \li One \c Domain_shell_inner_homothetic_nosym  centered at \f$ x_1 \f$.
         * \li One \c Domain_nucleus_nosym, of radius  rinstar2, centered at \f$ x_2 \f$.
         * \li One \c Domain_shell_outer_adapted_nosym  centered at \f$ x_2 \f$.
         * \li One \c Domain_shell_inner_homothetic_nosym  centered at \f$ x_2 \f$.
         * \li One \c Domain_bispheric_chi_first_nosym near the first sphere.
         * \li One \c Domain_bispheric_rect_nosym near the first sphere.
         * \li One \c Domain_bispheric_eta_first_nosym inbetween the two spheres.
         * \li One \c Domain_bispheric_rect_nosym near the second sphere.
         * \li One \c Domain_bispheric_chi_first_nosym near the second sphere.
         * \li One \c Domain_shell_nosym centered on the origin
         * \li One \c Domain_compact_nosym centered on the origin, with inner radius \f$ R \f$.
         */
        Space_bhns_nosym(int ttype, double dist, const std::vector<double>& NS_bounds,
                         const std::vector<double>& BH_bounds, const std::vector<double>& outer_bounds, int nr);

        /**
         * Per-domain-resolution variant (see Space_bhns): each domain carries its
         * own \c Dim_array, used by per-domain AMR p-refinement (including the
         * bispheric phi block). \c res_per_domain is indexed by absolute domain
         * index; the five bispheric domains must share one \c Dim_array and the NS
         * adapted / BH homothetic pairs must each agree in theta and phi.
         */
        Space_bhns_nosym(int ttype, double dist, const std::vector<double>& NS_bounds,
                         const std::vector<double>& BH_bounds, const std::vector<double>& outer_bounds,
                         const std::vector<Dim_array>& res_per_domain);

        Space_bhns_nosym(BinarySource& source, bool oldspace = false); ///< Modern API.

        ~Space_bhns_nosym() override; ///< Destructor
        void save(BinarySink& sink) const override; ///< Modern API.
        void add_eq_int_BH(System_of_eqs& sys, const char* nom);

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
        void affecte_coef_to_variable_domains(int&, int, Array<int>&) const override;
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
         * Adds an equation being the value of some field at the origin of the NS nucleus
         * @param syst : the \c System_of_eqs.
         * @param eq : the string describing the quantity that must be zero at the origin
         */
        void add_eq_ori_NS(System_of_eqs& syst, const char* eq);
        /**
         * Adds an equation being a volume integral over contiguous domains.
         * @param syst : the \c System_of_eqs.
         * @param dmin : the integral is taken beginning at this domain.
         * @param dmax : the integral is taken up to this domain.
         * @param eq : the string describing the equation (should contain something like integvolume(f)=b)
         */
        void add_eq_int_volume(System_of_eqs& sys, int dmin, int dmax, const char* nom);
        /**
         * Adds an equation being a surface integral over the outer boundary of the NS adapted domain
         * @param syst : the \c System_of_eqs.
         * @param dmin : the integral is taken beginning at this domain.
         * @param dmax : the integral is taken up to this domain.
         * @param eq : the string describing the equation (should contain something like integvolume(f)=b)
         */
        void add_eq_int_outer_NS(System_of_eqs& sys, const char* nom);

        int get_n_shells_outer() const { return n_shells_outer; }
    };
} // namespace Kadath
