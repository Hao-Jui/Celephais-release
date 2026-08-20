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

#include "space.hpp"
#include "For_Kadath/Domain/spheric.hpp"
#include "For_Kadath/Domain/bispheric.hpp"
#include "For_Kadath/IO/binary_sink.hpp"
#include "For_Kadath/IO/binary_source.hpp"

namespace Kadath
{
    /**
     * Spacetime intended for fake binary neutron star configurations (see constructor for details about the domains
     * used). By fake one means that the shape of the stars kept fixed to spherica. The radii of the holes are
     * variables.
     * \ingroup domain
     */

    class Space_bin_fake : public Space
    {
      public:
        /**
         * Standard constructor.
         * @param ttype [input] : the type of basis.
         * @param dist [input] : distance \f$ d \f$ between the centers of the two spheres.
         * @param r1 [input] : radius \f$ r_1 \f$ of the first sphere.
         * @param r2 [input] : radius \f$ r_2 \f$ of the second sphere.
         * @param rbi [input] : radius of the outer boundary of the bispherical part.
         * @param rext [input] : inner radius of the compactified domain.
         * @param nr [input] : number of points in each dimension
         *
         * The various domains are then :
         * \li One \c Domain_nucleus, of radius \f$ r_1 \f$, centered at \f$ x_1 \f$.
         * \li One \c Domain_nucleus, of radius \f$ r_2 \f$, centered at \f$ x_2 \f$.
         * \li One \c Domain_bispheric_chi_first near the first sphere.
         * \li One \c Domain_bispheric_rect near the first sphere.
         * \li One \c Domain_bispheric_eta_first inbetween the two spheres.
         * \li One \c Domain_bispheric_rect near the second sphere.
         * \li One \c Domain_bispheric_chi_first near the second sphere.
         * \li One \c Domain_shell_surr centered on the origin.
         * \li One \c Domain_compact centered on the origin.
         */
        Space_bin_fake(int ttype, double dist, double r1, double r2, double rbi, double rext, int nr);
        Space_bin_fake(BinarySource& source); ///< Modern API.

        ~Space_bin_fake() override; ///< Destructor
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
         * Adds an equation being a surface integral at infinity.
         * @param syst : the \c System_of_eqs.
         * @param eq : the string describing the equation (should contain something like integ(f)=b)
         */
        void add_eq_int_inf(System_of_eqs& syst, const char* eq);
        Array<int> get_indices_matching_non_std(int dom, int bound) const override;
    };
} // namespace Kadath
