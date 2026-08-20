/*
    Copyright 2022 Samuel Tootle

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
#include "For_Kadath/Term_eq/term_eq.hpp"
#include "For_Kadath/Domain/spheric.hpp"
#include "For_Kadath/Metric/metric.hpp"
#include "For_Kadath/Domain/adapted.hpp"
#include "For_Kadath/IO/binary_sink.hpp"
#include "For_Kadath/IO/binary_source.hpp"
#include <vector>

namespace Kadath
{
    /**
     * @class Space_KerrSchild_bh
     * @brief 3D black hole spacetime with adapted inner boundary.
     * @ingroup domain
     */
    class Space_KerrSchild_bh : public Space
    {
      public:
        /**
         * Standard constructor.
         * @param ttype      Type of basis.
         * @param cr         Center of coordinates.
         * @param nbr        Number of points in each domain.
         * @param BH_bounds  Radii of the various shells.
         */
        Space_KerrSchild_bh(int ttype, const Point& cr, const Dim_array& nbr, const std::vector<double>& BH_bounds);
        Space_KerrSchild_bh(BinarySource& source); ///< Modern API.
        void add_eq(System_of_eqs& syst, const char* eq, const char* rac, const char* rac_der, int nused = -1,
                    Array<int>** pused = nullptr);
        void add_bc_bh(System_of_eqs& syst, const char* eq, int nused = -1, Array<int>** pused = nullptr);
        void add_bc_inf(System_of_eqs& syst, const char* eq, int nused = -1, Array<int>** pused = nullptr);
        void add_eq_int_inf(System_of_eqs&, const char*);
        void add_eq_int_bh(System_of_eqs&, const char*);
        void add_eq_int_volume(System_of_eqs&, int, int, const char*);
        void add_eq_zero_mode_inf(System_of_eqs&, const char*, int, int);

        ~Space_KerrSchild_bh() override; ///< Destructor
        void save(BinarySink& sink) const override; ///< Modern API.

        int nbr_unknowns_from_variable_domains() const override;
        void affecte_coef_to_variable_domains(int&, int, Array<int>&) const override;
        void xx_to_ders_variable_domains(const Array<double>&, int&) const override;
        void xx_to_vars_variable_domains(System_of_eqs*, const Array<double>&, int&) const override;
        Array<int> get_indices_matching_non_std(int, int) const override;
    };
} // namespace Kadath
