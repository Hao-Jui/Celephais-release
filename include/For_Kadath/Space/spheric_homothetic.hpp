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
 *   2026-06-17  Added NS homothetic spherical space for the NOROT stage.
 */

#pragma once

#include <vector>

#include "space.hpp"
#include "For_Kadath/Term_eq/term_eq.hpp"
#include "For_Kadath/Domain/spheric.hpp"
#include "For_Kadath/Metric/metric.hpp"
#include "For_Kadath/Domain/homothetic.hpp"
#include "For_Kadath/IO/binary_sink.hpp"
#include "For_Kadath/IO/binary_source.hpp"

namespace Kadath
{

    class Space_spheric_homothetic : public Space
    {
      public:
        Space_spheric_homothetic(int ttype, const Point& cr, const Dim_array& nbr, const Array<double>& bounds);
        Space_spheric_homothetic(int ttype, const Point& cr, const Dim_array& nbr,
                                 const std::vector<double>& bounds);
        Space_spheric_homothetic(BinarySource& source); ///< Modern API.
        ~Space_spheric_homothetic() override;

        void save(BinarySink& sink) const override; ///< Modern API.

        int nbr_unknowns_from_variable_domains() const override;
        void affecte_coef_to_variable_domains(int& position, int start_index, Array<int>& affected_doms) const override;
        void xx_to_ders_variable_domains(const Array<double>& solution_array, int& position) const override;
        void xx_to_vars_variable_domains(System_of_eqs* system, const Array<double>& solution_array,
                                         int& position) const override;

        void add_eq_ori(System_of_eqs& syst, const char* eq);

        void add_eq(System_of_eqs& syst, const char* eq, const char* rac, const char* rac_der, int nused = -1,
                    Array<int>** pused = nullptr);

        void add_eq_int_inf(System_of_eqs& syst, const char* eq);

        void add_eq_int(System_of_eqs& sys, const int dom, const int bc, const char* eq);

        void add_eq_int_volume(System_of_eqs& syst, int nz, const char* eq);

        Array<int> get_indices_matching_non_std(int dom_index, int bound_index) const override;
    };

} // namespace Kadath
