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
 *   2026-08-06  RAII/span modernization.
 */

#include "For_Kadath/Param/param.hpp"
#include "For_Kadath/Array/exceptions.hpp"

#include <sstream>
#include <utility>
// Headers C
#include "For_Kadath/Array/headcpp.hpp"
#include <assert.h>
namespace Kadath
{

    //------------------------//
    //	Constructor	  //
    //------------------------//

    Param::Param() : n_int(0), n_double(0) {}

    //----------------------//
    //	Destructor	//
    //----------------------//

    Param::~Param() = default;

    //------------------------------------//
    //		int storage		  //
    //------------------------------------//

    // Total number of stored int
    // --------------------------

    int Param::get_n_int() const
    {
        return n_int;
    }

    // Addition
    // --------

    void Param::add_int(int ti, int index)
    {

        if (index >= n_int) { // p_int must be rescaled
            int n_int_nouveau = index + 1;
            auto p_int_nouveau = std::make_unique_for_overwrite<int[]>(n_int_nouveau);

            for (int i = 0; i < n_int; i++)
                p_int_nouveau[i] = p_int[i];
            p_int_nouveau[index] = ti;

            p_int = std::move(p_int_nouveau);
            n_int = n_int_nouveau;
        } else {

            if (p_int[index] != 0) {
                std::ostringstream oss;
                oss << "Param::add_int : the position " << index << " is already occupied !" << endl;
                KADATH_THROW(oss.str());
            } else {
                p_int[index] = ti;
            }
        }
    }

    // Extraction
    // ----------

    const int& Param::get_int(int index) const
    {

        assert(index >= 0);
        assert(index < n_int);

        return (p_int[index]);
    }

    //------------------------------------//
    //		double storage		  //
    //------------------------------------//

    // Total number of stored doubles
    // ------------------------------

    int Param::get_n_double() const
    {
        return n_double;
    }

    // Addition
    // --------

    void Param::add_double(double ti, int index)
    {

        if (index >= n_double) { // p_double must be rescaled

            int n_double_nouveau = index + 1;
            auto p_double_nouveau = std::make_unique_for_overwrite<double[]>(n_double_nouveau);

            for (int i = 0; i < n_double; i++)
                p_double_nouveau[i] = p_double[i];
            p_double_nouveau[index] = ti;

            p_double = std::move(p_double_nouveau);
            n_double = n_double_nouveau;
        } else {
            if (p_double[index] != 0) {
                std::ostringstream oss;
                oss << "Param::add_double : the position " << index << " is already occupied !" << endl;
                KADATH_THROW(oss.str());
            } else {
                p_double[index] = ti;
            }
        }
    }

    // Extraction
    // ----------

    const double& Param::get_double(int index) const
    {

        assert(index >= 0);
        assert(index < n_double);

        return (p_double[index]);
    }
} // namespace Kadath
