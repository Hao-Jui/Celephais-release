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

#pragma once
#include "stdio.h"
#include <cmath>
#include <span>
#include <string.h>

namespace Kadath
{
    class Param;
    class Point;

    /** Computes x0 in a given interval [a, b] such that f(x0, par) = 0
     *  @param (*f)(double, const Param\&) [input], parameters are
     *		    stored in an object of the Lorene class
     *  @param par [input] Parameters of the function f.
     *  @param a [input] Lower bound of the search interval [a, b]
     *  @param b [input] Higher bound of the search interval [a, b]
     *  @param precis [input] Required precision in the determination of x0 :
     *			the returned solution will be x0 +/- precis
     *  @param nitermax [input] Maximum number of iterations in the secant
     *			    method to compute x0.
     *  @param niter [output] Number of iterations effectively used in computing x0
     *  @return x0 (zero of function f)
     */
    double zerosec(double (*f)(double, const Param&), const Param& par, double a, double b, double precis, int nitermax,
                   int& niter);

    /** Writes integer(s) into a binary file according to the
     *  big endian convention.
     *
     *  Similar to fwrite but it ensures that the binary file is
     *  written in the big endian format, whatever the system is
     *  using little endian or big endian.
     *	@param aa [input] integer span to be written
     *	@param fich [input] binary file (must have been opened by fopen )
     *	@return number of integers effectively written in the file
     */
    int fwrite_be(std::span<const int> aa, FILE* fich);

    /** Writes double precision number(s) into a binary file according to the
     *  big endian convention.
     *
     *  Same as the previous function but for double (bytes must be 8)
     */
    int fwrite_be(std::span<const double> aa, FILE* fich);

    /** Reads integer(s) from a binary file according to the
     *  big endian convention.
     */
    int fread_be(std::span<int> aa, FILE* fich);

    /** Reads double precision number(s) from a binary file according to the
     *  big endian convention.
     */
    int fread_be(std::span<double> aa, FILE* fich);
} // namespace Kadath
