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

#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <map>
#include <string>
#include <typeinfo>
#include <utility>
#include <vector>

namespace Kadath
{

    // I/O streams
    using std::cerr;
    using std::cout;
    using std::endl;
    using std::fixed;
    using std::fstream;
    using std::ifstream;
    using std::ios;
    using std::istream;
    using std::ofstream;
    using std::ostream;
    using std::scientific;
    using std::setprecision;
    using std::setw;

    // Strings
    using std::getline;
    using std::string;
    using std::stringstream;
    using std::to_string;

    // I/O manipulators
    using std::flush;
    using std::left;
    using std::right;
    using std::setfill;
    using std::uppercase;

    // Containers
    using std::array;
    using std::make_pair;
    using std::map;
    using std::pair;
    using std::vector;

    // Algorithms
    using std::max;
    using std::min;
    using std::move;
    using std::sort;
    using std::swap;

    constexpr double PRECISION = 1e-14;

    // Math
    using std::abs;
    using std::acos;
    using std::asin;
    using std::atan;
    using std::atanh;
    using std::cos;
    using std::cosh;
    using std::exp;
    using std::fabs;
    using std::log;
    using std::log10;
    using std::pow;
    using std::sin;
    using std::sinh;
    using std::sqrt;
    using std::tan;

} // namespace Kadath