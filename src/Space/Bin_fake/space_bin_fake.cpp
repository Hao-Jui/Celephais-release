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

#include "For_Kadath/Array/headcpp.hpp"
#include "For_Kadath/Array/exceptions.hpp"
#include "For_Kadath/Space/bin_fake.hpp"
#include "For_Kadath/Utilities/utilities.hpp"
#include "For_Kadath/Array/point.hpp"
#include "For_Kadath/Scalar/scalar.hpp"
#include "For_Kadath/System_of_eqs/system_of_eqs.hpp"
#include "For_Kadath/Utilities/name_tools.hpp"
namespace Kadath
{

    double eta_lim_chi(double chi, double rext, double a, double eta_c);
    double chi_lim_eta(double chi, double rext, double a, double chi_c);

    double zerosec(double (*f)(double, const Param&), const Param& parf, double x1, double x2, double precis,
                   int nitermax, int& niter);

    double func_afake(double aa, const Param& par)
    {
        double r1 = par.get_double(0);
        double r2 = par.get_double(1);
        double d = par.get_double(2);
        return (sqrt(aa * aa + r1 * r1) + sqrt(aa * aa + r2 * r2) - d);
    }

    Space_bin_fake::Space_bin_fake(int ttype, double dist, double r1, double r2, double rbi, double rext, int nr)
    {

        ndim = 3;

        nbr_domains = 9;
        type_base = ttype;
        domains = new Domain*[nbr_domains];

        Dim_array res(ndim);
        res.set(0) = nr;
        res.set(1) = nr;
        res.set(2) = nr - 1;
        Dim_array res_bi(ndim);
        res_bi.set(0) = nr;
        res_bi.set(1) = nr;
        res_bi.set(2) = nr;

        // Bispheric :
        // Computation of aa
        Param par_a;
        par_a.add_double(r1, 0);
        par_a.add_double(r2, 1);
        par_a.add_double(dist, 2);
        double a_min = 0;
        double a_max = dist / 2.;
        double precis = PRECISION;
        int nitermax = 500;
        int niter;
        double aa = zerosec(func_afake, par_a, a_min, a_max, precis, nitermax, niter);
        double eta_plus = asinh(aa / r2);
        double eta_minus = -asinh(aa / r1);

        double chi_c = 2 * atan(aa / rbi);
        double eta_c = log((1 + rbi / aa) / (rbi / aa - 1));
        double eta_lim = eta_c / 2.;
        double chi_lim = chi_lim_eta(eta_lim, rbi, aa, chi_c);

        // Star 1
        Point center_minus(ndim);
        center_minus.set(1) = aa * cosh(eta_minus) / sinh(eta_minus);
        domains[0] = new Domain_nucleus(0, ttype, r1, center_minus, res);

        // Star 2
        Point center_plus(ndim);
        center_plus.set(1) = aa * cosh(eta_plus) / sinh(eta_plus);
        domains[1] = new Domain_nucleus(1, ttype, r2, center_plus, res);

        // Bispheric part
        domains[2] = new Domain_bispheric_chi_first(2, ttype, aa, eta_minus, rbi, chi_lim, res_bi);
        domains[3] = new Domain_bispheric_rect(3, ttype, aa, rbi, eta_minus, -eta_lim, chi_lim, res_bi);
        domains[4] = new Domain_bispheric_eta_first(4, ttype, aa, rbi, -eta_lim, eta_lim, res_bi);
        domains[5] = new Domain_bispheric_rect(5, ttype, aa, rbi, eta_plus, eta_lim, chi_lim, res_bi);
        domains[6] = new Domain_bispheric_chi_first(6, ttype, aa, eta_plus, rbi, chi_lim, res_bi);

        Point center(3);
        // Shell
        domains[7] = new Domain_shell_surr(7, ttype, rbi, rext, center, res);

        // Compactified domain
        domains[8] = new Domain_compact(8, ttype, rext, center, res);
    }

    Space_bin_fake::Space_bin_fake(BinarySource& source)
    {
        nbr_domains = source.read<int>();
        ndim = source.read<int>();
        type_base = source.read<int>();
        domains = new Domain*[nbr_domains];
        // Star 1
        domains[0] = new Domain_nucleus(0, source);

        // Star 2
        domains[1] = new Domain_nucleus(1, source);

        // Bispheric
        domains[2] = new Domain_bispheric_chi_first(2, source);
        domains[3] = new Domain_bispheric_rect(3, source);
        domains[4] = new Domain_bispheric_eta_first(4, source);
        domains[5] = new Domain_bispheric_rect(5, source);
        domains[6] = new Domain_bispheric_chi_first(6, source);

        // Shell
        domains[7] = new Domain_shell_surr(7, source);

        // Compactified
        domains[8] = new Domain_compact(8, source);
    }

    Space_bin_fake::~Space_bin_fake() {}

    void Space_bin_fake::save(BinarySink& sink) const
    {
        sink.write<int>(nbr_domains);
        sink.write<int>(ndim);
        sink.write<int>(type_base);
        for (int i = 0; i < nbr_domains; i++)
            domains[i]->save(sink);
    }

    Array<int> Space_bin_fake::get_indices_matching_non_std(int dom, int bound) const
    {

        if (dom == 0) {
            // First sphere ;
            Array<int> res(2, 2);
            switch (bound) {
                case OUTER_BC:
                    res.set(0, 0) = 2; // Matching with chi first
                    res.set(1, 0) = INNER_BC;
                    res.set(0, 1) = 3; // Matching with rect
                    res.set(1, 1) = INNER_BC;
                    break;
                default:
                    KADATH_THROW("Bad bound in Space_bin_fake::get_indices_matching_non_std");
            }
            return res;
        }

        if (dom == 1) {
            // second sphere ;
            Array<int> res(2, 2);
            switch (bound) {
                case OUTER_BC:
                    res.set(0, 0) = 5; // Matching with rect
                    res.set(1, 0) = INNER_BC;
                    res.set(0, 1) = 6; // Matching with chi_first
                    res.set(1, 1) = INNER_BC;
                    break;
                default:
                    KADATH_THROW("Bad bound in Space_bin_fake::get_indices_matching_non_std");
            }
            return res;
        }

        if (dom == 2) {
            // first chi first :
            Array<int> res(2, 1);
            switch (bound) {
                case INNER_BC:
                    res.set(0, 0) = 0; // First sphere
                    res.set(1, 0) = OUTER_BC;
                    break;
                case OUTER_BC:
                    res.set(0, 0) = 7; // Compactified domain or first shell
                    res.set(1, 0) = INNER_BC;
                    break;
                default:
                    KADATH_THROW("Bad bound in Space_bin_fake::get_indices_matching_non_std");
            }
            return res;
        }

        if (dom == 3) {
            // first rect :
            Array<int> res(2, 1);
            switch (bound) {
                case INNER_BC:
                    res.set(0, 0) = 0; // First sphere
                    res.set(1, 0) = OUTER_BC;
                    break;
                case OUTER_BC:
                    res.set(0, 0) = 7; // Compactified domain or first shell
                    res.set(1, 0) = INNER_BC;
                    break;
                default:
                    KADATH_THROW("Bad bound in Space_bin_fake::get_indices_matching_non_std");
            }
            return res;
        }

        if (dom == 4) {
            // eta first
            Array<int> res(2, 1);
            switch (bound) {
                case OUTER_BC:
                    res.set(0, 0) = 7; // Compactified domain or first shell
                    res.set(1, 0) = INNER_BC;
                    break;
                default:
                    KADATH_THROW("Bad bound in Space_bin_fake::get_indices_matching_non_std");
            }
            return res;
        }

        if (dom == 5) {
            // second rect :
            Array<int> res(2, 1);
            switch (bound) {
                case INNER_BC:
                    res.set(0, 0) = 1; // Second sphere
                    res.set(1, 0) = OUTER_BC;
                    break;
                case OUTER_BC:
                    res.set(0, 0) = 7; // Compactified domain or first shell
                    res.set(1, 0) = INNER_BC;
                    break;
                default:
                    KADATH_THROW("Bad bound in Space_bin_fake::get_indices_matching_non_std");
            }
            return res;
        }

        if (dom == 6) {
            // second chi first :
            Array<int> res(2, 1);
            switch (bound) {
                case INNER_BC:
                    res.set(0, 0) = 1; // second nucleus
                    res.set(1, 0) = OUTER_BC;
                    break;
                case OUTER_BC:
                    res.set(0, 0) = 7; // Compactified domain or first shell
                    res.set(1, 0) = INNER_BC;
                    break;
                default:
                    KADATH_THROW("Bad bound in Space_bin_fake::get_indices_matching_non_std");
            }
            return res;
        }

        if (dom == 7) {
            // outer shell :
            Array<int> res(2, 5);
            switch (bound) {
                case INNER_BC:
                    res.set(0, 0) = 2; // first chi first
                    res.set(0, 1) = 3; // first rect
                    res.set(0, 2) = 4; // eta first
                    res.set(0, 3) = 5; // second rect
                    res.set(0, 4) = 6; // second chi first
                    // Outer BC for all :
                    for (int i = 0; i < 5; i++)
                        res.set(1, i) = OUTER_BC;
                    break;
                default:
                    KADATH_THROW("Bad bound in Space_bin_fake::get_indices_matching_non_std");
            }
            return res;
        }

        KADATH_THROW("Bad domain in Space_bispheric::get_indices_matching_non_std");
    }

} // namespace Kadath
