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
 *   2026-08-03  Zero the un-gathered tail of the line scratch so a growing
 *               transform cannot consume the previous line's output.
 *   2026-08-04  Let the kernel read the source line and write the result line
 *               at the traversal stride, so the line scratch round trip only
 *               runs for the extent combinations the kernels decline.
 */

#include "For_Kadath/Base_spectral/base_spectral.hpp"
#include "For_Kadath/Base_spectral/transform_line_offsets.hpp"
#include "For_Kadath/Array/array.hpp"

namespace Kadath
{
    void coef_i_1d(int, Array<double>&);
    bool coef_i_1d(int, const double*, double*, int, int, int);

    namespace
    {
        template <typename BasisArrays>
        void coef_i_dim_into(int ndim, const BasisArrays& bases_1d, int dim,
                             int nbr_conf, const Array<double>& inout, Array<double>& res)
        {
            int after = 1;
            for (int i = 0; i < dim; i++)
                after *= inout.get_size(i);

            int before = 1;
            for (int i = dim + 1; i < ndim; i++)
                before *= inout.get_size(i);

            int nbr_coef = inout.get_size(dim);
            int nbr = (nbr_coef > nbr_conf) ? nbr_coef : nbr_conf;

            // Flat offsets of the line to transform, carried instead of rebuilt;
            // see the forward transform in coef.cpp and transform_line_offsets.hpp.
            // Same traversal order as the Index walk this replaces, so the line
            // scratch keeps seeing the same tail. Bit-identical (COO byte-hash gate).
            Transform_line_offsets outer_line(inout.get_dimensions(), dim + 1, ndim, 1, 1);
            Transform_line_offsets inner_line(inout.get_dimensions(), 0, dim, nbr_coef * before,
                                              nbr_conf * before);

            // The basis of axis `dim` is stored over the axes above it, so its
            // flat slot is the outer offset (see Base_spectral::allocate).
            assert(bases_1d[dim]->get_nbr() == static_cast<std::size_t>(before));
            const int* base_data = bases_1d[dim]->get_data();

            Array<double> tab_1d(nbr);
            const double* inout_data = inout.get_data();
            double* res_data = res.get_data();

            // Loop on dimensions before
            for (int i = 0; i < before; i++) {

                const int line_start = outer_line.in_offset();
                // On get la base

                int base = base_data[line_start];
                // Loop on dimensions after :
                for (int j = 0; j < after; j++) {

                    const int line_in = line_start + inner_line.in_offset();
                    const int line_out = line_start + inner_line.out_offset();

                    // The kernel reads the source line and writes the result
                    // line at the traversal stride, where the gather and the
                    // scatter below used to copy through `tab_1d`; see the
                    // forward transform in coef.cpp. It declines the extent
                    // combinations it is not written for.
                    if (!coef_i_1d(base, inout_data + line_in, res_data + line_out, nbr_coef, nbr_conf,
                                   before)) {

                        // Gather the 1D line along `dim` directly from the contiguous
                        // buffer: the old Index walk (lit_in = demarre_coef;
                        // lit_in.inc(after)) advances the flat offset by a constant
                        // stride `before` (the dim digit never overflows, k < nbr_coef).
                        int read_pos = line_in;
                        for (int k = 0; k < nbr_coef; k++) {
                            tab_1d.set(k) = inout_data[read_pos];
                            read_pos += before;
                        }

                        // Keep the kernel's input a function of the gathered line
                        // alone; see the forward transform in coef.cpp. The inverse
                        // shrinks its axis on the live paths, so this loop is empty
                        // there.
                        for (int k = nbr_coef; k < nbr; k++)
                            tab_1d.set(k) = 0.;

                        // Transformation
                        coef_i_1d(base, tab_1d);

                        // Scatter back into res with the same constant stride.
                        int write_pos = line_out;
                        for (int k = 0; k < nbr_conf; k++) {
                            res_data[write_pos] = tab_1d(k);
                            write_pos += before;
                        }
                    }
                    inner_line.advance();
                }
                outer_line.advance();
            }
            record_backward_transform_1d(
                static_cast<unsigned long long>(before) * static_cast<unsigned long long>(after));
        }
    } // namespace

    Array<double> Base_spectral::coef_i_dim(int dim, int nbr_conf, const Array<double>& inout) const
    {

        Dim_array res_out(inout.get_dimensions());
        res_out.set(dim) = nbr_conf;
        Array<double> res(res_out);

        coef_i_dim_into(ndim, bases_1d, dim, nbr_conf, inout, res);

        return res;
    }

    Array<double> Base_spectral::coef_i(const Dim_array& in_conf, const Array<double>& cof) const
    {
        // Match the forward 3D fast path: preserve allocation timing for the
        // first two passes, then reuse the first buffer for the final pass.
        if (ndim == 3 && in_conf == cof.get_dimensions()) {
            Array<double> first(in_conf);
            coef_i_dim_into(ndim, bases_1d, 0, in_conf(0), cof, first);
            Array<double> second(in_conf);
            coef_i_dim_into(ndim, bases_1d, 1, in_conf(1), first, second);
            coef_i_dim_into(ndim, bases_1d, 2, in_conf(2), second, first);
            return first;
        }

        // Unequal shapes retain the existing fresh-array path.
        Array<double> cur = coef_i_dim(0, in_conf(0), cof);
        for (int d = 1; d < ndim; d++)
            cur = coef_i_dim(d, in_conf(d), cur);
        return cur;
    }

} // namespace Kadath
