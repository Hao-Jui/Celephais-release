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
 *   2026-08-10  Cache the GCC/x86 COSSIN plan across a production line batch
 *               and dispatch its N=10--16 forward transform directly.
 */

#include "For_Kadath/Base_spectral/base_spectral.hpp"
#include "For_Kadath/Base_spectral/transform_line_offsets.hpp"
#include "For_Kadath/Array/array.hpp"
#if defined(__linux__) && defined(__x86_64__) && defined(__GNUC__) \
    && !defined(__clang__)
#include "For_Kadath/Base_spectral/base_r2hc.hpp"
#endif

namespace Kadath
{
    void coef_1d(int, Array<double>&);
    bool coef_1d(int, const double*, double*, int, int, int);
#if defined(__linux__) && defined(__x86_64__) && defined(__GNUC__) \
    && !defined(__clang__)
    r2hc_precomp_t& coef_1d_r2hc(int);
#endif

    namespace
    {
        struct Transform1dTrafficState
        {
            Transform1dTrafficSnapshot counts{};
            bool enabled = false;
        };

        Transform1dTrafficState transform_1d_traffic;

        template <typename BasisArrays>
        void coef_dim_into(int ndim, const BasisArrays& bases_1d, int dim, int nbr_coef,
                           const Array<double>& inout, Array<double>& res)
        {
            int after = 1;
            for (int i = 0; i < dim; i++)
                after *= inout.get_size(i);

            int before = 1;
            for (int i = dim + 1; i < ndim; i++)
                before *= inout.get_size(i);

            int nbr_conf = inout.get_size(dim);
            int nbr = (nbr_coef > nbr_conf) ? nbr_coef : nbr_conf;

            // Flat offsets of the line to transform, carried instead of rebuilt.
            // The axes above `dim` weigh the same in both buffers, so one offset
            // covers them; the axes below `dim` carry the transformed axis in
            // their weight and need one offset each. Same traversal order as the
            // Index walk this replaces (see transform_line_offsets.hpp), so the
            // line scratch keeps seeing the same tail. Bit-identical (COO
            // byte-hash gate).
            Transform_line_offsets outer_line(inout.get_dimensions(), dim + 1, ndim, 1, 1);
            Transform_line_offsets inner_line(inout.get_dimensions(), 0, dim, nbr_conf * before,
                                              nbr_coef * before);

            // The basis of axis `dim` is stored over the axes above it, so its
            // flat slot is the outer offset (see Base_spectral::allocate).
            assert(bases_1d[dim]->get_nbr() == static_cast<std::size_t>(before));
            const int* base_data = bases_1d[dim]->get_data();

            Array<double> tab_1d(nbr);
            const double* inout_data = inout.get_data();
            double* res_data = res.get_data();
#if defined(__linux__) && defined(__x86_64__) && defined(__GNUC__) \
    && !defined(__clang__)
            // Ordinary COSSIN is the one growing production line. Resolve its
            // fixed-size plan once for the whole batch instead of re-entering
            // the public wrapper and global plan cache for every line.
            r2hc_precomp_t* cached_cossin_plan = nullptr;
#endif

            // Loop on dimensions before
            for (int i = 0; i < before; i++) {

                const int line_start = outer_line.in_offset();
                // On get la base

                int base = base_data[line_start];
#if defined(__linux__) && defined(__x86_64__) && defined(__GNUC__) \
    && !defined(__clang__)
                r2hc_precomp_t* line_cossin_plan = nullptr;
                if (after > 0 && base == COSSIN && nbr_coef > nbr_conf
                    && (nbr_coef == 12 || nbr_coef == 14 || nbr_coef == 16
                        || nbr_coef == 18)) {
                    if (cached_cossin_plan == nullptr)
                        cached_cossin_plan = &coef_1d_r2hc(nbr_coef - 2);
                    line_cossin_plan = cached_cossin_plan;
                }
#endif
                // Loop on dimensions after :
                for (int j = 0; j < after; j++) {

                    const int line_in = line_start + inner_line.in_offset();
                    const int line_out = line_start + inner_line.out_offset();

                    // The kernel reads the source line and writes the result
                    // line at the traversal stride, where the gather and the
                    // scatter below used to copy through `tab_1d`. It declines
                    // the extent combinations it is not written for; those fall
                    // back to the gathering path unchanged.
#if defined(__linux__) && defined(__x86_64__) && defined(__GNUC__) \
    && !defined(__clang__)
                    const bool cached_cossin_done =
                        line_cossin_plan != nullptr
                        && line_cossin_plan->try_execute_cached_cossin_forward(
                            inout_data + line_in, res_data + line_out,
                            nbr_conf, nbr_coef, before);
                    if (!cached_cossin_done
                        && !coef_1d(base, inout_data + line_in,
                                    res_data + line_out, nbr_conf, nbr_coef,
                                    before)) {
#else
                    if (!coef_1d(base, inout_data + line_in,
                                 res_data + line_out, nbr_conf, nbr_coef,
                                 before)) {
#endif

                        // Gather the 1D line along `dim`. The old form walked an Index
                        // (lit_in = demarre_conf; lit_in.inc(after)) and called
                        // (*inout)(lit_in) per element, recomputing the flat offset
                        // each time. inc(after) advances the flat offset by a constant
                        // stride `before` (= the dim-`dim` flat stride; the dim digit
                        // never overflows since k < nbr_conf), so walk the contiguous
                        // buffer directly.
                        int read_pos = line_in;
                        for (int k = 0; k < nbr_conf; k++) {
                            tab_1d.set(k) = inout_data[read_pos];
                            read_pos += before;
                        }

                        // A growing transform gathers fewer values than the line
                        // scratch holds, and the scratch is reused across lines, so
                        // without this the kernel would read the previous line's
                        // output and its result would depend on call history. The
                        // gathered line is all the data the axis has; the extra
                        // slots exist only because the coefficient representation is
                        // wider. The live case is the COSSIN phi axis, whose
                        // nbr_coefs is nbr_points+2 and whose kernel transforms
                        // exactly nbr-2 samples, so these zeros are never read and
                        // the values are unchanged.
                        for (int k = nbr_conf; k < nbr; k++)
                            tab_1d.set(k) = 0.;

                        // Transformation
                        coef_1d(base, tab_1d);

                        // Scatter back into res with the same constant stride.
                        int write_pos = line_out;
                        for (int k = 0; k < nbr_coef; k++) {
                            res_data[write_pos] = tab_1d(k);
                            write_pos += before;
                        }
                    }
                    inner_line.advance();
                }
                outer_line.advance();
            }
            record_forward_transform_1d(
                static_cast<unsigned long long>(before) * static_cast<unsigned long long>(after));
        }
    } // namespace

    void begin_transform_1d_traffic_profile(bool const enabled)
    {
        transform_1d_traffic = {};
        transform_1d_traffic.enabled = enabled;
    }

    void record_forward_transform_1d(unsigned long long const count)
    {
        if (transform_1d_traffic.enabled)
            transform_1d_traffic.counts.forward += count;
    }

    void record_backward_transform_1d(unsigned long long const count)
    {
        if (transform_1d_traffic.enabled)
            transform_1d_traffic.counts.backward += count;
    }

    Transform1dTrafficSnapshot end_transform_1d_traffic_profile()
    {
        transform_1d_traffic.enabled = false;
        return transform_1d_traffic.counts;
    }

    Array<double> Base_spectral::coef_dim(int dim, int nbr_coef, const Array<double>& inout) const
    {

        Dim_array res_out(inout.get_dimensions());
        res_out.set(dim) = nbr_coef;
        Array<double> res(res_out);

        coef_dim_into(ndim, bases_1d, dim, nbr_coef, inout, res);

        return res;
    }

    Array<double> Base_spectral::coef(const Dim_array& in_coef, const Array<double>& coloc) const
    {
        // The 3D equal-shape path covers the solver's normal transforms. Delay
        // the second allocation until after the first pass, matching the
        // historical allocation/scratch order, then reuse the first buffer for
        // the final pass. Transform and line traversal order are unchanged.
        if (ndim == 3 && in_coef == coloc.get_dimensions()) {
            Array<double> first(in_coef);
            coef_dim_into(ndim, bases_1d, 2, in_coef(2), coloc, first);
            Array<double> second(in_coef);
            coef_dim_into(ndim, bases_1d, 1, in_coef(1), first, second);
            coef_dim_into(ndim, bases_1d, 0, in_coef(0), second, first);
            return first;
        }

        // Unequal shapes retain the existing fresh-array path because each
        // intermediate has a different dimensional contract.
        Array<double> cur = coef_dim(ndim - 1, in_coef(ndim - 1), coloc);
        for (int d = ndim - 2; d >= 0; d--)
            cur = coef_dim(d, in_coef(d), cur);
        return cur;
    }

} // namespace Kadath
