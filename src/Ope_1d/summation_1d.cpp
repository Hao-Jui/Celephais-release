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
 *   2026-08-11  Reused bounded worker-local spectral-summation scratch.
 *   2026-08-12  Evaluated Chebyshev lines directly at constant stride and added
 *               exact-order four-point spectral lanes.
 */

#include "For_Kadath/Array/point.hpp"
#include "For_Kadath/Array/exceptions.hpp"
#include "For_Kadath/Array/index.hpp"
#include "For_Kadath/Base_spectral/base_spectral.hpp"
#include "For_Kadath/Array/array.hpp"

#include <array>
#include <cstddef>
#include <memory>
#include <vector>
namespace Kadath
{
    double summation_1d(int base, double xx, const Array<double>& tab);

    namespace
    {
        // One retained slot covers the non-recursive production call graph.
        // Re-entry falls back to call-local storage rather than multiplying each
        // worker's persistent high-water capacity.
        constexpr std::size_t summation_scratch_slots = 1;
        constexpr std::size_t summation_scratch_max_dimensions = 3;
        constexpr std::size_t summation_scratch_max_retained_bytes = 64U * 1024U;
        constexpr std::size_t summation_scratch_max_retained_doubles =
            summation_scratch_max_retained_bytes / sizeof(double);

        struct SummationWorkspace {
            std::vector<std::unique_ptr<Array<double>>> stages;
            std::vector<std::unique_ptr<Array<double>>> lines;
            std::vector<std::vector<double>> point_stages;

            static Array<double>& ensure(std::unique_ptr<Array<double>>& buffer, const Dim_array& dimensions)
            {
                if (buffer == nullptr || buffer->get_dimensions() != dimensions)
                    buffer = std::make_unique<Array<double>>(dimensions);
                return *buffer;
            }

            static Array<double>& ensure(std::unique_ptr<Array<double>>& buffer, int length)
            {
                if (buffer == nullptr || buffer->get_ndim() != 1 || buffer->get_size(0) != length)
                    buffer = std::make_unique<Array<double>>(length);
                return *buffer;
            }

            static std::vector<double>& ensure(std::vector<double>& buffer, std::size_t length)
            {
                if (buffer.size() != length)
                    buffer.resize(length);
                return buffer;
            }

            Array<double> take_stage(std::size_t index)
            {
                Array<double> result(std::move(*stages[index]));
                stages[index].reset();
                return result;
            }
        };

        struct SummationScratchPool {
            std::array<SummationWorkspace, summation_scratch_slots> workspaces;
            std::array<bool, summation_scratch_slots> in_use{};
        };

        SummationScratchPool& summation_scratch_pool()
        {
            thread_local SummationScratchPool pool;
            return pool;
        }

        bool summation_scratch_can_retain(int ndim, const Array<double>& coefficients)
        {
            if (ndim < 1 || static_cast<std::size_t>(ndim) > summation_scratch_max_dimensions)
                return false;

            std::size_t remaining = coefficients.get_nbr();
            std::size_t retained = 0;

            for (int d = 0; d < ndim - 1; ++d) {
                const int line_length = coefficients.get_size(d);
                if (line_length <= 0 || remaining % static_cast<std::size_t>(line_length) != 0)
                    return false;
                remaining /= static_cast<std::size_t>(line_length);

                const std::size_t available = summation_scratch_max_retained_doubles - retained;
                const std::size_t line_size = static_cast<std::size_t>(line_length);
                if (line_size > available || remaining > available - line_size)
                    return false;
                retained += remaining + line_size;
            }
            return true;
        }

        bool summation_points4_scratch_can_retain(int ndim, const Array<double>& coefficients)
        {
            if (ndim < 1 || static_cast<std::size_t>(ndim) > summation_scratch_max_dimensions)
                return false;

            std::size_t remaining = coefficients.get_nbr();
            std::size_t retained = static_cast<std::size_t>(coefficients.get_size(ndim - 1));
            if (retained > summation_scratch_max_retained_doubles)
                return false;
            for (int d = 0; d < ndim - 1; ++d) {
                const int line_length = coefficients.get_size(d);
                if (line_length <= 0 || remaining % static_cast<std::size_t>(line_length) != 0)
                    return false;
                remaining /= static_cast<std::size_t>(line_length);

                const std::size_t available = summation_scratch_max_retained_doubles - retained;
                if (remaining > available / 4U)
                    return false;
                const std::size_t point_stage = 4U * remaining;
                const std::size_t line_size = static_cast<std::size_t>(line_length);
                if (point_stage > available || line_size > available - point_stage)
                    return false;
                retained += point_stage + line_size;
            }
            return true;
        }

        class SummationScratchLease
        {
          public:
            explicit SummationScratchLease(bool retain)
            {
                if (!retain)
                    return;

                SummationScratchPool& pool = summation_scratch_pool();
                for (std::size_t index = 0; index < pool.in_use.size(); ++index) {
                    if (pool.in_use[index])
                        continue;
                    pool.in_use[index] = true;
                    pool_ = &pool;
                    slot_ = index;
                    workspace_ = &pool.workspaces[index];
                    return;
                }
            }

            ~SummationScratchLease()
            {
                if (pool_ != nullptr)
                    pool_->in_use[slot_] = false;
            }

            SummationScratchLease(const SummationScratchLease&) = delete;
            SummationScratchLease& operator=(const SummationScratchLease&) = delete;

            [[nodiscard]] SummationWorkspace& workspace() { return workspace_ == nullptr ? local_ : *workspace_; }

            [[nodiscard]] bool retained() const { return workspace_ != nullptr; }

          private:
            SummationScratchPool* pool_ = nullptr;
            std::size_t slot_ = 0;
            SummationWorkspace* workspace_ = nullptr;
            SummationWorkspace local_;
        };

        double summation_1d_cheb_strided(double xx, const double* values, int length, int stride)
        {
            double tm2 = 1.;
            double tm1 = xx;
            double tm;

            double res = values[0] * tm2;
            if (length > 1)
                res += values[stride] * tm1;
            for (int i = 2; i < length; ++i) {
                tm = 2 * xx * tm1 - tm2;
                res += values[i * stride] * tm;
                tm2 = tm1;
                tm1 = tm;
            }
            return res;
        }

        double summation_1d_cheb_even_strided(double xx, const double* values, int length, int stride)
        {
            double tm2 = 1.;
            double tm1 = xx;
            double tm;

            double res = values[0] * tm2;
            for (int i = 1; i < length; ++i) {
                tm = 2 * xx * tm1 - tm2;
                res += values[i * stride] * tm;
                tm2 = tm1;
                tm1 = tm;
                tm = 2 * xx * tm1 - tm2;
                tm2 = tm1;
                tm1 = tm;
            }
            return res;
        }

        double summation_1d_cheb_odd_strided(double xx, const double* values, int length, int stride)
        {
            double tm2 = 1.;
            double tm1 = xx;
            double tm;

            double res = values[0] * tm1;
            for (int i = 1; i < length; ++i) {
                tm = 2 * xx * tm1 - tm2;
                tm2 = tm1;
                tm1 = tm;
                tm = 2 * xx * tm1 - tm2;
                res += values[i * stride] * tm;
                tm2 = tm1;
                tm1 = tm;
            }
            return res;
        }

        bool summation_1d_strided(int base, double xx, const double* values, int length, int stride,
                                  double& result)
        {
            switch (base) {
                case CHEB:
                    result = summation_1d_cheb_strided(xx, values, length, stride);
                    return true;
                case CHEB_EVEN:
                    result = summation_1d_cheb_even_strided(xx, values, length, stride);
                    return true;
                case CHEB_ODD:
                    result = summation_1d_cheb_odd_strided(xx, values, length, stride);
                    return true;
                default:
                    return false;
            }
        }

        template <bool shared_input>
        void summation_1d_cheb_points4(const std::array<double, 4>& xx,
                                       const double* values, int length,
                                       int coefficient_stride,
                                       std::array<double, 4>& result)
        {
            std::array<double, 4> tm2{1., 1., 1., 1.};
            std::array<double, 4> tm1 = xx;
            std::array<double, 4> tm{};

            for (std::size_t point = 0; point < result.size(); ++point)
                result[point] = values[shared_input ? 0 : point] * tm2[point];
            if (length > 1)
                for (std::size_t point = 0; point < result.size(); ++point)
                    result[point] +=
                        values[coefficient_stride + (shared_input ? 0 : point)] * tm1[point];
            for (int coefficient = 2; coefficient < length; ++coefficient) {
                const int offset = coefficient * coefficient_stride;
                for (std::size_t point = 0; point < result.size(); ++point) {
                    tm[point] = 2 * xx[point] * tm1[point] - tm2[point];
                    result[point] +=
                        values[offset + (shared_input ? 0 : point)] * tm[point];
                    tm2[point] = tm1[point];
                    tm1[point] = tm[point];
                }
            }
        }

        template <bool shared_input>
        void summation_1d_cheb_even_points4(const std::array<double, 4>& xx,
                                            const double* values, int length,
                                            int coefficient_stride,
                                            std::array<double, 4>& result)
        {
            std::array<double, 4> tm2{1., 1., 1., 1.};
            std::array<double, 4> tm1 = xx;
            std::array<double, 4> tm{};

            for (std::size_t point = 0; point < result.size(); ++point)
                result[point] = values[shared_input ? 0 : point] * tm2[point];
            for (int coefficient = 1; coefficient < length; ++coefficient) {
                const int offset = coefficient * coefficient_stride;
                for (std::size_t point = 0; point < result.size(); ++point) {
                    tm[point] = 2 * xx[point] * tm1[point] - tm2[point];
                    result[point] +=
                        values[offset + (shared_input ? 0 : point)] * tm[point];
                    tm2[point] = tm1[point];
                    tm1[point] = tm[point];
                    tm[point] = 2 * xx[point] * tm1[point] - tm2[point];
                    tm2[point] = tm1[point];
                    tm1[point] = tm[point];
                }
            }
        }

        template <bool shared_input>
        void summation_1d_cheb_odd_points4(const std::array<double, 4>& xx,
                                           const double* values, int length,
                                           int coefficient_stride,
                                           std::array<double, 4>& result)
        {
            std::array<double, 4> tm2{1., 1., 1., 1.};
            std::array<double, 4> tm1 = xx;
            std::array<double, 4> tm{};

            for (std::size_t point = 0; point < result.size(); ++point)
                result[point] = values[shared_input ? 0 : point] * tm1[point];
            for (int coefficient = 1; coefficient < length; ++coefficient) {
                const int offset = coefficient * coefficient_stride;
                for (std::size_t point = 0; point < result.size(); ++point) {
                    tm[point] = 2 * xx[point] * tm1[point] - tm2[point];
                    tm2[point] = tm1[point];
                    tm1[point] = tm[point];
                    tm[point] = 2 * xx[point] * tm1[point] - tm2[point];
                    result[point] +=
                        values[offset + (shared_input ? 0 : point)] * tm[point];
                    tm2[point] = tm1[point];
                    tm1[point] = tm[point];
                }
            }
        }

        template <bool shared_input>
        bool summation_1d_points4(int base, const std::array<double, 4>& xx,
                                  const double* values, int length,
                                  int coefficient_stride,
                                  std::array<double, 4>& result)
        {
            switch (base) {
                case CHEB:
                    summation_1d_cheb_points4<shared_input>(
                        xx, values, length, coefficient_stride, result);
                    return true;
                case CHEB_EVEN:
                    summation_1d_cheb_even_points4<shared_input>(
                        xx, values, length, coefficient_stride, result);
                    return true;
                case CHEB_ODD:
                    summation_1d_cheb_odd_points4<shared_input>(
                        xx, values, length, coefficient_stride, result);
                    return true;
                default:
                    return false;
            }
        }

        void collapse_summation_points4(const Base_spectral& base, int ndim,
                                        std::span<const Point* const, 4> num,
                                        const Array<double>& cf,
                                        SummationWorkspace& workspace,
                                        std::span<double, 4> result)
        {
            workspace.point_stages.resize(static_cast<std::size_t>(ndim - 1));
            workspace.lines.resize(static_cast<std::size_t>(ndim));
            Dim_array nbr_coefs(cf.get_dimensions());

            for (int d = 0; d < ndim - 1; ++d) {
                const int dim_output = ndim - 1 - d;
                Dim_array nbr_output(dim_output);
                for (int k = 0; k < dim_output; ++k)
                    nbr_output.set(k) = nbr_coefs(k + d + 1);

                std::size_t output_count = 1;
                for (int k = 0; k < dim_output; ++k)
                    output_count *= static_cast<std::size_t>(nbr_output(k));
                std::vector<double>& output = SummationWorkspace::ensure(
                    workspace.point_stages[static_cast<std::size_t>(d)], 4U * output_count);
                const bool shared_input = d == 0;
                const double* input = shared_input
                    ? cf.get_data()
                    : workspace.point_stages[static_cast<std::size_t>(d - 1)].data();
                const int coefficient_count = cf.get_size(d);
                const int coefficient_stride = static_cast<int>(
                    output_count * (shared_input ? 1U : 4U));
                std::array<double, 4> coordinates{};
                for (std::size_t point = 0; point < coordinates.size(); ++point)
                    coordinates[point] = (*num[point])(d + 1);

                Index inout(nbr_output);
                bool loop = true;
                while (loop) {
                    const int base_1d = (*base.get_base_1d(d))(inout);
                    int flat = inout(0);
                    for (int k = 1; k < dim_output; ++k)
                        flat = flat * nbr_output(k) + inout(k);

                    const int input_offset = flat * (shared_input ? 1 : 4);
                    std::array<double, 4> values{};
                    const bool handled = shared_input
                        ? summation_1d_points4<true>(base_1d, coordinates,
                              input + input_offset, coefficient_count,
                              coefficient_stride, values)
                        : summation_1d_points4<false>(base_1d, coordinates,
                              input + input_offset, coefficient_count,
                              coefficient_stride, values);
                    if (!handled) {
                        Array<double>& line = SummationWorkspace::ensure(
                            workspace.lines[static_cast<std::size_t>(d)], coefficient_count);
                        for (std::size_t point = 0; point < values.size(); ++point) {
                            for (int coefficient = 0; coefficient < coefficient_count; ++coefficient)
                                line.set(coefficient) = input[input_offset +
                                    coefficient * coefficient_stride +
                                    (shared_input ? 0 : static_cast<int>(point))];
                            values[point] = summation_1d(base_1d, coordinates[point], line);
                        }
                    }
                    for (std::size_t point = 0; point < values.size(); ++point)
                        output[4U * static_cast<std::size_t>(flat) + point] = values[point];
                    loop = inout.inc();
                }
            }

            const int last_length = cf.get_size(ndim - 1);
            const int last_base = (*base.get_base_1d(ndim - 1))(0);
            Array<double>& line = SummationWorkspace::ensure(
                workspace.lines[static_cast<std::size_t>(ndim - 1)], last_length);
            const double* partial = ndim == 1
                ? cf.get_data()
                : workspace.point_stages[static_cast<std::size_t>(ndim - 2)].data();
            for (std::size_t point = 0; point < result.size(); ++point) {
                for (int coefficient = 0; coefficient < last_length; ++coefficient)
                    line.set(coefficient) = ndim == 1
                        ? partial[coefficient]
                        : partial[4U * static_cast<std::size_t>(coefficient) + point];
                result[point] = summation_1d(last_base, (*num[point])(ndim), line);
            }
        }

        const Array<double>& collapse_summation(const Base_spectral& base, int ndim, const Point& num,
                                                const Array<double>& cf, SummationWorkspace& workspace)
        {
            workspace.stages.resize(static_cast<std::size_t>(ndim - 1));
            workspace.lines.resize(static_cast<std::size_t>(ndim - 1));
            Dim_array nbr_coefs(cf.get_dimensions());

            // Loop on dimensions (except the last):
            for (int d = 0; d < ndim - 1; d++) {
                const int dim_output = ndim - 1 - d;
                Dim_array nbr_output(dim_output);
                for (int k = 0; k < dim_output; k++)
                    nbr_output.set(k) = nbr_coefs(k + d + 1);
                Array<double>& output =
                    SummationWorkspace::ensure(workspace.stages[static_cast<std::size_t>(d)], nbr_output);

                Index inout(nbr_output);
                const int nbr = cf.get_size(d);

                const double xx = num(d + 1);
                // Dimension d is the slowest-varying axis of courant, so the 1D line
                // feeding output position `flat` starts at `flat` and advances by the flat
                // size of the output. Same contiguous walk as Base_spectral::ope_1d, in
                // place of recomputing a full multi-dimensional Index per coefficient.
                const int stride = static_cast<int>(output.get_nbr());
                const double* src =
                    d == 0 ? cf.get_data() : workspace.stages[static_cast<std::size_t>(d - 1)]->get_data();
                double* dst = output.get_data();

                // The coordinate is fixed for the whole sweep, so every output point repeats
                // the same 1D basis recurrence. Hoisting those basis values out and replaying
                // the accumulation over them is not value-neutral: separating the recurrence
                // from the accumulation changes which multiply-adds fuse, and the results then
                // differ by an ulp from the 1D kernels above. Keep the recurrence inline.

                // Loop on the points :
                bool loop = true;
                while (loop) {
                    const int base_1d = (*base.get_base_1d(d))(inout);

                    int flat = inout(0);
                    for (int k = 1; k < dim_output; k++)
                        flat = flat * nbr_output(k) + inout(k);

                    if (!summation_1d_strided(base_1d, xx, src + flat, nbr, stride, dst[flat])) {
                        Array<double>& tab_1d = SummationWorkspace::ensure(
                            workspace.lines[static_cast<std::size_t>(d)], nbr);
                        int read = flat;
                        for (int k = 0; k < nbr; k++) {
                            tab_1d.set(k) = src[read];
                            read += stride;
                        }
                        dst[flat] = summation_1d(base_1d, xx, tab_1d);
                    }
                    loop = inout.inc();
                }
            }

            return ndim == 1 ? cf : *workspace.stages[static_cast<std::size_t>(ndim - 2)];
        }
    } // namespace

    double summation_1d_pasprevu(double, const Array<double>&)
    {
        KADATH_THROW("Summation_1d not implemented.");
    }

    double summation_1d_cheb(double xx, const Array<double>& tab)
    {
        assert(tab.get_ndim() == 1);
        int nr = tab.get_size(0);
        double tm2 = 1.;
        double tm1 = xx;
        double tm;

        double res = tab(0) * tm2;
        if (nr > 1)
            res += tab(1) * tm1;
        for (int i = 2; i < nr; i++) {
            tm = 2 * xx * tm1 - tm2;
            res += tab(i) * tm;
            tm2 = tm1;
            tm1 = tm;
        }
        return res;
    }

    double summation_1d_cheb_even(double xx, const Array<double>& tab)
    {
        assert(tab.get_ndim() == 1);
        int nr = tab.get_size(0);
        double tm2 = 1.;
        double tm1 = xx;
        double tm;

        double res = tab(0) * tm2;
        for (int i = 1; i < nr; i++) {
            tm = 2 * xx * tm1 - tm2;
            res += tab(i) * tm;
            tm2 = tm1;
            tm1 = tm;
            tm = 2 * xx * tm1 - tm2;
            tm2 = tm1;
            tm1 = tm;
        }
        return res;
    }

    double summation_1d_cheb_odd(double xx, const Array<double>& tab)
    {
        assert(tab.get_ndim() == 1);
        int nr = tab.get_size(0);
        double tm2 = 1.;
        double tm1 = xx;
        double tm;

        double res = tab(0) * tm1;
        for (int i = 1; i < nr; i++) {
            tm = 2 * xx * tm1 - tm2;
            tm2 = tm1;
            tm1 = tm;
            tm = 2 * xx * tm1 - tm2;
            res += tab(i) * tm;
            tm2 = tm1;
            tm1 = tm;
        }
        return res;
    }

    double summation_1d_leg(double xx, const Array<double>& tab)
    {
        assert(tab.get_ndim() == 1);
        int nr = tab.get_size(0);
        double plm2 = 1.;
        double plm1 = xx;
        double pl;

        double res = tab(0) * plm2;
        if (nr > 1)
            res += tab(1) * plm1;
        for (int l = 2; l < nr; l++) {
            pl = ((2 * l - 1) * xx * plm1 - (l - 1) * plm2) / l;
            res += tab(l) * pl;
            plm2 = plm1;
            plm1 = pl;
        }
        return res;
    }

    double summation_1d_leg_even(double xx, const Array<double>& tab)
    {

        assert(tab.get_ndim() == 1);
        int nr = tab.get_size(0);
        double plm2 = 1.;
        double plm1 = xx;
        double pl;

        double res = tab(0) * plm2;
        int courant;
        for (int l = 1; l < nr; l++) {
            courant = 2 * l;
            pl = ((2 * courant - 1) * xx * plm1 - (courant - 1) * plm2) / courant;
            res += tab(l) * pl;
            plm2 = plm1;
            plm1 = pl;
            courant++;
            pl = ((2 * courant - 1) * xx * plm1 - (courant - 1) * plm2) / courant;
            plm2 = plm1;
            plm1 = pl;
        }

        return res;
    }

    double summation_1d_leg_odd(double xx, const Array<double>& tab)
    {
        assert(tab.get_ndim() == 1);
        int nr = tab.get_size(0);
        double plm2 = 1.;
        double plm1 = xx;
        double pl;

        double res = tab(0) * plm1;
        int courant;
        for (int l = 1; l < nr; l++) {
            courant = 2 * l;
            pl = ((2 * courant - 1) * xx * plm1 - (courant - 1) * plm2) / courant;
            plm2 = plm1;
            plm1 = pl;
            courant++;
            pl = ((2 * courant - 1) * xx * plm1 - (courant - 1) * plm2) / courant;
            res += tab(l) * pl;
            plm2 = plm1;
            plm1 = pl;
        }
        return res;
    }

    double summation_1d_cossin(double xx, const Array<double>& tab)
    {
        assert(tab.get_ndim() == 1);
        int nbr = tab.get_size(0);
        double res = 0;
        for (int i = 0; i < nbr; i++)
            res += (i % 2 == 0) ? tab(i) * cos((i / 2) * xx) : tab(i) * sin((i - 1) / 2 * xx);
        return res;
    }

    double summation_1d_cossin_even(double xx, const Array<double>& tab)
    {
        assert(tab.get_ndim() == 1);
        int nbr = tab.get_size(0);
        double res = 0;
        for (int i = 0; i < nbr; i++)
            res += (i % 2 == 0) ? tab(i) * cos(i * xx) : tab(i) * sin((i - 1) * xx);
        return res;
    }

    double summation_1d_cossin_odd(double xx, const Array<double>& tab)
    {
        assert(tab.get_ndim() == 1);
        int nbr = tab.get_size(0);
        double res = 0;
        for (int i = 0; i < nbr; i++)
            res += (i % 2 == 0) ? tab(i) * cos((i + 1) * xx) : tab(i) * sin(i * xx);
        return res;
    }

    double summation_1d_cos(double xx, const Array<double>& tab)
    {
        assert(tab.get_ndim() == 1);
        int nbr = tab.get_size(0);
        double res = 0;
        for (int i = 0; i < nbr; i++)
            res += tab(i) * cos(i * xx);
        return res;
    }

    double summation_1d_sin(double xx, const Array<double>& tab)
    {
        assert(tab.get_ndim() == 1);
        int nbr = tab.get_size(0);
        double res = 0;
        for (int i = 1; i < nbr; i++)
            res += tab(i) * sin(i * xx);
        return res;
    }

    double summation_1d_cos_even(double xx, const Array<double>& tab)
    {
        assert(tab.get_ndim() == 1);
        int nbr = tab.get_size(0);
        double res = 0;
        for (int i = 0; i < nbr; i++)
            res += tab(i) * cos(2 * i * xx);
        return res;
    }

    double summation_1d_cos_odd(double xx, const Array<double>& tab)
    {
        assert(tab.get_ndim() == 1);
        int nbr = tab.get_size(0);
        double res = 0;
        for (int i = 0; i < nbr; i++)
            res += tab(i) * cos((2 * i + 1) * xx);
        return res;
    }

    double summation_1d_sin_even(double xx, const Array<double>& tab)
    {
        assert(tab.get_ndim() == 1);
        int nbr = tab.get_size(0);
        double res = 0;
        for (int i = 1; i < nbr; i++)
            res += tab(i) * sin(2 * i * xx);
        return res;
    }

    double summation_1d_sin_odd(double xx, const Array<double>& tab)
    {
        assert(tab.get_ndim() == 1);
        int nbr = tab.get_size(0);
        double res = 0;
        for (int i = 0; i < nbr; i++)
            res += tab(i) * sin((2 * i + 1) * xx);
        return res;
    }

    double summation_1d(int base, double xx, const Array<double>& tab)
    {
        using Kernel = double (*)(double, const Array<double>&);
        static const std::array<Kernel, NBR_MAX_BASE> kernels = [] {
            std::array<Kernel, NBR_MAX_BASE> result;
            result.fill(summation_1d_pasprevu);
            result[CHEB] = summation_1d_cheb;
            result[CHEB_EVEN] = summation_1d_cheb_even;
            result[CHEB_ODD] = summation_1d_cheb_odd;
            result[COSSIN] = summation_1d_cossin;
            result[COS_EVEN] = summation_1d_cos_even;
            result[COS_ODD] = summation_1d_cos_odd;
            result[SIN_ODD] = summation_1d_sin_odd;
            result[SIN_EVEN] = summation_1d_sin_even;
            result[COS] = summation_1d_cos;
            result[SIN] = summation_1d_sin;
            result[LEG] = summation_1d_leg;
            result[LEG_EVEN] = summation_1d_leg_even;
            result[LEG_ODD] = summation_1d_leg_odd;
            result[COSSIN_EVEN] = summation_1d_cossin_even;
            result[COSSIN_ODD] = summation_1d_cossin_odd;
            return result;
        }();

        return kernels[static_cast<std::size_t>(base)](xx, tab);
    }

    Array<double> Base_spectral::summation_but_last(const Point& num, const Array<double>& cf) const
    {
        SummationScratchLease lease(summation_scratch_can_retain(ndim, cf));
        SummationWorkspace& workspace = lease.workspace();
        const Array<double>& partial = collapse_summation(*this, ndim, num, cf, workspace);
        assert(partial.get_ndim() == 1);

        // Retained storage remains worker-owned. The small final 1-D result is
        // detached so callers keep ordinary Array ownership beyond this call.
        if (lease.retained())
            return Array<double>(partial);
        if (ndim == 1)
            return Array<double>(partial);
        return workspace.take_stage(static_cast<std::size_t>(ndim - 2));
    }

    double Base_spectral::summation_last_dim(double x, const Array<double>& partial) const
    {
        assert(partial.get_ndim() == 1);
        assert(bases_1d[ndim - 1]->get_ndim() == 1);
        assert(bases_1d[ndim - 1]->get_size(0) == 1);
        return summation_1d((*bases_1d[ndim - 1])(0), x, partial);
    }

    double Base_spectral::summation(const Point& num, const Array<double>& cf) const
    {
        // Collapse all but the last dimension, then evaluate the last one. The
        // split is value-identical to the fused loop; callers sweeping the last
        // coordinate (e.g. bispherical volume quadrature) can reuse the partial.
        SummationScratchLease lease(summation_scratch_can_retain(ndim, cf));
        const Array<double>& partial = collapse_summation(*this, ndim, num, cf, lease.workspace());
        return summation_last_dim(num(ndim), partial);
    }

    void Base_spectral::summation_points4(std::span<const Point* const, 4> num,
                                          const Array<double>& cf,
                                          std::span<double, 4> result) const
    {
        for (const Point* point : num)
            if (point == nullptr)
                KADATH_THROW("summation_points4 requires non-null points");
        SummationScratchLease lease(summation_points4_scratch_can_retain(ndim, cf));
        collapse_summation_points4(*this, ndim, num, cf, lease.workspace(), result);
    }
} // namespace Kadath
