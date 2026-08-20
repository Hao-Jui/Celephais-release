/*
    Copyright 2026 Hao-Jui Kuan

    This file is part of Celephais.

    Celephais is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    Celephais is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with Celephais.  If not, see <http://www.gnu.org/licenses/>.
*/

#include "For_Kadath/Base_spectral/base_r2hc.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <new>
#include <numbers>
#include <stdexcept>

#ifdef CELEPHAIS_ENABLE_FFTW_ORACLE
#include <fftw3.h>
#endif

#if defined(__aarch64__) && (defined(__clang__) || defined(__GNUC__))
#include <arm_neon.h>
#endif

namespace Kadath
{
    namespace
    {
        // The generated schedules and runtime wrappers below use fused
        // operations deliberately: each helper is a single rounded operation.
        inline double multiply_add(double const left, double const right, double const addend)
        {
            return std::fma(left, right, addend);
        }

        inline double multiply_subtract(double const left, double const right, double const subtrahend)
        {
            return std::fma(left, right, -subtrahend);
        }

        inline double negative_multiply_subtract(double const left, double const right, double const addend)
        {
            return std::fma(-left, right, addend);
        }

        template <int N> void execute_r2hc_codelet(double* data);

        template <int N> void execute_hc2r_codelet(double* data);

        template <> void execute_r2hc_codelet<2>(double* const data)
        {
            const double first = data[0];
            const double second = data[1];
            data[0] = first + second;
            data[1] = first - second;
        }

        template <> void execute_hc2r_codelet<2>(double* const data)
        {
            execute_r2hc_codelet<2>(data);
        }

        template <> void execute_r2hc_codelet<4>(double* const data)
        {
            const double point_zero = data[0];
            const double point_one = data[1];
            const double point_two = data[2];
            const double point_three = data[3];
            const double even_sum = point_zero + point_two;
            const double even_difference = point_zero - point_two;
            const double odd_sum = point_one + point_three;
            const double odd_difference = point_three - point_one;
            data[0] = even_sum + odd_sum;
            data[1] = even_difference;
            data[2] = even_sum - odd_sum;
            data[3] = odd_difference;
        }

        template <> void execute_hc2r_codelet<4>(double* const data)
        {
            const double dc = data[0];
            const double mode_one = data[1];
            const double nyquist = data[2];
            const double imaginary_one = data[3];
            const double even_sum = dc + nyquist;
            const double odd_sum = dc - nyquist;
            data[0] = even_sum + 2. * mode_one;
            data[1] = odd_sum - 2. * imaginary_one;
            data[2] = even_sum - 2. * mode_one;
            data[3] = odd_sum + 2. * imaginary_one;
        }

#include "native_r2hc_schedules.inl"

        template <> void execute_r2hc_codelet<18>(double* const data)
        {
            constexpr std::array<double, 4> cosine{0.93969262078590838405410927732473146994,
                                                   0.76604444311897803520239265055541667394, 0.5,
                                                   0.17364817766693034885171662676931479600};
            constexpr std::array<double, 4> sine{
                0.34202014332566873304409961468225958076, 0.64278760968653932632264340990726343291,
                0.86602540378443864676372317075293618347, 0.98480775301220805936674302458952301367};
            std::array<double, 9> even{};
            std::array<double, 9> odd{};
            for (int index = 0; index < 9; ++index) {
                even[static_cast<std::size_t>(index)] = data[2 * index];
                odd[static_cast<std::size_t>(index)] = data[2 * index + 1];
            }
            execute_r2hc_codelet_9(even.data());
            execute_r2hc_codelet_9(odd.data());

            data[0] = even[0] + odd[0];
            data[9] = even[0] - odd[0];
            for (int k = 1; k <= 4; ++k) {
                const double even_real = even[static_cast<std::size_t>(k)];
                const double even_imaginary = even[static_cast<std::size_t>(9 - k)];
                const double odd_real = odd[static_cast<std::size_t>(k)];
                const double odd_imaginary = odd[static_cast<std::size_t>(9 - k)];
                const double rotated_real = multiply_add(cosine[static_cast<std::size_t>(k - 1)], odd_real,
                                                         sine[static_cast<std::size_t>(k - 1)] * odd_imaginary);
                const double rotated_imaginary =
                    negative_multiply_subtract(sine[static_cast<std::size_t>(k - 1)], odd_real,
                                               cosine[static_cast<std::size_t>(k - 1)] * odd_imaginary);
                data[k] = even_real + rotated_real;
                data[18 - k] = even_imaginary + rotated_imaginary;
                data[9 - k] = even_real - rotated_real;
                data[9 + k] = rotated_imaginary - even_imaginary;
            }
        }

        template <> void execute_hc2r_codelet<18>(double* const data)
        {
            constexpr std::array<double, 4> cosine{0.93969262078590838405410927732473146994,
                                                   0.76604444311897803520239265055541667394, 0.5,
                                                   0.17364817766693034885171662676931479600};
            constexpr std::array<double, 4> sine{
                0.34202014332566873304409961468225958076, 0.64278760968653932632264340990726343291,
                0.86602540378443864676372317075293618347, 0.98480775301220805936674302458952301367};
            std::array<double, 9> even{};
            std::array<double, 9> odd{};
            even[0] = data[0] + data[9];
            odd[0] = data[0] - data[9];
            for (int k = 1; k <= 4; ++k) {
                const double mode_real = data[k];
                const double mode_imaginary = data[18 - k];
                const double reflected_real = data[9 - k];
                const double reflected_imaginary = data[9 + k];
                const double difference_real = mode_real - reflected_real;
                const double difference_imaginary = mode_imaginary + reflected_imaginary;
                even[static_cast<std::size_t>(k)] = mode_real + reflected_real;
                even[static_cast<std::size_t>(9 - k)] = mode_imaginary - reflected_imaginary;
                odd[static_cast<std::size_t>(k)] =
                    negative_multiply_subtract(sine[static_cast<std::size_t>(k - 1)], difference_imaginary,
                                               cosine[static_cast<std::size_t>(k - 1)] * difference_real);
                odd[static_cast<std::size_t>(9 - k)] =
                    multiply_add(sine[static_cast<std::size_t>(k - 1)], difference_real,
                                 cosine[static_cast<std::size_t>(k - 1)] * difference_imaginary);
            }
            execute_hc2r_codelet_9(even.data());
            execute_hc2r_codelet_9(odd.data());
            for (int index = 0; index < 9; ++index) {
                data[2 * index] = even[static_cast<std::size_t>(index)];
                data[2 * index + 1] = odd[static_cast<std::size_t>(index)];
            }
        }

        template <int HalfSize> struct real_radix2_twiddles;

        template <> struct real_radix2_twiddles<11>
        {
            inline static constexpr std::array<double, 5> cosine{
                0.95949297361449738989036805706632769906,
                0.84125353283118116886181164891936771751,
                0.65486073394528506405692507246629355318,
                0.41541501300188642552927414922962320352,
                0.14231483827328514044379266861636966879};
            inline static constexpr std::array<double, 5> sine{
                0.28173255684142969771141791534661689904,
                0.54064081745559758210763595431869169543,
                0.75574957435425828377403584397234442018,
                0.90963199535451837141171538307902846006,
                0.98982144188093273237609203777671878738};
        };

        template <> struct real_radix2_twiddles<12>
        {
            inline static constexpr std::array<double, 5> cosine{
                0.96592582628906828674974319972889736763,
                0.86602540378443864676372317075293618347,
                0.70710678118654752440084436210484903928,
                0.5,
                0.25881904510252076234889883762404832835};
            inline static constexpr std::array<double, 5> sine{
                0.25881904510252076234889883762404832835,
                0.5,
                0.70710678118654752440084436210484903928,
                0.86602540378443864676372317075293618347,
                0.96592582628906828674974319972889736763};
        };

        template <> struct real_radix2_twiddles<13>
        {
            inline static constexpr std::array<double, 6> cosine{
                0.97094181742605202715698227629378922725,
                0.88545602565320989590037552201509887851,
                0.74851074817110109863463059970135138385,
                0.56806474673115580251180755912751662489,
                0.35460488704253562596963789260001847484,
                0.12053668025532305334906768745254358227};
            inline static constexpr std::array<double, 6> sine{
                0.23931566428755776714875372626021189520,
                0.46472317204376854565601533513310477756,
                0.66312265824079520237678549266676627952,
                0.82298386589365639457961742343938199065,
                0.93501624268541482343978459983783072905,
                0.99270887409805399280075164949252017934};
        };

        template <> struct real_radix2_twiddles<14>
        {
            inline static constexpr std::array<double, 6> cosine{
                0.97492791218182360701813168299393121723,
                0.90096886790241912623610231950744505117,
                0.78183148246802980870844452667405775023,
                0.62348980185873353052500488400423981063,
                0.43388373911755812047576833284835875461,
                0.22252093395631440428890256449679475947};
            inline static constexpr std::array<double, 6> sine{
                0.22252093395631440428890256449679475947,
                0.43388373911755812047576833284835875461,
                0.62348980185873353052500488400423981063,
                0.78183148246802980870844452667405775023,
                0.90096886790241912623610231950744505117,
                0.97492791218182360701813168299393121723};
        };

        template <> struct real_radix2_twiddles<15>
        {
            inline static constexpr std::array<double, 7> cosine{
                0.97814760073380563792856674786959953246,
                0.91354545764260089550212757198531717794,
                0.80901699437494742410229341718281905886,
                0.66913060635885821382627333068678047360,
                0.5,
                0.30901699437494742410229341718281905886,
                0.10452846326765347139983415480249811908};
            inline static constexpr std::array<double, 7> sine{
                0.20791169081775933710174228440512516622,
                0.40673664307580020775398599034149798730,
                0.58778525229247312916870595463907276860,
                0.74314482547739423501469704897425697719,
                0.86602540378443864676372317075293618347,
                0.95105651629515357211643933337938214341,
                0.99452189536827333692269194498057038152};
        };

        template <int HalfSize> inline void execute_r2hc_radix2_codelet(double* const data)
        {
            constexpr int transform_size = 2 * HalfSize;
            constexpr int paired_modes = (HalfSize - 1) / 2;
            std::array<double, HalfSize> even{};
            std::array<double, HalfSize> odd{};
            for (int index = 0; index < HalfSize; ++index) {
                even[static_cast<std::size_t>(index)] = data[2 * index];
                odd[static_cast<std::size_t>(index)] = data[2 * index + 1];
            }
            execute_r2hc_codelet<HalfSize>(even.data());
            execute_r2hc_codelet<HalfSize>(odd.data());

            const double even_dc = even[0];
            const double odd_dc = odd[0];
            data[0] = even_dc + odd_dc;
            data[HalfSize] = even_dc - odd_dc;
            for (int k = 1; k <= paired_modes; ++k) {
                const double even_real = even[static_cast<std::size_t>(k)];
                const double even_imaginary = even[static_cast<std::size_t>(HalfSize - k)];
                const double odd_real = odd[static_cast<std::size_t>(k)];
                const double odd_imaginary = odd[static_cast<std::size_t>(HalfSize - k)];
                const double cosine = real_radix2_twiddles<HalfSize>::cosine[
                    static_cast<std::size_t>(k - 1)];
                const double sine = real_radix2_twiddles<HalfSize>::sine[
                    static_cast<std::size_t>(k - 1)];
                const double rotated_real = multiply_add(cosine, odd_real,
                                                         sine * odd_imaginary);
                const double rotated_imaginary =
                    negative_multiply_subtract(sine, odd_real,
                                               cosine * odd_imaginary);
                data[k] = even_real + rotated_real;
                data[transform_size - k] = even_imaginary + rotated_imaginary;
                data[HalfSize - k] = even_real - rotated_real;
                data[HalfSize + k] = rotated_imaginary - even_imaginary;
            }
            if constexpr (HalfSize % 2 == 0) {
                const double even_nyquist = even[HalfSize / 2];
                const double odd_nyquist = odd[HalfSize / 2];
                data[HalfSize / 2] = even_nyquist;
                data[transform_size - HalfSize / 2] = -odd_nyquist;
            }
        }

        template <int HalfSize> inline void execute_hc2r_radix2_codelet(double* const data)
        {
            constexpr int transform_size = 2 * HalfSize;
            constexpr int paired_modes = (HalfSize - 1) / 2;
            std::array<double, HalfSize> even{};
            std::array<double, HalfSize> odd{};
            const double dc = data[0];
            const double nyquist = data[HalfSize];
            even[0] = dc + nyquist;
            odd[0] = dc - nyquist;
            for (int k = 1; k <= paired_modes; ++k) {
                const double mode_real = data[k];
                const double mode_imaginary = data[transform_size - k];
                const double reflected_real = data[HalfSize - k];
                const double reflected_imaginary = data[HalfSize + k];
                const double difference_real = mode_real - reflected_real;
                const double difference_imaginary = mode_imaginary + reflected_imaginary;
                const double cosine = real_radix2_twiddles<HalfSize>::cosine[
                    static_cast<std::size_t>(k - 1)];
                const double sine = real_radix2_twiddles<HalfSize>::sine[
                    static_cast<std::size_t>(k - 1)];
                even[static_cast<std::size_t>(k)] = mode_real + reflected_real;
                even[static_cast<std::size_t>(HalfSize - k)] =
                    mode_imaginary - reflected_imaginary;
                odd[static_cast<std::size_t>(k)] = negative_multiply_subtract(
                    sine, difference_imaginary, cosine * difference_real);
                odd[static_cast<std::size_t>(HalfSize - k)] =
                    multiply_add(sine, difference_real, cosine * difference_imaginary);
            }
            if constexpr (HalfSize % 2 == 0) {
                even[HalfSize / 2] = data[HalfSize / 2];
                odd[HalfSize / 2] = data[HalfSize + HalfSize / 2];
                even[HalfSize / 2] *= 2.;
                odd[HalfSize / 2] *= -2.;
            }

            execute_hc2r_codelet<HalfSize>(even.data());
            execute_hc2r_codelet<HalfSize>(odd.data());
            for (int index = 0; index < HalfSize; ++index) {
                data[2 * index] = even[static_cast<std::size_t>(index)];
                data[2 * index + 1] = odd[static_cast<std::size_t>(index)];
            }
        }

#define CELEPHAIS_REAL_RADIX2_CODELET(N, HALF)                                                                         \
    template <> void execute_r2hc_codelet<N>(double* const data)                                                       \
    {                                                                                                                  \
        execute_r2hc_radix2_codelet<HALF>(data);                                                                       \
    }                                                                                                                  \
    template <> void execute_hc2r_codelet<N>(double* const data)                                                       \
    {                                                                                                                  \
        execute_hc2r_radix2_codelet<HALF>(data);                                                                       \
    }

        CELEPHAIS_REAL_RADIX2_CODELET(22, 11)
        CELEPHAIS_REAL_RADIX2_CODELET(24, 12)
        CELEPHAIS_REAL_RADIX2_CODELET(26, 13)
        CELEPHAIS_REAL_RADIX2_CODELET(28, 14)
        CELEPHAIS_REAL_RADIX2_CODELET(30, 15)

#undef CELEPHAIS_REAL_RADIX2_CODELET

        template <std::size_t N> std::array<double, N> load_strided_line(const double* const src, int const stride)
        {
            std::array<double, N> values{};
            for (std::size_t i = 0; i < N; ++i)
                values[i] = src[i * static_cast<std::size_t>(stride)];
            return values;
        }

        template <std::size_t N>
        void store_strided_line(const std::array<double, N>& values, double* const dst, int const stride)
        {
            for (std::size_t i = 0; i < N; ++i)
                dst[i * static_cast<std::size_t>(stride)] = values[i];
        }

        template <int TransformSize, native_spectral_family Family>
        bool execute_fused_cheb_forward(const double* const src, double* const dst, int const nbr_in, int const nbr_out,
                                        int const stride, const r2hc_precomp_t& transform)
        {
            static_assert(Family == native_spectral_family::cheb || Family == native_spectral_family::cheb_even ||
                          Family == native_spectral_family::cheb_odd);
            constexpr int nr = TransformSize + 1;
            if (nbr_in != nr || nbr_out != nr)
                return false;

            const std::array<double, nr> input = load_strided_line<nr>(src, stride);
            std::array<double, nr - 1> transformed{};
            const double* const sin_pi = transform.sin_pi_over_n.data();

            if constexpr (Family == native_spectral_family::cheb_odd) {
                std::array<double, nr> coefficients{};
                const double* const sin_half = transform.sin_half_pi_i_over_n.data();
                for (int i = 0; i < nr; ++i)
                    coefficients[static_cast<std::size_t>(i)] = input[static_cast<std::size_t>(i)] * sin_half[i];

                const double fmoins0 = -0.5 * (coefficients[0] - coefficients[nr - 1]);
                for (int i = 1; i < (nr - 1) / 2; ++i) {
                    const double fp = 0.5 * (coefficients[static_cast<std::size_t>(i)] +
                                             coefficients[static_cast<std::size_t>(nr - 1 - i)]);
                    const double fms = 0.5 *
                                       (-coefficients[static_cast<std::size_t>(i)] +
                                        coefficients[static_cast<std::size_t>(nr - 1 - i)]) *
                                       sin_pi[i];
                    transformed[static_cast<std::size_t>(i)] = fp + fms;
                    transformed[static_cast<std::size_t>(nr - 1 - i)] = fp - fms;
                }
                transformed[0] = 0.5 * (coefficients[0] + coefficients[nr - 1]);
                transformed[(nr - 1) / 2] = coefficients[(nr - 1) / 2];
                execute_r2hc_codelet<TransformSize>(transformed.data());

                coefficients[0] = transformed[0] / (nr - 1);
                for (int i = 2; i < nr - 1; i += 2)
                    coefficients[static_cast<std::size_t>(i)] =
                        2 * transformed[static_cast<std::size_t>(i / 2)] / (nr - 1);
                coefficients[nr - 1] = transformed[(nr - 1) / 2] / (nr - 1);
                coefficients[1] = 0.;
                double sum = 0.;
                for (int i = 3; i < nr; i += 2) {
                    coefficients[static_cast<std::size_t>(i)] =
                        coefficients[static_cast<std::size_t>(i - 2)] +
                        4 * transformed[static_cast<std::size_t>(nr - 1 - i / 2)] / (nr - 1);
                    sum += coefficients[static_cast<std::size_t>(i)];
                }
                const double c1 = (fmoins0 - sum) / ((nr - 1) / 2);
                coefficients[1] = c1;
                for (int i = 3; i < nr; i += 2)
                    coefficients[static_cast<std::size_t>(i)] += c1;

                coefficients[0] = 2 * coefficients[0];
                for (int i = 1; i < nr - 1; ++i)
                    coefficients[static_cast<std::size_t>(i)] =
                        2 * coefficients[static_cast<std::size_t>(i)]
                        - coefficients[static_cast<std::size_t>(i - 1)];
                coefficients[nr - 1] = 0.;
                store_strided_line(coefficients, dst, stride);
                return true;
            } else {
                constexpr bool even = Family == native_spectral_family::cheb_even;
                const double fmoins0 =
                    (even ? -0.5 : 0.5) * (input[0] - input[nr - 1]);
                for (int i = 1; i < (nr - 1) / 2; ++i) {
                    const double fp = 0.5 * (input[static_cast<std::size_t>(i)]
                                             + input[static_cast<std::size_t>(nr - 1 - i)]);
                    const double fms =
                        even
                            ? 0.5 * (-input[static_cast<std::size_t>(i)]
                                     + input[static_cast<std::size_t>(nr - 1 - i)])
                                  * sin_pi[i]
                            : 0.5 * (input[static_cast<std::size_t>(i)]
                                     - input[static_cast<std::size_t>(nr - 1 - i)])
                                  * sin_pi[i];
                    transformed[static_cast<std::size_t>(i)] = fp + fms;
                    transformed[static_cast<std::size_t>(nr - 1 - i)] = fp - fms;
                }
                transformed[0] = 0.5 * (input[0] + input[nr - 1]);
                transformed[(nr - 1) / 2] = input[(nr - 1) / 2];
                execute_r2hc_codelet<TransformSize>(transformed.data());

                std::array<double, nr> coefficients{};
                coefficients[0] = transformed[0] / (nr - 1);
#if defined(__aarch64__) && (defined(__clang__) || defined(__GNUC__))
                if constexpr (TransformSize == 10
                              && Family == native_spectral_family::cheb) {
                    coefficients[2] = 2 * transformed[1] / (nr - 1);

                    const double numerator_four = 2 * transformed[2];
                    const double numerator_six = 2 * transformed[3];
                    const double numerator_eight = 2 * transformed[4];
                    const double numerator_ten = transformed[5];
                    const float64x2_t divisor =
                        vdupq_n_f64(static_cast<double>(nr - 1));

                    float64x2_t numerators = vdupq_n_f64(numerator_four);
                    numerators = vsetq_lane_f64(numerator_six, numerators, 1);
                    float64x2_t quotients = vdivq_f64(numerators, divisor);
                    coefficients[4] = vgetq_lane_f64(quotients, 0);
                    coefficients[6] = vgetq_lane_f64(quotients, 1);

                    numerators = vdupq_n_f64(numerator_eight);
                    numerators = vsetq_lane_f64(numerator_ten, numerators, 1);
                    quotients = vdivq_f64(numerators, divisor);
                    coefficients[8] = vgetq_lane_f64(quotients, 0);
                    coefficients[10] = vgetq_lane_f64(quotients, 1);
                } else if constexpr (TransformSize == 12
                                     && Family == native_spectral_family::cheb) {
                    coefficients[2] = 2 * transformed[1] / (nr - 1);

                    const double numerator_four = 2 * transformed[2];
                    const double numerator_six = 2 * transformed[3];
                    const double numerator_eight = 2 * transformed[4];
                    const double numerator_ten = 2 * transformed[5];
                    const float64x2_t divisor =
                        vdupq_n_f64(static_cast<double>(nr - 1));

                    float64x2_t numerators = vdupq_n_f64(numerator_four);
                    numerators = vsetq_lane_f64(numerator_six, numerators, 1);
                    float64x2_t quotients = vdivq_f64(numerators, divisor);
                    coefficients[4] = vgetq_lane_f64(quotients, 0);
                    coefficients[6] = vgetq_lane_f64(quotients, 1);

                    numerators = vdupq_n_f64(numerator_eight);
                    numerators = vsetq_lane_f64(numerator_ten, numerators, 1);
                    quotients = vdivq_f64(numerators, divisor);
                    coefficients[8] = vgetq_lane_f64(quotients, 0);
                    coefficients[10] = vgetq_lane_f64(quotients, 1);
                    coefficients[12] = transformed[6] / (nr - 1);
                } else if constexpr (TransformSize == 14
                                     && Family == native_spectral_family::cheb) {
                    coefficients[2] = 2 * transformed[1] / (nr - 1);

                    const float64x2_t divisor =
                        vdupq_n_f64(static_cast<double>(nr - 1));

                    float64x2_t numerators = vdupq_n_f64(2 * transformed[2]);
                    numerators = vsetq_lane_f64(2 * transformed[3], numerators, 1);
                    float64x2_t quotients = vdivq_f64(numerators, divisor);
                    coefficients[4] = vgetq_lane_f64(quotients, 0);
                    coefficients[6] = vgetq_lane_f64(quotients, 1);

                    numerators = vdupq_n_f64(2 * transformed[4]);
                    numerators = vsetq_lane_f64(2 * transformed[5], numerators, 1);
                    quotients = vdivq_f64(numerators, divisor);
                    coefficients[8] = vgetq_lane_f64(quotients, 0);
                    coefficients[10] = vgetq_lane_f64(quotients, 1);

                    numerators = vdupq_n_f64(2 * transformed[6]);
                    numerators = vsetq_lane_f64(transformed[7], numerators, 1);
                    quotients = vdivq_f64(numerators, divisor);
                    coefficients[12] = vgetq_lane_f64(quotients, 0);
                    coefficients[14] = vgetq_lane_f64(quotients, 1);
                } else
#endif
                {
                    for (int i = 2; i < nr - 1; i += 2)
                        coefficients[static_cast<std::size_t>(i)] =
                            2 * transformed[static_cast<std::size_t>(i / 2)] / (nr - 1);
                    coefficients[nr - 1] = transformed[(nr - 1) / 2] / (nr - 1);
                }
                coefficients[1] = 0.;
                double sum = 0.;
                for (int i = 3; i < nr; i += 2) {
                    if constexpr (even)
                        coefficients[static_cast<std::size_t>(i)] =
                            coefficients[static_cast<std::size_t>(i - 2)]
                            + 4 * transformed[static_cast<std::size_t>(nr - 1 - i / 2)]
                                  / (nr - 1);
                    else
                        coefficients[static_cast<std::size_t>(i)] =
                            coefficients[static_cast<std::size_t>(i - 2)]
                            - 4 * transformed[static_cast<std::size_t>(nr - 1 - (i - 1) / 2)]
                                  / (nr - 1);
                    sum += coefficients[static_cast<std::size_t>(i)];
                }
                const double c1 = even ? (fmoins0 - sum) / ((nr - 1) / 2)
                                       : -(fmoins0 + sum) / ((nr - 1) / 2);
                coefficients[1] = c1;
                for (int i = 3; i < nr; i += 2)
                    coefficients[static_cast<std::size_t>(i)] += c1;
                store_strided_line(coefficients, dst, stride);
                return true;
            }
        }

        template <int TransformSize>
        bool execute_fused_cos_forward(const double* const src, double* const dst,
                                       int const nbr_in, int const nbr_out,
                                       int const stride, const r2hc_precomp_t& transform)
        {
            constexpr int nbr = TransformSize + 1;
            if (nbr_in != nbr || nbr_out != nbr)
                return false;

            const std::array<double, nbr> input = load_strided_line<nbr>(src, stride);
            std::array<double, TransformSize> transformed{};
            std::array<double, nbr> coefficients{};
            const double* const sin_pi = transform.sin_pi_over_n.data();

            double fmoins0 = 0.5 * (input[0] - input[nbr - 1]);

            for (int i = 1; i < (nbr - 1) / 2; i++) {
                double fp = 0.5 * (input[i] + input[nbr - 1 - i]);
                double fms = 0.5 * (input[i] - input[nbr - 1 - i]) * sin_pi[i];
                transformed[i] = fp + fms;
                transformed[nbr - 1 - i] = fp - fms;
            }

            transformed[0] = 0.5 * (input[0] + input[nbr - 1]);
            transformed[(nbr - 1) / 2] = input[(nbr - 1) / 2];

            execute_r2hc_codelet<TransformSize>(transformed.data());

            coefficients[0] = transformed[0] / (nbr - 1);
            for (int i = 2; i < nbr - 1; i += 2)
                coefficients[i] = 2 * transformed[i / 2] / (nbr - 1);
            coefficients[nbr - 1] = transformed[(nbr - 1) / 2] / (nbr - 1);

            coefficients[1] = 0;
            double som = 0;
            for (int i = 3; i < nbr; i += 2) {
                coefficients[i] = coefficients[i - 2]
                                  + 4 * transformed[nbr - 1 - i / 2] / (nbr - 1);
                som += coefficients[i];
            }

            double c1 = (fmoins0 - som) / ((nbr - 1) / 2);

            coefficients[1] = c1;
            for (int i = 3; i < nbr; i += 2)
                coefficients[i] += c1;

            store_strided_line(coefficients, dst, stride);
            return true;
        }

        template <int TransformSize>
        bool execute_fused_sin_forward(const double* const src, double* const dst,
                                       int const nbr_in, int const nbr_out,
                                       int const stride, const r2hc_precomp_t& transform)
        {
            constexpr int nbr = TransformSize + 1;
            if (nbr_in != nbr || nbr_out != nbr)
                return false;

            const std::array<double, nbr> input = load_strided_line<nbr>(src, stride);
            std::array<double, TransformSize> transformed{};
            std::array<double, nbr> coefficients{};
            const double* const sin_pi = transform.sin_pi_over_n.data();

            for (int i = 1; i < (nbr - 1) / 2; i++) {
                double fp = 0.5 * (input[i] + input[nbr - 1 - i]) * sin_pi[i];
                double fms = 0.5 * (input[i] - input[nbr - 1 - i]);
                transformed[i] = fp + fms;
                transformed[nbr - 1 - i] = fp - fms;
            }

            transformed[0] = 0.5 * (input[0] + input[nbr - 1]);
            transformed[(nbr - 1) / 2] = input[(nbr - 1) / 2];

            execute_r2hc_codelet<TransformSize>(transformed.data());

            coefficients[0] = 0;
            for (int i = 2; i < nbr - 1; i += 2)
                coefficients[i] = -2 * transformed[nbr - 1 - i / 2] / (nbr - 1);
            coefficients[nbr - 1] = 0;

            coefficients[1] = 2 * transformed[0] / (nbr - 1);
            for (int i = 3; i < nbr; i += 2)
                coefficients[i] = coefficients[i - 2]
                                  + 4 * transformed[i / 2] / (nbr - 1);

            store_strided_line(coefficients, dst, stride);
            return true;
        }

        template <int TransformSize>
        bool execute_fused_cos_even_forward(const double* const src, double* const dst,
                                            int const nbr_in, int const nbr_out,
                                            int const stride,
                                            const r2hc_precomp_t& transform)
        {
            constexpr int nbr = TransformSize + 1;
            if (nbr_in != nbr || nbr_out != nbr)
                return false;

            const std::array<double, nbr> input = load_strided_line<nbr>(src, stride);
            std::array<double, TransformSize> transformed{};
            std::array<double, nbr> coefficients{};
            const double* const sin_pi = transform.sin_pi_over_n.data();

            double fmoins0 = 0.5 * (input[0] - input[nbr - 1]);

            for (int i = 1; i < (nbr - 1) / 2; i++) {
                double fp = 0.5 * (input[i] + input[nbr - 1 - i]);
                double fms = 0.5 * (input[i] - input[nbr - 1 - i]) * sin_pi[i];
                transformed[i] = fp + fms;
                transformed[nbr - 1 - i] = fp - fms;
            }

            transformed[0] = 0.5 * (input[0] + input[nbr - 1]);
            transformed[(nbr - 1) / 2] = input[(nbr - 1) / 2];

            execute_r2hc_codelet<TransformSize>(transformed.data());

            coefficients[0] = transformed[0] / (nbr - 1);
            for (int i = 2; i < nbr - 1; i += 2)
                coefficients[i] = 2 * transformed[i / 2] / (nbr - 1);
            coefficients[nbr - 1] = transformed[(nbr - 1) / 2] / (nbr - 1);

            coefficients[1] = 0;
            double som = 0;
            for (int i = 3; i < nbr; i += 2) {
                coefficients[i] = coefficients[i - 2]
                                  + 4 * transformed[nbr - 1 - i / 2] / (nbr - 1);
                som += coefficients[i];
            }

            double c1 = (fmoins0 - som) / ((nbr - 1) / 2);

            coefficients[1] = c1;
            for (int i = 3; i < nbr; i += 2)
                coefficients[i] += c1;

            store_strided_line(coefficients, dst, stride);
            return true;
        }

        template <int TransformSize>
        bool execute_fused_cos_odd_forward(const double* const src, double* const dst,
                                           int const nbr_in, int const nbr_out,
                                           int const stride,
                                           const r2hc_precomp_t& transform)
        {
            constexpr int nbr = TransformSize + 1;
            if (nbr_in != nbr || nbr_out != nbr)
                return false;

            const std::array<double, nbr> input = load_strided_line<nbr>(src, stride);
            std::array<double, TransformSize> transformed{};
            std::array<double, nbr> cf{};
            const double* const sin_pi = transform.sin_pi_over_n.data();
            const double* const sin_half_pi = transform.sin_pi_i_over_2n.data();

            for (int i = 0; i < nbr - 1; i++)
                cf[i] = input[i] * sin_half_pi[nbr - 1 - i];
            cf[nbr - 1] = 0;
            double fmoins0 = 0.5 * (cf[0] - cf[nbr - 1]);

            for (int i = 1; i < (nbr - 1) / 2; i++) {
                double fp = 0.5 * (cf[i] + cf[nbr - 1 - i]);
                double fms = 0.5 * (cf[i] - cf[nbr - 1 - i]) * sin_pi[i];
                transformed[i] = fp + fms;
                transformed[nbr - 1 - i] = fp - fms;
            }

            transformed[0] = 0.5 * (cf[0] + cf[nbr - 1]);
            transformed[(nbr - 1) / 2] = cf[(nbr - 1) / 2];

            execute_r2hc_codelet<TransformSize>(transformed.data());

            cf[0] = transformed[0] / (nbr - 1);
            for (int i = 2; i < nbr - 1; i += 2)
                cf[i] = 2 * transformed[i / 2] / (nbr - 1);
            cf[nbr - 1] = transformed[(nbr - 1) / 2];

            cf[1] = 0;
            double som = 0;
            for (int i = 3; i < nbr; i += 2) {
                cf[i] = cf[i - 2]
                        + 4 * transformed[nbr - 1 - i / 2] / (nbr - 1);
                som += cf[i];
            }

            double c1 = (fmoins0 - som) / ((nbr - 1) / 2);

            cf[1] = c1;
            for (int i = 3; i < nbr; i += 2)
                cf[i] += c1;

            cf[0] = 2 * cf[0];
            for (int i = 1; i < nbr - 1; i++)
                cf[i] = 2 * cf[i] - cf[i - 1];
            cf[nbr - 1] = 0;

            store_strided_line(cf, dst, stride);
            return true;
        }

        template <int TransformSize>
        bool execute_fused_sin_even_forward(const double* const src, double* const dst,
                                            int const nbr_in, int const nbr_out,
                                            int const stride,
                                            const r2hc_precomp_t& transform)
        {
            constexpr int nbr = TransformSize + 1;
            if (nbr_in != nbr || nbr_out != nbr)
                return false;

            const std::array<double, nbr> input = load_strided_line<nbr>(src, stride);
            std::array<double, TransformSize> transformed{};
            std::array<double, nbr> coefficients{};
            const double* const sin_pi = transform.sin_pi_over_n.data();

            for (int i = 1; i < (nbr - 1) / 2; i++) {
                double fp = 0.5 * (input[i] + input[nbr - 1 - i]) * sin_pi[i];
                double fms = 0.5 * (input[i] - input[nbr - 1 - i]);
                transformed[i] = fp + fms;
                transformed[nbr - 1 - i] = fp - fms;
            }

            transformed[0] = 0.5 * (input[0] - input[nbr - 1]);
            transformed[(nbr - 1) / 2] = input[(nbr - 1) / 2];

            execute_r2hc_codelet<TransformSize>(transformed.data());

            coefficients[0] = 0.;
            for (int i = 2; i < nbr - 1; i += 2)
                coefficients[i] = -2 * transformed[nbr - 1 - i / 2] / (nbr - 1);
            coefficients[nbr - 1] = 0;

            coefficients[1] = 2 * transformed[0] / (nbr - 1);
            for (int i = 3; i < nbr; i += 2)
                coefficients[i] = coefficients[i - 2]
                                  + 4 * transformed[i / 2] / (nbr - 1);

            store_strided_line(coefficients, dst, stride);
            return true;
        }

        template <int TransformSize>
        bool execute_fused_sin_odd_forward(const double* const src, double* const dst,
                                           int const nbr_in, int const nbr_out,
                                           int const stride,
                                           const r2hc_precomp_t& transform)
        {
            constexpr int nbr = TransformSize + 1;
            if (nbr_in != nbr || nbr_out != nbr)
                return false;

            const std::array<double, nbr> input = load_strided_line<nbr>(src, stride);
            std::array<double, TransformSize> transformed{};
            std::array<double, nbr> cf{};
            const double* const sin_pi = transform.sin_pi_over_n.data();
            const double* const sin_half_pi = transform.sin_pi_i_over_2n.data();

            cf[0] = 0;
            for (int i = 1; i < nbr; i++)
                cf[i] = input[i] * sin_half_pi[i];
            double fmoins0 = 0.5 * (cf[0] - cf[nbr - 1]);

            for (int i = 1; i < (nbr - 1) / 2; i++) {
                double fp = 0.5 * (cf[i] + cf[nbr - 1 - i]);
#if defined(__GNUC__) && !defined(__clang__)
                // Preserve the buffered route's rounded fms intermediate. GCC
                // otherwise contracts this fold and breaks its exact corpus.
                volatile double fms =
                    0.5 * (cf[i] - cf[nbr - 1 - i]) * sin_pi[i];
#else
                double fms = 0.5 * (cf[i] - cf[nbr - 1 - i]) * sin_pi[i];
#endif
                transformed[i] = fp + fms;
                transformed[nbr - 1 - i] = fp - fms;
            }

            transformed[0] = 0.5 * (cf[0] + cf[nbr - 1]);
            transformed[(nbr - 1) / 2] = cf[(nbr - 1) / 2];

            execute_r2hc_codelet<TransformSize>(transformed.data());

            cf[0] = transformed[0] / (nbr - 1);
            for (int i = 2; i < nbr - 1; i += 2)
                cf[i] = 2 * transformed[i / 2] / (nbr - 1);
            cf[nbr - 1] = transformed[(nbr - 1) / 2] / (nbr - 1);

            cf[1] = 0;
            double som = 0;
            for (int i = 3; i < nbr; i += 2) {
                cf[i] = cf[i - 2]
                        + 4 * transformed[nbr - 1 - i / 2] / (nbr - 1);
                som += cf[i];
            }

            double c1 = (fmoins0 - som) / ((nbr - 1) / 2);

            cf[1] = c1;
            for (int i = 3; i < nbr; i += 2)
                cf[i] += c1;

            cf[0] = 2 * cf[0];
            for (int i = 1; i < nbr - 1; i++)
                cf[i] = 2 * cf[i] + cf[i - 1];
            cf[nbr - 1] = 0;

            store_strided_line(cf, dst, stride);
            return true;
        }

        template <int TransformSize>
        bool execute_fused_cos_inverse(const double* const src, double* const dst,
                                       int const nbr_in, int const nbr_out,
                                       int const stride,
                                       const r2hc_precomp_t& transform)
        {
            constexpr int nbr = TransformSize + 1;
            if (nbr_in != nbr || nbr_out != nbr)
                return false;

            const std::array<double, nbr> input = load_strided_line<nbr>(src, stride);
            std::array<double, TransformSize> transformed{};
            std::array<double, nbr> cf{};
            std::array<double, nbr> output{};
            const double* const sin_pi = transform.sin_pi_over_n.data();

            double c1 = input[1];
            double somme = 0;
            cf[1] = 0;
            for (int i = 3; i < nbr; i += 2) {
                cf[i] = input[i] - c1;
                somme += cf[i];
            }
            double fmoins0 = (nbr - 1) / 2 * c1 + somme;
            for (int i = 3; i < nbr; i += 2)
                transformed[nbr - 1 - i / 2] = 0.25 * (cf[i] - cf[i - 2]);
            transformed[0] = input[0];
            for (int i = 1; i < (nbr - 1) / 2; i++)
                transformed[i] = 0.5 * input[2 * i];
            transformed[(nbr - 1) / 2] = input[nbr - 1];

#if defined(__clang__)
            if constexpr (TransformSize == 14 || TransformSize == 16) {
                [[clang::always_inline]] execute_hc2r_codelet<TransformSize>(transformed.data());
            } else
#endif
                execute_hc2r_codelet<TransformSize>(transformed.data());

            for (int i = 1; i < (nbr - 1) / 2; i++) {
                double fp = 0.5 * (transformed[i] + transformed[nbr - 1 - i]);
                double fm = 0.5 * (transformed[i] - transformed[nbr - 1 - i]) / sin_pi[i];
                output[i] = fp + fm;
                output[nbr - i - 1] = fp - fm;
            }
            output[0] = transformed[0] + fmoins0;
            output[nbr - 1] = transformed[0] - fmoins0;
            output[(nbr - 1) / 2] = transformed[(nbr - 1) / 2];

            store_strided_line(output, dst, stride);
            return true;
        }

        template <int TransformSize>
        bool execute_fused_sin_inverse(const double* const src, double* const dst,
                                       int const nbr_in, int const nbr_out,
                                       int const stride,
                                       const r2hc_precomp_t& transform)
        {
            constexpr int nbr = TransformSize + 1;
            if (nbr_in != nbr || nbr_out != nbr)
                return false;

            const std::array<double, nbr> input = load_strided_line<nbr>(src, stride);
            std::array<double, TransformSize> transformed{};
            std::array<double, nbr> output{};
            const double* const sin_pi = transform.sin_pi_over_n.data();

            for (int i = 2; i < nbr - 1; i += 2)
                transformed[nbr - 1 - i / 2] = -0.5 * input[i];
            transformed[0] = 0.5 * input[1];
            for (int i = 3; i < nbr; i += 2)
                transformed[i / 2] = 0.25 * (input[i] - input[i - 2]);
            transformed[(nbr - 1) / 2] = -0.5 * input[nbr - 2];

#if defined(__clang__)
            if constexpr (TransformSize == 14 || TransformSize == 16) {
                [[clang::always_inline]] execute_hc2r_codelet<TransformSize>(transformed.data());
            } else
#endif
                execute_hc2r_codelet<TransformSize>(transformed.data());

            for (int i = 1; i < (nbr - 1) / 2; i++) {
                double fp = 0.5 * (transformed[i] + transformed[nbr - 1 - i]) / sin_pi[i];
                double fm = 0.5 * (transformed[i] - transformed[nbr - 1 - i]);
                output[i] = fp + fm;
                output[nbr - i - 1] = fp - fm;
            }
            output[0] = 0;
            output[nbr - 1] = -2 * transformed[0];
            output[(nbr - 1) / 2] = transformed[(nbr - 1) / 2];

            store_strided_line(output, dst, stride);
            return true;
        }

        template <int TransformSize>
        bool execute_fused_cos_even_inverse(const double* const src, double* const dst,
                                            int const nbr_in, int const nbr_out,
                                            int const stride,
                                            const r2hc_precomp_t& transform)
        {
            constexpr int nbr = TransformSize + 1;
            if (nbr_in != nbr || nbr_out != nbr)
                return false;

            const std::array<double, nbr> input = load_strided_line<nbr>(src, stride);
            std::array<double, TransformSize> transformed{};
            std::array<double, nbr> cf{};
            std::array<double, nbr> output{};
            const double* const sin_pi = transform.sin_pi_over_n.data();

            double c1 = input[1];
            double somme = 0;
            cf[1] = 0;
            for (int i = 3; i < nbr; i += 2) {
                cf[i] = input[i] - c1;
                somme += cf[i];
            }
            double fmoins0 = (nbr - 1) / 2 * c1 + somme;
            for (int i = 3; i < nbr; i += 2)
                transformed[nbr - 1 - i / 2] = 0.25 * (cf[i] - cf[i - 2]);
            transformed[0] = input[0];
            for (int i = 1; i < (nbr - 1) / 2; i++)
                transformed[i] = 0.5 * input[2 * i];
            transformed[(nbr - 1) / 2] = input[nbr - 1];

#if defined(__clang__)
            if constexpr (TransformSize == 14 || TransformSize == 16) {
                [[clang::always_inline]] execute_hc2r_codelet<TransformSize>(transformed.data());
            } else
#endif
                execute_hc2r_codelet<TransformSize>(transformed.data());

            for (int i = 1; i < (nbr - 1) / 2; i++) {
                double fp = 0.5 * (transformed[i] + transformed[nbr - 1 - i]);
                double fm = 0.5 * (transformed[i] - transformed[nbr - 1 - i]) / sin_pi[i];
                output[i] = fp + fm;
                output[nbr - i - 1] = fp - fm;
            }
            output[0] = transformed[0] + fmoins0;
            output[nbr - 1] = transformed[0] - fmoins0;
            output[(nbr - 1) / 2] = transformed[(nbr - 1) / 2];

            store_strided_line(output, dst, stride);
            return true;
        }

        template <int TransformSize>
        bool execute_fused_cos_odd_inverse(const double* const src, double* const dst,
                                           int const nbr_in, int const nbr_out,
                                           int const stride,
                                           const r2hc_precomp_t& transform)
        {
            constexpr int nbr = TransformSize + 1;
            if (nbr_in != nbr || nbr_out != nbr)
                return false;

            const std::array<double, nbr> input = load_strided_line<nbr>(src, stride);
            std::array<double, TransformSize> transformed{};
            std::array<double, nbr> ti{};
            std::array<double, nbr> cf{};
            std::array<double, nbr> output{};
            const double* const sin_pi = transform.sin_pi_over_n.data();
            const double* const sin_half_pi = transform.sin_pi_i_over_2n.data();

            ti[0] = 0.5 * input[0];
            for (int i = 1; i < nbr - 1; i++)
                ti[i] = 0.5 * (input[i] + input[i - 1]);
            ti[nbr - 1] = 0.5 * input[nbr - 2];

            double c1 = ti[1];
            double somme = 0;
            cf[1] = 0;
            for (int i = 3; i < nbr; i += 2) {
                cf[i] = ti[i] - c1;
                somme += cf[i];
            }
            double fmoins0 = (nbr - 1) / 2 * c1 + somme;
            for (int i = 3; i < nbr; i += 2)
                transformed[nbr - 1 - i / 2] = 0.25 * (cf[i] - cf[i - 2]);
            transformed[0] = ti[0];
            for (int i = 1; i < (nbr - 1) / 2; i++)
                transformed[i] = 0.5 * ti[2 * i];
            transformed[(nbr - 1) / 2] = ti[nbr - 1];

#if defined(__clang__)
            if constexpr (TransformSize == 14 || TransformSize == 16) {
                [[clang::always_inline]] execute_hc2r_codelet<TransformSize>(transformed.data());
            } else
#endif
                execute_hc2r_codelet<TransformSize>(transformed.data());

            for (int i = 1; i < (nbr - 1) / 2; i++) {
                double fp = 0.5 * (transformed[i] + transformed[nbr - 1 - i]);
                double fm = 0.5 * (transformed[i] - transformed[nbr - 1 - i]) / sin_pi[i];
                output[i] = (fp + fm) / sin_half_pi[nbr - 1 - i];
                output[nbr - i - 1] = (fp - fm) / sin_half_pi[i];
            }
            output[0] = transformed[0] + fmoins0;
            output[nbr - 1] = 0;
            output[(nbr - 1) / 2] = transformed[(nbr - 1) / 2] / transform.sin_pi_quarter;

            store_strided_line(output, dst, stride);
            return true;
        }

        template <int TransformSize>
        bool execute_fused_sin_even_inverse(const double* const src, double* const dst,
                                            int const nbr_in, int const nbr_out,
                                            int const stride,
                                            const r2hc_precomp_t& transform)
        {
            constexpr int nbr = TransformSize + 1;
            if (nbr_in != nbr || nbr_out != nbr)
                return false;

            const std::array<double, nbr> input = load_strided_line<nbr>(src, stride);
            std::array<double, TransformSize> transformed{};
            std::array<double, nbr> output{};
            const double* const sin_pi = transform.sin_pi_over_n.data();

            for (int i = 2; i < nbr - 1; i += 2)
                transformed[nbr - 1 - i / 2] = -0.5 * input[i];
            transformed[0] = 0.5 * input[1];
            for (int i = 1; i < (nbr - 1) / 2; i++)
                transformed[i] = 0.25 * (input[2 * i + 1] - input[2 * i - 1]);
            transformed[(nbr - 1) / 2] = -0.5 * input[nbr - 2];

#if defined(__clang__)
            if constexpr (TransformSize == 14 || TransformSize == 16) {
                [[clang::always_inline]] execute_hc2r_codelet<TransformSize>(transformed.data());
            } else
#endif
                execute_hc2r_codelet<TransformSize>(transformed.data());

            for (int i = 1; i < (nbr - 1) / 2; i++) {
                double fp = 0.5 * (transformed[i] + transformed[nbr - 1 - i]) / sin_pi[i];
                double fm = 0.5 * (transformed[i] - transformed[nbr - 1 - i]);
                output[i] = fp + fm;
                output[nbr - i - 1] = fp - fm;
            }
            output[0] = 0;
            output[nbr - 1] = -2 * transformed[0];
            output[(nbr - 1) / 2] = transformed[(nbr - 1) / 2];

            store_strided_line(output, dst, stride);
            return true;
        }

        template <int TransformSize>
        bool execute_fused_sin_odd_inverse(const double* const src, double* const dst,
                                           int const nbr_in, int const nbr_out,
                                           int const stride,
                                           const r2hc_precomp_t& transform)
        {
            constexpr int nbr = TransformSize + 1;
            if (nbr_in != nbr || nbr_out != nbr)
                return false;

            const std::array<double, nbr> input = load_strided_line<nbr>(src, stride);
            std::array<double, TransformSize> transformed{};
            std::array<double, nbr> ti{};
            std::array<double, nbr> cf{};
            std::array<double, nbr> output{};
            const double* const sin_pi = transform.sin_pi_over_n.data();
            const double* const sin_half_pi = transform.sin_pi_i_over_2n.data();

            ti[0] = 0.5 * input[0];
            for (int i = 1; i < nbr - 1; i++)
                ti[i] = 0.5 * (input[i] - input[i - 1]);
            ti[nbr - 1] = -0.5 * input[nbr - 2];

            double c1 = ti[1];
            double somme = 0;
            cf[1] = 0;
            for (int i = 3; i < nbr; i += 2) {
                cf[i] = ti[i] - c1;
                somme += cf[i];
            }
            double fmoins0 = (nbr - 1) / 2 * c1 + somme;
            for (int i = 3; i < nbr; i += 2)
                transformed[nbr - 1 - i / 2] = 0.25 * (cf[i] - cf[i - 2]);
            transformed[0] = ti[0];
            for (int i = 1; i < (nbr - 1) / 2; i++)
                transformed[i] = 0.5 * ti[2 * i];
            transformed[(nbr - 1) / 2] = ti[nbr - 1];

#if defined(__clang__)
            if constexpr (TransformSize == 14 || TransformSize == 16) {
                [[clang::always_inline]] execute_hc2r_codelet<TransformSize>(transformed.data());
            } else
#endif
                execute_hc2r_codelet<TransformSize>(transformed.data());

            for (int i = 1; i < (nbr - 1) / 2; i++) {
                double fp = 0.5 * (transformed[i] + transformed[nbr - 1 - i]);
                double fm = 0.5 * (transformed[i] - transformed[nbr - 1 - i]) / sin_pi[i];
                output[i] = (fp + fm) / sin_half_pi[i];
                output[nbr - i - 1] = (fp - fm) / sin_half_pi[nbr - 1 - i];
            }
            output[0] = 0;
            output[nbr - 1] = transformed[0] - fmoins0;
            output[(nbr - 1) / 2] = transformed[(nbr - 1) / 2] / transform.sin_pi_quarter;

            store_strided_line(output, dst, stride);
            return true;
        }

        template <int TransformSize, native_spectral_family Family>
        bool execute_fused_cheb_inverse(const double* const src, double* const dst,
                                        int const nbr_in, int const nbr_out,
                                        int const stride,
                                        const r2hc_precomp_t& transform)
        {
            static_assert(Family == native_spectral_family::cheb
                          || Family == native_spectral_family::cheb_even
                          || Family == native_spectral_family::cheb_odd);
            constexpr int nbr = TransformSize + 1;
            if (nbr_in != nbr || nbr_out != nbr)
                return false;

            const std::array<double, nbr> input = load_strided_line<nbr>(src, stride);
            std::array<double, nbr - 1> transformed{};
            std::array<double, nbr> output{};
            const double* const sin_pi = transform.sin_pi_over_n.data();

            if constexpr (Family == native_spectral_family::cheb_odd) {
                std::array<double, nbr> adjacent{};
                std::array<double, nbr> coefficients{};
                adjacent[0] = 0.5 * input[0];
                for (int i = 1; i < nbr - 1; ++i)
                    adjacent[static_cast<std::size_t>(i)] =
                        0.5 * (input[static_cast<std::size_t>(i)]
                               + input[static_cast<std::size_t>(i - 1)]);
                adjacent[nbr - 1] = 0.5 * input[nbr - 2];

                const double c1 = adjacent[1];
                double sum = 0.;
                coefficients[1] = 0.;
                for (int i = 3; i < nbr; i += 2) {
                    coefficients[static_cast<std::size_t>(i)] =
                        adjacent[static_cast<std::size_t>(i)] - c1;
                    sum += coefficients[static_cast<std::size_t>(i)];
                }
                const double fmoins0 = (nbr - 1) / 2 * c1 + sum;
                for (int i = 3; i < nbr; i += 2)
                    transformed[static_cast<std::size_t>(nbr - 1 - i / 2)] =
                        0.25 * (coefficients[static_cast<std::size_t>(i)]
                                - coefficients[static_cast<std::size_t>(i - 2)]);
                transformed[0] = adjacent[0];
                for (int i = 1; i < (nbr - 1) / 2; ++i)
                    transformed[static_cast<std::size_t>(i)] =
                        0.5 * adjacent[static_cast<std::size_t>(2 * i)];
                transformed[(nbr - 1) / 2] = adjacent[nbr - 1];
#if defined(__clang__)
                if constexpr (TransformSize == 14 || TransformSize == 16) {
                    [[clang::always_inline]] execute_hc2r_codelet<TransformSize>(transformed.data());
                } else
#endif
                    execute_hc2r_codelet<TransformSize>(transformed.data());

                const double* const sin_half = transform.sin_pi_i_over_2n.data();
                for (int i = 1; i < (nbr - 1) / 2; ++i) {
                    const double fp = 0.5 * (transformed[static_cast<std::size_t>(i)]
                                             + transformed[static_cast<std::size_t>(nbr - 1 - i)]);
                    const double fm = 0.5 * (transformed[static_cast<std::size_t>(i)]
                                             - transformed[static_cast<std::size_t>(nbr - 1 - i)])
                                      / sin_pi[i];
                    output[static_cast<std::size_t>(nbr - 1 - i)] =
                        (fp + fm) / sin_half[nbr - 1 - i];
                    output[static_cast<std::size_t>(i)] = (fp - fm) / sin_half[i];
                }
                output[0] = 0.;
                output[nbr - 1] = transformed[0] + fmoins0;
                output[(nbr - 1) / 2] =
                    transformed[(nbr - 1) / 2] / transform.sin_pi_quarter;
            }
            else {
                constexpr bool even = Family == native_spectral_family::cheb_even;
                std::array<double, nbr> coefficients{};
                const double c1 = input[1];
                double sum = 0.;
                coefficients[1] = 0.;
                for (int i = 3; i < nbr; i += 2) {
                    coefficients[static_cast<std::size_t>(i)] =
                        input[static_cast<std::size_t>(i)] - c1;
                    sum += coefficients[static_cast<std::size_t>(i)];
                }
                const double fmoins0 = even ? (nbr - 1) / 2 * c1 + sum
                                           : -(nbr - 1) / 2 * c1 - sum;
                for (int i = 3; i < nbr; i += 2) {
                    const double difference = coefficients[static_cast<std::size_t>(i)]
                                              - coefficients[static_cast<std::size_t>(i - 2)];
                    transformed[static_cast<std::size_t>(nbr - 1 - i / 2)] =
                        (even ? 0.25 : -0.25) * difference;
                }
                transformed[0] = input[0];
                for (int i = 1; i < (nbr - 1) / 2; ++i)
                    transformed[static_cast<std::size_t>(i)] =
                        0.5 * input[static_cast<std::size_t>(2 * i)];
                transformed[(nbr - 1) / 2] = input[nbr - 1];
#if defined(__clang__)
                if constexpr (TransformSize == 14 || TransformSize == 16) {
                    [[clang::always_inline]] execute_hc2r_codelet<TransformSize>(transformed.data());
                } else
#endif
                    execute_hc2r_codelet<TransformSize>(transformed.data());

                for (int i = 1; i < (nbr - 1) / 2; ++i) {
                    const double fp = 0.5 * (transformed[static_cast<std::size_t>(i)]
                                             + transformed[static_cast<std::size_t>(nbr - 1 - i)]);
                    const double fm = 0.5 * (transformed[static_cast<std::size_t>(i)]
                                             - transformed[static_cast<std::size_t>(nbr - 1 - i)])
                                      / sin_pi[i];
                    if constexpr (even) {
                        output[static_cast<std::size_t>(nbr - 1 - i)] = fp + fm;
                        output[static_cast<std::size_t>(i)] = fp - fm;
                    }
                    else {
                        output[static_cast<std::size_t>(i)] = fp + fm;
                        output[static_cast<std::size_t>(nbr - i - 1)] = fp - fm;
                    }
                }
                if constexpr (even) {
                    output[0] = transformed[0] - fmoins0;
                    output[nbr - 1] = transformed[0] + fmoins0;
                }
                else {
                    output[0] = transformed[0] + fmoins0;
                    output[nbr - 1] = transformed[0] - fmoins0;
                }
                output[(nbr - 1) / 2] = transformed[(nbr - 1) / 2];
            }

            store_strided_line(output, dst, stride);
            return true;
        }

        template <int TransformSize>
        bool execute_fused_cossin_forward(const double* const src, double* const dst,
                                          int const nbr_in, int const nbr_out,
                                          int const stride)
        {
            constexpr int nbr = TransformSize + 2;
            if (nbr_out != nbr || nbr_in > nbr || nbr_in < 0)
                return false;

            std::array<double, TransformSize> transformed{};
            const int gathered = std::min(TransformSize, nbr_in);
            for (int i = 0; i < gathered; ++i)
                transformed[static_cast<std::size_t>(i)] =
                    src[static_cast<std::size_t>(i) * stride];
            execute_r2hc_codelet<TransformSize>(transformed.data());

            dst[0] = transformed[0] / double(TransformSize);
            dst[stride] = 0.;
            int index = 2;
            for (int i = 1; i < TransformSize / 2; ++i) {
                dst[static_cast<std::size_t>(index++) * stride] =
                    2. * transformed[static_cast<std::size_t>(i)] / double(TransformSize);
                dst[static_cast<std::size_t>(index++) * stride] =
                    -2. * transformed[static_cast<std::size_t>(TransformSize - i)]
                    / double(TransformSize);
            }
            dst[static_cast<std::size_t>(index++) * stride] =
                transformed[TransformSize / 2] / double(TransformSize);
            dst[static_cast<std::size_t>(index) * stride] = 0.;
            return true;
        }

        template <int TransformSize>
        bool execute_fused_cossin_inverse(const double* const src, double* const dst,
                                          int const nbr_in, int const nbr_out,
                                          int const stride)
        {
            constexpr int nbr = TransformSize + 2;
            if (nbr_in != nbr || nbr_out < 0 || nbr_out > nbr)
                return false;

            const std::array<double, nbr> input = load_strided_line<nbr>(src, stride);
            std::array<double, TransformSize> transformed{};
            transformed[0] = input[0];
            for (int i = 1; i < TransformSize / 2; ++i) {
                transformed[static_cast<std::size_t>(i)] =
                    0.5 * input[static_cast<std::size_t>(2 * i)];
                transformed[static_cast<std::size_t>(TransformSize - i)] =
                    -0.5 * input[static_cast<std::size_t>(2 * i + 1)];
            }
            transformed[TransformSize / 2] = input[TransformSize];
            execute_hc2r_codelet<TransformSize>(transformed.data());

            const int written = std::min(TransformSize, nbr_out);
            for (int i = 0; i < written; ++i)
                dst[static_cast<std::size_t>(i) * stride] =
                    transformed[static_cast<std::size_t>(i)];
            for (int i = written; i < nbr_out; ++i)
                dst[static_cast<std::size_t>(i) * stride] = input[static_cast<std::size_t>(i)];
            return true;
        }

#if defined(__linux__) && defined(__x86_64__) && defined(__GNUC__) \
    && !defined(__clang__)
        // GCC keeps the fused route's automatic stage arrays live across the
        // fixed-size codelet.  Reuse the plan-owned transform buffer and one
        // auxiliary lane so the production N=10--16 route has no stage copies.
        // Keep only the family helpers out of line with loop vectorization
        // disabled to avoid alias-versioned pointer loops; their called
        // codelets retain straight-line SLP/FMA.
#define CELEPHAIS_GCC_PERSISTENT_FAMILY_HELPER \
    __attribute__((noinline, optimize("no-tree-loop-vectorize")))
        template <int TransformSize, native_spectral_family Family>
        CELEPHAIS_GCC_PERSISTENT_FAMILY_HELPER bool execute_persistent_cheb_forward(
            const double* const src, double* const dst, int const nbr_in,
            int const nbr_out, int const stride,
            const r2hc_precomp_t& transform, double* const transformed,
            double* const scratch)
        {
            static_assert(Family == native_spectral_family::cheb
                          || Family == native_spectral_family::cheb_even
                          || Family == native_spectral_family::cheb_odd);
            constexpr int nbr = TransformSize + 1;
            if (nbr_in != nbr || nbr_out != nbr)
                return false;

            double* const coefficients = scratch;
            const double* const sin_pi = transform.sin_pi_over_n.data();

            if constexpr (Family == native_spectral_family::cheb_odd) {
                const double* const sin_half =
                    transform.sin_half_pi_i_over_n.data();
                for (int i = 0; i < nbr; ++i)
                    coefficients[i] = src[i * stride] * sin_half[i];

                const double fmoins0 =
                    -0.5 * (coefficients[0] - coefficients[nbr - 1]);
                for (int i = 1; i < (nbr - 1) / 2; ++i) {
                    const double fp =
                        0.5 * (coefficients[i] + coefficients[nbr - 1 - i]);
                    const double fms =
                        0.5 * (-coefficients[i] + coefficients[nbr - 1 - i])
                        * sin_pi[i];
                    transformed[i] = fp + fms;
                    transformed[nbr - 1 - i] = fp - fms;
                }
                transformed[0] =
                    0.5 * (coefficients[0] + coefficients[nbr - 1]);
                transformed[(nbr - 1) / 2] = coefficients[(nbr - 1) / 2];
                execute_r2hc_codelet<TransformSize>(transformed);

                dst[0] = transformed[0] / (nbr - 1);
                for (int i = 2; i < nbr - 1; i += 2)
                    dst[i * stride] = 2 * transformed[i / 2] / (nbr - 1);
                dst[(nbr - 1) * stride] =
                    transformed[(nbr - 1) / 2] / (nbr - 1);
                dst[stride] = 0.;
                double sum = 0.;
                for (int i = 3; i < nbr; i += 2) {
                    dst[i * stride] =
                        dst[(i - 2) * stride]
                        + 4 * transformed[nbr - 1 - i / 2] / (nbr - 1);
                    sum += dst[i * stride];
                }
                const double c1 = (fmoins0 - sum) / ((nbr - 1) / 2);
                dst[stride] = c1;
                for (int i = 3; i < nbr; i += 2)
                    dst[i * stride] += c1;

                dst[0] = 2 * dst[0];
                for (int i = 1; i < nbr - 1; ++i)
                    dst[i * stride] =
                        2 * dst[i * stride] - dst[(i - 1) * stride];
                dst[(nbr - 1) * stride] = 0.;
            }
            else {
                constexpr bool even = Family == native_spectral_family::cheb_even;
                const double fmoins0 =
                    (even ? -0.5 : 0.5) * (src[0] - src[(nbr - 1) * stride]);
                for (int i = 1; i < (nbr - 1) / 2; ++i) {
                    const double fp =
                        0.5 * (src[i * stride] + src[(nbr - 1 - i) * stride]);
                    const double fms =
                        even ? 0.5
                                   * (-src[i * stride]
                                      + src[(nbr - 1 - i) * stride])
                                   * sin_pi[i]
                             : 0.5
                                   * (src[i * stride]
                                      - src[(nbr - 1 - i) * stride])
                                   * sin_pi[i];
                    transformed[i] = fp + fms;
                    transformed[nbr - 1 - i] = fp - fms;
                }
                transformed[0] = 0.5 * (src[0] + src[(nbr - 1) * stride]);
                transformed[(nbr - 1) / 2] = src[(nbr - 1) / 2 * stride];
                execute_r2hc_codelet<TransformSize>(transformed);

                dst[0] = transformed[0] / (nbr - 1);
                for (int i = 2; i < nbr - 1; i += 2)
                    dst[i * stride] = 2 * transformed[i / 2] / (nbr - 1);
                dst[(nbr - 1) * stride] =
                    transformed[(nbr - 1) / 2] / (nbr - 1);
                dst[stride] = 0.;
                double sum = 0.;
                for (int i = 3; i < nbr; i += 2) {
                    if constexpr (even)
                        dst[i * stride] =
                            dst[(i - 2) * stride]
                            + 4 * transformed[nbr - 1 - i / 2] / (nbr - 1);
                    else
                        dst[i * stride] =
                            dst[(i - 2) * stride]
                            - 4 * transformed[nbr - 1 - (i - 1) / 2]
                                  / (nbr - 1);
                    sum += dst[i * stride];
                }
                const double c1 = even ? (fmoins0 - sum) / ((nbr - 1) / 2)
                                       : -(fmoins0 + sum) / ((nbr - 1) / 2);
                dst[stride] = c1;
                for (int i = 3; i < nbr; i += 2)
                    dst[i * stride] += c1;
            }
            return true;
        }

        template <int TransformSize>
        CELEPHAIS_GCC_PERSISTENT_FAMILY_HELPER bool execute_persistent_cos_forward(
            const double* const src, double* const dst, int const nbr_in,
            int const nbr_out, int const stride,
            const r2hc_precomp_t& transform, double* const transformed)
        {
            constexpr int nbr = TransformSize + 1;
            if (nbr_in != nbr || nbr_out != nbr)
                return false;

            const double* const sin_pi = transform.sin_pi_over_n.data();
            const double fmoins0 =
                0.5 * (src[0] - src[(nbr - 1) * stride]);

            for (int i = 1; i < (nbr - 1) / 2; ++i) {
                const double fp =
                    0.5 * (src[i * stride] + src[(nbr - 1 - i) * stride]);
                const double fms =
                    0.5 * (src[i * stride] - src[(nbr - 1 - i) * stride])
                    * sin_pi[i];
                transformed[i] = fp + fms;
                transformed[nbr - 1 - i] = fp - fms;
            }
            transformed[0] = 0.5 * (src[0] + src[(nbr - 1) * stride]);
            transformed[(nbr - 1) / 2] = src[(nbr - 1) / 2 * stride];
            execute_r2hc_codelet<TransformSize>(transformed);

            dst[0] = transformed[0] / (nbr - 1);
            for (int i = 2; i < nbr - 1; i += 2)
                dst[i * stride] = 2 * transformed[i / 2] / (nbr - 1);
            dst[(nbr - 1) * stride] =
                transformed[(nbr - 1) / 2] / (nbr - 1);
            dst[stride] = 0;
            double sum = 0;
            for (int i = 3; i < nbr; i += 2) {
                dst[i * stride] =
                    dst[(i - 2) * stride]
                    + 4 * transformed[nbr - 1 - i / 2] / (nbr - 1);
                sum += dst[i * stride];
            }
            const double c1 = (fmoins0 - sum) / ((nbr - 1) / 2);
            dst[stride] = c1;
            for (int i = 3; i < nbr; i += 2)
                dst[i * stride] += c1;

            return true;
        }

        template <int TransformSize, bool Even>
        CELEPHAIS_GCC_PERSISTENT_FAMILY_HELPER bool execute_persistent_sin_forward(
            const double* const src, double* const dst, int const nbr_in,
            int const nbr_out, int const stride,
            const r2hc_precomp_t& transform, double* const transformed)
        {
            constexpr int nbr = TransformSize + 1;
            if (nbr_in != nbr || nbr_out != nbr)
                return false;

            const double* const sin_pi = transform.sin_pi_over_n.data();

            for (int i = 1; i < (nbr - 1) / 2; ++i) {
                const double fp =
                    0.5 * (src[i * stride] + src[(nbr - 1 - i) * stride])
                    * sin_pi[i];
                const double fms =
                    0.5 * (src[i * stride] - src[(nbr - 1 - i) * stride]);
                transformed[i] = fp + fms;
                transformed[nbr - 1 - i] = fp - fms;
            }
            if constexpr (Even)
                transformed[0] = 0.5 * (src[0] - src[(nbr - 1) * stride]);
            else
                transformed[0] = 0.5 * (src[0] + src[(nbr - 1) * stride]);
            transformed[(nbr - 1) / 2] = src[(nbr - 1) / 2 * stride];
            execute_r2hc_codelet<TransformSize>(transformed);

            dst[0] = 0.;
            for (int i = 2; i < nbr - 1; i += 2)
                dst[i * stride] =
                    -2 * transformed[nbr - 1 - i / 2] / (nbr - 1);
            dst[(nbr - 1) * stride] = 0;
            dst[stride] = 2 * transformed[0] / (nbr - 1);
            for (int i = 3; i < nbr; i += 2)
                dst[i * stride] =
                    dst[(i - 2) * stride]
                    + 4 * transformed[i / 2] / (nbr - 1);

            return true;
        }

        template <int TransformSize, native_spectral_family Family>
        CELEPHAIS_GCC_PERSISTENT_FAMILY_HELPER bool execute_persistent_odd_forward(
            const double* const src, double* const dst, int const nbr_in,
            int const nbr_out, int const stride,
            const r2hc_precomp_t& transform, double* const transformed,
            double* const scratch)
        {
            static_assert(Family == native_spectral_family::cos_odd
                          || Family == native_spectral_family::sin_odd);
            constexpr bool cosine = Family == native_spectral_family::cos_odd;
            constexpr int nbr = TransformSize + 1;
            if (nbr_in != nbr || nbr_out != nbr)
                return false;

            double* const coefficients = scratch;
            const double* const sin_pi = transform.sin_pi_over_n.data();
            const double* const sin_half = transform.sin_pi_i_over_2n.data();

            if constexpr (cosine) {
                for (int i = 0; i < nbr - 1; ++i)
                    coefficients[i] = src[i * stride] * sin_half[nbr - 1 - i];
                coefficients[nbr - 1] = 0;
            }
            else {
                coefficients[0] = 0;
                for (int i = 1; i < nbr; ++i)
                    coefficients[i] = src[i * stride] * sin_half[i];
            }
            const double fmoins0 =
                0.5 * (coefficients[0] - coefficients[nbr - 1]);

            for (int i = 1; i < (nbr - 1) / 2; ++i) {
                const double fp =
                    0.5 * (coefficients[i] + coefficients[nbr - 1 - i]);
                if constexpr (cosine) {
                    const double fms =
                        0.5 * (coefficients[i] - coefficients[nbr - 1 - i])
                        * sin_pi[i];
                    transformed[i] = fp + fms;
                    transformed[nbr - 1 - i] = fp - fms;
                }
                else {
                    volatile double fms =
                        0.5 * (coefficients[i] - coefficients[nbr - 1 - i])
                        * sin_pi[i];
                    transformed[i] = fp + fms;
                    transformed[nbr - 1 - i] = fp - fms;
                }
            }
            transformed[0] =
                0.5 * (coefficients[0] + coefficients[nbr - 1]);
            transformed[(nbr - 1) / 2] = coefficients[(nbr - 1) / 2];
            execute_r2hc_codelet<TransformSize>(transformed);

            dst[0] = transformed[0] / (nbr - 1);
            for (int i = 2; i < nbr - 1; i += 2)
                dst[i * stride] = 2 * transformed[i / 2] / (nbr - 1);
            if constexpr (cosine)
                dst[(nbr - 1) * stride] = transformed[(nbr - 1) / 2];
            else
                dst[(nbr - 1) * stride] =
                    transformed[(nbr - 1) / 2] / (nbr - 1);
            dst[stride] = 0;
            double sum = 0;
            for (int i = 3; i < nbr; i += 2) {
                dst[i * stride] =
                    dst[(i - 2) * stride]
                    + 4 * transformed[nbr - 1 - i / 2] / (nbr - 1);
                sum += dst[i * stride];
            }
            const double c1 = (fmoins0 - sum) / ((nbr - 1) / 2);
            dst[stride] = c1;
            for (int i = 3; i < nbr; i += 2)
                dst[i * stride] += c1;

            dst[0] = 2 * dst[0];
            for (int i = 1; i < nbr - 1; ++i) {
                if constexpr (cosine)
                    dst[i * stride] =
                        2 * dst[i * stride] - dst[(i - 1) * stride];
                else
                    dst[i * stride] =
                        2 * dst[i * stride] + dst[(i - 1) * stride];
            }
            dst[(nbr - 1) * stride] = 0;

            return true;
        }

        template <int TransformSize, native_spectral_family Family>
        CELEPHAIS_GCC_PERSISTENT_FAMILY_HELPER bool execute_persistent_cheb_inverse(
            const double* const src, double* const dst, int const nbr_in,
            int const nbr_out, int const stride,
            const r2hc_precomp_t& transform, double* const transformed,
            double* const scratch)
        {
            static_assert(Family == native_spectral_family::cheb
                          || Family == native_spectral_family::cheb_even
                          || Family == native_spectral_family::cheb_odd);
            constexpr int nbr = TransformSize + 1;
            if (nbr_in != nbr || nbr_out != nbr)
                return false;

            double* const auxiliary = scratch;
            const double* const sin_pi = transform.sin_pi_over_n.data();

            if constexpr (Family == native_spectral_family::cheb_odd) {
                auxiliary[0] = 0.5 * src[0];
                for (int i = 1; i < nbr - 1; ++i)
                    auxiliary[i] =
                        0.5 * (src[i * stride] + src[(i - 1) * stride]);
                auxiliary[nbr - 1] = 0.5 * src[(nbr - 2) * stride];

                const double c1 = auxiliary[1];
                double sum = 0.;
                auxiliary[1] = 0.;
                for (int i = 3; i < nbr; i += 2) {
                    auxiliary[i] = auxiliary[i] - c1;
                    sum += auxiliary[i];
                }
                const double fmoins0 = (nbr - 1) / 2 * c1 + sum;
                for (int i = 3; i < nbr; i += 2)
                    transformed[nbr - 1 - i / 2] =
                        0.25 * (auxiliary[i] - auxiliary[i - 2]);
                transformed[0] = auxiliary[0];
                for (int i = 1; i < (nbr - 1) / 2; ++i)
                    transformed[i] = 0.5 * auxiliary[2 * i];
                transformed[(nbr - 1) / 2] = auxiliary[nbr - 1];
                execute_hc2r_codelet<TransformSize>(transformed);

                const double* const sin_half =
                    transform.sin_pi_i_over_2n.data();
                for (int i = 1; i < (nbr - 1) / 2; ++i) {
                    const double fp =
                        0.5 * (transformed[i] + transformed[nbr - 1 - i]);
                    const double fm =
                        0.5 * (transformed[i] - transformed[nbr - 1 - i])
                        / sin_pi[i];
                    dst[(nbr - 1 - i) * stride] =
                        (fp + fm) / sin_half[nbr - 1 - i];
                    dst[i * stride] = (fp - fm) / sin_half[i];
                }
                dst[0] = 0.;
                dst[(nbr - 1) * stride] = transformed[0] + fmoins0;
                dst[(nbr - 1) / 2 * stride] =
                    transformed[(nbr - 1) / 2] / transform.sin_pi_quarter;
            }
            else {
                constexpr bool even = Family == native_spectral_family::cheb_even;
                const double c1 = src[stride];
                double sum = 0.;
                auxiliary[1] = 0.;
                for (int i = 3; i < nbr; i += 2) {
                    auxiliary[i] = src[i * stride] - c1;
                    sum += auxiliary[i];
                }
                const double fmoins0 = even ? (nbr - 1) / 2 * c1 + sum
                                           : -(nbr - 1) / 2 * c1 - sum;
                for (int i = 3; i < nbr; i += 2) {
                    const double difference = auxiliary[i] - auxiliary[i - 2];
                    transformed[nbr - 1 - i / 2] =
                        (even ? 0.25 : -0.25) * difference;
                }
                transformed[0] = src[0];
                for (int i = 1; i < (nbr - 1) / 2; ++i)
                    transformed[i] = 0.5 * src[2 * i * stride];
                transformed[(nbr - 1) / 2] = src[(nbr - 1) * stride];
                execute_hc2r_codelet<TransformSize>(transformed);

                for (int i = 1; i < (nbr - 1) / 2; ++i) {
                    const double fp =
                        0.5 * (transformed[i] + transformed[nbr - 1 - i]);
                    const double fm =
                        0.5 * (transformed[i] - transformed[nbr - 1 - i])
                        / sin_pi[i];
                    if constexpr (even) {
                        dst[(nbr - 1 - i) * stride] = fp + fm;
                        dst[i * stride] = fp - fm;
                    }
                    else {
                        dst[i * stride] = fp + fm;
                        dst[(nbr - i - 1) * stride] = fp - fm;
                    }
                }
                if constexpr (even) {
                    dst[0] = transformed[0] - fmoins0;
                    dst[(nbr - 1) * stride] = transformed[0] + fmoins0;
                }
                else {
                    dst[0] = transformed[0] + fmoins0;
                    dst[(nbr - 1) * stride] = transformed[0] - fmoins0;
                }
                dst[(nbr - 1) / 2 * stride] = transformed[(nbr - 1) / 2];
            }

            return true;
        }

        template <int TransformSize>
        CELEPHAIS_GCC_PERSISTENT_FAMILY_HELPER bool execute_persistent_cos_inverse(
            const double* const src, double* const dst, int const nbr_in,
            int const nbr_out, int const stride,
            const r2hc_precomp_t& transform, double* const transformed,
            double* const scratch)
        {
            constexpr int nbr = TransformSize + 1;
            if (nbr_in != nbr || nbr_out != nbr)
                return false;

            double* const coefficients = scratch;
            const double* const sin_pi = transform.sin_pi_over_n.data();

            const double c1 = src[stride];
            double sum = 0;
            coefficients[1] = 0;
            for (int i = 3; i < nbr; i += 2) {
                coefficients[i] = src[i * stride] - c1;
                sum += coefficients[i];
            }
            const double fmoins0 = (nbr - 1) / 2 * c1 + sum;
            for (int i = 3; i < nbr; i += 2)
                transformed[nbr - 1 - i / 2] =
                    0.25 * (coefficients[i] - coefficients[i - 2]);
            transformed[0] = src[0];
            for (int i = 1; i < (nbr - 1) / 2; ++i)
                transformed[i] = 0.5 * src[2 * i * stride];
            transformed[(nbr - 1) / 2] = src[(nbr - 1) * stride];
            execute_hc2r_codelet<TransformSize>(transformed);

            for (int i = 1; i < (nbr - 1) / 2; ++i) {
                const double fp =
                    0.5 * (transformed[i] + transformed[nbr - 1 - i]);
                const double fm =
                    0.5 * (transformed[i] - transformed[nbr - 1 - i]) / sin_pi[i];
                dst[i * stride] = fp + fm;
                dst[(nbr - i - 1) * stride] = fp - fm;
            }
            dst[0] = transformed[0] + fmoins0;
            dst[(nbr - 1) * stride] = transformed[0] - fmoins0;
            dst[(nbr - 1) / 2 * stride] = transformed[(nbr - 1) / 2];

            return true;
        }

        template <int TransformSize>
        CELEPHAIS_GCC_PERSISTENT_FAMILY_HELPER bool execute_persistent_sin_inverse(
            const double* const src, double* const dst, int const nbr_in,
            int const nbr_out, int const stride,
            const r2hc_precomp_t& transform, double* const transformed)
        {
            constexpr int nbr = TransformSize + 1;
            if (nbr_in != nbr || nbr_out != nbr)
                return false;

            const double* const sin_pi = transform.sin_pi_over_n.data();

            for (int i = 2; i < nbr - 1; i += 2)
                transformed[nbr - 1 - i / 2] = -0.5 * src[i * stride];
            transformed[0] = 0.5 * src[stride];
            for (int i = 3; i < nbr; i += 2)
                transformed[i / 2] =
                    0.25 * (src[i * stride] - src[(i - 2) * stride]);
            transformed[(nbr - 1) / 2] = -0.5 * src[(nbr - 2) * stride];
            execute_hc2r_codelet<TransformSize>(transformed);

            for (int i = 1; i < (nbr - 1) / 2; ++i) {
                const double fp =
                    0.5 * (transformed[i] + transformed[nbr - 1 - i]) / sin_pi[i];
                const double fm =
                    0.5 * (transformed[i] - transformed[nbr - 1 - i]);
                dst[i * stride] = fp + fm;
                dst[(nbr - i - 1) * stride] = fp - fm;
            }
            dst[0] = 0;
            dst[(nbr - 1) * stride] = -2 * transformed[0];
            dst[(nbr - 1) / 2 * stride] = transformed[(nbr - 1) / 2];

            return true;
        }

        template <int TransformSize, native_spectral_family Family>
        CELEPHAIS_GCC_PERSISTENT_FAMILY_HELPER bool execute_persistent_odd_inverse(
            const double* const src, double* const dst, int const nbr_in,
            int const nbr_out, int const stride,
            const r2hc_precomp_t& transform, double* const transformed,
            double* const scratch)
        {
            static_assert(Family == native_spectral_family::cos_odd
                          || Family == native_spectral_family::sin_odd);
            constexpr bool cosine = Family == native_spectral_family::cos_odd;
            constexpr int nbr = TransformSize + 1;
            if (nbr_in != nbr || nbr_out != nbr)
                return false;

            double* const auxiliary = scratch;
            const double* const sin_pi = transform.sin_pi_over_n.data();
            const double* const sin_half = transform.sin_pi_i_over_2n.data();

            auxiliary[0] = 0.5 * src[0];
            for (int i = 1; i < nbr - 1; ++i) {
                if constexpr (cosine)
                    auxiliary[i] =
                        0.5 * (src[i * stride] + src[(i - 1) * stride]);
                else
                    auxiliary[i] =
                        0.5 * (src[i * stride] - src[(i - 1) * stride]);
            }
            if constexpr (cosine)
                auxiliary[nbr - 1] = 0.5 * src[(nbr - 2) * stride];
            else
                auxiliary[nbr - 1] = -0.5 * src[(nbr - 2) * stride];

            const double c1 = auxiliary[1];
            double sum = 0;
            auxiliary[1] = 0;
            for (int i = 3; i < nbr; i += 2) {
                auxiliary[i] = auxiliary[i] - c1;
                sum += auxiliary[i];
            }
            const double fmoins0 = (nbr - 1) / 2 * c1 + sum;
            for (int i = 3; i < nbr; i += 2)
                transformed[nbr - 1 - i / 2] =
                    0.25 * (auxiliary[i] - auxiliary[i - 2]);
            transformed[0] = auxiliary[0];
            for (int i = 1; i < (nbr - 1) / 2; ++i)
                transformed[i] = 0.5 * auxiliary[2 * i];
            transformed[(nbr - 1) / 2] = auxiliary[nbr - 1];
            execute_hc2r_codelet<TransformSize>(transformed);

            for (int i = 1; i < (nbr - 1) / 2; ++i) {
                const double fp =
                    0.5 * (transformed[i] + transformed[nbr - 1 - i]);
                const double fm =
                    0.5 * (transformed[i] - transformed[nbr - 1 - i]) / sin_pi[i];
                if constexpr (cosine) {
                    dst[i * stride] = (fp + fm) / sin_half[nbr - 1 - i];
                    dst[(nbr - i - 1) * stride] = (fp - fm) / sin_half[i];
                }
                else {
                    dst[i * stride] = (fp + fm) / sin_half[i];
                    dst[(nbr - i - 1) * stride] =
                        (fp - fm) / sin_half[nbr - 1 - i];
                }
            }
            if constexpr (cosine) {
                dst[0] = transformed[0] + fmoins0;
                dst[(nbr - 1) * stride] = 0;
            }
            else {
                dst[0] = 0;
                dst[(nbr - 1) * stride] = transformed[0] - fmoins0;
            }
            dst[(nbr - 1) / 2 * stride] =
                transformed[(nbr - 1) / 2] / transform.sin_pi_quarter;

            return true;
        }

        template <int TransformSize>
        CELEPHAIS_GCC_PERSISTENT_FAMILY_HELPER bool execute_persistent_cossin_forward(
            const double* const src, double* const dst, int const nbr_in,
            int const nbr_out, int const stride, double* const transformed)
        {
            constexpr int nbr = TransformSize + 2;
            if (nbr_out != nbr || nbr_in > nbr || nbr_in < 0)
                return false;

            // Exact workspace aliasing is diverted before this helper.  Copy
            // first so a full production input does not write every slot twice.
            const int gathered = std::min(TransformSize, nbr_in);
            for (int i = 0; i < gathered; ++i)
                transformed[i] = src[static_cast<std::size_t>(i) * stride];
            for (int i = gathered; i < TransformSize; ++i)
                transformed[i] = 0.;
            execute_r2hc_codelet<TransformSize>(transformed);

            dst[0] = transformed[0] / double(TransformSize);
            dst[stride] = 0.;
            int index = 2;
            for (int i = 1; i < TransformSize / 2; ++i) {
                dst[static_cast<std::size_t>(index++) * stride] =
                    2. * transformed[i] / double(TransformSize);
                dst[static_cast<std::size_t>(index++) * stride] =
                    -2. * transformed[TransformSize - i] / double(TransformSize);
            }
            dst[static_cast<std::size_t>(index++) * stride] =
                transformed[TransformSize / 2] / double(TransformSize);
            dst[static_cast<std::size_t>(index) * stride] = 0.;
            return true;
        }

        template <int TransformSize>
        CELEPHAIS_GCC_PERSISTENT_FAMILY_HELPER bool execute_persistent_cossin_inverse(
            const double* const src, double* const dst, int const nbr_in,
            int const nbr_out, int const stride, double* const transformed)
        {
            constexpr int nbr = TransformSize + 2;
            if (nbr_in != nbr || nbr_out < 0 || nbr_out > nbr)
                return false;

            const double first_tail =
                nbr_out > TransformSize ? src[TransformSize * stride] : 0.;
            const double second_tail =
                nbr_out > TransformSize + 1
                    ? src[(TransformSize + 1) * stride]
                    : 0.;
            transformed[0] = src[0];
            for (int i = 1; i < TransformSize / 2; ++i) {
                transformed[i] = 0.5 * src[2 * i * stride];
                transformed[TransformSize - i] =
                    -0.5 * src[(2 * i + 1) * stride];
            }
            transformed[TransformSize / 2] = src[TransformSize * stride];
            execute_hc2r_codelet<TransformSize>(transformed);

            const int written = std::min(TransformSize, nbr_out);
            for (int i = 0; i < written; ++i)
                dst[static_cast<std::size_t>(i) * stride] = transformed[i];
            if (nbr_out > TransformSize)
                dst[static_cast<std::size_t>(TransformSize) * stride] = first_tail;
            if (nbr_out > TransformSize + 1)
                dst[static_cast<std::size_t>(TransformSize + 1) * stride] =
                    second_tail;
            return true;
        }

#undef CELEPHAIS_GCC_PERSISTENT_FAMILY_HELPER

        template <int TransformSize>
        bool dispatch_persistent_forward(
            native_spectral_family const family, const double* const src,
            double* const dst, int const nbr_in, int const nbr_out,
            int const stride, const r2hc_precomp_t& transform,
            double* const transformed, double* const scratch)
        {
            switch (family) {
                case native_spectral_family::cheb:
                    return execute_persistent_cheb_forward<
                        TransformSize, native_spectral_family::cheb>(
                        src, dst, nbr_in, nbr_out, stride, transform,
                        transformed, scratch);
                case native_spectral_family::cheb_even:
                    return execute_persistent_cheb_forward<
                        TransformSize, native_spectral_family::cheb_even>(
                        src, dst, nbr_in, nbr_out, stride, transform,
                        transformed, scratch);
                case native_spectral_family::cheb_odd:
                    return execute_persistent_cheb_forward<
                        TransformSize, native_spectral_family::cheb_odd>(
                        src, dst, nbr_in, nbr_out, stride, transform,
                        transformed, scratch);
                case native_spectral_family::cossin:
                    // COSSIN is the only forward family whose N-value input
                    // can occupy the public N-slot transform buffer.  Keep
                    // that representable workspace alias on the existing
                    // stack-fused helper before the persistent path clears it.
                    if (src == transformed)
                        return execute_fused_cossin_forward<TransformSize>(
                            src, dst, nbr_in, nbr_out, stride);
                    return execute_persistent_cossin_forward<TransformSize>(
                        src, dst, nbr_in, nbr_out, stride, transformed);
                case native_spectral_family::cos:
                case native_spectral_family::cos_even:
                    return execute_persistent_cos_forward<TransformSize>(
                        src, dst, nbr_in, nbr_out, stride, transform,
                        transformed);
                case native_spectral_family::sin:
                    return execute_persistent_sin_forward<TransformSize, false>(
                        src, dst, nbr_in, nbr_out, stride, transform,
                        transformed);
                case native_spectral_family::cos_odd:
                    return execute_persistent_odd_forward<
                        TransformSize, native_spectral_family::cos_odd>(
                        src, dst, nbr_in, nbr_out, stride, transform,
                        transformed, scratch);
                case native_spectral_family::sin_even:
                    return execute_persistent_sin_forward<TransformSize, true>(
                        src, dst, nbr_in, nbr_out, stride, transform,
                        transformed);
                case native_spectral_family::sin_odd:
                    return execute_persistent_odd_forward<
                        TransformSize, native_spectral_family::sin_odd>(
                        src, dst, nbr_in, nbr_out, stride, transform,
                        transformed, scratch);
                default: return false;
            }
        }

        template <int TransformSize>
        bool dispatch_persistent_inverse(
            native_spectral_family const family, const double* const src,
            double* const dst, int const nbr_in, int const nbr_out,
            int const stride, const r2hc_precomp_t& transform,
            double* const transformed, double* const scratch)
        {
            switch (family) {
                case native_spectral_family::cheb:
                    return execute_persistent_cheb_inverse<
                        TransformSize, native_spectral_family::cheb>(
                        src, dst, nbr_in, nbr_out, stride, transform,
                        transformed, scratch);
                case native_spectral_family::cheb_even:
                    return execute_persistent_cheb_inverse<
                        TransformSize, native_spectral_family::cheb_even>(
                        src, dst, nbr_in, nbr_out, stride, transform,
                        transformed, scratch);
                case native_spectral_family::cheb_odd:
                    return execute_persistent_cheb_inverse<
                        TransformSize, native_spectral_family::cheb_odd>(
                        src, dst, nbr_in, nbr_out, stride, transform,
                        transformed, scratch);
                case native_spectral_family::cossin:
                    return execute_persistent_cossin_inverse<TransformSize>(
                        src, dst, nbr_in, nbr_out, stride, transformed);
                case native_spectral_family::cos:
                case native_spectral_family::cos_even:
                    return execute_persistent_cos_inverse<TransformSize>(
                        src, dst, nbr_in, nbr_out, stride, transform,
                        transformed, scratch);
                case native_spectral_family::sin:
                case native_spectral_family::sin_even:
                    return execute_persistent_sin_inverse<TransformSize>(
                        src, dst, nbr_in, nbr_out, stride, transform,
                        transformed);
                case native_spectral_family::cos_odd:
                    return execute_persistent_odd_inverse<
                        TransformSize, native_spectral_family::cos_odd>(
                        src, dst, nbr_in, nbr_out, stride, transform,
                        transformed, scratch);
                case native_spectral_family::sin_odd:
                    return execute_persistent_odd_inverse<
                        TransformSize, native_spectral_family::sin_odd>(
                        src, dst, nbr_in, nbr_out, stride, transform,
                        transformed, scratch);
                default: return false;
            }
        }
#endif

        template <int TransformSize>
        bool dispatch_fused_forward(native_spectral_family const family,
                                    const double* const src, double* const dst,
                                    int const nbr_in, int const nbr_out, int const stride,
                                    const r2hc_precomp_t& transform)
        {
            switch (family) {
                case native_spectral_family::cheb:
                    return execute_fused_cheb_forward<TransformSize,
                                                      native_spectral_family::cheb>(
                        src, dst, nbr_in, nbr_out, stride, transform);
                case native_spectral_family::cheb_even:
                    return execute_fused_cheb_forward<TransformSize,
                                                      native_spectral_family::cheb_even>(
                        src, dst, nbr_in, nbr_out, stride, transform);
                case native_spectral_family::cheb_odd:
                    return execute_fused_cheb_forward<TransformSize,
                                                      native_spectral_family::cheb_odd>(
                        src, dst, nbr_in, nbr_out, stride, transform);
                case native_spectral_family::cossin:
                    return execute_fused_cossin_forward<TransformSize>(
                        src, dst, nbr_in, nbr_out, stride);
                case native_spectral_family::cos:
                    return execute_fused_cos_forward<TransformSize>(
                        src, dst, nbr_in, nbr_out, stride, transform);
                case native_spectral_family::sin:
                    return execute_fused_sin_forward<TransformSize>(
                        src, dst, nbr_in, nbr_out, stride, transform);
                case native_spectral_family::cos_even:
                    return execute_fused_cos_even_forward<TransformSize>(
                        src, dst, nbr_in, nbr_out, stride, transform);
                case native_spectral_family::cos_odd:
                    return execute_fused_cos_odd_forward<TransformSize>(
                        src, dst, nbr_in, nbr_out, stride, transform);
                case native_spectral_family::sin_even:
                    return execute_fused_sin_even_forward<TransformSize>(
                        src, dst, nbr_in, nbr_out, stride, transform);
                case native_spectral_family::sin_odd:
                    return execute_fused_sin_odd_forward<TransformSize>(
                        src, dst, nbr_in, nbr_out, stride, transform);
                default: return false;
            }
        }

        template <int TransformSize>
        bool dispatch_fused_inverse(native_spectral_family const family,
                                    const double* const src, double* const dst,
                                    int const nbr_in, int const nbr_out, int const stride,
                                    const r2hc_precomp_t& transform)
        {
            switch (family) {
                case native_spectral_family::cheb:
                    return execute_fused_cheb_inverse<TransformSize,
                                                      native_spectral_family::cheb>(
                        src, dst, nbr_in, nbr_out, stride, transform);
                case native_spectral_family::cheb_even:
                    return execute_fused_cheb_inverse<TransformSize,
                                                      native_spectral_family::cheb_even>(
                        src, dst, nbr_in, nbr_out, stride, transform);
                case native_spectral_family::cheb_odd:
                    return execute_fused_cheb_inverse<TransformSize,
                                                      native_spectral_family::cheb_odd>(
                        src, dst, nbr_in, nbr_out, stride, transform);
                case native_spectral_family::cossin:
                    return execute_fused_cossin_inverse<TransformSize>(
                        src, dst, nbr_in, nbr_out, stride);
                case native_spectral_family::cos:
                    return execute_fused_cos_inverse<TransformSize>(
                        src, dst, nbr_in, nbr_out, stride, transform);
                case native_spectral_family::sin:
                    return execute_fused_sin_inverse<TransformSize>(
                        src, dst, nbr_in, nbr_out, stride, transform);
                case native_spectral_family::cos_even:
                    return execute_fused_cos_even_inverse<TransformSize>(
                        src, dst, nbr_in, nbr_out, stride, transform);
                case native_spectral_family::cos_odd:
                    return execute_fused_cos_odd_inverse<TransformSize>(
                        src, dst, nbr_in, nbr_out, stride, transform);
                case native_spectral_family::sin_even:
                    return execute_fused_sin_even_inverse<TransformSize>(
                        src, dst, nbr_in, nbr_out, stride, transform);
                case native_spectral_family::sin_odd:
                    return execute_fused_sin_odd_inverse<TransformSize>(
                        src, dst, nbr_in, nbr_out, stride, transform);
                default: return false;
            }
        }

#define CELEPHAIS_EVEN_TRANSFORM_CASE(N)                                                       \
    case N:                                                                                    \
        if constexpr (Inverse)                                                                 \
            execute_hc2r_codelet<N>(buffer);                                                   \
        else                                                                                   \
            execute_r2hc_codelet<N>(buffer);                                                   \
        return

        template <bool Inverse, typename Generic>
        void dispatch_even_hot_size(int const n, double* const buffer, Generic&& generic)
        {
            switch (n) {
                CELEPHAIS_EVEN_TRANSFORM_CASE(2);
                CELEPHAIS_EVEN_TRANSFORM_CASE(4);
                CELEPHAIS_EVEN_TRANSFORM_CASE(6);
                CELEPHAIS_EVEN_TRANSFORM_CASE(8);
                CELEPHAIS_EVEN_TRANSFORM_CASE(10);
                CELEPHAIS_EVEN_TRANSFORM_CASE(12);
                CELEPHAIS_EVEN_TRANSFORM_CASE(14);
                CELEPHAIS_EVEN_TRANSFORM_CASE(16);
                CELEPHAIS_EVEN_TRANSFORM_CASE(18);
                CELEPHAIS_EVEN_TRANSFORM_CASE(20);
                CELEPHAIS_EVEN_TRANSFORM_CASE(22);
                CELEPHAIS_EVEN_TRANSFORM_CASE(24);
                CELEPHAIS_EVEN_TRANSFORM_CASE(26);
                CELEPHAIS_EVEN_TRANSFORM_CASE(28);
                CELEPHAIS_EVEN_TRANSFORM_CASE(30);
                CELEPHAIS_EVEN_TRANSFORM_CASE(32);
                default: generic(); return;
            }
        }

#undef CELEPHAIS_EVEN_TRANSFORM_CASE
    } // namespace

    int r2hc_precomp_t::checked_transform_size(int const n)
    {
        if (n < 2 || n > 32 || n % 2 != 0)
            throw std::invalid_argument("Transform size must be even and in [2, 32]");
        return n;
    }

    int r2hc_precomp_t::checked_backend_transform_size(
        int const n, r2hc_backend const backend)
    {
        const int checked_size = checked_transform_size(n);
#ifndef CELEPHAIS_ENABLE_FFTW_ORACLE
        if (backend == r2hc_backend::fftw)
            throw std::invalid_argument(
                "The FFTW backend is unavailable in this production build; use "
                "CELEPHAIS_FFT_BACKEND=native or leave CELEPHAIS_FFT_BACKEND unset");
#endif
        return checked_size;
    }

    void r2hc_precomp_t::aligned_buffer_deleter::operator()(double* const pointer) const noexcept
    {
        ::operator delete[](pointer, std::align_val_t{64});
    }

    r2hc_precomp_t::aligned_buffer r2hc_precomp_t::allocate_aligned_buffer(int const n)
    {
        auto* const pointer = static_cast<double*>(
            ::operator new[](static_cast<std::size_t>(n) * sizeof(double),
                             std::align_val_t{64}));
        return aligned_buffer(pointer);
    }

    r2hc_backend r2hc_precomp_t::configured_backend()
    {
        const char* const value = std::getenv("CELEPHAIS_FFT_BACKEND");
        if (value == nullptr || value[0] == '\0' || std::strcmp(value, "native") == 0)
            return r2hc_backend::native;
        if (std::strcmp(value, "fftw") == 0) {
#ifdef CELEPHAIS_ENABLE_FFTW_ORACLE
            return r2hc_backend::fftw;
#else
            throw std::invalid_argument(
                "CELEPHAIS_FFT_BACKEND=fftw requested a test-only oracle unavailable "
                "in this production build; use 'native' or leave it unset");
#endif
        }
        throw std::invalid_argument(
            "CELEPHAIS_FFT_BACKEND must be 'native' or the enabled test-only 'fftw' oracle");
    }

    bool r2hc_precomp_t::configured_fused_spectral()
    {
        const char* const value = std::getenv("CELEPHAIS_NATIVE_FUSED");
        if (value == nullptr || value[0] == '\0' || std::strcmp(value, "1") == 0)
            return true;
        if (std::strcmp(value, "0") == 0)
            return false;
        throw std::invalid_argument("CELEPHAIS_NATIVE_FUSED must be either '0' or '1'");
    }

    r2hc_precomp_t::r2hc_precomp_t(int const n, r2hc_direction const direction)
        : r2hc_precomp_t(n, direction, configured_backend())
    {
    }

    r2hc_precomp_t::r2hc_precomp_t(int const n, r2hc_direction const direction,
                                   r2hc_backend const backend)
        : buffer_storage_(allocate_aligned_buffer(checked_backend_transform_size(n, backend))),
#if defined(__linux__) && defined(__x86_64__) && defined(__GNUC__) \
    && !defined(__clang__)
          scratch_(backend == r2hc_backend::native
                       ? static_cast<std::size_t>(n + 1)
                       : std::size_t{}),
#else
          scratch_(backend == r2hc_backend::native ? static_cast<std::size_t>(n)
                                                   : std::size_t{}),
#endif
          cos_table_(backend == r2hc_backend::native
                         ? static_cast<std::size_t>(n / 2 + 1) * static_cast<std::size_t>(n)
                         : std::size_t{}),
          sin_table_(backend == r2hc_backend::native
                         ? static_cast<std::size_t>(n / 2 + 1) * static_cast<std::size_t>(n)
                         : std::size_t{}),
          oracle_plan_(nullptr), backend_(backend), direction_(direction),
          fused_spectral_enabled_(backend == r2hc_backend::native
                                      ? configured_fused_spectral()
                                      : false),
          transform_size(n), sin_pi_over_n(static_cast<std::size_t>(n + 1)),
          sin_pi_i_over_2n(static_cast<std::size_t>(n + 1)),
          sin_half_pi_i_over_n(static_cast<std::size_t>(n + 1)),
          sin_pi_quarter(std::sin(std::numbers::pi_v<double> * n / 4 / n)),
          buffer(buffer_storage_.get())
    {
        for (int i = 0; i <= transform_size; ++i) {
            sin_pi_over_n[static_cast<std::size_t>(i)] =
                std::sin(std::numbers::pi_v<double> * i / n);
            sin_pi_i_over_2n[static_cast<std::size_t>(i)] =
                std::sin(std::numbers::pi_v<double> * i / 2. / n);
            sin_half_pi_i_over_n[static_cast<std::size_t>(i)] =
                std::sin(std::numbers::pi_v<double> / 2. * i / n);
        }

#ifdef CELEPHAIS_ENABLE_FFTW_ORACLE
        if (backend_ == r2hc_backend::fftw) {
            const fftw_r2r_kind transform = direction_ == r2hc_direction::forward
                                                ? FFTW_R2HC
                                                : FFTW_HC2R;
            oracle_plan_ = fftw_plan_r2r_1d(
                transform_size, buffer, buffer, transform, FFTW_MEASURE);
            if (oracle_plan_ == nullptr)
                throw std::runtime_error("FFTW could not create the requested test-oracle plan");
        }
        else
#endif
        {
            for (int k = 0; k <= transform_size / 2; ++k) {
                for (int j = 0; j < transform_size; ++j) {
                    const double angle = 2. * std::numbers::pi_v<double> * k * j / transform_size;
                    const auto index = static_cast<std::size_t>(k) * transform_size + j;
                    cos_table_[index] = std::cos(angle);
                    sin_table_[index] = std::sin(angle);
                }
            }
        }
    }

    r2hc_precomp_t::~r2hc_precomp_t()
    {
#ifdef CELEPHAIS_ENABLE_FFTW_ORACLE
        if (oracle_plan_ != nullptr)
            fftw_destroy_plan(static_cast<fftw_plan>(oracle_plan_));
#endif
    }

    r2hc_backend r2hc_precomp_t::backend() const noexcept
    {
        return backend_;
    }

#if defined(__linux__) && defined(__x86_64__) && defined(__GNUC__) \
    && !defined(__clang__)
    bool r2hc_precomp_t::execute_cached_cossin_forward(
        const double* const src, double* const dst, int const nbr_in,
        int const nbr_out, int const stride) const
    {
        switch (transform_size) {
            case 10:
                if (src == buffer)
                    return execute_fused_cossin_forward<10>(
                        src, dst, nbr_in, nbr_out, stride);
                return execute_persistent_cossin_forward<10>(
                    src, dst, nbr_in, nbr_out, stride, buffer);
            case 12:
                if (src == buffer)
                    return execute_fused_cossin_forward<12>(
                        src, dst, nbr_in, nbr_out, stride);
                return execute_persistent_cossin_forward<12>(
                    src, dst, nbr_in, nbr_out, stride, buffer);
            case 14:
                if (src == buffer)
                    return execute_fused_cossin_forward<14>(
                        src, dst, nbr_in, nbr_out, stride);
                return execute_persistent_cossin_forward<14>(
                    src, dst, nbr_in, nbr_out, stride, buffer);
            case 16:
                if (src == buffer)
                    return execute_fused_cossin_forward<16>(
                        src, dst, nbr_in, nbr_out, stride);
                return execute_persistent_cossin_forward<16>(
                    src, dst, nbr_in, nbr_out, stride, buffer);
            default: return false;
        }
    }
#endif

    bool r2hc_precomp_t::execute_fused_forward(
        native_spectral_family const family, const double* const src, double* const dst,
        int const nbr_in, int const nbr_out, int const stride) const
    {
        switch (transform_size) {
            case 6: return dispatch_fused_forward<6>(family, src, dst, nbr_in, nbr_out,
                                                      stride, *this);
            case 8: return dispatch_fused_forward<8>(family, src, dst, nbr_in, nbr_out,
                                                      stride, *this);
#if defined(__linux__) && defined(__x86_64__) && defined(__GNUC__) \
    && !defined(__clang__)
            case 10: return dispatch_persistent_forward<10>(
                         family, src, dst, nbr_in, nbr_out, stride, *this,
                         buffer, scratch_.data());
            case 12: return dispatch_persistent_forward<12>(
                         family, src, dst, nbr_in, nbr_out, stride, *this,
                         buffer, scratch_.data());
            case 14: return dispatch_persistent_forward<14>(
                         family, src, dst, nbr_in, nbr_out, stride, *this,
                         buffer, scratch_.data());
            case 16: return dispatch_persistent_forward<16>(
                         family, src, dst, nbr_in, nbr_out, stride, *this,
                         buffer, scratch_.data());
#else
            case 10: return dispatch_fused_forward<10>(family, src, dst, nbr_in, nbr_out,
                                                        stride, *this);
            case 12: return dispatch_fused_forward<12>(family, src, dst, nbr_in, nbr_out,
                                                        stride, *this);
            case 14: return dispatch_fused_forward<14>(family, src, dst, nbr_in, nbr_out,
                                                        stride, *this);
            case 16: return dispatch_fused_forward<16>(family, src, dst, nbr_in, nbr_out,
                                                        stride, *this);
#endif
            case 18: return dispatch_fused_forward<18>(family, src, dst, nbr_in, nbr_out,
                                                        stride, *this);
            case 20: return dispatch_fused_forward<20>(family, src, dst, nbr_in, nbr_out,
                                                        stride, *this);
            default: return false;
        }
    }

    bool r2hc_precomp_t::execute_fused_inverse(
        native_spectral_family const family, const double* const src, double* const dst,
        int const nbr_in, int const nbr_out, int const stride) const
    {
        switch (transform_size) {
            case 6: return dispatch_fused_inverse<6>(family, src, dst, nbr_in, nbr_out,
                                                      stride, *this);
            case 8: return dispatch_fused_inverse<8>(family, src, dst, nbr_in, nbr_out,
                                                      stride, *this);
#if defined(__linux__) && defined(__x86_64__) && defined(__GNUC__) \
    && !defined(__clang__)
            case 10: return dispatch_persistent_inverse<10>(
                         family, src, dst, nbr_in, nbr_out, stride, *this,
                         buffer, scratch_.data());
            case 12: return dispatch_persistent_inverse<12>(
                         family, src, dst, nbr_in, nbr_out, stride, *this,
                         buffer, scratch_.data());
            case 14: return dispatch_persistent_inverse<14>(
                         family, src, dst, nbr_in, nbr_out, stride, *this,
                         buffer, scratch_.data());
            case 16: return dispatch_persistent_inverse<16>(
                         family, src, dst, nbr_in, nbr_out, stride, *this,
                         buffer, scratch_.data());
#else
            case 10: return dispatch_fused_inverse<10>(family, src, dst, nbr_in, nbr_out,
                                                        stride, *this);
            case 12: return dispatch_fused_inverse<12>(family, src, dst, nbr_in, nbr_out,
                                                        stride, *this);
            case 14: return dispatch_fused_inverse<14>(family, src, dst, nbr_in, nbr_out,
                                                        stride, *this);
            case 16: return dispatch_fused_inverse<16>(family, src, dst, nbr_in, nbr_out,
                                                        stride, *this);
#endif
            case 18: return dispatch_fused_inverse<18>(family, src, dst, nbr_in, nbr_out,
                                                        stride, *this);
            case 20: return dispatch_fused_inverse<20>(family, src, dst, nbr_in, nbr_out,
                                                        stride, *this);
            default: return false;
        }
    }

    void r2hc_precomp_t::execute_r2hc_generic()
    {
        const int n = transform_size;
        const int half = n / 2;

        double dc = buffer[0];
        for (int j = 1; j < n; ++j)
            dc += buffer[j];
        scratch_[0] = dc;

        const int paired_modes = (n - 1) / 2;
        for (int k = 1; k <= paired_modes; ++k) {
            const double* const cos_row = cos_table_.data() + k * n;
            const double* const sin_row = sin_table_.data() + k * n;
            double real = buffer[0];
            if (n % 2 == 0)
                real += (k & 1) ? -buffer[half] : buffer[half];
            double imag = 0.;
            for (int j = 1; j <= (n - 1) / 2; ++j) {
                real += (buffer[j] + buffer[n - j]) * cos_row[j];
                imag += (buffer[n - j] - buffer[j]) * sin_row[j];
            }
            scratch_[static_cast<std::size_t>(k)] = real;
            scratch_[static_cast<std::size_t>(n - k)] = imag;
        }

        if (n % 2 == 0) {
            double nyquist = buffer[0];
            for (int j = 1; j < n; ++j)
                nyquist += (j & 1) ? -buffer[j] : buffer[j];
            scratch_[static_cast<std::size_t>(half)] = nyquist;
        }

        for (int j = 0; j < n; ++j)
            buffer[j] = scratch_[static_cast<std::size_t>(j)];
    }

    void r2hc_precomp_t::execute_hc2r_generic()
    {
        const int n = transform_size;
        const int paired_modes = (n - 1) / 2;

        for (int j = 0; j < n; ++j) {
            double value = buffer[0];
            if (n % 2 == 0)
                value += (j & 1) ? -buffer[n / 2] : buffer[n / 2];
            for (int k = 1; k <= paired_modes; ++k) {
                value += 2. * (buffer[k] * cos_table_[static_cast<std::size_t>(k) * n + j]
                               - buffer[n - k] * sin_table_[static_cast<std::size_t>(k) * n + j]);
            }
            scratch_[static_cast<std::size_t>(j)] = value;
        }

        for (int j = 0; j < n; ++j)
            buffer[j] = scratch_[static_cast<std::size_t>(j)];
    }

    void r2hc_precomp_t::execute_r2hc()
    {
        if (direction_ != r2hc_direction::forward)
            throw std::logic_error("Inverse transform workspace cannot execute R2HC");
#ifdef CELEPHAIS_ENABLE_FFTW_ORACLE
        if (backend_ == r2hc_backend::fftw) {
            fftw_execute(static_cast<fftw_plan>(oracle_plan_));
            return;
        }
#endif
        dispatch_even_hot_size<false>(transform_size, buffer,
                                      [this] { execute_r2hc_generic(); });
    }

    void r2hc_precomp_t::execute_hc2r()
    {
        if (direction_ != r2hc_direction::inverse)
            throw std::logic_error("Forward transform workspace cannot execute HC2R");
#ifdef CELEPHAIS_ENABLE_FFTW_ORACLE
        if (backend_ == r2hc_backend::fftw) {
            fftw_execute(static_cast<fftw_plan>(oracle_plan_));
            return;
        }
#endif
        dispatch_even_hot_size<true>(transform_size, buffer,
                                     [this] { execute_hc2r_generic(); });
    }

} // namespace Kadath
