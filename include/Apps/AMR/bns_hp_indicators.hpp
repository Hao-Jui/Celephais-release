#pragma once

#include "For_Kadath/Scalar/scalar.hpp"
#include "For_Kadath/Tensor/tensor.hpp"
#include "For_Kadath/Val_domain/val_domain.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace Kadath {
namespace bns_hp {

// Which coordinate axes' highest modes form the "tail" when judging spectral
// convergence. Radial = axis 0, Theta = axis 1, Phi = axis 2 -- each refinable
// independently per domain now that non-conforming seams are coupled with
// coefficient-space (import) matching. AngularOnly = axes 1.. (theta+phi
// together, the legacy combined angular tail); All = every axis (the plain
// coefficient-decay test). Per-direction AMR judges RadialOnly / ThetaOnly /
// PhiOnly separately against the same thresholds.
enum class TailAxisSelection : std::uint8_t { All, RadialOnly, AngularOnly, ThetaOnly, PhiOnly };

struct SpectralTailOptions {
    int tail_width = 2;
    double norm_floor = 0.0;
    // Which axes contribute the tail modes. Default All keeps the plain decay
    // test; callers narrow to RadialOnly (the refine trigger, which can only add
    // radial points) or AngularOnly (the polar-resolution diagnostic).
    TailAxisSelection axes = TailAxisSelection::All;
};

struct SpectralTailRatio {
    int domain = -1;
    double l2_ratio = 0.0;
    double linf_ratio = 0.0;
    double tail_l2 = 0.0;
    double total_l2 = 0.0;
    double tail_linf = 0.0;
    double total_linf = 0.0;
    std::size_t tail_modes = 0;
    std::size_t total_modes = 0;
};

namespace detail {

struct TailAccumulator {
    double tail_l2_sq = 0.0;
    double total_l2_sq = 0.0;
    double tail_linf = 0.0;
    double total_linf = 0.0;
    std::size_t tail_modes = 0;
    std::size_t total_modes = 0;

    void add(double coefficient, bool is_tail_mode)
    {
        const double magnitude = std::abs(coefficient);
        total_l2_sq += magnitude * magnitude;
        total_linf = std::max(total_linf, magnitude);
        ++total_modes;

        if (is_tail_mode) {
            tail_l2_sq += magnitude * magnitude;
            tail_linf = std::max(tail_linf, magnitude);
            ++tail_modes;
        }
    }

    void merge(const TailAccumulator& other)
    {
        tail_l2_sq += other.tail_l2_sq;
        total_l2_sq += other.total_l2_sq;
        tail_linf = std::max(tail_linf, other.tail_linf);
        total_linf = std::max(total_linf, other.total_linf);
        tail_modes += other.tail_modes;
        total_modes += other.total_modes;
    }

    SpectralTailRatio finish(int domain, double norm_floor) const
    {
        SpectralTailRatio ratio;
        ratio.domain = domain;
        ratio.tail_l2 = std::sqrt(tail_l2_sq);
        ratio.total_l2 = std::sqrt(total_l2_sq);
        ratio.tail_linf = tail_linf;
        ratio.total_linf = total_linf;
        ratio.tail_modes = tail_modes;
        ratio.total_modes = total_modes;

        if (ratio.total_l2 > norm_floor)
            ratio.l2_ratio = ratio.tail_l2 / ratio.total_l2;
        if (ratio.total_linf > norm_floor)
            ratio.linf_ratio = ratio.tail_linf / ratio.total_linf;
        return ratio;
    }
};

inline bool is_tail_mode(const Index& position, const Dim_array& dimensions, int tail_width,
                         TailAxisSelection axes = TailAxisSelection::All)
{
    if (tail_width <= 0)
        return false;

    const int ndim = dimensions.get_ndim();
    int first_axis = 0;
    int last_axis = ndim;
    switch (axes) {
        case TailAxisSelection::RadialOnly:  // axis 0
            last_axis = std::min(1, ndim);
            break;
        case TailAxisSelection::AngularOnly:  // axes 1.. (theta + phi)
            first_axis = std::min(1, ndim);
            break;
        case TailAxisSelection::ThetaOnly:  // axis 1
            first_axis = std::min(1, ndim);
            last_axis = std::min(2, ndim);
            break;
        case TailAxisSelection::PhiOnly:  // axis 2
            first_axis = std::min(2, ndim);
            last_axis = std::min(3, ndim);
            break;
        case TailAxisSelection::All:
            break;
    }
    for (int axis = first_axis; axis < last_axis; ++axis) {
        const int first_tail_mode = std::max(0, dimensions(axis) - tail_width);
        if (position(axis) >= first_tail_mode)
            return true;
    }
    return false;
}

inline TailAccumulator accumulate_tail(const Val_domain& value, const SpectralTailOptions& options)
{
    value.coef();

    TailAccumulator accumulator;
    const Dim_array dimensions = value.get_domain()->get_nbr_coefs();
    Index position(dimensions);
    do {
        accumulator.add(value.get_coef(position),
                        is_tail_mode(position, dimensions, options.tail_width, options.axes));
    } while (position.inc());
    return accumulator;
}

// Robust azimuthal (phi, axis 2) tail estimator. The phi basis is a cos/sin
// Fourier expansion, so (i) the raw coefficient index alternates cos/sin with
// structural zeros (sin of the constant mode, the Nyquist sin), and (ii) the
// highest retained azimuthal mode picks up ALIASED high-m energy (a single
// anomalously large top coefficient). Reading the literal top-w coefficients
// therefore over-reports the truncation error and is non-monotone under
// refinement. This estimator instead (1) FOLDS each cos/sin pair into a per-
// azimuthal-number amplitude a_m = sqrt(cos_m^2 + sin_m^2) (killing the
// structural zeros and giving a monotone envelope) and (2) takes the MIN over
// the top tail_width+1 folded bins as the tail magnitude (an envelope floor that
// ignores the aliased spike). It accumulates the per-mode marginal across tensor
// components, so the scalar and aggregate paths share one structure.
// NOTE: the fold + envelope-floor yields a SYSTEMATICALLY LOWER ratio than the
// legacy literal top-w tail (it removes the aliased-spike inflation), judged
// against the same l2/linf thresholds. The PhiOnly direction is therefore
// intentionally more conservative about marking than radial/polar; recalibrate
// the thresholds if a different azimuthal sensitivity is wanted.
struct PhiMarginal {
    std::vector<double> energy_by_mode;  ///< per phi-mode sum of squared coefficients
    double total_l2_sq = 0.0;

    void add_coefficient(double coefficient, int phi_index, int nphi)
    {
        if (energy_by_mode.empty())
            energy_by_mode.assign(static_cast<std::size_t>(nphi), 0.0);
        const double energy = coefficient * coefficient;
        energy_by_mode[static_cast<std::size_t>(phi_index)] += energy;
        total_l2_sq += energy;
    }

    void add(const Val_domain& value)
    {
        value.coef();
        const Dim_array dims = value.get_domain()->get_nbr_coefs();
        if (dims.get_ndim() < 3)
            return;  // no azimuthal axis (e.g. a 2D domain)
        const int nphi = dims(2);
        Index position(dims);
        do {
            add_coefficient(value.get_coef(position), position(2), nphi);
        } while (position.inc());
    }

    SpectralTailRatio finish(int domain, int tail_width, double norm_floor) const
    {
        SpectralTailRatio ratio;
        ratio.domain = domain;
        const int nphi = static_cast<int>(energy_by_mode.size());
        if (nphi == 0 || tail_width <= 0)
            return ratio;  // no phi axis -> zero ratio (treated as resolved)

        // Fold cos/sin pairs (k, k+1) into per-azimuthal-number amplitudes.
        std::vector<double> amp;
        amp.reserve(static_cast<std::size_t>(nphi) / 2 + 1);
        for (int k = 0; k + 1 < nphi; k += 2)
            amp.push_back(std::sqrt(energy_by_mode[static_cast<std::size_t>(k)] +
                                    energy_by_mode[static_cast<std::size_t>(k) + 1]));
        if (nphi % 2 == 1)
            amp.push_back(std::sqrt(energy_by_mode[static_cast<std::size_t>(nphi) - 1]));

        // Envelope-floor tail: MIN over the top tail_width+1 folded bins. The
        // aliased top mode is a spike (a maximum), so the min skips it; when the
        // spectrum decays cleanly the top bin IS the min, recovering the usual
        // tail magnitude.
        const int k = std::min(tail_width + 1, static_cast<int>(amp.size()));
        double tail = amp.back();
        for (int j = static_cast<int>(amp.size()) - k; j < static_cast<int>(amp.size()); ++j)
            tail = std::min(tail, amp[static_cast<std::size_t>(j)]);

        // Both ratios stay in the FOLDED amplitude space: l2 against the total
        // folded energy, linf against the peak folded amplitude. (Using the raw
        // per-coefficient max for the linf denominator would mix the folded
        // numerator with an unfolded denominator and could exceed 1.)
        const double total_l2 = std::sqrt(total_l2_sq);
        const double amp_peak = *std::max_element(amp.begin(), amp.end());
        ratio.tail_l2 = tail;
        ratio.total_l2 = total_l2;
        ratio.tail_linf = tail;
        ratio.total_linf = amp_peak;
        ratio.tail_modes = static_cast<std::size_t>(k);
        ratio.total_modes = amp.size();
        if (total_l2 > norm_floor)
            ratio.l2_ratio = tail / total_l2;
        if (amp_peak > norm_floor)
            ratio.linf_ratio = tail / amp_peak;
        return ratio;
    }
};

using AxisTailAccumulators = std::array<TailAccumulator, 2>;

inline std::array<SpectralTailRatio, 3> accumulate_axis_tails(
    int domain, const Val_domain& value, const SpectralTailOptions& options)
{
    value.coef();
    const Dim_array dimensions = value.get_domain()->get_nbr_coefs();
    AxisTailAccumulators literal;
    PhiMarginal phi;
    const int radial_tail = std::max(0, dimensions(0) - options.tail_width);
    const int theta_tail = dimensions.get_ndim() > 1
                               ? std::max(0, dimensions(1) - options.tail_width)
                               : dimensions(0);
    const int nphi = dimensions.get_ndim() > 2 ? dimensions(2) : 0;

    Index position(dimensions);
    do {
        const double coefficient = value.get_coef(position);
        literal[0].add(coefficient, options.tail_width > 0 && position(0) >= radial_tail);
        literal[1].add(coefficient, options.tail_width > 0 && dimensions.get_ndim() > 1 &&
                                        position(1) >= theta_tail);
        if (nphi > 0)
            phi.add_coefficient(coefficient, position(2), nphi);
    } while (position.inc());

    return {literal[0].finish(domain, options.norm_floor),
            literal[1].finish(domain, options.norm_floor),
            phi.finish(domain, options.tail_width, options.norm_floor)};
}

} // namespace detail

inline SpectralTailRatio spectral_tail_ratio(
    int domain,
    const Val_domain& value,
    const SpectralTailOptions& options = SpectralTailOptions())
{
    // The azimuthal (Fourier) axis uses the fold + envelope-floor estimator,
    // robust to the cos/sin structural zeros and the un-aliasing spike at the
    // top mode; the radial/polar axes keep the literal top-w tail.
    if (options.axes == TailAxisSelection::PhiOnly) {
        detail::PhiMarginal marginal;
        marginal.add(value);
        return marginal.finish(domain, options.tail_width, options.norm_floor);
    }
    return detail::accumulate_tail(value, options).finish(domain, options.norm_floor);
}

inline SpectralTailRatio spectral_tail_ratio(
    const Val_domain& value,
    const SpectralTailOptions& options = SpectralTailOptions())
{
    return spectral_tail_ratio(-1, value, options);
}

inline std::vector<SpectralTailRatio> scalar_spectral_tail_ratios(
    const Scalar& field,
    const SpectralTailOptions& options = SpectralTailOptions())
{
    std::vector<SpectralTailRatio> ratios;
    ratios.reserve(static_cast<std::size_t>(field.get_nbr_domains()));
    for (int domain = 0; domain < field.get_nbr_domains(); ++domain)
        ratios.push_back(spectral_tail_ratio(domain, field(domain), options));
    return ratios;
}

using SpectralAxisRatioVectors = std::array<std::vector<SpectralTailRatio>, 3>;

inline SpectralAxisRatioVectors scalar_spectral_axis_tail_ratios(
    const Scalar& field, const SpectralTailOptions& options = SpectralTailOptions())
{
    SpectralAxisRatioVectors ratios;
    for (auto& axis : ratios)
        axis.reserve(static_cast<std::size_t>(field.get_nbr_domains()));
    for (int domain = 0; domain < field.get_nbr_domains(); ++domain) {
        const auto domain_ratios = detail::accumulate_axis_tails(domain, field(domain), options);
        for (std::size_t axis = 0; axis < ratios.size(); ++axis)
            ratios[axis].push_back(domain_ratios[axis]);
    }
    return ratios;
}

inline std::vector<SpectralTailRatio> tensor_component_spectral_tail_ratios(
    const Tensor& field,
    int component,
    const SpectralTailOptions& options = SpectralTailOptions())
{
    if (field.get_valence() == 0)
        return scalar_spectral_tail_ratios(field(), options);

    const Array<int> indices = field.indices(component);
    return scalar_spectral_tail_ratios(field(indices), options);
}

inline SpectralAxisRatioVectors tensor_component_spectral_axis_tail_ratios(
    const Tensor& field, int component,
    const SpectralTailOptions& options = SpectralTailOptions())
{
    if (field.get_valence() == 0)
        return scalar_spectral_axis_tail_ratios(field(), options);

    const Array<int> indices = field.indices(component);
    return scalar_spectral_axis_tail_ratios(field(indices), options);
}

inline std::vector<SpectralTailRatio> tensor_aggregate_spectral_tail_ratios(
    const Tensor& field,
    const SpectralTailOptions& options = SpectralTailOptions())
{
    if (field.get_valence() == 0)
        return scalar_spectral_tail_ratios(field(), options);

    const int domains = field.get_space().get_nbr_domains();

    // Azimuthal (Fourier) axis: aggregate the per-phi-mode marginal across
    // components, then fold + envelope-floor (matches the scalar PhiOnly path).
    if (options.axes == TailAxisSelection::PhiOnly) {
        std::vector<detail::PhiMarginal> marginals(static_cast<std::size_t>(domains));
        for (int component = 0; component < field.get_n_comp(); ++component) {
            const Array<int> indices = field.indices(component);
            const Scalar& component_field = field(indices);
            for (int domain = 0; domain < domains; ++domain)
                marginals[static_cast<std::size_t>(domain)].add(component_field(domain));
        }
        std::vector<SpectralTailRatio> ratios;
        ratios.reserve(static_cast<std::size_t>(domains));
        for (int domain = 0; domain < domains; ++domain)
            ratios.push_back(marginals[static_cast<std::size_t>(domain)].finish(
                domain, options.tail_width, options.norm_floor));
        return ratios;
    }

    std::vector<detail::TailAccumulator> accumulators(static_cast<std::size_t>(domains));

    for (int component = 0; component < field.get_n_comp(); ++component) {
        const Array<int> indices = field.indices(component);
        const Scalar& component_field = field(indices);
        for (int domain = 0; domain < domains; ++domain) {
            accumulators[static_cast<std::size_t>(domain)].merge(
                detail::accumulate_tail(component_field(domain), options));
        }
    }

    std::vector<SpectralTailRatio> ratios;
    ratios.reserve(static_cast<std::size_t>(domains));
    for (int domain = 0; domain < domains; ++domain) {
        ratios.push_back(accumulators[static_cast<std::size_t>(domain)].finish(
            domain, options.norm_floor));
    }
    return ratios;
}

} // namespace bns_hp
} // namespace Kadath
