#include "For_Kadath/Domain/bispheric_nosym.hpp"
#include "For_Kadath/Domain/spheric_adapted_nosym.hpp"
#include "For_Kadath/Domain/spheric_nosym.hpp"

namespace Kadath
{
    namespace
    {
        constexpr double spherical_lookup_precision = 1e-12;
        constexpr double adapted_lookup_precision = 1e-9;
        constexpr double bispheric_lookup_precision = 1e-3;
    }

    bool Domain_nucleus_nosym::import_lanes_native(int numdom, int bound, int n_ope,
                                                   const Array<int>& zedoms, int lane_count,
                                                   Tensor* const* lane_parts, Tensor** lane_results) const
    {
        if (bound != OUTER_BC)
            return false;
        return import_lanes_point_major(numdom, bound, n_ope, zedoms, lane_count, lane_parts, lane_results,
                                        ImportLanePlanLayout::RadialBoundary, spherical_lookup_precision, true);
    }

    bool Domain_shell_nosym::import_lanes_native(int numdom, int bound, int n_ope,
                                                 const Array<int>& zedoms, int lane_count,
                                                 Tensor* const* lane_parts, Tensor** lane_results) const
    {
        return import_lanes_point_major(numdom, bound, n_ope, zedoms, lane_count, lane_parts, lane_results,
                                        ImportLanePlanLayout::RadialBoundary, spherical_lookup_precision, true);
    }

    bool Domain_compact_nosym::import_lanes_native(int numdom, int bound, int n_ope,
                                                   const Array<int>& zedoms, int lane_count,
                                                   Tensor* const* lane_parts, Tensor** lane_results) const
    {
        return import_lanes_point_major(numdom, bound, n_ope, zedoms, lane_count, lane_parts, lane_results,
                                        ImportLanePlanLayout::RadialBoundary, spherical_lookup_precision, true);
    }

    bool Domain_shell_inner_adapted_nosym::import_lanes_native(
        int numdom, int bound, int n_ope, const Array<int>& zedoms, int lane_count,
        Tensor* const* lane_parts, Tensor** lane_results) const
    {
        return import_lanes_point_major(numdom, bound, n_ope, zedoms, lane_count, lane_parts, lane_results,
                                        ImportLanePlanLayout::RadialBoundary, adapted_lookup_precision, true);
    }

    bool Domain_shell_outer_adapted_nosym::import_lanes_native(
        int numdom, int bound, int n_ope, const Array<int>& zedoms, int lane_count,
        Tensor* const* lane_parts, Tensor** lane_results) const
    {
        return import_lanes_point_major(numdom, bound, n_ope, zedoms, lane_count, lane_parts, lane_results,
                                        ImportLanePlanLayout::RadialBoundary, adapted_lookup_precision, true);
    }

    bool Domain_bispheric_chi_first_nosym::import_lanes_native(
        int numdom, int bound, int n_ope, const Array<int>& zedoms, int lane_count,
        Tensor* const* lane_parts, Tensor** lane_results) const
    {
        return import_lanes_point_major(numdom, bound, n_ope, zedoms, lane_count, lane_parts, lane_results,
                                        ImportLanePlanLayout::RadialBoundary, bispheric_lookup_precision, false);
    }

    bool Domain_bispheric_rect_nosym::import_lanes_native(
        int numdom, int bound, int n_ope, const Array<int>& zedoms, int lane_count,
        Tensor* const* lane_parts, Tensor** lane_results) const
    {
        if (bound == INNER_BC)
            return import_lanes_point_major(numdom, bound, n_ope, zedoms, lane_count, lane_parts, lane_results,
                                            ImportLanePlanLayout::RadialBoundary, spherical_lookup_precision, false);
        if (bound == OUTER_BC)
            return import_lanes_point_major(numdom, bound, n_ope, zedoms, lane_count, lane_parts, lane_results,
                                            ImportLanePlanLayout::BisphericRectOuter, spherical_lookup_precision,
                                            false);
        return false;
    }

    bool Domain_bispheric_eta_first_nosym::import_lanes_native(
        int numdom, int bound, int n_ope, const Array<int>& zedoms, int lane_count,
        Tensor* const* lane_parts, Tensor** lane_results) const
    {
        if (bound != OUTER_BC)
            return false;
        return import_lanes_point_major(numdom, bound, n_ope, zedoms, lane_count, lane_parts, lane_results,
                                        ImportLanePlanLayout::RadialBoundary, bispheric_lookup_precision, false);
    }
} // namespace Kadath
