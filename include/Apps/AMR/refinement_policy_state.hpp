#pragma once

#include <array>
#include <string>

namespace KadathApps::bns_amr
{
    struct FieldTailMaximum
    {
        std::string field{};
        int domain = -1;
        double l2_ratio = 0.0;
        double linf_ratio = 0.0;
        int axis = -1;
    };

    struct RefinementPolicyState
    {
        bool has_h_baseline = false;
        double previous_radial_demand = 0.0;
        std::array<double, 3> previous_radial_region_demand{{0.0, 0.0, 0.0}};
        std::array<char, 3> previous_h_regions{{0, 0, 0}};
        int previous_radial_flagged_domains = 0;
        bool h_closed = false;

        void record_h_baseline(double radial_demand,
                               int radial_flagged_domains,
                               const FieldTailMaximum&,
                               std::array<double, 3> radial_region_demand = {{0.0, 0.0, 0.0}},
                               std::array<char, 3> h_regions = {{0, 0, 0}})
        {
            has_h_baseline = true;
            previous_radial_demand = radial_demand;
            previous_radial_region_demand = radial_region_demand;
            previous_h_regions = h_regions;
            previous_radial_flagged_domains = radial_flagged_domains;
            h_closed = false;
        }

        void clear_h_baseline()
        {
            has_h_baseline = false;
            previous_radial_demand = 0.0;
            previous_radial_region_demand = {{0.0, 0.0, 0.0}};
            previous_h_regions = {{0, 0, 0}};
            previous_radial_flagged_domains = 0;
        }

        void close_h_gate()
        {
            clear_h_baseline();
            h_closed = true;
        }

        void reset_h_state()
        {
            clear_h_baseline();
            h_closed = false;
        }
    };
} // namespace KadathApps::bns_amr
