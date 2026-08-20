/*
    Added 2026 -- diagnostic only.

    y = 0 reflection-parity mass probe for the assembled Jacobian.  Default off;
    armed with JACOBIAN_PARITY_MASS=<report path>.  Reads the centralized
    rank-0 COO and writes the four sector masses, the structural 2-colouring
    check and the per-domain phi-column occupancy used to gate the parity
    reformulation (R1 / R2).
*/

#pragma once

#include "For_Kadath/System_of_eqs/system_of_eqs.hpp"

#include "Linear_algebra/jacobian_assembler.hpp"

#include <string>
#include <vector>

namespace Kadath
{
    namespace jacobian_parity_mass_detail
    {
        /// Compact row-domain x column-domain census. Bucket `domain_count`
        /// represents metadata domain -1 (global rows/columns).
        struct DomainCensus {
            int domain_count = 0;
            std::vector<long long> row_dimensions;
            std::vector<long long> column_dimensions;
            std::vector<long long> stored_coo_entries;
            std::vector<long long> matching_stored_coo_entries;
        };

        DomainCensus build_domain_census(
            int domain_count,
            const std::vector<RowMetadata>& rows,
            const std::vector<ColumnMetadata>& columns,
            const AssembledJacobianCoo& coo);
    } // namespace jacobian_parity_mass_detail

    /// Analyse \c coo for y -> -y block decoupling and write the report to
    /// \c report_path.  Rank-0 only; \c coo must be the centralized COO.
    void jacobian_parity_mass_report(System_of_eqs& system,
                                     const AssembledJacobianCoo& coo,
                                     const std::string& report_path);
} // namespace Kadath
