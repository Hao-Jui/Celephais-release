/*
    Copyright 2026 Kadath contributors

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

#include <mpi.h>

#include <array>
#include <cstdint>
#include <filesystem>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace Kadath
{

    using CapturedLinearSystemHash = std::array<std::uint8_t, 32>;

    struct CapturedAnalysisSettings {
        static constexpr std::size_t mumps_version_bytes = 32;

        std::array<std::int32_t, 60> icntl{};
        std::array<double, 15> cntl{};
        std::array<char, mumps_version_bytes> mumps_version{};
        std::int32_t requested_ordering = 0;
        std::int32_t actual_ordering = 0;
        std::int32_t communicator_size = 1;
        std::int32_t analysis_rank_count = 1;
        std::int32_t factor_ranks_per_node = 0;
        std::int32_t successful_factor_icntl14 = 1;
        std::int32_t factor_retry_count = 0;
    };

    struct CapturedLinearSystemHashes {
        CapturedLinearSystemHash archive{};
        CapturedLinearSystemHash ordered_matrix{};
        CapturedLinearSystemHash rhs{};
        CapturedLinearSystemHash canonical_pattern{};
        CapturedLinearSystemHash canonical_values{};
    };

    // Stable archive codes, deliberately independent of the underlying values of
    // RowTaxonomy and ColumnClass.
    enum class CapturedRowTaxonomy : std::int32_t {
        Unknown = 0,
        Vol = 1,
        TauBc = 2,
        TauMatch = 3,
        GlobalInt = 4,
    };

    enum class CapturedColumnClass : std::int32_t {
        Unknown = 0,
        FieldUnknown = 1,
        FieldInterior = 2,
        FieldBoundary = 3,
        FieldInteriorVol = 4,
        FieldBoundaryTau = 5,
        FieldOuterShellTau = 6,
        FieldMatching = 7,
        FieldGauge = 8,
        VarDomain = 9,
        ScalarGlobal = 10,
    };

    struct CapturedRowTag {
        std::int32_t original_row = -1;
        CapturedRowTaxonomy taxonomy = CapturedRowTaxonomy::Unknown;
        std::int32_t domain = -1;
        std::int32_t domain_pair = -1;
    };

    struct CapturedColumnTag {
        std::int32_t original_column = -1;
        CapturedColumnClass column_class = CapturedColumnClass::Unknown;
        std::int32_t incidence_role = -1;
        std::int32_t domain = -1;
        std::int32_t term_idx = -1;
        std::int32_t var_idx = -1;
        std::int32_t var_double_idx = -1;
        std::int32_t vardom_param = -1;
        std::int32_t basis_mode = -1;
        std::uint64_t var_name_hash = 0;
        // Schema-v2 FIELD semantics. Schema-v1 readers leave every member at
        // the explicit unknown marker (-1).
        std::int32_t domain_type_id = -1;
        std::int32_t tensor_component = -1;
        std::int32_t coefficient_i = -1;
        std::int32_t coefficient_j = -1;
        std::int32_t coefficient_k = -1;
        std::int32_t coefficient_nr = -1;
        std::int32_t coefficient_nt = -1;
        std::int32_t coefficient_np = -1;
    };

    /// Non-owning input to the capture writer. COO indices and the captured MUMPS
    /// permutation are one-based. Full-index maps are zero-based.
    struct CapturedLinearSystemView {
        std::uint64_t rows = 0;
        std::uint64_t columns = 0;
        std::uint64_t full_rows = 0;
        std::uint64_t full_columns = 0;
        std::span<const int> row_indices_1based;
        std::span<const int> column_indices_1based;
        std::span<const double> values;
        std::span<const double> rhs;
        std::span<const std::uint32_t> row_full_indices_zero_based;
        std::span<const std::uint32_t> column_full_indices_zero_based;
        std::span<const CapturedRowTag> row_tags;
        std::span<const CapturedColumnTag> column_tags;
        std::span<const int> analysis_permutation_1based;
        CapturedAnalysisSettings analysis;
    };

    /// Owned replay form returned only after the complete file has passed all
    /// structural, finite-value, hash, truncation, and trailing-byte checks.
    struct CapturedLinearSystem {
        std::uint32_t schema_version = 0;
        std::uint64_t rows = 0;
        std::uint64_t columns = 0;
        std::uint64_t full_rows = 0;
        std::uint64_t full_columns = 0;
        std::vector<int> row_indices_1based;
        std::vector<int> column_indices_1based;
        std::vector<double> values;
        std::vector<double> rhs;
        std::vector<std::uint32_t> row_full_indices_zero_based;
        std::vector<std::uint32_t> column_full_indices_zero_based;
        std::vector<CapturedRowTag> row_tags;
        std::vector<CapturedColumnTag> column_tags;
        std::vector<int> analysis_permutation_1based;
        CapturedAnalysisSettings analysis;
        CapturedLinearSystemHashes hashes;
    };

    struct CapturedBackwardError {
        double residual_inf = 0.0;
        double matrix_inf = 0.0;
        double solution_inf = 0.0;
        double rhs_inf = 0.0;
        double denominator = 0.0;
        double value = 0.0;
        double componentwise_value = 0.0;
        std::uint64_t worst_factor_row_zero_based = 0;
        std::uint64_t worst_full_row_zero_based = 0;
    };

    CapturedLinearSystemHashes write_captured_linear_system(const std::filesystem::path& path,
                                                            const CapturedLinearSystemView& system);

    // Hash the ordered COO exactly as the archive does. Capture uses this both
    // before and after JOB=2 so a solver-side mutation of borrowed COO storage is
    // detected before an archive can be published.
    CapturedLinearSystemHash captured_ordered_matrix_hash(std::uint64_t rows, std::uint64_t columns,
                                                          std::span<const int> row_indices_1based,
                                                          std::span<const int> column_indices_1based,
                                                          std::span<const double> values);

    CapturedLinearSystem read_captured_linear_system(const std::filesystem::path& path);

    CapturedBackwardError scale_aware_backward_error(const CapturedLinearSystem& system,
                                                     std::span<const double> solution);

    std::string captured_linear_system_hash_hex(const CapturedLinearSystemHash& hash);

    // All ranks must make the same capture decision before any conditional MPI
    // phase. This exact collective gate refuses enable/path/ordinal/reuse
    // disagreement and the capture/analyze-reuse combination itself.
    void validate_captured_linear_system_collective_request(MPI_Comm communicator, std::string_view path, int ordinal,
                                                            bool sparse_analyze_reuse);

} // namespace Kadath
