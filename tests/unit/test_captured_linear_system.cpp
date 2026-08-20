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

#include <catch2/catch_approx.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <catch2/catch_test_macros.hpp>

#include "Linear_algebra/captured_linear_system.hpp"
#include "Linear_algebra/mumps_linear_solver.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unistd.h>

namespace
{

    void ensure_mpi_initialized()
    {
        struct MpiProcessLifetime {
            MpiProcessLifetime()
            {
                int initialized = 0;
                if (MPI_Initialized(&initialized) != MPI_SUCCESS)
                    throw std::runtime_error("MPI_Initialized failed");
                if (initialized == 0) {
                    int argc = 0;
                    char** argv = nullptr;
                    if (MPI_Init(&argc, &argv) != MPI_SUCCESS)
                        throw std::runtime_error("MPI_Init failed");
                }
            }

            ~MpiProcessLifetime()
            {
                int finalized = 0;
                if (MPI_Finalized(&finalized) == MPI_SUCCESS && finalized == 0)
                    MPI_Finalize();
            }
        };
        static MpiProcessLifetime lifetime;
        (void)lifetime;
    }

    using Kadath::CapturedAnalysisSettings;
    using Kadath::CapturedColumnClass;
    using Kadath::CapturedColumnTag;
    using Kadath::CapturedLinearSystemView;
    using Kadath::CapturedRowTag;
    using Kadath::CapturedRowTaxonomy;

    struct TemporaryCapture {
        std::filesystem::path path;

        explicit TemporaryCapture(std::string_view stem)
        {
            static std::atomic<unsigned long long> sequence{0};
            path = std::filesystem::temp_directory_path() /
                   (std::string(stem) + "." + std::to_string(static_cast<long long>(::getpid())) + "." +
                    std::to_string(sequence.fetch_add(1)) + ".kcsr");
            std::error_code ignored;
            std::filesystem::remove(path, ignored);
        }

        ~TemporaryCapture()
        {
            std::error_code ignored;
            std::filesystem::remove(path, ignored);
        }
    };

    struct SampleSystem {
        std::uint64_t rows = 3;
        std::uint64_t columns = 3;
        std::uint64_t full_rows = 8;
        std::uint64_t full_columns = 9;
        std::vector<int> irn{2, 2, 1, 1, 1, 3};
        std::vector<int> jcn{2, 2, 1, 1, 1, 3};
        std::vector<double> values{-0.0, 3.0, 1.0e16, -1.0e16, 1.0, 4.0};
        std::vector<double> rhs{1.0, 6.0, 4.0};
        std::vector<std::uint32_t> row_map{1, 4, 7};
        std::vector<std::uint32_t> column_map{0, 3, 8};
        std::vector<CapturedRowTag> row_tags{
            {1, CapturedRowTaxonomy::Vol, 0, -1},
            {4, CapturedRowTaxonomy::TauMatch, 0, 1},
            {7, CapturedRowTaxonomy::GlobalInt, -1, -1},
        };
        std::vector<CapturedColumnTag> column_tags{
            {0, CapturedColumnClass::FieldInteriorVol, 0, 0, 0, 0, -1, -1, 2, 0x11},
            {3, CapturedColumnClass::FieldMatching, 1, 0, 1, 0, -1, -1, 3, 0x22},
            {8, CapturedColumnClass::ScalarGlobal, 1, -1, 2, -1, 0, -1, -1, 0x33},
        };
        std::vector<int> permutation{2, 3, 1};
        CapturedAnalysisSettings analysis;

        SampleSystem()
        {
            column_tags[0].domain_type_id = 2;
            column_tags[0].tensor_component = 0;
            column_tags[0].coefficient_i = 2;
            column_tags[0].coefficient_j = 0;
            column_tags[0].coefficient_k = 0;
            column_tags[0].coefficient_nr = 4;
            column_tags[0].coefficient_nt = 3;
            column_tags[0].coefficient_np = 3;
            column_tags[1].domain_type_id = 7;
            column_tags[1].tensor_component = 1;
            column_tags[1].coefficient_i = 0;
            column_tags[1].coefficient_j = 1;
            column_tags[1].coefficient_k = 1;
            column_tags[1].coefficient_nr = 5;
            column_tags[1].coefficient_nt = 4;
            column_tags[1].coefficient_np = 4;
            constexpr std::string_view mumps_version = "5.7.3-test";
            std::copy(mumps_version.begin(), mumps_version.end(), analysis.mumps_version.begin());
            analysis.icntl[6] = 5;
            analysis.icntl[13] = 200;
            analysis.icntl[46] = 1;
            analysis.cntl[0] = 1.0e-6;
            analysis.requested_ordering = 5;
            analysis.actual_ordering = 5;
            analysis.communicator_size = 4;
            analysis.analysis_rank_count = 1;
            analysis.factor_ranks_per_node = 1;
            analysis.successful_factor_icntl14 = 200;
        }

        CapturedLinearSystemView view() const
        {
            CapturedLinearSystemView result;
            result.rows = rows;
            result.columns = columns;
            result.full_rows = full_rows;
            result.full_columns = full_columns;
            result.row_indices_1based = irn;
            result.column_indices_1based = jcn;
            result.values = values;
            result.rhs = rhs;
            result.row_full_indices_zero_based = row_map;
            result.column_full_indices_zero_based = column_map;
            result.row_tags = row_tags;
            result.column_tags = column_tags;
            result.analysis_permutation_1based = permutation;
            result.analysis = analysis;
            return result;
        }
    };

    void prepare_successful_replay_fixture(SampleSystem& sample)
    {
        ensure_mpi_initialized();
        int rank = 0;
        int size = 0;
        REQUIRE(MPI_Comm_rank(MPI_COMM_WORLD, &rank) == MPI_SUCCESS);
        REQUIRE(MPI_Comm_size(MPI_COMM_WORLD, &size) == MPI_SUCCESS);
        REQUIRE(size == 1);

        constexpr int requested_ordering = 5;
        constexpr int factor_ranks_per_node = 0;
        Kadath::MumpsLinearSolver solver(static_cast<int>(sample.rows), requested_ordering, false, 0, 200,
                                         MPI_COMM_WORLD, factor_ranks_per_node, false);
        solver.set_pattern(static_cast<int>(sample.rows), static_cast<long long>(sample.values.size()),
                           sample.irn.data(), sample.jcn.data());
        solver.analyze_pattern();
        solver.copy_analysis_controls(sample.analysis.icntl, sample.analysis.cntl);
        solver.copy_symmetric_permutation_1based(sample.permutation);
        sample.analysis.mumps_version.fill('\0');
        constexpr std::string_view mumps_version = MUMPS_VERSION;
        static_assert(mumps_version.size() < CapturedAnalysisSettings::mumps_version_bytes);
        std::copy(mumps_version.begin(), mumps_version.end(), sample.analysis.mumps_version.begin());
        sample.analysis.requested_ordering = requested_ordering;
        sample.analysis.actual_ordering = solver.last_actual_ordering();
        sample.analysis.communicator_size = size;
        sample.analysis.analysis_rank_count = solver.analysis_rank_count();
        sample.analysis.factor_ranks_per_node = solver.factor_ranks_per_node();
        solver.factor_analyzed(rank == 0 ? sample.values.data() : nullptr);
        sample.analysis.successful_factor_icntl14 = solver.successful_factor_icntl14();
        sample.analysis.factor_retry_count = solver.factor_retry_count();
        solver.reset();
    }

    void flip_byte(const std::filesystem::path& path, std::streamoff offset)
    {
        std::fstream stream(path, std::ios::binary | std::ios::in | std::ios::out);
        REQUIRE(stream.good());
        stream.seekg(offset);
        char value = 0;
        stream.read(&value, 1);
        REQUIRE(stream.gcount() == 1);
        value ^= 0x01;
        stream.seekp(offset);
        stream.write(&value, 1);
        stream.close();
        REQUIRE(stream.good());
    }

    void set_u64_be(const std::filesystem::path& path, std::streamoff offset, std::uint64_t value)
    {
        std::array<char, 8> bytes{};
        for (int index = 7; index >= 0; --index) {
            bytes[static_cast<std::size_t>(index)] = static_cast<char>(value & 0xff);
            value >>= 8;
        }
        std::fstream stream(path, std::ios::binary | std::ios::in | std::ios::out);
        REQUIRE(stream.good());
        stream.seekp(offset);
        stream.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
        stream.close();
        REQUIRE(stream.good());
    }

    std::uint8_t hex_nibble(char value)
    {
        if (value >= '0' && value <= '9')
            return static_cast<std::uint8_t>(value - '0');
        if (value >= 'a' && value <= 'f')
            return static_cast<std::uint8_t>(value - 'a' + 10);
        throw std::runtime_error("invalid hex fixture");
    }

    void write_hex_fixture(const std::filesystem::path& path, std::string_view hex)
    {
        REQUIRE(hex.size() % 2 == 0);
        std::ofstream output(path, std::ios::binary | std::ios::trunc);
        REQUIRE(output.good());
        for (std::size_t index = 0; index < hex.size(); index += 2) {
            const auto value = static_cast<char>((hex_nibble(hex[index]) << 4) | hex_nibble(hex[index + 1]));
            output.put(value);
        }
        output.close();
        REQUIRE(output.good());
    }

} // namespace

TEST_CASE("captured linear system round-trips raw COO, metadata, and analysis state", "[captured-linear-system]")
{
    TemporaryCapture capture("kadath_capture_roundtrip");
    const SampleSystem sample;
    const auto written = Kadath::write_captured_linear_system(capture.path, sample.view());
    const auto replay = Kadath::read_captured_linear_system(capture.path);

    CHECK(replay.schema_version == 2);
    CHECK(replay.rows == sample.rows);
    CHECK(replay.columns == sample.columns);
    CHECK(replay.full_rows == sample.full_rows);
    CHECK(replay.full_columns == sample.full_columns);
    CHECK(replay.row_indices_1based == sample.irn);
    CHECK(replay.column_indices_1based == sample.jcn);
    REQUIRE(replay.values.size() == sample.values.size());
    for (std::size_t index = 0; index < replay.values.size(); ++index) {
        CHECK(std::bit_cast<std::uint64_t>(replay.values[index]) == std::bit_cast<std::uint64_t>(sample.values[index]));
    }
    CHECK(replay.rhs == sample.rhs);
    CHECK(replay.row_full_indices_zero_based == sample.row_map);
    CHECK(replay.column_full_indices_zero_based == sample.column_map);
    CHECK(replay.analysis_permutation_1based == sample.permutation);
    CHECK(replay.analysis.icntl == sample.analysis.icntl);
    CHECK(replay.analysis.cntl == sample.analysis.cntl);
    CHECK(replay.analysis.mumps_version == sample.analysis.mumps_version);
    CHECK(replay.analysis.actual_ordering == 5);
    CHECK(replay.analysis.successful_factor_icntl14 == 200);
    CHECK(replay.analysis.factor_retry_count == 0);
    CHECK(replay.row_tags[1].taxonomy == CapturedRowTaxonomy::TauMatch);
    CHECK(replay.column_tags[2].column_class == CapturedColumnClass::ScalarGlobal);
    CHECK(replay.column_tags[2].var_name_hash == 0x33);
    CHECK(replay.column_tags[0].domain_type_id == 2);
    CHECK(replay.column_tags[0].tensor_component == 0);
    CHECK(replay.column_tags[0].coefficient_i == 2);
    CHECK(replay.column_tags[0].coefficient_j == 0);
    CHECK(replay.column_tags[0].coefficient_k == 0);
    CHECK(replay.column_tags[0].coefficient_nr == 4);
    CHECK(replay.column_tags[0].coefficient_nt == 3);
    CHECK(replay.column_tags[0].coefficient_np == 3);
    CHECK(replay.column_tags[1].domain_type_id == 7);
    CHECK(replay.column_tags[1].tensor_component == 1);
    CHECK(replay.column_tags[1].coefficient_i == 0);
    CHECK(replay.column_tags[1].coefficient_j == 1);
    CHECK(replay.column_tags[1].coefficient_k == 1);
    CHECK(replay.column_tags[1].coefficient_nr == 5);
    CHECK(replay.column_tags[1].coefficient_nt == 4);
    CHECK(replay.column_tags[1].coefficient_np == 4);
    CHECK(replay.column_tags[2].domain_type_id == -1);
    CHECK(replay.column_tags[2].tensor_component == -1);
    CHECK(replay.hashes.archive == written.archive);
    CHECK(replay.hashes.ordered_matrix == written.ordered_matrix);
    CHECK(replay.hashes.rhs == written.rhs);
    CHECK(replay.hashes.canonical_pattern == written.canonical_pattern);
    CHECK(replay.hashes.canonical_values == written.canonical_values);
    CHECK(Kadath::captured_ordered_matrix_hash(sample.rows, sample.columns, sample.irn, sample.jcn, sample.values) ==
          written.ordered_matrix);
    CHECK(Kadath::captured_linear_system_hash_hex(written.archive).size() == 64);
    CHECK_THROWS(Kadath::write_captured_linear_system(capture.path, sample.view()));
}

TEST_CASE("captured linear system reader maps schema-v1 field semantics to explicit unknown markers",
          "[captured-linear-system][schema-v1]")
{
    // This is a byte-for-byte schema-v1 archive emitted by the previous writer.
    // Keeping it independent of the current writer makes the compatibility check
    // sensitive to tag-width, offset, and archive-hash regressions.
    constexpr std::string_view schema_v1_hex =
        "4b41444b43535200000000010000001f010203040000022000000000000000030000000000000003"
        "00000000000000080000000000000009000000000000000600000000000000030000000000000003"
        "00000000000000030000000000000003000000000000000300000000000000030000000000000418"
        "0000000500000005000000010000000100000000000000c80000000000000000ffffffffffffffff"
        "ffffffff000000000000000000000007000000050000004d00000001000000000000000000000001"
        "00000000000000c80000000000000000000000000000000000000000000000000000000000000000"
        "00000000000000000000000000000000ffffffe00000000100000000000000000000000000000000"
        "000000000000000000000000000000010000000000000258000001f4000000000000000000000000"
        "00000000000000000000000000000000000000000000000000000000000000000000000000000000"
        "0000000000000000000000000000000000000000000000020000000000000000bff0000000000000"
        "3e500000000000000000000000000000bff000000000000000000000000000000000000000000000"
        "00000000000000000000000000000000000000000000000000000000000000000000000000000000"
        "0000000000000000000000000000000000000000000000000000000000000000352e392e30000000"
        "00000000000000000000000000000000000000000000000000000002000000020000000000000000"
        "00000002000000024008000000000000000000010000000140000000000000000000000100000001"
        "400800000000000000000001000000013ff000000000000000000003000000034010000000000000"
        "3ff00000000000004018000000000000401000000000000000000001000000040000000700000000"
        "0000000300000008000000010000000100000000ffffffff00000004000000030000000000000001"
        "0000000700000004ffffffffffffffff000000000000000400000000000000000000000000000000"
        "ffffffffffffffff0000000200000000000000110000000300000007000000010000000000000001"
        "00000000ffffffffffffffff000000030000000000000022000000080000000a00000001ffffffff"
        "00000002ffffffff00000000ffffffffffffffff0000000000000033000000020000000100000003"
        "bfb04840b7e25f38c0eb68294de84965dff5b7455b738cf3bbb577055559e92ca9f12261425c24b1"
        "ced960411733b22e7a06e18e39634e46fd8f13c3e681021416ed84c7f9854177d487207a2ece1a2a"
        "1abe9a1ab6317437a589077f997bf741a2d3d5284f62439189053bfc35ebf70d2327c79d65528aba"
        "2ace4226057c23f917675dd01463655286b9eaec409f381066f87aa752359bdfaab7478ad5e0a0c0"
        "4b435352454e4400";

    TemporaryCapture capture("kadath_capture_schema_v1");
    write_hex_fixture(capture.path, schema_v1_hex);
    REQUIRE(std::filesystem::file_size(capture.path) == 1048);
    const auto replay = Kadath::read_captured_linear_system(capture.path);

    CHECK(replay.schema_version == 1);
    CHECK(replay.rows == 3);
    CHECK(replay.columns == 3);
    CHECK(replay.column_tags[0].column_class == CapturedColumnClass::FieldInteriorVol);
    CHECK(replay.column_tags[1].column_class == CapturedColumnClass::FieldMatching);
    for (const CapturedColumnTag& tag : replay.column_tags) {
        CHECK(tag.domain_type_id == -1);
        CHECK(tag.tensor_component == -1);
        CHECK(tag.coefficient_i == -1);
        CHECK(tag.coefficient_j == -1);
        CHECK(tag.coefficient_k == -1);
        CHECK(tag.coefficient_nr == -1);
        CHECK(tag.coefficient_nt == -1);
        CHECK(tag.coefficient_np == -1);
    }
}

TEST_CASE("capture request validation is collective and exact", "[captured-linear-system][capture-request-mpi]")
{
    ensure_mpi_initialized();
    int rank = 0;
    int size = 0;
    REQUIRE(MPI_Comm_rank(MPI_COMM_WORLD, &rank) == MPI_SUCCESS);
    REQUIRE(MPI_Comm_size(MPI_COMM_WORLD, &size) == MPI_SUCCESS);

    CHECK_NOTHROW(Kadath::validate_captured_linear_system_collective_request(MPI_COMM_WORLD, {}, 0, false));
    CHECK_THROWS_WITH(Kadath::validate_captured_linear_system_collective_request(MPI_COMM_WORLD,
                                                                                 "/tmp/invalid-ordinal.kcsr", 0, false),
                      Catch::Matchers::ContainsSubstring("same positive integer"));
    if (size < 2)
        return;

    CHECK_THROWS_WITH(
        Kadath::validate_captured_linear_system_collective_request(
            MPI_COMM_WORLD, rank == 0 ? std::string_view{"/tmp/enabled.kcsr"} : std::string_view{}, 1, false),
        Catch::Matchers::ContainsSubstring("enable state differs"));
    CHECK_THROWS_WITH(Kadath::validate_captured_linear_system_collective_request(MPI_COMM_WORLD, "/tmp/ordinal.kcsr",
                                                                                 rank + 1, false),
                      Catch::Matchers::ContainsSubstring("same positive integer"));
    CHECK_THROWS_WITH(Kadath::validate_captured_linear_system_collective_request(
                          MPI_COMM_WORLD,
                          rank == 0 ? std::string_view{"/tmp/path-a.kcsr"} : std::string_view{"/tmp/path-b.kcsr"}, 1,
                          false),
                      Catch::Matchers::ContainsSubstring("path differs"));
    CHECK_THROWS_WITH(Kadath::validate_captured_linear_system_collective_request(
                          MPI_COMM_WORLD, "/tmp/reuse-divergence.kcsr", 1, rank == 0),
                      Catch::Matchers::ContainsSubstring("analyze-reuse state differs"));
    CHECK_THROWS_WITH(
        Kadath::validate_captured_linear_system_collective_request(MPI_COMM_WORLD, "/tmp/reuse-enabled.kcsr", 1, true),
        Catch::Matchers::ContainsSubstring("incompatible with sparse analyze reuse"));
}

TEST_CASE("capture hashes preserve raw order and signed zero but canonicalize duplicates", "[captured-linear-system]")
{
    SampleSystem first;
    SampleSystem second = first;
    second.irn = {3, 1, 1, 1, 2, 2};
    second.jcn = {3, 1, 1, 1, 2, 2};
    second.values = {4.0, 1.0e16, 1.0, -1.0e16, 3.0, -0.0};
    TemporaryCapture first_path("kadath_capture_hash_first");
    TemporaryCapture second_path("kadath_capture_hash_second");
    const auto first_hashes = Kadath::write_captured_linear_system(first_path.path, first.view());
    const auto second_hashes = Kadath::write_captured_linear_system(second_path.path, second.view());
    CHECK(first_hashes.ordered_matrix != second_hashes.ordered_matrix);
    CHECK(first_hashes.canonical_pattern == second_hashes.canonical_pattern);
    CHECK(first_hashes.canonical_values == second_hashes.canonical_values);

    SampleSystem positive_zero = first;
    positive_zero.values[0] = 0.0;
    TemporaryCapture zero_path("kadath_capture_hash_zero");
    const auto zero_hashes = Kadath::write_captured_linear_system(zero_path.path, positive_zero.view());
    CHECK(first_hashes.ordered_matrix != zero_hashes.ordered_matrix);
    CHECK(first_hashes.canonical_pattern == zero_hashes.canonical_pattern);
    CHECK(first_hashes.canonical_values == zero_hashes.canonical_values);

    SampleSystem changed_value = first;
    changed_value.values.back() = 5.0;
    TemporaryCapture changed_path("kadath_capture_hash_changed");
    const auto changed_hashes = Kadath::write_captured_linear_system(changed_path.path, changed_value.view());
    CHECK(first_hashes.canonical_pattern == changed_hashes.canonical_pattern);
    CHECK(first_hashes.canonical_values != changed_hashes.canonical_values);
}

TEST_CASE("capture refuses invalid indices, nonfinite data, and split column groups", "[captured-linear-system]")
{
    SECTION("bad index")
    {
        SampleSystem sample;
        sample.irn[0] = 0;
        TemporaryCapture path("kadath_capture_bad_index");
        CHECK_THROWS(Kadath::write_captured_linear_system(path.path, sample.view()));
        CHECK_FALSE(std::filesystem::exists(path.path));
    }
    SECTION("nonfinite matrix")
    {
        SampleSystem sample;
        sample.values[0] = std::numeric_limits<double>::infinity();
        TemporaryCapture path("kadath_capture_bad_matrix");
        CHECK_THROWS(Kadath::write_captured_linear_system(path.path, sample.view()));
    }
    SECTION("nonfinite RHS")
    {
        SampleSystem sample;
        sample.rhs[0] = std::numeric_limits<double>::quiet_NaN();
        TemporaryCapture path("kadath_capture_bad_rhs");
        CHECK_THROWS(Kadath::write_captured_linear_system(path.path, sample.view()));
    }
    SECTION("split column")
    {
        SampleSystem sample;
        sample.jcn = {2, 1, 2, 1, 1, 3};
        TemporaryCapture path("kadath_capture_split_column");
        CHECK_THROWS(Kadath::write_captured_linear_system(path.path, sample.view()));
    }
    SECTION("unknown row taxonomy")
    {
        SampleSystem sample;
        sample.row_tags[0].taxonomy = CapturedRowTaxonomy::Unknown;
        TemporaryCapture path("kadath_capture_unknown_row");
        CHECK_THROWS(Kadath::write_captured_linear_system(path.path, sample.view()));
    }
    SECTION("unknown column class")
    {
        SampleSystem sample;
        sample.column_tags[0].column_class = CapturedColumnClass::Unknown;
        TemporaryCapture path("kadath_capture_unknown_column");
        CHECK_THROWS(Kadath::write_captured_linear_system(path.path, sample.view()));
    }
    SECTION("missing schema-v2 field semantics")
    {
        SampleSystem sample;
        sample.column_tags[0].domain_type_id = -1;
        TemporaryCapture path("kadath_capture_missing_field_semantics");
        CHECK_THROWS_WITH(Kadath::write_captured_linear_system(path.path, sample.view()),
                          Catch::Matchers::ContainsSubstring("unknown mode semantics"));
    }
    SECTION("schema-v2 field coordinate outside its captured extent")
    {
        SampleSystem sample;
        sample.column_tags[0].coefficient_i = sample.column_tags[0].coefficient_nr;
        TemporaryCapture path("kadath_capture_field_coordinate_range");
        CHECK_THROWS_WITH(Kadath::write_captured_linear_system(path.path, sample.view()),
                          Catch::Matchers::ContainsSubstring("out-of-range mode coordinate"));
    }
    SECTION("schema-v2 non-field column contains field semantics")
    {
        SampleSystem sample;
        sample.column_tags[2].domain_type_id = 1;
        TemporaryCapture path("kadath_capture_non_field_semantics");
        CHECK_THROWS_WITH(Kadath::write_captured_linear_system(path.path, sample.view()),
                          Catch::Matchers::ContainsSubstring("non-field column"));
    }
    SECTION("missing MUMPS version")
    {
        SampleSystem sample;
        sample.analysis.mumps_version.fill('\0');
        TemporaryCapture path("kadath_capture_missing_mumps_version");
        CHECK_THROWS(Kadath::write_captured_linear_system(path.path, sample.view()));
    }
    SECTION("nonzero MUMPS version padding")
    {
        SampleSystem sample;
        sample.analysis.mumps_version.back() = 'x';
        TemporaryCapture path("kadath_capture_bad_mumps_version_padding");
        CHECK_THROWS(Kadath::write_captured_linear_system(path.path, sample.view()));
    }
    SECTION("factor retry provenance mismatch")
    {
        SampleSystem sample;
        sample.analysis.factor_retry_count = 1;
        sample.analysis.successful_factor_icntl14 = 301;
        TemporaryCapture path("kadath_capture_bad_factor_retry");
        CHECK_THROWS_WITH(Kadath::write_captured_linear_system(path.path, sample.view()),
                          Catch::Matchers::ContainsSubstring("seed and retry count"));
    }
    SECTION("factor retry provenance is exact")
    {
        SampleSystem sample;
        sample.analysis.factor_retry_count = 1;
        sample.analysis.successful_factor_icntl14 = 300;
        TemporaryCapture path("kadath_capture_factor_retry");
        CHECK_NOTHROW(Kadath::write_captured_linear_system(path.path, sample.view()));
        const auto replay = Kadath::read_captured_linear_system(path.path);
        CHECK(replay.analysis.factor_retry_count == 1);
        CHECK(replay.analysis.successful_factor_icntl14 == 300);
    }
    SECTION("factor retry provenance refuses signed-int growth overflow")
    {
        SampleSystem sample;
        sample.analysis.icntl[13] = std::numeric_limits<std::int32_t>::max();
        sample.analysis.successful_factor_icntl14 = std::numeric_limits<std::int32_t>::max();
        sample.analysis.factor_retry_count = 1;
        TemporaryCapture path("kadath_capture_factor_retry_overflow");
        CHECK_THROWS_WITH(Kadath::write_captured_linear_system(path.path, sample.view()),
                          Catch::Matchers::ContainsSubstring("overflows int32"));
    }
}

TEST_CASE("ordered matrix hashing refuses malformed numeric input", "[captured-linear-system][failure]")
{
    const std::array<int, 1> index{{1}};
    const std::array<double, 1> finite{{1.0}};
    const std::array<double, 1> nonfinite{{std::numeric_limits<double>::infinity()}};
    CHECK_THROWS(Kadath::captured_ordered_matrix_hash(0, 1, index, index, finite));
    CHECK_THROWS(Kadath::captured_ordered_matrix_hash(1, 1, std::span<const int>{}, index, finite));
    CHECK_THROWS(Kadath::captured_ordered_matrix_hash(1, 1, index, index, nonfinite));
}

TEST_CASE("capture reader rejects corruption, count overflow, truncation, and trailing bytes",
          "[captured-linear-system]")
{
    const SampleSystem sample;
    SECTION("payload corruption")
    {
        TemporaryCapture path("kadath_capture_corrupt");
        Kadath::write_captured_linear_system(path.path, sample.view());
        flip_byte(path.path, 544 + 8);
        CHECK_THROWS(Kadath::read_captured_linear_system(path.path));
    }
    SECTION("count overflow")
    {
        TemporaryCapture path("kadath_capture_count");
        Kadath::write_captured_linear_system(path.path, sample.view());
        set_u64_be(path.path, 56, std::numeric_limits<std::uint64_t>::max());
        CHECK_THROWS(Kadath::read_captured_linear_system(path.path));
    }
    SECTION("truncation")
    {
        TemporaryCapture path("kadath_capture_truncated");
        Kadath::write_captured_linear_system(path.path, sample.view());
        const auto size = std::filesystem::file_size(path.path);
        std::filesystem::resize_file(path.path, size - 1);
        CHECK_THROWS(Kadath::read_captured_linear_system(path.path));
    }
    SECTION("trailing byte")
    {
        TemporaryCapture path("kadath_capture_trailing");
        Kadath::write_captured_linear_system(path.path, sample.view());
        std::ofstream output(path.path, std::ios::binary | std::ios::app);
        output.put('\0');
        output.close();
        REQUIRE(output.good());
        CHECK_THROWS(Kadath::read_captured_linear_system(path.path));
    }
    SECTION("symbolic link")
    {
        TemporaryCapture path("kadath_capture_symlink_target");
        TemporaryCapture link("kadath_capture_symlink");
        Kadath::write_captured_linear_system(path.path, sample.view());
        std::filesystem::create_symlink(path.path, link.path);
        CHECK_THROWS(Kadath::read_captured_linear_system(link.path));
    }
}

TEST_CASE("scale-aware backward error uses duplicate-coalesced matrix rows", "[captured-linear-system]")
{
    TemporaryCapture path("kadath_capture_backward");
    const SampleSystem sample;
    Kadath::write_captured_linear_system(path.path, sample.view());
    const auto replay = Kadath::read_captured_linear_system(path.path);

    const std::array<double, 3> exact_solution{{1.0, 2.0, 1.0}};
    const auto exact = Kadath::scale_aware_backward_error(replay, exact_solution);
    CHECK(exact.value == 0.0);
    CHECK(exact.componentwise_value == 0.0);

    const std::array<double, 3> wrong_solution{{0.0, 2.0, 1.0}};
    const auto wrong = Kadath::scale_aware_backward_error(replay, wrong_solution);
    CHECK(wrong.value == Catch::Approx(1.0 / 14.0));
    CHECK(wrong.componentwise_value == Catch::Approx(1.0));
    CHECK(wrong.worst_factor_row_zero_based == 0);
    CHECK(wrong.worst_full_row_zero_based == 1);
}

TEST_CASE("scale-aware backward error avoids an overflowing normalization ratio", "[captured-linear-system]")
{
    Kadath::CapturedLinearSystem system;
    system.rows = 1;
    system.columns = 1;
    system.row_indices_1based = {1};
    system.column_indices_1based = {1};
    system.values = {1.0e200};
    system.rhs = {0.0};
    system.row_full_indices_zero_based = {7};
    const std::array<double, 1> solution{{1.0e200}};
    const auto error = Kadath::scale_aware_backward_error(system, solution);
    CHECK(error.value == Catch::Approx(1.0));
    CHECK(error.componentwise_value == Catch::Approx(1.0));
    CHECK(error.worst_full_row_zero_based == 7);
}

TEST_CASE("captured linear system test can emit a replay smoke fixture",
          "[captured-linear-system][captured-linear-system-fixture]")
{
    const char* requested_path = std::getenv("CAPTURED_LINEAR_SYSTEM_FIXTURE");
    const char* singular_path = std::getenv("CAPTURED_LINEAR_SYSTEM_SINGULAR_FIXTURE");
    const bool regular_requested = requested_path != nullptr && requested_path[0] != '\0';
    const bool singular_requested = singular_path != nullptr && singular_path[0] != '\0';
    if (!regular_requested && !singular_requested) {
        SUCCEED("fixture output not requested");
        return;
    }
    SampleSystem sample;
    // Keep duplicate COO entries in the replay fixture, but avoid the extreme
    // cancellation used by the hash tests: MUMPS is free to accumulate raw
    // duplicates in a numerically different order.
    sample.values = {0.0, 3.0, 2.0, 3.0, 1.0, 4.0};
    prepare_successful_replay_fixture(sample);
    if (regular_requested) {
        const std::filesystem::path output(requested_path);
        Kadath::write_captured_linear_system(output, sample.view());
        REQUIRE(std::filesystem::is_regular_file(output));
        const auto replay = Kadath::read_captured_linear_system(output);
        CHECK(replay.rows == sample.rows);
    }
    if (singular_requested) {
        sample.values.back() = 0.0;
        const std::filesystem::path output(singular_path);
        Kadath::write_captured_linear_system(output, sample.view());
        REQUIRE(std::filesystem::is_regular_file(output));
    }
}
