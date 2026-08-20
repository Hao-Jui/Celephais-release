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

#include <catch2/matchers/catch_matchers_string.hpp>
#include <catch2/catch_test_macros.hpp>

#include "Linear_algebra/mumps_tree_cache.hpp"

#include <array>
#include <atomic>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <string>
#include <string_view>
#include <unistd.h>
#include <vector>

namespace
{

    struct TemporaryCache {
        std::filesystem::path path;

        explicit TemporaryCache(std::string_view stem, std::string_view extension = ".mumpstree")
        {
            static std::atomic<unsigned long long> sequence{0};
            path = std::filesystem::temp_directory_path() /
                   (std::string(stem) + "." + std::to_string(static_cast<long long>(::getpid())) + "." +
                    std::to_string(sequence.fetch_add(1)) + std::string(extension));
            std::error_code ignored;
            std::filesystem::remove(path, ignored);
        }

        ~TemporaryCache()
        {
            std::error_code ignored;
            std::filesystem::remove(path, ignored);
        }
    };

    struct SampleCache {
        int dimension = 4;
        long long pattern_nnz = 37;
        bool matching_applied = true;
        std::vector<int> column_permutation{3, 1, 4, 2};
        std::vector<int> symmetric_permutation{2, 4, 1, 3};

        Kadath::MumpsTreeCacheView view() const
        {
            return {dimension, pattern_nnz, matching_applied, column_permutation, symmetric_permutation};
        }
    };

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

    void set_u32_be(const std::filesystem::path& path, std::streamoff offset, std::uint32_t value)
    {
        std::array<char, 4> bytes{};
        for (int index = 3; index >= 0; --index) {
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

} // namespace

TEST_CASE("mumpstree cache round-trips both analysis permutations", "[mumps-tree-cache]")
{
    TemporaryCache path("kadath_mumpstree_roundtrip");
    const SampleCache sample;
    Kadath::write_mumps_tree_cache(path.path, sample.view());
    const auto replay = Kadath::read_mumps_tree_cache(path.path, sample.dimension);

    CHECK(replay.dimension == sample.dimension);
    CHECK(replay.pattern_nnz == sample.pattern_nnz);
    CHECK(replay.matching_applied);
    CHECK(replay.column_permutation_1based == sample.column_permutation);
    CHECK(replay.symmetric_permutation_1based == sample.symmetric_permutation);

    TemporaryCache identity_path("kadath_mumpstree_identity");
    SampleCache identity = sample;
    identity.matching_applied = false;
    identity.column_permutation = {1, 2, 3, 4};
    Kadath::write_mumps_tree_cache(identity_path.path, identity.view());
    const auto identity_replay = Kadath::read_mumps_tree_cache(identity_path.path, identity.dimension);
    CHECK_FALSE(identity_replay.matching_applied);
    CHECK(identity_replay.column_permutation_1based == identity.column_permutation);
}

TEST_CASE("mumpstree writer validates permutations and provenance", "[mumps-tree-cache]")
{
    SECTION("UNS_PERM length")
    {
        TemporaryCache path("kadath_mumpstree_uns_length");
        SampleCache sample;
        sample.column_permutation.pop_back();
        CHECK_THROWS(Kadath::write_mumps_tree_cache(path.path, sample.view()));
        CHECK_FALSE(std::filesystem::exists(path.path));
    }
    SECTION("UNS_PERM range")
    {
        TemporaryCache path("kadath_mumpstree_uns_range");
        SampleCache sample;
        sample.column_permutation[0] = 0;
        CHECK_THROWS(Kadath::write_mumps_tree_cache(path.path, sample.view()));
    }
    SECTION("SYM_PERM bijection")
    {
        TemporaryCache path("kadath_mumpstree_sym_duplicate");
        SampleCache sample;
        sample.symmetric_permutation[0] = sample.symmetric_permutation[1];
        CHECK_THROWS(Kadath::write_mumps_tree_cache(path.path, sample.view()));
    }
    SECTION("unmatched UNS_PERM identity")
    {
        TemporaryCache path("kadath_mumpstree_unmatched_nonidentity");
        SampleCache sample;
        sample.matching_applied = false;
        CHECK_THROWS_WITH(Kadath::write_mumps_tree_cache(path.path, sample.view()),
                          Catch::Matchers::ContainsSubstring("must be identity"));
    }
    SECTION("empty dimension")
    {
        TemporaryCache path("kadath_mumpstree_empty");
        SampleCache sample;
        sample.dimension = 0;
        sample.column_permutation.clear();
        sample.symmetric_permutation.clear();
        CHECK_THROWS(Kadath::write_mumps_tree_cache(path.path, sample.view()));
    }
    SECTION("empty pattern")
    {
        TemporaryCache path("kadath_mumpstree_empty_pattern");
        SampleCache sample;
        sample.pattern_nnz = 0;
        CHECK_THROWS(Kadath::write_mumps_tree_cache(path.path, sample.view()));
    }
    SECTION("extension")
    {
        TemporaryCache path("kadath_mumpstree_extension", ".bin");
        const SampleCache sample;
        CHECK_THROWS(Kadath::write_mumps_tree_cache(path.path, sample.view()));
        CHECK_THROWS(Kadath::read_mumps_tree_cache(path.path, sample.dimension));
        CHECK_THROWS(Kadath::remove_mumps_tree_cache(path.path));
    }
}

TEST_CASE("mumpstree writer publishes without replacement", "[mumps-tree-cache]")
{
    TemporaryCache path("kadath_mumpstree_no_replace");
    const SampleCache sample;
    Kadath::write_mumps_tree_cache(path.path, sample.view());
    const auto original_size = std::filesystem::file_size(path.path);

    SampleCache changed = sample;
    changed.pattern_nnz = 99;
    CHECK_THROWS_WITH(Kadath::write_mumps_tree_cache(path.path, changed.view()),
                      Catch::Matchers::ContainsSubstring("refusing to overwrite"));
    CHECK(std::filesystem::file_size(path.path) == original_size);
    CHECK(Kadath::read_mumps_tree_cache(path.path, sample.dimension).pattern_nnz == sample.pattern_nnz);

    TemporaryCache dangling("kadath_mumpstree_dangling_link");
    TemporaryCache missing("kadath_mumpstree_missing_target");
    std::filesystem::create_symlink(missing.path, dangling.path);
    CHECK_THROWS(Kadath::write_mumps_tree_cache(dangling.path, sample.view()));
    CHECK(std::filesystem::is_symlink(dangling.path));
}

TEST_CASE("mumpstree reader rejects incompatible and malformed files", "[mumps-tree-cache]")
{
    const SampleCache sample;
    SECTION("expected dimension mismatch")
    {
        TemporaryCache path("kadath_mumpstree_dimension_mismatch");
        Kadath::write_mumps_tree_cache(path.path, sample.view());
        CHECK_THROWS_WITH(Kadath::read_mumps_tree_cache(path.path, sample.dimension + 1),
                          Catch::Matchers::ContainsSubstring("does not match expected"));
    }
    SECTION("oversized declared dimension")
    {
        TemporaryCache path("kadath_mumpstree_oversized_dimension");
        Kadath::write_mumps_tree_cache(path.path, sample.view());
        set_u64_be(path.path, 24, std::numeric_limits<std::uint64_t>::max());
        CHECK_THROWS_WITH(Kadath::read_mumps_tree_cache(path.path, sample.dimension),
                          Catch::Matchers::ContainsSubstring("outside the supported range"));
    }
    SECTION("oversized actual file")
    {
        TemporaryCache path("kadath_mumpstree_oversized_file");
        Kadath::write_mumps_tree_cache(path.path, sample.view());
        std::filesystem::resize_file(path.path, 80'000'105);
        CHECK_THROWS_WITH(Kadath::read_mumps_tree_cache(path.path, sample.dimension),
                          Catch::Matchers::ContainsSubstring("maximum size"));
    }
    SECTION("bad schema version")
    {
        TemporaryCache path("kadath_mumpstree_version");
        Kadath::write_mumps_tree_cache(path.path, sample.view());
        set_u32_be(path.path, 8, 2);
        CHECK_THROWS_WITH(Kadath::read_mumps_tree_cache(path.path, sample.dimension),
                          Catch::Matchers::ContainsSubstring("schema version"));
    }
    SECTION("unknown flags")
    {
        TemporaryCache path("kadath_mumpstree_flags");
        Kadath::write_mumps_tree_cache(path.path, sample.view());
        set_u32_be(path.path, 12, 0x2);
        CHECK_THROWS_WITH(Kadath::read_mumps_tree_cache(path.path, sample.dimension),
                          Catch::Matchers::ContainsSubstring("unsupported flags"));
    }
    SECTION("permutation count mismatch")
    {
        TemporaryCache path("kadath_mumpstree_count");
        Kadath::write_mumps_tree_cache(path.path, sample.view());
        set_u64_be(path.path, 40, static_cast<std::uint64_t>(sample.dimension + 1));
        CHECK_THROWS_WITH(Kadath::read_mumps_tree_cache(path.path, sample.dimension),
                          Catch::Matchers::ContainsSubstring("counts do not match"));
    }
    SECTION("declared total mismatch")
    {
        TemporaryCache path("kadath_mumpstree_total");
        Kadath::write_mumps_tree_cache(path.path, sample.view());
        set_u64_be(path.path, 56, 1);
        CHECK_THROWS_WITH(Kadath::read_mumps_tree_cache(path.path, sample.dimension),
                          Catch::Matchers::ContainsSubstring("declared file size"));
    }
    SECTION("invalid stored nnz provenance")
    {
        TemporaryCache path("kadath_mumpstree_stored_nnz");
        Kadath::write_mumps_tree_cache(path.path, sample.view());
        set_u64_be(path.path, 32, 0);
        CHECK_THROWS_WITH(Kadath::read_mumps_tree_cache(path.path, sample.dimension),
                          Catch::Matchers::ContainsSubstring("nnz provenance"));
    }
    SECTION("invalid expected dimension")
    {
        TemporaryCache path("kadath_mumpstree_expected_dimension");
        Kadath::write_mumps_tree_cache(path.path, sample.view());
        CHECK_THROWS_WITH(Kadath::read_mumps_tree_cache(path.path, 0),
                          Catch::Matchers::ContainsSubstring("expected dimension"));
    }
    SECTION("bad magic")
    {
        TemporaryCache path("kadath_mumpstree_magic");
        Kadath::write_mumps_tree_cache(path.path, sample.view());
        flip_byte(path.path, 0);
        CHECK_THROWS_WITH(Kadath::read_mumps_tree_cache(path.path, sample.dimension),
                          Catch::Matchers::ContainsSubstring("bad magic"));
    }
    SECTION("payload corruption")
    {
        TemporaryCache path("kadath_mumpstree_hash");
        Kadath::write_mumps_tree_cache(path.path, sample.view());
        flip_byte(path.path, 64);
        CHECK_THROWS_WITH(Kadath::read_mumps_tree_cache(path.path, sample.dimension),
                          Catch::Matchers::ContainsSubstring("archive hash mismatch"));
    }
    SECTION("truncation")
    {
        TemporaryCache path("kadath_mumpstree_truncated");
        Kadath::write_mumps_tree_cache(path.path, sample.view());
        const auto size = std::filesystem::file_size(path.path);
        std::filesystem::resize_file(path.path, size - 1);
        CHECK_THROWS_WITH(Kadath::read_mumps_tree_cache(path.path, sample.dimension),
                          Catch::Matchers::ContainsSubstring("truncated"));
    }
    SECTION("trailing byte")
    {
        TemporaryCache path("kadath_mumpstree_trailing");
        Kadath::write_mumps_tree_cache(path.path, sample.view());
        std::ofstream output(path.path, std::ios::binary | std::ios::app);
        output.put('\0');
        output.close();
        REQUIRE(output.good());
        CHECK_THROWS_WITH(Kadath::read_mumps_tree_cache(path.path, sample.dimension),
                          Catch::Matchers::ContainsSubstring("trailing bytes"));
    }
}

TEST_CASE("mumpstree read and delete refuse symlinks and nonregular paths", "[mumps-tree-cache]")
{
    const SampleCache sample;
    TemporaryCache target("kadath_mumpstree_symlink_target");
    TemporaryCache link("kadath_mumpstree_symlink");
    Kadath::write_mumps_tree_cache(target.path, sample.view());
    std::filesystem::create_symlink(target.path, link.path);

    CHECK_THROWS_WITH(Kadath::read_mumps_tree_cache(link.path, sample.dimension),
                      Catch::Matchers::ContainsSubstring("symbolic-link"));
    CHECK_THROWS_WITH(Kadath::remove_mumps_tree_cache(link.path), Catch::Matchers::ContainsSubstring("symbolic-link"));
    CHECK(std::filesystem::is_regular_file(target.path));
    CHECK(std::filesystem::is_symlink(link.path));

    TemporaryCache directory("kadath_mumpstree_directory");
    std::filesystem::create_directory(directory.path);
    CHECK_THROWS_WITH(Kadath::read_mumps_tree_cache(directory.path, sample.dimension),
                      Catch::Matchers::ContainsSubstring("not a regular file"));
    CHECK_THROWS_WITH(Kadath::remove_mumps_tree_cache(directory.path),
                      Catch::Matchers::ContainsSubstring("non-regular"));
}

TEST_CASE("mumpstree delete is durable and idempotent", "[mumps-tree-cache]")
{
    TemporaryCache path("kadath_mumpstree_delete");
    const SampleCache sample;
    CHECK_FALSE(Kadath::remove_mumps_tree_cache(path.path));
    Kadath::write_mumps_tree_cache(path.path, sample.view());
    REQUIRE(std::filesystem::is_regular_file(path.path));
    CHECK(Kadath::remove_mumps_tree_cache(path.path));
    CHECK_FALSE(std::filesystem::exists(path.path));
    CHECK_FALSE(Kadath::remove_mumps_tree_cache(path.path));
}

TEST_CASE("mumpstree replay composes COO columns with inverse UNS_PERM",
          "[mumps-tree-cache][replay]")
{
    const std::vector<int> columns{1, 2, 3, 4, 3, 1};
    const std::vector<int> uns_permutation{3, 1, 4, 2};
    const std::vector<int> expected{2, 4, 1, 3, 1, 2};

    CHECK(Kadath::compose_mumps_tree_column_indices_1based(
              columns, uns_permutation, 4) == expected);

    CHECK_THROWS(Kadath::compose_mumps_tree_column_indices_1based(
        columns, std::vector<int>{3, 1, 4}, 4));
    CHECK_THROWS(Kadath::compose_mumps_tree_column_indices_1based(
        columns, std::vector<int>{3, 1, 3, 2}, 4));
    CHECK_THROWS(Kadath::compose_mumps_tree_column_indices_1based(
        std::vector<int>{0}, uns_permutation, 4));
    CHECK_THROWS(Kadath::compose_mumps_tree_column_indices_1based(
        columns, uns_permutation, 0));
}

TEST_CASE("mumpstree active stage path can be replaced and cleared",
          "[mumps-tree-cache][cleanup]")
{
    Kadath::clear_active_mumps_tree_cache_path();
    TemporaryCache first("kadath_mumpstree_active_first");
    TemporaryCache second("kadath_mumpstree_active_second");

    Kadath::set_active_mumps_tree_cache_path(first.path);
    CHECK(Kadath::active_mumps_tree_cache_path() == first.path);
    Kadath::set_active_mumps_tree_cache_path(second.path);
    CHECK(Kadath::active_mumps_tree_cache_path() == second.path);
    Kadath::clear_active_mumps_tree_cache_path();
    CHECK(Kadath::active_mumps_tree_cache_path().empty());

    TemporaryCache wrong_extension("kadath_mumpstree_active_bad", ".bin");
    CHECK_THROWS(Kadath::set_active_mumps_tree_cache_path(
        wrong_extension.path));
    CHECK(Kadath::active_mumps_tree_cache_path().empty());
}
