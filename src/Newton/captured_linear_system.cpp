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

#include "Linear_algebra/captured_linear_system.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <fcntl.h>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string_view>
#include <sys/stat.h>
#include <system_error>
#include <unistd.h>

namespace Kadath
{
    namespace
    {

        constexpr std::array<std::uint8_t, 8> kMagic{{'K', 'A', 'D', 'K', 'C', 'S', 'R', '\0'}};
        constexpr std::array<std::uint8_t, 8> kFooter{{'K', 'C', 'S', 'R', 'E', 'N', 'D', '\0'}};
        constexpr std::uint32_t kLegacySchemaVersion = 1;
        constexpr std::uint32_t kSchemaVersion = 2;
        constexpr std::uint32_t kRequiredFlags = 0x1f;
        constexpr std::uint32_t kEndianMarker = 0x01020304;
        constexpr std::uint32_t kHeaderBytes = 544;
        constexpr std::uint64_t kSemanticHashBytes = 4 * 32;
        constexpr std::uint64_t kArchiveHashBytes = 32;
        constexpr std::uint64_t kFooterBytes = 8;

        [[noreturn]] void fail(const std::string& message)
        {
            throw std::runtime_error("captured linear system: " + message);
        }

        std::uint64_t checked_add(std::uint64_t left, std::uint64_t right, std::string_view what)
        {
            if (right > std::numeric_limits<std::uint64_t>::max() - left)
                fail(std::string(what) + " byte count overflows uint64");
            return left + right;
        }

        std::uint64_t checked_multiply(std::uint64_t left, std::uint64_t right, std::string_view what)
        {
            if (left != 0 && right > std::numeric_limits<std::uint64_t>::max() / left)
                fail(std::string(what) + " byte count overflows uint64");
            return left * right;
        }

        std::size_t checked_size(std::uint64_t count, std::string_view what)
        {
            if (count > std::numeric_limits<std::size_t>::max())
                fail(std::string(what) + " count does not fit size_t");
            return static_cast<std::size_t>(count);
        }

        std::uint64_t column_tag_bytes(std::uint32_t schema_version)
        {
            switch (schema_version) {
                case kLegacySchemaVersion:
                    return 44;
                case kSchemaVersion:
                    return 76;
                default:
                    fail("unsupported schema version");
            }
        }

        std::uint64_t expected_file_bytes(std::uint32_t schema_version, std::uint64_t rows, std::uint64_t columns,
                                          std::uint64_t nnz)
        {
            std::uint64_t bytes = kHeaderBytes;
            bytes = checked_add(bytes, checked_multiply(nnz, 16, "COO"), "COO");
            bytes = checked_add(bytes, checked_multiply(rows, 8, "RHS"), "RHS");
            bytes = checked_add(bytes, checked_multiply(rows, 4, "row map"), "row map");
            bytes = checked_add(bytes, checked_multiply(columns, 4, "column map"), "column map");
            bytes = checked_add(bytes, checked_multiply(rows, 16, "row tags"), "row tags");
            bytes = checked_add(bytes, checked_multiply(columns, column_tag_bytes(schema_version), "column tags"),
                                "column tags");
            bytes = checked_add(bytes, checked_multiply(columns, 4, "analysis permutation"), "analysis permutation");
            bytes = checked_add(bytes, kSemanticHashBytes, "semantic hashes");
            bytes = checked_add(bytes, kArchiveHashBytes, "archive hash");
            return checked_add(bytes, kFooterBytes, "footer");
        }

        std::array<std::uint8_t, 4> encode_u32(std::uint32_t value)
        {
            return {{static_cast<std::uint8_t>(value >> 24), static_cast<std::uint8_t>(value >> 16),
                     static_cast<std::uint8_t>(value >> 8), static_cast<std::uint8_t>(value)}};
        }

        std::array<std::uint8_t, 8> encode_u64(std::uint64_t value)
        {
            return {{static_cast<std::uint8_t>(value >> 56), static_cast<std::uint8_t>(value >> 48),
                     static_cast<std::uint8_t>(value >> 40), static_cast<std::uint8_t>(value >> 32),
                     static_cast<std::uint8_t>(value >> 24), static_cast<std::uint8_t>(value >> 16),
                     static_cast<std::uint8_t>(value >> 8), static_cast<std::uint8_t>(value)}};
        }

        std::uint32_t decode_u32(const std::uint8_t* bytes)
        {
            return (static_cast<std::uint32_t>(bytes[0]) << 24) | (static_cast<std::uint32_t>(bytes[1]) << 16) |
                   (static_cast<std::uint32_t>(bytes[2]) << 8) | static_cast<std::uint32_t>(bytes[3]);
        }

        std::uint64_t decode_u64(const std::uint8_t* bytes)
        {
            std::uint64_t value = 0;
            for (int index = 0; index < 8; ++index)
                value = (value << 8) | bytes[index];
            return value;
        }

        class Sha256
        {
          public:
            Sha256()
                : state_{{0x6a09e667U, 0xbb67ae85U, 0x3c6ef372U, 0xa54ff53aU, 0x510e527fU, 0x9b05688cU, 0x1f83d9abU,
                          0x5be0cd19U}}
            {
            }

            void update(const void* input, std::size_t size)
            {
                if (finished_)
                    fail("internal SHA-256 update after finish");
                if (size > (std::numeric_limits<std::uint64_t>::max() - byte_count_))
                    fail("SHA-256 input length overflows uint64");
                byte_count_ += static_cast<std::uint64_t>(size);
                const auto* bytes = static_cast<const std::uint8_t*>(input);
                while (size != 0) {
                    const std::size_t take = std::min(size, block_.size() - block_size_);
                    std::memcpy(block_.data() + block_size_, bytes, take);
                    block_size_ += take;
                    bytes += take;
                    size -= take;
                    if (block_size_ == block_.size()) {
                        compress(block_.data());
                        block_size_ = 0;
                    }
                }
            }

            CapturedLinearSystemHash finish()
            {
                if (finished_)
                    fail("internal SHA-256 finish called twice");
                if (byte_count_ > std::numeric_limits<std::uint64_t>::max() / 8)
                    fail("SHA-256 bit length overflows uint64");
                const std::uint64_t bit_count = byte_count_ * 8;
                block_[block_size_++] = 0x80;
                if (block_size_ > 56) {
                    std::fill(block_.begin() + static_cast<std::ptrdiff_t>(block_size_), block_.end(), 0);
                    compress(block_.data());
                    block_size_ = 0;
                }
                std::fill(block_.begin() + static_cast<std::ptrdiff_t>(block_size_), block_.begin() + 56, 0);
                const auto encoded_count = encode_u64(bit_count);
                std::copy(encoded_count.begin(), encoded_count.end(), block_.begin() + 56);
                compress(block_.data());
                finished_ = true;

                CapturedLinearSystemHash digest{};
                for (std::size_t word = 0; word < state_.size(); ++word) {
                    const auto encoded = encode_u32(state_[word]);
                    std::copy(encoded.begin(), encoded.end(), digest.begin() + static_cast<std::ptrdiff_t>(4 * word));
                }
                return digest;
            }

          private:
            static constexpr std::array<std::uint32_t, 64> constants_{{
                0x428a2f98U, 0x71374491U, 0xb5c0fbcfU, 0xe9b5dba5U, 0x3956c25bU, 0x59f111f1U, 0x923f82a4U, 0xab1c5ed5U,
                0xd807aa98U, 0x12835b01U, 0x243185beU, 0x550c7dc3U, 0x72be5d74U, 0x80deb1feU, 0x9bdc06a7U, 0xc19bf174U,
                0xe49b69c1U, 0xefbe4786U, 0x0fc19dc6U, 0x240ca1ccU, 0x2de92c6fU, 0x4a7484aaU, 0x5cb0a9dcU, 0x76f988daU,
                0x983e5152U, 0xa831c66dU, 0xb00327c8U, 0xbf597fc7U, 0xc6e00bf3U, 0xd5a79147U, 0x06ca6351U, 0x14292967U,
                0x27b70a85U, 0x2e1b2138U, 0x4d2c6dfcU, 0x53380d13U, 0x650a7354U, 0x766a0abbU, 0x81c2c92eU, 0x92722c85U,
                0xa2bfe8a1U, 0xa81a664bU, 0xc24b8b70U, 0xc76c51a3U, 0xd192e819U, 0xd6990624U, 0xf40e3585U, 0x106aa070U,
                0x19a4c116U, 0x1e376c08U, 0x2748774cU, 0x34b0bcb5U, 0x391c0cb3U, 0x4ed8aa4aU, 0x5b9cca4fU, 0x682e6ff3U,
                0x748f82eeU, 0x78a5636fU, 0x84c87814U, 0x8cc70208U, 0x90befffaU, 0xa4506cebU, 0xbef9a3f7U, 0xc67178f2U,
            }};

            static std::uint32_t choose(std::uint32_t x, std::uint32_t y, std::uint32_t z)
            {
                return (x & y) ^ (~x & z);
            }

            static std::uint32_t majority(std::uint32_t x, std::uint32_t y, std::uint32_t z)
            {
                return (x & y) ^ (x & z) ^ (y & z);
            }

            void compress(const std::uint8_t* block)
            {
                std::array<std::uint32_t, 64> words{};
                for (std::size_t index = 0; index < 16; ++index)
                    words[index] = decode_u32(block + 4 * index);
                for (std::size_t index = 16; index < words.size(); ++index) {
                    const std::uint32_t s0 =
                        std::rotr(words[index - 15], 7) ^ std::rotr(words[index - 15], 18) ^ (words[index - 15] >> 3);
                    const std::uint32_t s1 =
                        std::rotr(words[index - 2], 17) ^ std::rotr(words[index - 2], 19) ^ (words[index - 2] >> 10);
                    words[index] = words[index - 16] + s0 + words[index - 7] + s1;
                }

                std::uint32_t a = state_[0];
                std::uint32_t b = state_[1];
                std::uint32_t c = state_[2];
                std::uint32_t d = state_[3];
                std::uint32_t e = state_[4];
                std::uint32_t f = state_[5];
                std::uint32_t g = state_[6];
                std::uint32_t h = state_[7];
                for (std::size_t index = 0; index < words.size(); ++index) {
                    const std::uint32_t sum1 = std::rotr(e, 6) ^ std::rotr(e, 11) ^ std::rotr(e, 25);
                    const std::uint32_t temporary1 = h + sum1 + choose(e, f, g) + constants_[index] + words[index];
                    const std::uint32_t sum0 = std::rotr(a, 2) ^ std::rotr(a, 13) ^ std::rotr(a, 22);
                    const std::uint32_t temporary2 = sum0 + majority(a, b, c);
                    h = g;
                    g = f;
                    f = e;
                    e = d + temporary1;
                    d = c;
                    c = b;
                    b = a;
                    a = temporary1 + temporary2;
                }
                state_[0] += a;
                state_[1] += b;
                state_[2] += c;
                state_[3] += d;
                state_[4] += e;
                state_[5] += f;
                state_[6] += g;
                state_[7] += h;
            }

            std::array<std::uint32_t, 8> state_{};
            std::array<std::uint8_t, 64> block_{};
            std::size_t block_size_ = 0;
            std::uint64_t byte_count_ = 0;
            bool finished_ = false;
        };

        void hash_domain(Sha256& hash, std::string_view domain)
        {
            hash.update(domain.data(), domain.size());
            const std::uint8_t terminator = 0;
            hash.update(&terminator, 1);
        }

        void hash_u32(Sha256& hash, std::uint32_t value)
        {
            const auto encoded = encode_u32(value);
            hash.update(encoded.data(), encoded.size());
        }

        void hash_u64(Sha256& hash, std::uint64_t value)
        {
            const auto encoded = encode_u64(value);
            hash.update(encoded.data(), encoded.size());
        }

        // A finite binary64 is an integer multiple of 2^-1074. The largest finite
        // significand occupies bits through 2097 in that integer representation. A
        // signed-64-bit entry count adds at most 63 carry bits, so 35 limbs provide a
        // checked exact accumulator with margin. Canonical duplicate sums are rounded
        // once, ties-to-even, and exact zero is normalized to +0.
        class ExactBinary64Sum
        {
          public:
            void add(double value)
            {
                const std::uint64_t bits = std::bit_cast<std::uint64_t>(value);
                const std::uint64_t exponent = (bits >> 52) & 0x7ffULL;
                const std::uint64_t fraction = bits & ((1ULL << 52) - 1);
                if (exponent == 0x7ffULL)
                    fail("nonfinite value reached exact duplicate accumulator");
                const std::uint64_t significand = exponent == 0 ? fraction : fraction | (1ULL << 52);
                if (significand == 0)
                    return;
                const unsigned shift = exponent == 0 ? 0U : static_cast<unsigned>(exponent - 1);
                Magnitude term{};
                const std::size_t limb = shift / 64;
                const unsigned offset = shift % 64;
                term[limb] = significand << offset;
                if (offset != 0)
                    term[limb + 1] = significand >> (64 - offset);
                const bool negative = (bits >> 63) != 0;
                if (is_zero(magnitude_)) {
                    magnitude_ = term;
                    negative_ = negative;
                } else if (negative_ == negative) {
                    add_magnitude(magnitude_, term);
                } else {
                    const int comparison = compare_magnitude(magnitude_, term);
                    if (comparison == 0) {
                        magnitude_.fill(0);
                        negative_ = false;
                    } else if (comparison > 0) {
                        subtract_magnitude(magnitude_, term);
                    } else {
                        Magnitude result = term;
                        subtract_magnitude(result, magnitude_);
                        magnitude_ = result;
                        negative_ = negative;
                    }
                }
            }

            std::uint64_t rounded_bits() const
            {
                if (is_zero(magnitude_))
                    return 0;
                int highest = -1;
                for (std::size_t index = magnitude_.size(); index-- > 0;) {
                    if (magnitude_[index] != 0) {
                        highest = static_cast<int>(index * 64 + (63 - std::countl_zero(magnitude_[index])));
                        break;
                    }
                }
                if (highest < 0)
                    return 0;

                std::uint64_t magnitude_bits = 0;
                if (highest <= 51) {
                    magnitude_bits = magnitude_[0];
                } else {
                    unsigned shift = static_cast<unsigned>(highest - 52);
                    std::uint64_t significand = shifted_low_u64(shift);
                    if (shift != 0) {
                        const bool half = bit_is_set(shift - 1);
                        const bool below_half = any_bits_below(shift - 1);
                        if (half && (below_half || (significand & 1ULL) != 0))
                            ++significand;
                    }
                    if (significand == (1ULL << 53)) {
                        significand >>= 1;
                        ++highest;
                    }
                    const int exponent_bits = highest - 51;
                    if (exponent_bits >= 0x7ff)
                        fail("duplicate COO sum overflows binary64");
                    magnitude_bits =
                        (static_cast<std::uint64_t>(exponent_bits) << 52) | (significand & ((1ULL << 52) - 1));
                }
                return magnitude_bits | (negative_ ? (1ULL << 63) : 0ULL);
            }

          private:
            using Magnitude = std::array<std::uint64_t, 35>;

            static bool is_zero(const Magnitude& value)
            {
                return std::all_of(value.begin(), value.end(), [](std::uint64_t word) { return word == 0; });
            }

            static int compare_magnitude(const Magnitude& left, const Magnitude& right)
            {
                for (std::size_t index = left.size(); index-- > 0;) {
                    if (left[index] != right[index])
                        return left[index] < right[index] ? -1 : 1;
                }
                return 0;
            }

            static void add_magnitude(Magnitude& left, const Magnitude& right)
            {
                std::uint64_t carry = 0;
                for (std::size_t index = 0; index < left.size(); ++index) {
                    const std::uint64_t first = left[index] + right[index];
                    const bool first_carry = first < left[index];
                    const std::uint64_t second = first + carry;
                    const bool second_carry = second < first;
                    left[index] = second;
                    carry = (first_carry || second_carry) ? 1 : 0;
                }
                if (carry != 0)
                    fail("exact duplicate accumulator overflow");
            }

            static void subtract_magnitude(Magnitude& left, const Magnitude& right)
            {
                std::uint64_t borrow = 0;
                for (std::size_t index = 0; index < left.size(); ++index) {
                    const std::uint64_t original = left[index];
                    const std::uint64_t first = original - right[index];
                    const bool first_borrow = original < right[index];
                    const std::uint64_t second = first - borrow;
                    const bool second_borrow = first < borrow;
                    left[index] = second;
                    borrow = (first_borrow || second_borrow) ? 1 : 0;
                }
                if (borrow != 0)
                    fail("internal exact duplicate accumulator underflow");
            }

            std::uint64_t shifted_low_u64(unsigned shift) const
            {
                const std::size_t limb = shift / 64;
                const unsigned offset = shift % 64;
                std::uint64_t value = magnitude_[limb] >> offset;
                if (offset != 0 && limb + 1 < magnitude_.size())
                    value |= magnitude_[limb + 1] << (64 - offset);
                return value;
            }

            bool bit_is_set(unsigned bit) const { return (magnitude_[bit / 64] & (1ULL << (bit % 64))) != 0; }

            bool any_bits_below(unsigned bit_count) const
            {
                const std::size_t full_limbs = bit_count / 64;
                for (std::size_t index = 0; index < full_limbs; ++index) {
                    if (magnitude_[index] != 0)
                        return true;
                }
                const unsigned remaining = bit_count % 64;
                if (remaining == 0)
                    return false;
                const std::uint64_t mask = (1ULL << remaining) - 1;
                return (magnitude_[full_limbs] & mask) != 0;
            }

            Magnitude magnitude_{};
            bool negative_ = false;
        };

        struct ColumnSpan {
            std::size_t begin = 0;
            std::size_t count = 0;
        };

        template <typename Callback>
        void for_each_canonical_coordinate(const CapturedLinearSystemView& system, Callback&& callback)
        {
            std::vector<ColumnSpan> spans(checked_size(system.columns, "column spans"));
            std::size_t begin = 0;
            while (begin < system.values.size()) {
                const int column_1based = system.column_indices_1based[begin];
                std::size_t end = begin + 1;
                while (end < system.values.size() && system.column_indices_1based[end] == column_1based)
                    ++end;
                ColumnSpan& span = spans[static_cast<std::size_t>(column_1based - 1)];
                if (span.count != 0)
                    fail("COO column reappears after its contiguous input group");
                span = {begin, end - begin};
                begin = end;
            }

            std::vector<std::size_t> order;
            for (std::size_t column = 0; column < spans.size(); ++column) {
                const ColumnSpan span = spans[column];
                if (span.count == 0)
                    continue;
                order.resize(span.count);
                std::iota(order.begin(), order.end(), span.begin);
                std::sort(order.begin(), order.end(), [&](std::size_t left, std::size_t right) {
                    const int left_row = system.row_indices_1based[left];
                    const int right_row = system.row_indices_1based[right];
                    return left_row != right_row ? left_row < right_row : left < right;
                });
                std::size_t group_begin = 0;
                while (group_begin < order.size()) {
                    const int row_1based = system.row_indices_1based[order[group_begin]];
                    ExactBinary64Sum sum;
                    std::size_t group_end = group_begin;
                    while (group_end < order.size() && system.row_indices_1based[order[group_end]] == row_1based) {
                        sum.add(system.values[order[group_end]]);
                        ++group_end;
                    }
                    callback(static_cast<std::uint32_t>(column + 1), static_cast<std::uint32_t>(row_1based),
                             static_cast<std::uint64_t>(group_end - group_begin), sum.rounded_bits());
                    group_begin = group_end;
                }
            }
        }

        bool valid_row_taxonomy(CapturedRowTaxonomy taxonomy)
        {
            const auto code = static_cast<std::int32_t>(taxonomy);
            return code >= static_cast<std::int32_t>(CapturedRowTaxonomy::Unknown) &&
                   code <= static_cast<std::int32_t>(CapturedRowTaxonomy::GlobalInt);
        }

        bool valid_column_class(CapturedColumnClass column_class)
        {
            const auto code = static_cast<std::int32_t>(column_class);
            return code >= static_cast<std::int32_t>(CapturedColumnClass::Unknown) &&
                   code <= static_cast<std::int32_t>(CapturedColumnClass::ScalarGlobal);
        }

        bool field_column_class(CapturedColumnClass column_class)
        {
            const auto code = static_cast<std::int32_t>(column_class);
            return code >= static_cast<std::int32_t>(CapturedColumnClass::FieldUnknown) &&
                   code <= static_cast<std::int32_t>(CapturedColumnClass::FieldGauge);
        }

        template <typename T>
        void validate_strict_map(std::span<const T> map, std::uint64_t full_count, std::string_view name)
        {
            std::uint64_t previous = 0;
            for (std::size_t index = 0; index < map.size(); ++index) {
                const std::uint64_t value = map[index];
                if (value >= full_count)
                    fail(std::string(name) + " contains an out-of-range full index");
                if (index != 0 && value <= previous)
                    fail(std::string(name) + " must be strictly increasing");
                previous = value;
            }
        }

        void validate_permutation(std::span<const int> permutation, std::uint64_t dimension)
        {
            if (permutation.size() != dimension)
                fail("analysis permutation length does not match matrix dimension");
            std::vector<bool> seen(checked_size(dimension, "permutation"), false);
            for (int position : permutation) {
                if (position < 1 || static_cast<std::uint64_t>(position) > dimension)
                    fail("analysis permutation contains an out-of-range position");
                const std::size_t index = static_cast<std::size_t>(position - 1);
                if (seen[index])
                    fail("analysis permutation is not a bijection");
                seen[index] = true;
            }
        }

        void validate_view(const CapturedLinearSystemView& system, std::uint32_t schema_version)
        {
            static_assert(sizeof(int) == sizeof(std::int32_t));
            static_assert(sizeof(double) == sizeof(std::uint64_t));
            if (!std::numeric_limits<double>::is_iec559)
                fail("host double is not IEEE-754 binary64");
            if (system.rows == 0 || system.columns == 0 || system.rows != system.columns)
                fail("captured matrix must be nonempty and square");
            if (system.rows > static_cast<std::uint64_t>(INT_MAX) ||
                system.columns > static_cast<std::uint64_t>(INT_MAX) ||
                system.full_rows > static_cast<std::uint64_t>(INT_MAX) ||
                system.full_columns > static_cast<std::uint64_t>(INT_MAX))
                fail("matrix dimensions exceed the supported 32-bit index range");
            if (system.full_rows < system.rows || system.full_columns < system.columns)
                fail("full dimensions are smaller than captured dimensions");

            const std::uint64_t nnz = system.values.size();
            if (nnz == 0)
                fail("captured matrix must contain at least one COO entry");
            if (system.row_indices_1based.size() != nnz || system.column_indices_1based.size() != nnz)
                fail("COO row, column, and value lengths differ");
            if (system.rhs.size() != system.rows)
                fail("RHS length does not match matrix rows");
            if (system.row_full_indices_zero_based.size() != system.rows || system.row_tags.size() != system.rows)
                fail("row map/tag length does not match matrix rows");
            if (system.column_full_indices_zero_based.size() != system.columns ||
                system.column_tags.size() != system.columns)
                fail("column map/tag length does not match matrix columns");

            validate_strict_map(system.row_full_indices_zero_based, system.full_rows, "row map");
            validate_strict_map(system.column_full_indices_zero_based, system.full_columns, "column map");
            validate_permutation(system.analysis_permutation_1based, system.columns);

            for (std::size_t entry = 0; entry < system.values.size(); ++entry) {
                const int row = system.row_indices_1based[entry];
                const int column = system.column_indices_1based[entry];
                if (row < 1 || static_cast<std::uint64_t>(row) > system.rows || column < 1 ||
                    static_cast<std::uint64_t>(column) > system.columns)
                    fail("COO contains an out-of-range one-based index");
                if (!std::isfinite(system.values[entry]))
                    fail("COO contains a nonfinite value");
            }
            for (double value : system.rhs) {
                if (!std::isfinite(value))
                    fail("RHS contains a nonfinite value");
            }

            for (std::size_t index = 0; index < system.row_tags.size(); ++index) {
                const CapturedRowTag& tag = system.row_tags[index];
                if (!valid_row_taxonomy(tag.taxonomy) || tag.taxonomy == CapturedRowTaxonomy::Unknown)
                    fail("row tag contains an unknown taxonomy code");
                if (tag.original_row < 0 ||
                    static_cast<std::uint32_t>(tag.original_row) != system.row_full_indices_zero_based[index])
                    fail("row tag original index disagrees with row map");
                if (tag.domain < -1 || tag.domain_pair < -1)
                    fail("row tag contains an invalid domain");
            }
            for (std::size_t index = 0; index < system.column_tags.size(); ++index) {
                const CapturedColumnTag& tag = system.column_tags[index];
                if (!valid_column_class(tag.column_class) || tag.column_class == CapturedColumnClass::Unknown)
                    fail("column tag contains an unknown class code");
                if (tag.original_column < 0 ||
                    static_cast<std::uint32_t>(tag.original_column) != system.column_full_indices_zero_based[index])
                    fail("column tag original index disagrees with column map");
                if (tag.incidence_role < -1 || tag.incidence_role > 1 || tag.domain < -1 || tag.term_idx < -1 ||
                    tag.var_idx < -1 || tag.var_double_idx < -1 || tag.vardom_param < -1 || tag.basis_mode < -1)
                    fail("column tag contains an invalid numeric field");
                const std::array<std::int32_t, 8> semantics{{
                    tag.domain_type_id, tag.tensor_component, tag.coefficient_i, tag.coefficient_j,
                    tag.coefficient_k, tag.coefficient_nr, tag.coefficient_nt, tag.coefficient_np,
                }};
                if (std::any_of(semantics.begin(), semantics.end(), [](std::int32_t value) { return value < -1; }))
                    fail("column tag contains an invalid schema-v2 semantic field");
                if (schema_version == kLegacySchemaVersion) {
                    if (std::any_of(semantics.begin(), semantics.end(), [](std::int32_t value) { return value != -1; }))
                        fail("schema-v1 column tag contains schema-v2 semantics");
                } else if (field_column_class(tag.column_class)) {
                    if (tag.domain_type_id < 1 || tag.tensor_component < 0 || tag.coefficient_i < 0 ||
                        tag.coefficient_j < 0 || tag.coefficient_k < 0 || tag.coefficient_nr < 1 ||
                        tag.coefficient_nt < 1 || tag.coefficient_np < 1)
                        fail("schema-v2 field column tag has unknown mode semantics");
                    if (tag.coefficient_i >= tag.coefficient_nr || tag.coefficient_j >= tag.coefficient_nt ||
                        tag.coefficient_k >= tag.coefficient_np)
                        fail("schema-v2 field column tag has an out-of-range mode coordinate");
                } else if (std::any_of(semantics.begin(), semantics.end(),
                                       [](std::int32_t value) { return value != -1; })) {
                    fail("schema-v2 non-field column tag contains field mode semantics");
                }
            }

            if (system.analysis.communicator_size < 1 || system.analysis.analysis_rank_count < 1 ||
                system.analysis.analysis_rank_count > system.analysis.communicator_size ||
                system.analysis.factor_ranks_per_node < 0)
                fail("analysis settings contain invalid rank counts");
            if (system.analysis.icntl[13] < 1 || system.analysis.successful_factor_icntl14 < 1 ||
                system.analysis.factor_retry_count < 0 || system.analysis.factor_retry_count > 40)
                fail("analysis settings contain invalid factor-workspace provenance");
            std::int32_t expected_factor_icntl14 = system.analysis.icntl[13];
            for (std::int32_t retry = 0; retry < system.analysis.factor_retry_count; ++retry) {
                const std::int32_t increase = expected_factor_icntl14 / 2;
                if (increase > std::numeric_limits<std::int32_t>::max() - expected_factor_icntl14)
                    fail("factor ICNTL(14) retry history overflows int32");
                expected_factor_icntl14 += increase;
            }
            if (expected_factor_icntl14 != system.analysis.successful_factor_icntl14)
                fail("successful factor ICNTL(14) disagrees with its seed and retry count");
            for (double value : system.analysis.cntl) {
                if (!std::isfinite(value))
                    fail("analysis CNTL settings contain a nonfinite value");
            }
            bool version_terminated = false;
            for (char character : system.analysis.mumps_version) {
                const auto byte = static_cast<unsigned char>(character);
                if (version_terminated) {
                    if (byte != 0)
                        fail("MUMPS version must be zero-padded after its terminator");
                } else if (byte == 0) {
                    version_terminated = true;
                } else if (byte < 0x20 || byte > 0x7e) {
                    fail("MUMPS version contains a non-printable ASCII byte");
                }
            }
            if (system.analysis.mumps_version[0] == '\0' || !version_terminated)
                fail("MUMPS version must be a nonempty, null-terminated string");

            (void)expected_file_bytes(schema_version, system.rows, system.columns, nnz);
        }

        CapturedLinearSystemHashes semantic_hashes(const CapturedLinearSystemView& system)
        {
            Sha256 rhs_hash;
            hash_domain(rhs_hash, "KADATH-KCSR-RHS-V1");
            hash_u64(rhs_hash, system.rhs.size());
            for (double value : system.rhs)
                hash_u64(rhs_hash, std::bit_cast<std::uint64_t>(value));

            Sha256 pattern_hash;
            hash_domain(pattern_hash, "KADATH-KCSR-PATTERN-MULTISET-V1");
            hash_u64(pattern_hash, system.rows);
            hash_u64(pattern_hash, system.columns);
            hash_u64(pattern_hash, system.values.size());
            Sha256 canonical_value_hash;
            hash_domain(canonical_value_hash, "KADATH-KCSR-CANONICAL-VALUES-V1");
            hash_u64(canonical_value_hash, system.rows);
            hash_u64(canonical_value_hash, system.columns);
            hash_u64(canonical_value_hash, system.values.size());
            std::uint64_t unique_nnz = 0;
            for_each_canonical_coordinate(system, [&](std::uint32_t column, std::uint32_t row,
                                                      std::uint64_t multiplicity, std::uint64_t coalesced_bits) {
                hash_u32(pattern_hash, column);
                hash_u32(pattern_hash, row);
                hash_u64(pattern_hash, multiplicity);
                hash_u32(canonical_value_hash, column);
                hash_u32(canonical_value_hash, row);
                hash_u64(canonical_value_hash, multiplicity);
                hash_u64(canonical_value_hash, coalesced_bits);
                ++unique_nnz;
            });
            hash_u64(pattern_hash, unique_nnz);
            hash_u64(canonical_value_hash, unique_nnz);

            CapturedLinearSystemHashes hashes;
            hashes.ordered_matrix = captured_ordered_matrix_hash(system.rows, system.columns, system.row_indices_1based,
                                                                 system.column_indices_1based, system.values);
            hashes.rhs = rhs_hash.finish();
            hashes.canonical_pattern = pattern_hash.finish();
            hashes.canonical_values = canonical_value_hash.finish();
            return hashes;
        }

        class AtomicFile
        {
          public:
            explicit AtomicFile(const std::filesystem::path& target) : target_(target)
            {
                const std::filesystem::path parent =
                    target.parent_path().empty() ? std::filesystem::path{"."} : target.parent_path();
                static std::atomic<unsigned long long> sequence{0};
                for (int attempt = 0; attempt < 100; ++attempt) {
                    temporary_ = parent / ("." + target.filename().string() + ".tmp." +
                                           std::to_string(static_cast<long long>(::getpid())) + "." +
                                           std::to_string(sequence.fetch_add(1)));
                    int flags = O_WRONLY | O_CREAT | O_EXCL;
#ifdef O_CLOEXEC
                    flags |= O_CLOEXEC;
#endif
                    fd_ = ::open(temporary_.c_str(), flags, 0600);
                    if (fd_ >= 0)
                        return;
                    if (errno != EEXIST)
                        fail("cannot create temporary file '" + temporary_.string() +
                             "': " + std::system_category().message(errno));
                }
                fail("could not allocate an exclusive temporary filename");
            }

            AtomicFile(const AtomicFile&) = delete;
            AtomicFile& operator=(const AtomicFile&) = delete;

            ~AtomicFile()
            {
                if (fd_ >= 0)
                    ::close(fd_);
                if (!committed_) {
                    std::error_code ignored;
                    std::filesystem::remove(temporary_, ignored);
                }
            }

            void write_prefix(const void* data, std::size_t size)
            {
                write_raw(data, size);
                archive_hash_.update(data, size);
            }

            void write_tail(const void* data, std::size_t size) { write_raw(data, size); }

            void write_prefix_u32(std::uint32_t value)
            {
                const auto bytes = encode_u32(value);
                write_prefix(bytes.data(), bytes.size());
            }

            void write_prefix_i32(std::int32_t value) { write_prefix_u32(std::bit_cast<std::uint32_t>(value)); }

            void write_prefix_u64(std::uint64_t value)
            {
                const auto bytes = encode_u64(value);
                write_prefix(bytes.data(), bytes.size());
            }

            CapturedLinearSystemHash finish_archive_hash() { return archive_hash_.finish(); }

            void commit()
            {
                flush_buffer();
                if (::fsync(fd_) != 0)
                    fail("fsync failed for temporary file: " + std::system_category().message(errno));
                if (::close(fd_) != 0) {
                    fd_ = -1;
                    fail("close failed for temporary file: " + std::system_category().message(errno));
                }
                fd_ = -1;
                if (::link(temporary_.c_str(), target_.c_str()) != 0)
                    fail("atomic no-replace publish failed for '" + target_.string() +
                         "': " + std::system_category().message(errno));
                if (::unlink(temporary_.c_str()) != 0)
                    fail_after_publish("cannot unlink published temporary file: " +
                                       std::system_category().message(errno));

                const std::filesystem::path parent =
                    target_.parent_path().empty() ? std::filesystem::path{"."} : target_.parent_path();
                int directory_flags = O_RDONLY;
#ifdef O_DIRECTORY
                directory_flags |= O_DIRECTORY;
#endif
#ifdef O_CLOEXEC
                directory_flags |= O_CLOEXEC;
#endif
                const int directory_fd = ::open(parent.c_str(), directory_flags);
                if (directory_fd < 0)
                    fail_after_publish("cannot open output directory for fsync: " +
                                       std::system_category().message(errno));
                if (::fsync(directory_fd) != 0) {
                    const std::string message =
                        "output directory fsync failed: " + std::system_category().message(errno);
                    (void)::close(directory_fd);
                    fail_after_publish(message);
                }
                if (::close(directory_fd) != 0)
                    fail_after_publish("output directory close failed: " + std::system_category().message(errno));
                committed_ = true;
            }

          private:
            [[noreturn]] void fail_after_publish(const std::string& message)
            {
                if (::unlink(target_.c_str()) != 0)
                    fail(message + "; rollback unlink also failed: " + std::system_category().message(errno));
                fail(message + "; published output was rolled back");
            }

            void flush_buffer()
            {
                std::size_t offset = 0;
                while (offset != output_size_) {
                    const ssize_t written = ::write(fd_, output_buffer_.data() + offset, output_size_ - offset);
                    if (written < 0) {
                        if (errno == EINTR)
                            continue;
                        fail("write failed for temporary file: " + std::system_category().message(errno));
                    }
                    if (written == 0)
                        fail("zero-length write to temporary file");
                    offset += static_cast<std::size_t>(written);
                }
                output_size_ = 0;
            }

            void write_raw(const void* data, std::size_t size)
            {
                const auto* bytes = static_cast<const std::uint8_t*>(data);
                while (size != 0) {
                    if (output_size_ == output_buffer_.size())
                        flush_buffer();
                    const std::size_t count = std::min(size, output_buffer_.size() - output_size_);
                    std::memcpy(output_buffer_.data() + output_size_, bytes, count);
                    output_size_ += count;
                    bytes += count;
                    size -= count;
                }
            }

            std::filesystem::path target_;
            std::filesystem::path temporary_;
            int fd_ = -1;
            bool committed_ = false;
            std::array<std::uint8_t, 1024 * 1024> output_buffer_{};
            std::size_t output_size_ = 0;
            Sha256 archive_hash_;
        };

        void write_header(AtomicFile& file, const CapturedLinearSystemView& system, std::uint64_t total_bytes)
        {
            file.write_prefix(kMagic.data(), kMagic.size());
            file.write_prefix_u32(kSchemaVersion);
            file.write_prefix_u32(kRequiredFlags);
            file.write_prefix_u32(kEndianMarker);
            file.write_prefix_u32(kHeaderBytes);
            file.write_prefix_u64(system.rows);
            file.write_prefix_u64(system.columns);
            file.write_prefix_u64(system.full_rows);
            file.write_prefix_u64(system.full_columns);
            file.write_prefix_u64(system.values.size());
            file.write_prefix_u64(system.rhs.size());
            file.write_prefix_u64(system.row_full_indices_zero_based.size());
            file.write_prefix_u64(system.column_full_indices_zero_based.size());
            file.write_prefix_u64(system.row_tags.size());
            file.write_prefix_u64(system.column_tags.size());
            file.write_prefix_u64(system.analysis_permutation_1based.size());
            file.write_prefix_u64(total_bytes);
            file.write_prefix_i32(system.analysis.requested_ordering);
            file.write_prefix_i32(system.analysis.actual_ordering);
            file.write_prefix_i32(system.analysis.communicator_size);
            file.write_prefix_i32(system.analysis.analysis_rank_count);
            file.write_prefix_i32(system.analysis.factor_ranks_per_node);
            file.write_prefix_i32(system.analysis.successful_factor_icntl14);
            file.write_prefix_i32(system.analysis.factor_retry_count);
            file.write_prefix_i32(0);
            for (std::int32_t value : system.analysis.icntl)
                file.write_prefix_i32(value);
            for (double value : system.analysis.cntl)
                file.write_prefix_u64(std::bit_cast<std::uint64_t>(value));
            file.write_prefix(system.analysis.mumps_version.data(), system.analysis.mumps_version.size());
        }

        class FileReader
        {
          public:
            explicit FileReader(const std::filesystem::path& path) : fd_(open_capture(path)), path_(path)
            {
                if (fd_ < 0)
                    fail("cannot open '" + path.string() + "': " + std::system_category().message(errno));
                struct stat status{};
                if (::fstat(fd_, &status) != 0)
                    fail_open("cannot stat '" + path.string() + "': " + std::system_category().message(errno));
                if (!S_ISREG(status.st_mode))
                    fail_open("capture path is not a regular file");
                if (status.st_size < 0)
                    fail_open("capture file has a negative size");
                size_ = static_cast<std::uint64_t>(status.st_size);
            }

            FileReader(const FileReader&) = delete;
            FileReader& operator=(const FileReader&) = delete;

            ~FileReader()
            {
                if (fd_ >= 0)
                    ::close(fd_);
            }

            std::uint64_t size() const { return size_; }
            std::uint64_t position() const { return position_; }

            void seek(std::uint64_t offset)
            {
                if (offset > static_cast<std::uint64_t>(std::numeric_limits<off_t>::max()))
                    fail("file offset does not fit off_t");
                if (::lseek(fd_, static_cast<off_t>(offset), SEEK_SET) < 0)
                    fail("seek failed: " + std::system_category().message(errno));
                position_ = offset;
            }

            void read_exact(void* destination, std::size_t size)
            {
                auto* bytes = static_cast<std::uint8_t*>(destination);
                while (size != 0) {
                    const ssize_t count = ::read(fd_, bytes, size);
                    if (count < 0) {
                        if (errno == EINTR)
                            continue;
                        fail("read failed: " + std::system_category().message(errno));
                    }
                    if (count == 0)
                        fail("truncated file");
                    bytes += count;
                    size -= static_cast<std::size_t>(count);
                    position_ += static_cast<std::uint64_t>(count);
                }
            }

            std::uint32_t read_u32()
            {
                std::array<std::uint8_t, 4> bytes{};
                read_exact(bytes.data(), bytes.size());
                return decode_u32(bytes.data());
            }

            std::int32_t read_i32() { return std::bit_cast<std::int32_t>(read_u32()); }

            std::uint64_t read_u64()
            {
                std::array<std::uint8_t, 8> bytes{};
                read_exact(bytes.data(), bytes.size());
                return decode_u64(bytes.data());
            }

          private:
            [[noreturn]] void fail_open(const std::string& message)
            {
                (void)::close(fd_);
                fd_ = -1;
                fail(message);
            }

            static int open_capture(const std::filesystem::path& path)
            {
                int flags = O_RDONLY;
#ifdef O_NOFOLLOW
                flags |= O_NOFOLLOW;
#endif
#ifdef O_CLOEXEC
                flags |= O_CLOEXEC;
#endif
                return ::open(path.c_str(), flags);
            }

            int fd_ = -1;
            std::filesystem::path path_;
            std::uint64_t size_ = 0;
            std::uint64_t position_ = 0;
        };

        struct Header {
            std::uint32_t schema_version = 0;
            std::uint64_t rows = 0;
            std::uint64_t columns = 0;
            std::uint64_t full_rows = 0;
            std::uint64_t full_columns = 0;
            std::uint64_t nnz = 0;
            std::uint64_t rhs_count = 0;
            std::uint64_t row_map_count = 0;
            std::uint64_t column_map_count = 0;
            std::uint64_t row_tag_count = 0;
            std::uint64_t column_tag_count = 0;
            std::uint64_t permutation_count = 0;
            std::uint64_t total_bytes = 0;
            CapturedAnalysisSettings analysis;
        };

        Header read_header(FileReader& file)
        {
            std::array<std::uint8_t, 8> magic{};
            file.read_exact(magic.data(), magic.size());
            if (magic != kMagic)
                fail("bad magic");
            const std::uint32_t schema_version = file.read_u32();
            if (schema_version != kLegacySchemaVersion && schema_version != kSchemaVersion)
                fail("unsupported schema version");
            if (file.read_u32() != kRequiredFlags)
                fail("unsupported or missing required flags");
            if (file.read_u32() != kEndianMarker)
                fail("bad endian marker");
            if (file.read_u32() != kHeaderBytes)
                fail("unexpected header size");

            Header header;
            header.schema_version = schema_version;
            header.rows = file.read_u64();
            header.columns = file.read_u64();
            header.full_rows = file.read_u64();
            header.full_columns = file.read_u64();
            header.nnz = file.read_u64();
            header.rhs_count = file.read_u64();
            header.row_map_count = file.read_u64();
            header.column_map_count = file.read_u64();
            header.row_tag_count = file.read_u64();
            header.column_tag_count = file.read_u64();
            header.permutation_count = file.read_u64();
            header.total_bytes = file.read_u64();
            header.analysis.requested_ordering = file.read_i32();
            header.analysis.actual_ordering = file.read_i32();
            header.analysis.communicator_size = file.read_i32();
            header.analysis.analysis_rank_count = file.read_i32();
            header.analysis.factor_ranks_per_node = file.read_i32();
            header.analysis.successful_factor_icntl14 = file.read_i32();
            header.analysis.factor_retry_count = file.read_i32();
            if (file.read_i32() != 0)
                fail("nonzero reserved header field");
            for (std::int32_t& value : header.analysis.icntl)
                value = file.read_i32();
            for (double& value : header.analysis.cntl)
                value = std::bit_cast<double>(file.read_u64());
            file.read_exact(header.analysis.mumps_version.data(), header.analysis.mumps_version.size());
            if (file.position() != kHeaderBytes)
                fail("internal header size mismatch");
            return header;
        }

        void validate_header(const Header& header, std::uint64_t actual_bytes)
        {
            if (header.rows == 0 || header.columns == 0 || header.rows != header.columns)
                fail("captured matrix dimensions are invalid");
            if (header.rows > static_cast<std::uint64_t>(INT_MAX) ||
                header.columns > static_cast<std::uint64_t>(INT_MAX) ||
                header.full_rows > static_cast<std::uint64_t>(INT_MAX) ||
                header.full_columns > static_cast<std::uint64_t>(INT_MAX))
                fail("captured dimensions exceed the supported index range");
            if (header.full_rows < header.rows || header.full_columns < header.columns)
                fail("captured full dimensions are invalid");
            if (header.nnz == 0)
                fail("captured matrix contains no entries");
            if (header.rhs_count != header.rows || header.row_map_count != header.rows ||
                header.row_tag_count != header.rows || header.column_map_count != header.columns ||
                header.column_tag_count != header.columns || header.permutation_count != header.columns)
                fail("captured section counts do not match dimensions");
            const std::uint64_t expected =
                expected_file_bytes(header.schema_version, header.rows, header.columns, header.nnz);
            if (header.total_bytes != expected)
                fail("declared file size does not match section counts");
            if (actual_bytes < expected)
                fail("truncated file");
            if (actual_bytes > expected)
                fail("trailing bytes after capture footer");
            (void)checked_size(header.nnz, "COO");
        }

        CapturedLinearSystemHash verify_archive_hash(FileReader& file, const Header& header)
        {
            const std::uint64_t archive_offset = header.total_bytes - kFooterBytes - kArchiveHashBytes;
            file.seek(0);
            Sha256 hash;
            std::array<std::uint8_t, 1024 * 1024> buffer{};
            std::uint64_t remaining = archive_offset;
            while (remaining != 0) {
                const std::size_t count = static_cast<std::size_t>(std::min<std::uint64_t>(remaining, buffer.size()));
                file.read_exact(buffer.data(), count);
                hash.update(buffer.data(), count);
                remaining -= count;
            }
            CapturedLinearSystemHash stored{};
            file.read_exact(stored.data(), stored.size());
            std::array<std::uint8_t, 8> footer{};
            file.read_exact(footer.data(), footer.size());
            if (footer != kFooter)
                fail("bad footer magic");
            if (hash.finish() != stored)
                fail("archive hash mismatch");
            return stored;
        }

        CapturedLinearSystemView owned_view(const CapturedLinearSystem& system)
        {
            CapturedLinearSystemView view;
            view.rows = system.rows;
            view.columns = system.columns;
            view.full_rows = system.full_rows;
            view.full_columns = system.full_columns;
            view.row_indices_1based = system.row_indices_1based;
            view.column_indices_1based = system.column_indices_1based;
            view.values = system.values;
            view.rhs = system.rhs;
            view.row_full_indices_zero_based = system.row_full_indices_zero_based;
            view.column_full_indices_zero_based = system.column_full_indices_zero_based;
            view.row_tags = system.row_tags;
            view.column_tags = system.column_tags;
            view.analysis_permutation_1based = system.analysis_permutation_1based;
            view.analysis = system.analysis;
            return view;
        }

        class ScaledNumber
        {
          public:
            static ScaledNumber from(double value)
            {
                ScaledNumber result;
                if (value != 0.0)
                    result.mantissa_ = std::frexp(static_cast<long double>(value), &result.exponent_);
                return result;
            }

            static ScaledNumber product(double left, double right)
            {
                if (left == 0.0 || right == 0.0)
                    return {};
                int left_exponent = 0;
                int right_exponent = 0;
                const long double left_mantissa = std::frexp(static_cast<long double>(left), &left_exponent);
                const long double right_mantissa = std::frexp(static_cast<long double>(right), &right_exponent);
                ScaledNumber result;
                result.mantissa_ = left_mantissa * right_mantissa;
                result.exponent_ = left_exponent + right_exponent;
                result.normalize();
                return result;
            }

            static ScaledNumber product(const ScaledNumber& left, const ScaledNumber& right)
            {
                if (left.is_zero() || right.is_zero())
                    return {};
                ScaledNumber result;
                result.mantissa_ = left.mantissa_ * right.mantissa_;
                result.exponent_ = left.exponent_ + right.exponent_;
                result.normalize();
                return result;
            }

            bool is_zero() const { return mantissa_ == 0.0L; }

            ScaledNumber absolute() const
            {
                ScaledNumber result = *this;
                result.mantissa_ = std::abs(result.mantissa_);
                return result;
            }

            ScaledNumber negated() const
            {
                ScaledNumber result = *this;
                result.mantissa_ = -result.mantissa_;
                return result;
            }

            void add(const ScaledNumber& other)
            {
                if (other.is_zero())
                    return;
                if (is_zero()) {
                    *this = other;
                    return;
                }
                constexpr int insignificant = std::numeric_limits<long double>::digits + 4;
                if (exponent_ >= other.exponent_) {
                    const int difference = exponent_ - other.exponent_;
                    if (difference <= insignificant)
                        mantissa_ += std::scalbn(other.mantissa_, -difference);
                } else {
                    const int difference = other.exponent_ - exponent_;
                    if (difference > insignificant) {
                        *this = other;
                        return;
                    }
                    mantissa_ = std::scalbn(mantissa_, -difference) + other.mantissa_;
                    exponent_ = other.exponent_;
                }
                normalize();
            }

            int compare_nonnegative(const ScaledNumber& other) const
            {
                if (is_zero())
                    return other.is_zero() ? 0 : -1;
                if (other.is_zero())
                    return 1;
                if (exponent_ != other.exponent_)
                    return exponent_ < other.exponent_ ? -1 : 1;
                if (mantissa_ == other.mantissa_)
                    return 0;
                return mantissa_ < other.mantissa_ ? -1 : 1;
            }

            double to_double() const
            {
                if (is_zero())
                    return 0.0;
                const long double value = std::scalbn(mantissa_, exponent_);
                if (!std::isfinite(value) ||
                    std::abs(value) > static_cast<long double>(std::numeric_limits<double>::max()))
                    return std::copysign(std::numeric_limits<double>::infinity(), static_cast<double>(mantissa_));
                return static_cast<double>(value);
            }

            static double nonnegative_ratio(const ScaledNumber& numerator, const ScaledNumber& denominator)
            {
                if (denominator.is_zero())
                    return numerator.is_zero() ? 0.0 : std::numeric_limits<double>::infinity();
                if (numerator.is_zero())
                    return 0.0;
                const long double mantissa = numerator.mantissa_ / denominator.mantissa_;
                const int exponent = numerator.exponent_ - denominator.exponent_;
                const long double value = std::scalbn(mantissa, exponent);
                if (!std::isfinite(value) || value > static_cast<long double>(std::numeric_limits<double>::max()))
                    return std::numeric_limits<double>::infinity();
                return static_cast<double>(value);
            }

          private:
            void normalize()
            {
                if (mantissa_ == 0.0L) {
                    exponent_ = 0;
                    return;
                }
                int adjustment = 0;
                mantissa_ = std::frexp(mantissa_, &adjustment);
                exponent_ += adjustment;
            }

            long double mantissa_ = 0.0L;
            int exponent_ = 0;
        };

    } // namespace

    void validate_captured_linear_system_collective_request(MPI_Comm communicator, std::string_view path, int ordinal,
                                                            bool sparse_analyze_reuse)
    {
        const auto mpi_check = [](int code, std::string_view operation) {
            if (code == MPI_SUCCESS)
                return;
            std::array<char, MPI_MAX_ERROR_STRING> message{};
            int length = 0;
            MPI_Error_string(code, message.data(), &length);
            fail(std::string(operation) + " failed: " + std::string(message.data(), static_cast<std::size_t>(length)));
        };

        int rank = 0;
        mpi_check(MPI_Comm_rank(communicator, &rank), "MPI_Comm_rank");
        const int enabled = path.empty() ? 0 : 1;
        const std::array<int, 2> local_enabled{{enabled, -enabled}};
        std::array<int, 2> enabled_range{};
        mpi_check(MPI_Allreduce(local_enabled.data(), enabled_range.data(), static_cast<int>(enabled_range.size()),
                                MPI_INT, MPI_MIN, communicator),
                  "MPI_Allreduce(capture enable range)");
        const int minimum_enabled = enabled_range[0];
        const int maximum_enabled = -enabled_range[1];
        if (minimum_enabled != maximum_enabled)
            fail("capture enable state differs across MPI ranks");
        if (maximum_enabled == 0)
            return;

        const int path_fits = path.size() <= static_cast<std::size_t>(std::numeric_limits<int>::max()) ? 1 : 0;
        const long long wide_ordinal = ordinal;
        const long long reuse = sparse_analyze_reuse ? 1 : 0;
        const long long path_size = path_fits != 0 ? static_cast<long long>(path.size()) : -1;
        const std::array<long long, 7> local_request{
            {wide_ordinal, -wide_ordinal, path_fits, reuse, -reuse, path_size, -path_size}};
        std::array<long long, 7> request_range{};
        mpi_check(MPI_Allreduce(local_request.data(), request_range.data(), static_cast<int>(request_range.size()),
                                MPI_LONG_LONG, MPI_MIN, communicator),
                  "MPI_Allreduce(capture request range)");
        const long long minimum_ordinal = request_range[0];
        const long long maximum_ordinal = -request_range[1];
        if (minimum_ordinal <= 0 || minimum_ordinal != maximum_ordinal) {
            fail("capture ordinal must be the same positive integer on every MPI rank");
        }
        if (request_range[2] == 0)
            fail("capture path exceeds MPI count range");
        const long long minimum_reuse = request_range[3];
        const long long maximum_reuse = -request_range[4];
        if (minimum_reuse != maximum_reuse)
            fail("sparse analyze-reuse state differs across MPI ranks while capture is enabled");
        if (maximum_reuse != 0) {
            fail("direct replay capture is incompatible with sparse analyze reuse because its cached superset is "
                 "not the exact factor input");
        }

        const long long minimum_path_size = request_range[5];
        const long long maximum_path_size = -request_range[6];
        if (minimum_path_size != maximum_path_size)
            fail("capture path length differs across MPI ranks");

        int local_matches = 1;
        std::array<char, 4096> root_path_chunk{};
        std::size_t path_offset = 0;
        while (path_offset < path.size()) {
            const int count = static_cast<int>(std::min(path.size() - path_offset, root_path_chunk.size()));
            if (rank == 0)
                std::memcpy(root_path_chunk.data(), path.data() + path_offset, static_cast<std::size_t>(count));
            mpi_check(MPI_Bcast(root_path_chunk.data(), count, MPI_CHAR, 0, communicator),
                      "MPI_Bcast(capture path chunk)");
            if (std::memcmp(path.data() + path_offset, root_path_chunk.data(), static_cast<std::size_t>(count)) != 0)
                local_matches = 0;
            path_offset += static_cast<std::size_t>(count);
        }
        int every_path_matches = 0;
        mpi_check(MPI_Allreduce(&local_matches, &every_path_matches, 1, MPI_INT, MPI_MIN, communicator),
                  "MPI_Allreduce(capture path match)");
        if (every_path_matches == 0)
            fail("capture path differs across MPI ranks");
    }

    CapturedLinearSystemHash captured_ordered_matrix_hash(std::uint64_t rows, std::uint64_t columns,
                                                          std::span<const int> row_indices_1based,
                                                          std::span<const int> column_indices_1based,
                                                          std::span<const double> values)
    {
        if (rows == 0 || columns == 0 || rows > static_cast<std::uint64_t>(INT_MAX) ||
            columns > static_cast<std::uint64_t>(INT_MAX))
            fail("ordered matrix hash requires supported nonempty dimensions");
        if (row_indices_1based.size() != values.size() || column_indices_1based.size() != values.size())
            fail("ordered matrix hash COO lengths differ");

        Sha256 matrix_hash;
        hash_domain(matrix_hash, "KADATH-KCSR-ORDERED-MATRIX-V1");
        hash_u64(matrix_hash, rows);
        hash_u64(matrix_hash, columns);
        hash_u64(matrix_hash, values.size());
        for (std::size_t entry = 0; entry < values.size(); ++entry) {
            const int row = row_indices_1based[entry];
            const int column = column_indices_1based[entry];
            if (row < 1 || static_cast<std::uint64_t>(row) > rows || column < 1 ||
                static_cast<std::uint64_t>(column) > columns)
                fail("ordered matrix hash COO index lies outside its dimensions");
            if (!std::isfinite(values[entry]))
                fail("ordered matrix hash COO contains a nonfinite value");
            hash_u32(matrix_hash, static_cast<std::uint32_t>(row));
            hash_u32(matrix_hash, static_cast<std::uint32_t>(column));
            hash_u64(matrix_hash, std::bit_cast<std::uint64_t>(values[entry]));
        }
        return matrix_hash.finish();
    }

    CapturedLinearSystemHashes write_captured_linear_system(const std::filesystem::path& path,
                                                            const CapturedLinearSystemView& system)
    {
        if (path.empty() || path.extension() != ".kcsr")
            fail("output path must have a .kcsr extension");
        std::error_code status_error;
        const std::filesystem::file_status status = std::filesystem::symlink_status(path, status_error);
        if (status_error && status_error != std::errc::no_such_file_or_directory)
            fail("cannot inspect output path: " + status_error.message());
        if (!status_error && status.type() != std::filesystem::file_type::not_found)
            fail("refusing to overwrite existing output path");

        validate_view(system, kSchemaVersion);
        CapturedLinearSystemHashes hashes = semantic_hashes(system);
        const std::uint64_t total_bytes =
            expected_file_bytes(kSchemaVersion, system.rows, system.columns, system.values.size());
        AtomicFile file(path);
        write_header(file, system, total_bytes);

        for (std::size_t entry = 0; entry < system.values.size(); ++entry) {
            file.write_prefix_i32(system.row_indices_1based[entry]);
            file.write_prefix_i32(system.column_indices_1based[entry]);
            file.write_prefix_u64(std::bit_cast<std::uint64_t>(system.values[entry]));
        }
        for (double value : system.rhs)
            file.write_prefix_u64(std::bit_cast<std::uint64_t>(value));
        for (std::uint32_t value : system.row_full_indices_zero_based)
            file.write_prefix_u32(value);
        for (std::uint32_t value : system.column_full_indices_zero_based)
            file.write_prefix_u32(value);
        for (const CapturedRowTag& tag : system.row_tags) {
            file.write_prefix_i32(tag.original_row);
            file.write_prefix_i32(static_cast<std::int32_t>(tag.taxonomy));
            file.write_prefix_i32(tag.domain);
            file.write_prefix_i32(tag.domain_pair);
        }
        for (const CapturedColumnTag& tag : system.column_tags) {
            file.write_prefix_i32(tag.original_column);
            file.write_prefix_i32(static_cast<std::int32_t>(tag.column_class));
            file.write_prefix_i32(tag.incidence_role);
            file.write_prefix_i32(tag.domain);
            file.write_prefix_i32(tag.term_idx);
            file.write_prefix_i32(tag.var_idx);
            file.write_prefix_i32(tag.var_double_idx);
            file.write_prefix_i32(tag.vardom_param);
            file.write_prefix_i32(tag.basis_mode);
            file.write_prefix_u64(tag.var_name_hash);
            file.write_prefix_i32(tag.domain_type_id);
            file.write_prefix_i32(tag.tensor_component);
            file.write_prefix_i32(tag.coefficient_i);
            file.write_prefix_i32(tag.coefficient_j);
            file.write_prefix_i32(tag.coefficient_k);
            file.write_prefix_i32(tag.coefficient_nr);
            file.write_prefix_i32(tag.coefficient_nt);
            file.write_prefix_i32(tag.coefficient_np);
        }
        for (int value : system.analysis_permutation_1based)
            file.write_prefix_i32(value);
        file.write_prefix(hashes.ordered_matrix.data(), hashes.ordered_matrix.size());
        file.write_prefix(hashes.rhs.data(), hashes.rhs.size());
        file.write_prefix(hashes.canonical_pattern.data(), hashes.canonical_pattern.size());
        file.write_prefix(hashes.canonical_values.data(), hashes.canonical_values.size());
        hashes.archive = file.finish_archive_hash();
        file.write_tail(hashes.archive.data(), hashes.archive.size());
        file.write_tail(kFooter.data(), kFooter.size());
        file.commit();
        return hashes;
    }

    CapturedLinearSystem read_captured_linear_system(const std::filesystem::path& path)
    {
        if (path.empty() || path.extension() != ".kcsr")
            fail("input path must have a .kcsr extension");
        std::error_code status_error;
        const std::filesystem::file_status status = std::filesystem::symlink_status(path, status_error);
        if (status_error)
            fail("cannot inspect input path: " + status_error.message());
        if (status.type() == std::filesystem::file_type::symlink)
            fail("refusing to read a symbolic-link capture path");
        if (status.type() != std::filesystem::file_type::regular)
            fail("capture path is not a regular file");

        FileReader file(path);
        Header header = read_header(file);
        validate_header(header, file.size());
        const CapturedLinearSystemHash archive_hash = verify_archive_hash(file, header);

        file.seek(0);
        header = read_header(file);
        CapturedLinearSystem system;
        system.schema_version = header.schema_version;
        system.rows = header.rows;
        system.columns = header.columns;
        system.full_rows = header.full_rows;
        system.full_columns = header.full_columns;
        system.analysis = header.analysis;
        const std::size_t nnz = checked_size(header.nnz, "COO");
        const std::size_t rows = checked_size(header.rows, "rows");
        const std::size_t columns = checked_size(header.columns, "columns");
        system.row_indices_1based.resize(nnz);
        system.column_indices_1based.resize(nnz);
        system.values.resize(nnz);
        system.rhs.resize(rows);
        system.row_full_indices_zero_based.resize(rows);
        system.column_full_indices_zero_based.resize(columns);
        system.row_tags.resize(rows);
        system.column_tags.resize(columns);
        system.analysis_permutation_1based.resize(columns);

        for (std::size_t entry = 0; entry < nnz; ++entry) {
            system.row_indices_1based[entry] = file.read_i32();
            system.column_indices_1based[entry] = file.read_i32();
            system.values[entry] = std::bit_cast<double>(file.read_u64());
        }
        for (double& value : system.rhs)
            value = std::bit_cast<double>(file.read_u64());
        for (std::uint32_t& value : system.row_full_indices_zero_based)
            value = file.read_u32();
        for (std::uint32_t& value : system.column_full_indices_zero_based)
            value = file.read_u32();
        for (CapturedRowTag& tag : system.row_tags) {
            tag.original_row = file.read_i32();
            tag.taxonomy = static_cast<CapturedRowTaxonomy>(file.read_i32());
            tag.domain = file.read_i32();
            tag.domain_pair = file.read_i32();
        }
        for (CapturedColumnTag& tag : system.column_tags) {
            tag.original_column = file.read_i32();
            tag.column_class = static_cast<CapturedColumnClass>(file.read_i32());
            tag.incidence_role = file.read_i32();
            tag.domain = file.read_i32();
            tag.term_idx = file.read_i32();
            tag.var_idx = file.read_i32();
            tag.var_double_idx = file.read_i32();
            tag.vardom_param = file.read_i32();
            tag.basis_mode = file.read_i32();
            tag.var_name_hash = file.read_u64();
            if (header.schema_version >= kSchemaVersion) {
                tag.domain_type_id = file.read_i32();
                tag.tensor_component = file.read_i32();
                tag.coefficient_i = file.read_i32();
                tag.coefficient_j = file.read_i32();
                tag.coefficient_k = file.read_i32();
                tag.coefficient_nr = file.read_i32();
                tag.coefficient_nt = file.read_i32();
                tag.coefficient_np = file.read_i32();
            }
        }
        for (int& value : system.analysis_permutation_1based)
            value = file.read_i32();
        file.read_exact(system.hashes.ordered_matrix.data(), system.hashes.ordered_matrix.size());
        file.read_exact(system.hashes.rhs.data(), system.hashes.rhs.size());
        file.read_exact(system.hashes.canonical_pattern.data(), system.hashes.canonical_pattern.size());
        file.read_exact(system.hashes.canonical_values.data(), system.hashes.canonical_values.size());
        file.read_exact(system.hashes.archive.data(), system.hashes.archive.size());
        std::array<std::uint8_t, 8> footer{};
        file.read_exact(footer.data(), footer.size());
        if (footer != kFooter || file.position() != file.size())
            fail("capture footer or final length is invalid");
        if (system.hashes.archive != archive_hash)
            fail("archive hash changed between verification and decode");

        const CapturedLinearSystemView view = owned_view(system);
        validate_view(view, system.schema_version);
        const CapturedLinearSystemHashes computed = semantic_hashes(view);
        if (computed.ordered_matrix != system.hashes.ordered_matrix)
            fail("ordered matrix hash mismatch");
        if (computed.rhs != system.hashes.rhs)
            fail("RHS hash mismatch");
        if (computed.canonical_pattern != system.hashes.canonical_pattern)
            fail("canonical pattern hash mismatch");
        if (computed.canonical_values != system.hashes.canonical_values)
            fail("canonical value hash mismatch");
        return system;
    }

    CapturedBackwardError scale_aware_backward_error(const CapturedLinearSystem& system,
                                                     std::span<const double> solution)
    {
        if (system.rows == 0 || system.rows != system.columns || system.rows > static_cast<std::uint64_t>(INT_MAX))
            fail("backward error requires a nonempty square matrix");
        if (solution.size() != system.columns)
            fail("solution length does not match matrix columns");
        if (system.row_indices_1based.size() != system.values.size() ||
            system.column_indices_1based.size() != system.values.size() || system.rhs.size() != system.rows ||
            system.row_full_indices_zero_based.size() != system.rows)
            fail("backward-error matrix storage is inconsistent");
        for (double value : solution) {
            if (!std::isfinite(value))
                fail("solution contains a nonfinite value");
        }
        for (double value : system.rhs) {
            if (!std::isfinite(value))
                fail("RHS contains a nonfinite value");
        }

        const std::size_t rows = checked_size(system.rows, "backward-error rows");
        for (std::size_t entry = 0; entry < system.values.size(); ++entry) {
            const int row = system.row_indices_1based[entry];
            const int column = system.column_indices_1based[entry];
            if (row < 1 || static_cast<std::uint64_t>(row) > system.rows || column < 1 ||
                static_cast<std::uint64_t>(column) > system.columns || !std::isfinite(system.values[entry]))
                fail("backward-error COO contains an invalid entry");
        }

        std::vector<ScaledNumber> product(rows);
        std::vector<ScaledNumber> row_norm(rows);
        std::vector<ScaledNumber> component_scale(rows);
        CapturedLinearSystemView view;
        view.rows = system.rows;
        view.columns = system.columns;
        view.row_indices_1based = system.row_indices_1based;
        view.column_indices_1based = system.column_indices_1based;
        view.values = system.values;
        for_each_canonical_coordinate(view, [&](std::uint32_t column_1based, std::uint32_t row_1based, std::uint64_t,
                                                std::uint64_t coalesced_bits) {
            const double coefficient = std::bit_cast<double>(coalesced_bits);
            const std::size_t row = static_cast<std::size_t>(row_1based - 1);
            const std::size_t column = static_cast<std::size_t>(column_1based - 1);
            product[row].add(ScaledNumber::product(coefficient, solution[column]));
            row_norm[row].add(ScaledNumber::from(std::abs(coefficient)));
            component_scale[row].add(ScaledNumber::product(std::abs(coefficient), std::abs(solution[column])));
        });

        ScaledNumber matrix_inf;
        ScaledNumber solution_inf;
        ScaledNumber rhs_inf;
        ScaledNumber residual_inf;
        double componentwise_value = 0.0;
        std::size_t worst_row = 0;
        for (const ScaledNumber& value : row_norm) {
            if (value.compare_nonnegative(matrix_inf) > 0)
                matrix_inf = value;
        }
        for (double value : solution) {
            const ScaledNumber magnitude = ScaledNumber::from(std::abs(value));
            if (magnitude.compare_nonnegative(solution_inf) > 0)
                solution_inf = magnitude;
        }
        for (std::size_t row = 0; row < rows; ++row) {
            const ScaledNumber rhs = ScaledNumber::from(system.rhs[row]);
            const ScaledNumber rhs_magnitude = rhs.absolute();
            if (rhs_magnitude.compare_nonnegative(rhs_inf) > 0)
                rhs_inf = rhs_magnitude;
            ScaledNumber residual = rhs;
            residual.add(product[row].negated());
            residual = residual.absolute();
            if (residual.compare_nonnegative(residual_inf) > 0)
                residual_inf = residual;
            ScaledNumber component_denominator = rhs_magnitude;
            component_denominator.add(component_scale[row]);
            const double component_error = ScaledNumber::nonnegative_ratio(residual, component_denominator);
            if (component_error > componentwise_value) {
                componentwise_value = component_error;
                worst_row = row;
            }
        }
        ScaledNumber denominator = ScaledNumber::product(matrix_inf, solution_inf);
        denominator.add(rhs_inf);
        const double error = ScaledNumber::nonnegative_ratio(residual_inf, denominator);

        CapturedBackwardError result;
        result.residual_inf = residual_inf.to_double();
        result.matrix_inf = matrix_inf.to_double();
        result.solution_inf = solution_inf.to_double();
        result.rhs_inf = rhs_inf.to_double();
        result.denominator = denominator.to_double();
        result.value = error;
        result.componentwise_value = componentwise_value;
        result.worst_factor_row_zero_based = worst_row;
        result.worst_full_row_zero_based = system.row_full_indices_zero_based[worst_row];
        return result;
    }

    std::string captured_linear_system_hash_hex(const CapturedLinearSystemHash& hash)
    {
        constexpr char digits[] = "0123456789abcdef";
        std::string output;
        output.resize(hash.size() * 2);
        for (std::size_t index = 0; index < hash.size(); ++index) {
            output[2 * index] = digits[hash[index] >> 4];
            output[2 * index + 1] = digits[hash[index] & 0x0f];
        }
        return output;
    }

} // namespace Kadath
