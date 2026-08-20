#include <catch2/catch_test_macros.hpp>

#include "Linear_algebra/jacobian_coo_hash.hpp"

#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

using namespace Kadath;

namespace
{

template <typename T>
void legacy_hash_scalar(std::uint64_t& hash, const T& value)
{
    constexpr std::uint64_t fnv_prime = 1099511628211ULL;
    const auto* bytes = reinterpret_cast<const unsigned char*>(&value);
    for (std::size_t i = 0; i < sizeof(T); ++i) {
        hash ^= bytes[i];
        hash *= fnv_prime;
    }
}

template <std::size_t N>
std::uint64_t legacy_raw_hash(
    int n, long long nnz, double drop_tol, const std::array<int, N>& rows,
    const std::array<int, N>& columns, const std::array<double, N>& values)
{
    std::uint64_t hash = 1469598103934665603ULL;
    legacy_hash_scalar(hash, n);
    legacy_hash_scalar(hash, nnz);
    legacy_hash_scalar(hash, drop_tol);
    for (std::size_t entry = 0; entry < N; ++entry) {
        legacy_hash_scalar(hash, rows[entry]);
        legacy_hash_scalar(hash, columns[entry]);
        legacy_hash_scalar(hash, std::bit_cast<std::uint64_t>(values[entry]));
    }
    return hash;
}

} // namespace

TEST_CASE("canonical Jacobian COO bit hash ignores entry order", "[jacobian-coo-hash]")
{
    const std::array<int, 5> rows_a = {1, 3, 2, 1, 4};
    const std::array<int, 5> cols_a = {1, 1, 2, 3, 3};
    const std::array<double, 5> vals_a = {2.0, -1.0, 4.0, 7.0, -0.0};

    const std::array<int, 5> rows_b = {4, 1, 2, 3, 1};
    const std::array<int, 5> cols_b = {3, 3, 2, 1, 1};
    const std::array<double, 5> vals_b = {-0.0, 7.0, 4.0, -1.0, 2.0};

    const std::uint64_t raw_a = jacobian_coo_bit_hash(4, 5, 1e-14, rows_a.data(), cols_a.data(), vals_a.data());
    const std::uint64_t raw_b = jacobian_coo_bit_hash(4, 5, 1e-14, rows_b.data(), cols_b.data(), vals_b.data());
    const std::uint64_t canonical_a =
        canonical_jacobian_coo_bit_hash(4, 5, 1e-14, rows_a.data(), cols_a.data(), vals_a.data());
    const std::uint64_t canonical_b =
        canonical_jacobian_coo_bit_hash(4, 5, 1e-14, rows_b.data(), cols_b.data(), vals_b.data());

    CHECK(raw_a == legacy_raw_hash(4, 5, 1e-14, rows_a, cols_a, vals_a));
    CHECK(raw_a != raw_b);
    CHECK(canonical_a == canonical_b);
}

TEST_CASE("canonical Jacobian COO bit hash remains value-bit sensitive", "[jacobian-coo-hash]")
{
    const int row = 1;
    const int column = 1;
    const double positive_zero = 0.0;
    const double negative_zero = -0.0;

    CHECK(canonical_jacobian_coo_bit_hash(1, 1, 1e-14, &row, &column, &positive_zero) !=
          canonical_jacobian_coo_bit_hash(1, 1, 1e-14, &row, &column, &negative_zero));
}

TEST_CASE("Jacobian COO bit hashes handle empty input and reject invalid storage",
          "[jacobian-coo-hash]")
{
    CHECK_NOTHROW(jacobian_coo_bit_hash(0, 0, 1e-14, nullptr, nullptr, nullptr));
    CHECK_NOTHROW(canonical_jacobian_coo_bit_hash(
        0, 0, 1e-14, nullptr, nullptr, nullptr));
    CHECK_THROWS_AS(jacobian_coo_bit_hash(
                        1, -1, 1e-14, nullptr, nullptr, nullptr),
                    std::invalid_argument);
    CHECK_THROWS_AS(canonical_jacobian_coo_bit_hash(
                        1, 1, 1e-14, nullptr, nullptr, nullptr),
                    std::invalid_argument);
}

TEST_CASE("canonical Jacobian COO hash rejects order-sensitive duplicate coordinates",
          "[jacobian-coo-hash]")
{
    const std::array<int, 3> rows = {1, 1, 1};
    const std::array<int, 3> columns = {1, 1, 1};
    const std::array<double, 3> first_order = {1e16, -1e16, 1.0};
    const std::array<double, 3> second_order = {1e16, 1.0, -1e16};

    CHECK_THROWS_AS(canonical_jacobian_coo_bit_hash(
                        1, 3, 0.0, rows.data(), columns.data(), first_order.data()),
                    std::invalid_argument);
    CHECK_THROWS_AS(canonical_jacobian_coo_bit_hash(
                        1, 3, 0.0, rows.data(), columns.data(), second_order.data()),
                    std::invalid_argument);
}
