#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "../../src/System_of_eqs/coloring_internals.hpp"

using namespace Kadath::coloring_internals;

TEST_CASE("sparse column comparison accepts equivalent support", "[sparse-jacobian-compare]")
{
    const SparseColumnMap reference_values = {
        {1, 1.0},
        {4, -2.0},
    };
    const SparseColumnMap decoded_values = reference_values;

    const SparseColumnComparison comparison =
        compare_sparse_columns(reference_values, decoded_values, 1e-12, 1e-14);

    CHECK(comparison.passed);
    CHECK(comparison.reference_nnz == 2);
    CHECK(comparison.decoded_nnz == 2);
    CHECK(comparison.missing_rows == 0);
    CHECK(comparison.extra_rows == 0);
}

TEST_CASE("sparse column comparison reports value mismatch", "[sparse-jacobian-compare]")
{
    const SparseColumnMap reference_values = {
        {1, 1.0},
        {4, -2.0},
    };
    const SparseColumnMap decoded_values = {
        {1, 1.0},
        {4, -1.5},
    };

    const SparseColumnComparison comparison =
        compare_sparse_columns(reference_values, decoded_values, 1e-12, 1e-14);

    CHECK_FALSE(comparison.passed);
    CHECK(comparison.first_mismatch_row == 4);
    CHECK(comparison.first_reference_value == Catch::Approx(-2.0));
    CHECK(comparison.first_decoded_value == Catch::Approx(-1.5));
    CHECK(comparison.max_abs_error == Catch::Approx(0.5));
}

TEST_CASE("sparse COO comparison reports missing and extra entries", "[sparse-jacobian-compare]")
{
    const SparseCooMap reference_entries = {
        {{1, 2}, 3.0},
        {{5, 8}, 4.0},
    };
    const SparseCooMap seeded_entries = {
        {{1, 2}, 3.0},
        {{7, 9}, -1.0},
    };

    const SparseCooComparison comparison =
        compare_sparse_coo(reference_entries, seeded_entries, 1e-12, 1e-14);

    CHECK_FALSE(comparison.passed);
    CHECK(comparison.reference_nnz == 2);
    CHECK(comparison.seeded_nnz == 2);
    CHECK(comparison.missing_entries == 1);
    CHECK(comparison.extra_entries == 1);
}
