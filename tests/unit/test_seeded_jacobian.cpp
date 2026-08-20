#include <catch2/catch_test_macros.hpp>

#include "For_Kadath/System_of_eqs/system_of_eqs.hpp"

using namespace Kadath;

TEST_CASE("seeded Jacobian public diagnostics stay available", "[seeded-jacobian]")
{
    using SeededColumns =
        void (System_of_eqs::*)(const std::vector<int>&, std::vector<Array<double>>&);
    using FallbackBucketValidation =
        bool (System_of_eqs::*)(std::ostream&, int, int, int, double, double, double);
    using SeededCooValidation =
        bool (System_of_eqs::*)(std::ostream&, int, int, int, double, double, double);
    using DumpSeededProfile =
        void (System_of_eqs::*)() const;

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif
    SeededColumns do_seeded_columns = &System_of_eqs::do_cols_J_seeded;
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#endif
    FallbackBucketValidation validate_fallback_bucket =
        &System_of_eqs::validate_fallback_coloring_bucket;
    SeededCooValidation validate_seeded_entries =
        &System_of_eqs::validate_seeded_coo_equivalence;
    DumpSeededProfile dump_profile = &System_of_eqs::dump_do_cols_J_seeded_profile;

    CHECK(do_seeded_columns != nullptr);
    CHECK(validate_fallback_bucket != nullptr);
    CHECK(validate_seeded_entries != nullptr);
    CHECK(dump_profile != nullptr);
}
