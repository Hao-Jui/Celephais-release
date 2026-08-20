/*
    Copyright 2026 Kadath contributors

    This file is part of Kadath.

    Kadath is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    Kadath is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with Kadath. If not, see <http://www.gnu.org/licenses/>.
*/

#include "Linear_algebra/mumps_prolongation.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <cstddef>
#include <cstdint>
#include <vector>

namespace
{
    using Kadath::CapturedColumnClass;
    using Kadath::CapturedColumnTag;
    using Kadath::ProlongationOrderingObservation;
    using Kadath::ProlongationSemanticColumn;

    CapturedColumnTag field_tag(std::int32_t original_column,
                                std::int32_t coefficient_i,
                                std::int32_t coefficient_nr)
    {
        CapturedColumnTag tag;
        tag.original_column = original_column;
        tag.column_class = CapturedColumnClass::FieldInteriorVol;
        tag.incidence_role = 0;
        tag.domain = 2;
        tag.term_idx = 3;
        tag.var_idx = 1;
        tag.basis_mode = original_column + 40;
        tag.var_name_hash = 0x91ab;
        tag.domain_type_id = 7;
        tag.tensor_component = 2;
        tag.coefficient_i = coefficient_i;
        tag.coefficient_j = 0;
        tag.coefficient_k = 0;
        tag.coefficient_nr = coefficient_nr;
        tag.coefficient_nt = 1;
        tag.coefficient_np = 1;
        return tag;
    }

    CapturedColumnTag scalar_tag(std::int32_t original_column)
    {
        CapturedColumnTag tag;
        tag.original_column = original_column;
        tag.column_class = CapturedColumnClass::ScalarGlobal;
        tag.incidence_role = 1;
        tag.domain = -1;
        tag.term_idx = 8;
        tag.var_idx = -1;
        tag.var_double_idx = 0;
        tag.basis_mode = -1;
        tag.var_name_hash = 0x5ca1a;
        return tag;
    }

    std::vector<ProlongationSemanticColumn>
    semantics(const std::vector<CapturedColumnTag>& tags,
              const char* role = "synthetic archive")
    {
        return Kadath::extract_prolongation_semantics(
            Kadath::kProlongationSemanticSchemaVersion, tags, role);
    }

    std::vector<CapturedColumnTag> field_tags(std::int32_t count,
                                              std::int32_t coefficient_nr)
    {
        std::vector<CapturedColumnTag> tags;
        tags.reserve(static_cast<std::size_t>(count));
        for (std::int32_t mode = 0; mode < count; ++mode)
            tags.push_back(field_tag(mode, mode, coefficient_nr));
        return tags;
    }

    ProlongationOrderingObservation observation(
        double resolution,
        const std::vector<CapturedColumnTag>& tags,
        std::vector<int> permutation)
    {
        return {resolution, semantics(tags), std::move(permutation)};
    }
} // namespace

TEST_CASE("Schema-v2 prolongation keys use exact FIELD semantics and refuse v1",
          "[mumps-prolongation][semantics]")
{
    auto field = field_tag(17, 2, 5);
    field.coefficient_j = 1;
    field.coefficient_k = 3;
    field.coefficient_nt = 4;
    field.coefficient_np = 6;
    const std::vector<CapturedColumnTag> tags{
        field, scalar_tag(21), scalar_tag(29)};

    const auto columns = semantics(tags, "coarse archive");
    REQUIRE(columns.size() == 3);
    CHECK(columns[0].key.is_field);
    CHECK(columns[0].key.domain_type_id == 7);
    CHECK(columns[0].key.tensor_component == 2);
    CHECK(columns[0].key.coefficient_i == 2);
    CHECK(columns[0].key.coefficient_j == 1);
    CHECK(columns[0].key.coefficient_k == 3);
    CHECK(columns[0].coefficient_nr == 5);
    CHECK(columns[0].coefficient_nt == 4);
    CHECK(columns[0].coefficient_np == 6);
    CHECK_FALSE(columns[1].key.is_field);
    CHECK(columns[1].key.ordinal_within_group == 0);
    CHECK(columns[2].key.ordinal_within_group == 1);
    CHECK(Kadath::infer_prolongation_resolution(columns) == Catch::Approx(6.0));

    REQUIRE_THROWS_WITH(
        Kadath::extract_prolongation_semantics(1, tags, "coarse archive"),
        Catch::Matchers::ContainsSubstring("coarse archive has schema v1"));
}

TEST_CASE("Ensemble prolongation distinguishes stable drift and non-monotone keys",
          "[mumps-prolongation][ensemble]")
{
    // The same five exact semantic keys occur at three synthetic resolutions.
    // Their rank fractions are, respectively:
    //   stable 0; increasing .25/.5/.75; decreasing .75/.25/.25;
    //   non-monotone .5/.75/.5; stable 1.
    const auto tags = field_tags(5, 5);
    const std::vector<ProlongationOrderingObservation> coarse{
        observation(4.0, tags, {1, 2, 4, 3, 5}),
        observation(6.0, tags, {1, 3, 2, 4, 5}),
        observation(8.0, tags, {1, 4, 2, 3, 5}),
    };
    const auto target = semantics(tags, "fine archive");

    Kadath::ProlongationOptions options;
    options.stable_rank_spread_tolerance = 0.01;
    const auto result =
        Kadath::build_prolonged_order(coarse, 10.0, target, options);
    const auto repeated =
        Kadath::build_prolonged_order(coarse, 10.0, target, options);

    REQUIRE(result.predicted_rank_fractions.size() == 5);
    CHECK(result.predicted_rank_fractions[0] == Catch::Approx(0.0));
    CHECK(result.predicted_rank_fractions[1] > 0.75);
    CHECK(result.predicted_rank_fractions[1] <= 1.0);
    CHECK(result.predicted_rank_fractions[2] < 0.25);
    CHECK(result.predicted_rank_fractions[2] >= 0.0);
    CHECK(result.predicted_rank_fractions[3] == Catch::Approx(0.5));
    CHECK(result.predicted_rank_fractions[4] == Catch::Approx(1.0));
    CHECK(result.diagnostics.exact_semantic_hits == 5);
    CHECK(result.diagnostics.interpolated_columns == 0);
    CHECK(result.diagnostics.stable_keys == 2);
    CHECK(result.diagnostics.monotone_increasing_keys == 1);
    CHECK(result.diagnostics.monotone_decreasing_keys == 1);
    CHECK(result.diagnostics.nonmonotone_keys == 1);
    CHECK(result.diagnostics.observed_rank_spread_keys == 5);
    CHECK(result.diagnostics.mean_observed_rank_spread ==
          Catch::Approx(0.25));
    CHECK(result.diagnostics.maximum_observed_rank_spread ==
          Catch::Approx(0.5));
    CHECK(result.permutation_1based == repeated.permutation_1based);
    CHECK(result.predicted_rank_fractions == repeated.predicted_rank_fractions);
    CHECK_NOTHROW(Kadath::validate_prolonged_permutation(
        result.permutation_1based));
}

TEST_CASE("New FIELD modes use deterministic nearest-neighbor ranks and non-field bands",
          "[mumps-prolongation][interpolation]")
{
    // Coarse FIELD modes are i=0 and i=2 at nr=3. Fine i=1 at nr=5 is
    // equidistant from both in normalized coordinates, so lexical key order
    // selects i=0. The extra scalar is placed within the observed scalar band.
    std::vector<CapturedColumnTag> coarse_tags{
        field_tag(0, 0, 3), field_tag(2, 2, 3), scalar_tag(3), scalar_tag(4)};
    std::vector<CapturedColumnTag> target_tags{
        field_tag(10, 0, 5), field_tag(11, 1, 5), field_tag(12, 2, 5),
        scalar_tag(13), scalar_tag(14), scalar_tag(15)};
    const std::vector<ProlongationOrderingObservation> coarse{
        observation(3.0, coarse_tags, {2, 1, 3, 4}),
        observation(5.0, coarse_tags, {2, 1, 3, 4}),
    };
    const auto target = semantics(target_tags, "fine archive");

    const auto result = Kadath::build_prolonged_order(coarse, 7.0, target);
    const auto repeated = Kadath::build_prolonged_order(coarse, 7.0, target);

    REQUIRE(result.predicted_rank_fractions.size() == 6);
    CHECK(result.predicted_rank_fractions[0] == Catch::Approx(1.0 / 3.0));
    CHECK(result.predicted_rank_fractions[1] ==
          Catch::Approx(result.predicted_rank_fractions[0]));
    CHECK(result.predicted_rank_fractions[2] == Catch::Approx(0.0));
    CHECK(result.predicted_rank_fractions[3] == Catch::Approx(2.0 / 3.0));
    CHECK(result.predicted_rank_fractions[4] == Catch::Approx(5.0 / 6.0));
    CHECK(result.predicted_rank_fractions[5] == Catch::Approx(1.0));
    CHECK(result.permutation_1based == std::vector<int>{2, 3, 1, 4, 5, 6});
    CHECK(result.permutation_1based == repeated.permutation_1based);
    CHECK(result.diagnostics.exact_semantic_hits == 4);
    CHECK(result.diagnostics.interpolated_columns == 2);
    CHECK_NOTHROW(Kadath::validate_prolonged_permutation(
        result.permutation_1based));
}

TEST_CASE("Prolongation permutation validation enforces a one-based bijection",
          "[mumps-prolongation][failure]")
{
    CHECK_NOTHROW(Kadath::validate_prolonged_permutation(
        std::vector<int>{3, 1, 2}));
    CHECK_THROWS_WITH(
        Kadath::validate_prolonged_permutation(std::vector<int>{1, 1}),
        Catch::Matchers::ContainsSubstring("not a bijection"));
    CHECK_THROWS_WITH(
        Kadath::validate_prolonged_permutation(std::vector<int>{0, 1}),
        Catch::Matchers::ContainsSubstring("out-of-range"));
}

TEST_CASE("Prolongation recovers zero-hit tail groups by top anchoring",
          "[mumps-prolongation][anchor]")
{
    // Tail coefficients (tau/quotient remainders) sit at i = nr - 2, so
    // their absolute radial indices never coincide across resolutions.
    const auto tail_tag = [](std::int32_t original_column, std::int32_t i,
                             std::int32_t nr) {
        CapturedColumnTag tag = field_tag(original_column, i, nr);
        tag.term_idx = 9;
        return tag;
    };
    const auto tags_at = [&](std::int32_t nr, bool extra_tail) {
        std::vector<CapturedColumnTag> tags{field_tag(0, 0, nr),
                                            field_tag(1, 1, nr),
                                            tail_tag(2, nr - 2, nr)};
        if (extra_tail)
            tags.push_back(tail_tag(3, nr - 1, nr));
        return tags;
    };
    const std::vector<ProlongationOrderingObservation> coarse{
        observation(4.0, tags_at(4, false), {1, 2, 3}),
        observation(6.0, tags_at(6, false), {1, 2, 3}),
    };
    const auto target = semantics(tags_at(8, true), "fine archive");

    const auto result = Kadath::build_prolonged_order(coarse, 8.0, target);
    const auto repeated = Kadath::build_prolonged_order(coarse, 8.0, target);

    REQUIRE(result.predicted_rank_fractions.size() == 4);
    CHECK(result.predicted_rank_fractions[0] == Catch::Approx(0.0));
    CHECK(result.predicted_rank_fractions[1] == Catch::Approx(0.5));
    CHECK(result.predicted_rank_fractions[2] == Catch::Approx(1.0));
    CHECK(result.predicted_rank_fractions[3] == Catch::Approx(1.0));
    CHECK(result.permutation_1based == std::vector<int>{1, 2, 3, 4});
    CHECK(result.permutation_1based == repeated.permutation_1based);
    CHECK(result.diagnostics.exact_semantic_hits == 2);
    CHECK(result.diagnostics.anchor_recovered_groups == 1);
    CHECK(result.diagnostics.anchor_recovered_columns == 1);
    CHECK(result.diagnostics.interpolated_columns == 1);
}

TEST_CASE("Prolongation reports a census of groups absent from every coarse archive",
          "[mumps-prolongation][failure]")
{
    // A foreign domain defeats every widening pass, so the census throw is
    // still reachable.
    const auto foreign_tag = [](std::int32_t original_column) {
        CapturedColumnTag tag = field_tag(original_column, 0, 5);
        tag.var_name_hash = 0xfeed;
        tag.domain = 6;
        return tag;
    };
    const auto tags = field_tags(3, 5);
    const std::vector<ProlongationOrderingObservation> coarse{
        observation(4.0, tags, {1, 2, 3}),
        observation(6.0, tags, {1, 3, 2}),
    };
    auto fine_tags = tags;
    fine_tags.push_back(foreign_tag(3));
    const auto target = semantics(fine_tags, "fine archive");

    CHECK_THROWS_WITH(
        Kadath::build_prolonged_order(coarse, 8.0, target),
        Catch::Matchers::ContainsSubstring("no coarse hit for a new FIELD mode") &&
            Catch::Matchers::ContainsSubstring("var=0xfeed"));
}

TEST_CASE("Prolongation places groups born at the target resolution next to sibling components",
          "[mumps-prolongation][anchor]")
{
    // The fine archive gains a tensor component whose semantic group exists
    // at no coarse resolution; it lands beside the nearest mode of a sibling
    // component of the same variable, domain, and term.
    const auto tags = field_tags(3, 5);
    const std::vector<ProlongationOrderingObservation> coarse{
        observation(4.0, tags, {1, 2, 3}),
        observation(6.0, tags, {1, 2, 3}),
    };
    auto fine_tags = tags;
    CapturedColumnTag born = field_tag(3, 2, 5);
    born.tensor_component = 3;
    fine_tags.push_back(born);
    const auto target = semantics(fine_tags, "fine archive");

    const auto result = Kadath::build_prolonged_order(coarse, 8.0, target);

    REQUIRE(result.predicted_rank_fractions.size() == 4);
    CHECK(result.predicted_rank_fractions[3] ==
          Catch::Approx(result.predicted_rank_fractions[2]));
    CHECK(result.diagnostics.component_widened_columns == 1);
    CHECK(result.diagnostics.interpolated_columns == 1);
    CHECK_NOTHROW(Kadath::validate_prolonged_permutation(
        result.permutation_1based));
}

TEST_CASE("Prolongation places born groups without sibling components by domain and class",
          "[mumps-prolongation][anchor]")
{
    // The born component's class has no sibling within its own variable, so
    // it lands beside the nearest same-class hit of the same domain.
    const auto matching_tag = [](std::int32_t original_column,
                                 std::uint64_t var_name_hash,
                                 std::int32_t term_idx,
                                 std::int32_t tensor_component) {
        CapturedColumnTag tag = field_tag(original_column, 2, 5);
        tag.column_class = CapturedColumnClass::FieldMatching;
        tag.var_name_hash = var_name_hash;
        tag.term_idx = term_idx;
        tag.tensor_component = tensor_component;
        return tag;
    };
    auto tags = field_tags(2, 5);
    tags.push_back(matching_tag(2, 0xbeef, 4, 0));
    const std::vector<ProlongationOrderingObservation> coarse{
        observation(4.0, tags, {1, 2, 3}),
        observation(6.0, tags, {1, 2, 3}),
    };
    auto fine_tags = tags;
    fine_tags.push_back(matching_tag(3, 0x91ab, 3, 3));
    const auto target = semantics(fine_tags, "fine archive");

    const auto result = Kadath::build_prolonged_order(coarse, 8.0, target);

    REQUIRE(result.predicted_rank_fractions.size() == 4);
    CHECK(result.predicted_rank_fractions[3] ==
          Catch::Approx(result.predicted_rank_fractions[2]));
    CHECK(result.diagnostics.domain_widened_columns == 1);
    CHECK(result.diagnostics.component_widened_columns == 0);
    CHECK_NOTHROW(Kadath::validate_prolonged_permutation(
        result.permutation_1based));
}

TEST_CASE("Prolongation defaults rare unplaceable columns to the elimination tail",
          "[mumps-prolongation][anchor]")
{
    // One born column in a foreign domain among ~1500 columns stays under
    // the defaulted-column budget and lands at the tail instead of aborting.
    auto tags = field_tags(1500, 1500);
    const std::vector<int> identity = [] {
        std::vector<int> permutation(1500);
        for (int position = 0; position < 1500; ++position)
            permutation[static_cast<std::size_t>(position)] = position + 1;
        return permutation;
    }();
    const std::vector<ProlongationOrderingObservation> coarse{
        observation(4.0, tags, identity),
        observation(6.0, tags, identity),
    };
    auto fine_tags = tags;
    CapturedColumnTag born = field_tag(1500, 0, 5);
    born.var_name_hash = 0xfeed;
    born.domain = 6;
    fine_tags.push_back(born);
    const auto target = semantics(fine_tags, "fine archive");

    const auto result = Kadath::build_prolonged_order(coarse, 8.0, target);

    REQUIRE(result.predicted_rank_fractions.size() == 1501);
    CHECK(result.predicted_rank_fractions[1500] == Catch::Approx(1.0));
    CHECK(result.diagnostics.defaulted_tail_columns == 1);
    CHECK_NOTHROW(Kadath::validate_prolonged_permutation(
        result.permutation_1based));
}
