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

#include "Linear_algebra/mumps_analysis_hints.hpp"

#include <catch2/matchers/catch_matchers_string.hpp>
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <numeric>
#include <vector>

namespace
{
    Kadath::MumpsAnalysisPatternView make_view(int n, const std::vector<int>& irn, const std::vector<int>& jcn,
                                               const std::vector<Kadath::MumpsAnalysisRowTag>& rows,
                                               const std::vector<Kadath::MumpsAnalysisColumnTag>& columns)
    {
        return {n, static_cast<long long>(irn.size()), irn.data(), jcn.data(), rows, columns};
    }

    Kadath::MumpsAnalysisColumnTag column(int original, Kadath::ColumnClass type, int domain, int role)
    {
        Kadath::MumpsAnalysisColumnTag result;
        result.original_column = original;
        result.column_class = type;
        result.domain = domain;
        result.incidence_role = role;
        result.var_idx = original;
        result.basis_mode = original;
        return result;
    }

    Kadath::MumpsAnalysisRowTag row(int original, Kadath::RowTaxonomy taxonomy, int domain, int pair = -1)
    {
        return {original, taxonomy, domain, pair};
    }
} // namespace

TEST_CASE("Topology ordering is deterministic and keeps bulk before seams and globals", "[topology-ordering]")
{
    // Two bulk columns (domains 0,1), one matching column, one global. The
    // matching row couples both domains through the actual captured incidence.
    const std::vector<int> irn{1, 1, 2, 2, 3, 3, 3, 4};
    const std::vector<int> jcn{1, 3, 2, 3, 1, 2, 3, 4};
    const std::vector<Kadath::MumpsAnalysisRowTag> rows{
        row(0, Kadath::RowTaxonomy::Vol, 0), row(1, Kadath::RowTaxonomy::Vol, 1),
        row(2, Kadath::RowTaxonomy::TauMatch, 0, 1), row(3, Kadath::RowTaxonomy::GlobalInt, -1)};
    const std::vector<Kadath::MumpsAnalysisColumnTag> columns{
        column(0, Kadath::ColumnClass::FieldInteriorVol, 0, 0), column(1, Kadath::ColumnClass::FieldInteriorVol, 1, 0),
        column(2, Kadath::ColumnClass::FieldMatching, 0, 1), column(3, Kadath::ColumnClass::ScalarGlobal, -1, 1)};

    const auto hints = Kadath::build_topology_ordering_hints(make_view(4, irn, jcn, rows, columns));
    CHECK(hints.permutation_1based[0] < hints.permutation_1based[2]);
    CHECK(hints.permutation_1based[1] < hints.permutation_1based[2]);
    CHECK(hints.permutation_1based[2] < hints.permutation_1based[3]);
    CHECK(hints.bulk_columns == 2);
    CHECK(hints.seam_columns == 1);
    CHECK(hints.global_columns == 1);
    CHECK(hints.unresolved_matching_rows == 0);

    std::vector<int> reverse_irn(irn.rbegin(), irn.rend());
    std::vector<int> reverse_jcn(jcn.rbegin(), jcn.rend());
    const auto reordered = Kadath::build_topology_ordering_hints(make_view(4, reverse_irn, reverse_jcn, rows, columns));
    CHECK(reordered.permutation_1based == hints.permutation_1based);
    CHECK(reordered.domain_elimination_order == hints.domain_elimination_order);
}

TEST_CASE("Topology quotient minimum fill peels chain leaves deterministically", "[topology-ordering]")
{
    const std::vector<int> irn{1, 1, 2, 2, 3, 3};
    const std::vector<int> jcn{1, 2, 2, 3, 3, 4};
    const std::vector<Kadath::MumpsAnalysisRowTag> rows{
        row(0, Kadath::RowTaxonomy::TauMatch, 0, 1), row(1, Kadath::RowTaxonomy::TauMatch, 1, 2),
        row(2, Kadath::RowTaxonomy::Vol, 2), row(3, Kadath::RowTaxonomy::Vol, 2)};
    const std::vector<Kadath::MumpsAnalysisColumnTag> columns{
        column(0, Kadath::ColumnClass::FieldMatching, 0, 1), column(1, Kadath::ColumnClass::FieldMatching, 1, 1),
        column(2, Kadath::ColumnClass::FieldMatching, 2, 1), column(3, Kadath::ColumnClass::FieldInteriorVol, 2, 0)};
    const auto hints = Kadath::build_topology_ordering_hints(make_view(4, irn, jcn, rows, columns));
    REQUIRE(hints.domain_elimination_order.size() == 3);
    CHECK(hints.domain_elimination_order.front() == 0);
}

TEST_CASE("Application block hints group only exact closed-neighbourhood twins",
          "[topology-ordering][mumps-block-analysis]")
{
    // Columns/rows 1 and 2 are true twins: each closed adjacency is {1,2,3}.
    // Variable 3 has a distinct closed adjacency and remains a singleton.
    const std::vector<int> irn{1, 1, 1, 2, 2, 2, 3, 3, 3};
    const std::vector<int> jcn{1, 2, 3, 1, 2, 3, 1, 2, 3};
    const std::vector<Kadath::MumpsAnalysisRowTag> rows{row(0, Kadath::RowTaxonomy::Vol, 0),
                                                        row(1, Kadath::RowTaxonomy::Vol, 0),
                                                        row(2, Kadath::RowTaxonomy::TauBc, 0)};
    const std::vector<Kadath::MumpsAnalysisColumnTag> columns{column(0, Kadath::ColumnClass::FieldInteriorVol, 0, 0),
                                                              column(1, Kadath::ColumnClass::FieldInteriorVol, 0, 0),
                                                              column(2, Kadath::ColumnClass::FieldBoundaryTau, 0, 1)};
    const auto view = make_view(3, irn, jcn, rows, columns);
    const auto topology = Kadath::build_topology_ordering_hints(view);
    const auto blocks = Kadath::build_exact_topology_blocks(view, topology);
    CHECK(blocks.blkptr_1based.front() == 1);
    CHECK(blocks.blkptr_1based.back() == 4);
    CHECK(blocks.blkvar_1based.size() == 3);
    CHECK(blocks.nonsingleton_blocks == 1);
    CHECK(blocks.compressed_variables == 1);
}

TEST_CASE("Symbolic diagnostics expose etree and singleton front histogram",
          "[topology-ordering][symbolic-elimination]")
{
    const std::vector<int> irn{1, 2, 2, 3, 3, 4};
    const std::vector<int> jcn{1, 1, 2, 2, 3, 4};
    std::vector<Kadath::MumpsAnalysisRowTag> rows;
    std::vector<Kadath::MumpsAnalysisColumnTag> columns;
    for (int i = 0; i < 4; ++i) {
        rows.push_back(row(i, Kadath::RowTaxonomy::Vol, 0));
        columns.push_back(column(i, Kadath::ColumnClass::FieldInteriorVol, 0, 0));
    }
    const std::vector<int> identity{1, 2, 3, 4};
    const auto diagnostics = Kadath::analyze_symbolic_elimination(make_view(4, irn, jcn, rows, columns), identity);
    CHECK(diagnostics.parent_1based == std::vector<int>{2, 3, 0, 0});
    CHECK(diagnostics.max_tree_depth == 2);
    CHECK(diagnostics.filled_lower_nnz_proxy == 6);
    CHECK(diagnostics.max_singleton_front_width == 2);
    CHECK(diagnostics.parent_hash != 0);
    CHECK(std::accumulate(diagnostics.depth_histogram.begin(), diagnostics.depth_histogram.end(), std::uint64_t{0}) ==
          4);
}

TEST_CASE("Topology and symbolic builders refuse malformed input", "[topology-ordering][failure]")
{
    const std::vector<int> irn{0};
    const std::vector<int> jcn{1};
    const std::vector<Kadath::MumpsAnalysisRowTag> rows{row(0, Kadath::RowTaxonomy::Vol, 0)};
    const std::vector<Kadath::MumpsAnalysisColumnTag> columns{column(0, Kadath::ColumnClass::FieldInteriorVol, 0, 0)};
    CHECK_THROWS_AS(Kadath::build_topology_ordering_hints(make_view(1, irn, jcn, rows, columns)), std::out_of_range);

    const std::vector<int> valid_irn{1};
    const std::vector<int> duplicate_permutation{1, 1};
    std::vector<Kadath::MumpsAnalysisRowTag> rows2{row(0, Kadath::RowTaxonomy::Vol, 0),
                                                   row(1, Kadath::RowTaxonomy::Vol, 0)};
    std::vector<Kadath::MumpsAnalysisColumnTag> columns2{column(0, Kadath::ColumnClass::FieldInteriorVol, 0, 0),
                                                         column(1, Kadath::ColumnClass::FieldInteriorVol, 0, 0)};
    CHECK_THROWS_AS(Kadath::analyze_symbolic_elimination(make_view(2, valid_irn, valid_irn, rows2, columns2),
                                                         duplicate_permutation),
                    std::invalid_argument);

    columns2[0].incidence_role = 2;
    CHECK_THROWS_WITH(Kadath::build_topology_ordering_hints(make_view(2, valid_irn, valid_irn, rows2, columns2)),
                      Catch::Matchers::ContainsSubstring("incidence role"));

    columns2[0].incidence_role = 0;
    columns2[0].basis_mode = -2;
    CHECK_THROWS_WITH(Kadath::build_topology_ordering_hints(make_view(2, valid_irn, valid_irn, rows2, columns2)),
                      Catch::Matchers::ContainsSubstring("column numeric fields"));

    columns2[0].basis_mode = 0;
    rows2[0].domain_pair = -2;
    CHECK_THROWS_WITH(Kadath::build_topology_ordering_hints(make_view(2, valid_irn, valid_irn, rows2, columns2)),
                      Catch::Matchers::ContainsSubstring("row domains"));
}

TEST_CASE("Topology handles a domainless scalar-only system", "[topology-ordering][edge]")
{
    const std::vector<int> irn{1};
    const std::vector<int> jcn{1};
    const std::vector<Kadath::MumpsAnalysisRowTag> rows{row(0, Kadath::RowTaxonomy::GlobalInt, -1)};
    const std::vector<Kadath::MumpsAnalysisColumnTag> columns{column(0, Kadath::ColumnClass::ScalarGlobal, -1, 1)};
    const auto view = make_view(1, irn, jcn, rows, columns);
    const auto topology = Kadath::build_topology_ordering_hints(view);
    CHECK(topology.permutation_1based == std::vector<int>{1});
    CHECK(topology.domain_elimination_order.empty());
    CHECK(topology.global_columns == 1);

    const auto blocks = Kadath::build_exact_topology_blocks(view, topology);
    CHECK(blocks.blkptr_1based == std::vector<int>{1, 2});
    CHECK(blocks.blkvar_1based == std::vector<int>{1});
}

TEST_CASE("Topology reports unresolved matching rows without inventing endpoints", "[topology-ordering][edge]")
{
    const std::vector<int> irn{1, 2};
    const std::vector<int> jcn{1, 2};
    const std::vector<Kadath::MumpsAnalysisRowTag> rows{row(0, Kadath::RowTaxonomy::TauMatch, 0),
                                                        row(1, Kadath::RowTaxonomy::Vol, 1)};
    const std::vector<Kadath::MumpsAnalysisColumnTag> columns{column(0, Kadath::ColumnClass::FieldMatching, 0, 1),
                                                              column(1, Kadath::ColumnClass::FieldInteriorVol, 1, 0)};

    const auto topology = Kadath::build_topology_ordering_hints(make_view(2, irn, jcn, rows, columns));
    CHECK(topology.unresolved_matching_rows == 1);
    CHECK(topology.permutation_1based.size() == 2);
}

TEST_CASE("Topology deterministically orders an empty disconnected pattern", "[topology-ordering][edge]")
{
    const std::vector<int> irn;
    const std::vector<int> jcn;
    const std::vector<Kadath::MumpsAnalysisRowTag> rows{row(4, Kadath::RowTaxonomy::Vol, 0),
                                                        row(7, Kadath::RowTaxonomy::Vol, 1)};
    const std::vector<Kadath::MumpsAnalysisColumnTag> columns{column(8, Kadath::ColumnClass::FieldInteriorVol, 0, 0),
                                                              column(2, Kadath::ColumnClass::FieldInteriorVol, 1, 0)};

    const auto view = make_view(2, irn, jcn, rows, columns);
    const auto first = Kadath::build_topology_ordering_hints(view);
    const auto second = Kadath::build_topology_ordering_hints(view);
    CHECK(first.permutation_1based == second.permutation_1based);
    CHECK(first.domain_elimination_order == std::vector<int>{0, 1});
    CHECK(first.permutation_1based == std::vector<int>{1, 2});

    const auto symbolic = Kadath::analyze_symbolic_elimination(view, first.permutation_1based);
    CHECK(symbolic.parent_1based == std::vector<int>{0, 0});
    CHECK(symbolic.filled_lower_nnz_proxy == 2);
}
