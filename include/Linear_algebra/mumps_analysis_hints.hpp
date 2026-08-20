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

#include "For_Kadath/System_of_eqs/Jacobian/column_types.hpp"
#include "For_Kadath/System_of_eqs/Jacobian/tagged_metadata.hpp"

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace Kadath
{

    // Compact, archive-friendly semantic tags used by the offline MUMPS replay
    // comparator. `incidence_role` follows IncidenceColumnPartition (0=bulk,
    // 1=transfer); -1 asks the builder to derive the role from the captured
    // pattern. Original indices are zero based.
    struct MumpsAnalysisColumnTag {
        int original_column = -1;
        ColumnClass column_class = ColumnClass::Unknown;
        int incidence_role = -1;
        int domain = -1;
        int term_idx = -1;
        int var_idx = -1;
        int var_double_idx = -1;
        int vardom_param = -1;
        int basis_mode = -1;
        std::uint64_t var_name_hash = 0;
    };

    struct MumpsAnalysisRowTag {
        int original_row = -1;
        RowTaxonomy taxonomy = RowTaxonomy::Unknown;
        int domain = -1;
        int domain_pair = -1;
    };

    struct MumpsAnalysisPatternView {
        int n = 0;
        long long nnz = 0;
        const int* irn_1based = nullptr;
        const int* jcn_1based = nullptr;
        std::span<const MumpsAnalysisRowTag> rows;
        std::span<const MumpsAnalysisColumnTag> columns;
    };

    struct TopologyOrderingHints {
        // MUMPS PERM_IN convention: entry i is the 1-based elimination position
        // of original/factor variable i+1.
        std::vector<int> permutation_1based;
        std::vector<int> domain_elimination_order;
        int unresolved_matching_rows = 0;
        int bulk_columns = 0;
        int local_boundary_columns = 0;
        int seam_columns = 0;
        int auxiliary_columns = 0;
        int global_columns = 0;
    };

    // Deterministic parity-neutral topology order for an already-selected square
    // factor system. Domains are ordered by minimum fill on the captured quotient
    // graph; private bulk variables precede local boundaries, seams, auxiliaries,
    // and globals. Malformed or semantically unknown input is refused.
    TopologyOrderingHints build_topology_ordering_hints(const MumpsAnalysisPatternView& pattern);

    struct ExactBlockAnalysisHints {
        // MUMPS ICNTL(15)=1 arrays, both 1 based. Every block is a subset of one
        // semantic topology bucket and has an exactly identical closed adjacency
        // set in the symmetrized captured graph. This is a conservative sufficient
        // first-party criterion, not a claim about MUMPS' internal block finder.
        std::vector<int> blkptr_1based;
        std::vector<int> blkvar_1based;
        int nonsingleton_blocks = 0;
        int compressed_variables = 0;
    };

    ExactBlockAnalysisHints build_exact_topology_blocks(const MumpsAnalysisPatternView& pattern,
                                                        const TopologyOrderingHints& topology);

    struct SymbolicFrontHistogramBin {
        // Width interval is (lower_exclusive, upper_inclusive]. The first bin is
        // [1,1]. Width is the filled singleton-column count, a first-party front
        // proxy rather than a MUMPS amalgamated multifrontal size.
        std::uint64_t lower_exclusive = 0;
        std::uint64_t upper_inclusive = 0;
        std::uint64_t count = 0;
    };

    struct SymbolicEliminationDiagnostics {
        std::vector<int> parent_1based; // 0 denotes a root.
        std::vector<std::uint64_t> depth_histogram;
        std::vector<SymbolicFrontHistogramBin> front_width_histogram;
        std::uint64_t parent_hash = 0;
        std::uint64_t filled_lower_nnz_proxy = 0;
        std::uint64_t max_singleton_front_width = 0;
        int max_tree_depth = 0;
    };

    // Compute an elimination tree and exact singleton-column fill counts for the
    // symmetrized captured pattern under one supplied permutation. This is an
    // O(nnz(L)) diagnostic and is deliberately offline-only.
    SymbolicEliminationDiagnostics analyze_symbolic_elimination(const MumpsAnalysisPatternView& pattern,
                                                                std::span<const int> permutation_1based);

} // namespace Kadath
