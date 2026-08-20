/*
    Copyright 2017 Philippe Grandclement & Gregoire Martinon

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

#include <cstddef>
#include <set>
#include <string>
#include <vector>

namespace Kadath
{
    enum class ColumnClass {
        Unknown,
        FieldUnknown,
        FieldInterior,
        FieldBoundary,
        FieldInteriorVol,
        FieldBoundaryTau,
        /// Tau-coefficient unknown stored in the FieldInteriorVol basis slot
        /// of an unmatched outer-shell face. Kadath emits the decay closure
        /// row as TauBc and stores its conjugate coefficient mode as
        /// FieldInteriorVol because the basis layout does not separate
        /// boundary-tau cells from interior cells on outer-decay shells.
        /// Identified post-classification by row-support: a FieldInteriorVol
        /// col is promoted to FieldOuterShellTau iff its row support
        /// contains a same-domain TauBc row AND no TauMatch row anywhere.
        /// Interface-partition diagnostics keep this class on the transfer
        /// side when IncidenceColumnPartition classifies it that way.
        FieldOuterShellTau,
        FieldMatching,
        FieldGauge,
        VarDomain,
        ScalarGlobal,
    };

    /** Stable semantic identifiers for domain classes captured in column tags. */
    enum class ColumnDomainType : int {
        Unknown = -1,
        SphericNucleus = 1,
        SphericShell = 2,
        SphericCompact = 3,
        SphericShellOuterAdapted = 4,
        SphericShellInnerAdapted = 5,
        BisphericChiFirst = 6,
        BisphericRect = 7,
        BisphericEtaFirst = 8,
        SphericNucleusNoSym = 9,
        SphericShellNoSym = 10,
        SphericCompactNoSym = 11,
        SphericShellOuterAdaptedNoSym = 12,
        SphericShellInnerAdaptedNoSym = 13,
        BisphericChiFirstNoSym = 14,
        BisphericRectNoSym = 15,
        BisphericEtaFirstNoSym = 16,
    };

    /**
     * Metadata for a single Jacobian column.
     * Maps column index to the variable/domain it represents.
     */
    struct ColumnInfo {
        int var_idx = -1;           ///< Index into var[] array (-1 if not a field variable)
        int var_double_idx = -1;    ///< Index into var_double[] array (-1 if not a scalar)
        int domain = -1;            ///< Domain index (-1 if not domain-specific)
        int term_idx = -1;          ///< Index into term[] array
        int basis_mode = -1;        ///< Coefficient index inside the term/domain block
        int domain_type_id = static_cast<int>(ColumnDomainType::Unknown);
        int tensor_component = -1;
        int coefficient_i = -1;
        int coefficient_j = -1;
        int coefficient_k = -1;
        int coefficient_nr = -1;
        int coefficient_nt = -1;
        int coefficient_np = -1;
        ColumnClass field_class = ColumnClass::FieldUnknown; ///< Field coefficient role, when known.
        std::string var_name;       ///< Variable name for debugging
        bool is_var_domain = false; ///< True if this is a variable domain coefficient
    };

    struct ColumnMetadata {
        int column = -1;
        ColumnClass column_class = ColumnClass::Unknown;
        int domain = -1;
        int term_idx = -1;
        int var_idx = -1;
        int var_double_idx = -1;
        int vardom_param = -1;
        std::string var_name;
        int basis_mode = -1;
        int domain_type_id = static_cast<int>(ColumnDomainType::Unknown);
        int tensor_component = -1;
        int coefficient_i = -1;
        int coefficient_j = -1;
        int coefficient_k = -1;
        int coefficient_nr = -1;
        int coefficient_nt = -1;
        int coefficient_np = -1;
        std::string reason = "unknown";
    };

    struct DirectJacobianEntry {
        int row = -1;
        double value = 0.0;
    };

    struct DirectJacobianColumn {
        std::size_t first_entry_index = 0;
        std::size_t entry_count = 0;

        bool has_entries() const { return entry_count > 0; }
    };

    struct DirectJacobianColumnPlan {
        std::vector<DirectJacobianColumn> columns;
        std::vector<DirectJacobianEntry> entries;
    };

    /**
     * Immutable assembler planning products that are safe to retain across
     * JacobianAssembler instances.  The probe-heavy ColumnInfo map is an
     * intentional transient and is never retained in this cache value.
     */
    struct JacobianAssemblerStructuralPlan {
        DirectJacobianColumnPlan direct_singleton_plan;
        std::vector<ColumnMetadata> column_metadata;
    };

    /** Per-access evidence for the assembler structural-plan cache. */
    struct JacobianAssemblerStructuralPlanAccess {
        bool cache_hit = false;
        double cache_check_seconds = 0.0;
        double cache_miss_build_seconds = 0.0;
    };

    /**
     * Graph coloring for Jacobian columns.
     * Columns with the same color affect disjoint equation rows and can be computed together.
     */
    struct ColumnColoring {
        std::vector<int> color;               ///< color[col_idx] = color_id
        std::vector<std::vector<int>> groups; ///< groups[color_id] = {col_indices...}
        int num_colors = 0;                   ///< Number of distinct colors
        std::vector<ColumnInfo> column_map;   ///< Maps column index to variable info
        bool initialized = false;             ///< Cache validity flag
        const void* system_ptr = nullptr;     ///< System pointer for cache validation

        // Cached data for seeded computation optimization
        std::vector<std::set<int>> rows_per_column; ///< rows_per_column[col] = set of affected rows
        std::vector<int> col_to_domain;             ///< col_to_domain[col] = domain (-1 if not domain-specific)
    };
} // namespace Kadath
