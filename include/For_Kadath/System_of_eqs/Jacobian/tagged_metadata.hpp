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

#include "column_types.hpp"

#include <set>
#include <string>
#include <vector>

namespace Kadath
{
    enum class RowClass {
        Galerkin,
        Constraint,
        FirstIntegral,
    };

    enum class RowTaxonomy {
        Unknown,
        Vol,
        TauBc,
        TauMatch,
        GlobalInt,
    };

    struct RowMetadata {
        int row = -1;
        RowClass legacy_class = RowClass::Constraint;
        RowTaxonomy taxonomy = RowTaxonomy::Unknown;
        int dom = -1;
        int dom_pair = -1;
        int eq_index = -1;
        int eq_local_row = -1;
        int comp = -1;
        std::string equation_type;
        std::string owner_var_name; ///< Field-unknown name owning the equation; mirrors eq_column_attachments[eq_index].owner_var_name when available.
        int basis_mode = -1;
    };

    /// Unified row/column tags for the flat Jacobian ordering used by sec_member() and do_col_J().
    struct TaggedJacobianMetadata {
        int nrows = 0;
        int ncols = 0;
        std::vector<RowMetadata> rows;
        std::vector<ColumnMetadata> columns;
        std::vector<std::set<int>> rows_per_column;
    };

    /// Count/order checks for TaggedJacobianMetadata before block extraction code trusts it.
    struct TaggedJacobianMetadataValidation {
        bool ok = false;
        int expected_rows = 0;
        int expected_cols = 0;
        int actual_rows = 0;
        int actual_cols = 0;
        int row_support_columns = 0;
        long long row_support_entries = 0;
        std::vector<std::string> errors;
    };

    struct IncidenceColumnPartition {
        std::vector<int> col_role;       ///< 0=bulk, 1=transfer
        std::vector<int> bulk_cols;
        std::vector<int> transfer_cols;
        std::vector<int> domains;
    };
} // namespace Kadath
