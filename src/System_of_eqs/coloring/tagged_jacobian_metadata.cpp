#include "coloring_internals.hpp"

#include <algorithm>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace Kadath
{
    using namespace coloring_internals;

    namespace
    {
        const char* row_taxonomy_label(System_of_eqs::RowTaxonomy taxonomy)
        {
            switch (taxonomy) {
                case System_of_eqs::RowTaxonomy::Vol: return "Vol";
                case System_of_eqs::RowTaxonomy::TauBc: return "TauBc";
                case System_of_eqs::RowTaxonomy::TauMatch: return "TauMatch";
                case System_of_eqs::RowTaxonomy::GlobalInt: return "GlobalInt";
                case System_of_eqs::RowTaxonomy::Unknown: return "Unknown";
            }
            return "Unknown";
        }

        const char* row_class_label(System_of_eqs::RowClass row_class)
        {
            switch (row_class) {
                case System_of_eqs::RowClass::Galerkin: return "Galerkin";
                case System_of_eqs::RowClass::Constraint: return "Constraint";
                case System_of_eqs::RowClass::FirstIntegral: return "FirstIntegral";
            }
            return "Constraint";
        }

        const char* column_class_label(ColumnClass column_class)
        {
            switch (column_class) {
                case ColumnClass::FieldUnknown: return "FieldUnknown";
                case ColumnClass::FieldInterior: return "FieldInterior";
                case ColumnClass::FieldBoundary: return "FieldBoundary";
                case ColumnClass::FieldInteriorVol: return "FieldInteriorVol";
                case ColumnClass::FieldBoundaryTau: return "FieldBoundaryTau";
                case ColumnClass::FieldOuterShellTau: return "FieldOuterShellTau";
                case ColumnClass::FieldMatching: return "FieldMatching";
                case ColumnClass::FieldGauge: return "FieldGauge";
                case ColumnClass::VarDomain: return "VarDomain";
                case ColumnClass::ScalarGlobal: return "ScalarGlobal";
                case ColumnClass::Unknown: return "Unknown";
            }
            return "Unknown";
        }

        /// Promotes FieldInteriorVol cols stored in the interior basis slot
        /// of an unmatched outer-shell face to FieldOuterShellTau.
        ///
        /// Discriminator: per (domain, owner_var_name), count Vol rows owned
        /// by `var` in `domain` and count FieldInteriorVol cols at the same
        /// (domain, var). When FieldInteriorVol cols outnumber Vol rows, the
        /// excess cols are decay-tau coefficient modes stored in the interior
        /// basis slot. Demote the surplus by selecting the highest
        /// basis_mode values (the standard tau-coefficient convention in
        /// Kadath's spectral expansion).
        ///
        /// No effect on fixtures where Vol_count(dom,var) ==
        /// FieldInteriorVol_count(dom,var) — including all of BH2d and the
        /// majority of BNS domains.
        void promote_outer_shell_tau_columns(TaggedJacobianMetadata& metadata)
        {
            if (metadata.rows_per_column.empty()) {
                return;
            }
            if (static_cast<int>(metadata.rows_per_column.size()) != metadata.ncols) {
                return;
            }
            if (static_cast<int>(metadata.rows.size()) != metadata.nrows) {
                return;
            }
            using DomVar = std::pair<int, std::string>;
            std::map<DomVar, int> vol_rows_per_dv;
            for (const RowMetadata& row : metadata.rows) {
                if (row.taxonomy != RowTaxonomy::Vol) {
                    continue;
                }
                if (row.dom < 0 || row.owner_var_name.empty()) {
                    continue;
                }
                ++vol_rows_per_dv[{row.dom, row.owner_var_name}];
            }

            std::map<DomVar, std::vector<int>> field_interior_cols_per_dv;
            for (int col = 0; col < metadata.ncols; ++col) {
                const ColumnMetadata& column =
                    metadata.columns[static_cast<std::size_t>(col)];
                if (column.column_class != ColumnClass::FieldInteriorVol) {
                    continue;
                }
                if (column.domain < 0 || column.var_name.empty()) {
                    continue;
                }
                field_interior_cols_per_dv[{column.domain, column.var_name}]
                    .push_back(col);
            }

            for (auto& kv : field_interior_cols_per_dv) {
                const DomVar& dv = kv.first;
                std::vector<int>& cols = kv.second;
                const auto it = vol_rows_per_dv.find(dv);
                const int vol_count = (it != vol_rows_per_dv.end()) ? it->second : 0;
                const int surplus = static_cast<int>(cols.size()) - vol_count;
                if (surplus <= 0) {
                    continue;
                }
                std::sort(cols.begin(), cols.end(),
                          [&metadata](int a, int b) {
                              return metadata.columns[static_cast<std::size_t>(a)].basis_mode <
                                     metadata.columns[static_cast<std::size_t>(b)].basis_mode;
                          });
                for (int i = static_cast<int>(cols.size()) - surplus;
                     i < static_cast<int>(cols.size()); ++i) {
                    ColumnMetadata& demoted =
                        metadata.columns[static_cast<std::size_t>(cols[i])];
                    demoted.column_class = ColumnClass::FieldOuterShellTau;
                    demoted.reason = "field_coefficient_outer_shell_tau_decay_closure";
                }
            }
        }

        void append_metadata_error(System_of_eqs::TaggedJacobianMetadataValidation& validation,
                                   const std::string& error)
        {
            if (validation.errors.size() < 8) {
                validation.errors.push_back(error);
            }
        }

        std::string csv_escape(const std::string& value)
        {
            bool quote = false;
            for (char ch : value) {
                if (ch == ',' || ch == '"' || ch == '\n' || ch == '\r') {
                    quote = true;
                    break;
                }
            }
            if (!quote) {
                return value;
            }

            std::string out = "\"";
            for (char ch : value) {
                if (ch == '"') {
                    out += "\"\"";
                } else {
                    out += ch;
                }
            }
            out += '"';
            return out;
        }

    } // namespace

    void System_of_eqs::build_tagged_jacobian_metadata(TaggedJacobianMetadata& out,
                                                       bool include_row_incidence) const
    {
        if (nbr_conditions < 0) {
            KADATH_THROW(kColoringNeedsSecMember);
        }

        out = TaggedJacobianMetadata{};
        out.nrows = nbr_conditions;
        out.ncols = nbr_unknowns;

        classify_equation_row_metadata(out.rows);
        classify_columns(out.columns);

        if (include_row_incidence) {
            std::vector<ColumnInfo> column_map;
            build_column_map(column_map);
            if (static_cast<int>(column_map.size()) != nbr_unknowns) {
                KADATH_THROW("Tagged Jacobian metadata column map size mismatch");
            }
            build_column_row_incidence(column_map, out.rows_per_column);
            promote_outer_shell_tau_columns(out);
        }
    }

    System_of_eqs::TaggedJacobianMetadataValidation
    System_of_eqs::validate_tagged_jacobian_metadata(const TaggedJacobianMetadata& metadata) const
    {
        TaggedJacobianMetadataValidation validation;
        validation.expected_rows = nbr_conditions;
        validation.expected_cols = nbr_unknowns;
        validation.actual_rows = static_cast<int>(metadata.rows.size());
        validation.actual_cols = static_cast<int>(metadata.columns.size());
        validation.row_support_columns = static_cast<int>(metadata.rows_per_column.size());

        if (nbr_conditions < 0) {
            append_metadata_error(validation, kColoringNeedsSecMember);
        }
        if (metadata.nrows != nbr_conditions) {
            append_metadata_error(validation, "metadata.nrows does not match System_of_eqs::nbr_conditions");
        }
        if (metadata.ncols != nbr_unknowns) {
            append_metadata_error(validation, "metadata.ncols does not match System_of_eqs::nbr_unknowns");
        }
        if (validation.actual_rows != nbr_conditions) {
            append_metadata_error(validation, "row metadata count does not match System_of_eqs::nbr_conditions");
        }
        if (validation.actual_cols != nbr_unknowns) {
            append_metadata_error(validation, "column metadata count does not match System_of_eqs::nbr_unknowns");
        }

        const int rows_to_check = std::min(validation.actual_rows, std::max(0, nbr_conditions));
        for (int row = 0; row < rows_to_check; ++row) {
            const RowMetadata& item = metadata.rows[static_cast<std::size_t>(row)];
            if (item.row != row) {
                append_metadata_error(validation, "row metadata is not in canonical sec_member row order");
                break;
            }
        }

        const int cols_to_check = std::min(validation.actual_cols, std::max(0, nbr_unknowns));
        for (int col = 0; col < cols_to_check; ++col) {
            const ColumnMetadata& item = metadata.columns[static_cast<std::size_t>(col)];
            if (item.column != col) {
                append_metadata_error(validation, "column metadata is not in canonical do_col_J column order");
                break;
            }
        }

        if (!metadata.rows_per_column.empty()) {
            if (validation.row_support_columns != nbr_unknowns) {
                append_metadata_error(validation, "column-row incidence count does not match System_of_eqs::nbr_unknowns");
            }
            const int support_cols =
                std::min(validation.row_support_columns, std::max(0, nbr_unknowns));
            bool found_bad_support = false;
            for (int col = 0; col < support_cols && !found_bad_support; ++col) {
                for (int row : metadata.rows_per_column[static_cast<std::size_t>(col)]) {
                    ++validation.row_support_entries;
                    if (row < 0 || row >= nbr_conditions) {
                        append_metadata_error(validation, "column-row incidence contains row outside Jacobian range");
                        found_bad_support = true;
                        break;
                    }
                }
            }
        }

        validation.ok = validation.errors.empty();
        return validation;
    }

    void System_of_eqs::build_incidence_column_partition(
        const std::vector<RowMetadata>& row_metadata,
        const std::vector<ColumnMetadata>& column_metadata,
        const std::vector<std::set<int>>& rows_per_column,
        IncidenceColumnPartition& out) const
    {
        if (nbr_conditions < 0) {
            KADATH_THROW(kColoringNeedsSecMember);
        }
        if (static_cast<int>(row_metadata.size()) != nbr_conditions) {
            KADATH_THROW("Incidence column partition row metadata size mismatch");
        }
        if (static_cast<int>(column_metadata.size()) != nbr_unknowns) {
            KADATH_THROW("Incidence column partition column metadata size mismatch");
        }
        if (static_cast<int>(rows_per_column.size()) != nbr_unknowns) {
            KADATH_THROW("Incidence column partition row-support size mismatch");
        }

        out = IncidenceColumnPartition{};
        out.col_role.assign(static_cast<std::size_t>(nbr_unknowns), 1);

        for (int col = 0; col < nbr_unknowns; ++col) {
            const ColumnMetadata& meta = column_metadata[static_cast<std::size_t>(col)];
            bool bulk = false;
            if (meta.column_class == ColumnClass::FieldInteriorVol && meta.domain >= 0) {
                bool has_same_domain_vol = false;
                bool has_cross_domain_vol = false;
                for (int row : rows_per_column[static_cast<std::size_t>(col)]) {
                    if (row < 0 || row >= nbr_conditions) {
                        has_cross_domain_vol = true;
                        break;
                    }
                    const RowMetadata& row_meta = row_metadata[static_cast<std::size_t>(row)];
                    if (row_meta.taxonomy != RowTaxonomy::Vol) {
                        continue;
                    }
                    if (row_meta.dom != meta.domain) {
                        has_cross_domain_vol = true;
                        break;
                    }
                    has_same_domain_vol = true;
                }
                bulk = has_same_domain_vol && !has_cross_domain_vol;
            }

            if (bulk) {
                out.col_role[static_cast<std::size_t>(col)] = 0;
                out.bulk_cols.push_back(col);
                if (std::find(out.domains.begin(), out.domains.end(), meta.domain) ==
                    out.domains.end()) {
                    out.domains.push_back(meta.domain);
                }
            } else {
                out.transfer_cols.push_back(col);
            }
        }

        std::sort(out.domains.begin(), out.domains.end());
    }

    void System_of_eqs::dump_tagged_jacobian_metadata_summary(std::ostream& os) const
    {
        if (nbr_conditions < 0) {
            os << kColoringNeedsSecMember << std::endl;
            return;
        }

        TaggedJacobianMetadata metadata;
        build_tagged_jacobian_metadata(metadata);
        const TaggedJacobianMetadataValidation validation =
            validate_tagged_jacobian_metadata(metadata);

        int vol_rows = 0;
        int tau_bc_rows = 0;
        int tau_match_rows = 0;
        int global_int_rows = 0;
        int unknown_rows = 0;
        for (const RowMetadata& row : metadata.rows) {
            switch (row.taxonomy) {
                case RowTaxonomy::Vol: ++vol_rows; break;
                case RowTaxonomy::TauBc: ++tau_bc_rows; break;
                case RowTaxonomy::TauMatch: ++tau_match_rows; break;
                case RowTaxonomy::GlobalInt: ++global_int_rows; break;
                case RowTaxonomy::Unknown: ++unknown_rows; break;
            }
        }

        int field_unknown_cols = 0;
        int field_interior_cols = 0;
        int field_boundary_cols = 0;
        int field_interior_vol_cols = 0;
        int field_boundary_tau_cols = 0;
        int field_outer_shell_tau_cols = 0;
        int field_matching_cols = 0;
        int field_gauge_cols = 0;
        int var_domain_cols = 0;
        int scalar_global_cols = 0;
        int unknown_cols = 0;
        for (const ColumnMetadata& col : metadata.columns) {
            switch (col.column_class) {
                case ColumnClass::FieldUnknown: ++field_unknown_cols; break;
                case ColumnClass::FieldInterior: ++field_interior_cols; break;
                case ColumnClass::FieldBoundary: ++field_boundary_cols; break;
                case ColumnClass::FieldInteriorVol: ++field_interior_vol_cols; break;
                case ColumnClass::FieldBoundaryTau: ++field_boundary_tau_cols; break;
                case ColumnClass::FieldOuterShellTau: ++field_outer_shell_tau_cols; break;
                case ColumnClass::FieldMatching: ++field_matching_cols; break;
                case ColumnClass::FieldGauge: ++field_gauge_cols; break;
                case ColumnClass::VarDomain: ++var_domain_cols; break;
                case ColumnClass::ScalarGlobal: ++scalar_global_cols; break;
                case ColumnClass::Unknown: ++unknown_cols; break;
            }
        }

        os << "Tagged Jacobian Metadata:" << std::endl;
        os << "  Validation: " << (validation.ok ? "ok" : "failed") << std::endl;
        for (const std::string& error : validation.errors) {
            os << "    error: " << error << std::endl;
        }
        os << "  Shape: rows=" << metadata.nrows << " cols=" << metadata.ncols << std::endl;
        os << "  Row taxonomy: "
           << row_taxonomy_label(RowTaxonomy::Vol) << "=" << vol_rows << " "
           << row_taxonomy_label(RowTaxonomy::TauBc) << "=" << tau_bc_rows << " "
           << row_taxonomy_label(RowTaxonomy::TauMatch) << "=" << tau_match_rows << " "
           << row_taxonomy_label(RowTaxonomy::GlobalInt) << "=" << global_int_rows << " "
           << row_taxonomy_label(RowTaxonomy::Unknown) << "=" << unknown_rows << std::endl;
        os << "  Column classes: "
           << column_class_label(ColumnClass::FieldUnknown) << "=" << field_unknown_cols << " "
           << column_class_label(ColumnClass::FieldInterior) << "=" << field_interior_cols << " "
           << column_class_label(ColumnClass::FieldBoundary) << "=" << field_boundary_cols << " "
           << column_class_label(ColumnClass::FieldInteriorVol) << "=" << field_interior_vol_cols << " "
           << column_class_label(ColumnClass::FieldBoundaryTau) << "=" << field_boundary_tau_cols << " "
           << column_class_label(ColumnClass::FieldOuterShellTau) << "=" << field_outer_shell_tau_cols << " "
           << column_class_label(ColumnClass::FieldMatching) << "=" << field_matching_cols << " "
           << column_class_label(ColumnClass::FieldGauge) << "=" << field_gauge_cols << " "
           << column_class_label(ColumnClass::VarDomain) << "=" << var_domain_cols << " "
           << column_class_label(ColumnClass::ScalarGlobal) << "=" << scalar_global_cols << " "
           << column_class_label(ColumnClass::Unknown) << "=" << unknown_cols << std::endl;
        print_support_summary(os, "Metadata row incidence",
                              summarize_support(metadata.rows_per_column));
    }

    void System_of_eqs::dump_tagged_jacobian_metadata_csv(std::ostream& row_os,
                                                          std::ostream& column_os) const
    {
        TaggedJacobianMetadata metadata;
        build_tagged_jacobian_metadata(metadata, false);

        row_os << "row,taxonomy,legacy_class,dom,dom_pair,eq_index,eq_local_row,comp,basis_mode,equation_type\n";
        for (const RowMetadata& row : metadata.rows) {
            row_os << row.row << ','
                   << row_taxonomy_label(row.taxonomy) << ','
                   << row_class_label(row.legacy_class) << ','
                   << row.dom << ','
                   << row.dom_pair << ','
                   << row.eq_index << ','
                   << row.eq_local_row << ','
                   << row.comp << ','
                   << row.basis_mode << ','
                   << csv_escape(row.equation_type) << '\n';
        }

        column_os << "column,column_class,domain,term_idx,var_idx,var_double_idx,vardom_param,var_name,basis_mode,reason\n";
        for (const ColumnMetadata& column : metadata.columns) {
            column_os << column.column << ','
                      << column_class_label(column.column_class) << ','
                      << column.domain << ','
                      << column.term_idx << ','
                      << column.var_idx << ','
                      << column.var_double_idx << ','
                      << column.vardom_param << ','
                      << csv_escape(column.var_name) << ','
                      << column.basis_mode << ','
                      << csv_escape(column.reason) << '\n';
        }
    }

} // namespace Kadath
