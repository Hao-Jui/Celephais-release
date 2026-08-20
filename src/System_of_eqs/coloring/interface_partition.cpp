/*
    Interface-partition diagnostic consumer (implementation).
*/

#include "For_Kadath/System_of_eqs/Jacobian/interface_partition.hpp"

#include <algorithm>
#include <map>
#include <ostream>
#include <sstream>
#include <string>
#include <utility>

namespace Kadath
{
    namespace
    {
        const char* row_taxonomy_label(RowTaxonomy taxonomy)
        {
            switch (taxonomy) {
                case RowTaxonomy::Vol: return "Vol";
                case RowTaxonomy::TauBc: return "TauBc";
                case RowTaxonomy::TauMatch: return "TauMatch";
                case RowTaxonomy::GlobalInt: return "GlobalInt";
                case RowTaxonomy::Unknown: return "Unknown";
            }
            return "Unknown";
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

        void tally_row_taxonomy(RowTaxonomy taxonomy, InterfacePartitionCensus& census)
        {
            switch (taxonomy) {
                case RowTaxonomy::Vol: ++census.rows_vol; break;
                case RowTaxonomy::TauBc: ++census.rows_tau_bc; break;
                case RowTaxonomy::TauMatch: ++census.rows_tau_match; break;
                case RowTaxonomy::GlobalInt: ++census.rows_global_int; break;
                case RowTaxonomy::Unknown: ++census.rows_unknown; break;
            }
        }

        void tally_column_class(ColumnClass column_class, InterfacePartitionCensus& census)
        {
            switch (column_class) {
                case ColumnClass::FieldUnknown: ++census.cols_field_unknown; break;
                case ColumnClass::FieldInterior: ++census.cols_field_interior; break;
                case ColumnClass::FieldBoundary: ++census.cols_field_boundary; break;
                case ColumnClass::FieldInteriorVol: ++census.cols_field_interior_vol; break;
                case ColumnClass::FieldBoundaryTau: ++census.cols_field_boundary_tau; break;
                case ColumnClass::FieldOuterShellTau: ++census.cols_field_outer_shell_tau; break;
                case ColumnClass::FieldMatching: ++census.cols_field_matching; break;
                case ColumnClass::FieldGauge: ++census.cols_field_gauge; break;
                case ColumnClass::VarDomain: ++census.cols_var_domain; break;
                case ColumnClass::ScalarGlobal: ++census.cols_scalar_global; break;
                case ColumnClass::Unknown: ++census.cols_unknown; break;
            }
        }

        void append_error(InterfacePartitionValidation& validation, const std::string& msg)
        {
            if (validation.errors.size() < 16) {
                validation.errors.push_back(msg);
            }
        }
    } // namespace

    namespace
    {
        bool tau_bc_is_bulk(InterfaceRowPolicy policy)
        {
            return policy == InterfaceRowPolicy::TauBcBulk;
        }
    } // namespace

    void build_interface_partition(const TaggedJacobianMetadata& metadata,
                                   const IncidenceColumnPartition& incidence,
                                   InterfacePartition& out,
                                   InterfaceRowPolicy policy)
    {
        out = InterfacePartition{};
        out.policy = policy;
        const int nrows = metadata.nrows;
        const int ncols = metadata.ncols;

        out.census.total_rows = nrows;
        out.census.total_cols = ncols;
        out.row_role.assign(static_cast<std::size_t>(std::max(0, nrows)), -1);

        if (nrows < 0 || ncols < 0) {
            return;
        }
        if (static_cast<int>(metadata.rows.size()) != nrows ||
            static_cast<int>(metadata.columns.size()) != ncols ||
            static_cast<int>(incidence.col_role.size()) != ncols) {
            return;
        }

        std::map<int, InterfaceDomainBlock> bulk_by_dom;
        const bool tau_bc_rows_are_bulk = tau_bc_is_bulk(policy);

        for (int row = 0; row < nrows; ++row) {
            const RowMetadata& meta = metadata.rows[static_cast<std::size_t>(row)];
            tally_row_taxonomy(meta.taxonomy, out.census);
            switch (meta.taxonomy) {
                case RowTaxonomy::Vol: {
                    if (meta.dom < 0) {
                        out.row_role[static_cast<std::size_t>(row)] = -1;
                        ++out.census.unclassified_rows;
                        break;
                    }
                    out.row_role[static_cast<std::size_t>(row)] = 0;
                    ++out.census.bulk_rows;
                    bulk_by_dom[meta.dom].bulk_rows.push_back(row);
                    break;
                }
                case RowTaxonomy::TauBc: {
                    if (tau_bc_rows_are_bulk) {
                        if (meta.dom < 0) {
                            out.row_role[static_cast<std::size_t>(row)] = -1;
                            ++out.census.unclassified_rows;
                            break;
                        }
                        out.row_role[static_cast<std::size_t>(row)] = 0;
                        ++out.census.bulk_rows;
                        bulk_by_dom[meta.dom].bulk_rows.push_back(row);
                    } else {
                        out.row_role[static_cast<std::size_t>(row)] = 1;
                        ++out.census.interface_rows;
                        out.interface_rows.push_back(row);
                    }
                    break;
                }
                case RowTaxonomy::TauMatch:
                case RowTaxonomy::GlobalInt: {
                    out.row_role[static_cast<std::size_t>(row)] = 1;
                    ++out.census.interface_rows;
                    out.interface_rows.push_back(row);
                    break;
                }
                case RowTaxonomy::Unknown: {
                    out.row_role[static_cast<std::size_t>(row)] = -1;
                    ++out.census.unclassified_rows;
                    break;
                }
            }
        }

        for (int col = 0; col < ncols; ++col) {
            const ColumnMetadata& meta = metadata.columns[static_cast<std::size_t>(col)];
            tally_column_class(meta.column_class, out.census);
            const int role = incidence.col_role[static_cast<std::size_t>(col)];
            if (role == 0) {
                ++out.census.bulk_cols;
                if (meta.domain < 0) {
                    ++out.census.unclassified_cols;
                    continue;
                }
                bulk_by_dom[meta.domain].bulk_cols.push_back(col);
            } else if (role == 1) {
                ++out.census.interface_cols;
                out.interface_cols.push_back(col);
            } else {
                ++out.census.unclassified_cols;
            }
        }

        for (auto& kv : bulk_by_dom) {
            kv.second.domain = kv.first;
            out.domains.push_back(kv.first);
            out.blocks.push_back(std::move(kv.second));
        }
        std::sort(out.interface_rows.begin(), out.interface_rows.end());
        std::sort(out.interface_cols.begin(), out.interface_cols.end());

        out.shapes.resize(out.domains.size());
        out.all_square = !out.domains.empty();
        for (std::size_t i = 0; i < out.domains.size(); ++i) {
            InterfaceDomainShape& shape = out.shapes[i];
            shape.domain = out.domains[i];
            shape.bulk_rows = static_cast<int>(out.blocks[i].bulk_rows.size());
            shape.bulk_cols = static_cast<int>(out.blocks[i].bulk_cols.size());
            shape.square = (shape.bulk_rows == shape.bulk_cols);
            if (!shape.square) {
                out.all_square = false;
            }
        }
    }

    InterfacePartitionValidation
    validate_interface_partition(const TaggedJacobianMetadata& metadata,
                                 const IncidenceColumnPartition& incidence,
                                 const InterfacePartition& partition)
    {
        InterfacePartitionValidation validation;
        const InterfacePartitionCensus& c = partition.census;

        if (c.total_rows != metadata.nrows) {
            append_error(validation, "interface partition census total_rows != metadata.nrows");
        }
        if (c.total_cols != metadata.ncols) {
            append_error(validation, "interface partition census total_cols != metadata.ncols");
        }
        if (c.bulk_rows + c.interface_rows + c.unclassified_rows != c.total_rows) {
            append_error(validation, "interface partition row counts do not sum to total_rows");
        }
        if (c.bulk_cols + c.interface_cols + c.unclassified_cols != c.total_cols) {
            append_error(validation, "interface partition col counts do not sum to total_cols");
        }
        if (c.unclassified_rows != 0) {
            std::ostringstream msg;
            msg << "interface partition has " << c.unclassified_rows
                << " unclassified rows (RowTaxonomy::Unknown or missing dom)";
            append_error(validation, msg.str());
        }
        if (c.unclassified_cols != 0) {
            std::ostringstream msg;
            msg << "interface partition has " << c.unclassified_cols
                << " unclassified cols (col_role outside {0,1} or bulk-col missing domain)";
            append_error(validation, msg.str());
        }
        const int expected_bulk_cols = static_cast<int>(incidence.bulk_cols.size());
        const int expected_interface_cols = static_cast<int>(incidence.transfer_cols.size());
        if (c.bulk_cols != expected_bulk_cols) {
            append_error(validation,
                         "interface partition bulk_cols disagrees with IncidenceColumnPartition.bulk_cols");
        }
        if (c.interface_cols != expected_interface_cols) {
            append_error(validation,
                         "interface partition interface_cols disagrees with IncidenceColumnPartition.transfer_cols");
        }
        if (partition.blocks.size() != partition.domains.size()) {
            append_error(validation, "interface partition blocks/domains size mismatch");
        }
        for (std::size_t i = 0; i < partition.blocks.size(); ++i) {
            if (partition.blocks[i].domain != partition.domains[i]) {
                append_error(validation, "interface partition block.domain misaligned with domains[]");
                break;
            }
            if (partition.blocks[i].bulk_rows.empty() && partition.blocks[i].bulk_cols.empty()) {
                std::ostringstream msg;
                msg << "interface partition empty block for domain " << partition.blocks[i].domain;
                append_error(validation, msg.str());
            }
        }
        validation.ok = validation.errors.empty();
        return validation;
    }

    void dump_interface_partition_census(const TaggedJacobianMetadata& metadata,
                                         const InterfacePartition& partition,
                                         const InterfacePartitionValidation& validation,
                                         std::ostream& os)
    {
        const InterfacePartitionCensus& c = partition.census;
        const char* policy_label = "TauBcBulk (Vol+TauBc bulk)";
        switch (partition.policy) {
            case InterfaceRowPolicy::TauBcBulk:
                policy_label = "TauBcBulk (Vol+TauBc bulk)";
                break;
            case InterfaceRowPolicy::TauBcInterface:
                policy_label = "TauBcInterface (Vol bulk; TauBc interface)";
                break;
        }
        os << "Interface Partition Census [policy: " << policy_label << "]:\n";
        os << "  Validation: " << (validation.ok ? "ok" : "failed") << "\n";
        for (const std::string& err : validation.errors) {
            os << "    error: " << err << "\n";
        }
        os << "  Shape: rows=" << metadata.nrows << " cols=" << metadata.ncols << "\n";
        switch (partition.policy) {
            case InterfaceRowPolicy::TauBcBulk:
                os << "  Row split (Vol+TauBc=bulk; TauMatch+GlobalInt=interface):\n";
                break;
            case InterfaceRowPolicy::TauBcInterface:
                os << "  Row split (Vol=bulk; TauBc+TauMatch+GlobalInt=interface):\n";
                break;
        }
        os << "    bulk=" << c.bulk_rows
           << "  interface=" << c.interface_rows
           << "  unclassified=" << c.unclassified_rows << "\n";
        os << "  Col split (IncidenceColumnPartition):\n";
        os << "    bulk=" << c.bulk_cols
           << "  interface=" << c.interface_cols
           << "  unclassified=" << c.unclassified_cols << "\n";
        os << "  Row taxonomy: "
           << row_taxonomy_label(RowTaxonomy::Vol) << "=" << c.rows_vol << " "
           << row_taxonomy_label(RowTaxonomy::TauBc) << "=" << c.rows_tau_bc << " "
           << row_taxonomy_label(RowTaxonomy::TauMatch) << "=" << c.rows_tau_match << " "
           << row_taxonomy_label(RowTaxonomy::GlobalInt) << "=" << c.rows_global_int << " "
           << row_taxonomy_label(RowTaxonomy::Unknown) << "=" << c.rows_unknown << "\n";
        os << "  Column classes: "
           << column_class_label(ColumnClass::FieldUnknown) << "=" << c.cols_field_unknown << " "
           << column_class_label(ColumnClass::FieldInterior) << "=" << c.cols_field_interior << " "
           << column_class_label(ColumnClass::FieldBoundary) << "=" << c.cols_field_boundary << " "
           << column_class_label(ColumnClass::FieldInteriorVol) << "=" << c.cols_field_interior_vol << " "
           << column_class_label(ColumnClass::FieldBoundaryTau) << "=" << c.cols_field_boundary_tau << " "
           << column_class_label(ColumnClass::FieldOuterShellTau) << "=" << c.cols_field_outer_shell_tau << " "
           << column_class_label(ColumnClass::FieldMatching) << "=" << c.cols_field_matching << " "
           << column_class_label(ColumnClass::FieldGauge) << "=" << c.cols_field_gauge << " "
           << column_class_label(ColumnClass::VarDomain) << "=" << c.cols_var_domain << " "
           << column_class_label(ColumnClass::ScalarGlobal) << "=" << c.cols_scalar_global << " "
           << column_class_label(ColumnClass::Unknown) << "=" << c.cols_unknown << "\n";
        os << "  Per-domain bulk blocks (domain: bulk_rows x bulk_cols [square?]):\n";
        for (std::size_t i = 0; i < partition.blocks.size(); ++i) {
            const InterfaceDomainBlock& blk = partition.blocks[i];
            const bool square =
                i < partition.shapes.size() ? partition.shapes[i].square
                                            : (blk.bulk_rows.size() == blk.bulk_cols.size());
            os << "    d" << blk.domain << ": "
               << blk.bulk_rows.size() << " x " << blk.bulk_cols.size()
               << "  square=" << (square ? "true" : "false") << "\n";
        }
        os << "  all_square=" << (partition.all_square ? "true" : "false") << "\n";
    }
} // namespace Kadath
