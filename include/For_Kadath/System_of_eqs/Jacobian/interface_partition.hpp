/*
    Interface-partition diagnostic consumer.

    Thin reader over TaggedJacobianMetadata + IncidenceColumnPartition. It does
    not classify rows or columns from scratch; it only groups the canonical
    taxonomy into bulk rows/columns and interface rows/columns for diagnostics.
*/

#pragma once

#include "tagged_metadata.hpp"

#include <iosfwd>
#include <vector>

namespace Kadath
{
    /// Per-domain bulk-row / bulk-col group.
    struct InterfaceDomainBlock {
        int domain = -1;
        std::vector<int> bulk_rows;  ///< Local rows assigned to the bulk side.
        std::vector<int> bulk_cols;  ///< Bulk columns from IncidenceColumnPartition whose domain == this.
    };

    /// Census counters for a single InterfacePartition build.
    struct InterfacePartitionCensus {
        int total_rows = 0;
        int total_cols = 0;
        int bulk_rows = 0;
        int interface_rows = 0;
        int bulk_cols = 0;
        int interface_cols = 0;
        int unclassified_rows = 0;
        int unclassified_cols = 0;

        int rows_vol = 0;
        int rows_tau_bc = 0;
        int rows_tau_match = 0;
        int rows_global_int = 0;
        int rows_unknown = 0;

        int cols_field_unknown = 0;
        int cols_field_interior = 0;
        int cols_field_boundary = 0;
        int cols_field_interior_vol = 0;
        int cols_field_boundary_tau = 0;
        int cols_field_outer_shell_tau = 0;
        int cols_field_matching = 0;
        int cols_field_gauge = 0;
        int cols_var_domain = 0;
        int cols_scalar_global = 0;
        int cols_unknown = 0;
    };

    /// Row-side assignment for TauBc rows. Column-side assignment comes
    /// directly from IncidenceColumnPartition: bulk columns stay bulk, transfer
    /// columns stay interface.
    enum class InterfaceRowPolicy {
        TauBcBulk,
        TauBcInterface,
    };

    /// Per-domain bulk-block rectangularity report (computed by the builder
    /// alongside the partition). square == (bulk_rows == bulk_cols).
    struct InterfaceDomainShape {
        int domain = -1;
        int bulk_rows = 0;
        int bulk_cols = 0;
        bool square = false;
    };

    struct InterfacePartition {
        InterfaceRowPolicy policy = InterfaceRowPolicy::TauBcBulk;
        std::vector<int> row_role;             ///< Per-row 0=bulk, 1=interface, -1=unclassified (size = metadata.nrows)
        std::vector<int> interface_rows;       ///< Sorted ascending.
        std::vector<int> interface_cols;       ///< Reuses IncidenceColumnPartition.transfer_cols.
        std::vector<int> domains;              ///< Sorted ascending. Domains with non-empty bulk row OR col list.
        std::vector<InterfaceDomainBlock> blocks; ///< Parallel to domains. blocks[i].domain == domains[i].
        std::vector<InterfaceDomainShape> shapes; ///< Parallel to domains; per-domain rectangularity.
        bool all_square = false;
        InterfacePartitionCensus census;
    };

    struct InterfacePartitionValidation {
        bool ok = false;
        std::vector<std::string> errors;
    };

    /// Build the interface partition view from already-built metadata.
    /// Pre-conditions:
    ///   - metadata.nrows / ncols set
    ///   - metadata.rows.size() == metadata.nrows, metadata.columns.size() == metadata.ncols
    ///   - incidence.col_role.size() == metadata.ncols
    /// Postcondition: out is fully populated; census counters are exact.
    void build_interface_partition(const TaggedJacobianMetadata& metadata,
                                   const IncidenceColumnPartition& incidence,
                                   InterfacePartition& out,
                                   InterfaceRowPolicy policy = InterfaceRowPolicy::TauBcBulk);

    /// Cross-check the partition against the metadata. Reports unclassified
    /// rows/cols, role-count consistency, per-domain non-emptiness, and
    /// agreement with IncidenceColumnPartition column counts.
    InterfacePartitionValidation
    validate_interface_partition(const TaggedJacobianMetadata& metadata,
                                 const IncidenceColumnPartition& incidence,
                                 const InterfacePartition& partition);

    /// Human-readable single-block census suitable for stderr or run.log.
    void dump_interface_partition_census(const TaggedJacobianMetadata& metadata,
                                         const InterfacePartition& partition,
                                         const InterfacePartitionValidation& validation,
                                         std::ostream& os);
} // namespace Kadath
