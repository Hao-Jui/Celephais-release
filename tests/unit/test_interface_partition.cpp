/*
    Unit tests for the interface partition diagnostic consumer.

    Synthetic TaggedJacobianMetadata + IncidenceColumnPartition inputs verify
    the row policy (Vol+TauBc -> bulk by row.dom; TauMatch+GlobalInt -> interface;
    Unknown -> unclassified) and the bulk-col regroup-by-domain logic.
*/

#include <catch2/catch_test_macros.hpp>

#include "For_Kadath/System_of_eqs/Jacobian/interface_partition.hpp"

#include <sstream>
#include <vector>

using namespace Kadath;

namespace
{
    RowMetadata make_row(int row, RowTaxonomy taxonomy, int dom, int dom_pair = -1)
    {
        RowMetadata r;
        r.row = row;
        r.taxonomy = taxonomy;
        r.dom = dom;
        r.dom_pair = dom_pair;
        return r;
    }

    ColumnMetadata make_col(int column, ColumnClass column_class, int domain)
    {
        ColumnMetadata c;
        c.column = column;
        c.column_class = column_class;
        c.domain = domain;
        return c;
    }
} // namespace

TEST_CASE("Interface partition handles empty input", "[interface-partition]")
{
    TaggedJacobianMetadata metadata;
    IncidenceColumnPartition incidence;

    InterfacePartition partition;
    build_interface_partition(metadata, incidence, partition);
    const InterfacePartitionValidation validation =
        validate_interface_partition(metadata, incidence, partition);

    CHECK(partition.census.total_rows == 0);
    CHECK(partition.census.total_cols == 0);
    CHECK(partition.census.bulk_rows == 0);
    CHECK(partition.census.interface_rows == 0);
    CHECK(partition.census.unclassified_rows == 0);
    CHECK(partition.domains.empty());
    CHECK(validation.ok);
}

TEST_CASE("Interface partition: 2 domains with Vol/TauBc/TauMatch rows",
          "[interface-partition]")
{
    TaggedJacobianMetadata metadata;
    metadata.nrows = 7;
    metadata.ncols = 5;
    metadata.rows = {
        make_row(0, RowTaxonomy::Vol, 0),
        make_row(1, RowTaxonomy::Vol, 0),
        make_row(2, RowTaxonomy::TauBc, 0),
        make_row(3, RowTaxonomy::Vol, 1),
        make_row(4, RowTaxonomy::TauBc, 1),
        make_row(5, RowTaxonomy::TauMatch, 0, /*dom_pair=*/1),
        make_row(6, RowTaxonomy::GlobalInt, -1),
    };
    metadata.columns = {
        make_col(0, ColumnClass::FieldInteriorVol, 0),
        make_col(1, ColumnClass::FieldInteriorVol, 0),
        make_col(2, ColumnClass::FieldInteriorVol, 1),
        make_col(3, ColumnClass::FieldMatching, 0),
        make_col(4, ColumnClass::ScalarGlobal, -1),
    };
    metadata.rows_per_column.resize(metadata.ncols);

    IncidenceColumnPartition incidence;
    incidence.col_role = {0, 0, 0, 1, 1};
    incidence.bulk_cols = {0, 1, 2};
    incidence.transfer_cols = {3, 4};
    incidence.domains = {0, 1};

    InterfacePartition partition;
    build_interface_partition(metadata, incidence, partition);
    const InterfacePartitionValidation validation =
        validate_interface_partition(metadata, incidence, partition);

    REQUIRE(validation.ok);

    CHECK(partition.census.total_rows == 7);
    CHECK(partition.census.total_cols == 5);
    CHECK(partition.census.bulk_rows == 5);
    CHECK(partition.census.interface_rows == 2);
    CHECK(partition.census.unclassified_rows == 0);
    CHECK(partition.census.bulk_cols == 3);
    CHECK(partition.census.interface_cols == 2);
    CHECK(partition.census.unclassified_cols == 0);

    CHECK(partition.census.rows_vol == 3);
    CHECK(partition.census.rows_tau_bc == 2);
    CHECK(partition.census.rows_tau_match == 1);
    CHECK(partition.census.rows_global_int == 1);

    CHECK(partition.census.cols_field_interior_vol == 3);
    CHECK(partition.census.cols_field_matching == 1);
    CHECK(partition.census.cols_scalar_global == 1);

    REQUIRE(partition.domains.size() == 2);
    CHECK(partition.domains[0] == 0);
    CHECK(partition.domains[1] == 1);

    REQUIRE(partition.blocks.size() == 2);
    CHECK(partition.blocks[0].domain == 0);
    CHECK(partition.blocks[0].bulk_rows == std::vector<int>{0, 1, 2});
    CHECK(partition.blocks[0].bulk_cols == std::vector<int>{0, 1});
    CHECK(partition.blocks[1].domain == 1);
    CHECK(partition.blocks[1].bulk_rows == std::vector<int>{3, 4});
    CHECK(partition.blocks[1].bulk_cols == std::vector<int>{2});

    CHECK(partition.interface_rows == std::vector<int>{5, 6});
    CHECK(partition.interface_cols == std::vector<int>{3, 4});

    CHECK(partition.row_role[0] == 0);
    CHECK(partition.row_role[5] == 1);
    CHECK(partition.row_role[6] == 1);
}

TEST_CASE("Interface partition flags Unknown rows as unclassified",
          "[interface-partition]")
{
    TaggedJacobianMetadata metadata;
    metadata.nrows = 3;
    metadata.ncols = 1;
    metadata.rows = {
        make_row(0, RowTaxonomy::Vol, 0),
        make_row(1, RowTaxonomy::Unknown, 0),
        make_row(2, RowTaxonomy::Vol, -1),
    };
    metadata.columns = {make_col(0, ColumnClass::FieldInteriorVol, 0)};
    metadata.rows_per_column.resize(metadata.ncols);

    IncidenceColumnPartition incidence;
    incidence.col_role = {0};
    incidence.bulk_cols = {0};
    incidence.domains = {0};

    InterfacePartition partition;
    build_interface_partition(metadata, incidence, partition);
    const InterfacePartitionValidation validation =
        validate_interface_partition(metadata, incidence, partition);

    CHECK(partition.census.unclassified_rows == 2);
    CHECK_FALSE(validation.ok);
    CHECK_FALSE(validation.errors.empty());
}

TEST_CASE("Interface partition TauBcInterface policy moves TauBc rows to interface",
          "[interface-partition][tau-bc-interface]")
{
    TaggedJacobianMetadata metadata;
    metadata.nrows = 4;
    metadata.ncols = 2;
    metadata.rows = {
        make_row(0, RowTaxonomy::Vol, 0),
        make_row(1, RowTaxonomy::Vol, 0),
        make_row(2, RowTaxonomy::TauBc, 0),
        make_row(3, RowTaxonomy::TauMatch, 0, 1),
    };
    metadata.columns = {
        make_col(0, ColumnClass::FieldInteriorVol, 0),
        make_col(1, ColumnClass::FieldInteriorVol, 0),
    };
    metadata.rows_per_column.resize(metadata.ncols);

    IncidenceColumnPartition incidence;
    incidence.col_role = {0, 0};
    incidence.bulk_cols = {0, 1};
    incidence.transfer_cols = {};
    incidence.domains = {0};

    InterfacePartition partition;
    build_interface_partition(metadata, incidence, partition,
                              InterfaceRowPolicy::TauBcInterface);

    CHECK(partition.policy == InterfaceRowPolicy::TauBcInterface);
    CHECK(partition.census.bulk_rows == 2);
    CHECK(partition.census.interface_rows == 2);
    CHECK(partition.row_role[0] == 0);
    CHECK(partition.row_role[1] == 0);
    CHECK(partition.row_role[2] == 1);  // TauBc -> interface under this policy.
    CHECK(partition.row_role[3] == 1);
    CHECK(partition.interface_rows == std::vector<int>{2, 3});
    REQUIRE(partition.blocks.size() == 1);
    CHECK(partition.blocks[0].bulk_rows == std::vector<int>{0, 1});
    CHECK(partition.blocks[0].bulk_cols == std::vector<int>{0, 1});
    REQUIRE(partition.shapes.size() == 1);
    CHECK(partition.shapes[0].square);
    CHECK(partition.all_square);
}

TEST_CASE("FieldOuterShellTau col follows incidence transfer role",
          "[interface-partition][outer-shell-tau]")
{
    // FieldOuterShellTau is diagnostic metadata here; the partition must not
    // promote it when IncidenceColumnPartition marks it as transfer.
    TaggedJacobianMetadata metadata;
    metadata.nrows = 2;
    metadata.ncols = 2;
    metadata.rows = {
        make_row(0, RowTaxonomy::Vol, 0),
        make_row(1, RowTaxonomy::TauBc, 0),
    };
    metadata.columns = {
        make_col(0, ColumnClass::FieldInteriorVol, 0),
        make_col(1, ColumnClass::FieldOuterShellTau, 0),
    };
    metadata.rows_per_column.resize(metadata.ncols);

    // IncidenceColumnPartition's bulk filter only matches FieldInteriorVol;
    // FieldOuterShellTau falls into transfer_cols by default.
    IncidenceColumnPartition incidence;
    incidence.col_role = {0, 1};
    incidence.bulk_cols = {0};
    incidence.transfer_cols = {1};
    incidence.domains = {0};

    InterfacePartition partition;
    build_interface_partition(metadata, incidence, partition,
                              InterfaceRowPolicy::TauBcInterface);
    const InterfacePartitionValidation validation =
        validate_interface_partition(metadata, incidence, partition);

    REQUIRE(validation.ok);
    CHECK(partition.census.bulk_rows == 1);
    CHECK(partition.census.interface_rows == 1);
    CHECK(partition.census.bulk_cols == 1);
    CHECK(partition.census.interface_cols == 1);
    CHECK(partition.census.cols_field_outer_shell_tau == 1);
    REQUIRE(partition.shapes.size() == 1);
    CHECK(partition.shapes[0].square);
    CHECK(partition.all_square);
}

TEST_CASE("Interface partition TauBcBulk default keeps TauBc in bulk",
          "[interface-partition][tau-bc-bulk]")
{
    TaggedJacobianMetadata metadata;
    metadata.nrows = 2;
    metadata.ncols = 1;
    metadata.rows = {
        make_row(0, RowTaxonomy::Vol, 0),
        make_row(1, RowTaxonomy::TauBc, 0),
    };
    metadata.columns = {make_col(0, ColumnClass::FieldInteriorVol, 0)};
    metadata.rows_per_column.resize(metadata.ncols);

    IncidenceColumnPartition incidence;
    incidence.col_role = {0};
    incidence.bulk_cols = {0};
    incidence.domains = {0};

    InterfacePartition partition;
    build_interface_partition(metadata, incidence, partition);

    CHECK(partition.policy == InterfaceRowPolicy::TauBcBulk);
    CHECK(partition.census.bulk_rows == 2);
    CHECK(partition.census.interface_rows == 0);
    CHECK(partition.row_role[1] == 0);  // TauBc -> bulk under default policy.
    CHECK_FALSE(partition.all_square);  // 2 rows x 1 col.
}

TEST_CASE("Interface partition census dump emits headline counts", "[interface-partition]")
{
    TaggedJacobianMetadata metadata;
    metadata.nrows = 2;
    metadata.ncols = 2;
    metadata.rows = {
        make_row(0, RowTaxonomy::Vol, 0),
        make_row(1, RowTaxonomy::TauMatch, 0, 1),
    };
    metadata.columns = {
        make_col(0, ColumnClass::FieldInteriorVol, 0),
        make_col(1, ColumnClass::FieldMatching, 0),
    };
    metadata.rows_per_column.resize(metadata.ncols);

    IncidenceColumnPartition incidence;
    incidence.col_role = {0, 1};
    incidence.bulk_cols = {0};
    incidence.transfer_cols = {1};
    incidence.domains = {0};

    InterfacePartition partition;
    build_interface_partition(metadata, incidence, partition);
    const InterfacePartitionValidation validation =
        validate_interface_partition(metadata, incidence, partition);

    std::ostringstream os;
    dump_interface_partition_census(metadata, partition, validation, os);
    const std::string out = os.str();

    CHECK(out.find("Interface Partition Census") != std::string::npos);
    CHECK(out.find("Validation: ok") != std::string::npos);
    CHECK(out.find("bulk=1") != std::string::npos);
    CHECK(out.find("interface=1") != std::string::npos);
    CHECK(out.find("Vol=1") != std::string::npos);
    CHECK(out.find("TauMatch=1") != std::string::npos);
    CHECK(out.find("FieldInteriorVol=1") != std::string::npos);
}
