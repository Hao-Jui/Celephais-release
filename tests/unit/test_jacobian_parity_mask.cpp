#include <catch2/catch_test_macros.hpp>

#include "For_Kadath/Array/point.hpp"
#include "For_Kadath/Domain/bispheric_nosym.hpp"
#include "For_Kadath/Domain/spheric_nosym.hpp"
#include "For_Kadath/Scalar/scalar.hpp"
#include "For_Kadath/Space/bin_ns.hpp"
#include "For_Kadath/Space/bin_ns_nosym.hpp"
#include "For_Kadath/Space/space.hpp"
#include "For_Kadath/System_of_eqs/system_of_eqs.hpp"
#include "For_Kadath/Val_domain/val_domain.hpp"
#include "Linear_algebra/jacobian_assembler.hpp"
#include "Linear_algebra/jacobian_parity_mask.hpp"
#include "Linear_algebra/jacobian_parity_mass.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <memory>
#include <numeric>
#include <sstream>
#include <span>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

using namespace Kadath;

namespace
{
    class CoutCapture
    {
      public:
        CoutCapture() : original_(std::cout.rdbuf(output_.rdbuf())) {}
        ~CoutCapture() { std::cout.rdbuf(original_); }

        std::string str() const { return output_.str(); }

      private:
        std::ostringstream output_;
        std::streambuf* original_;
    };

    class UnsupportedTauSeedDomain final : public Domain_nucleus_nosym
    {
      public:
        using Domain_nucleus_nosym::Domain_nucleus_nosym;

        bool describe_tau_seed_block(
            const Tensor&, int,
            std::vector<TauSeedDescriptor>& descriptors) const override
        {
            descriptors.clear();
            return false;
        }
    };

    // A deliberately unsupported domain keeps the refusal-path regression
    // stable when R7 teaches every production symmetric domain this API.
    class UnsupportedTauSeedSpace : public Space
    {
      public:
        UnsupportedTauSeedSpace()
        {
            Point center(3);
            center.set(1) = 0.0;
            center.set(2) = 0.0;
            center.set(3) = 0.0;
            Dim_array resolution(3);
            resolution.set(0) = 5;
            resolution.set(1) = 5;
            resolution.set(2) = 4;

            nbr_domains = 1;
            ndim = 3;
            type_base = CHEB_TYPE;
            domains = new Domain*[1];
            domains[0] = new UnsupportedTauSeedDomain(
                0, CHEB_TYPE, 1.0, center, resolution);
        }

        ~UnsupportedTauSeedSpace() override = default;
    };

    class UnsupportedVariableDomainLayoutSpace final
        : public UnsupportedTauSeedSpace
    {
      public:
        int nbr_unknowns_from_variable_domains() const override { return 1; }

        bool describe_variable_domain_blocks(
            std::vector<VariableDomainBlock>& blocks) const override
        {
            blocks.clear();
            return false;
        }
    };

    class TemporaryReport
    {
      public:
        explicit TemporaryReport(const void* identity)
            : path(std::filesystem::temp_directory_path() /
                   ("kadath_parity_mass_unavailable_" +
                    std::to_string(reinterpret_cast<std::uintptr_t>(identity)) +
                    ".csv"))
        {
            std::error_code error;
            std::filesystem::remove(path, error);
        }

        ~TemporaryReport()
        {
            std::error_code error;
            std::filesystem::remove(path, error);
        }

        std::filesystem::path path;
    };

    // Two symmetric and two antisymmetric columns, the layout every engaged
    // decision needs: as many symmetric rows as symmetric columns.
    JacobianParityMaskState square_state()
    {
        JacobianParityMaskState state;
        state.n = 4;
        state.column_sector = {1, 1, -1, -1};
        return state;
    }

    JacobianParityRowPrediction complete_row_prediction(
        std::vector<signed char> sector = {+1, +1, -1, -1})
    {
        JacobianParityRowPrediction prediction;
        prediction.sector = std::move(sector);
        prediction.all_rows_available = true;
        return prediction;
    }

    JacobianParityColumnGrading complete_column_grading(
        std::vector<signed char> sector = {+1, +1, -1, -1})
    {
        JacobianParityColumnGrading grading;
        grading.sector = std::move(sector);
        return grading;
    }

    Dim_array parity_test_resolution()
    {
        Dim_array resolution(3);
        resolution.set(0) = 9;
        resolution.set(1) = 9;
        resolution.set(2) = 14;
        return resolution;
    }

    Dim_array symmetric_bispheric_parity_test_resolution()
    {
        Dim_array resolution = parity_test_resolution();
        resolution.set(2) = 15;
        return resolution;
    }

    Point parity_test_center()
    {
        Point center(3);
        center.set(1) = 0.0;
        center.set(2) = 0.0;
        center.set(3) = 0.0;
        return center;
    }

    void check_phi_parity_table(const Domain& domain, const char* label,
                                const std::array<int, 14>& expected,
                                int phi_basis = COSSIN)
    {
        for (std::size_t k = 0; k < expected.size(); ++k) {
            INFO("domain=" << label << " k=" << k);
            if (phi_basis == COSSIN) {
                CHECK(domain.phi_coefficient_parity(static_cast<int>(k)) ==
                      expected[k]);
            } else {
                CHECK(domain.phi_coefficient_parity(
                          static_cast<int>(k), phi_basis) == expected[k]);
            }
        }
    }

    Dim_array bns_test_resolution(int theta, int phi)
    {
        Dim_array resolution(3);
        resolution.set(0) = 5;
        resolution.set(1) = theta;
        resolution.set(2) = phi;
        return resolution;
    }

    std::vector<Dim_array> unequal_bns_surface_resolutions(int bispheric_phi)
    {
        std::vector<Dim_array> resolutions;
        resolutions.reserve(12);
        for (int domain = 0; domain < 12; ++domain) {
            const bool bispheric = domain >= 6 && domain <= 10;
            resolutions.push_back(bns_test_resolution(5, bispheric ? bispheric_phi : 4));
        }
        // The two members of each adapted pair share one surface grid.  Give
        // star 2 a different surface block so an offset or owner-domain error
        // cannot pass by comparing two accidentally identical tables.
        resolutions[4] = bns_test_resolution(7, 6);
        resolutions[5] = bns_test_resolution(7, 6);
        return resolutions;
    }

    int probed_phi_index(const Val_domain& value, int phi_size)
    {
        if (value.check_if_zero())
            return -1;
        const Array<double>& coefficients = value.get_coef_ref();
        int found = -1;
        for (int offset = 0; offset < coefficients.get_nbr(); ++offset) {
            if (coefficients.get_data()[offset] == 0.0)
                continue;
            const int phi = offset % phi_size;
            if (found >= 0 && found != phi)
                return -2;
            found = phi;
        }
        return found;
    }

    std::vector<int> directly_probe_surface_block(const Domain& domain,
                                                  int column_count)
    {
        Array<double> one_hot(column_count);
        std::vector<int> result(static_cast<std::size_t>(column_count), -1);
        const int phi_size = domain.get_nbr_coefs()(2);
        for (int column = 0; column < column_count; ++column) {
            one_hot = 0.0;
            one_hot.set(column) = 1.0;
            Val_domain shape(&domain);
            int position = 0;
            domain.xx_to_vars_from_adapted(shape, one_hot, position);
            REQUIRE(position == column_count);
            result[static_cast<std::size_t>(column)] =
                probed_phi_index(shape, phi_size);
        }
        return result;
    }
} // namespace

TEST_CASE("spheric_nosym phi coefficients follow phi-to-minus-phi parity",
          "[parity_mask]")
{
    Domain_nucleus_nosym domain(0, CHEB_TYPE, 1.0, parity_test_center(),
                                parity_test_resolution());
    check_phi_parity_table(
        domain, "spheric_nucleus_nosym",
        {+1, -1, +1, -1, +1, -1, +1, -1, +1, -1, +1, -1, +1, -1});
    check_phi_parity_table(
        domain, "spheric_nucleus_nosym_cos",
        {+1, +1, +1, +1, +1, +1, +1, +1, +1, +1, +1, +1, +1, +1},
        COS);
    check_phi_parity_table(
        domain, "spheric_nucleus_nosym_sin",
        {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        SIN);
    CHECK(domain.phi_coefficient_parity(0, CHEB) == 0);
}

TEST_CASE("bispheric_nosym phi coefficients follow pi-minus-phi parity",
          "[parity_mask]")
{
    const Dim_array resolution = parity_test_resolution();
    Domain_bispheric_rect_nosym rect(
        0, CHEB_TYPE, 1.0, 2.0, -0.5, 0.5, 0.5, resolution);
    Domain_bispheric_chi_first_nosym chi_first(
        0, CHEB_TYPE, 1.0, 0.5, 2.0, 1.5, resolution);
    Domain_bispheric_eta_first_nosym eta_first(
        0, CHEB_TYPE, 1.0, 2.0, -0.5, 0.5, resolution);
    const std::array<int, 14> expected = {
        +1, -1, -1, +1, +1, -1, -1, +1, +1, -1, -1, +1, +1, -1};

    check_phi_parity_table(rect, "bispheric_rect_nosym", expected);
    check_phi_parity_table(chi_first, "bispheric_chi_first_nosym", expected);
    check_phi_parity_table(eta_first, "bispheric_eta_first_nosym", expected);
    CHECK(rect.phi_coefficient_parity(0, COS) == 0);
    CHECK(chi_first.phi_coefficient_parity(0, SIN) == 0);
    CHECK(eta_first.phi_coefficient_parity(0, CHEB) == 0);
}

TEST_CASE("symmetric bispheric phi coefficients use their COS or SIN basis",
          "[parity_mask][sym][domain-bispheric-rect]")
{
    Domain_bispheric_rect rect(
        0, CHEB_TYPE, 1.0, 2.0, -0.5, 0.5, 0.5,
        symmetric_bispheric_parity_test_resolution());
    check_phi_parity_table(
        rect, "bispheric_rect_cos",
        {+1, -1, +1, -1, +1, -1, +1, -1, +1, -1, +1, -1, +1, -1},
        COS);
    check_phi_parity_table(
        rect, "bispheric_rect_sin",
        {-1, +1, -1, +1, -1, +1, -1, +1, -1, +1, -1, +1, -1, +1},
        SIN);
    CHECK(rect.phi_coefficient_parity(0, COSSIN) == 0);
}

TEST_CASE("symmetric bispheric chi-first phi coefficients use their COS or SIN basis",
          "[parity_mask][sym][domain-bispheric-chi-first]")
{
    Domain_bispheric_chi_first chi_first(
        0, CHEB_TYPE, 1.0, 0.5, 2.0, 1.5,
        symmetric_bispheric_parity_test_resolution());
    check_phi_parity_table(
        chi_first, "bispheric_chi_first_cos",
        {+1, -1, +1, -1, +1, -1, +1, -1, +1, -1, +1, -1, +1, -1},
        COS);
    check_phi_parity_table(
        chi_first, "bispheric_chi_first_sin",
        {-1, +1, -1, +1, -1, +1, -1, +1, -1, +1, -1, +1, -1, +1},
        SIN);
    CHECK(chi_first.phi_coefficient_parity(0, COSSIN) == 0);
}

TEST_CASE("symmetric bispheric eta-first phi coefficients use their COS or SIN basis",
          "[parity_mask][sym][domain-bispheric-eta-first]")
{
    Domain_bispheric_eta_first eta_first(
        0, CHEB_TYPE, 1.0, 2.0, -0.5, 0.5,
        symmetric_bispheric_parity_test_resolution());
    check_phi_parity_table(
        eta_first, "bispheric_eta_first_cos",
        {+1, -1, +1, -1, +1, -1, +1, -1, +1, -1, +1, -1, +1, -1},
        COS);
    check_phi_parity_table(
        eta_first, "bispheric_eta_first_sin",
        {-1, +1, -1, +1, -1, +1, -1, +1, -1, +1, -1, +1, -1, +1},
        SIN);
    CHECK(eta_first.phi_coefficient_parity(0, COSSIN) == 0);
}

TEST_CASE("BNS-nosym parity grading follows both surface-column blocks",
          "[parity_mask]")
{
    const std::vector<double> star_bounds{0.5, 1.0, 2.0};
    const std::vector<double> outer_bounds{10.0};
    Space_bin_ns_nosym space(
        CHEB_TYPE, 12.0, star_bounds, star_bounds, outer_bounds,
        unequal_bns_surface_resolutions(6));

    const Domain* star1 = space.get_domain(space.ADAPTED1);
    const Domain* star2 = space.get_domain(space.ADAPTED2);
    const int star1_columns = star1->nbr_unknowns_from_adapted();
    const int star2_columns = star2->nbr_unknowns_from_adapted();
    REQUIRE(star1_columns != star2_columns);

    std::vector<VariableDomainBlock> blocks;
    REQUIRE(space.describe_variable_domain_blocks(blocks));
    REQUIRE(blocks.size() == 2);
    CHECK(blocks[0].domain == space.ADAPTED1);
    CHECK(blocks[0].first_column == 0);
    CHECK(blocks[0].column_count == star1_columns);
    CHECK(blocks[1].domain == space.ADAPTED2);
    CHECK(blocks[1].first_column == star1_columns);
    CHECK(blocks[1].column_count == star2_columns);

    System_of_eqs system(space);
    std::vector<int> phi_index;
    std::vector<int> phi_basis;
    std::vector<int> phi_domain;
    std::vector<int> component_index;
    int unsupported_tau_seed_domain = -1;
    bool unsupported_variable_domain_layout = false;
    REQUIRE(system.build_column_phi_and_component_indices(
        phi_index, phi_basis, phi_domain, component_index,
        unsupported_tau_seed_domain,
        unsupported_variable_domain_layout));
    CHECK_FALSE(unsupported_variable_domain_layout);
    CHECK(unsupported_tau_seed_domain == -1);
    REQUIRE(phi_index.size() ==
            static_cast<std::size_t>(star1_columns + star2_columns));

    const std::vector<int> expected_star1 =
        directly_probe_surface_block(*star1, star1_columns);
    const std::vector<int> expected_star2 =
        directly_probe_surface_block(*star2, star2_columns);
    CHECK(std::vector<int>(phi_index.begin(),
                           phi_index.begin() + star1_columns) == expected_star1);
    CHECK(std::vector<int>(phi_index.begin() + star1_columns,
                           phi_index.end()) == expected_star2);
    CHECK(std::all_of(phi_index.begin(), phi_index.end(),
                      [](int phi) { return phi >= 0; }));
    REQUIRE(phi_basis.size() == phi_index.size());
    CHECK(std::all_of(phi_basis.begin(), phi_basis.end(),
                      [](int basis) { return basis == COSSIN; }));
    REQUIRE(phi_domain.size() == phi_index.size());
    CHECK(std::all_of(phi_domain.begin(),
                      phi_domain.begin() + star1_columns,
                      [domain = space.ADAPTED1](int actual) {
                          return actual == domain;
                      }));
    CHECK(std::all_of(phi_domain.begin() + star1_columns, phi_domain.end(),
                      [domain = space.ADAPTED2](int actual) {
                          return actual == domain;
                      }));
    CHECK(std::all_of(component_index.begin(), component_index.end(),
                      [](int component) { return component == -1; }));
}

TEST_CASE("BNS symmetric surface-column blocks are contiguous and exhaustive",
          "[parity_mask][sym][variable-domain-blocks]")
{
    const std::vector<double> star_bounds{0.5, 1.0, 2.0};
    const std::vector<double> outer_bounds{10.0};
    Space_bin_ns space(
        CHEB_TYPE, 12.0, star_bounds, star_bounds, outer_bounds,
        unequal_bns_surface_resolutions(5));

    const int star1_columns =
        space.get_domain(space.ADAPTED1)->nbr_unknowns_from_adapted();
    const int star2_columns =
        space.get_domain(space.ADAPTED2)->nbr_unknowns_from_adapted();
    REQUIRE(star1_columns > 0);
    REQUIRE(star2_columns > 0);
    REQUIRE(star1_columns != star2_columns);

    std::vector<VariableDomainBlock> blocks;
    REQUIRE(space.describe_variable_domain_blocks(blocks));
    REQUIRE(blocks.size() == 2);
    CHECK(blocks[0].domain == space.ADAPTED1);
    CHECK(blocks[0].first_column == 0);
    CHECK(blocks[0].column_count == star1_columns);
    CHECK(blocks[1].domain == space.ADAPTED2);
    CHECK(blocks[1].first_column == star1_columns);
    CHECK(blocks[1].column_count == star2_columns);

    int next_column = 0;
    for (const VariableDomainBlock& block : blocks) {
        CHECK(block.first_column == next_column);
        next_column += block.column_count;
    }
    CHECK(next_column == space.nbr_unknowns_from_variable_domains());
    CHECK(next_column == star1_columns + star2_columns);

    System_of_eqs system(space);
    std::vector<int> phi_index;
    std::vector<int> phi_basis;
    std::vector<int> phi_domain;
    std::vector<int> component_index;
    int unsupported_tau_seed_domain = -1;
    bool unsupported_variable_domain_layout = false;
    REQUIRE(system.build_column_phi_and_component_indices(
        phi_index, phi_basis, phi_domain, component_index,
        unsupported_tau_seed_domain,
        unsupported_variable_domain_layout));
    CHECK_FALSE(unsupported_variable_domain_layout);
    CHECK(unsupported_tau_seed_domain == -1);
    REQUIRE(phi_basis.size() ==
            static_cast<std::size_t>(star1_columns + star2_columns));
    CHECK(std::all_of(phi_basis.begin(), phi_basis.end(),
                      [](int basis) { return basis == COSSIN; }));
    CHECK(std::all_of(phi_domain.begin(),
                      phi_domain.begin() + star1_columns,
                      [domain = space.ADAPTED1](int actual) {
                          return actual == domain;
                      }));
    CHECK(std::all_of(phi_domain.begin() + star1_columns, phi_domain.end(),
                      [domain = space.ADAPTED2](int actual) {
                          return actual == domain;
                      }));
}

TEST_CASE("parity field grading covers supported scalar fields", "[parity_mask]")
{
    for (const char* name : {"P", "N", "H", "xiScal", "varscal", "cstB"})
        CHECK(jacobian_parity_field_grading(name, -1) == +1);

    CHECK(jacobian_parity_field_grading("phi", -1) == -1);
    CHECK(jacobian_parity_field_grading("bet", 0) == -1);
    CHECK(jacobian_parity_field_grading("bet", 1) == +1);
    CHECK(jacobian_parity_field_grading("bet", 2) == -1);
    CHECK(jacobian_parity_field_grading("  H  ", -1) == +1);
    CHECK(jacobian_parity_field_grading("unknown", -1) == 0);
}

TEST_CASE("parity grading distinguishes odd numeric globals", "[parity_mask]")
{
    for (const char* name : {"ome", "xaxis", "Hc", "Hc1", "Hc2", "Hscale",
                             "Hscale1", "Hscale2", "Mb", "Madm", "qlMadm",
                             "qlMadm1", "qlMadm2", "omes1", "omes2", "muz1",
                             "muz2", "cstB"})
        CHECK(jacobian_parity_global_grading(name) == +1);

    for (const char* name : {"yaxis", "zvel", "adot"})
        CHECK(jacobian_parity_global_grading(name) == -1);

    // add_var stores fixed-width padded tokens in the column map.
    CHECK(jacobian_parity_global_grading("  yaxis  ") == -1);
    CHECK(jacobian_parity_global_grading("unknown") == 0);
}

TEST_CASE("parity column grading fails closed without user log chatter", "[parity_mask]")
{
    SECTION("a mixed-phi column disables silently") {
        JacobianParityColumnGrading grading;
        grading.mixed_phi_columns = 1;
        const std::string reason =
            jacobian_parity_column_grading_disable_reason(grading);
        REQUIRE(reason == "1 column has mixed phi indices");

        JacobianParityMaskState state = square_state();
        std::string output;
        {
            CoutCapture capture;
            disable_jacobian_parity_mask(state, reason, 0);
            output = capture.str();
        }
        REQUIRE(output.empty());
        REQUIRE(state.decision ==
                JacobianParityMaskState::Decision::Disabled);
        REQUIRE(state.column_sector.empty());
        REQUIRE(state.row_sector.empty());
    }

    SECTION("unsupported tau seeding takes precedence over other failures") {
        JacobianParityColumnGrading grading;
        grading.unsupported_tau_seed_domain = 7;
        grading.mixed_phi_columns = 2;
        grading.ungraded_columns = 3;
        REQUIRE(jacobian_parity_column_grading_disable_reason(grading) ==
                "domain 7 does not describe its tau seed block");
    }

    SECTION("unsupported variable-domain layouts fail closed") {
        JacobianParityColumnGrading grading;
        grading.unsupported_variable_domain_layout = true;
        grading.unsupported_tau_seed_domain = 7;
        REQUIRE(jacobian_parity_column_grading_disable_reason(grading) ==
                "space does not describe its variable-domain column blocks");
    }

    SECTION("ungraded names retain their existing reason") {
        JacobianParityColumnGrading grading;
        grading.ungraded_columns = 2;
        grading.ungraded_names = {"first", "second"};
        REQUIRE(jacobian_parity_column_grading_disable_reason(grading) ==
                "2 columns have no y parity: first second");
    }

    SECTION("unsupported phi bases fail closed with one reason") {
        JacobianParityColumnGrading grading;
        grading.unsupported_phi_basis_columns = 1;
        REQUIRE(jacobian_parity_column_grading_disable_reason(grading) ==
                "1 column has an unsupported or ambiguous phi basis");

        grading.unsupported_phi_basis_columns = 2;
        REQUIRE(jacobian_parity_column_grading_disable_reason(grading) ==
                "2 columns have unsupported or ambiguous phi bases");
    }

    SECTION("a complete grading has no disable reason") {
        REQUIRE(jacobian_parity_column_grading_disable_reason({}).empty());
    }

    SECTION("rank one does not print the disable reason") {
        JacobianParityMaskState state = square_state();
        std::string output;
        {
            CoutCapture capture;
            disable_jacobian_parity_mask(state, "test reason", 1);
            output = capture.str();
        }
        REQUIRE(output.empty());
    }
}

TEST_CASE("unsupported tau-seed domains disable grading and parity diagnostics",
          "[parity_mask]")
{
    UnsupportedTauSeedSpace space;
    Scalar field(space);
    field = 1.0;
    field.std_base();
    System_of_eqs system(space, 0, 0);
    system.add_var("P", field);

    JacobianParityColumnGrading grading;
    REQUIRE_NOTHROW(grading = grade_jacobian_parity_columns(system));
    REQUIRE(grading.unsupported_tau_seed_domain == 0);
    REQUIRE(grading.sector.empty());
    REQUIRE(jacobian_parity_column_grading_disable_reason(grading) ==
            "domain 0 does not describe its tau seed block");

    AssembledJacobianCoo coo;
    coo.n = system.get_nbr_unknowns();
    TemporaryReport report(&system);
    REQUIRE_NOTHROW(jacobian_parity_mass_report(
        system, coo, report.path.string()));

    std::ifstream input(report.path);
    REQUIRE(input.good());
    const std::string contents{std::istreambuf_iterator<char>{input},
                               std::istreambuf_iterator<char>{}};
    REQUIRE(contents.find("status,unavailable\n") != std::string::npos);
    REQUIRE(contents.find(
                "reason,domain 0 does not describe its tau seed block\n") !=
            std::string::npos);
}

TEST_CASE("Jacobian domain census preserves dimensions and stored-entry totals",
          "[parity_mask][domain_census]")
{
    std::vector<RowMetadata> rows(5);
    const std::array<int, 5> row_domains = {0, 0, 1, 1, -1};
    for (std::size_t row = 0; row < rows.size(); ++row) {
        rows[row].row = static_cast<int>(row);
        rows[row].dom = row_domains[row];
        rows[row].taxonomy = row == 1 || row == 3
                                 ? RowTaxonomy::TauMatch
                                 : (row == 4 ? RowTaxonomy::GlobalInt
                                             : RowTaxonomy::Vol);
    }

    std::vector<ColumnMetadata> columns(5);
    const std::array<int, 5> column_domains = {0, 0, 1, 1, -1};
    for (std::size_t column = 0; column < columns.size(); ++column) {
        columns[column].column = static_cast<int>(column);
        columns[column].domain = column_domains[column];
    }

    AssembledJacobianCoo coo;
    coo.n = 5;
    coo.nnz = 8;
    // The duplicate (1,1) verifies that this is a stored-entry census, not a
    // deduplicated mathematical-sparsity count.
    coo.irn = {1, 1, 2, 2, 3, 4, 4, 5};
    coo.jcn = {1, 1, 3, 5, 2, 1, 4, 5};

    const auto census = jacobian_parity_mass_detail::build_domain_census(
        2, rows, columns, coo);
    REQUIRE(census.domain_count == 2);
    CHECK(census.row_dimensions == std::vector<long long>{2, 2, 1});
    CHECK(census.column_dimensions == std::vector<long long>{2, 2, 1});

    const std::size_t buckets = 3;
    const auto entry = [&](const std::vector<long long>& table, int row_domain,
                           int column_domain) {
        const std::size_t row = row_domain < 0 ? 2 : static_cast<std::size_t>(row_domain);
        const std::size_t column =
            column_domain < 0 ? 2 : static_cast<std::size_t>(column_domain);
        return table[row * buckets + column];
    };
    CHECK(entry(census.stored_coo_entries, 0, 0) == 2);
    CHECK(entry(census.stored_coo_entries, 0, 1) == 1);
    CHECK(entry(census.stored_coo_entries, 0, -1) == 1);
    CHECK(entry(census.stored_coo_entries, 1, 0) == 2);
    CHECK(entry(census.stored_coo_entries, 1, 1) == 1);
    CHECK(entry(census.stored_coo_entries, -1, -1) == 1);
    CHECK(entry(census.matching_stored_coo_entries, 0, 1) == 1);
    CHECK(entry(census.matching_stored_coo_entries, 0, -1) == 1);
    CHECK(entry(census.matching_stored_coo_entries, 1, 0) == 1);
    CHECK(entry(census.matching_stored_coo_entries, 1, 1) == 1);
    CHECK(std::accumulate(census.stored_coo_entries.begin(),
                          census.stored_coo_entries.end(), 0LL) == coo.nnz);
    CHECK(std::accumulate(census.matching_stored_coo_entries.begin(),
                          census.matching_stored_coo_entries.end(), 0LL) == 4);

    SECTION("out-of-range COO coordinates fail closed") {
        coo.jcn[0] = 0;
        CHECK_THROWS(jacobian_parity_mass_detail::build_domain_census(
            2, rows, columns, coo));
    }
    SECTION("invalid metadata domains fail closed") {
        rows[0].dom = -2;
        CHECK_THROWS(jacobian_parity_mass_detail::build_domain_census(
            2, rows, columns, coo));
    }
}

TEST_CASE("unsupported variable-domain layouts disable grading",
          "[parity_mask]")
{
    UnsupportedVariableDomainLayoutSpace space;
    System_of_eqs system(space);

    JacobianParityColumnGrading grading;
    REQUIRE_NOTHROW(grading = grade_jacobian_parity_columns(system));
    REQUIRE(grading.unsupported_variable_domain_layout);
    REQUIRE(grading.unsupported_tau_seed_domain == -1);
    REQUIRE(grading.sector.empty());
    REQUIRE(jacobian_parity_column_grading_disable_reason(grading) ==
            "space does not describe its variable-domain column blocks");
}

TEST_CASE("parity row grading follows the dominant sector mass", "[parity_mask]")
{
    JacobianParityMaskState state = square_state();
    // Row 1 is antisymmetric, row 3 is a tie: ties fall to the symmetric side
    // so an empty row never lands in the sector with no mass at all.
    derive_jacobian_parity_row_sectors(state, {1.0, 1e-9, 4.0, 0.0},
                                       {1e-12, 3.0, 0.0, 0.0});
    REQUIRE(state.row_sector == std::vector<signed char>{1, -1, 1, 1});
}

TEST_CASE("full-J parity row grading uses structural rows only when exact",
          "[parity_mask][residual_row_descriptor]")
{
    const std::vector<signed char> column_sector = {+1, -1, +1, -1};
    const std::vector<double> symmetric_mass = {9.0, 8.0, 1.0, 0.0};
    const std::vector<double> antisymmetric_mass = {0.0, 1.0, 8.0, 7.0};
    const std::vector<signed char> matrix_grading = {+1, +1, -1, -1};

    JacobianParityRowPrediction prediction =
        complete_row_prediction({+1, -1, +1, -1});
    JacobianParityRowGradingSelection selection =
        select_jacobian_parity_row_grading(
            prediction, column_sector, symmetric_mass, antisymmetric_mass);
    CHECK(selection.source ==
          JacobianParityRowGradingSelection::Source::Structural);
    CHECK(selection.sector == prediction.sector);
    CHECK(selection.sector != matrix_grading);
    CHECK(selection.fallback_reason.empty());
    CHECK_FALSE(regrade_jacobian_parity_rows_after_structural_measurement(
        selection, 1e-14, 1.0, symmetric_mass, antisymmetric_mass));
    CHECK(selection.source ==
          JacobianParityRowGradingSelection::Source::Structural);
    CHECK(selection.sector == prediction.sector);

    JacobianParityRowGradingSelection boundary_selection =
        select_jacobian_parity_row_grading(
            prediction, column_sector, symmetric_mass, antisymmetric_mass);
    REQUIRE(regrade_jacobian_parity_rows_after_structural_measurement(
        boundary_selection, jacobian_parity_cross_tolerance, 1.0,
        symmetric_mass, antisymmetric_mass));
    CHECK(boundary_selection.source ==
          JacobianParityRowGradingSelection::Source::MatrixDerivedSecondPass);

    REQUIRE(regrade_jacobian_parity_rows_after_structural_measurement(
        selection, 1e-6, 1.0, symmetric_mass, antisymmetric_mass));
    CHECK(selection.source ==
          JacobianParityRowGradingSelection::Source::MatrixDerivedSecondPass);
    CHECK(selection.sector == matrix_grading);
    JacobianParityMaskState approximate_state;
    approximate_state.n = 4;
    approximate_state.column_sector = column_sector;
    approximate_state.row_sector = selection.sector;
    decide_jacobian_parity_mask(approximate_state, 3.7e-7, 1.0, 1);
    CHECK(approximate_state.decision ==
          JacobianParityMaskState::Decision::Engaged);

    prediction.all_rows_available = false;
    prediction.unavailable_rows = 1;
    selection = select_jacobian_parity_row_grading(
        prediction, column_sector, symmetric_mass, antisymmetric_mass);
    CHECK(selection.source ==
          JacobianParityRowGradingSelection::Source::MatrixDerivedFallback);
    CHECK(selection.sector == matrix_grading);
    CHECK(selection.fallback_reason ==
          "structural residual-row descriptors are incomplete "
          "(unavailable=1, ungraded=0, unsupported_phi_basis=0)");
    CHECK_FALSE(regrade_jacobian_parity_rows_after_structural_measurement(
        selection, 1.0, 1.0, symmetric_mass, antisymmetric_mass));
    CHECK(selection.source ==
          JacobianParityRowGradingSelection::Source::MatrixDerivedFallback);
    CHECK(selection.sector == matrix_grading);

    prediction.unavailable_rows = 0;
    prediction.ungraded_rows = 1;
    selection = select_jacobian_parity_row_grading(
        prediction, column_sector, symmetric_mass, antisymmetric_mass);
    CHECK(selection.source ==
          JacobianParityRowGradingSelection::Source::MatrixDerivedFallback);
    CHECK(selection.sector == matrix_grading);
    CHECK(selection.fallback_reason ==
          "structural residual-row descriptors are incomplete "
          "(unavailable=0, ungraded=1, unsupported_phi_basis=0)");
}

TEST_CASE("sector-reduction flag does not change a descriptor-covered parity mask",
          "[parity_mask][selection_plan][residual_row_descriptor]")
{
    const std::vector<signed char> column_sector = {+1, -1, +1, -1};
    const JacobianParityRowPrediction prediction =
        complete_row_prediction(column_sector);
    // Matrix mass deliberately gives a different row table.  This makes the
    // test sensitive to the former flag-dependent source switch.
    const std::vector<double> symmetric_mass = {9.0, 8.0, 1.0, 0.0};
    const std::vector<double> antisymmetric_mass = {0.0, 1.0, 8.0, 0.0};

    const auto masked_fixture = [&](bool sector_reduction_requested) {
        const JacobianParityRowGradingSelection selection =
            select_jacobian_parity_row_grading(
                prediction, column_sector, symmetric_mass,
                antisymmetric_mass);
        REQUIRE(selection.source ==
                JacobianParityRowGradingSelection::Source::Structural);

        JacobianParityMaskState state;
        state.n = 4;
        state.column_sector = column_sector;
        state.row_sector = selection.sector;
        // Exact structural grading is independent of the reduction flag.
        decide_jacobian_parity_mask(state, 1e-14, 1.0, 1);
        REQUIRE(state.decision == JacobianParityMaskState::Decision::Engaged);
        if (sector_reduction_requested) {
            decide_jacobian_parity_reduction(state, 1e-14, 1.0, 1);
            REQUIRE(state.reduction_decision ==
                    JacobianParityMaskState::ReductionDecision::Eligible);
        }

        std::vector<std::pair<int, int>> retained_entries;
        for (int row = 0; row < state.n; ++row) {
            for (int column = 0; column < state.n; ++column) {
                if (state.row_sector[static_cast<std::size_t>(row)] ==
                    state.column_sector[static_cast<std::size_t>(column)]) {
                    retained_entries.emplace_back(row, column);
                }
            }
        }
        return std::make_pair(state.row_sector, retained_entries);
    };

    const auto flag_off = masked_fixture(false);
    const auto flag_on = masked_fixture(true);
    CHECK(flag_off.first == flag_on.first);
    CHECK(flag_off.second == flag_on.second);
}

TEST_CASE("descriptor row oracle compares covered rows without claiming gaps",
          "[parity_mask][residual_row_descriptor]")
{
    JacobianParityRowPrediction prediction;
    prediction.sector = {1, 0, -1};
    prediction.unavailable_rows = 1;

    JacobianParityRowOracleComparison comparison =
        compare_jacobian_parity_row_prediction(prediction, {1, 1, 1});
    CHECK_FALSE(comparison.exact_on_covered_rows);
    CHECK_FALSE(comparison.whole_fixture_covered);
    CHECK(comparison.compared_rows == 2);
    CHECK(comparison.unavailable_rows == 1);
    CHECK(comparison.mismatched_rows == 1);
    CHECK(comparison.first_mismatch == 2);
    CHECK(comparison.failure_reason.empty());

    comparison = compare_jacobian_parity_row_prediction(
        prediction, {1, -1, -1});
    CHECK(comparison.exact_on_covered_rows);
    CHECK_FALSE(comparison.whole_fixture_covered);
    CHECK(comparison.mismatched_rows == 0);
    CHECK(comparison.first_mismatch == -1);

    comparison = compare_jacobian_parity_row_prediction(prediction, {1, -1});
    CHECK_FALSE(comparison.exact_on_covered_rows);
    CHECK(comparison.failure_reason ==
          "descriptor and matrix row-sector sizes differ");
}

TEST_CASE("parity mask engages only at roundoff-level coupling", "[parity_mask]")
{
    JacobianParityMaskState state = square_state();
    derive_jacobian_parity_row_sectors(state, {1.0, 1.0, 0.0, 0.0},
                                       {0.0, 0.0, 1.0, 1.0});
    decide_jacobian_parity_mask(state, 2.1e-9, 1.7e6, 1);
    REQUIRE(state.decision == JacobianParityMaskState::Decision::Engaged);
    REQUIRE(state.row_sector == std::vector<signed char>{1, 1, -1, -1});
}

TEST_CASE("parity mask declines a coupled Jacobian", "[parity_mask]")
{
    JacobianParityMaskState state = square_state();
    derive_jacobian_parity_row_sectors(state, {1.0, 1.0, 0.0, 0.0},
                                       {0.0, 0.0, 1.0, 1.0});
    decide_jacobian_parity_mask(state, 1.5e4, 1.7e6, 1);
    REQUIRE(state.decision == JacobianParityMaskState::Decision::Disabled);
    REQUIRE(state.row_sector.empty());
    REQUIRE(state.column_sector.empty());
}

TEST_CASE("parity mask fixed approximate tolerance has inclusive boundaries",
          "[parity_mask]")
{
    SECTION("the exact boundary approximate-engages with a distinct report") {
        JacobianParityMaskState state = square_state();
        derive_jacobian_parity_row_sectors(state, {1.0, 1.0, 0.0, 0.0},
                                           {0.0, 0.0, 1.0, 1.0});
        CoutCapture capture;
        decide_jacobian_parity_mask(
            state, jacobian_parity_cross_tolerance, 1.0, 0);
        REQUIRE(state.decision == JacobianParityMaskState::Decision::Engaged);
        // The engage summary is printed by the assembler as one combined line
        // after masking; the decision itself records the measurement silently.
        REQUIRE(state.approximate_engagement);
        REQUIRE(state.engaged_cross_ratio == jacobian_parity_cross_tolerance);
        REQUIRE(capture.str().empty());
    }

    SECTION("the approximate boundary engages") {
        JacobianParityMaskState state = square_state();
        derive_jacobian_parity_row_sectors(state, {1.0, 1.0, 0.0, 0.0},
                                           {0.0, 0.0, 1.0, 1.0});
        decide_jacobian_parity_mask(
            state, jacobian_parity_approximate_cross_tolerance, 1.0, 1);
        REQUIRE(state.decision == JacobianParityMaskState::Decision::Engaged);
    }

    SECTION("above the approximate boundary is refused") {
        JacobianParityMaskState state = square_state();
        derive_jacobian_parity_row_sectors(state, {1.0, 1.0, 0.0, 0.0},
                                           {0.0, 0.0, 1.0, 1.0});
        decide_jacobian_parity_mask(
            state,
            std::nextafter(jacobian_parity_approximate_cross_tolerance,
                           std::numeric_limits<double>::infinity()),
            1.0, 1);
        REQUIRE(state.decision == JacobianParityMaskState::Decision::Disabled);
    }
}

TEST_CASE("parity mask declines rectangular sectors", "[parity_mask]")
{
    JacobianParityMaskState state = square_state();
    // Three symmetric rows against two symmetric columns: the symmetric block
    // would be singular however small the cross entries are.
    derive_jacobian_parity_row_sectors(state, {1.0, 1.0, 1.0, 0.0},
                                       {0.0, 0.0, 0.0, 1.0});
    decide_jacobian_parity_mask(state, 0.0, 1.7e6, 0);
    REQUIRE(state.decision == JacobianParityMaskState::Decision::Disabled);
}

TEST_CASE("parity mask reports its own disable reason", "[parity_mask]")
{
    JacobianParityMaskState state = square_state();
    disable_jacobian_parity_mask(state, "test reason", 1);
    REQUIRE(state.decision == JacobianParityMaskState::Decision::Disabled);
    REQUIRE(state.column_sector.empty());
}

TEST_CASE("neutral Jacobian selection plans validate opaque two-block labels",
          "[parity_mask][selection_plan]")
{
    using Label = JacobianSelectionPlan::BlockLabel;

    SECTION("the selected square block exports ordered row and column lists") {
        const JacobianSelectionPlanBuild built = make_jacobian_selection_plan(
            std::vector<Label>{42, 7, 42, 7},
            std::vector<Label>{7, 42, 42, 7}, 42, 7);
        REQUIRE(built.plan);
        CHECK(built.fallback_reason.empty());
        CHECK(built.plan->selected_block() == 42);
        CHECK(built.plan->selected_rows() == std::vector<int>{0, 2});
        CHECK(built.plan->selected_columns() == std::vector<int>{1, 2});
    }

    SECTION("a rectangular selected block falls back") {
        const JacobianSelectionPlanBuild built = make_jacobian_selection_plan(
            std::vector<Label>{5, 5, 9, 9},
            std::vector<Label>{5, 9, 9, 9}, 5, 9);
        CHECK_FALSE(built.plan);
        CHECK(built.fallback_reason ==
              "selected block is not square (rows 2 vs columns 1)");
    }

    SECTION("a label outside the declared pair is malformed") {
        const JacobianSelectionPlanBuild built = make_jacobian_selection_plan(
            std::vector<Label>{5, 9}, std::vector<Label>{5, 0}, 5, 9);
        CHECK_FALSE(built.plan);
        CHECK(built.fallback_reason ==
              "column block label at index 1 is outside the declared two-block partition");
    }

    SECTION("row and column label tables must describe the same square system") {
        const JacobianSelectionPlanBuild built = make_jacobian_selection_plan(
            std::vector<Label>{5, 9}, std::vector<Label>{5}, 5, 9);
        CHECK_FALSE(built.plan);
        CHECK(built.fallback_reason ==
              "row and column block-label tables have different sizes");
    }

    SECTION("identical declared labels are malformed") {
        const JacobianSelectionPlanBuild built = make_jacobian_selection_plan(
            std::vector<Label>{5}, std::vector<Label>{5}, 5, 5);
        CHECK_FALSE(built.plan);
        CHECK(built.fallback_reason ==
              "selected and excluded block labels are identical");
    }
}

TEST_CASE("selection-plan vector wiring keeps row and column maps distinct",
          "[parity_mask][selection_plan][reduced_solve]")
{
    const std::array<double, 4> full{{10.0, 20.0, 30.0, 40.0}};
    const JacobianSelectedValues gathered = gather_jacobian_selected_values(
        full, {0, 3});
    REQUIRE(gathered);
    CHECK(gathered.values == std::vector<double>{10.0, 40.0});

    const std::array<double, 2> reduced{{7.0, 8.0}};
    const JacobianSelectedValues scattered = scatter_jacobian_selected_values(
        reduced, 4, {1, 2});
    REQUIRE(scattered);
    CHECK(scattered.values == std::vector<double>{0.0, 7.0, 8.0, 0.0});

    CHECK_FALSE(gather_jacobian_selected_values(full, {2, 2}));
    CHECK_FALSE(gather_jacobian_selected_values(full, {4}));
    CHECK_FALSE(gather_jacobian_selected_values(
        full, std::vector<int>{}));
    CHECK_FALSE(gather_jacobian_selected_values(full, {-1}));
    CHECK_FALSE(scatter_jacobian_selected_values(
        std::span<const double>{reduced}.first<1>(), 4, {1, 2}));
    CHECK_FALSE(scatter_jacobian_selected_values(
        std::span<const double>{}, 4, std::vector<int>{}));

    std::array<double, 4> nonfinite = full;
    nonfinite[3] = std::numeric_limits<double>::quiet_NaN();
    CHECK_FALSE(gather_jacobian_selected_values(
        nonfinite, {0, 3}));
    const std::array<double, 2> nonfinite_reduced{{
        1.0, std::numeric_limits<double>::infinity()}};
    CHECK_FALSE(scatter_jacobian_selected_values(
        nonfinite_reduced, 4, {1, 2}));
}

TEST_CASE("pre-J1 selection uses structural grading and the dual entry guard",
          "[parity_mask][selection_plan][reduced_solve]")
{
    const JacobianParityRowPrediction rows = complete_row_prediction();
    const JacobianParityColumnGrading columns = complete_column_grading();

    SECTION("absolute-floor branch includes its boundary") {
        std::array<double, 4> residual{{4.548e-4, 0.0, 1e-12, 0.0}};
        const JacobianPreJ1SelectionPlanBuild boundary =
            make_jacobian_pre_j1_selection_plan(
                rows, columns, residual.data(),
                static_cast<int>(residual.size()));
        REQUIRE(boundary);
        CHECK(boundary.entry_norms.active_linf == 4.548e-4);
        CHECK(boundary.entry_norms.forbidden_linf == 1e-12);
        CHECK(boundary.entry_limit ==
              jacobian_pre_j1_forbidden_absolute_floor);
        CHECK(boundary.plan->selected_rows() == std::vector<int>{0, 1});
        CHECK(boundary.plan->selected_columns() == std::vector<int>{0, 1});

        residual[2] = std::nextafter(
            jacobian_pre_j1_forbidden_absolute_floor,
            std::numeric_limits<double>::infinity());
        const JacobianPreJ1SelectionPlanBuild above =
            make_jacobian_pre_j1_selection_plan(
                rows, columns, residual.data(),
                static_cast<int>(residual.size()));
        CHECK_FALSE(above);
        CHECK(above.fallback_reason ==
              "entry-residual guard refused: forbidden Linf exceeds "
              "max(1e-10 * active Linf, 1e-12)");
    }

    SECTION("relative branch includes its boundary") {
        std::array<double, 4> residual{{20.0, 0.0, 2e-9, 0.0}};
        const JacobianPreJ1SelectionPlanBuild boundary =
            make_jacobian_pre_j1_selection_plan(
                rows, columns, residual.data(),
                static_cast<int>(residual.size()));
        REQUIRE(boundary);
        CHECK(boundary.entry_limit == 2e-9);
        CHECK(boundary.entry_norms.forbidden_linf == boundary.entry_limit);

        residual[2] = std::nextafter(
            boundary.entry_limit, std::numeric_limits<double>::infinity());
        CHECK_FALSE(make_jacobian_pre_j1_selection_plan(
            rows, columns, residual.data(),
            static_cast<int>(residual.size())));
    }

    SECTION("certified warm-entry scale is accepted by the floor") {
        const std::array<double, 4> residual{{4.548e-4, 0.0, 2.5e-14, 0.0}};
        const JacobianPreJ1SelectionPlanBuild built =
            make_jacobian_pre_j1_selection_plan(
                rows, columns, residual.data(),
                static_cast<int>(residual.size()));
        REQUIRE(built);
        CHECK(built.entry_limit == 1e-12);
    }
}

TEST_CASE("pre-J1 structural selection fails closed",
          "[parity_mask][selection_plan][reduced_solve][failure]")
{
    const std::array<double, 4> residual{{1.0, 0.0, 0.0, 0.0}};

    SECTION("any unavailable row descriptor refuses the plan") {
        JacobianParityRowPrediction rows = complete_row_prediction();
        rows.all_rows_available = false;
        rows.unavailable_rows = 1;
        const auto built = make_jacobian_pre_j1_selection_plan(
            rows, complete_column_grading(), residual.data(), 4);
        CHECK_FALSE(built);
        CHECK(built.fallback_reason ==
              "structural residual-row descriptors are unavailable");
    }

    SECTION("any ungradable column refuses the plan") {
        JacobianParityColumnGrading columns = complete_column_grading();
        columns.ungraded_columns = 1;
        columns.ungraded_names = {"mystery"};
        const auto built = make_jacobian_pre_j1_selection_plan(
            complete_row_prediction(), columns, residual.data(), 4);
        CHECK_FALSE(built);
        CHECK(built.fallback_reason ==
              "Jacobian columns are not all gradable: 1 columns have no y parity: mystery");
    }

    SECTION("row, column, and residual counts must agree") {
        const auto built = make_jacobian_pre_j1_selection_plan(
            complete_row_prediction({+1, +1, -1}),
            complete_column_grading(), residual.data(), 4);
        CHECK_FALSE(built);
        CHECK(built.fallback_reason ==
              "structural row, column, and residual counts do not match");
    }

    SECTION("active row and column counts must be equal") {
        const auto built = make_jacobian_pre_j1_selection_plan(
            complete_row_prediction({+1, +1, +1, -1}),
            complete_column_grading(), residual.data(), 4);
        CHECK_FALSE(built);
        CHECK(built.fallback_reason ==
              "selected block is not square (rows 3 vs columns 2)");
    }

    SECTION("labels outside the two structural sectors are malformed") {
        const auto built = make_jacobian_pre_j1_selection_plan(
            complete_row_prediction({+1, 0, -1, -1}),
            complete_column_grading(), residual.data(), 4);
        CHECK_FALSE(built);
        CHECK(built.fallback_reason ==
              "row block label at index 1 is outside the declared two-block partition");
    }

    SECTION("null residual storage refuses the plan") {
        const auto built = make_jacobian_pre_j1_selection_plan(
            complete_row_prediction(), complete_column_grading(), nullptr, 4);
        CHECK_FALSE(built);
        CHECK(built.fallback_reason ==
              "entry-residual guard cannot be evaluated: full vector storage is null");
    }

    SECTION("a non-finite entry residual refuses the plan") {
        std::array<double, 4> nonfinite = residual;
        nonfinite[2] = std::numeric_limits<double>::infinity();
        const auto built = make_jacobian_pre_j1_selection_plan(
            complete_row_prediction(), complete_column_grading(),
            nonfinite.data(), 4);
        CHECK_FALSE(built);
        CHECK(built.fallback_reason ==
              "entry-residual guard cannot be evaluated: full vector value is non-finite");
    }
}

TEST_CASE("selection-plan residual guard is relative to its first reduced step",
          "[parity_mask][selection_plan][reduced_solve]")
{
    const std::array<double, 4> residual{{0.25, 1e-14, 2e-14, 0.5}};
    const JacobianSelectionNorms norms = measure_jacobian_selection_norms(
        residual, {0, 3});
    REQUIRE(norms);
    CHECK(norms.active_linf == 0.5);
    CHECK(norms.forbidden_linf == 2e-14);
    double forbidden_baseline = 0.0;
    bool forbidden_baseline_installed = false;
    const JacobianForbiddenResidualCheck first =
        check_jacobian_forbidden_residual(
            norms, forbidden_baseline, forbidden_baseline_installed);
    CHECK(first.allowed);
    CHECK(forbidden_baseline_installed);
    CHECK(forbidden_baseline == norms.forbidden_linf);
    CHECK(first.limit == 10.0 * forbidden_baseline);

    const JacobianForbiddenResidualCheck constant =
        check_jacobian_forbidden_residual(
            norms, forbidden_baseline, forbidden_baseline_installed);
    CHECK(constant.allowed);
    CHECK(constant.limit == first.limit);

    JacobianSelectionNorms boundary = norms;
    boundary.forbidden_linf = first.limit;
    CHECK(check_jacobian_forbidden_residual(
              boundary, forbidden_baseline,
              forbidden_baseline_installed)
              .allowed);
    boundary.forbidden_linf = std::nextafter(
        boundary.forbidden_linf, std::numeric_limits<double>::infinity());
    CHECK_FALSE(check_jacobian_forbidden_residual(
                    boundary, forbidden_baseline,
                    forbidden_baseline_installed)
                    .allowed);

    std::array<double, 4> nonfinite = residual;
    nonfinite[1] = std::numeric_limits<double>::infinity();
    const JacobianSelectionNorms refused = measure_jacobian_selection_norms(
        nonfinite, {0, 3});
    CHECK_FALSE(refused);
    CHECK(std::isinf(refused.forbidden_linf));
    double nonfinite_baseline = 0.0;
    bool nonfinite_baseline_installed = false;
    const JacobianForbiddenResidualCheck nonfinite_check =
        check_jacobian_forbidden_residual(
            refused, nonfinite_baseline, nonfinite_baseline_installed);
    CHECK_FALSE(nonfinite_check.allowed);
    CHECK_FALSE(nonfinite_baseline_installed);
    JacobianParityMaskState nonfinite_state = square_state();
    nonfinite_state.row_sector = {1, 1, -1, -1};
    nonfinite_state.decision = JacobianParityMaskState::Decision::Engaged;
    decide_jacobian_parity_reduction(nonfinite_state, 0.0, 1.0, 1);
    REQUIRE(nonfinite_state.selection_plan);
    abandon_jacobian_parity_reduction(
        nonfinite_state, "forbidden residual norm is non-finite", 1);
    CHECK(nonfinite_state.reduction_decision ==
          JacobianParityMaskState::ReductionDecision::MaskedFullFallback);
    CHECK_FALSE(nonfinite_state.selection_plan);
    CHECK_FALSE(measure_jacobian_selection_norms(
        residual, std::vector<int>{}));
}

TEST_CASE("inactive-state guard snapshots an offset baseline and detects drift",
          "[parity_mask][selection_plan][reduced_solve]")
{
    UnsupportedTauSeedSpace space;
    System_of_eqs system(space, 0, 0);
    double active = 2.0;
    double inactive = 1.9e-13;
    system.add_var("ome", active);
    system.add_var("zvel", inactive);
    const JacobianSelectionPlan plan(
        +1, std::vector<int>{0}, std::vector<int>{0});

    std::vector<double> current;
    std::vector<double> baseline;
    bool baseline_installed = false;
    std::string failure_reason;
    REQUIRE(system.read_inactive_jacobian_state(
        plan, current, failure_reason));
    REQUIRE(current == std::vector<double>{inactive});
    const double installed_drift =
        install_or_measure_jacobian_inactive_state_drift(
            current, baseline, baseline_installed);
    CHECK(baseline_installed);
    CHECK(baseline == std::vector<double>{inactive});
    CHECK(installed_drift == 0.0);
    CHECK(jacobian_inactive_state_drift_allowed(installed_drift));

    REQUIRE(system.read_inactive_jacobian_state(
        plan, current, failure_reason));
    const double constant_drift =
        install_or_measure_jacobian_inactive_state_drift(
            current, baseline, baseline_installed);
    CHECK(constant_drift == 0.0);
    CHECK(jacobian_inactive_state_drift_allowed(constant_drift));
    CHECK(jacobian_inactive_state_drift_allowed(
        jacobian_inactive_state_drift_tolerance));
    CHECK_FALSE(jacobian_inactive_state_drift_allowed(std::nextafter(
        jacobian_inactive_state_drift_tolerance,
        std::numeric_limits<double>::infinity())));

    inactive += 2e-14;
    REQUIRE(system.read_inactive_jacobian_state(
        plan, current, failure_reason));
    const double drifted = install_or_measure_jacobian_inactive_state_drift(
        current, baseline, baseline_installed);
    REQUIRE(drifted > jacobian_inactive_state_drift_tolerance);
    CHECK_FALSE(jacobian_inactive_state_drift_allowed(drifted));

    inactive = std::numeric_limits<double>::infinity();
    CHECK_FALSE(system.read_inactive_jacobian_state(
        plan, current, failure_reason));
    CHECK(failure_reason == "inactive state coefficient is non-finite");
    CHECK_FALSE(jacobian_inactive_state_drift_allowed(
        std::numeric_limits<double>::infinity()));
}

TEST_CASE("retained sparse factors require the exact immutable selection role",
          "[parity_mask][selection_plan][reduced_solve]")
{
    auto plan = std::make_shared<const JacobianSelectionPlan>(
        5, std::vector<int>{0, 3}, std::vector<int>{1, 2});
    auto equivalent_but_distinct =
        std::make_shared<const JacobianSelectionPlan>(
            5, std::vector<int>{0, 3}, std::vector<int>{1, 2});
    CHECK(jacobian_selection_factor_compatible(plan, 2, plan, 2));
    CHECK_FALSE(jacobian_selection_factor_compatible(
        plan, 2, equivalent_but_distinct, 2));
    CHECK_FALSE(jacobian_selection_factor_compatible(plan, 2, plan, 3));

    std::shared_ptr<const JacobianSelectionPlan> full_role;
    CHECK(jacobian_selection_factor_compatible(
        full_role, 4, full_role, 4));
    CHECK_FALSE(jacobian_selection_factor_compatible(
        full_role, 0, full_role, 0));
}

TEST_CASE("pre-J1 reduced chord factor keeps the selected role",
          "[parity_mask][selection_plan][reduced_solve]")
{
    auto selected = std::make_shared<const JacobianSelectionPlan>(
        5, std::vector<int>{0, 3}, std::vector<int>{1, 2});
    std::shared_ptr<const JacobianSelectionPlan> full_role;

    CHECK(jacobian_selection_factor_compatible(selected, 2, selected, 2));
    CHECK_FALSE(jacobian_selection_factor_compatible(
        full_role, 4, selected, 2));
    CHECK_FALSE(jacobian_selection_factor_compatible(
        selected, 2, full_role, 4));
}

TEST_CASE("runtime reduction abandonment is sticky and preserves the mask",
          "[parity_mask][selection_plan][reduced_solve]")
{
    JacobianParityMaskState state = square_state();
    state.row_sector = {1, 1, -1, -1};
    state.decision = JacobianParityMaskState::Decision::Engaged;
    decide_jacobian_parity_reduction(state, 0.0, 1.0, 1);
    REQUIRE(state.selection_plan);
    state.inactive_state_baseline = {1.0};
    state.inactive_state_baseline_installed = true;
    JacobianSelectionNorms residual_norms;
    residual_norms.forbidden_linf = 2e-14;
    REQUIRE(check_jacobian_forbidden_residual(
                residual_norms, state.forbidden_baseline,
                state.forbidden_baseline_installed)
                .allowed);

    CoutCapture capture;
    abandon_jacobian_parity_reduction(state, "guard tripped", 0);
    abandon_jacobian_parity_reduction(state, "second reason", 0);
    CHECK(state.decision == JacobianParityMaskState::Decision::Engaged);
    CHECK(state.reduction_decision ==
          JacobianParityMaskState::ReductionDecision::MaskedFullFallback);
    CHECK_FALSE(state.selection_plan);
    CHECK(state.inactive_state_baseline.empty());
    CHECK_FALSE(state.inactive_state_baseline_installed);
    CHECK(state.forbidden_baseline == 0.0);
    CHECK_FALSE(state.forbidden_baseline_installed);
    CHECK(state.reduction_fallback_reason == "guard tripped");
    CHECK(capture.str() ==
          "Jacobian sector reduction: masked-full fallback, guard tripped\n");
}

TEST_CASE("parity reduction eligibility is exact-only and sticky per solve",
          "[parity_mask][selection_plan]")
{
    SECTION("strictly below the exact boundary is eligible") {
        JacobianParityMaskState state = square_state();
        state.row_sector = {1, 1, -1, -1};
        state.decision = JacobianParityMaskState::Decision::Engaged;
        CoutCapture capture;
        decide_jacobian_parity_reduction(
            state,
            std::nextafter(jacobian_parity_cross_tolerance, 0.0), 1.0, 0);
        REQUIRE(state.reduction_decision ==
                JacobianParityMaskState::ReductionDecision::Eligible);
        REQUIRE(state.selection_plan);
        CHECK(state.selection_plan->selected_rows() ==
              std::vector<int>{0, 1});
        CHECK(state.selection_plan->selected_columns() ==
              std::vector<int>{0, 1});
        CHECK(capture.str().empty());

        const auto original_plan = state.selection_plan;
        decide_jacobian_parity_reduction(state, 3.7e-7, 1.0, 0);
        CHECK(state.reduction_decision ==
              JacobianParityMaskState::ReductionDecision::Eligible);
        CHECK(state.selection_plan == original_plan);
        CHECK(capture.str().empty());
    }

    SECTION("the exact boundary is approximate and therefore ineligible") {
        JacobianParityMaskState state = square_state();
        state.row_sector = {1, 1, -1, -1};
        state.decision = JacobianParityMaskState::Decision::Engaged;
        decide_jacobian_parity_reduction(
            state, jacobian_parity_cross_tolerance, 1.0, 1);
        CHECK(state.reduction_decision ==
              JacobianParityMaskState::ReductionDecision::MaskedFullFallback);
        CHECK_FALSE(state.selection_plan);
        CHECK(state.reduction_fallback_reason ==
              "first-J parity is approximate (max cross / max |J| >= 1e-12)");
    }

    SECTION("a genuinely approximate engagement remains masked-full") {
        JacobianParityMaskState state = square_state();
        state.row_sector = {1, 1, -1, -1};
        state.decision = JacobianParityMaskState::Decision::Engaged;
        CoutCapture capture;
        decide_jacobian_parity_reduction(state, 3.7e-7, 1.0, 0);
        CHECK(state.reduction_decision ==
              JacobianParityMaskState::ReductionDecision::MaskedFullFallback);
        CHECK(state.reduction_fallback_reason ==
              "first-J parity is approximate (max cross / max |J| >= 1e-12)");
        CHECK(capture.str() ==
              "Jacobian sector reduction: masked-full fallback, first-J parity is approximate (max cross / max |J| >= 1e-12)\n");
    }

    SECTION("an unusable first-J measurement fails closed and stays failed") {
        for (const auto measurement :
             std::array<std::pair<double, double>, 5>{
                 std::pair{0.0, 0.0}, std::pair{-1.0, 1.0},
                 std::pair{0.0, -1.0},
                 std::pair{std::numeric_limits<double>::infinity(), 1.0},
                 std::pair{0.0, std::numeric_limits<double>::quiet_NaN()}}) {
            JacobianParityMaskState state = square_state();
            state.row_sector = {1, 1, -1, -1};
            state.decision = JacobianParityMaskState::Decision::Engaged;
            decide_jacobian_parity_reduction(
                state, measurement.first, measurement.second, 1);
            REQUIRE(state.reduction_decision ==
                    JacobianParityMaskState::ReductionDecision::MaskedFullFallback);
            CHECK_FALSE(state.selection_plan);
            CHECK(state.reduction_fallback_reason ==
                  "first-J parity measurement is invalid");

            decide_jacobian_parity_reduction(state, 0.0, 1.0, 1);
            CHECK(state.reduction_decision ==
                  JacobianParityMaskState::ReductionDecision::MaskedFullFallback);
            CHECK_FALSE(state.selection_plan);
            CHECK(state.reduction_fallback_reason ==
                  "first-J parity measurement is invalid");
        }
    }
}

TEST_CASE("parity reduction excludes odd globals with their paired odd rows",
          "[parity_mask][selection_plan]")
{
    JacobianParityMaskState state;
    state.n = 4;
    state.column_sector = {
        static_cast<signed char>(jacobian_parity_global_grading("ome")),
        static_cast<signed char>(jacobian_parity_global_grading("zvel")),
        +1, -1};
    // The first-J row grading puts the zvel-paired equation in the same odd
    // block. Both are excluded; neither disqualifies reduction.
    state.row_sector = {+1, -1, +1, -1};
    state.decision = JacobianParityMaskState::Decision::Engaged;

    decide_jacobian_parity_reduction(state, 0.0, 1.0, 1);
    REQUIRE(state.reduction_decision ==
            JacobianParityMaskState::ReductionDecision::Eligible);
    REQUIRE(state.selection_plan);
    CHECK(state.selection_plan->selected_rows() == std::vector<int>{0, 2});
    CHECK(state.selection_plan->selected_columns() == std::vector<int>{0, 2});
}

TEST_CASE("parity reduction malformed or rectangular labels fall back per solve",
          "[parity_mask][selection_plan]")
{
    SECTION("a rectangular active block falls back with the square assertion") {
        JacobianParityMaskState state = square_state();
        state.row_sector = {1, 1, 1, -1};
        state.decision = JacobianParityMaskState::Decision::Engaged;
        decide_jacobian_parity_reduction(state, 0.0, 1.0, 1);
        CHECK(state.reduction_decision ==
              JacobianParityMaskState::ReductionDecision::MaskedFullFallback);
        CHECK(state.reduction_fallback_reason ==
              "selected block is not square (rows 3 vs columns 2)");
    }

    SECTION("a parity label outside plus-or-minus one is malformed") {
        JacobianParityMaskState state = square_state();
        state.row_sector = {1, 0, -1, -1};
        state.decision = JacobianParityMaskState::Decision::Engaged;
        decide_jacobian_parity_reduction(state, 0.0, 1.0, 1);
        CHECK(state.reduction_decision ==
              JacobianParityMaskState::ReductionDecision::MaskedFullFallback);
        CHECK(state.reduction_fallback_reason ==
              "row block label at index 1 is outside the declared two-block partition");
    }
}

TEST_CASE("solver runtime reset clears the per-solve parity selection plan",
          "[parity_mask][selection_plan]")
{
    UnsupportedTauSeedSpace space;
    System_of_eqs system(space, 0, 0);
    auto state = std::make_shared<JacobianParityMaskState>();
    state->n = 2;
    state->column_sector = {+1, -1};
    state->row_sector = {+1, -1};
    state->decision = JacobianParityMaskState::Decision::Engaged;
    system.jacobian_parity_mask_state() = state;
    decide_jacobian_parity_reduction(*state, 0.0, 1.0, 1);
    REQUIRE(state->selection_plan);
    REQUIRE(system.jacobian_parity_mask_state());
    JacobianSelectionNorms first_solve_norms;
    first_solve_norms.forbidden_linf = 2e-14;
    REQUIRE(check_jacobian_forbidden_residual(
                first_solve_norms, state->forbidden_baseline,
                state->forbidden_baseline_installed)
                .allowed);
    REQUIRE(state->forbidden_baseline_installed);

    system.reset_solver_runtime_state();
    CHECK_FALSE(system.jacobian_parity_mask_state());
    // Existing external owners cannot mutate the immutable plan in a future
    // solve, and the System no longer retains the old decision.
    CHECK(state->selection_plan);

    auto next_state = std::make_shared<JacobianParityMaskState>();
    system.jacobian_parity_mask_state() = next_state;
    CHECK_FALSE(next_state->forbidden_baseline_installed);
    JacobianSelectionNorms second_solve_norms;
    second_solve_norms.forbidden_linf = 7e-14;
    const JacobianForbiddenResidualCheck second_solve_first =
        check_jacobian_forbidden_residual(
            second_solve_norms, next_state->forbidden_baseline,
            next_state->forbidden_baseline_installed);
    CHECK(second_solve_first.allowed);
    CHECK(next_state->forbidden_baseline == 7e-14);
}

TEST_CASE("fused parity emission and post-hoc retention select identical entries",
          "[parity_mask][fused]")
{
    struct Entry {
        int lane;
        int row;
        int column;
    };
    const std::vector<signed char> row_sector = {+1, -1, +1, -1};
    const std::vector<signed char> column_sector = {+1, +1, -1};
    const std::vector<Entry> entries = {
        {0, 0, 0}, {0, 1, 0}, {0, 3, 0},
        {1, 0, 1}, {1, 2, 1}, {1, 3, 1},
        {2, 0, 2}, {2, 1, 2}, {2, 2, 2},
    };

    std::vector<std::size_t> post_hoc_retained;
    for (std::size_t index = 0; index < entries.size(); ++index) {
        const Entry& entry = entries[index];
        if (jacobian_parity_entry_retained(
                column_sector[static_cast<std::size_t>(entry.column)],
                row_sector[static_cast<std::size_t>(entry.row)])) {
            post_hoc_retained.push_back(index);
        }
    }

    std::vector<std::size_t> fused_emitted;
    std::array<std::size_t, 3> lane_entry_counts{};
    for (std::size_t index = 0; index < entries.size(); ++index) {
        const Entry& entry = entries[index];
        if (!jacobian_parity_entry_retained(
                column_sector[static_cast<std::size_t>(entry.column)],
                row_sector[static_cast<std::size_t>(entry.row)])) {
            continue;
        }
        fused_emitted.push_back(index);
        ++lane_entry_counts[static_cast<std::size_t>(entry.lane)];
    }

    CHECK(fused_emitted == post_hoc_retained);
    CHECK(fused_emitted.size() == 4);
    CHECK(jacobian_packed_retained_entry_count(
              lane_entry_counts) ==
          fused_emitted.size());
}

TEST_CASE("all-cross packed lanes can decline without a false partial-emission guard",
          "[parity_mask][fused]")
{
    const std::array<signed char, 2> column_sector = {+1, +1};
    const std::array<signed char, 3> row_sector = {-1, -1, -1};
    std::array<std::size_t, 2> lane_entry_counts{};
    std::size_t cross_candidates = 0;
    for (std::size_t lane = 0; lane < column_sector.size(); ++lane) {
        for (signed char row : row_sector) {
            ++cross_candidates;
            if (jacobian_parity_entry_retained(column_sector[lane], row))
                ++lane_entry_counts[lane];
        }
    }

    REQUIRE(cross_candidates == 6);
    CHECK(jacobian_packed_retained_entry_count(
              lane_entry_counts) == 0);
    lane_entry_counts[1] = 1;
    CHECK(jacobian_packed_retained_entry_count(
              lane_entry_counts) == 1);
}

TEST_CASE("fused parity readiness requires a prior unmasked structural full J",
          "[parity_mask][fused]")
{
    JacobianParityMaskState state;
    state.n = 4;
    state.decision = JacobianParityMaskState::Decision::Engaged;
    state.column_sector = {+1, +1, -1, -1};
    state.row_sector = {+1, -1, +1, -1};
    state.structural_labels_available = true;

    CHECK_FALSE(jacobian_fused_parity_mask_ready(state, 4));
    state.unmasked_full_j_emitted = true;
    CHECK(jacobian_fused_parity_mask_ready(state, 4));
    state.row_sector[2] = 0;
    CHECK_FALSE(jacobian_fused_parity_mask_ready(state, 4));
}
