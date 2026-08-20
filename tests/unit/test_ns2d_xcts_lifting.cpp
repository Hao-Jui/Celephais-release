#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "Apps/Helper/converged_filename.hpp"
#include "Apps/Formalism/GR/NS_2D_XCTS/lifting.hpp"
#include "Apps/Formalism/Shared/NS_2D_XCTS/core.hpp"
#include "Apps/Formalism/Shared/ns_2d_seed.hpp"
#include "For_Kadath/Config/config_bco.hpp"
#include "For_Kadath/IO/be_file_source.hpp"
#include "For_Kadath/Kadath_point_h/kadath_adapted.hpp"
#include "For_Kadath/Kadath_point_h/kadath_adapted_polar.hpp"
#include "For_Kadath/Utilities/Exporters/bco_geometry.hpp"
#include "For_Kadath/Utilities/Exporters/coord_fields.hpp"
#include "Hydro/EOS.hh"

#include <chrono>
#include <array>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <limits>
#include <string>
#include <type_traits>

using namespace Kadath;
using namespace Kadath::Margherita;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

namespace {

using ns_config_t = kadath_config<BCO_NS_INFO>;

std::filesystem::path repo_root()
{
    const char* home = std::getenv("HOME_CELEPHAIS");
    REQUIRE(home != nullptr);
    return home;
}

std::filesystem::path ns2d_xcts_fixture()
{
    return repo_root() / "tests/fixtures/ns2d_xcts_lift/converged_NS_2D.dd2.1.4.0.3.0.13.uniform.toml";
}

struct TemporaryLiftOutput
{
    std::filesystem::path base;

    ~TemporaryLiftOutput()
    {
        std::filesystem::remove(base.string() + ".toml");
        std::filesystem::remove(base.string() + ".dat");
    }
};

TemporaryLiftOutput make_temporary_lift_output()
{
    const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
    return {std::filesystem::temp_directory_path() /
            ("kadath_ns2d_xcts_lift_mb_" + std::to_string(stamp))};
}


double max_axisymmetric_field_error(const Scalar& target, const Scalar& source)
{
    double max_error = 0.;
    const int ndom = target.get_nbr_domains();
    for (int dom = 0; dom < ndom; ++dom) {
        const Domain& domain = *target.get_space().get_domain(dom);
        const Dim_array& points = domain.get_nbr_points();
        Index pos(points);
        do {
            if (dom == ndom - 1 && pos(0) == points(0) - 1)
                continue;
            Point source_point(2);
            const double x = domain.get_cart(1)(pos);
            const double y = domain.get_cart(2)(pos);
            source_point.set(1) = std::sqrt(x * x + y * y);
            source_point.set(2) = domain.get_cart(3)(pos);
            max_error = std::max(max_error,
                                 std::abs(target(dom)(pos) - source.val_point(source_point)));
        } while (pos.inc());
    }
    return max_error;
}


template <class eos_t>
double integrated_mb_2d(ns_config_t bconfig)
{
    BeFileSource source(bconfig.space_filename());
    Space_polar_adapted space(source);
    Scalar conf(space, source);
    Scalar lapse(space, source);
    Scalar shift(space, source);
    Scalar logh(space, source);
    Scalar Omg(space, source);

    const int ndom = space.get_nbr_domains();
    const int adapted_outer = space.ADAPTED_OUTER;

    System_of_eqs syst(space, 0, ndom - 1);
    syst.add_cst("4piG", 4.0 * M_PI);
    syst.add_cst("ome", bconfig(OMEGA));
    syst.add_cst("cstA", bconfig(CSTA));
    syst.add_cst("P", conf);
    syst.add_cst("N", lapse);
    syst.add_cst("brsint", shift);
    syst.add_cst("Omg", Omg);
    syst.add_cst("H", logh);
    syst.add_def("bet = divrsint(brsint)");

    Param eos_params;
    syst.add_ope("rho", &EOS<eos_t, eos_var_t::DENSITY>::action, &eos_params);

    for (int d = 0; d <= adapted_outer; ++d) {
        syst.add_def(d, "h = exp(H)");
        syst.add_def(d, "U = multrsint(P^2 / N * (Omg + bet))");
        syst.add_def(d, "Usq = U*U");
        syst.add_def(d, "Wsquare = 1 / (1 - Usq)");
        syst.add_def(d, "W = sqrt(Wsquare)");
        syst.add_def(d, "intMb = W * rho(h) * P^6 * 4piG / 2");
    }

    double mb = 0.0;
    for (int d = 0; d <= adapted_outer; ++d)
        mb += syst.give_val_def("intMb")()(d).integ_volume();

    return mb;
}

template <class eos_t, class space_t = Space_spheric_adapted>
double integrated_mb_3d(ns_config_t bconfig, const double tilt_angle = 0.)
{
    BeFileSource source(bconfig.space_filename());
    space_t space(source);
    Scalar conf(space, source);
    Scalar lapse(space, source);
    Vector shift(space, source);
    Scalar logh(space, source);

    Base_tensor basis(space, CARTESIAN_BASIS);
    Metric_flat fmet(space, basis);
    CoordFields<space_t> coord_fields(space);
    vec_ary_t coord_vectors = default_co_vector_ary(space);
    const double xo = bco_utils::get_center(space, 0);
    update_fields_co(coord_fields, coord_vectors, {}, xo);

    Scalar Omg(space);
    Omg = bconfig(OMEGA);
    Omg.std_base();

    System_of_eqs syst(space, 0, space.get_nbr_domains() - 1);
    fmet.set_system(syst, "f");

    syst.add_cst("P", conf);
    syst.add_cst("N", lapse);
    syst.add_cst("bet", shift);
    syst.add_cst("Omg", Omg);
    syst.add_cst("H", logh);
    syst.add_cst("mmx", *coord_vectors[to_int(coord_vector::BCO1_ROTx)]);
    syst.add_cst("mmz", *coord_vectors[to_int(coord_vector::BCO1_ROTz)]);
    syst.add_cst("cosTilt", std::cos(tilt_angle));
    syst.add_cst("sinTilt", std::sin(tilt_angle));

    syst.add_def("mm^i = cosTilt * mmz^i + sinTilt * mmx^i");
    syst.add_def("omega^i = bet^i + Omg * mm^i");

    Param eos_params;
    syst.add_ope("rho", &EOS<eos_t, eos_var_t::DENSITY>::action, &eos_params);

    for (int d = 0; d <= 1; ++d) {
        syst.add_def(d, "h = exp(H)");
        syst.add_def(d, "U^i = omega^i / N");
        syst.add_def(d, "Usquare = P^4 * U_i * U^i");
        syst.add_def(d, "Wsquare = 1. / (1. - Usquare)");
        syst.add_def(d, "W = sqrt(Wsquare)");
        syst.add_def(d, "intMb = P^6 * rho(h) * W");
    }

    return syst.give_val_def("intMb")()(0).integ_volume() +
           syst.give_val_def("intMb")()(1).integ_volume();
}

std::pair<double, double> tilted_shift_axis_projection(ns_config_t bconfig, const double tilt_angle)
{
    BeFileSource source(bconfig.space_filename());
    Space_spheric_adapted_nosym space(source);
    Scalar conf(space, source);
    Scalar lapse(space, source);
    Vector shift(space, source);
    Scalar logh(space, source);
    (void)conf;
    (void)lapse;
    (void)logh;

    const double sine = std::sin(tilt_angle);
    const double cosine = std::cos(tilt_angle);
    double max_projection = 0.;
    double max_shift = 0.;
    for (int dom = 0; dom < space.get_nbr_domains(); ++dom) {
        const Domain& domain = *space.get_domain(dom);
        const Dim_array& points = domain.get_nbr_points();
        Index pos(points);
        do {
            if (dom == space.get_nbr_domains() - 1 && pos(0) == points(0) - 1)
                continue;
            const double beta_x = shift(1)(dom)(pos);
            const double beta_y = shift(2)(dom)(pos);
            const double beta_z = shift(3)(dom)(pos);
            max_projection = std::max(max_projection, std::abs(sine * beta_x + cosine * beta_z));
            max_shift = std::max(max_shift, std::sqrt(beta_x * beta_x + beta_y * beta_y + beta_z * beta_z));
        } while (pos.inc());
    }
    return {max_projection, max_shift};
}

template <class eos_t>
void initialize_fixture_eos(ns_config_t& bconfig);

template <class eos_t>
void require_tilted_lift_consistency(const std::filesystem::path& input_config,
                                     const double tilt_degrees)
{
    const double tilt_angle = tilt_degrees * M_PI / 180.;
    ns_config_t source_config(input_config.string());
    initialize_fixture_eos<eos_t>(source_config);
    const double source_integrated_mb = integrated_mb_2d<eos_t>(source_config);

    auto output = make_temporary_lift_output();
    ns_config_t lift_config(input_config.string());
    lift_config.set(DEG) = tilt_degrees;
    const int lift_status = ns_2d_xcts_lift_to_3d_as<Space_spheric_adapted_nosym>(
        lift_config, static_cast<int>(source_config(BCO_RES)), output.base.string(), true, tilt_angle);
    REQUIRE(lift_status == EXIT_SUCCESS);

    ns_config_t lifted_config(output.base.string() + ".toml");
    REQUIRE_THAT(lifted_config(DEG), WithinAbs(tilt_degrees, 1.e-14));
    const double lifted_integrated_mb =
        integrated_mb_3d<eos_t, Space_spheric_adapted_nosym>(lifted_config, tilt_angle);
    const auto [max_axis_projection, max_shift] = tilted_shift_axis_projection(lifted_config, tilt_angle);

    CAPTURE(source_integrated_mb);
    CAPTURE(lifted_integrated_mb);
    CAPTURE(max_axis_projection);
    CAPTURE(max_shift);
    REQUIRE(std::isfinite(lifted_integrated_mb));
    // A tilted axis is represented on the finite full-phi grid; the fixture's
    // res=13 lift has an O(1e-7) interpolation residue in the volume integral.
    REQUIRE_THAT(lifted_integrated_mb, WithinRel(source_integrated_mb, 2.e-7));
    REQUIRE(max_shift > 0.);
    REQUIRE(max_axis_projection <= 1.e-11 * max_shift + 1.e-14);
}

double integrated_j_2d(ns_config_t bconfig)
{
    BeFileSource source(bconfig.space_filename());
    Space_polar_adapted space(source);
    Scalar conf(space, source);
    Scalar lapse(space, source);
    Scalar shift(space, source);
    Scalar logh(space, source);
    Scalar Omg(space, source);
    (void)conf;
    (void)lapse;
    (void)logh;
    (void)Omg;

    const int ndom = space.get_nbr_domains();
    System_of_eqs syst(space, 0, ndom - 1);
    syst.add_cst("4piG", 4.0 * M_PI);
    syst.add_cst("brsint", shift);
    syst.add_def("bet = divrsint(brsint)");
    syst.add_def(ndom - 1, "intJ = multrsint(multrsint(dr(bet))) / 4 / 4piG");

    return space.get_domain(ndom - 1)->integ(syst.give_val_def("intJ")()(ndom - 1), OUTER_BC);
}

double integrated_j_3d_scalarized(ns_config_t bconfig)
{
    BeFileSource source(bconfig.space_filename());
    Space_spheric_adapted space(source);
    Scalar conf(space, source);
    Scalar lapse(space, source);
    Vector shift(space, source);
    Scalar logh(space, source);
    (void)conf;
    (void)lapse;
    (void)logh;

    const int ndom = space.get_nbr_domains();

    Scalar brsint3(space);
    brsint3.annule_hard();
    brsint3.set_in_conf();
    brsint3.allocate_conf();
    for (int dom = 0; dom < ndom; ++dom) {
        const Domain& domain = *space.get_domain(dom);
        const Dim_array& points = domain.get_nbr_points();
        Index pos(points);
        do {
            const double x = (dom == ndom - 1) ? domain.get_cart_surr(1)(pos) : domain.get_cart(1)(pos);
            const double y = (dom == ndom - 1) ? domain.get_cart_surr(2)(pos) : domain.get_cart(2)(pos);
            const double rho = std::sqrt(x * x + y * y);
            double brsint = 0.;
            if (rho > 0.)
                brsint = (-shift(1)(dom)(pos) * y + shift(2)(dom)(pos) * x) / rho;
            brsint3.set_domain(dom).set(pos) = brsint;
        } while (pos.inc());
    }
    brsint3.std_base_p_spher();

    System_of_eqs syst(space, 0, ndom - 1);
    syst.add_cst("4piG", 4.0 * M_PI);
    syst.add_cst("brsint3", brsint3);
    syst.add_def("betphi3 = divrsint(brsint3)");
    syst.add_def(ndom - 1, "intJ = multrsint(multrsint(dr(betphi3))) / 4 / 4piG");

    return space.get_domain(ndom - 1)->integ(syst.give_val_def("intJ")()(ndom - 1), OUTER_BC);
}

double integrated_j_3d_surface(ns_config_t bconfig)
{
    BeFileSource source(bconfig.space_filename());
    Space_spheric_adapted space(source);
    Scalar conf(space, source);
    Scalar lapse(space, source);
    Vector shift(space, source);
    Scalar logh(space, source);
    (void)logh;

    const int ndom = space.get_nbr_domains();
    Base_tensor basis(space, CARTESIAN_BASIS);
    Metric_flat fmet(space, basis);
    CoordFields<Space_spheric_adapted> coord_fields(space);
    vec_ary_t coord_vectors = default_co_vector_ary(space);
    update_fields_co(coord_fields, coord_vectors, {}, bco_utils::get_center(space, 0));

    System_of_eqs syst(space, 0, ndom - 1);
    fmet.set_system(syst, "f");
    syst.add_cst("PI", M_PI);
    syst.add_cst("P", conf);
    syst.add_cst("N", lapse);
    syst.add_cst("bet", shift);
    syst.add_cst("mg", *coord_vectors[to_int(coord_vector::GLOBAL_ROT)]);
    syst.add_cst("einf", *coord_vectors[to_int(coord_vector::S_INF)]);
    syst.add_def("NP = P*N");
    syst.add_def("Ntilde = N / P^6");
    syst.add_def("A^ij = (D^i bet^j + D^j bet^i - 2. / 3.* D_k bet^k * f^ij) / 2. / Ntilde");
    syst.add_def(ndom - 1, "intJ = multr(A_ij * mg^j * einf^i) / 8. / PI");

    return space.get_domain(ndom - 1)->integ(syst.give_val_def("intJ")()(ndom - 1), OUTER_BC);
}

template <class eos_t>
void initialize_fixture_eos(ns_config_t& bconfig)
{
    const double h_cut = bconfig.eos<double>(HCUT);
    const std::string eos_file = bconfig.eos<std::string>(EOSFILE);
    const std::string eos_type = bconfig.eos<std::string>(EOSTYPE);

    if constexpr (std::is_same_v<eos_t, Cold_Table>) {
        REQUIRE(eos_type == "Cold_Table");
        const int interp_pts = (bconfig.eos<int>(INTERP_PTS) == 0) ? 2000 : bconfig.eos<int>(INTERP_PTS);
        EOS<eos_t, eos_var_t::PRESSURE>::init(eos_file, h_cut, interp_pts);
    } else {
        REQUIRE(eos_type == "Cold_PWPoly");
        EOS<eos_t, eos_var_t::PRESSURE>::init(eos_file, h_cut);
    }
}

template <class eos_t>
void require_mb_consistency_after_lift(const std::filesystem::path& input_config)
{
    ns_config_t source_config(input_config.string());
    initialize_fixture_eos<eos_t>(source_config);

    const double configured_mb = source_config(MB);
    const double source_integrated_mb = integrated_mb_2d<eos_t>(source_config);
    REQUIRE(std::isfinite(source_integrated_mb));
    REQUIRE_THAT(source_integrated_mb, WithinRel(configured_mb, 1.0e-8));

    auto output = make_temporary_lift_output();
    ns_config_t lift_config(input_config.string());
    const int lift_status = ns_2d_xcts_lift_to_3d(
        lift_config, static_cast<int>(source_config(BCO_RES)), output.base.string());
    REQUIRE(lift_status == EXIT_SUCCESS);

    ns_config_t lifted_config(output.base.string() + ".toml");
    REQUIRE_THAT(lifted_config(MB), WithinAbs(configured_mb, 1.0e-14));

    const double lifted_integrated_mb = integrated_mb_3d<eos_t>(lifted_config);
    const double source_integrated_j = integrated_j_2d(source_config);
    const double lifted_scalarized_j = integrated_j_3d_scalarized(lifted_config);
    const double lifted_surface_j = integrated_j_3d_surface(lifted_config);
    CAPTURE(configured_mb);
    CAPTURE(source_integrated_mb);
    CAPTURE(lifted_integrated_mb);
    CAPTURE(source_integrated_j);
    CAPTURE(lifted_scalarized_j);
    CAPTURE(lifted_surface_j);
    REQUIRE(std::isfinite(lifted_integrated_mb));
    REQUIRE_THAT(lifted_integrated_mb, WithinRel(source_integrated_mb, 1.0e-10));
    REQUIRE_THAT(lifted_scalarized_j, WithinAbs(source_integrated_j, 1.0e-8));
}

} // namespace

TEST_CASE("NS2d batched scalar import matches the scalar import contract", "[ns2d_xcts_lifting]")
{
    ns_config_t config(ns2d_xcts_fixture().string());
    BeFileSource source(config.space_filename());
    Space_polar_adapted space(source);
    Scalar old_conf(space, source);
    Scalar old_lapse(space, source);
    Scalar old_shift(space, source);
    Scalar old_logh(space, source);
    Scalar old_Omg(space, source);

    Scalar legacy_conf(space); legacy_conf = 1.;
    Scalar legacy_lapse(space); legacy_lapse = 1.;
    Scalar legacy_shift(space); legacy_shift = 0.;
    Scalar legacy_logh(space); legacy_logh = 0.;
    Scalar legacy_Omg(space); legacy_Omg = 0.;
    legacy_conf.import(old_conf);
    legacy_lapse.import(old_lapse);
    legacy_shift.import(old_shift);
    legacy_logh.import(old_logh);
    legacy_Omg.import(old_Omg);

    Scalar batch_conf(space); batch_conf = 1.;
    Scalar batch_lapse(space); batch_lapse = 1.;
    Scalar batch_shift(space); batch_shift = 0.;
    Scalar batch_logh(space); batch_logh = 0.;
    Scalar batch_Omg(space); batch_Omg = 0.;
    const std::array fields{
        ns_2d_xcts_import::import_field(batch_conf, old_conf),
        ns_2d_xcts_import::import_field(batch_lapse, old_lapse),
        ns_2d_xcts_import::import_field(batch_shift, old_shift),
        ns_2d_xcts_import::import_field(batch_logh, old_logh),
        ns_2d_xcts_import::import_field(batch_Omg, old_Omg),
    };
    ns_2d_xcts_import::import_scalar_batch(fields);

    const auto require_same_values = [&](const Scalar& legacy, const Scalar& batch) {
        for (int dom = 0; dom < space.get_nbr_domains(); ++dom) {
            Index pos(space.get_domain(dom)->get_nbr_points());
            do {
                REQUIRE(batch(dom)(pos) == legacy(dom)(pos));
            } while (pos.inc());
        }
    };
    require_same_values(legacy_conf, batch_conf);
    require_same_values(legacy_lapse, batch_lapse);
    require_same_values(legacy_shift, batch_shift);
    require_same_values(legacy_logh, batch_logh);
    require_same_values(legacy_Omg, batch_Omg);

    BeFileSource other_source(config.space_filename());
    Space_polar_adapted other_space(other_source);
    Scalar other_conf(other_space, other_source);
    const std::array mismatched_sources{
        ns_2d_xcts_import::import_field(batch_conf, old_conf),
        ns_2d_xcts_import::import_field(batch_lapse, other_conf),
    };
    REQUIRE_THROWS(ns_2d_xcts_import::import_scalar_batch(mismatched_sources));
}

TEST_CASE("NS2d XCTS lift preserves baryonic mass and axial spin", "[ns2d_xcts_lifting]")
{
    const auto input_config = ns2d_xcts_fixture();
    REQUIRE(std::filesystem::exists(input_config));
    REQUIRE(std::filesystem::exists(input_config.parent_path() / (input_config.stem().string() + ".dat")));

    ns_config_t fixture_config(input_config.string());
    BeFileSource fixture_source(fixture_config.space_filename());
    Space_polar_adapted fixture_space(fixture_source);
    REQUIRE(converged_ns2d_diffrot_filename(fixture_config, fixture_space, "UNIROT") ==
            input_config.stem().string());
    auto example_config = fixture_config;
    example_config.set(CHI) = 0.1;
    const std::string expected_example =
        "converged_NS_2D.dd2.1.4.0.1.0." + converged_resolution(fixture_space) + ".uniform";
    REQUIRE(converged_ns2d_diffrot_filename(example_config, fixture_space, "UNIROT") ==
            expected_example);

    const std::string eos_type = fixture_config.eos<std::string>(EOSTYPE);

    if (eos_type == "Cold_Table") {
        require_mb_consistency_after_lift<Cold_Table>(input_config);
    } else if (eos_type == "Cold_PWPoly") {
        require_mb_consistency_after_lift<Cold_PWPoly>(input_config);
    } else {
        FAIL("Unsupported EOS type in NS2d XCTS lift fixture: " << eos_type);
    }
}

TEST_CASE("GR NS2d uniform rotation always retains the Omg field", "[ns2d_xcts_lifting]")
{
    ns_config_t config(ns2d_xcts_fixture().string());
    BeFileSource source(config.space_filename());
    Space_polar_adapted space(source);
    Scalar conf(space, source);
    Scalar lapse(space, source);
    Scalar shift(space, source);
    Scalar logh(space, source);
    Scalar Omg(space, source);

    config.set(CHI) = 0.;
    System_of_eqs static_system(space, 0, space.get_nbr_domains() - 1);
    ns_2d_xcts_core::add_base_system<Cold_PWPoly>(
        static_system, config, conf, lapse, logh, shift, Omg);

    config.set(CHI) = 0.3;
    System_of_eqs rotating_system(space, 0, space.get_nbr_domains() - 1);
    ns_2d_xcts_core::add_base_system<Cold_PWPoly>(
        rotating_system, config, conf, lapse, logh, shift, Omg);

    System_of_eqs reference_without_omg(space, 0, space.get_nbr_domains() - 1);
    reference_without_omg.add_var("P", conf);
    reference_without_omg.add_var("N", lapse);
    reference_without_omg.add_var("H", logh);
    reference_without_omg.add_var("brsint", shift);
    reference_without_omg.add_var("ome", config.set(OMEGA));
    if (config.control(MB_FIXING))
        reference_without_omg.add_var("Madm", config.set(MADM));
    else
        reference_without_omg.add_var("Mb", config.set(MB));

    int omega_field_dof = 0;
    for (int d = 0; d < space.get_nbr_domains(); ++d)
        omega_field_dof += space.get_domain(d)->nbr_unknowns(Omg, d);

    REQUIRE(static_system.get_nbr_unknowns() - reference_without_omg.get_nbr_unknowns() ==
            omega_field_dof);
    REQUIRE(rotating_system.get_nbr_unknowns() == static_system.get_nbr_unknowns());
    REQUIRE_THROWS(ns_2d_xcts_core::validate_spin(std::numeric_limits<double>::quiet_NaN()));
}

TEST_CASE("NS2d XCTS lift rotates fields and shift into a no-symmetry space", "[ns2d_xcts_lifting]")
{
    const auto input_config = ns2d_xcts_fixture();
    REQUIRE(std::filesystem::exists(input_config));

    ns_config_t source_config(input_config.string());
    const std::string eos_type = source_config.eos<std::string>(EOSTYPE);
    for (const double tilt_degrees : {30., -30.}) {
        DYNAMIC_SECTION("tilt " << tilt_degrees << " degrees")
        {
            if (eos_type == "Cold_Table") {
                require_tilted_lift_consistency<Cold_Table>(input_config, tilt_degrees);
            } else if (eos_type == "Cold_PWPoly") {
                require_tilted_lift_consistency<Cold_PWPoly>(input_config, tilt_degrees);
            } else {
                FAIL("Unsupported EOS type in tilted NS2d XCTS lift fixture: " << eos_type);
            }
        }
    }
}


TEST_CASE("NS2d seed stage selection bypasses TOV after lift", "[ns2d_xcts_lifting]")
{
    ns_config_t config(ns2d_xcts_fixture().string());
    config.set_stage(NOROT) = true;
    config.set_stage(UNIROT) = false;
    config.set_stage(BIN_BOOST) = true;

    const auto lifted_stages = ns_2d_xcts_lifted_stages(config.return_stages());
    REQUIRE_FALSE(lifted_stages[to_int(NOROT)]);
    REQUIRE(lifted_stages[to_int(UNIROT)]);
    REQUIRE(lifted_stages[to_int(BIN_BOOST)]);
}

TEST_CASE("NS2d lift refuses a missing source", "[ns2d_xcts_lifting]")
{
    ns_config_t config(ns2d_xcts_fixture().string());
    auto missing = make_temporary_lift_output();
    config.set_filename(missing.base.string());
    REQUIRE_THROWS(ns_2d_xcts_lift_to_3d(config, 13, missing.base.string() + "_out"));
}

TEST_CASE("NS2d lift refuses invalid resolution and tilt", "[ns2d_xcts_lifting]")
{
    ns_config_t config(ns2d_xcts_fixture().string());
    auto output = make_temporary_lift_output();
    REQUIRE_THROWS(ns_2d_xcts_lift_to_3d(config, 0, output.base.string()));
    REQUIRE_THROWS(ns_2d_xcts_lift_to_3d(config, -1, output.base.string()));
    REQUIRE_THROWS(ns_2d_xcts_lift_to_3d(config, 12, output.base.string()));
    REQUIRE_THROWS(ns_2d_xcts_lift_to_3d_as<Space_spheric_adapted_nosym>(
        config, 13, output.base.string(), true, std::numeric_limits<double>::quiet_NaN()));
    REQUIRE_THROWS(ns_2d_xcts_lift_to_3d_as<Space_spheric_adapted_nosym>(
        config, 13, output.base.string(), true, std::numeric_limits<double>::infinity()));
}
