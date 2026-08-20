#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "Apps/Seed/GR/trumpet_bh_seed.hpp"
#include "For_Kadath/IO/be_file_source.hpp"
#include "For_Kadath/Utilities/Exporters/bco_geometry.hpp"

#include <cmath>
#include <filesystem>

using Catch::Matchers::WithinAbs;

TEST_CASE("Trumpet BH seed populates the full adapted space", "[seed][trumpet]")
{
    const auto output_dir = std::filesystem::temp_directory_path() / "kadath_trumpet_seed_unit";
    std::filesystem::create_directories(output_dir);

    kadath_config<BCO_BH_INFO> config;
    config.set_defaults();
    config.set_outputdir(output_dir.string());
    config.set_filename("trumpet_seed");
    config.set(BCO_RES) = 5;
    config.set(NSHELLS) = 0;
    config.set(MCH) = 1.;
    config.set(MIRR) = 1.;
    config.set(CHI) = 0.;
    config.set(TRUMPET_BH_SEED) = true;

    Kadath::Seed::setup_co<BH, Kadath::Space_trumpet>(config);

    REQUIRE(std::filesystem::exists(config.space_filename()));

    BeFileSource source(config.space_filename());
    Kadath::Space_trumpet space(source);
    Scalar conf(space, source);
    Scalar lapse(space, source);
    Vector shift(space, source);

    Index origin(space.get_domain(0)->get_nbr_points());
    REQUIRE(conf(0)(origin) > 1.);
    REQUIRE(lapse(0)(origin) >= 0.);

    Kadath::Seed::NonRotatingTrumpetProfile profile(1.);
    REQUIRE_THAT(bco_utils::get_radius(space.get_domain(1), OUTER_BC),
                 WithinAbs(profile.horizon_isotropic_radius(), 1e-12));
}
