#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "Apps/AMR/bns_amr_policy.hpp"
#include "Apps/AMR/bhns_amr_policy.hpp"
#include "Apps/Helper/converged_filename.hpp"
#include "For_Kadath/Config/config_binary.hpp"
#include "For_Kadath/IO/be_file_sink.hpp"
#include "For_Kadath/IO/be_file_source.hpp"
#include "For_Kadath/Kadath_point_h/kadath_bin_ns.hpp"
#include "For_Kadath/Kadath_point_h/kadath_bin_ns_nosym.hpp"
#include "For_Kadath/Kadath_point_h/kadath_bhns.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <string>
#include <unistd.h>
#include <vector>

using namespace Kadath;

namespace {

// Minimal 12-domain binary-NS geometry (no per-star shells, no exterior
// shells): NS1=0, ADAPTED1=1/2, NS2=3, ADAPTED2=4/5, bispheric 6-10,
// compactified 11.
constexpr int test_domain_count = 12;
constexpr int test_base = 7;
constexpr double test_dist = 10.0;

const std::vector<double>& star_bounds()
{
    static const std::vector<double> bounds{0.5, 1.0, 2.0};
    return bounds;
}

const std::vector<double>& exterior_bounds()
{
    static const std::vector<double> bounds{10.0};
    return bounds;
}

Space_bin_ns make_space(const std::vector<int>& nr_per_domain)
{
    return Space_bin_ns(CHEB_TYPE, test_dist, star_bounds(), star_bounds(), exterior_bounds(), nr_per_domain);
}

Space_bhns make_bhns_space()
{
    return Space_bhns(CHEB_TYPE, test_dist, star_bounds(), star_bounds(), exterior_bounds(), test_base);
}

// 12-domain Dim_array layout at the shared base: spherical domains carry
// (nr=7, ntheta=7, nphi=6), the five bispheric domains (OUTER..OUTER+4 = 6..10)
// carry (7, 7, 7).  Returned ready to be patched per domain through the
// new full-Dim_array constructor.

Dim_array spherical_base_dims()
{
    Dim_array r(3);
    r.set(0) = test_base;
    r.set(1) = test_base;
    r.set(2) = test_base - 1;
    return r;
}

Dim_array bispheric_base_dims()
{
    Dim_array r(3);
    r.set(0) = test_base;
    r.set(1) = test_base;
    r.set(2) = test_base;
    return r;
}

// OUTER = 6, OUTER+4 = 10 in the 12-domain layout.
bool is_bispheric_domain(int d)
{
    return d >= 6 && d <= 10;
}

std::vector<Dim_array> base_dim_layout()
{
    std::vector<Dim_array> dims;
    dims.reserve(test_domain_count);
    for (int d = 0; d < test_domain_count; ++d)
        dims.push_back(is_bispheric_domain(d) ? bispheric_base_dims() : spherical_base_dims());
    return dims;
}

std::vector<Dim_array> nosym_base_dim_layout()
{
    std::vector<Dim_array> dims = base_dim_layout();
    for (int d = 6; d <= 10; ++d)
        dims[static_cast<std::size_t>(d)].set(2) = test_base + 1;
    return dims;
}

Space_bin_ns make_space(const std::vector<Dim_array>& res_per_domain)
{
    return Space_bin_ns(CHEB_TYPE, test_dist, star_bounds(), star_bounds(), exterior_bounds(), res_per_domain);
}

// Star 1 carries one ordinary shell (uniform split of the same band):
// NS1=0, ADAPTED1=1/2, shell=3, NS2=4, ADAPTED2=5/6, bispheric 7-11,
// compactified 12.
constexpr int shelled_domain_count = 13;

Space_bin_ns make_shelled_space(const std::vector<int>& nr_per_domain)
{
    static const std::vector<double> ns1_bounds{0.5, 1.0, 1.5, 2.0};
    return Space_bin_ns(CHEB_TYPE, test_dist, ns1_bounds, star_bounds(), exterior_bounds(), nr_per_domain);
}

std::string temp_space_path()
{
    namespace fs = std::filesystem;
    fs::path p = fs::temp_directory_path()
               / ("kadath_bin_ns_per_domain_test_" + std::to_string(::getpid()) + ".dat");
    return p.string();
}

std::vector<int> radial_points(const Space& space)
{
    std::vector<int> nr(static_cast<std::size_t>(space.get_nbr_domains()));
    for (int d = 0; d < space.get_nbr_domains(); ++d)
        nr[static_cast<std::size_t>(d)] = space.get_domain(d)->get_nbr_points()(0);
    return nr;
}

// Square-system + residual probe: one scalar Laplace problem with the full
// matching set of the binary space. u == 1 solves it exactly, so the residual
// (including every cross-resolution matching condition) must vanish to
// roundoff.
void require_square_laplace_system(Space_bin_ns& space)
{
    Scalar u(space);
    u = 1.;
    u.std_base();

    System_of_eqs syst(space);
    syst.add_var("u", u);
    space.add_eq(syst, "lap(u) = 0", "u", "dn(u)");
    syst.add_eq_bc(space.get_nbr_domains() - 1, OUTER_BC, "u = 1");
    // The adapted-surface shape coefficients are unknowns of any system on
    // this space; close them with a boundary condition on the variable at the
    // star surfaces (u == 1 stays exact).
    syst.add_eq_bc(space.ADAPTED1, OUTER_BC, "u = 1");
    syst.add_eq_bc(space.ADAPTED2, OUTER_BC, "u = 1");

    const Array<double> residual = syst.sec_member();
    REQUIRE(syst.get_nbr_unknowns() == syst.get_nbr_conditions());

    double max_abs = 0.0;
    for (int i = 0; i < syst.get_nbr_conditions(); ++i)
        max_abs = std::max(max_abs, std::abs(residual.get_data()[i]));
    REQUIRE(std::isfinite(max_abs));
    REQUIRE(max_abs < 1.e-10);
}

std::array<int, 2> bhns_laplace_counts(Space_bhns& space)
{
    Scalar u(space);
    u = 1.;
    u.std_base();

    System_of_eqs syst(space);
    syst.add_var("u", u);
    space.add_eq(syst, "lap(u) = 0", "u", "dn(u)");
    syst.add_eq_bc(space.get_nbr_domains() - 1, OUTER_BC, "u = 1");
    syst.add_eq_bc(space.ADAPTEDNS, OUTER_BC, "u = 1");
    syst.add_eq_bc(space.ADAPTEDBH, OUTER_BC, "u = 1");
    const Array<double> residual = syst.sec_member();
    REQUIRE(residual.get_size(0) == syst.get_nbr_conditions());
    return {syst.get_nbr_unknowns(), syst.get_nbr_conditions()};
}

using BnsAmrConfig = kadath_config<BIN_INFO>;

BnsAmrConfig make_amr_config()
{
    BnsAmrConfig config;
    config.initialize_binary({"ns", "ns"});
    config.set_defaults();
    config.set(BIN_RES) = 9;
    return config;
}

BnsAmrConfig make_bhns_amr_config()
{
    BnsAmrConfig config;
    config.initialize_binary({"ns", "bh"});
    config.set_defaults();
    config.set(BIN_RES) = test_base;
    return config;
}

// ---------------------------------------------------------------------------
// Per-direction TailDecision fixtures.  The dispatch reads current_res
// ({nr,nt,np} per domain) and refine_axis ({radial,theta,phi} flags per
// domain); these helpers build those vectors compactly so the fake-evaluate
// lambdas stay readable.
// ---------------------------------------------------------------------------
using KadathApps::bns_amr::TailDecision;

// Uniform {nr,nt,np} for every domain.
std::vector<std::array<int, 3>> uniform_res(int ndom, int nr, int nt, int np)
{
    return std::vector<std::array<int, 3>>(static_cast<std::size_t>(ndom),
                                           std::array<int, 3>{nr, nt, np});
}

// Per-domain radial counts, angular pinned to the shared base (nt, nt-1).
std::vector<std::array<int, 3>> radial_res(const std::vector<int>& nr, int nt)
{
    std::vector<std::array<int, 3>> res;
    res.reserve(nr.size());
    for (int n : nr)
        res.push_back({n, nt, nt - 1});
    return res;
}

// Arm a single axis (0=radial, 1=theta, 2=phi) on the listed domains.
std::vector<std::array<char, 3>> axis_flags(int ndom, int axis,
                                            const std::vector<int>& domains)
{
    std::vector<std::array<char, 3>> flags(static_cast<std::size_t>(ndom),
                                           std::array<char, 3>{0, 0, 0});
    for (int d : domains)
        flags[static_cast<std::size_t>(d)][static_cast<std::size_t>(axis)] = 1;
    return flags;
}

void set_tail_demand(TailDecision& decision, int domain, int axis, double demand)
{
    if (decision.refine_demand.size() != decision.refine_axis.size())
        decision.refine_demand.assign(decision.refine_axis.size(), {0.0, 0.0, 0.0});
    decision.refine_demand[static_cast<std::size_t>(domain)]
                            [static_cast<std::size_t>(axis)] = demand;
    if (axis == 0) {
        decision.radial_demand = std::max(decision.radial_demand, demand);
        const int region = KadathApps::bns_amr::h_region_index(decision, domain);
        if (region >= 0)
            decision.radial_region_demand[static_cast<std::size_t>(region)] =
                std::max(decision.radial_region_demand[static_cast<std::size_t>(region)],
                         demand);
    } else {
        decision.angular_demand = std::max(decision.angular_demand, demand);
    }
}

// Flatten a std::vector<Dim_array> regrid target to {nr,nt,np} triples so the
// assertions read like the old radial-only vector<int> comparisons.
std::vector<std::array<int, 3>> as_triples(const std::vector<Dim_array>& target)
{
    std::vector<std::array<int, 3>> out;
    out.reserve(target.size());
    for (const auto& dims : target)
        out.push_back({dims(0), dims(1), dims(2)});
    return out;
}

} // namespace

TEST_CASE("Space_bin_ns per-domain ctor applies the requested point patterns",
          "[bin-ns][per-domain-p]")
{
    std::vector<int> nr(test_domain_count, 7);
    nr[0] = 9;   // NS1 nucleus
    nr[2] = 11;  // NS1 inner adapted shell
    nr[11] = 9;  // compactified

    Space_bin_ns space = make_space(nr);
    REQUIRE(space.get_nbr_domains() == test_domain_count);

    // Radial-only: angular counts stay at the bispheric base (7) everywhere.
    for (int d = 0; d < test_domain_count; ++d) {
        const Dim_array points = space.get_domain(d)->get_nbr_points();
        const bool bispheric = d >= space.OUTER && d < space.OUTER + 5;
        CHECK(points(0) == nr[static_cast<std::size_t>(d)]);
        CHECK(points(1) == 7);
        CHECK(points(2) == (bispheric ? 7 : 6));
    }
}

TEST_CASE("Space_bin_ns uniform ctor matches per-domain ctor with uniform entries",
          "[bin-ns][per-domain-p]")
{
    Space_bin_ns uniform(CHEB_TYPE, test_dist, star_bounds(), star_bounds(), exterior_bounds(), 7);
    Space_bin_ns vectorized = make_space(std::vector<int>(test_domain_count, 7));
    REQUIRE(radial_points(uniform) == radial_points(vectorized));
}

TEST_CASE("Space_bin_ns per-domain ctor rejects malformed resolution vectors",
          "[bin-ns][per-domain-p]")
{
    CHECK_THROWS(make_space(std::vector<int>(test_domain_count - 1, 7)));
    std::vector<int> too_small(test_domain_count, 7);
    too_small[3] = 4;
    CHECK_THROWS(make_space(too_small));
    // The five bispheric domains must share ONE resolution: refining a single
    // bispheric domain (here index 8) out of lockstep with the other four still
    // throws. The AMR block-refinement path bumps all five together instead.
    std::vector<int> refined_bispheric(test_domain_count, 7);
    refined_bispheric[8] = 9;
    CHECK_THROWS(make_space(refined_bispheric));
}

TEST_CASE("Space_bin_ns mixed resolutions survive a save/load roundtrip",
          "[bin-ns][per-domain-p]")
{
    std::vector<int> nr(test_domain_count, 7);
    nr[1] = 9;
    nr[11] = 9;

    const std::string path = temp_space_path();
    {
        Space_bin_ns space = make_space(nr);
        BeFileSink sink(path);
        space.save(sink);
    }
    BeFileSource source(path);
    Space_bin_ns loaded(source);
    std::remove(path.c_str());

    REQUIRE(radial_points(loaded) == nr);
}

TEST_CASE("Mixed-resolution Space_bin_ns keeps the Laplace system square with zero residual",
          "[bin-ns][per-domain-p]")
{
    SECTION("uniform control")
    {
        Space_bin_ns space = make_space(std::vector<int>(test_domain_count, 7));
        require_square_laplace_system(space);
    }
    SECTION("refined nucleus")
    {
        std::vector<int> nr(test_domain_count, 7);
        nr[0] = 9;
        Space_bin_ns space = make_space(nr);
        require_square_laplace_system(space);
    }
    SECTION("refined adapted outer shell")
    {
        std::vector<int> nr(test_domain_count, 7);
        nr[1] = 9;
        Space_bin_ns space = make_space(nr);
        require_square_laplace_system(space);
    }
    SECTION("refined adapted inner shell")
    {
        std::vector<int> nr(test_domain_count, 7);
        nr[2] = 9;
        Space_bin_ns space = make_space(nr);
        require_square_laplace_system(space);
    }
    SECTION("refined NS2 nucleus and adapted pair together")
    {
        std::vector<int> nr(test_domain_count, 7);
        nr[3] = 9;
        nr[4] = 9;
        nr[5] = 9;
        Space_bin_ns space = make_space(nr);
        require_square_laplace_system(space);
    }
    SECTION("refined compactified exterior")
    {
        std::vector<int> nr(test_domain_count, 7);
        nr[11] = 9;
        Space_bin_ns space = make_space(nr);
        require_square_laplace_system(space);
    }
    SECTION("NSHELLS=1 layout with the shell and adapted pair refined")
    {
        std::vector<int> nr(shelled_domain_count, 7);
        nr[1] = 9;  // NS1 adapted-outer
        nr[2] = 9;  // NS1 adapted-inner
        nr[3] = 9;  // NS1 ordinary shell
        Space_bin_ns space = make_shelled_space(nr);
        require_square_laplace_system(space);
    }
}

// The new full-Dim_array constructor must reproduce the radial-ctor behaviour
// when every domain carries the shared angular base: spherical (nr, nt, nt-1)
// and bispheric (nr, nt, nt).  At uniform angular resolution the production
// (conforming) add_eq wiring stays square with a zero u == 1 residual -- this is
// the regression that the new constructor delegates correctly and that the
// conforming matching path is untouched.
TEST_CASE("Full-Dim_array ctor keeps the Laplace system square at the shared angular base",
          "[bin-ns][angular-p]")
{
    Space_bin_ns space = make_space(base_dim_layout());
    require_square_laplace_system(space);
}

TEST_CASE("converged_resolution keeps angular AMR layouts from colliding",
          "[bin-ns][angular-p][policy]")
{
    SECTION("ordinary spherical/bispheric base keeps the legacy radial token")
    {
        Space_bin_ns space = make_space(base_dim_layout());
        CHECK(converged_resolution(space) == "07");
    }

    SECTION("radial mixed layout keeps the existing radial range token")
    {
        std::vector<Dim_array> dims = base_dim_layout();
        dims[3].set(0) = 9;
        Space_bin_ns space = make_space(dims);
        CHECK(converged_resolution(space) == "07_09");
    }

    SECTION("theta-only mixed layout appends a theta range")
    {
        std::vector<Dim_array> dims = base_dim_layout();
        dims[3].set(1) = 9;
        Space_bin_ns space = make_space(dims);
        CHECK(converged_resolution(space) == "07_t07_09");
    }

    SECTION("phi-only mixed layout appends a nonstandard phi range")
    {
        std::vector<Dim_array> dims = base_dim_layout();
        dims[3].set(2) = 8;
        Space_bin_ns space = make_space(dims);
        CHECK(converged_resolution(space) == "07_p06_08");
    }

    SECTION("theta and phi mixed layouts encode both angular ranges")
    {
        std::vector<Dim_array> dims = base_dim_layout();
        dims[3].set(1) = 9;
        dims[3].set(2) = 8;
        Space_bin_ns space = make_space(dims);
        CHECK(converged_resolution(space) == "07_t07_09_p06_08");
    }
}

// Angular p-refinement: add_eq uses split Eq_matching_import for all fixed-radius
// star-internal seams (NS<->ADAPTED outer, shell-to-shell).  Eq_matching_import
// counts conditions in tau-coefficient space (nbr_conditions_boundary), so the
// system stays square when neighbouring domains carry different ntheta/nphi.
// The deformable star surface (ADAPTED.OUTER <-> (ADAPTED+1).INNER) stays
// conforming because both adapted domains share one surface object with the same
// angular counts by construction.
//
// Base: spherical (nr=7, nt=7, np=6), bispheric (7,7,7).
// Refined: nt=9, np=8 (np = nt-1 for spherical domains).
// np=6 (not 4) is used throughout so the np=4 tau/points coincidence cannot mask
// an import-vs-nonconforming counting error.
TEST_CASE("Angular p-refinement with split import keeps the Laplace system square",
          "[bin-ns][angular-p][finding]")
{
    // Base layout at np=6.
    auto np6_base_layout = []() {
        std::vector<Dim_array> dims;
        dims.reserve(test_domain_count);
        for (int d = 0; d < test_domain_count; ++d) {
            Dim_array r(3);
            r.set(0) = test_base;
            r.set(1) = test_base;
            r.set(2) = is_bispheric_domain(d) ? test_base : test_base - 1;
            dims.push_back(r);
        }
        return dims;
    };

    SECTION("uniform np=6 base — import wiring is a no-op regression")
    {
        Space_bin_ns space = make_space(np6_base_layout());
        require_square_laplace_system(space);
    }

    SECTION("refined NS1 nucleus angular (nt=9, np=8)")
    {
        std::vector<Dim_array> dims = np6_base_layout();
        dims[0].set(1) = 9;  // NS1 nucleus ntheta
        dims[0].set(2) = 8;  // nphi = ntheta - 1
        Space_bin_ns space = make_space(dims);
        require_square_laplace_system(space);
    }

    SECTION("refined post-adapted shell angular in NSHELLS=1 layout (nt=9, np=8)")
    {
        // A NSHELLS=1 layout gives one post-adapted shell (domain index 3 in the
        // 13-domain layout) sitting between ADAPTED1+1 (index 2) and the bispheric
        // region.  Refine only the post-adapted shell to exercise the split-import
        // seam at (ADAPTED1+1).OUTER <-> shell.INNER without touching the adapted pair.
        static const std::vector<double> ns1_bounds_1sh{0.5, 1.0, 1.5, 2.0};
        constexpr int ndom_1sh = 13; // 12 + 1 extra shell for star 1
        std::vector<Dim_array> dims;
        dims.reserve(ndom_1sh);
        for (int d = 0; d < ndom_1sh; ++d) {
            Dim_array r(3);
            r.set(0) = test_base;
            r.set(1) = test_base;
            // bispheric: OUTER=7..11 in 13-domain layout
            r.set(2) = (d >= 7 && d <= 11) ? test_base : test_base - 1;
            dims.push_back(r);
        }
        // Domain 3 = the post-adapted shell for star 1.
        dims[3].set(1) = 9;
        dims[3].set(2) = 8;
        Space_bin_ns space(CHEB_TYPE, test_dist, ns1_bounds_1sh, star_bounds(), exterior_bounds(), dims);
        require_square_laplace_system(space);
    }

    SECTION("refined adapted pair together angular (nt=9, np=8)")
    {
        // Both adapted domains in the pair share the star surface and must carry
        // the same nt/np.  The surface seam (ADAPTED1.OUTER) stays conforming.
        std::vector<Dim_array> dims = np6_base_layout();
        dims[1].set(1) = 9;  // ADAPTED1 outer
        dims[1].set(2) = 8;
        dims[2].set(1) = 9;  // ADAPTED1+1 inner (same surface)
        dims[2].set(2) = 8;
        Space_bin_ns space = make_space(dims);
        require_square_laplace_system(space);
    }
}

TEST_CASE("BHNS angular p-refinement with split import keeps the Laplace system square",
          "[bhns][angular-p][finding]")
{
    auto one_ns_shell_layout = []() {
        static const std::vector<double> ns_bounds_1sh{0.5, 1.0, 1.5, 2.0};
        constexpr int ndom_1sh = 13; // 12 + 1 extra NS shell
        constexpr int outer_1sh = 7; // OUTER = 6 + n_shells1 + n_shells2
        std::vector<Dim_array> dims;
        dims.reserve(ndom_1sh);
        for (int d = 0; d < ndom_1sh; ++d) {
            Dim_array r(3);
            r.set(0) = test_base;
            r.set(1) = test_base;
            r.set(2) = (d >= outer_1sh && d < outer_1sh + 5) ? test_base : test_base - 1;
            dims.push_back(r);
        }
        return std::pair{ns_bounds_1sh, dims};
    };

    auto [ns_bounds_1sh, base_dims] = one_ns_shell_layout();
    Space_bhns base_space(CHEB_TYPE, test_dist, ns_bounds_1sh, star_bounds(), exterior_bounds(), base_dims);
    const auto base_counts = bhns_laplace_counts(base_space);

    SECTION("uniform one-NS-shell base has stable minimal-system counts")
    {
        CHECK(base_counts[0] > base_counts[1]);
    }

    SECTION("post-adapted NS shell theta only matches the AMR failure layout")
    {
        auto dims = base_dims;
        dims[3].set(1) = 9; // domain 3 = NS shell 0: (7,7,6) -> (7,9,6)
        Space_bhns space(CHEB_TYPE, test_dist, ns_bounds_1sh, star_bounds(), exterior_bounds(), dims);
        const auto refined_counts = bhns_laplace_counts(space);
        CHECK(refined_counts[0] - base_counts[0] == refined_counts[1] - base_counts[1]);
    }

    SECTION("post-adapted NS shell theta and phi together")
    {
        auto dims = base_dims;
        dims[3].set(1) = 9;
        dims[3].set(2) = 8;
        Space_bhns space(CHEB_TYPE, test_dist, ns_bounds_1sh, star_bounds(), exterior_bounds(), dims);
        const auto refined_counts = bhns_laplace_counts(space);
        CHECK(refined_counts[0] - base_counts[0] == refined_counts[1] - base_counts[1]);
    }
}

TEST_CASE("observe_axis arms only the measured axis of eligible domains above threshold",
          "[bns-amr][per-domain-p]")
{
    TailDecision decision;
    decision.current_res = uniform_res(4, 9, 9, 8);
    decision.refine_axis.assign(4, {0, 0, 0});
    decision.domain_eligible = {1, 1, 0, 1};

    std::vector<bns_hp::SpectralTailRatio> ratios(4);
    for (int d = 0; d < 4; ++d)
        ratios[static_cast<std::size_t>(d)].domain = d;
    ratios[1].l2_ratio = 1.e-3;  // eligible, above threshold
    ratios[2].l2_ratio = 1.e-1;  // ineligible, above threshold
    ratios[3].l2_ratio = 1.e-12; // eligible, below threshold

    // Fold this measurement into the theta axis (index 1).
    KadathApps::bns_amr::observe_axis(decision, "conf", ratios, /*axis=*/1, 1.e-8, 1.e-7);

    CHECK(decision.refine);
    // Only domain 1, only the theta slot is armed; radial/phi stay clear.
    CHECK(decision.refine_axis[1] == std::array<char, 3>{0, 1, 0});
    CHECK(decision.refine_axis[0] == std::array<char, 3>{0, 0, 0});
    CHECK(decision.refine_axis[2] == std::array<char, 3>{0, 0, 0});  // masked
    CHECK(decision.refine_axis[3] == std::array<char, 3>{0, 0, 0});  // below threshold
    CHECK(decision.maximum.domain == 1);
    CHECK(decision.maximum.field == "conf");
}

TEST_CASE("observe_axis keeps each coordinate direction independent on the same domain",
          "[bns-amr][per-domain-p]")
{
    TailDecision decision;
    decision.current_res = uniform_res(3, 9, 9, 8);
    decision.refine_axis.assign(3, {0, 0, 0});
    decision.domain_eligible.assign(3, 1);

    // Radial tail on domain 0, phi tail on domain 2.
    std::vector<bns_hp::SpectralTailRatio> radial(3);
    for (int d = 0; d < 3; ++d)
        radial[static_cast<std::size_t>(d)].domain = d;
    radial[0].l2_ratio = 1.e-3;
    KadathApps::bns_amr::observe_axis(decision, "conf", radial, /*axis=*/0, 1.e-8, 1.e-7);

    std::vector<bns_hp::SpectralTailRatio> phi(3);
    for (int d = 0; d < 3; ++d)
        phi[static_cast<std::size_t>(d)].domain = d;
    phi[2].l2_ratio = 1.e-3;
    KadathApps::bns_amr::observe_axis(decision, "shift", phi, /*axis=*/2, 1.e-8, 1.e-7);

    CHECK(decision.refine);
    CHECK(decision.refine_axis[0] == std::array<char, 3>{1, 0, 0});  // radial only
    CHECK(decision.refine_axis[1] == std::array<char, 3>{0, 0, 0});
    CHECK(decision.refine_axis[2] == std::array<char, 3>{0, 0, 1});  // phi only
}

TEST_CASE("observe_axis records structurally blocked p axes without hiding tail demand",
          "[bns-amr][per-domain-p][policy]")
{
    TailDecision decision;
    decision.current_res = uniform_res(3, 9, 9, 8);
    decision.refine_axis.assign(3, {0, 0, 0});
    decision.domain_eligible.assign(3, 1);
    decision.p_refine_allowed.assign(3, {1, 0, 0});  // radial-only actionability

    std::vector<bns_hp::SpectralTailRatio> theta(3);
    for (int d = 0; d < 3; ++d)
        theta[static_cast<std::size_t>(d)].domain = d;
    theta[1].l2_ratio = 1.e-3;

    KadathApps::bns_amr::observe_axis(decision, "phi", theta, /*axis=*/1, 1.e-8, 1.e-7);

    CHECK(decision.refine);
    CHECK(decision.refine_axis[1] == std::array<char, 3>{0, 0, 0});
    REQUIRE(decision.blocked_refine_axis.size() == 3);
    CHECK(decision.blocked_refine_axis[1] == std::array<char, 3>{0, 1, 0});
    CHECK(decision.angular_demand == Catch::Approx(1.e5));
    CHECK(decision.maximum.domain == 1);
    CHECK(decision.maximum.axis == 1);
}

TEST_CASE("make_tail_decision masks only the bispheric domains",
          "[bns-amr][per-domain-p]")
{
    std::vector<int> nr(test_domain_count, 7);
    nr[5] = 9;
    Space_bin_ns space = make_space(nr);

    BnsAmrConfig config = make_amr_config();
    auto decision = KadathApps::bns_amr::make_tail_decision(config, space);
    REQUIRE(decision.current_nr == nr);
    REQUIRE(decision.first_exterior_domain == space.OUTER);
    REQUIRE(decision.second_star_first_domain == space.NS2);
    // The adapted-pair indices feed the angular lock; they name each star's
    // first adapted domain.
    REQUIRE(decision.adapted1 == space.ADAPTED1);
    REQUIRE(decision.adapted2 == space.ADAPTED2);

    // current_res mirrors each domain's {nr, nt, np} as the space carries it.
    REQUIRE(decision.current_res.size() == static_cast<std::size_t>(test_domain_count));
    for (int d = 0; d < test_domain_count; ++d) {
        const Dim_array points = space.get_domain(d)->get_nbr_points();
        CHECK(decision.current_res[static_cast<std::size_t>(d)] ==
              std::array<int, 3>{points(0), points(1), points(2)});
        CHECK(decision.current_res[static_cast<std::size_t>(d)][0] ==
              decision.current_nr[static_cast<std::size_t>(d)]);
    }
    // refine_axis starts cleared for every domain.
    REQUIRE(decision.refine_axis.size() == static_cast<std::size_t>(test_domain_count));
    for (const auto& axes : decision.refine_axis)
        CHECK(axes == std::array<char, 3>{0, 0, 0});

    // Default mask: bispheric domains (OUTER..OUTER+4) are structurally
    // ineligible; everything else (stars + compactified) is eligible.
    for (int d = 0; d < test_domain_count; ++d) {
        const bool bispheric = d >= space.OUTER && d < space.OUTER + 5;
        CHECK(decision.domain_eligible[static_cast<std::size_t>(d)] == (bispheric ? 0 : 1));
    }

    // The locked bispheric block is anchored at OUTER and enabled by default.
    CHECK(decision.bispheric_first == space.OUTER);
    CHECK(decision.bispheric_enabled);

    // Banner labels name each domain's role in the layout.
    REQUIRE(decision.domain_label.size() == static_cast<std::size_t>(test_domain_count));
    CHECK(decision.domain_label[0] == "NS1 nucleus");
    CHECK(decision.domain_label[1] == "NS1 adapted-outer");
    CHECK(decision.domain_label[2] == "NS1 adapted-inner");
    CHECK(decision.domain_label[3] == "NS2 nucleus");
    CHECK(decision.domain_label[8] == "bispheric eta-first");
    CHECK(decision.domain_label[11] == "compactified");

}

TEST_CASE("make_bhns_tail_decision allows ordinary angular p candidates",
          "[bns-amr][bhns][per-domain-p]")
{
    Space_bhns space = make_bhns_space();
    BnsAmrConfig config = make_bhns_amr_config();
    auto decision = KadathApps::bns_amr::make_bhns_tail_decision(config, space);

    REQUIRE(decision.current_res.size() == static_cast<std::size_t>(space.get_nbr_domains()));
    REQUIRE(decision.p_refine_allowed.size() == decision.current_res.size());
    for (int d = 0; d < space.get_nbr_domains(); ++d) {
        const bool bispheric = d >= space.OUTER && d < space.OUTER + 5;
        CHECK(decision.domain_eligible[static_cast<std::size_t>(d)] == (bispheric ? 0 : 1));
        CHECK(decision.p_refine_allowed[static_cast<std::size_t>(d)] ==
              std::array<char, 3>{1, 1, 1});
    }

    CHECK(decision.second_star_first_domain == space.BH);
    CHECK(decision.first_exterior_domain == space.OUTER);
    CHECK(decision.adapted1 == space.ADAPTEDNS);
    CHECK(decision.adapted2 == space.ADAPTEDBH);
    CHECK(decision.bispheric_first == space.OUTER);
    CHECK(decision.bispheric_enabled);
}

TEST_CASE("AMR defaults select h-first hp mode", "[bns-amr][hp]")
{
    BnsAmrConfig config = make_amr_config();
    CHECK(config.template amr_setting_as<std::string>(AMR_MODE) == "hp");
    CHECK(config.template amr_setting_as<int>(AMR_MAX_SHELLS) == 3);
    CHECK(config.template amr_setting_as<int>(AMR_MAX_NR) == 15);
    CHECK(config.template amr_setting_as<double>(AMR_H_GAIN_MIN) == Catch::Approx(2.0));
    CHECK(config.template amr_setting_as<int>(AMR_H_STALL_CYCLES) == 1);
    CHECK(config.template amr_setting_as<double>(AMR_H_SATURATION_LINF) == Catch::Approx(0.95));
    CHECK(config.template amr_setting_as<double>(AMR_H_ANGULAR_DOMINANCE) == Catch::Approx(1.0));
    CHECK(config.template amr_setting_as<double>(AMR_P_DOF_GROWTH_LIMIT) == Catch::Approx(2.0));
    CHECK(config.template amr_setting_as<double>(AMR_P_FALLBACK_DOF_GROWTH_LIMIT) == Catch::Approx(1.1));
    CHECK(config.template amr_setting_as<int>(AMR_P_FALLBACK_MAX_CANDIDATES) == 1);
}

TEST_CASE("amr_refine_if_needed in p mode refines exactly the flagged domain+axes by +2",
          "[bns-amr][per-domain-p]")
{
    BnsAmrConfig config = make_amr_config();
    config.amr_setting(AMR_ENABLED) = true;
    config.amr_setting(AMR_MODE) = std::string{"p"};

    auto fake_evaluate = [](BnsAmrConfig&) {
        TailDecision decision;
        decision.refine = true;
        decision.current_res = radial_res({9, 9, 11, 9}, /*nt=*/9);
        decision.refine_axis = axis_flags(4, /*axis=*/0, {1, 2});  // radial on 1,2
        decision.domain_eligible.assign(4, 1);
        decision.first_exterior_domain = 3;
        decision.maximum = {"conf", 2, 1.e-3, 1.e-3};
        return decision;
    };

    std::vector<Dim_array> regrid_target;
    std::string regrid_filename;
    const bool refined = KadathApps::bns_amr::amr_refine_if_needed<BnsAmrConfig, Space_bin_ns>(
        config, "", /*rank=*/0, /*cycle=*/1, fake_evaluate,
        [&](BnsAmrConfig&, const std::string& filename, const std::vector<Dim_array>& target) {
            regrid_filename = filename;
            regrid_target = target;
        });

    REQUIRE(refined);
    // Radial +2 on domains 1 and 2 only; angular untouched (stays base 9/8).
    CHECK(as_triples(regrid_target) ==
          std::vector<std::array<int, 3>>{{9, 9, 8}, {11, 9, 8}, {13, 9, 8}, {9, 9, 8}});
    CHECK(static_cast<int>(config(BIN_RES)) == 13);
    CHECK(regrid_filename == "bns_amr_cycle1_res13");
}

TEST_CASE("amr_refine_if_needed in p mode drives per-domain angular nt/np by +2",
          "[bns-amr][per-domain-p][angular-p]")
{
    BnsAmrConfig config = make_amr_config();
    config.amr_setting(AMR_ENABLED) = true;
    config.amr_setting(AMR_MODE) = std::string{"p"};

    auto fake_evaluate = [](BnsAmrConfig&) {
        TailDecision decision;
        decision.refine = true;
        decision.current_res = uniform_res(4, /*nr=*/9, /*nt=*/9, /*np=*/8);
        // theta tail on domain 0, phi tail on domain 2 — radial stays clear.
        decision.refine_axis.assign(4, {0, 0, 0});
        decision.refine_axis[0][1] = 1;  // theta on domain 0
        decision.refine_axis[2][2] = 1;  // phi on domain 2
        decision.domain_eligible.assign(4, 1);
        decision.first_exterior_domain = 3;
        decision.maximum = {"shift", 0, 1.e-3, 1.e-3};
        return decision;
    };

    std::vector<Dim_array> regrid_target;
    const bool refined = KadathApps::bns_amr::amr_refine_if_needed<BnsAmrConfig, Space_bin_ns>(
        config, "", /*rank=*/0, /*cycle=*/4, fake_evaluate,
        [&](BnsAmrConfig&, const std::string&, const std::vector<Dim_array>& target) {
            regrid_target = target;
        });

    REQUIRE(refined);
    // Domain 0 nt 9->11, domain 2 np 8->10; radial counts unchanged everywhere,
    // so the radial ladder (BIN_RES) stays at base 9.
    CHECK(as_triples(regrid_target) ==
          std::vector<std::array<int, 3>>{{9, 11, 8}, {9, 9, 8}, {9, 9, 10}, {9, 9, 8}});
    CHECK(static_cast<int>(config(BIN_RES)) == 9);
}

TEST_CASE("amr_refine_if_needed closes the h gate after a committed p-refinement",
          "[bns-amr][per-domain-p]")
{
    // Probe 11: once a p-cycle commits a regrid, the h gate must CLOSE so a later
    // h-cycle cannot uniform-regrid at base and erase the per-domain p gains just
    // installed. In hp mode an angular-only tail (no radial flag) takes no h-move
    // and falls straight to a p commit, which must set h_closed.
    using KadathApps::bns_amr::RefinementPolicyState;
    BnsAmrConfig config = make_amr_config();
    config.amr_setting(AMR_ENABLED) = true;
    config.amr_setting(AMR_MODE) = std::string{"hp"};

    RefinementPolicyState policy_state;

    auto fake_evaluate = [](BnsAmrConfig&) {
        TailDecision decision;
        decision.refine = true;
        decision.current_res = uniform_res(4, /*nr=*/9, /*nt=*/9, /*np=*/8);
        decision.refine_axis.assign(4, {0, 0, 0});
        decision.refine_axis[0][1] = 1;  // theta on domain 0 only — no radial h
        decision.domain_eligible.assign(4, 1);
        decision.first_exterior_domain = 3;
        decision.second_star_first_domain = 2;  // hp mode requires this set
        decision.maximum = {"shift", 0, 1.e-3, 1.e-3};
        return decision;
    };

    std::vector<Dim_array> regrid_target;
    const bool refined = KadathApps::bns_amr::amr_refine_if_needed<BnsAmrConfig, Space_bin_ns>(
        config, "", /*rank=*/0, /*cycle=*/1, fake_evaluate,
        [&](BnsAmrConfig&, const std::string&, const std::vector<Dim_array>& target) {
            regrid_target = target;
        },
        &policy_state);

    REQUIRE(refined);                       // a p commit happened
    CHECK(policy_state.h_closed);           // probe 11: gate closed after p commit
    CHECK_FALSE(policy_state.has_h_baseline);
}

TEST_CASE("amr_refine_if_needed locks the nucleus to its adapted pair's angular resolution",
          "[bns-amr][per-domain-p][angular-p]")
{
    // A theta tail on the adapted-outer (domain 1) but NOT the nucleus (domain 0)
    // must STILL lift the nucleus's nt to match. The nucleus uses a parity-
    // restricted angular basis whose import matching is non-square against a
    // different-nt adapted neighbour (observed end-to-end as an XCTS system
    // under-determined by Δnt·np). The star-core angular lock keeps the
    // nucleus<->adapted seam conforming. nr and the unrelated domain stay put.
    BnsAmrConfig config = make_amr_config();
    config.amr_setting(AMR_ENABLED) = true;
    config.amr_setting(AMR_MODE) = std::string{"p"};

    auto fake_evaluate = [](BnsAmrConfig&) {
        TailDecision decision;
        decision.refine = true;
        decision.current_res = uniform_res(4, /*nr=*/9, /*nt=*/9, /*np=*/8);
        decision.refine_axis.assign(4, {0, 0, 0});
        decision.refine_axis[1][1] = 1;  // theta on the adapted-outer ONLY
        decision.domain_eligible.assign(4, 1);
        decision.adapted1 = 1;  // nucleus=0, adapted pair = {1,2}; adapted2 unused (-1)
        decision.first_exterior_domain = 3;
        decision.maximum = {"shift", 1, 1.e-3, 1.e-3};
        return decision;
    };

    std::vector<Dim_array> regrid_target;
    const bool refined = KadathApps::bns_amr::amr_refine_if_needed<BnsAmrConfig, Space_bin_ns>(
        config, "", /*rank=*/0, /*cycle=*/1, fake_evaluate,
        [&](BnsAmrConfig&, const std::string&, const std::vector<Dim_array>& target) {
            regrid_target = target;
        });

    REQUIRE(refined);
    // The whole star core {nucleus, adapted-outer, adapted-inner} shares nt=11;
    // np and nr untouched, the unrelated domain 3 untouched. Without the lock the
    // nucleus would stay nt=9 -> a non-square import jump at the nucleus seam.
    CHECK(as_triples(regrid_target) ==
          std::vector<std::array<int, 3>>{{9, 11, 8}, {9, 11, 8}, {9, 11, 8}, {9, 9, 8}});
}

TEST_CASE("amr_refine_if_needed stops without regridding when nothing exceeds",
          "[bns-amr][per-domain-p]")
{
    BnsAmrConfig config = make_amr_config();

    auto fake_evaluate = [](BnsAmrConfig&) {
        TailDecision decision;
        decision.current_res = uniform_res(4, 9, 9, 8);
        decision.refine_axis.assign(4, {0, 0, 0});
        decision.domain_eligible.assign(4, 1);
        // The default config is hp mode, whose h-pass reads both region anchors.
        decision.second_star_first_domain = 2;
        decision.first_exterior_domain = 3;
        return decision;
    };

    bool regrid_called = false;
    const bool refined = KadathApps::bns_amr::amr_refine_if_needed<BnsAmrConfig, Space_bin_ns>(
        config, "", /*rank=*/0, /*cycle=*/0, fake_evaluate,
        [&](BnsAmrConfig&, const std::string&, const std::vector<Dim_array>&) { regrid_called = true; });

    REQUIRE_FALSE(refined);
    REQUIRE_FALSE(regrid_called);
    CHECK(static_cast<int>(config(BIN_RES)) == 9);
}

TEST_CASE("hp mode adds a shell first in flagged regions below max_shells",
          "[bns-amr][hp]")
{
    using KadathApps::bns_amr::TailDecision;
    BnsAmrConfig config = make_amr_config();
    // Plenty of radial headroom: h still goes first.
    config.amr_setting(AMR_MAX_NR) = 17;
    config.set(NSHELLS, BCO1) = 0;
    config.set(NSHELLS, BCO2) = 0;
    config.set(OUTER_SHELLS) = 0;

    // 6 fake domains: 0-1 star1, 2-3 star2, 4-5 exterior (4 = bispheric base).
    auto fake_evaluate = [](BnsAmrConfig&) {
        TailDecision decision;
        decision.refine = true;
        decision.current_res = uniform_res(6, 9, 9, 8);
        decision.refine_axis = axis_flags(6, /*axis=*/0, {1});  // radial on star1
        decision.domain_eligible.assign(6, 1);
        decision.second_star_first_domain = 2;
        decision.first_exterior_domain = 4;
        decision.maximum = {"logh", 1, 1.e-3, 1.e-3};
        return decision;
    };

    std::vector<Dim_array> regrid_target{Dim_array(3)};  // non-empty sentinel
    std::string regrid_filename;
    const bool refined = KadathApps::bns_amr::amr_refine_if_needed<BnsAmrConfig, Space_bin_ns>(
        config, "", /*rank=*/0, /*cycle=*/0, fake_evaluate,
        [&](BnsAmrConfig&, const std::string& filename, const std::vector<Dim_array>& target) {
            regrid_filename = filename;
            regrid_target = target;
        });

    REQUIRE(refined);
    CHECK(regrid_target.empty());                       // uniform-at-base regrid
    CHECK(static_cast<int>(config(BIN_RES)) == 9);      // base, not max
    CHECK(static_cast<int>(config(NSHELLS, BCO1)) == 1);
    CHECK(static_cast<int>(config(NSHELLS, BCO2)) == 0);
    CHECK(static_cast<int>(config(OUTER_SHELLS)) == 0);
    CHECK(regrid_filename == "bns_amr_cycle0_res9_h");
}

TEST_CASE("hp mode keeps h open when the previous h solve reduced radial demand enough",
          "[bns-amr][hp][policy]")
{
    using KadathApps::bns_amr::RefinementPolicyState;
    using KadathApps::bns_amr::TailDecision;
    BnsAmrConfig config = make_amr_config();
    config.set(NSHELLS, BCO1) = 0;
    config.set(NSHELLS, BCO2) = 0;
    config.set(OUTER_SHELLS) = 0;

    RefinementPolicyState policy_state;
    KadathApps::bns_amr::FieldTailMaximum previous_max{"logh", 1, 1.e-3, 0.2, 0};
    policy_state.record_h_baseline(/*radial_demand=*/30.0, /*radial_flagged_domains=*/1,
                                   previous_max);

    auto fake_evaluate = [](BnsAmrConfig&) {
        TailDecision decision;
        decision.refine = true;
        decision.current_res = uniform_res(6, 9, 9, 8);
        decision.refine_axis = axis_flags(6, /*axis=*/0, {1});
        decision.domain_eligible.assign(6, 1);
        decision.second_star_first_domain = 2;
        decision.first_exterior_domain = 4;
        decision.maximum = {"logh", 1, 1.e-3, 0.2, 0};
        set_tail_demand(decision, 1, 0, 10.0);  // 3x gain over the previous h baseline
        return decision;
    };

    std::vector<Dim_array> regrid_target{Dim_array(3)};
    const bool refined = KadathApps::bns_amr::amr_refine_if_needed<BnsAmrConfig, Space_bin_ns>(
        config, "", /*rank=*/0, /*cycle=*/1, fake_evaluate,
        [&](BnsAmrConfig&, const std::string&, const std::vector<Dim_array>& target) {
            regrid_target = target;
        },
        &policy_state);

    REQUIRE(refined);
    CHECK(regrid_target.empty());
    CHECK(static_cast<int>(config(NSHELLS, BCO1)) == 1);
    CHECK(policy_state.has_h_baseline);
    CHECK(policy_state.previous_radial_demand == Catch::Approx(10.0));
    CHECK_FALSE(policy_state.h_closed);
}

TEST_CASE("hp h-gain check follows the previously shelled region when the radial max hops",
          "[bns-amr][hp][policy]")
{
    using KadathApps::bns_amr::RefinementPolicyState;
    using KadathApps::bns_amr::TailDecision;
    BnsAmrConfig config = make_amr_config();
    config.set(NSHELLS, BCO1) = 1;  // previous h move was on star 1
    config.set(NSHELLS, BCO2) = 0;
    config.set(OUTER_SHELLS) = 0;

    RefinementPolicyState policy_state;
    policy_state.record_h_baseline(
        /*radial_demand=*/5.0, /*radial_flagged_domains=*/1,
        {"logh", 1, 1.e-3, 0.2, 0},
        /*radial_region_demand=*/std::array<double, 3>{{5.0, 4.8, 0.0}},
        /*h_regions=*/std::array<char, 3>{{1, 0, 0}});

    auto fake_evaluate = [](BnsAmrConfig&) {
        TailDecision decision;
        decision.refine = true;
        decision.current_res = uniform_res(6, 9, 9, 8);
        decision.refine_axis = axis_flags(6, /*axis=*/0, {3});
        decision.domain_eligible.assign(6, 1);
        decision.second_star_first_domain = 2;
        decision.first_exterior_domain = 4;
        decision.maximum = {"logh", 3, 1.e-3, 0.2, 0};
        set_tail_demand(decision, 1, 0, 0.5);  // star 1 improved 10x
        set_tail_demand(decision, 3, 0, 4.0);  // global max hopped to star 2
        return decision;
    };

    std::vector<Dim_array> regrid_target{Dim_array(3)};
    const bool refined = KadathApps::bns_amr::amr_refine_if_needed<BnsAmrConfig, Space_bin_ns>(
        config, "", /*rank=*/0, /*cycle=*/2, fake_evaluate,
        [&](BnsAmrConfig&, const std::string&, const std::vector<Dim_array>& target) {
            regrid_target = target;
        },
        &policy_state);

    REQUIRE(refined);
    CHECK(regrid_target.empty());
    CHECK(static_cast<int>(config(NSHELLS, BCO1)) == 1);
    CHECK(static_cast<int>(config(NSHELLS, BCO2)) == 1);
    CHECK(policy_state.previous_h_regions == std::array<char, 3>{{0, 1, 0}});
    CHECK(policy_state.previous_radial_region_demand[1] == Catch::Approx(4.0));
}

TEST_CASE("hp mode closes h after a failed layout-gain check and falls back to p",
          "[bns-amr][hp][policy]")
{
    using KadathApps::bns_amr::RefinementPolicyState;
    using KadathApps::bns_amr::TailDecision;
    BnsAmrConfig config = make_amr_config();
    config.amr_setting(AMR_MAX_NR) = 17;
    config.set(NSHELLS, BCO1) = 0;  // h has room, but the policy gate should close it
    config.set(NSHELLS, BCO2) = 0;
    config.set(OUTER_SHELLS) = 0;

    RefinementPolicyState policy_state;
    KadathApps::bns_amr::FieldTailMaximum previous_max{"phi", 1, 0.2, 1.0, 0};
    policy_state.record_h_baseline(/*radial_demand=*/20.0, /*radial_flagged_domains=*/1,
                                   previous_max);

    auto fake_evaluate = [](BnsAmrConfig&) {
        TailDecision decision;
        decision.refine = true;
        decision.current_res = uniform_res(4, 9, 9, 8);
        decision.refine_axis = axis_flags(4, /*axis=*/0, {0, 1});
        decision.domain_eligible.assign(4, 1);
        decision.second_star_first_domain = 2;
        decision.first_exterior_domain = 3;
        decision.maximum = {"phi", 1, 0.2, 1.0, 0};
        set_tail_demand(decision, 0, 0, 15.0);
        set_tail_demand(decision, 1, 0, 12.0);
        return decision;
    };

    std::vector<Dim_array> regrid_target;
    const bool refined = KadathApps::bns_amr::amr_refine_if_needed<BnsAmrConfig, Space_bin_ns>(
        config, "", /*rank=*/0, /*cycle=*/2, fake_evaluate,
        [&](BnsAmrConfig&, const std::string&, const std::vector<Dim_array>& target) {
            regrid_target = target;
        },
        &policy_state);

    REQUIRE(refined);
    CHECK(as_triples(regrid_target) ==
          std::vector<std::array<int, 3>>{{11, 9, 8}, {9, 9, 8}, {9, 9, 8}, {9, 9, 8}});
    CHECK(static_cast<int>(config(NSHELLS, BCO1)) == 0);
    CHECK_FALSE(policy_state.has_h_baseline);
    CHECK(policy_state.h_closed);
}

TEST_CASE("hp mode skips h when angular demand dominates after a useful h solve",
          "[bns-amr][hp][policy]")
{
    using KadathApps::bns_amr::RefinementPolicyState;
    using KadathApps::bns_amr::TailDecision;
    BnsAmrConfig config = make_amr_config();
    config.amr_setting(AMR_MAX_NR) = 17;
    config.set(NSHELLS, BCO1) = 0;  // h has room, but angular demand should own the next move
    config.set(NSHELLS, BCO2) = 0;
    config.set(OUTER_SHELLS) = 0;

    RefinementPolicyState policy_state;
    policy_state.record_h_baseline(/*radial_demand=*/100.0, /*radial_flagged_domains=*/1,
                                   {"logh", 1, 1.e-3, 0.2, 0});

    auto fake_evaluate = [](BnsAmrConfig&) {
        TailDecision decision;
        decision.refine = true;
        decision.current_res = uniform_res(4, 9, 9, 8);
        decision.refine_axis.assign(4, {0, 0, 0});
        decision.refine_axis[1][0] = 1;  // radial h candidate still exists
        decision.refine_axis[3][1] = 1;  // angular tail dominates the remaining error
        decision.domain_eligible.assign(4, 1);
        decision.second_star_first_domain = 2;
        decision.first_exterior_domain = 3;
        decision.maximum = {"phi", 3, 1.e-3, 1.0, 1};
        set_tail_demand(decision, 1, 0, 10.0);   // 10x radial gain versus the h baseline
        set_tail_demand(decision, 3, 1, 200.0);  // but A/R is now 20
        return decision;
    };

    std::vector<Dim_array> regrid_target;
    const bool refined = KadathApps::bns_amr::amr_refine_if_needed<BnsAmrConfig, Space_bin_ns>(
        config, "", /*rank=*/0, /*cycle=*/2, fake_evaluate,
        [&](BnsAmrConfig&, const std::string&, const std::vector<Dim_array>& target) {
            regrid_target = target;
        },
        &policy_state);

    REQUIRE(refined);
    CHECK(as_triples(regrid_target) ==
          std::vector<std::array<int, 3>>{{9, 9, 8}, {9, 9, 8}, {9, 9, 8}, {9, 11, 8}});
    CHECK(static_cast<int>(config(NSHELLS, BCO1)) == 0);
    CHECK_FALSE(policy_state.has_h_baseline);
    // Probe 11: the angular-dominance cycle commits a p-move, which now CLOSES the
    // h gate (was left open before). This deliberately forgoes a later shell on the
    // still-pending radial demand to protect the angular p gains from an h-cycle's
    // uniform-base regrid; an explicit layout reset re-arms h via reset_h_state().
    CHECK(policy_state.h_closed);
}

TEST_CASE("hp mode keeps h closed after a failed layout trial but still allows radial p",
          "[bns-amr][hp][policy]")
{
    using KadathApps::bns_amr::RefinementPolicyState;
    using KadathApps::bns_amr::TailDecision;
    BnsAmrConfig config = make_amr_config();
    config.amr_setting(AMR_MAX_NR) = 17;
    config.set(NSHELLS, BCO1) = 0;  // h has room, but the gate was closed earlier

    RefinementPolicyState policy_state;
    policy_state.close_h_gate();

    auto fake_evaluate = [](BnsAmrConfig&) {
        TailDecision decision;
        decision.refine = true;
        decision.current_res = uniform_res(4, 9, 9, 8);
        decision.refine_axis.assign(4, {0, 0, 0});
        decision.refine_axis[1][0] = 1;  // radial h candidate still exists
        decision.domain_eligible.assign(4, 1);
        decision.second_star_first_domain = 2;
        decision.first_exterior_domain = 3;
        decision.maximum = {"logh", 1, 1.e-3, 1.0, 0};
        set_tail_demand(decision, 1, 0, 200.0);
        return decision;
    };

    std::vector<Dim_array> regrid_target;
    const bool refined = KadathApps::bns_amr::amr_refine_if_needed<BnsAmrConfig, Space_bin_ns>(
        config, "", /*rank=*/0, /*cycle=*/3, fake_evaluate,
        [&](BnsAmrConfig&, const std::string&, const std::vector<Dim_array>& target) {
            regrid_target = target;
        },
        &policy_state);

    REQUIRE(refined);
    CHECK(as_triples(regrid_target) ==
          std::vector<std::array<int, 3>>{{9, 9, 8}, {11, 9, 8}, {9, 9, 8}, {9, 9, 8}});
    CHECK(static_cast<int>(config(NSHELLS, BCO1)) == 0);
    CHECK(policy_state.h_closed);
}

TEST_CASE("hp mode stops when the dominant post-h p target is structurally blocked",
          "[bns-amr][hp][policy]")
{
    using KadathApps::bns_amr::RefinementPolicyState;
    using KadathApps::bns_amr::TailDecision;
    BnsAmrConfig config = make_amr_config();
    config.amr_setting(AMR_MAX_NR) = 17;
    config.set(NSHELLS, BCO1) = 0;  // h has room, but angular demand should own p

    RefinementPolicyState policy_state;
    policy_state.record_h_baseline(/*radial_demand=*/100.0, /*radial_flagged_domains=*/1,
                                   {"logh", 1, 1.e-3, 0.2, 0});

    auto fake_evaluate = [](BnsAmrConfig&) {
        TailDecision decision;
        decision.refine = true;
        decision.current_res = uniform_res(4, 9, 9, 8);
        decision.refine_axis.assign(4, {0, 0, 0});
        decision.blocked_refine_axis.assign(4, {0, 0, 0});
        decision.refine_axis[1][0] = 1;          // lower-priority radial p candidate
        decision.blocked_refine_axis[3][1] = 1;  // dominant angular p target is unsupported
        decision.domain_eligible.assign(4, 1);
        decision.second_star_first_domain = 2;
        decision.first_exterior_domain = 3;
        decision.maximum = {"phi", 3, 1.e-3, 1.0, 1};
        set_tail_demand(decision, 1, 0, 10.0);
        set_tail_demand(decision, 3, 1, 200.0);
        return decision;
    };

    bool regrid_called = false;
    const bool refined = KadathApps::bns_amr::amr_refine_if_needed<BnsAmrConfig, Space_bin_ns>(
        config, "", /*rank=*/0, /*cycle=*/2, fake_evaluate,
        [&](BnsAmrConfig&, const std::string&, const std::vector<Dim_array>&) {
            regrid_called = true;
        },
        &policy_state);

    REQUIRE_FALSE(refined);
    CHECK_FALSE(regrid_called);
    CHECK(static_cast<int>(config(NSHELLS, BCO1)) == 0);
    CHECK_FALSE(policy_state.has_h_baseline);
}

TEST_CASE("hp mode tries h again after angular demand no longer dominates",
          "[bns-amr][hp][policy]")
{
    using KadathApps::bns_amr::RefinementPolicyState;
    using KadathApps::bns_amr::TailDecision;
    BnsAmrConfig config = make_amr_config();
    config.amr_setting(AMR_MAX_NR) = 17;
    config.set(NSHELLS, BCO1) = 0;

    RefinementPolicyState policy_state;

    auto fake_evaluate = [](BnsAmrConfig&) {
        TailDecision decision;
        decision.refine = true;
        decision.current_res = uniform_res(4, 9, 9, 8);
        decision.refine_axis = axis_flags(4, /*axis=*/0, {1});
        decision.domain_eligible.assign(4, 1);
        decision.second_star_first_domain = 2;
        decision.first_exterior_domain = 3;
        decision.maximum = {"logh", 1, 1.e-3, 0.2, 0};
        set_tail_demand(decision, 1, 0, 200.0);
        set_tail_demand(decision, 3, 1, 10.0);
        return decision;
    };

    std::vector<Dim_array> regrid_target{Dim_array(3)};
    const bool refined = KadathApps::bns_amr::amr_refine_if_needed<BnsAmrConfig, Space_bin_ns>(
        config, "", /*rank=*/0, /*cycle=*/4, fake_evaluate,
        [&](BnsAmrConfig&, const std::string&, const std::vector<Dim_array>& target) {
            regrid_target = target;
        },
        &policy_state);

    REQUIRE(refined);
    CHECK(regrid_target.empty());
    CHECK(static_cast<int>(config(NSHELLS, BCO1)) == 1);
    CHECK(policy_state.has_h_baseline);
    CHECK_FALSE(policy_state.h_closed);
}

TEST_CASE("hp mode h-moves only the flagged regions below max_shells",
          "[bns-amr][hp]")
{
    using KadathApps::bns_amr::TailDecision;
    BnsAmrConfig config = make_amr_config();
    config.amr_setting(AMR_MAX_SHELLS) = 1;
    config.set(NSHELLS, BCO1) = 0;   // below cap -> takes the shell
    config.set(NSHELLS, BCO2) = 0;   // not flagged -> untouched
    config.set(OUTER_SHELLS) = 1;    // flagged but at cap -> untouched

    auto fake_evaluate = [](BnsAmrConfig&) {
        TailDecision decision;
        decision.refine = true;
        decision.current_res = uniform_res(6, 9, 9, 8);
        decision.refine_axis = axis_flags(6, /*axis=*/0, {0, 5});  // radial star1 + exterior
        decision.domain_eligible.assign(6, 1);
        decision.second_star_first_domain = 2;
        decision.first_exterior_domain = 4;
        decision.maximum = {"conf", 0, 1.e-3, 1.e-3};
        return decision;
    };

    std::vector<Dim_array> regrid_target{Dim_array(3)};
    const bool refined = KadathApps::bns_amr::amr_refine_if_needed<BnsAmrConfig, Space_bin_ns>(
        config, "", /*rank=*/0, /*cycle=*/2, fake_evaluate,
        [&](BnsAmrConfig&, const std::string&, const std::vector<Dim_array>& target) {
            regrid_target = target;
        });

    REQUIRE(refined);
    CHECK(regrid_target.empty());
    CHECK(static_cast<int>(config(NSHELLS, BCO1)) == 1);
    CHECK(static_cast<int>(config(NSHELLS, BCO2)) == 0);
    CHECK(static_cast<int>(config(OUTER_SHELLS)) == 1);
}

TEST_CASE("hp mode falls back to per-domain p once the flagged region is shell-capped",
          "[bns-amr][hp]")
{
    using KadathApps::bns_amr::TailDecision;
    BnsAmrConfig config = make_amr_config();
    config.amr_setting(AMR_MAX_SHELLS) = 1;
    config.amr_setting(AMR_MAX_NR) = 17;
    config.set(NSHELLS, BCO1) = 1;   // flagged region already at the cap

    auto fake_evaluate = [](BnsAmrConfig&) {
        TailDecision decision;
        decision.refine = true;
        decision.current_res = uniform_res(4, 9, 9, 8);
        decision.refine_axis = axis_flags(4, /*axis=*/0, {1});  // radial on star1 domain
        decision.domain_eligible.assign(4, 1);
        decision.second_star_first_domain = 2;
        decision.first_exterior_domain = 3;
        decision.maximum = {"conf", 1, 1.e-3, 1.e-3};
        return decision;
    };

    std::vector<Dim_array> regrid_target;
    const bool refined = KadathApps::bns_amr::amr_refine_if_needed<BnsAmrConfig, Space_bin_ns>(
        config, "", /*rank=*/0, /*cycle=*/1, fake_evaluate,
        [&](BnsAmrConfig&, const std::string&, const std::vector<Dim_array>& target) {
            regrid_target = target;
        });

    REQUIRE(refined);
    CHECK(as_triples(regrid_target) ==
          std::vector<std::array<int, 3>>{{9, 9, 8}, {11, 9, 8}, {9, 9, 8}, {9, 9, 8}});
    CHECK(static_cast<int>(config(BIN_RES)) == 11);
    CHECK(static_cast<int>(config(NSHELLS, BCO1)) == 1);  // no further shell
}

TEST_CASE("p fallback greedily respects the projected DOF growth trust limit",
          "[bns-amr][per-domain-p][policy]")
{
    using KadathApps::bns_amr::TailDecision;
    BnsAmrConfig config = make_amr_config();
    config.amr_setting(AMR_ENABLED) = true;
    config.amr_setting(AMR_MODE) = std::string{"p"};
    config.amr_setting(AMR_MAX_NR) = 17;
    config.amr_setting(AMR_P_DOF_GROWTH_LIMIT) = 1.12;

    auto fake_evaluate = [](BnsAmrConfig&) {
        TailDecision decision;
        decision.refine = true;
        decision.current_res = uniform_res(4, 9, 9, 8);
        decision.refine_axis.assign(4, {0, 0, 0});
        decision.refine_axis[3][0] = 1;  // cheap radial move
        decision.refine_axis[1][1] = 1;  // expensive star-core angular lock
        decision.domain_eligible.assign(4, 1);
        decision.adapted1 = 1;
        decision.second_star_first_domain = 2;
        decision.first_exterior_domain = 3;
        decision.maximum = {"shift", 1, 1.e-3, 1.e-3, 1};
        set_tail_demand(decision, 3, 0, 20.0);
        set_tail_demand(decision, 1, 1, 10.0);
        return decision;
    };

    std::vector<Dim_array> regrid_target;
    const bool refined = KadathApps::bns_amr::amr_refine_if_needed<BnsAmrConfig, Space_bin_ns>(
        config, "", /*rank=*/0, /*cycle=*/3, fake_evaluate,
        [&](BnsAmrConfig&, const std::string&, const std::vector<Dim_array>& target) {
            regrid_target = target;
        });

    REQUIRE(refined);
    CHECK(as_triples(regrid_target) ==
          std::vector<std::array<int, 3>>{{9, 9, 8}, {9, 9, 8}, {9, 9, 8}, {11, 9, 8}});
    CHECK(static_cast<int>(config(BIN_RES)) == 11);
}

TEST_CASE("post-h p fallback uses the tighter fallback DOF trust limit",
          "[bns-amr][hp][policy]")
{
    using KadathApps::bns_amr::RefinementPolicyState;
    using KadathApps::bns_amr::TailDecision;
    BnsAmrConfig config = make_amr_config();
    config.amr_setting(AMR_MAX_SHELLS) = 0;  // force hp into p fallback
    config.amr_setting(AMR_MAX_NR) = 17;
    config.amr_setting(AMR_P_DOF_GROWTH_LIMIT) = 2.0;
    config.amr_setting(AMR_P_FALLBACK_DOF_GROWTH_LIMIT) = 1.12;

    RefinementPolicyState policy_state;
    policy_state.record_h_baseline(/*radial_demand=*/20.0, /*radial_flagged_domains=*/1,
                                   {"shift", 1, 1.e-3, 1.e-3, 0});

    auto fake_evaluate = [](BnsAmrConfig&) {
        TailDecision decision;
        decision.refine = true;
        decision.current_res = uniform_res(4, 9, 9, 8);
        decision.refine_axis.assign(4, {0, 0, 0});
        decision.refine_axis[3][0] = 1;  // cheap radial move
        decision.refine_axis[1][1] = 1;  // expensive star-core angular lock
        decision.domain_eligible.assign(4, 1);
        decision.adapted1 = 1;
        decision.second_star_first_domain = 2;
        decision.first_exterior_domain = 3;
        decision.maximum = {"shift", 1, 1.e-3, 1.e-3, 1};
        set_tail_demand(decision, 3, 0, 20.0);
        set_tail_demand(decision, 1, 1, 10.0);
        return decision;
    };

    std::vector<Dim_array> regrid_target;
    const bool refined = KadathApps::bns_amr::amr_refine_if_needed<BnsAmrConfig, Space_bin_ns>(
        config, "", /*rank=*/0, /*cycle=*/3, fake_evaluate,
        [&](BnsAmrConfig&, const std::string&, const std::vector<Dim_array>& target) {
            regrid_target = target;
        },
        &policy_state);

    REQUIRE(refined);
    CHECK(as_triples(regrid_target) ==
          std::vector<std::array<int, 3>>{{9, 9, 8}, {9, 9, 8}, {9, 9, 8}, {11, 9, 8}});
    CHECK(static_cast<int>(config(BIN_RES)) == 11);
    CHECK_FALSE(policy_state.has_h_baseline);
}

TEST_CASE("post-h p fallback stages candidate batches after h closes",
          "[bns-amr][hp][policy]")
{
    using KadathApps::bns_amr::RefinementPolicyState;
    using KadathApps::bns_amr::TailDecision;
    BnsAmrConfig config = make_amr_config();
    config.amr_setting(AMR_MAX_SHELLS) = 0;  // force hp into p fallback
    config.amr_setting(AMR_MAX_NR) = 17;
    config.amr_setting(AMR_P_FALLBACK_DOF_GROWTH_LIMIT) = 2.0;
    config.amr_setting(AMR_P_FALLBACK_MAX_CANDIDATES) = 2;

    RefinementPolicyState policy_state;
    policy_state.record_h_baseline(/*radial_demand=*/40.0, /*radial_flagged_domains=*/2,
                                   {"phi", 1, 1.e-3, 1.0, 1});

    auto fake_evaluate = [](BnsAmrConfig&) {
        TailDecision decision;
        decision.refine = true;
        decision.current_res = uniform_res(4, 9, 9, 8);
        decision.refine_axis.assign(4, {0, 0, 0});
        decision.refine_axis[3][0] = 1;
        decision.refine_axis[1][1] = 1;
        decision.refine_axis[2][2] = 1;
        decision.refine_axis[0][1] = 1;
        decision.domain_eligible.assign(4, 1);
        decision.adapted1 = 1;
        decision.second_star_first_domain = 2;
        decision.first_exterior_domain = 3;
        decision.maximum = {"phi", 1, 1.e-3, 1.0, 1};
        set_tail_demand(decision, 3, 0, 40.0);
        set_tail_demand(decision, 1, 1, 30.0);
        set_tail_demand(decision, 2, 2, 20.0);
        set_tail_demand(decision, 0, 1, 10.0);
        return decision;
    };

    std::vector<Dim_array> regrid_target;
    const bool refined = KadathApps::bns_amr::amr_refine_if_needed<BnsAmrConfig, Space_bin_ns>(
        config, "", /*rank=*/0, /*cycle=*/4, fake_evaluate,
        [&](BnsAmrConfig&, const std::string&, const std::vector<Dim_array>& target) {
            regrid_target = target;
        },
        &policy_state);

    REQUIRE(refined);
    CHECK(as_triples(regrid_target) ==
          std::vector<std::array<int, 3>>{{9, 11, 8}, {9, 11, 8}, {9, 11, 8}, {11, 9, 8}});
    CHECK(static_cast<int>(config(BIN_RES)) == 11);
    CHECK_FALSE(policy_state.has_h_baseline);
}

TEST_CASE("hp mode p-fallback drops domains whose target would exceed max_nr",
          "[bns-amr][hp]")
{
    using KadathApps::bns_amr::TailDecision;
    BnsAmrConfig config = make_amr_config();
    config.amr_setting(AMR_MAX_SHELLS) = 0;  // h exhausted from the start
    config.amr_setting(AMR_MAX_NR) = 11;

    auto fake_evaluate = [](BnsAmrConfig&) {
        TailDecision decision;
        decision.refine = true;
        decision.current_res = radial_res({9, 11, 9, 9}, /*nt=*/9);
        decision.refine_axis = axis_flags(4, /*axis=*/0, {0, 1});  // radial on 0,1
        decision.domain_eligible.assign(4, 1);
        decision.second_star_first_domain = 2;
        decision.first_exterior_domain = 3;
        decision.maximum = {"conf", 1, 1.e-3, 1.e-3};
        return decision;
    };

    std::vector<Dim_array> regrid_target;
    const bool refined = KadathApps::bns_amr::amr_refine_if_needed<BnsAmrConfig, Space_bin_ns>(
        config, "", /*rank=*/0, /*cycle=*/0, fake_evaluate,
        [&](BnsAmrConfig&, const std::string&, const std::vector<Dim_array>& target) {
            regrid_target = target;
        });

    REQUIRE(refined);
    // Domain 0 (9 -> 11) fits under the cap; domain 1 (11 -> 13) would exceed
    // it and drops out unrefined.
    CHECK(as_triples(regrid_target) ==
          std::vector<std::array<int, 3>>{{11, 9, 8}, {11, 9, 8}, {9, 9, 8}, {9, 9, 8}});
    CHECK(static_cast<int>(config(BIN_RES)) == 11);
}

TEST_CASE("hp mode stops when shells and radial points are both capped",
          "[bns-amr][hp]")
{
    using KadathApps::bns_amr::TailDecision;
    BnsAmrConfig config = make_amr_config();
    config.amr_setting(AMR_MAX_SHELLS) = 0;
    config.amr_setting(AMR_MAX_NR) = 9;

    auto fake_evaluate = [](BnsAmrConfig&) {
        TailDecision decision;
        decision.refine = true;
        decision.current_res = uniform_res(4, 9, 9, 8);
        decision.refine_axis = axis_flags(4, /*axis=*/0, {1});  // radial
        decision.domain_eligible.assign(4, 1);
        decision.second_star_first_domain = 2;
        decision.first_exterior_domain = 3;
        decision.maximum = {"conf", 1, 1.e-3, 1.e-3};
        return decision;
    };

    bool regrid_called = false;
    const bool refined = KadathApps::bns_amr::amr_refine_if_needed<BnsAmrConfig, Space_bin_ns>(
        config, "", /*rank=*/0, /*cycle=*/0, fake_evaluate,
        [&](BnsAmrConfig&, const std::string&, const std::vector<Dim_array>&) { regrid_called = true; });

    REQUIRE_FALSE(refined);
    REQUIRE_FALSE(regrid_called);
    CHECK(static_cast<int>(config(BIN_RES)) == 9);
}

TEST_CASE("hp mode h-moves every flagged region with shell headroom",
          "[bns-amr][hp]")
{
    using KadathApps::bns_amr::TailDecision;
    BnsAmrConfig config = make_amr_config();
    config.set(NSHELLS, BCO1) = 0;
    config.set(NSHELLS, BCO2) = 0;
    // OUTER_SHELLS deliberately left unset: an absent shell key counts as 0.

    auto fake_evaluate = [](BnsAmrConfig&) {
        TailDecision decision;
        decision.refine = true;
        decision.current_res = uniform_res(6, 9, 9, 8);
        decision.refine_axis = axis_flags(6, /*axis=*/0, {2, 5});  // NS2 + exterior radial
        decision.domain_eligible.assign(6, 1);
        decision.second_star_first_domain = 2;
        decision.first_exterior_domain = 4;
        decision.maximum = {"phi", 5, 1.e-3, 1.e-3};
        return decision;
    };

    std::vector<Dim_array> regrid_target{Dim_array(3)};
    const bool refined = KadathApps::bns_amr::amr_refine_if_needed<BnsAmrConfig, Space_bin_ns>(
        config, "", /*rank=*/0, /*cycle=*/0, fake_evaluate,
        [&](BnsAmrConfig&, const std::string&, const std::vector<Dim_array>& target) {
            regrid_target = target;
        });

    REQUIRE(refined);
    CHECK(regrid_target.empty());
    CHECK(static_cast<int>(config(NSHELLS, BCO1)) == 0);
    CHECK(static_cast<int>(config(NSHELLS, BCO2)) == 1);
    CHECK(static_cast<int>(config(OUTER_SHELLS)) == 1);
}

TEST_CASE("hp mode rejects a negative max_shells", "[bns-amr][hp]")
{
    using KadathApps::bns_amr::TailDecision;
    BnsAmrConfig config = make_amr_config();
    config.amr_setting(AMR_MAX_SHELLS) = -1;

    auto fake_evaluate = [](BnsAmrConfig&) {
        TailDecision decision;
        decision.refine = true;
        decision.current_res = uniform_res(4, 9, 9, 8);
        decision.refine_axis = axis_flags(4, /*axis=*/0, {1});
        decision.domain_eligible.assign(4, 1);
        decision.second_star_first_domain = 2;
        decision.first_exterior_domain = 3;
        return decision;
    };

    CHECK_THROWS(KadathApps::bns_amr::amr_refine_if_needed<BnsAmrConfig, Space_bin_ns>(
        config, "", 0, 0, fake_evaluate,
        [](BnsAmrConfig&, const std::string&, const std::vector<Dim_array>&) {}));
}

TEST_CASE("hp mode requires a blended fixed r_bisph target", "[bns-amr][hp]")
{
    using KadathApps::bns_amr::TailDecision;
    BnsAmrConfig config = make_amr_config();
    config.set(RBISPH_FILL) = 0.;

    auto fake_evaluate = [](BnsAmrConfig&) { return TailDecision{}; };
    CHECK_THROWS(KadathApps::bns_amr::amr_refine_if_needed<BnsAmrConfig, Space_bin_ns>(
        config, "", 0, 0, fake_evaluate,
        [](BnsAmrConfig&, const std::string&, const std::vector<Dim_array>&) {}));
}

// ---------------------------------------------------------------------------
// Per-direction angular refinement.  The retired global-uniform "angular bump"
// is gone: a theta or phi tail now bumps ONLY the flagged domain's nt/np by +2
// (the radial layout and every other domain stay put), and the adapted pair of
// a star moves together because both adapted domains share one surface object.
// The five bispheric domains pin the shared angular base and are never armed.
// ---------------------------------------------------------------------------

// A full 12-domain BNS-layout decision (NS1=0, ADAPTED1=1/2, NS2=3,
// ADAPTED2=4/5, bispheric 6-10, compactified 11), every domain at the shared
// angular base (nt=9, np=8) with the bispheric block ineligible.
TailDecision make_layout12_decision()
{
    TailDecision decision;
    decision.refine = true;
    // Spherical domains carry np = nt-1; the five bispheric domains share the
    // full angular base nt=np=9.
    std::vector<std::array<int, 3>> res;
    res.reserve(12);
    for (int d = 0; d < 12; ++d) {
        const bool bispheric = d >= 6 && d <= 10;
        res.push_back({9, 9, bispheric ? 9 : 8});
    }
    decision.current_res = res;
    decision.refine_axis.assign(12, {0, 0, 0});
    decision.domain_eligible.assign(12, 1);
    for (int d = 6; d <= 10; ++d)
        decision.domain_eligible[static_cast<std::size_t>(d)] = 0;
    decision.second_star_first_domain = 3;
    decision.first_exterior_domain = 6;
    decision.adapted1 = 1;
    decision.adapted2 = 4;
    return decision;
}

TEST_CASE("a theta tail on a star nucleus lifts the whole star core (nucleus + adapted pair)",
          "[bns-amr][per-domain-p][angular-p]")
{
    BnsAmrConfig config = make_amr_config();
    config.amr_setting(AMR_ENABLED) = true;
    config.amr_setting(AMR_MODE) = std::string{"p"};  // uncapped p path

    auto fake_evaluate = [](BnsAmrConfig&) {
        TailDecision decision = make_layout12_decision();
        // Theta tail on the NS2 nucleus (domain 3). The star-core angular lock
        // must carry its adapted pair (domains 4,5) along: the nucleus parity
        // basis cannot sit at a different nt from the adapted domain across an
        // import seam (non-square), and the adapted pair shares the surface.
        decision.refine_axis[3][1] = 1;
        decision.maximum = {"logh", 3, 1.e-3, 1.e-3};
        return decision;
    };

    std::vector<Dim_array> regrid_target;
    std::string regrid_filename;
    const bool refined = KadathApps::bns_amr::amr_refine_if_needed<BnsAmrConfig, Space_bin_ns>(
        config, "", /*rank=*/0, /*cycle=*/3, fake_evaluate,
        [&](BnsAmrConfig&, const std::string& filename, const std::vector<Dim_array>& target) {
            regrid_filename = filename;
            regrid_target = target;
        });

    REQUIRE(refined);
    const auto triples = as_triples(regrid_target);
    REQUIRE(triples.size() == 12);
    // The whole star-2 core {NS2 nucleus=3, adapted pair=4,5} moves to nt=11;
    // nr and np untouched.
    CHECK(triples[3] == std::array<int, 3>{9, 11, 8});
    CHECK(triples[4] == std::array<int, 3>{9, 11, 8});
    CHECK(triples[5] == std::array<int, 3>{9, 11, 8});
    // Every domain outside the star-2 core is unchanged, bispheric block (6-10)
    // included. Star-1 core (0,1,2) is untouched (its tail did not flag).
    for (int d = 0; d < 12; ++d) {
        if (d >= 3 && d <= 5)
            continue;
        const bool bispheric = d >= 6 && d <= 10;
        CHECK(triples[static_cast<std::size_t>(d)] ==
              std::array<int, 3>{9, 9, bispheric ? 9 : 8});
    }
    // No radial change, so the radial ladder stays at base 9 (no "_global"
    // suffix, no global resolution bump).
    CHECK(static_cast<int>(config(BIN_RES)) == 9);
    CHECK(regrid_filename == "bns_amr_cycle3_res9");
}

TEST_CASE("a theta tail on one adapted domain moves the whole star core together",
          "[bns-amr][per-domain-p][angular-p]")
{
    BnsAmrConfig config = make_amr_config();
    config.amr_setting(AMR_ENABLED) = true;
    config.amr_setting(AMR_MODE) = std::string{"p"};

    auto fake_evaluate = [](BnsAmrConfig&) {
        TailDecision decision = make_layout12_decision();
        // Theta tail flagged on only ADAPTED1 (domain 1); the surface partner
        // (domain 2) AND the nucleus (domain 0) must follow — the star core
        // shares one nt so every intra-core seam stays conforming.
        decision.refine_axis[1][1] = 1;
        decision.maximum = {"conf", 1, 1.e-3, 1.e-3};
        return decision;
    };

    std::vector<Dim_array> regrid_target;
    const bool refined = KadathApps::bns_amr::amr_refine_if_needed<BnsAmrConfig, Space_bin_ns>(
        config, "", /*rank=*/0, /*cycle=*/1, fake_evaluate,
        [&](BnsAmrConfig&, const std::string&, const std::vector<Dim_array>& target) {
            regrid_target = target;
        });

    REQUIRE(refined);
    const auto triples = as_triples(regrid_target);
    // The whole star-1 core {nucleus=0, adapted pair=1,2} carries nt=11; np stays
    // at the shared base 8 (every intra-core seam stays conforming).
    CHECK(triples[0] == std::array<int, 3>{9, 11, 8});
    CHECK(triples[1] == std::array<int, 3>{9, 11, 8});
    CHECK(triples[2] == std::array<int, 3>{9, 11, 8});
    // Star-2 core (NS2=3, ADAPTED2 pair=4,5) and the bispheric block are untouched.
    CHECK(triples[3] == std::array<int, 3>{9, 9, 8});
    CHECK(triples[4] == std::array<int, 3>{9, 9, 8});
    CHECK(triples[5] == std::array<int, 3>{9, 9, 8});
    for (int d = 6; d <= 10; ++d)
        CHECK(triples[static_cast<std::size_t>(d)] == std::array<int, 3>{9, 9, 9});
}

TEST_CASE("an angular tail at the max_nr ceiling drops out unrefined in hp mode",
          "[bns-amr][per-domain-p][angular-p][hp]")
{
    BnsAmrConfig config = make_amr_config();  // hp mode
    config.amr_setting(AMR_MAX_SHELLS) = 0;   // h exhausted -> p path
    config.amr_setting(AMR_MAX_NR) = 9;       // base 9 -> 9+2=11 > 9: angular capped

    auto fake_evaluate = [](BnsAmrConfig&) {
        TailDecision decision = make_layout12_decision();
        decision.refine_axis[0][1] = 1;  // theta tail on NS1 nucleus, but capped
        decision.maximum = {"conf", 0, 1.e-3, 1.e-3};
        return decision;
    };

    bool regrid_called = false;
    const bool refined = KadathApps::bns_amr::amr_refine_if_needed<BnsAmrConfig, Space_bin_ns>(
        config, "", /*rank=*/0, /*cycle=*/0, fake_evaluate,
        [&](BnsAmrConfig&, const std::string&, const std::vector<Dim_array>&) { regrid_called = true; });

    REQUIRE_FALSE(refined);
    REQUIRE_FALSE(regrid_called);
    CHECK(static_cast<int>(config(BIN_RES)) == 9);
}

TEST_CASE("radial and angular tails on different domains refine independently in one cycle",
          "[bns-amr][per-domain-p][angular-p]")
{
    BnsAmrConfig config = make_amr_config();
    config.amr_setting(AMR_ENABLED) = true;
    config.amr_setting(AMR_MODE) = std::string{"p"};

    auto fake_evaluate = [](BnsAmrConfig&) {
        TailDecision decision = make_layout12_decision();
        decision.refine_axis[0][0] = 1;  // radial tail on NS1 nucleus
        decision.refine_axis[3][2] = 1;  // phi tail on NS2 nucleus
        decision.maximum = {"shift", 3, 1.e-3, 1.e-3};
        return decision;
    };

    std::vector<Dim_array> regrid_target;
    const bool refined = KadathApps::bns_amr::amr_refine_if_needed<BnsAmrConfig, Space_bin_ns>(
        config, "", /*rank=*/0, /*cycle=*/2, fake_evaluate,
        [&](BnsAmrConfig&, const std::string&, const std::vector<Dim_array>& target) {
            regrid_target = target;
        });

    REQUIRE(refined);
    const auto triples = as_triples(regrid_target);
    CHECK(triples[0] == std::array<int, 3>{11, 9, 8});  // radial +2
    CHECK(triples[3] == std::array<int, 3>{9, 9, 10});  // phi +2
    // The radial ladder tracks the largest radial count = 11.
    CHECK(static_cast<int>(config(BIN_RES)) == 11);
}

// ---------------------------------------------------------------------------
// Bispheric block refinement.  The five bispheric domains share ONE Dim_array
// and refine as a LOCKED BLOCK (all five take the same +2 on a flagged axis):
// their star-side and outer seams are unconditional import (non-conforming),
// so an order jump to the spherical neighbours stays square, while their mutual
// seams stay conforming because the five move together.  The end-to-end gate is
// the square-Laplace probe on a real Space_bin_ns: it exercises every seam
// (including the bispheric<->spherical import seams) across the order jump.
// ---------------------------------------------------------------------------

// Patch only the bispheric block (domains 6..10) to (nr,nt,np).
std::vector<Dim_array> raise_bispheric_block(std::vector<Dim_array> dims, int nr, int nt, int np)
{
    for (int d = 6; d <= 10; ++d) {  // OUTER..OUTER+4 in the 12-domain layout
        dims[static_cast<std::size_t>(d)].set(0) = nr;
        dims[static_cast<std::size_t>(d)].set(1) = nt;
        dims[static_cast<std::size_t>(d)].set(2) = np;
    }
    return dims;
}

// Condition balance of the full Laplace matching set on a conforming layout.
int bispheric_layout_condition_balance(const std::vector<Dim_array>& dims)
{
    Space_bin_ns space = make_space(dims);
    Scalar u(space);
    u = 1.;
    u.std_base();
    System_of_eqs syst(space);
    syst.add_var("u", u);
    space.add_eq(syst, "lap(u) = 0", "u", "dn(u)");
    syst.add_eq_bc(space.get_nbr_domains() - 1, OUTER_BC, "u = 1");
    syst.add_eq_bc(space.ADAPTED1, OUTER_BC, "u = 1");
    syst.add_eq_bc(space.ADAPTED2, OUTER_BC, "u = 1");
    const Array<double> residual = syst.sec_member();  // triggers the condition count
    (void)residual;
    return syst.get_nbr_conditions() - syst.get_nbr_unknowns();
}

// Decisive gate: a real binary space whose bispheric block is raised in PHI above
// its spherical neighbours must keep the full matching set square with a zero
// u==1 residual.  Phi is the only refinable block axis -- it is the same physical
// (azimuthal) coordinate in every bipolar variant, so a uniform +2 keeps the
// conforming internal seams square while the star-side / outer import seams absorb
// the order jump.  Base spherical (7,7,6), bispheric base (7,7,7).
TEST_CASE("A phi-raised bispheric block keeps the Laplace system square with zero residual",
          "[bin-ns][bispheric-block][finding]")
{
    SECTION("uniform base control")
    {
        Space_bin_ns space = make_space(base_dim_layout());
        require_square_laplace_system(space);
    }
    SECTION("phi block raise (bispheric np 7->9)")
    {
        Space_bin_ns space = make_space(raise_bispheric_block(base_dim_layout(), 7, 7, 9));
        require_square_laplace_system(space);
    }
    SECTION("phi block raise twice (bispheric np 7->11)")
    {
        Space_bin_ns space = make_space(raise_bispheric_block(base_dim_layout(), 7, 7, 11));
        require_square_laplace_system(space);
    }
}

// Why the block is phi-ONLY: the bipolar chi/eta axes (Dim_array slots 0 and 1)
// map to DIFFERENT physical directions across the five variants
// (chi_first/rect/eta_first), so a uniform +2 makes the internal seams
// nonconforming.  Eq_matching export requires equal row counts on both sides, so
// the negative probe compares the public counts directly instead of exporting a
// deliberately invalid seam into a buffer sized from only its local side.
TEST_CASE("Raising a bipolar bispheric axis makes its internal seam nonconforming",
          "[bin-ns][bispheric-block][finding]")
{
    CHECK(bispheric_layout_condition_balance(base_dim_layout()) == 0);            // square base
    CHECK(bispheric_layout_condition_balance(raise_bispheric_block(
              base_dim_layout(), 7, 7, 9)) == 0);                                 // phi stays square

    Space_bin_ns space = make_space(raise_bispheric_block(
        base_dim_layout(), 9, 7, 7));
    Scalar u(space);
    u = 1.;
    u.std_base();
    const int rect_domain = space.OUTER + 1;
    const int eta_domain = space.OUTER + 2;
    const Array<int> rect_counts = space.get_domain(rect_domain)->nbr_conditions_boundary(
        u, rect_domain, ETA_PLUS_BC);
    const Array<int> eta_counts = space.get_domain(eta_domain)->nbr_conditions_boundary(
        u, eta_domain, ETA_MINUS_BC);
    REQUIRE(rect_counts.get_nbr() == 1);
    REQUIRE(eta_counts.get_nbr() == 1);
    CHECK(rect_counts(0) != eta_counts(0));
}

// The same phi-block raise must stay square on the NO-SYM binary space: the
// bispheric region (and its seam wiring, add_bispheric_and_outer_wiring) is
// shared between the sym and nosym classes, so the refinable axis is the same.
// refine_bispheric defaults on for the nosym BNS apps too, so this is verified,
// not assumed.  12-domain nosym layout: spherical (7,7,6), bispheric (7,7,8).
TEST_CASE("A phi-raised bispheric block keeps the NOSYM Laplace system square with zero residual",
          "[bin-ns-nosym][bispheric-block][finding]")
{
    auto make_nosym = [](const std::vector<Dim_array>& dims) {
        return Space_bin_ns_nosym(CHEB_TYPE, test_dist, star_bounds(), star_bounds(),
                                  exterior_bounds(), dims);
    };
    // Squareness only: the nosym add_eq carries an asymmetric-star gauge for which
    // u==1 is not the exact Laplace solution (so the residual is not a meaningful
    // probe here), but well-posedness == squareness is exactly what the phi-block
    // raise must preserve, and that IS checkable on the full matching set.
    auto require_square_nosym = [](Space_bin_ns_nosym& space) {
        Scalar u(space);
        u = 1.;
        u.std_base();
        System_of_eqs syst(space);
        syst.add_var("u", u);
        space.add_eq(syst, "lap(u) = 0", "u", "dn(u)");
        syst.add_eq_bc(space.get_nbr_domains() - 1, OUTER_BC, "u = 1");
        syst.add_eq_bc(space.ADAPTED1, OUTER_BC, "u = 1");
        syst.add_eq_bc(space.ADAPTED2, OUTER_BC, "u = 1");
        const Array<double> residual = syst.sec_member();  // triggers the condition count
        (void)residual;
        REQUIRE(syst.get_nbr_unknowns() == syst.get_nbr_conditions());
    };

    SECTION("uniform base control")
    {
        Space_bin_ns_nosym space = make_nosym(nosym_base_dim_layout());
        require_square_nosym(space);
    }
    SECTION("phi block raise (bispheric np 8->10)")
    {
        Space_bin_ns_nosym space = make_nosym(raise_bispheric_block(nosym_base_dim_layout(), 7, 7, 10));
        require_square_nosym(space);
    }
}

TEST_CASE("observe_axis folds the five bispheric domains into the locked block",
          "[bns-amr][bispheric-block]")
{
    TailDecision decision;
    decision.current_res = uniform_res(12, 9, 9, 9);
    decision.refine_axis.assign(12, {0, 0, 0});
    decision.domain_eligible.assign(12, 1);
    for (int d = 6; d <= 10; ++d)
        decision.domain_eligible[static_cast<std::size_t>(d)] = 0;  // masked per-domain
    decision.bispheric_first = 6;
    decision.bispheric_enabled = true;

    std::vector<bns_hp::SpectralTailRatio> ratios(12);
    for (int d = 0; d < 12; ++d)
        ratios[static_cast<std::size_t>(d)].domain = d;
    ratios[8].l2_ratio = 1.e-3;  // a bispheric domain over threshold (phi axis)

    KadathApps::bns_amr::observe_axis(decision, "conf", ratios, /*axis=*/2, 1.e-8, 1.e-7);

    CHECK(decision.refine);
    // The block axis is armed; the per-domain refine_axis stays clear (the five
    // are masked) and the eligible-domain maximum is untouched.
    CHECK(decision.bispheric_refine_axis == std::array<char, 3>{0, 0, 1});
    CHECK(decision.refine_axis[8] == std::array<char, 3>{0, 0, 0});
    CHECK(decision.bispheric_maximum.domain == 8);
    CHECK(decision.bispheric_maximum.field == "conf");
    CHECK(decision.maximum.field.empty());  // no eligible domain contributed
}

TEST_CASE("amr_refine_if_needed bumps the whole bispheric block together in phi",
          "[bns-amr][bispheric-block]")
{
    BnsAmrConfig config = make_amr_config();
    config.amr_setting(AMR_ENABLED) = true;
    config.amr_setting(AMR_MODE) = std::string{"p"};  // uncapped p path

    auto fake_evaluate = [](BnsAmrConfig&) {
        TailDecision decision = make_layout12_decision();
        decision.bispheric_first = 6;
        decision.bispheric_enabled = true;
        decision.bispheric_refine_axis[2] = 1;  // phi tail on the block (the refinable axis)
        decision.bispheric_maximum = {"conf", 8, 1.e-3, 1.e-3};
        return decision;
    };

    std::vector<Dim_array> regrid_target;
    const bool refined = KadathApps::bns_amr::amr_refine_if_needed<BnsAmrConfig, Space_bin_ns>(
        config, "", /*rank=*/0, /*cycle=*/1, fake_evaluate,
        [&](BnsAmrConfig&, const std::string&, const std::vector<Dim_array>& target) {
            regrid_target = target;
        });

    REQUIRE(refined);
    const auto triples = as_triples(regrid_target);
    // All five bispheric domains (6-10) take the same phi +2 -> {9,9,11}; they
    // stay identical (the ctor requires it) and nothing else moves.
    for (int d = 6; d <= 10; ++d)
        CHECK(triples[static_cast<std::size_t>(d)] == std::array<int, 3>{9, 9, 11});
    for (int d = 0; d < 6; ++d)
        CHECK(triples[static_cast<std::size_t>(d)] == std::array<int, 3>{9, 9, 8});
    CHECK(triples[11] == std::array<int, 3>{9, 9, 8});
    // Phi-only: no radial change, so the radial ladder stays at base 9.
    CHECK(static_cast<int>(config(BIN_RES)) == 9);
}

TEST_CASE("amr_refine_if_needed refines the bispheric block in phi only, never the bipolar axes",
          "[bns-amr][bispheric-block]")
{
    BnsAmrConfig config = make_amr_config();
    config.amr_setting(AMR_ENABLED) = true;
    config.amr_setting(AMR_MODE) = std::string{"p"};

    auto fake_evaluate = [](BnsAmrConfig&) {
        TailDecision decision = make_layout12_decision();
        decision.bispheric_first = 6;
        decision.bispheric_enabled = true;
        // All three axes flagged (e.g. the indicator saw chi/eta AND phi tails),
        // but only phi is refinable: the bipolar chi/eta axes must stay put or the
        // conforming internal seams go non-square.
        decision.bispheric_refine_axis = {1, 1, 1};
        decision.bispheric_maximum = {"conf", 7, 1.e-3, 1.e-3};
        return decision;
    };

    std::vector<Dim_array> regrid_target;
    const bool refined = KadathApps::bns_amr::amr_refine_if_needed<BnsAmrConfig, Space_bin_ns>(
        config, "", /*rank=*/0, /*cycle=*/1, fake_evaluate,
        [&](BnsAmrConfig&, const std::string&, const std::vector<Dim_array>& target) {
            regrid_target = target;
        });

    REQUIRE(refined);
    const auto triples = as_triples(regrid_target);
    // Only phi (axis 2) moved: nr and the bipolar axis stay at the base 9.
    for (int d = 6; d <= 10; ++d)
        CHECK(triples[static_cast<std::size_t>(d)] == std::array<int, 3>{9, 9, 11});
}

TEST_CASE("amr_refine_if_needed leaves the bispheric block pinned when disabled",
          "[bns-amr][bispheric-block]")
{
    BnsAmrConfig config = make_amr_config();
    config.amr_setting(AMR_ENABLED) = true;
    config.amr_setting(AMR_MODE) = std::string{"p"};

    auto fake_evaluate = [](BnsAmrConfig&) {
        TailDecision decision = make_layout12_decision();
        decision.bispheric_first = 6;
        decision.bispheric_enabled = false;       // block refinement off
        decision.bispheric_refine_axis[2] = 1;    // phi armed, but must be ignored
        decision.refine_axis[0][0] = 1;           // an eligible radial tail drives the cycle
        decision.maximum = {"conf", 0, 1.e-3, 1.e-3};
        return decision;
    };

    std::vector<Dim_array> regrid_target;
    const bool refined = KadathApps::bns_amr::amr_refine_if_needed<BnsAmrConfig, Space_bin_ns>(
        config, "", /*rank=*/0, /*cycle=*/1, fake_evaluate,
        [&](BnsAmrConfig&, const std::string&, const std::vector<Dim_array>& target) {
            regrid_target = target;
        });

    REQUIRE(refined);
    const auto triples = as_triples(regrid_target);
    CHECK(triples[0] == std::array<int, 3>{11, 9, 8});  // the eligible domain refined
    for (int d = 6; d <= 10; ++d)                        // the block stayed at base
        CHECK(triples[static_cast<std::size_t>(d)] == std::array<int, 3>{9, 9, 9});
}

TEST_CASE("hp mode drops the bispheric block phi bump at the max_nr ceiling",
          "[bns-amr][bispheric-block][hp]")
{
    BnsAmrConfig config = make_amr_config();  // hp mode
    config.amr_setting(AMR_MAX_SHELLS) = 0;   // h exhausted -> p path
    config.amr_setting(AMR_MAX_NR) = 9;       // base 9 -> 9+2=11 > 9: block capped

    auto fake_evaluate = [](BnsAmrConfig&) {
        TailDecision decision = make_layout12_decision();
        decision.bispheric_first = 6;
        decision.bispheric_enabled = true;
        decision.bispheric_refine_axis[2] = 1;  // phi tail on the block, but capped
        decision.bispheric_maximum = {"conf", 7, 1.e-3, 1.e-3};
        return decision;
    };

    bool regrid_called = false;
    const bool refined = KadathApps::bns_amr::amr_refine_if_needed<BnsAmrConfig, Space_bin_ns>(
        config, "", /*rank=*/0, /*cycle=*/0, fake_evaluate,
        [&](BnsAmrConfig&, const std::string&, const std::vector<Dim_array>&) { regrid_called = true; });

    REQUIRE_FALSE(refined);
    REQUIRE_FALSE(regrid_called);
    CHECK(static_cast<int>(config(BIN_RES)) == 9);
}
