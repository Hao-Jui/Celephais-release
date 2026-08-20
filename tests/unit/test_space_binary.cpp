#include <catch2/catch_test_macros.hpp>

#include "For_Kadath/IO/be_file_sink.hpp"
#include "For_Kadath/IO/memory_sink.hpp"
#include "For_Kadath/Domain/oned.hpp"
#include "For_Kadath/Domain/critic.hpp"
#include "For_Kadath/Domain/polar.hpp"
#include "For_Kadath/Domain/spheric.hpp"
#include "For_Kadath/Domain/spheric_symphi.hpp"
#include "For_Kadath/Domain/spheric_periodic.hpp"
#include "For_Kadath/Domain/spheric_time.hpp"
#include "For_Kadath/Domain/polar_periodic.hpp"
#include "For_Kadath/Domain/fourD_periodic.hpp"
#include "For_Kadath/Domain/bispheric.hpp"
#include "For_Kadath/Domain/adapted.hpp"
#include "For_Kadath/Domain/adapted_polar.hpp"
#include "For_Kadath/Space/adapted_bh.hpp"
#include "For_Kadath/Space/adapted_bh_polar.hpp"
#include "For_Kadath/Space/spheric_homothetic.hpp"
#include "For_Kadath/Space/bin_bh.hpp"
#include "For_Kadath/Space/bbh.hpp"
#include "For_Kadath/Space/kerrbin_bh.hpp"
#include "For_Kadath/Space/kerrschild_bh.hpp"
#include "For_Kadath/Space/bin_fake.hpp"
#include "For_Kadath/Space/bin_ns.hpp"
#include "For_Kadath/Space/bin_ns_nosym.hpp"
#include "For_Kadath/Space/bhns.hpp"
#include "For_Kadath/Space/bhns_nosym.hpp"

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

using namespace Kadath;

namespace {
std::vector<unsigned char> read_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    return std::vector<unsigned char>{
        std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>()};
}

Dim_array nbr1d() {
    Dim_array nbr(1);
    nbr.set(0) = 7;
    return nbr;
}

Dim_array nbr2d() {
    Dim_array nbr(2);
    nbr.set(0) = 9;
    nbr.set(1) = 5;
    return nbr;
}

Dim_array nbr3d() {
    Dim_array nbr(3);
    nbr.set(0) = 9;
    nbr.set(1) = 5;
    nbr.set(2) = 4;
    return nbr;
}

Point center3d() {
    Point p(3);
    p.set(1) = 0.0;
    p.set(2) = 0.0;
    p.set(3) = 0.0;
    return p;
}

Point center2d() {
    Point p(2);
    p.set(1) = 0.0;
    p.set(2) = 0.0;
    return p;
}

Array<double> bounds_two() {
    Array<double> b(2);
    b.set(0) = 1.0;
    b.set(1) = 2.0;
    return b;
}

Array<double> bounds_one() {
    Array<double> b(1);
    b.set(0) = 1.0;
    return b;
}

Array<double> bounds_three() {
    Array<double> b(3);
    b.set(0) = 1.0;
    b.set(1) = 2.0;
    b.set(2) = 3.0;
    return b;
}

std::vector<double> bounds_three_vec() {
    return std::vector<double>{1.0, 2.0, 3.0};
}
} // namespace

TEST_CASE("Space_oned round-trip via MemorySink", "[space_binary]") {
    Space_oned original(CHEB_TYPE, nbr1d(), bounds_two());
    MemorySink sink1;
    original.save(sink1);
    MemorySource source(sink1.buffer());
    Space_oned restored(source);
    REQUIRE(restored.get_nbr_domains() == original.get_nbr_domains());
    REQUIRE(restored.get_ndim() == original.get_ndim());
    REQUIRE(restored.get_type_base() == original.get_type_base());
    REQUIRE(restored.get_domain(0)->get_nbr_points()(0) ==
            original.get_domain(0)->get_nbr_points()(0));
    MemorySink sink2;
    restored.save(sink2);
    REQUIRE(sink1.buffer() == sink2.buffer());
}

TEST_CASE("Space_critic round-trip via MemorySink", "[space_binary]") {
    Space_critic original(CHEB_TYPE, /*xlim=*/0.5, nbr2d(), nbr2d());
    MemorySink sink1;
    original.save(sink1);
    MemorySource source(sink1.buffer());
    Space_critic restored(source);
    REQUIRE(restored.get_nbr_domains() == original.get_nbr_domains());
    REQUIRE(restored.get_ndim() == original.get_ndim());
    REQUIRE(restored.get_type_base() == original.get_type_base());
    MemorySink sink2;
    restored.save(sink2);
    REQUIRE(sink1.buffer() == sink2.buffer());
}

TEST_CASE("Space_polar round-trip via MemorySink", "[space_binary]") {
    Space_polar original(CHEB_TYPE, center2d(), nbr2d(), bounds_two());
    MemorySink sink1;
    original.save(sink1);
    MemorySource source(sink1.buffer());
    Space_polar restored(source);
    REQUIRE(restored.get_nbr_domains() == original.get_nbr_domains());
    REQUIRE(restored.get_ndim() == original.get_ndim());
    REQUIRE(restored.get_type_base() == original.get_type_base());
    MemorySink sink2;
    restored.save(sink2);
    REQUIRE(sink1.buffer() == sink2.buffer());
}

TEST_CASE("Space_spheric round-trip via MemorySink (withzec=true)", "[space_binary]") {
    Space_spheric original(CHEB_TYPE, center3d(), nbr3d(), bounds_two(), /*withzec=*/true);
    MemorySink sink1;
    original.save(sink1);
    MemorySource source(sink1.buffer());
    Space_spheric restored(source, /*withzec=*/true);
    REQUIRE(restored.get_nbr_domains() == original.get_nbr_domains());
    REQUIRE(restored.get_ndim() == original.get_ndim());
    REQUIRE(restored.get_type_base() == original.get_type_base());
    MemorySink sink2;
    restored.save(sink2);
    REQUIRE(sink1.buffer() == sink2.buffer());
}

TEST_CASE("Space_spheric round-trip via MemorySink (withzec=false)", "[space_binary]") {
    Space_spheric original(CHEB_TYPE, center3d(), nbr3d(), bounds_two(), /*withzec=*/false);
    MemorySink sink1;
    original.save(sink1);
    MemorySource source(sink1.buffer());
    Space_spheric restored(source, /*withzec=*/false);
    REQUIRE(restored.get_nbr_domains() == original.get_nbr_domains());
    REQUIRE(restored.get_ndim() == original.get_ndim());
    MemorySink sink2;
    restored.save(sink2);
    REQUIRE(sink1.buffer() == sink2.buffer());
}

TEST_CASE("Space_spheric_symphi round-trip via MemorySink (withzec=false)", "[space_binary]") {
    Space_spheric_symphi original(CHEB_TYPE, center3d(), nbr3d(), bounds_two(), /*withzec=*/false);
    MemorySink sink1;
    original.save(sink1);
    MemorySource source(sink1.buffer());
    Space_spheric_symphi restored(source, /*withzec=*/false);
    REQUIRE(restored.get_nbr_domains() == original.get_nbr_domains());
    REQUIRE(restored.get_ndim() == original.get_ndim());
    REQUIRE(restored.get_type_base() == original.get_type_base());
    MemorySink sink2;
    restored.save(sink2);
    REQUIRE(sink1.buffer() == sink2.buffer());
}

TEST_CASE("Space_spheric_periodic round-trip via MemorySink", "[space_binary]") {
    Space_spheric_periodic original(CHEB_TYPE, /*typet=*/TO_PI, nbr2d(), bounds_two(), /*ome=*/1.0);
    MemorySink sink1;
    original.save(sink1);
    MemorySource source(sink1.buffer());
    Space_spheric_periodic restored(source);
    REQUIRE(restored.get_nbr_domains() == original.get_nbr_domains());
    REQUIRE(restored.get_ndim() == original.get_ndim());
    REQUIRE(restored.get_type_base() == original.get_type_base());
    MemorySink sink2;
    restored.save(sink2);
    REQUIRE(sink1.buffer() == sink2.buffer());
}

TEST_CASE("Space_spheric_time round-trip via MemorySink", "[space_binary]") {
    Space_spheric_time original(CHEB_TYPE, nbr2d(), bounds_two(), /*tmin=*/0.0, /*tmax=*/1.0,
                                /*wc=*/false);
    MemorySink sink1;
    original.save(sink1);
    MemorySource source(sink1.buffer());
    Space_spheric_time restored(source);
    REQUIRE(restored.get_nbr_domains() == original.get_nbr_domains());
    REQUIRE(restored.get_ndim() == original.get_ndim());
    REQUIRE(restored.get_type_base() == original.get_type_base());
    MemorySink sink2;
    restored.save(sink2);
    REQUIRE(sink1.buffer() == sink2.buffer());
}

TEST_CASE("Space_polar_periodic round-trip via MemorySink", "[space_binary]") {
    Space_polar_periodic original(CHEB_TYPE, /*omega=*/1.0, nbr3d(), bounds_one());
    MemorySink sink1;
    original.save(sink1);
    MemorySource source(sink1.buffer());
    Space_polar_periodic restored(source);
    REQUIRE(restored.get_nbr_domains() == original.get_nbr_domains());
    REQUIRE(restored.get_ndim() == original.get_ndim());
    REQUIRE(restored.get_type_base() == original.get_type_base());
    MemorySink sink2;
    restored.save(sink2);
    REQUIRE(sink1.buffer() == sink2.buffer());
}

TEST_CASE("Space_fourD_periodic round-trip via MemorySink", "[space_binary]") {
    Dim_array nbr(4);
    nbr.set(0) = 9;
    nbr.set(1) = 5;
    nbr.set(2) = 4;
    nbr.set(3) = 4;
    Space_fourD_periodic original(CHEB_TYPE, /*omega=*/1.0, nbr, bounds_one());
    MemorySink sink1;
    original.save(sink1);
    MemorySource source(sink1.buffer());
    Space_fourD_periodic restored(source);
    REQUIRE(restored.get_nbr_domains() == original.get_nbr_domains());
    REQUIRE(restored.get_ndim() == original.get_ndim());
    REQUIRE(restored.get_type_base() == original.get_type_base());
    MemorySink sink2;
    restored.save(sink2);
    REQUIRE(sink1.buffer() == sink2.buffer());
}

TEST_CASE("Space_bispheric round-trip via MemorySink", "[space_binary]") {
    Space_bispheric original(CHEB_TYPE, /*distance=*/4.0,
                             /*rhor1=*/0.5, /*rshell1=*/1.0,
                             /*rhor2=*/0.5, /*rshell2=*/1.0,
                             /*rext=*/8.0, /*nr=*/5);
    MemorySink sink1;
    original.save(sink1);
    MemorySource source(sink1.buffer());
    Space_bispheric restored(source);
    REQUIRE(restored.get_nbr_domains() == original.get_nbr_domains());
    REQUIRE(restored.get_ndim() == original.get_ndim());
    REQUIRE(restored.get_type_base() == original.get_type_base());
    REQUIRE(restored.get_ndom_minus() == original.get_ndom_minus());
    REQUIRE(restored.get_ndom_plus() == original.get_ndom_plus());
    REQUIRE(restored.get_nshells() == original.get_nshells());
    MemorySink sink2;
    restored.save(sink2);
    REQUIRE(sink1.buffer() == sink2.buffer());
}

TEST_CASE("Space_spheric_adapted round-trip via MemorySink", "[space_binary]") {
    Space_spheric_adapted original(CHEB_TYPE, center3d(), nbr3d(), bounds_three());
    MemorySink sink1;
    original.save(sink1);
    MemorySource source(sink1.buffer());
    Space_spheric_adapted restored(source);
    REQUIRE(restored.get_nbr_domains() == original.get_nbr_domains());
    REQUIRE(restored.get_ndim() == original.get_ndim());
    REQUIRE(restored.get_type_base() == original.get_type_base());
    MemorySink sink2;
    restored.save(sink2);
    REQUIRE(sink1.buffer() == sink2.buffer());
}

TEST_CASE("Space_spheric_homothetic round-trip via MemorySink", "[space_binary]") {
    Space_spheric_homothetic original(CHEB_TYPE, center3d(), nbr3d(), bounds_three());
    REQUIRE(original.nbr_unknowns_from_variable_domains() == 1);
    MemorySink sink1;
    original.save(sink1);
    MemorySource source(sink1.buffer());
    Space_spheric_homothetic restored(source);
    REQUIRE(restored.get_nbr_domains() == original.get_nbr_domains());
    REQUIRE(restored.get_ndim() == original.get_ndim());
    REQUIRE(restored.get_type_base() == original.get_type_base());
    REQUIRE(restored.nbr_unknowns_from_variable_domains() == 1);
    MemorySink sink2;
    restored.save(sink2);
    REQUIRE(sink1.buffer() == sink2.buffer());
}

TEST_CASE("Space_polar_adapted round-trip via MemorySink", "[space_binary]") {
    Space_polar_adapted original(CHEB_TYPE, center2d(), nbr2d(), bounds_three());
    MemorySink sink1;
    original.save(sink1);
    MemorySource source(sink1.buffer());
    Space_polar_adapted restored(source);
    REQUIRE(restored.get_nbr_domains() == original.get_nbr_domains());
    REQUIRE(restored.get_ndim() == original.get_ndim());
    REQUIRE(restored.get_type_base() == original.get_type_base());
    REQUIRE(restored.ADAPTED_OUTER == original.ADAPTED_OUTER);
    REQUIRE(restored.ADAPTED_INNER == original.ADAPTED_INNER);
    MemorySink sink2;
    restored.save(sink2);
    REQUIRE(sink1.buffer() == sink2.buffer());
}

TEST_CASE("Space_polar_bilateral_adapted round-trip via MemorySink", "[space_binary]") {
    Space_polar_bilateral_adapted original(CHEB_TYPE, center2d(), nbr2d(), bounds_three());
    MemorySink sink1;
    original.save(sink1);
    MemorySource source(sink1.buffer());
    Space_polar_bilateral_adapted restored(source);
    REQUIRE(restored.get_nbr_domains() == original.get_nbr_domains());
    REQUIRE(restored.get_ndim() == original.get_ndim());
    REQUIRE(restored.get_type_base() == original.get_type_base());
    REQUIRE(restored.ADAPTED_BILATERAL == original.ADAPTED_BILATERAL);
    MemorySink sink2;
    restored.save(sink2);
    REQUIRE(sink1.buffer() == sink2.buffer());
}

TEST_CASE("Space_adapted_bh round-trip via MemorySink", "[space_binary]") {
    Space_adapted_bh original(CHEB_TYPE, center3d(), nbr3d(), bounds_three_vec());
    MemorySink sink1;
    original.save(sink1);
    MemorySource source(sink1.buffer());
    Space_adapted_bh restored(source);
    REQUIRE(restored.get_nbr_domains() == original.get_nbr_domains());
    REQUIRE(restored.get_ndim() == original.get_ndim());
    REQUIRE(restored.get_type_base() == original.get_type_base());
    MemorySink sink2;
    restored.save(sink2);
    REQUIRE(sink1.buffer() == sink2.buffer());
}

TEST_CASE("Space_adapted_bh_polar round-trip via MemorySink", "[space_binary]") {
    Space_adapted_bh_polar original(CHEB_TYPE, center2d(), nbr2d(), bounds_three_vec());
    MemorySink sink1;
    original.save(sink1);
    MemorySource source(sink1.buffer());
    Space_adapted_bh_polar restored(source);
    REQUIRE(restored.get_nbr_domains() == original.get_nbr_domains());
    REQUIRE(restored.get_ndim() == original.get_ndim());
    REQUIRE(restored.get_type_base() == original.get_type_base());
    REQUIRE(restored.HOMOTHETIC_OUTER == original.HOMOTHETIC_OUTER);
    REQUIRE(restored.HOMOTHETIC_INNER == original.HOMOTHETIC_INNER);
    MemorySink sink2;
    restored.save(sink2);
    REQUIRE(sink1.buffer() == sink2.buffer());
}

TEST_CASE("Space_bin_bh round-trip via MemorySink", "[space_binary]") {
    Space_bin_bh original(CHEB_TYPE, /*dist=*/8.0, /*rbh1=*/1.0, /*rbh2=*/1.0,
                          /*rext=*/8.0, /*nr=*/5);
    MemorySink sink1;
    original.save(sink1);
    MemorySource source(sink1.buffer());
    Space_bin_bh restored(source);
    REQUIRE(restored.get_nbr_domains() == original.get_nbr_domains());
    REQUIRE(restored.get_ndim() == original.get_ndim());
    REQUIRE(restored.get_type_base() == original.get_type_base());
    REQUIRE(restored.BH1 == original.BH1);
    REQUIRE(restored.BH2 == original.BH2);
    REQUIRE(restored.OUTER == original.OUTER);
    MemorySink sink2;
    restored.save(sink2);
    REQUIRE(sink1.buffer() == sink2.buffer());
}

TEST_CASE("Space_bbh round-trip via MemorySink", "[space_binary]") {
    Space_bbh original(CHEB_TYPE, /*dist=*/8.0, /*rbh1=*/1.0, /*rbh2=*/1.0,
                       /*rbi=*/8.0, /*rext=*/16.0, /*nr=*/5);
    MemorySink sink1;
    original.save(sink1);
    MemorySource source(sink1.buffer());
    Space_bbh restored(source);
    REQUIRE(restored.get_nbr_domains() == original.get_nbr_domains());
    REQUIRE(restored.get_ndim() == original.get_ndim());
    REQUIRE(restored.get_type_base() == original.get_type_base());
    MemorySink sink2;
    restored.save(sink2);
    REQUIRE(sink1.buffer() == sink2.buffer());
}

TEST_CASE("Space_Kerr_bbh round-trip via MemorySink", "[space_binary]") {
    Space_Kerr_bbh original(CHEB_TYPE, /*dist=*/8.0,
                            std::vector<double>{1.0, 2.0, 4.0},
                            std::vector<double>{1.0, 2.0, 4.0},
                            std::vector<double>{8.0}, /*nr=*/5);
    MemorySink sink1;
    original.save(sink1);
    MemorySource source(sink1.buffer());
    Space_Kerr_bbh restored(source);
    REQUIRE(restored.get_nbr_domains() == original.get_nbr_domains());
    REQUIRE(restored.get_ndim() == original.get_ndim());
    REQUIRE(restored.get_type_base() == original.get_type_base());
    REQUIRE(restored.BH1 == original.BH1);
    REQUIRE(restored.BH2 == original.BH2);
    REQUIRE(restored.OUTER == original.OUTER);
    MemorySink sink2;
    restored.save(sink2);
    REQUIRE(sink1.buffer() == sink2.buffer());
}

TEST_CASE("Space_KerrSchild_bh round-trip via MemorySink", "[space_binary]") {
    Space_KerrSchild_bh original(CHEB_TYPE, center3d(), nbr3d(),
                                 std::vector<double>{1.0, 2.0, 4.0});
    MemorySink sink1;
    original.save(sink1);
    MemorySource source(sink1.buffer());
    Space_KerrSchild_bh restored(source);
    REQUIRE(restored.get_nbr_domains() == original.get_nbr_domains());
    REQUIRE(restored.get_ndim() == original.get_ndim());
    REQUIRE(restored.get_type_base() == original.get_type_base());
    MemorySink sink2;
    restored.save(sink2);
    REQUIRE(sink1.buffer() == sink2.buffer());
}

TEST_CASE("Space_bin_fake round-trip via MemorySink", "[space_binary]") {
    Space_bin_fake original(CHEB_TYPE, /*dist=*/8.0, /*r1=*/1.0, /*r2=*/1.0,
                            /*rbi=*/8.0, /*rext=*/16.0, /*nr=*/5);
    MemorySink sink1;
    original.save(sink1);
    MemorySource source(sink1.buffer());
    Space_bin_fake restored(source);
    REQUIRE(restored.get_nbr_domains() == original.get_nbr_domains());
    REQUIRE(restored.get_ndim() == original.get_ndim());
    REQUIRE(restored.get_type_base() == original.get_type_base());
    MemorySink sink2;
    restored.save(sink2);
    REQUIRE(sink1.buffer() == sink2.buffer());
}

TEST_CASE("Space_bin_ns round-trip via MemorySink", "[space_binary]") {
    Space_bin_ns original(CHEB_TYPE, /*dist=*/8.0,
                          /*rinstar1=*/0.5, /*rstar1=*/1.0, /*routstar1=*/1.5,
                          /*rinstar2=*/0.5, /*rstar2=*/1.0, /*routstar2=*/1.5,
                          /*rext=*/8.0, /*nr=*/5);
    MemorySink sink1;
    original.save(sink1);
    MemorySource source(sink1.buffer());
    Space_bin_ns restored(source);
    REQUIRE(restored.get_nbr_domains() == original.get_nbr_domains());
    REQUIRE(restored.get_ndim() == original.get_ndim());
    REQUIRE(restored.get_type_base() == original.get_type_base());
    REQUIRE(restored.NS1 == original.NS1);
    REQUIRE(restored.NS2 == original.NS2);
    REQUIRE(restored.ADAPTED1 == original.ADAPTED1);
    REQUIRE(restored.ADAPTED2 == original.ADAPTED2);
    REQUIRE(restored.OUTER == original.OUTER);
    MemorySink sink2;
    restored.save(sink2);
    REQUIRE(sink1.buffer() == sink2.buffer());
}

TEST_CASE("Space_bin_ns_nosym round-trip via MemorySink", "[space_binary]") {
    Space_bin_ns_nosym original(CHEB_TYPE, /*dist=*/8.0,
                                std::vector<double>{0.5, 1.0, 1.5}, // n_shells1 = 0
                                std::vector<double>{0.5, 1.0, 1.5}, // n_shells2 = 0
                                std::vector<double>{8.0}, /*nr=*/5);
    MemorySink sink1;
    original.save(sink1);
    MemorySource source(sink1.buffer());
    Space_bin_ns_nosym restored(source);
    REQUIRE(restored.get_nbr_domains() == original.get_nbr_domains());
    REQUIRE(restored.get_ndim() == original.get_ndim());
    REQUIRE(restored.get_type_base() == original.get_type_base());
    REQUIRE(restored.NS1 == original.NS1);
    REQUIRE(restored.NS2 == original.NS2);
    REQUIRE(restored.ADAPTED1 == original.ADAPTED1);
    REQUIRE(restored.ADAPTED2 == original.ADAPTED2);
    REQUIRE(restored.OUTER == original.OUTER);
    MemorySink sink2;
    restored.save(sink2);
    REQUIRE(sink1.buffer() == sink2.buffer());
}

TEST_CASE("Space_bhns round-trip via MemorySink", "[space_binary]") {
    // Minimal bounds: NS_bounds.size()=3 -> n_shells1=0; BH_bounds.size()=3 -> n_shells2=0;
    // outer_bounds.size()=1 -> n_shells_outer=0.
    Space_bhns original(CHEB_TYPE, /*dist=*/8.0,
                        std::vector<double>{0.5, 1.0, 1.5},
                        std::vector<double>{0.5, 1.0, 1.5},
                        std::vector<double>{8.0}, /*nr=*/5);
    MemorySink sink1;
    original.save(sink1);
    MemorySource source(sink1.buffer());
    Space_bhns restored(source);
    REQUIRE(restored.get_nbr_domains() == original.get_nbr_domains());
    REQUIRE(restored.get_ndim() == original.get_ndim());
    REQUIRE(restored.get_type_base() == original.get_type_base());
    REQUIRE(restored.NS == original.NS);
    REQUIRE(restored.BH == original.BH);
    REQUIRE(restored.ADAPTEDNS == original.ADAPTEDNS);
    REQUIRE(restored.ADAPTEDBH == original.ADAPTEDBH);
    REQUIRE(restored.OUTER == original.OUTER);
    MemorySink sink2;
    restored.save(sink2);
    REQUIRE(sink1.buffer() == sink2.buffer());
}

// Per-domain-resolution BHNS constructor (sym + nosym), used by BHNS AMR. The
// 12-domain no-shell layout puts the five bispheric domains at indices 6..10
// (NS=0, ADAPTEDNS=1/2, BH=3, ADAPTEDBH=4/5, bispheric 6..10, compact 11). The
// bispheric phi block (axis 2) is the refinable axis; raising it must build a
// valid space leaving every other domain at the base.
namespace {
std::vector<Dim_array> bhns_per_domain_layout(int np_bisph)
{
    std::vector<Dim_array> dims;
    dims.reserve(12);
    for (int d = 0; d < 12; ++d) {
        Dim_array r(3);
        const bool bisph = (d >= 6 && d <= 10);
        r.set(0) = 5;
        r.set(1) = 5;
        r.set(2) = bisph ? np_bisph : 4;
        dims.push_back(r);
    }
    return dims;
}
const std::vector<double> bhns_ns_bounds{0.5, 1.0, 1.5};
const std::vector<double> bhns_bh_bounds{0.5, 1.0, 1.5};
const std::vector<double> bhns_out_bounds{8.0};

template <typename space_t>
void check_bhns_phi_block_build(int base_phi, int raised_phi)
{
    space_t base(CHEB_TYPE, 8.0, bhns_ns_bounds, bhns_bh_bounds, bhns_out_bounds,
                 bhns_per_domain_layout(base_phi));
    REQUIRE(base.get_nbr_domains() == 12);
    REQUIRE(base.OUTER == 6);

    space_t raised(CHEB_TYPE, 8.0, bhns_ns_bounds, bhns_bh_bounds, bhns_out_bounds,
                   bhns_per_domain_layout(raised_phi));
    // The five bispheric domains take the raised phi together; every other domain
    // (NS/BH cores, compactified) stays at the base np=4.
    for (int d = 6; d <= 10; ++d)
        CHECK(raised.get_domain(d)->get_nbr_points()(2) == raised_phi);
    for (int d = 0; d <= 5; ++d)
        CHECK(raised.get_domain(d)->get_nbr_points()(2) == 4);
    CHECK(raised.get_domain(11)->get_nbr_points()(2) == 4);
    // Radial/theta untouched everywhere.
    for (int d = 0; d < 12; ++d) {
        CHECK(raised.get_domain(d)->get_nbr_points()(0) == 5);
        CHECK(raised.get_domain(d)->get_nbr_points()(1) == 5);
    }
}

template <typename space_t>
void check_bhns_per_domain_rejects(int base_phi)
{
    // Construct in place (the Space classes own a raw domain table and are not
    // copyable) so CHECK_THROWS sees the ctor exception directly.
    auto make = [](const std::vector<Dim_array>& dims) {
        space_t space(CHEB_TYPE, 8.0, bhns_ns_bounds, bhns_bh_bounds, bhns_out_bounds, dims);
    };
    // Wrong entry count.
    {
        auto dims = bhns_per_domain_layout(base_phi);
        dims.pop_back();
        CHECK_THROWS(make(dims));
    }
    // A single bispheric domain raised out of lockstep with the other four.
    {
        auto dims = bhns_per_domain_layout(base_phi);
        dims[8].set(2) = 8;
        CHECK_THROWS(make(dims));
    }
    // NS adapted pair (domains 1,2) disagreeing in theta.
    {
        auto dims = bhns_per_domain_layout(base_phi);
        dims[1].set(1) = 7;
        CHECK_THROWS(make(dims));
    }
    // BH homothetic pair (domains 4,5) disagreeing in theta.
    {
        auto dims = bhns_per_domain_layout(base_phi);
        dims[4].set(1) = 7;
        CHECK_THROWS(make(dims));
    }
    // Radial resolution below the floor.
    {
        auto dims = bhns_per_domain_layout(base_phi);
        dims[0].set(0) = 4;
        CHECK_THROWS(make(dims));
    }
}
} // namespace

TEST_CASE("Space_bhns per-domain ctor builds a phi-raised bispheric block",
          "[space_binary][bhns][bispheric-block]")
{
    check_bhns_phi_block_build<Space_bhns>(5, 7);
}

TEST_CASE("Space_bhns_nosym per-domain ctor builds a phi-raised bispheric block",
          "[space_binary][bhns][bispheric-block]")
{
    check_bhns_phi_block_build<Space_bhns_nosym>(6, 8);
}

TEST_CASE("Space_bhns per-domain ctor rejects malformed layouts",
          "[space_binary][bhns][bispheric-block]")
{
    check_bhns_per_domain_rejects<Space_bhns>(5);
}

TEST_CASE("Space_bhns_nosym per-domain ctor rejects malformed layouts",
          "[space_binary][bhns][bispheric-block]")
{
    check_bhns_per_domain_rejects<Space_bhns_nosym>(6);
}

// The four cases below exercise the per-star outer shells (NSHELLS>0) via the
// production (bounds-vector) constructors, which lay domains out in BHNS order:
// nucleus, outer-adapted, inner-adapted, then the n_shells outer shells. They
// guard the n_shells persistence in save() and the shared read_adapted_star_domains
// deserialise path: a wrong domain order would misread an adapted domain's bytes
// as a Domain_shell, desyncing the stream so the re-saved buffer differs.
// NS1_bounds = [rin, rmid, s1, rout] (4 entries) -> n_shells1 = 1.

TEST_CASE("Space_bin_ns round-trip with outer shells via MemorySink", "[space_binary]") {
    Space_bin_ns original(CHEB_TYPE, /*dist=*/8.0,
                          std::vector<double>{0.5, 1.0, 1.25, 1.5}, // n_shells1 = 1
                          std::vector<double>{0.5, 1.0, 1.5},       // n_shells2 = 0
                          std::vector<double>{8.0}, /*nr=*/5);
    MemorySink sink1;
    original.save(sink1);
    MemorySource source(sink1.buffer());
    Space_bin_ns restored(source);
    REQUIRE(restored.get_nbr_domains() == 13); // 12 + 1 outer shell
    REQUIRE(restored.get_nbr_domains() == original.get_nbr_domains());
    REQUIRE(restored.NS1 == 0);
    REQUIRE(restored.ADAPTED1 == 1);
    REQUIRE(restored.NS2 == 4); // ADAPTED1 + 2 + n_shells1
    REQUIRE(restored.OUTER == 7);
    REQUIRE(restored.NS2 == original.NS2);
    REQUIRE(restored.OUTER == original.OUTER);
    MemorySink sink2;
    restored.save(sink2);
    REQUIRE(sink1.buffer() == sink2.buffer());
}

TEST_CASE("Space_bin_ns_nosym round-trip with outer shells via MemorySink", "[space_binary]") {
    Space_bin_ns_nosym original(CHEB_TYPE, /*dist=*/8.0,
                                std::vector<double>{0.5, 1.0, 1.25, 1.5}, // n_shells1 = 1
                                std::vector<double>{0.5, 1.0, 1.5},       // n_shells2 = 0
                                std::vector<double>{8.0}, /*nr=*/5);
    MemorySink sink1;
    original.save(sink1);
    MemorySource source(sink1.buffer());
    Space_bin_ns_nosym restored(source);
    REQUIRE(restored.get_nbr_domains() == 13);
    REQUIRE(restored.get_nbr_domains() == original.get_nbr_domains());
    REQUIRE(restored.NS1 == 0);
    REQUIRE(restored.ADAPTED1 == 1);
    REQUIRE(restored.NS2 == 4); // BHNS order: shells after the adapted pair
    REQUIRE(restored.OUTER == 7);
    REQUIRE(restored.NS2 == original.NS2);
    REQUIRE(restored.OUTER == original.OUTER);
    MemorySink sink2;
    restored.save(sink2);
    REQUIRE(sink1.buffer() == sink2.buffer());
}

TEST_CASE("Space_bhns round-trip with outer shells via MemorySink", "[space_binary]") {
    Space_bhns original(CHEB_TYPE, /*dist=*/8.0,
                        std::vector<double>{0.5, 1.0, 1.25, 1.5}, // NS: n_shells1 = 1
                        std::vector<double>{0.5, 1.0, 1.5},       // BH: n_shells2 = 0
                        std::vector<double>{8.0}, /*nr=*/5);
    MemorySink sink1;
    original.save(sink1);
    MemorySource source(sink1.buffer());
    Space_bhns restored(source);
    REQUIRE(restored.get_nbr_domains() == 13);
    REQUIRE(restored.get_nbr_domains() == original.get_nbr_domains());
    REQUIRE(restored.NS == 0);
    REQUIRE(restored.ADAPTEDNS == 1);
    REQUIRE(restored.BH == 4); // NS nucleus + adapted pair + 1 shell
    REQUIRE(restored.OUTER == 7);
    REQUIRE(restored.BH == original.BH);
    REQUIRE(restored.OUTER == original.OUTER);
    MemorySink sink2;
    restored.save(sink2);
    REQUIRE(sink1.buffer() == sink2.buffer());
}

TEST_CASE("Space_bhns_nosym round-trip with outer shells via MemorySink", "[space_binary]") {
    Space_bhns_nosym original(CHEB_TYPE, /*dist=*/8.0,
                              std::vector<double>{0.5, 1.0, 1.25, 1.5}, // NS: n_shells1 = 1
                              std::vector<double>{0.5, 1.0, 1.5},       // BH: n_shells2 = 0
                              std::vector<double>{8.0}, /*nr=*/5);
    MemorySink sink1;
    original.save(sink1);
    MemorySource source(sink1.buffer());
    Space_bhns_nosym restored(source);
    REQUIRE(restored.get_nbr_domains() == 13);
    REQUIRE(restored.get_nbr_domains() == original.get_nbr_domains());
    REQUIRE(restored.NS == 0);
    REQUIRE(restored.ADAPTEDNS == 1);
    REQUIRE(restored.BH == 4);
    REQUIRE(restored.OUTER == 7);
    REQUIRE(restored.BH == original.BH);
    REQUIRE(restored.OUTER == original.OUTER);
    MemorySink sink2;
    restored.save(sink2);
    REQUIRE(sink1.buffer() == sink2.buffer());
}
