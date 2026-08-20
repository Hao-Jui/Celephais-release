#include <catch2/catch_test_macros.hpp>

#include "For_Kadath/IO/be_file_sink.hpp"
#include "For_Kadath/IO/memory_sink.hpp"
#include "For_Kadath/Domain/adapted.hpp"
#include "For_Kadath/Domain/adapted_polar.hpp"
#include "For_Kadath/Domain/homothetic.hpp"
#include "For_Kadath/Domain/homothetic_polar.hpp"

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

Dim_array nbr3d() {
    Dim_array nbr(3);
    nbr.set(0) = 9;
    nbr.set(1) = 5;
    nbr.set(2) = 4;
    return nbr;
}

Dim_array nbr2d() {
    Dim_array nbr(2);
    nbr.set(0) = 9;
    nbr.set(1) = 5;
    return nbr;
}

Point center3d() {
    Point p(3);
    p.set(1) = 0.5;
    p.set(2) = -0.25;
    p.set(3) = 1.0;
    return p;
}

Point center2d() {
    Point p(2);
    p.set(1) = 0.0;
    p.set(2) = 0.0;
    return p;
}

Array<double> bounds_three() {
    Array<double> b(3);
    b.set(0) = 1.0;
    b.set(1) = 2.0;
    b.set(2) = 3.0;
    return b;
}

template <typename Domain, typename Space>
void check_round_trip(const Space& sp, Domain& original) {
    MemorySink sink1;
    original.save(sink1);
    MemorySource source(sink1.buffer());
    Domain restored(sp, 0, source);
    MemorySink sink2;
    restored.save(sink2);
    REQUIRE(sink1.buffer() == sink2.buffer());
}

} // namespace

TEST_CASE("Domain_shell_inner_homothetic binary round-trip",
          "[domain_homothetic_binary]") {
    Space_spheric_adapted sp(CHEB_TYPE, center3d(), nbr3d(), bounds_three());
    Domain_shell_inner_homothetic dom(sp, 0, CHEB_TYPE, 1.0, 2.0, center3d(), nbr3d());
    check_round_trip(sp, dom);
}

TEST_CASE("Domain_shell_outer_homothetic binary round-trip",
          "[domain_homothetic_binary]") {
    Space_spheric_adapted sp(CHEB_TYPE, center3d(), nbr3d(), bounds_three());
    Domain_shell_outer_homothetic dom(sp, 0, CHEB_TYPE, 1.0, 2.0, center3d(), nbr3d());
    check_round_trip(sp, dom);
}

TEST_CASE("Domain_polar_shell_inner_homothetic binary round-trip",
          "[domain_homothetic_binary]") {
    Space_polar_adapted sp(CHEB_TYPE, center2d(), nbr2d(), bounds_three());
    Domain_polar_shell_inner_homothetic dom(sp, 0, CHEB_TYPE, 1.0, 2.0, center2d(), nbr2d());
    check_round_trip(sp, dom);
}

TEST_CASE("Domain_polar_shell_outer_homothetic binary round-trip",
          "[domain_homothetic_binary]") {
    Space_polar_adapted sp(CHEB_TYPE, center2d(), nbr2d(), bounds_three());
    Domain_polar_shell_outer_homothetic dom(sp, 0, CHEB_TYPE, 1.0, 2.0, center2d(), nbr2d());
    check_round_trip(sp, dom);
}
