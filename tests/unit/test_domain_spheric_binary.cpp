#include <catch2/catch_test_macros.hpp>

#include "For_Kadath/IO/be_file_sink.hpp"
#include "For_Kadath/IO/memory_sink.hpp"
#include "For_Kadath/Domain/spheric.hpp"

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

Point center3d() {
    Point p(3);
    p.set(1) = 0.5;
    p.set(2) = -0.25;
    p.set(3) = 1.0;
    return p;
}

template <typename T>
void check_round_trip(T& original) {
    MemorySink sink1;
    original.save(sink1);
    MemorySource source(sink1.buffer());
    T restored(0, source);
    MemorySink sink2;
    restored.save(sink2);
    REQUIRE(sink1.buffer() == sink2.buffer());
}

} // namespace

TEST_CASE("Domain_nucleus binary round-trip", "[domain_spheric_binary]") {
    Dim_array nbr = nbr3d();
    Point c = center3d();
    Domain_nucleus dom(0, CHEB_TYPE, 1.0, c, nbr);
    check_round_trip(dom);
}

TEST_CASE("Domain_shell binary round-trip", "[domain_spheric_binary]") {
    Dim_array nbr = nbr3d();
    Point c = center3d();
    Domain_shell dom(0, CHEB_TYPE, 1.0, 2.0, c, nbr);
    check_round_trip(dom);
}

TEST_CASE("Domain_compact binary round-trip", "[domain_spheric_binary]") {
    Dim_array nbr = nbr3d();
    Point c = center3d();
    Domain_compact dom(0, CHEB_TYPE, 1.0, c, nbr);
    check_round_trip(dom);
}

TEST_CASE("Domain_shell_log binary round-trip", "[domain_spheric_binary]") {
    Dim_array nbr = nbr3d();
    Point c = center3d();
    Domain_shell_log dom(0, CHEB_TYPE, 1.0, 2.0, c, nbr);
    check_round_trip(dom);
}

TEST_CASE("Domain_shell_surr binary round-trip", "[domain_spheric_binary]") {
    Dim_array nbr = nbr3d();
    Point c = center3d();
    Domain_shell_surr dom(0, CHEB_TYPE, 1.0, 2.0, c, nbr);
    check_round_trip(dom);
}
