#include <catch2/catch_test_macros.hpp>

#include "For_Kadath/IO/be_file_sink.hpp"
#include "For_Kadath/IO/memory_sink.hpp"
#include "For_Kadath/Domain/bispheric.hpp"
#include "For_Kadath/Domain/bispheric_nosym.hpp"
#include "For_Kadath/Val_domain/val_domain.hpp"

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

Dim_array nbr3d(int phi) {
    Dim_array nbr(3);
    nbr.set(0) = 9;
    nbr.set(1) = 5;
    nbr.set(2) = phi;
    return nbr;
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

TEST_CASE("Domain_bispheric_chi_first binary round-trip",
          "[domain_bispheric_binary]") {
    Dim_array nbr = nbr3d(5);
    Domain_bispheric_chi_first dom(0, CHEB_TYPE, /*aa=*/1.0, /*etalim=*/0.5,
                                   /*rr=*/2.0, /*chi_max=*/1.5, nbr);
    check_round_trip(dom);
}

TEST_CASE("Domain_bispheric_eta_first binary round-trip",
          "[domain_bispheric_binary]") {
    Dim_array nbr = nbr3d(5);
    Domain_bispheric_eta_first dom(0, CHEB_TYPE, /*aa=*/1.0, /*rr=*/2.0,
                                   /*eta_min=*/-0.5, /*eta_max=*/0.5, nbr);
    check_round_trip(dom);
}

TEST_CASE("Domain_bispheric_rect binary round-trip",
          "[domain_bispheric_binary]") {
    Dim_array nbr = nbr3d(5);
    Domain_bispheric_rect dom(0, CHEB_TYPE, /*aa=*/1.0, /*rext=*/2.0,
                              /*eta_minus=*/-0.5, /*eta_plus=*/0.5,
                              /*chi_min=*/0.5, nbr);
    check_round_trip(dom);
}

TEST_CASE("Domain_bispheric_chi_first_nosym binary round-trip",
          "[domain_bispheric_nosym_binary]") {
    Dim_array nbr = nbr3d(6);
    Domain_bispheric_chi_first_nosym dom(0, CHEB_TYPE, /*aa=*/1.0, /*etalim=*/0.5,
                                         /*rr=*/2.0, /*chi_max=*/1.5, nbr);
    check_round_trip(dom);
}

TEST_CASE("Domain_bispheric_eta_first_nosym binary round-trip",
          "[domain_bispheric_nosym_binary]") {
    Dim_array nbr = nbr3d(6);
    Domain_bispheric_eta_first_nosym dom(0, CHEB_TYPE, /*aa=*/1.0, /*rr=*/2.0,
                                         /*eta_min=*/-0.5, /*eta_max=*/0.5, nbr);
    check_round_trip(dom);
}

TEST_CASE("Domain_bispheric_rect_nosym binary round-trip",
          "[domain_bispheric_nosym_binary]") {
    Dim_array nbr = nbr3d(6);
    Domain_bispheric_rect_nosym dom(0, CHEB_TYPE, /*aa=*/1.0, /*rext=*/2.0,
                                    /*eta_minus=*/-0.5, /*eta_plus=*/0.5,
                                    /*chi_min=*/0.5, nbr);
    check_round_trip(dom);
}

TEST_CASE("Bispheric_nosym boundary points use full COSSIN phi grid",
          "[domain_bispheric_nosym_binary]") {
    Dim_array nbr = nbr3d(6);

    Domain_bispheric_rect_nosym rect(0, CHEB_TYPE, /*aa=*/1.0, /*rext=*/2.0,
                                     /*eta_minus=*/-0.5, /*eta_plus=*/0.5,
                                     /*chi_min=*/0.5, nbr);
    Val_domain rect_field(&rect);
    rect_field = 1.0;
    rect_field.std_base();
    REQUIRE(rect_field.get_base().get_base_1d(2)->get_data()[0] == COSSIN);
    CHECK(rect.nbr_points_boundary(INNER_BC, rect_field.get_base()) == nbr(2) * (nbr(1) - 1) + 1);
    CHECK(rect.nbr_points_boundary(OUTER_BC, rect_field.get_base()) == nbr(2));

    Domain_bispheric_chi_first_nosym chi_first(0, CHEB_TYPE, /*aa=*/1.0, /*etalim=*/0.5,
                                               /*rr=*/2.0, /*chi_max=*/1.5, nbr);
    Val_domain chi_field(&chi_first);
    chi_field = 1.0;
    chi_field.std_base();
    REQUIRE(chi_field.get_base().get_base_1d(2)->get_data()[0] == COSSIN);
    CHECK(chi_first.nbr_points_boundary(INNER_BC, chi_field.get_base()) == nbr(2) * (nbr(1) - 1) + 1);
    CHECK(chi_first.nbr_points_boundary(OUTER_BC, chi_field.get_base()) == nbr(2) * (nbr(1) - 1) + 1);

    Domain_bispheric_eta_first_nosym eta_first(0, CHEB_TYPE, /*aa=*/1.0, /*rr=*/2.0,
                                               /*eta_min=*/-0.5, /*eta_max=*/0.5, nbr);
    Val_domain eta_field(&eta_first);
    eta_field = 1.0;
    eta_field.std_base();
    REQUIRE(eta_field.get_base().get_base_1d(2)->get_data()[0] == COSSIN);
    CHECK(eta_first.nbr_points_boundary(OUTER_BC, eta_field.get_base()) == nbr(2) * nbr(1));
}
