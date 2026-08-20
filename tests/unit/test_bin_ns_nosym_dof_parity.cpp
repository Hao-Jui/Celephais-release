#include <catch2/catch_test_macros.hpp>

#include "For_Kadath/Space/bin_ns.hpp"
#include "For_Kadath/Space/bin_ns_nosym.hpp"
#include "For_Kadath/Scalar/scalar.hpp"
#include "For_Kadath/System_of_eqs/system_of_eqs.hpp"

#include <vector>

// DOF-count parity: a well-formed spectral domain layout produces a SQUARE
// Newton system (number of discretized equations == number of unknown
// coefficients) at *every* resolution, independent of whether the physical
// problem converges. The symmetric Space_bin_ns satisfies this; the no-symmetry
// Space_bin_ns_nosym must too. A resolution-dependent surplus (observed at
// res9 in the BNS_nosym solver: 10644 equations vs 10639 unknowns) is a
// structural bug in the nosym domain counting (nbr_conditions vs nbr_unknowns
// drifting out of lockstep), not a physics/gauge issue. This test pins the
// invariant cheaply so the regression cannot reappear silently.

using namespace Kadath;

namespace {

std::vector<double> ns_bounds()
{
    return {1.0, 2.0, 4.0};
}

std::vector<double> outer_bounds()
{
    return {10.0};
}

template <typename SpaceT>
Scalar make_scalar(SpaceT& space)
{
    Scalar u(space);
    u = 1.0;
    u.std_base();
    u.coef();
    return u;
}

template <typename SpaceT>
void check_scalar_square(SpaceT& space)
{
    System_of_eqs sys(space, 0, space.get_nbr_domains() - 1);
    Scalar u = make_scalar(space);
    sys.add_var("u", u);
    space.add_eq(sys, "u = 0", "u", "dn(u)");
    space.add_bc_sphere_one(sys, "u = 0");
    space.add_bc_sphere_two(sys, "u = 0");
    space.add_bc_outer(sys, "u = 0");
    const int nbr_conditions = sys.sec_member().get_nbr();
    INFO("scalar: nbr_conditions=" << nbr_conditions
                                   << " nbr_unknowns=" << sys.get_nbr_unknowns());
    REQUIRE(nbr_conditions == sys.get_nbr_unknowns());
}

} // namespace

// Control: the symmetric Space_bin_ns scalar BVP is exactly square at every
// resolution, confirming the closed test (add_eq + inner/outer boundary
// conditions) is well-posed.
TEST_CASE("Space_bin_ns scalar system is square across resolutions",
          "[bin_ns_nosym_dof_parity]")
{
    for (int nr : {5, 7, 9, 11}) {
        DYNAMIC_SECTION("nr = " << nr)
        {
            Space_bin_ns plane(CHEB_TYPE, 12.0, ns_bounds(), ns_bounds(), outer_bounds(), nr);
            check_scalar_square(plane);
        }
    }
}

// Regression: the no-symmetry Space_bin_ns_nosym scalar BVP must be square at
// every resolution too. It is currently short by (nr-3) conditions (deficit
// 2,4,6,8 at nr 5,7,9,11) — a structural bug in the nosym domain mode counting
// (nbr_conditions vs nbr_unknowns out of lockstep), the same defect that makes
// the BNS_nosym Newton system non-square at res9 (10644 != 10639).
TEST_CASE("Space_bin_ns_nosym scalar system is square across resolutions",
          "[bin_ns_nosym_dof_parity]")
{
    for (int nr : {5, 7, 9, 11}) {
        DYNAMIC_SECTION("nr = " << nr)
        {
            Space_bin_ns_nosym nosym(CHEB_TYPE, 12.0, ns_bounds(), ns_bounds(), outer_bounds(), nr);
            check_scalar_square(nosym);
        }
    }
}
