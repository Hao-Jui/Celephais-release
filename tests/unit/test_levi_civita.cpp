// Unit tests for the flat Cartesian Levi-Civita SYMBOL builder
// (build_levi_civita_symbol, include/For_Kadath/Utilities/levi_civita.hpp).
//
// Three layers of increasing strength:
//   a. components / antisymmetry of the bare symbol hat-eps_ijk.
//   b. the contraction identity  eps^{ijk} eps_{ilm} = d^j_l d^k_m - d^j_m d^k_l
//      evaluated through the real System_of_eqs tensor-contraction machinery.
//   c. DSL end-to-end: inject eps as a constant, build the flat curl
//      curl^i = eps^ijk D_j V_k of analytic vector fields with known curl,
//      and check the result against the analytic answer at sample points.
//      This exercises constant injection + index raising (with the flat
//      metric) + covariant derivative + tensor contraction together.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "For_Kadath/Array/headcpp.hpp"
#include "For_Kadath/Domain/spheric.hpp"
#include "For_Kadath/Scalar/scalar.hpp"
#include "For_Kadath/Tensor/tensor.hpp"
#include "For_Kadath/Metric/metric.hpp"
#include "For_Kadath/System_of_eqs/system_of_eqs.hpp"
#include "For_Kadath/Array/point.hpp"
#include "For_Kadath/Utilities/levi_civita.hpp"

#include <cmath>

using namespace Kadath;
using Catch::Matchers::WithinAbs;

namespace {

// nucleus + 1 shell, NO compact (zec) domain: finite Cartesian coordinates in
// every domain, so the analytic coordinate vector fields below are well defined
// throughout and give_val_def can assemble the curl across all domains.
Space_spheric build_levi_civita_space()
{
    Point center(3);
    center.set(1) = 0.; center.set(2) = 0.; center.set(3) = 0.;

    Dim_array res(3);
    res.set(0) = 9; res.set(1) = 9; res.set(2) = 8;

    Dim_array bd(1);
    bd.set(0) = 2;
    Array<double> bounds(bd);
    bounds.set(0) = 1.0;
    bounds.set(1) = 2.0;

    return Space_spheric(CHEB_TYPE, center, res, bounds, /*withzec=*/false);
}

// Cartesian coordinate component x_dir (dir = 1,2,3) on every domain, as a raw
// per-domain Val_domain. Used to populate vector-component slots; the spectral
// base is fixed afterwards by Vector::std_base() (which assigns the correct
// Cartesian-component base per slot), mirroring coord_fields.hpp::cart()/rot_z().
Scalar cartesian_coordinate(const Space_spheric& space, int dir)
{
    Scalar coordinate(space);
    coordinate = 0.;
    for (int d = 0; d < space.get_nbr_domains(); ++d)
        coordinate.set_domain(d) = space.get_domain(d)->get_cart(dir);
    return coordinate;
}

// Covariant vector field V = (-y, x, 0): a rigid rotation, with curl (0,0,2).
// Assembled exactly like coord_fields.hpp::rot_z(): set the component slots from
// the Cartesian coordinates, then let Vector::std_base() pick each slot's
// Cartesian-component spectral base.
Vector rotation_vector(const Space_spheric& space, const Base_tensor& basis)
{
    const Scalar coord_x = cartesian_coordinate(space, 1);
    const Scalar coord_y = cartesian_coordinate(space, 2);

    Vector field(space, COV, basis);
    field.set(1) = -coord_y;
    field.set(2) = coord_x;
    field.set(3) = 0.;
    field.std_base();
    return field;
}

// Covariant vector field V = (x, y, z) = grad(r^2/2): curl-free, curl = 0.
Vector radial_gradient_vector(const Space_spheric& space, const Base_tensor& basis)
{
    Vector field(space, COV, basis);
    field.set(1) = cartesian_coordinate(space, 1);
    field.set(2) = cartesian_coordinate(space, 2);
    field.set(3) = cartesian_coordinate(space, 3);
    field.std_base();
    return field;
}

} // namespace

// ---------------------------------------------------------------------------
// (a) Component values and total antisymmetry of the bare symbol.
// ---------------------------------------------------------------------------
TEST_CASE("Levi-Civita symbol: component values and antisymmetry", "[levi-civita]")
{
    auto space = build_levi_civita_space();
    const Tensor eps = build_levi_civita_symbol(space);

    REQUIRE(eps.get_valence() == 3);
    REQUIRE(eps.get_index_type(0) == COV);
    REQUIRE(eps.get_index_type(1) == COV);
    REQUIRE(eps.get_index_type(2) == COV);

    Point sample(3);
    sample.set(1) = 0.3; sample.set(2) = -0.4; sample.set(3) = 0.5;

    auto component = [&](int i, int j, int k) {
        return eps(i, j, k).val_point(sample);
    };

    // Even permutations of (1,2,3) -> +1.
    REQUIRE_THAT(component(1, 2, 3), WithinAbs(1.0, 1e-12));
    REQUIRE_THAT(component(2, 3, 1), WithinAbs(1.0, 1e-12));
    REQUIRE_THAT(component(3, 1, 2), WithinAbs(1.0, 1e-12));

    // Odd permutations -> -1.
    REQUIRE_THAT(component(2, 1, 3), WithinAbs(-1.0, 1e-12));
    REQUIRE_THAT(component(1, 3, 2), WithinAbs(-1.0, 1e-12));
    REQUIRE_THAT(component(3, 2, 1), WithinAbs(-1.0, 1e-12));

    // Any repeated index -> 0.
    REQUIRE_THAT(component(1, 1, 2), WithinAbs(0.0, 1e-12));
    REQUIRE_THAT(component(2, 2, 2), WithinAbs(0.0, 1e-12));
    REQUIRE_THAT(component(3, 1, 3), WithinAbs(0.0, 1e-12));
    REQUIRE_THAT(component(1, 2, 2), WithinAbs(0.0, 1e-12));

    // Full antisymmetry under every pairwise index swap, all index triples.
    for (int i = 1; i <= 3; ++i)
        for (int j = 1; j <= 3; ++j)
            for (int k = 1; k <= 3; ++k) {
                const double base = component(i, j, k);
                REQUIRE_THAT(component(j, i, k), WithinAbs(-base, 1e-12)); // swap 1<->2
                REQUIRE_THAT(component(i, k, j), WithinAbs(-base, 1e-12)); // swap 2<->3
                REQUIRE_THAT(component(k, j, i), WithinAbs(-base, 1e-12)); // swap 1<->3
            }
}

// ---------------------------------------------------------------------------
// (b) Contraction identity  eps^{ijk} eps_{ilm} = d^j_l d^k_m - d^j_m d^k_l,
//     evaluated through the real tensor-contraction machinery (add_def + the
//     flat metric to raise the first factor's indices).
// ---------------------------------------------------------------------------
TEST_CASE("Levi-Civita symbol: epsilon-epsilon contraction identity", "[levi-civita]")
{
    auto space = build_levi_civita_space();
    Base_tensor basis(space, CARTESIAN_BASIS);
    Metric_flat fmet(space, basis);

    System_of_eqs syst(space);
    fmet.set_system(syst, "f");
    syst.add_cst("eps", build_levi_civita_symbol(space));

    // identity^{jklm} = eps^{ijk} eps_i^{lm} ; flat metric makes ^ the identity
    // raise, so this is numerically eps_{ijk} eps_{ilm} summed over i.
    syst.add_def("identity^jklm = eps^ijk * eps_i^lm");
    const Tensor identity = syst.give_val_def("identity");

    Point sample(3);
    sample.set(1) = 0.25; sample.set(2) = 0.5; sample.set(3) = -0.35;

    auto delta = [](int a, int b) { return a == b ? 1.0 : 0.0; };

    // Check every (j,k,l,m) combination against the Kronecker-delta identity.
    for (int j = 1; j <= 3; ++j)
        for (int k = 1; k <= 3; ++k)
            for (int l = 1; l <= 3; ++l)
                for (int m = 1; m <= 3; ++m) {
                    const double expected = delta(j, l) * delta(k, m)
                                          - delta(j, m) * delta(k, l);
                    const double actual = identity(j, k, l, m).val_point(sample);
                    REQUIRE_THAT(actual, WithinAbs(expected, 1e-10));
                }
}

// ---------------------------------------------------------------------------
// (c) DSL end-to-end: flat curl of a rigid rotation V = (-y, x, 0) has the
//     known value curl = (0, 0, 2). Exercises injection + raise + derivative +
//     contraction in one add_def string.
// ---------------------------------------------------------------------------
TEST_CASE("Levi-Civita symbol: curl of rigid rotation equals (0,0,2)", "[levi-civita]")
{
    auto space = build_levi_civita_space();
    Base_tensor basis(space, CARTESIAN_BASIS);
    Metric_flat fmet(space, basis);

    System_of_eqs syst(space);
    fmet.set_system(syst, "f");
    syst.add_cst("eps", build_levi_civita_symbol(space));
    syst.add_cst("V", rotation_vector(space, basis));

    syst.add_def("curl^i = eps^ijk * D_j V_k");
    const Tensor curl = syst.give_val_def("curl");

    // Sample in both the nucleus (r < 1) and the shell (1 < r < 2).
    for (const auto& coords : { std::array<double, 3>{0.3, 0.2, 0.1},
                                std::array<double, 3>{1.2, -0.7, 0.5} }) {
        Point sample(3);
        sample.set(1) = coords[0];
        sample.set(2) = coords[1];
        sample.set(3) = coords[2];

        REQUIRE_THAT(curl(1).val_point(sample), WithinAbs(0.0, 1e-9));
        REQUIRE_THAT(curl(2).val_point(sample), WithinAbs(0.0, 1e-9));
        REQUIRE_THAT(curl(3).val_point(sample), WithinAbs(2.0, 1e-9));
    }
}

// ---------------------------------------------------------------------------
// (c') DSL end-to-end, curl-free control: V = (x, y, z) = grad(r^2/2) has
//      zero curl everywhere. Confirms the curl operator returns 0 on a
//      gradient field (no spurious antisymmetric leakage).
// ---------------------------------------------------------------------------
TEST_CASE("Levi-Civita symbol: curl of a gradient field vanishes", "[levi-civita]")
{
    auto space = build_levi_civita_space();
    Base_tensor basis(space, CARTESIAN_BASIS);
    Metric_flat fmet(space, basis);

    System_of_eqs syst(space);
    fmet.set_system(syst, "f");
    syst.add_cst("eps", build_levi_civita_symbol(space));
    syst.add_cst("V", radial_gradient_vector(space, basis));

    syst.add_def("curl^i = eps^ijk * D_j V_k");
    const Tensor curl = syst.give_val_def("curl");

    for (const auto& coords : { std::array<double, 3>{0.3, 0.2, 0.1},
                                std::array<double, 3>{1.2, -0.7, 0.5} }) {
        Point sample(3);
        sample.set(1) = coords[0];
        sample.set(2) = coords[1];
        sample.set(3) = coords[2];

        REQUIRE_THAT(curl(1).val_point(sample), WithinAbs(0.0, 1e-9));
        REQUIRE_THAT(curl(2).val_point(sample), WithinAbs(0.0, 1e-9));
        REQUIRE_THAT(curl(3).val_point(sample), WithinAbs(0.0, 1e-9));
    }
}
