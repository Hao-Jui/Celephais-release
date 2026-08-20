// Square-system gate for the single-star Space_spheric_adapted_nosym geometry.
//
// WHY THIS FILE EXISTS
// --------------------
// Commit 1e555af9 rewrote the no-symmetry adapted-domain mode counting
// (src/Domain/Spheric_adapted_nosym/, src/Domain/Spheric_nosym/) and silently
// broke the NS_nosym UNIROT (rotating XCTS) solver: its Newton system became
// NON-SQUARE (equations m != unknowns n, a fixed surplus). It went UNDETECTED
// because every pre-existing Space_spheric_adapted_nosym unit test
// ([spheric_adapted_nosym_parity], [_surface_dofs], [_newton_parity]) builds
// only a SCALAR system (sys.add_var("u")). None exercised the VECTOR field
// `bet` (shift) over the full adapted-shell stack — and the real UNIROT stage
// adds exactly that via `space.add_eq(syst, "eqbet^i= 0", "bet^i", "dn(bet^i)")`
// (include/Apps/Formalism/GR/NS_3D_XCTS_nosym/uniform_rot_stage.cpp).
//
// A well-formed spectral domain layout produces a SQUARE Newton system
// (discretized equations == unknown coefficients) at EVERY resolution,
// independent of whether the physics converges. The z-symmetric
// Space_spheric_adapted satisfies this; the no-symmetry
// Space_spheric_adapted_nosym must too. This file pins that invariant for the
// single-star adapted geometry with a VECTOR field set, the path that drifted.
//
// RED-UNTIL-FIXED: the NONSYM cases (both the scalar
// "[spheric_adapted_nosym_square_scalar]" regression case and the vector
// "[spheric_adapted_nosym_square_vector]" case) are EXPECTED TO FAIL at present
// — the src/Domain/* counting is mid-restore (deficit grows with resolution:
// nbr_conditions - nbr_unknowns = -2,-3,-4 at nr=5,7,9). They turn GREEN once
// m == n is restored. The SYM control cases stay GREEN throughout. Do NOT
// weaken any of these to make them pass.
//
// Three independent anchors protect against the "formula matches code" trap
// (where 1e555af9 changed the code AND a duplicated test formula together):
//   1. SYM Space_spheric_adapted is asserted square directly (m == n) — the
//      symmetric counting is the validated reference and was untouched.
//   2. A HARDCODED golden total-DOF count is pinned for the SYM control at each
//      resolution. If the SYM counting itself ever drifts, this trips.
//   3. The NONSYM system is asserted square AND on the SAME geometry as the
//      SYM control, so the expectation cannot silently co-drift with the nosym
//      implementation.

#include <catch2/catch_test_macros.hpp>

#include "For_Kadath/Domain/adapted.hpp"
#include "For_Kadath/Array/array.hpp"
#include "For_Kadath/Base_tensor/base_tensor.hpp"
#include "For_Kadath/Array/dim_array.hpp"
#include "For_Kadath/Array/point.hpp"
#include "For_Kadath/Scalar/scalar.hpp"
#include "For_Kadath/Space/space.hpp"
#include "For_Kadath/Domain/spheric_adapted_nosym.hpp"
#include "For_Kadath/System_of_eqs/system_of_eqs.hpp"
#include "For_Kadath/Tensor/tensor.hpp"
#include "For_Kadath/Tensor/vector.hpp"

using namespace Kadath;

namespace {

Point make_origin()
{
    Point center(3);
    center.set(1) = 0.0;
    center.set(2) = 0.0;
    center.set(3) = 0.0;
    return center;
}

Dim_array make_resolution(int nr)
{
    Dim_array res(3);
    res.set(0) = nr;
    res.set(1) = nr;
    res.set(2) = nr - 1;
    return res;
}

Array<double> make_bounds()
{
    Dim_array dim(1);
    dim.set(0) = 3;
    Array<double> bounds(dim);
    bounds.set(0) = 1.0;
    bounds.set(1) = 2.0;
    bounds.set(2) = 10.0;
    return bounds;
}

// Smooth z-symmetric scalar field (the symmetric subspace is shared by both
// spaces, so a z-symmetric seed is a fair common input).
template <typename SpaceT>
Scalar make_scalar(SpaceT& space)
{
    Scalar u(space);
    for (int d = 0; d < space.get_nbr_domains(); ++d) {
        const Domain* domain = space.get_domain(d);
        const Val_domain x = domain->get_cart(1);
        const Val_domain y = domain->get_cart(2);
        const Val_domain z = domain->get_cart(3);
        u.set_domain(d) = 1.0 + 0.01 * x + 0.02 * y + 0.03 * z * z;
    }
    u.std_base();
    u.coef();
    return u;
}

// Cartesian-basis vector field, z-symmetric, mirroring the construction idiom
// in tests/unit/test_bin_ns_nosym_matching.cpp::make_z_symmetric_vector. This
// is the shift-field analog the UNIROT stage solves for.
template <typename SpaceT>
Vector make_vector(SpaceT& space)
{
    Base_tensor basis(space, CARTESIAN_BASIS);
    Vector v(space, CON, basis);
    for (int d = 0; d < space.get_nbr_domains(); ++d) {
        const Domain* domain = space.get_domain(d);
        const Val_domain x = domain->get_cart(1);
        const Val_domain y = domain->get_cart(2);
        const Val_domain z = domain->get_cart(3);
        v.set(1).set_domain(d) = 0.1 + 0.01 * x + 0.02 * y + 0.03 * z * z;
        v.set(2).set_domain(d) = -0.2 + 0.03 * x - 0.01 * y + 0.02 * z * z;
        v.set(3).set_domain(d) = z * (0.04 + 0.01 * x + 0.02 * y);
    }
    v.std_base();
    v.coef();
    return v;
}

struct SystemDims {
    int nbr_conditions = 0;
    int nbr_unknowns = 0;
};

// HARDCODED golden total-DOF counts for the SYM Space_spheric_adapted control
// at res=(nr,nr,nr-1), bounds=[1,2,10]. Captured at the known-good baseline so
// the SYM expectation cannot silently co-drift with any nosym change. Update
// ONLY if the symmetric adapted counting is intentionally re-derived.
//   scalar  : field eq + outer Dirichlet + adapted-shell surface-DOF closure.
//   vector  : 3-component field + scalar surface-DOF closer (see builder).
int golden_sym_scalar_dofs(int nr)
{
    switch (nr) {
        case 5: return 335;
        case 7: return 994;
        case 9: return 2241;
        default: return -1;
    }
}

int golden_sym_vector_dofs(int nr)
{
    switch (nr) {
        case 5: return 1212;
        case 7: return 3705;
        case 9: return 8494;
        default: return -1;
    }
}

// Scalar field BVP over the full adapted-shell stack: one second-order field
// equation matched across every domain interface, with an outer Dirichlet BC.
// Mirrors the closed scalar system in test_bin_ns_nosym_dof_parity.cpp.
template <typename SpaceT>
SystemDims scalar_system_dims(SpaceT& space)
{
    const int ndom = space.get_nbr_domains();
    System_of_eqs sys(space, 0, ndom - 1);
    Scalar u = make_scalar(space);
    sys.add_var("u", u);
    // Field equation matched across every interface (closes the field
    // coefficients) ...
    space.add_eq(sys, "u = 0", "u", "dn(u)");
    // ... outer Dirichlet (closes the outermost field block) ...
    sys.add_eq_bc(ndom - 1, OUTER_BC, "u=0");
    // ... and a boundary on the adapted shell, which supplies exactly
    // nbr_unknowns_from_variable_domains conditions and closes the
    // surface-shape DOFs. This mirrors the UNIROT stage's
    // `add_eq_bc(1, OUTER_BC, "H = 0")` (uniform_rot_stage.cpp).
    sys.add_eq_bc(1, OUTER_BC, "u=0");
    SystemDims dims;
    dims.nbr_conditions = sys.sec_member().get_nbr();
    dims.nbr_unknowns = sys.get_nbr_unknowns();
    return dims;
}

// Vector (3-component, Cartesian basis) field BVP over the full adapted-shell
// stack — the path 1e555af9 broke. One vector field equation matched across
// every interface plus an outer Dirichlet BC per component.
// Combined scalar + Cartesian-vector field system, modelled on the UNIROT
// stage's (N, P, bet^i) + H closure:
//   - vector field `v`  : 3-component field eq matched across interfaces +
//                         outer Dirichlet on all 3 components. Stresses the
//                         per-mode COLUMN bookkeeping for a vector field — the
//                         3x multiplier on coefficient counts that 1e555af9
//                         desynchronized and that scalar-only tests miss.
//   - scalar field `u`  : field eq + outer Dirichlet, PLUS a single boundary
//                         on the adapted shell that closes the surface-shape
//                         DOFs (exactly nbr_unknowns_from_variable_domains
//                         conditions). One scalar boundary is the correct
//                         multiplicity for the geometry DOFs, mirroring the
//                         single `add_eq_bc(1, OUTER_BC, "H = 0")` in the real
//                         stage (uniform_rot_stage.cpp).
template <typename SpaceT>
SystemDims vector_system_dims(SpaceT& space)
{
    const int ndom = space.get_nbr_domains();
    System_of_eqs sys(space, 0, ndom - 1);
    Vector v = make_vector(space);
    Scalar u = make_scalar(space);
    sys.add_var("v", v);
    sys.add_var("u", u);

    space.add_eq(sys, "v^i = 0", "v^i", "dn(v^i)");
    sys.add_eq_bc(ndom - 1, OUTER_BC, "v^i=0");

    space.add_eq(sys, "u = 0", "u", "dn(u)");
    sys.add_eq_bc(ndom - 1, OUTER_BC, "u=0");
    sys.add_eq_bc(1, OUTER_BC, "u=0"); // closes the surface-shape DOFs

    SystemDims dims;
    dims.nbr_conditions = sys.sec_member().get_nbr();
    dims.nbr_unknowns = sys.get_nbr_unknowns();
    return dims;
}

} // namespace

// Control: the SYM Space_spheric_adapted scalar BVP is exactly square at every
// resolution. Confirms the closed test (full-field add_eq + outer BC) is
// well-posed, so the NONSYM failures below cannot be blamed on the harness.
TEST_CASE("Space_spheric_adapted scalar system is square across resolutions",
          "[spheric_adapted_nosym_square_scalar]")
{
    Point center = make_origin();
    Array<double> bounds = make_bounds();

    for (int nr : {5, 7, 9}) {
        DYNAMIC_SECTION("nr = " << nr)
        {
            Space_spheric_adapted plane(CHEB_TYPE, center, make_resolution(nr), bounds);
            const SystemDims d = scalar_system_dims(plane);
            INFO("Space_spheric_adapted scalar nbr_conditions=" << d.nbr_conditions
                                                                << " nbr_unknowns=" << d.nbr_unknowns
                                                                << " (golden " << golden_sym_scalar_dofs(nr) << ")");
            REQUIRE(d.nbr_conditions == d.nbr_unknowns);
            // Anchor 2: pin the absolute SYM count so the reference cannot drift.
            REQUIRE(d.nbr_unknowns == golden_sym_scalar_dofs(nr));
        }
    }
}

// Regression (SCALAR): the no-symmetry Space_spheric_adapted_nosym scalar BVP
// must be square at every resolution too — and must match its SYM control.
TEST_CASE("Space_spheric_adapted_nosym scalar system is square across resolutions",
          "[spheric_adapted_nosym_square_scalar]")
{
    Point center = make_origin();
    Array<double> bounds = make_bounds();

    for (int nr : {5, 7, 9}) {
        DYNAMIC_SECTION("nr = " << nr)
        {
            Space_spheric_adapted plane(CHEB_TYPE, center, make_resolution(nr), bounds);
            Space_spheric_adapted_nosym nosym(CHEB_TYPE, center, make_resolution(nr), bounds);
            REQUIRE(plane.get_nbr_domains() == nosym.get_nbr_domains());

            const SystemDims sym = scalar_system_dims(plane);
            const SystemDims dims = scalar_system_dims(nosym);
            INFO("Space_spheric_adapted_nosym scalar nbr_conditions=" << dims.nbr_conditions
                                                                      << " nbr_unknowns=" << dims.nbr_unknowns
                                                                      << " (SYM control "
                                                                      << sym.nbr_conditions << "=="
                                                                      << sym.nbr_unknowns << ")");
            // Cross-check: the SYM control is independently square, so a
            // square NONSYM system is required regardless of any duplicated
            // formula.
            REQUIRE(sym.nbr_conditions == sym.nbr_unknowns);
            REQUIRE(dims.nbr_conditions == dims.nbr_unknowns);
        }
    }
}

// Regression (VECTOR) — THE GATE THAT WOULD HAVE CAUGHT 1e555af9.
//
// The UNIROT stage solves for the Cartesian shift vector `bet` over the
// adapted-shell stack. A 3-component vector field triples the per-mode column
// bookkeeping that 1e555af9 desynchronized, so the m != n surplus surfaces here
// where it was invisible to every scalar-only test. Asserts m == n directly and
// against the SYM control on identical geometry.
//
// EXPECTED RED until the src/Domain/* nosym counting is restored.
TEST_CASE("Space_spheric_adapted_nosym vector (shift) system is square across resolutions",
          "[spheric_adapted_nosym_square_vector]")
{
    Point center = make_origin();
    Array<double> bounds = make_bounds();

    for (int nr : {5, 7, 9}) {
        DYNAMIC_SECTION("nr = " << nr)
        {
            Space_spheric_adapted plane(CHEB_TYPE, center, make_resolution(nr), bounds);
            Space_spheric_adapted_nosym nosym(CHEB_TYPE, center, make_resolution(nr), bounds);
            REQUIRE(plane.get_nbr_domains() == nosym.get_nbr_domains());

            const SystemDims sym = vector_system_dims(plane);
            const SystemDims dims = vector_system_dims(nosym);
            INFO("Space_spheric_adapted_nosym vector nbr_conditions=" << dims.nbr_conditions
                                                                      << " nbr_unknowns=" << dims.nbr_unknowns
                                                                      << " surplus=" << (dims.nbr_conditions - dims.nbr_unknowns)
                                                                      << " (SYM control "
                                                                      << sym.nbr_conditions << "=="
                                                                      << sym.nbr_unknowns << ")");
            // The SYM control must be square (sanity on the harness)...
            REQUIRE(sym.nbr_conditions == sym.nbr_unknowns);
            // ...pinned to its golden absolute count (anchor 2)...
            REQUIRE(sym.nbr_unknowns == golden_sym_vector_dofs(nr));
            // ...and so must the NONSYM vector system. This is the assertion
            // 1e555af9 would have tripped.
            REQUIRE(dims.nbr_conditions == dims.nbr_unknowns);
        }
    }
}
