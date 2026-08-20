// Matrix-free Jacobian-vector product parity gate for System_of_eqs::do_JX.
//
// CONTRACT UNDER TEST
//   For any unknown perturbation x, the matrix-free product do_JX(x) must equal
//   J . x exactly (to round-off), where J is the Jacobian assembled column by
//   column with do_col_J. Both routines share the same forward-mode seeding
//   (xx_to_ders / JacobianColumnEngine::compute_column seed term derivatives
//   with the entries of x / a unit basis vector e_cc), so this is the exact
//   linearity identity
//       do_JX(x) == sum_cc x[cc] * do_col_J(cc).
//
// WHY THIS TEST EXISTS
//   The seeding loop in System_of_eqs::xx_to_ders walks every domain slot of
//   each numeric (double) unknown:
//       for (i in nvar_double)
//           for (dd in [dom_min, dom_max]) term_double[pos++]->set_der_d(x[conte]);
//   An off-by-one in that domain range (dd < dom_max instead of dd <= dom_max)
//   leaves the LAST domain's slot of every double unknown unseeded and shifts
//   later variables' seeds into earlier variables' high-domain slots. do_col_J
//   seeds all domain slots of a double via its own loop, so the column-built
//   Jacobian stays correct while do_JX silently drops the contribution of any
//   double referenced on the last domain. No existing test covers this: the
//   BH2d smoke parity gate registers a single numeric unknown that is not read
//   on the last domain, so its Jv contribution is zero either way.
//
//   This test registers TWO numeric unknowns, one of which (b) is referenced by
//   a boundary equation evaluated on dom_max, and asserts do_JX == J.x. With the
//   bug present, b's column is dropped from do_JX but retained in the
//   column-built J, so the parity residual is O(1) and the test fails. On the
//   fixed code the residual is at round-off.
//
//   nproc == 1 (this is a plain unit binary, no MPI launch), so do_JX takes the
//   fully replicated path: DO_JX_MPI's row partition only activates for
//   nproc > 1, hence xx_to_ders is invoked unfiltered here -- exactly the seeding
//   code the bug lived in.
//
// COLUMN ORDERING (verified against the seeding code, src/System_of_eqs/assembly)
//   The Jacobian column index cc indexes the entries of x in this order:
//     1. variable / adapted-domain unknowns  (espace.xx_to_ders_variable_domains)
//     2. the nvar_double numeric unknowns    (one column each, add order)
//     3. the field / tensor coefficient unknowns
//   A plain Space_spheric (not adapted) contributes zero variable-domain
//   unknowns, so the two numeric unknowns occupy columns [0, nvar_double) -- the
//   first columns. The field columns follow.

#include <catch2/catch_test_macros.hpp>

#include "For_Kadath/Array/array.hpp"
#include "For_Kadath/Array/dim_array.hpp"
#include "For_Kadath/Array/point.hpp"
#include "For_Kadath/Scalar/scalar.hpp"
#include "For_Kadath/Domain/spheric.hpp"
#include "For_Kadath/System_of_eqs/system_of_eqs.hpp"
#include "mpi.h"

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <vector>

using namespace Kadath;

namespace {

void ensure_mpi_initialized()
{
    static const bool initialized_for_tests = []() {
        int finalized = 0;
        if (MPI_Finalized(&finalized) != MPI_SUCCESS || finalized != 0)
            throw std::runtime_error("MPI is already finalized");

        int initialized = 0;
        if (MPI_Initialized(&initialized) != MPI_SUCCESS)
            throw std::runtime_error("MPI_Initialized failed");
        if (initialized == 0) {
            int argc = 0;
            char** argv = nullptr;
            if (MPI_Init(&argc, &argv) != MPI_SUCCESS)
                throw std::runtime_error("MPI_Init failed");
            std::atexit([]() {
                int mpi_finalized = 0;
                if (MPI_Finalized(&mpi_finalized) == MPI_SUCCESS && mpi_finalized == 0)
                    MPI_Finalize();
            });
        }
        return true;
    }();
    (void)initialized_for_tests;
}

// Minimal multi-domain spheric space (nucleus + 2 shells = 3 domains), low
// resolution, matching the make_minimal_space pattern of the double-der seeding
// scaffold. withzec=false suppresses the compactified outer domain so every
// domain has a direct nucleus/shell dispatch.
Space_spheric make_minimal_space()
{
    Point center(3);
    center.set(1) = 0.0;
    center.set(2) = 0.0;
    center.set(3) = 0.0;

    Dim_array res(3);
    res.set(0) = 5;
    res.set(1) = 5;
    res.set(2) = 4;

    Dim_array bounds_dim(1);
    bounds_dim.set(0) = 3;            // 3 boundaries -> nucleus + 2 shells = 3 domains
    Array<double> bounds(bounds_dim);
    bounds.set(0) = 1.0;
    bounds.set(1) = 2.0;
    bounds.set(2) = 10.0;

    return Space_spheric(CHEB_TYPE, center, res, bounds, false);
}

// Deterministic pseudo-random perturbation: no RNG, fully reproducible.
// x[i] = sin(0.7*i + 1.3) spans both signs and is O(1) for every entry.
Array<double> make_deterministic_perturbation(int n)
{
    Array<double> xx(n);
    for (int i = 0; i < n; ++i) {
        xx.set(i) = std::sin(0.7 * static_cast<double>(i) + 1.3);
    }
    return xx;
}

double max_abs(const Array<double>& v)
{
    double m = 0.0;
    const int n = v.get_nbr();
    for (int i = 0; i < n; ++i) {
        const double a = std::abs(v.get_data()[i]);
        if (a > m) m = a;
    }
    return m;
}

// Assemble the dense Jacobian column by column and return it as a flat
// row-major m x n matrix (J[r*n + c]).
std::vector<double> assemble_dense_jacobian(System_of_eqs& sys, int m, int n)
{
    std::vector<double> jac(static_cast<std::size_t>(m) * static_cast<std::size_t>(n), 0.0);
    for (int c = 0; c < n; ++c) {
        Array<double> col = sys.do_col_J(c);
        REQUIRE(col.get_nbr() == m);
        for (int r = 0; r < m; ++r) {
            jac[static_cast<std::size_t>(r) * static_cast<std::size_t>(n) +
                static_cast<std::size_t>(c)] = col.get_data()[r];
        }
    }
    return jac;
}

// Compare do_JX(x) against the dense product J.x and return the measured
// infinity-norm parity residual. Asserts the residual is at round-off scale.
double check_jx_parity(System_of_eqs& sys)
{
    const int n = sys.get_nbr_unknowns();
    const int m = sys.get_nbr_conditions();
    REQUIRE(n > 0);
    REQUIRE(m > 0);

    const std::vector<double> jac = assemble_dense_jacobian(sys, m, n);

    const Array<double> xx = make_deterministic_perturbation(n);

    // Dense reference product Jx = J . x.
    std::vector<double> jx(static_cast<std::size_t>(m), 0.0);
    double jx_scale = 0.0;
    for (int r = 0; r < m; ++r) {
        double acc = 0.0;
        for (int c = 0; c < n; ++c) {
            acc += jac[static_cast<std::size_t>(r) * static_cast<std::size_t>(n) +
                       static_cast<std::size_t>(c)] * xx(c);
        }
        jx[static_cast<std::size_t>(r)] = acc;
        if (std::abs(acc) > jx_scale) jx_scale = std::abs(acc);
    }

    // Matrix-free product. The write-into overload must retain its caller-owned
    // storage and produce the same values as the compatibility return-by-value
    // wrapper used by existing call sites.
    Array<double> jv(m);
    double* const jv_storage = jv.set_data();
    sys.do_JX(xx, jv);
    REQUIRE(jv.set_data() == jv_storage);
    REQUIRE(jv.get_nbr() == m);
    const Array<double> returned_jv = sys.do_JX(xx);
    REQUIRE(returned_jv.get_nbr() == m);
    for (int r = 0; r < m; ++r) {
        REQUIRE(jv.get_data()[r] == returned_jv.get_data()[r]);
    }

    // A mismatched output cannot be written without violating the fixed-layout
    // contract; reject it before any Jacobian work or collective is entered.
    Array<double> wrong_size(m + 1);
    REQUIRE_THROWS(sys.do_JX(xx, wrong_size));

    double parity_residual = 0.0;
    for (int r = 0; r < m; ++r) {
        const double d = std::abs(jx[static_cast<std::size_t>(r)] - jv.get_data()[r]);
        if (d > parity_residual) parity_residual = d;
    }

    INFO("nbr_unknowns = " << n << "  nbr_conditions = " << m);
    INFO("max|Jx|          = " << jx_scale);
    INFO("max|Jx - do_JX|  = " << parity_residual);

    // Round-off scale: relative to the magnitude of the dense product. The
    // measured residual on the fixed code is ~3.6e-15 at jx_scale ~17.7
    // (machine epsilon times the product magnitude), so the 1e-11 relative
    // bound has > 4 orders of headroom. With the seeding off-by-one present,
    // the last-domain double's column is dropped from do_JX and the residual
    // is O(1) -- this assertion fails hard.
    REQUIRE(parity_residual <= 1e-11 * (1.0 + jx_scale));
    return parity_residual;
}

} // namespace

// Primary gate: field unknown + two numeric unknowns, one referenced on the
// LAST domain (the exact seeding-bug-exposing condition).
TEST_CASE("do_JX equals J.x for a field plus double-variable system referenced on the last domain",
          "[do_jx][system_of_eqs][jacobian]")
{
    Space_spheric space = make_minimal_space();
    const int ndom = space.get_nbr_domains();
    REQUIRE(ndom >= 3); // need a multi-domain range to exercise dom_max seeding

    const int dom_max = ndom - 1;

    System_of_eqs sys(space, 0, dom_max);

    // Scalar field unknown over the whole domain range.
    Scalar u(space);
    for (int d = 0; d < ndom; ++d) {
        u.set_domain(d) = 0.3 + 0.1 * static_cast<double>(d);
    }
    u.std_base();
    u.coef();
    sys.add_var("u", u);

    // Two numeric unknowns. `a` sources an interior equation on domain 0;
    // `b` sources the OUTER boundary equation on the LAST domain -- the slot the
    // off-by-one left unseeded in do_JX.
    double a = 0.5;
    double b = 0.25;
    sys.add_var("a", a);
    sys.add_var("b", b);

    // Equation set chosen so sec_member / do_col_J run cleanly and the system is
    // square. Mirrors the matching/order/bc structure of the existing nosym
    // newton-parity tests, with the algebraic RHS made double-dependent:
    //   - interior algebraic equation on the nucleus sourced by `a`,
    //   - a 2nd-order regularity equation on each shell,
    //   - an OUTER boundary equation on the last domain sourced by `b`,
    //   - matching across every internal interface.
    sys.add_eq_inside(0, "u = a");
    for (int d = 1; d < ndom; ++d) {
        sys.add_eq_order(d, 2, "u = 0");
    }
    sys.add_eq_bc(dom_max, OUTER_BC, "u = b");
    for (int d = 0; d < dom_max; ++d) {
        sys.add_eq_matching(d, OUTER_BC, "u");
    }

    // Realize nbr_conditions / nbr_unknowns and seed terms.
    Array<double> residual = sys.sec_member();
    REQUIRE(sys.get_nbr_conditions() > 0);
    REQUIRE(residual.get_nbr() == sys.get_nbr_conditions());

    const int n = sys.get_nbr_unknowns();
    const int m = sys.get_nbr_conditions();

    // Column layout: plain (non-adapted) Space_spheric has no variable-domain
    // unknowns, so the two doubles occupy columns 0 and 1 (add order: a, b).
    // The field columns follow. Guard the assumption: there must be at least two
    // more columns than the two doubles (the field DOFs).
    REQUIRE(n >= 3);
    const int col_a = 0; // numeric unknown `a`
    const int col_b = 1; // numeric unknown `b`, referenced on dom_max

    // Non-vacuity on the buggy axis: the double columns must be genuinely
    // nonzero, proving an equation really reads each double's derivative. In
    // particular col_b (referenced ONLY on the last domain) is the column the
    // seeding off-by-one would silently zero out of do_JX.
    const Array<double> column_a = sys.do_col_J(col_a);
    const Array<double> column_b = sys.do_col_J(col_b);
    INFO("max|do_col_J(a)| = " << max_abs(column_a));
    INFO("max|do_col_J(b)| = " << max_abs(column_b));
    REQUIRE(max_abs(column_a) > 1e-12);
    REQUIRE(max_abs(column_b) > 1e-12);

    // Core contract: do_JX(x) == J . x for a deterministic pseudo-random x.
    const double parity_residual = check_jx_parity(sys);

    // Reported for the run log (see VERIFY in the task): the measured residual.
    INFO("measured max|Jx - do_JX| (field + doubles) = " << parity_residual);
    SUCCEED("do_JX matched the column-built Jacobian product to round-off");
}

// Control gate: pure field system, no numeric unknowns. Confirms the parity
// harness itself is sound on the path that the BH2d smoke gate already exercises
// (so a failure of the primary case localizes to the double-seeding axis, not to
// the harness).
TEST_CASE("do_JX equals J.x for a pure field system (no double variables)",
          "[do_jx][system_of_eqs][jacobian]")
{
    Space_spheric space = make_minimal_space();
    const int ndom = space.get_nbr_domains();
    REQUIRE(ndom >= 3);

    const int dom_max = ndom - 1;

    System_of_eqs sys(space, 0, dom_max);

    Scalar u(space);
    for (int d = 0; d < ndom; ++d) {
        u.set_domain(d) = 0.3 + 0.1 * static_cast<double>(d);
    }
    u.std_base();
    u.coef();
    sys.add_var("u", u);

    sys.add_eq_inside(0, "u = 0");
    for (int d = 1; d < ndom; ++d) {
        sys.add_eq_order(d, 2, "u = 0");
    }
    sys.add_eq_bc(dom_max, OUTER_BC, "u = 0");
    for (int d = 0; d < dom_max; ++d) {
        sys.add_eq_matching(d, OUTER_BC, "u");
    }

    Array<double> residual = sys.sec_member();
    REQUIRE(sys.get_nbr_conditions() > 0);
    REQUIRE(residual.get_nbr() == sys.get_nbr_conditions());

    const double parity_residual = check_jx_parity(sys);
    INFO("measured max|Jx - do_JX| (pure field) = " << parity_residual);
    SUCCEED("do_JX matched the column-built Jacobian product to round-off");
}

TEST_CASE("partitioned residual reuses MPI workspaces for integral parts and field rows",
          "[system_of_eqs][partitioned-residual][mpi]")
{
    ensure_mpi_initialized();

    int nproc = 0;
    REQUIRE(MPI_Comm_size(MPI_COMM_WORLD, &nproc) == MPI_SUCCESS);
    if (nproc < 2)
        SKIP("requires at least two MPI ranks");

    Space_spheric space = make_minimal_space();
    const int ndom = space.get_nbr_domains();
    const int dom_max = ndom - 1;
    REQUIRE(ndom == 3);

    System_of_eqs sys(space, 0, dom_max);
    Scalar u(space);
    for (int d = 0; d < ndom; ++d)
        u.set_domain(d) = 0.3 + 0.1 * static_cast<double>(d);
    u.std_base();
    u.coef();
    sys.add_var("u", u);

    double volume_target = 0.125;
    sys.add_var("voltarget", volume_target);

    sys.add_eq_inside(0, "u = 0");
    for (int d = 1; d < ndom; ++d)
        sys.add_eq_order(d, 2, "u = 0");
    sys.add_eq_bc(dom_max, OUTER_BC, "u = 0");
    for (int d = 0; d < dom_max; ++d)
        sys.add_eq_matching(d, OUTER_BC, "u");
    space.add_eq_int_volume(sys, "integvolume(u) = voltarget");

    const Array<double> reference = sys.sec_member();
    REQUIRE(reference.get_nbr() == sys.get_nbr_conditions());

    // Build, self-test, and reverify the row partition before exercising the
    // value tail. Both calls are collective and deterministic across ranks.
    const Array<double> perturbation =
        make_deterministic_perturbation(sys.get_nbr_unknowns());
    const Array<double> initial_jv = sys.do_JX(perturbation);
    const Array<double> steady_jv = sys.do_JX(perturbation);
    REQUIRE(initial_jv.get_nbr() == reference.get_nbr());
    REQUIRE(steady_jv.get_nbr() == reference.get_nbr());

    // First call runs the partitioned-value parity gate; the second exercises
    // the steady-state path with the same topology-sized gather/cursor storage.
    const Array<double> first = sys.sec_member_partitioned();
    const Array<double> steady = sys.sec_member_partitioned();
    REQUIRE(first.get_nbr() == reference.get_nbr());
    REQUIRE(steady.get_nbr() == reference.get_nbr());
    REQUIRE(std::memcmp(first.get_data(), reference.get_data(),
                        sizeof(double) * reference.get_nbr()) == 0);
    REQUIRE(std::memcmp(steady.get_data(), reference.get_data(),
                        sizeof(double) * reference.get_nbr()) == 0);
}
