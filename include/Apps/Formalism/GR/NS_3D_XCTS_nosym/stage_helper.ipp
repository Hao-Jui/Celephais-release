// ns_3d_xcts_nosym solver stage scaffolding plus per-stage aggregation. Holds
// the nosym-specific shared helpers (save_converged_solution, add_* equation
// registration); run_newton_loop and add_mass_fixing are inherited from the
// XCTS_Solver base. #includes the individual stage bodies at the end. Body
// fragment included by solver.hpp; not a standalone
// translation unit.
#include "mpi.h"
#include <algorithm>
#include <cmath>
#include <string>
#include "Apps/Bco_utils/bco_io.hpp"

namespace Kadath {


namespace {
double ns_nosym_forwarded_residual_infinity_norm(const Kadath::System_of_eqs& syst)
{
    double error = 0.;
    if (!syst.forwarded_residual_infinity_norm(error))
        KADATH_THROW("NS_nosym post-update residual was not forwarded");
    return error;
}
} // namespace

template <class eos_t, typename config_t, typename space_t>
void ns_3d_xcts_nosym_solver<eos_t, config_t, space_t>::save_converged_solution(const std::string& stage)
{
    Scalar phi_zero(space);
    phi_zero.annule_hard();
    phi_zero.std_base();
    save_converged_solution(stage, phi_zero);
}

template <class eos_t, typename config_t, typename space_t>
void ns_3d_xcts_nosym_solver<eos_t, config_t, space_t>::save_converged_solution(const std::string& stage,
                                                                                const Scalar& phi)
{
    bconfig.set_filename(converged_filename(stage));
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0)
        bco_utils::save_to_file(space, bconfig, conf, lapse, shift, logh, phi);
}

template <class eos_t, typename config_t, typename space_t>
void ns_3d_xcts_nosym_solver<eos_t, config_t, space_t>::add_spin_constants(System_of_eqs& syst)
{
    syst.add_cst("chi", bconfig(CHI));
    const double spin_axis_angle = bconfig(DEG) * std::acos(-1.) / 180.;
    syst.add_cst("angs", spin_axis_angle);
    syst.add_var("ome", bconfig(OMEGA));
}

template <class eos_t, typename config_t, typename space_t>
void ns_3d_xcts_nosym_solver<eos_t, config_t, space_t>::add_extrinsic_curvature(System_of_eqs& syst)
{
    syst.add_def("A^ij = (D^i bet^j + D^j bet^i - 2. / 3.* D_k bet^k * f^ij) / 2. / Ntilde");
    syst.add_def(ndom - 1, "intJ = multr(A_ij * mm^j * einf^i) / 2. / 4piG");
}

template <class eos_t, typename config_t, typename space_t>
void ns_3d_xcts_nosym_solver<eos_t, config_t, space_t>::add_constraint_equations(System_of_eqs& syst, int d)
{
    if (is_vacuum_domain(d)) {
        // Vacuum: no matter sources.
        syst.add_def(d, "eqP = D^i D_i P + A_ij * A^ij / P^7 / 8");
        syst.add_def(d, "eqNP = D^i D_i NP - 7. / 8. * NP / P^8 * A_ij * A^ij");
        syst.add_def(d, "eqbet^i = D_j D^j bet^i + D^i D_j bet^j / 3. - 2. * A^ij * D_j Ntilde");
    } else {
        // Matter: Etilde/Stilde/ptilde must already be defined for this domain.
        syst.add_def(d, "eqP    = delta * D^i D_i P + A_ij * A^ij / P^7 / 8 * delta + 4piG / 2. * P^5 * Etilde");
        syst.add_def(d, "eqNP   = delta * D^i D_i NP - 7. / 8. * NP / P^8 * delta * A_ij *A^ij "
                        "- 4piG / 2. * N * P^5 * (Etilde + 2. * Stilde)");
        syst.add_def(d, "eqbet^i= delta * D_j D^j bet^i + delta * D^i D_j bet^j / 3. "
                        "- 2. * delta * A^ij * D_j Ntilde - 4. * 4piG * N * P^4 * ptilde^i");
    }
}

template <class eos_t, typename config_t, typename space_t>
void ns_3d_xcts_nosym_solver<eos_t, config_t, space_t>::add_conformal_field_bcs(System_of_eqs& syst)
{
    space.add_eq(syst, "eqNP= 0", "N", "dn(N)");
    space.add_eq(syst, "eqP= 0", "P", "dn(P)");
    space.add_eq(syst, "eqbet^i= 0", "bet^i", "dn(bet^i)");

    syst.add_eq_bc(ndom - 1, OUTER_BC, "N=1");
    syst.add_eq_bc(ndom - 1, OUTER_BC, "P=1");
    syst.add_eq_bc(ndom - 1, OUTER_BC, "bet^i=0");
}

// Individual stage bodies (each a body fragment, not a standalone TU).
#include "norot_stage.ipp"        // norot_stage (static TOV)
#include "uniform_rot_stage.ipp"  // uniform_rot_stage (UNIROT)
#include "binary_boost_stage.ipp" // binary_boost_stage (BIN_BOOST)

} // namespace Kadath
