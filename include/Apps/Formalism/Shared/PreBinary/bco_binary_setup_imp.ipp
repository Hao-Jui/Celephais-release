#include "Apps/Helper/solver_base.hpp"
#include "For_Kadath/Kadath_point_h/kadath_adapted.hpp"
#include "For_Kadath/Array/exceptions.hpp"
#include "mpi.h"
#include "bco_binary_setup.hpp"
#include "Apps/Policy/app_resolution.hpp"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>
#include <utility>

namespace Kadath {

namespace {

// Copy BCO, gravity, control, and sequence settings from a binary config slot → isolated config.
template <typename isolated_t, typename binary_t>
void copy_params_from_binary(isolated_t& dst, binary_t& src, NODES bco)
{
    for (int i = 0; i < NUM_BCO_PARAMS_V;    ++i) dst.set(i)          = src.set(i, bco);
    for (int i = 0; i < NUM_GRAVITY_PARAMS_V; ++i) dst.set_gravity(i) = src.gravity_setting(i);
    for (int i = 0; i < NUM_CONTROLS_V;       ++i) dst.control(i)     = src.control(i);
    for (int i = 0; i < NUM_SEQ_SETTINGS_V;   ++i) dst.seq_setting(i) = src.seq_setting(i);
}

// Copy BCO results from an isolated config → binary config slot.
template <typename binary_t, typename isolated_t>
void copy_params_to_binary(binary_t& dst, NODES bco, isolated_t& src)
{
    for (int i = 0; i < NUM_BCO_PARAMS_V; ++i) dst.set(i, bco) = src.set(i);
}

inline void mpi_barrier_if_initialized()
{
    int initialized = 0;
    MPI_Initialized(&initialized);
    if (initialized)
        MPI_Barrier(MPI_COMM_WORLD);
}

} // anonymous namespace

template <typename config_t, typename NsSeedSolve>
std::string solve_NS_from_binary(config_t& bconfig, NODES bco, NsSeedSolve&& ns_seed_solve)
{
    std::string output_path = get_cos_path();
    fs::create_directories(output_path);

    kadath_config<BCO_NS_INFO> nsconfig;
    nsconfig.set_defaults();

    copy_params_from_binary(nsconfig, bconfig, bco);
    for (int i = 0; i < NUM_EOS_PARAMS_V; ++i)
        nsconfig.set_eos(i) = bconfig.set_eos(i, bco);

    // The binary ladder owns [binary].initial_resolution. Component seeds always
    // climb from kDefaultInitialResolution to their own [binary.ns*].res target,
    // so that target must be >= kDefaultInitialResolution.
    nsconfig.seq_setting(INIT_RES) = Kadath::kDefaultInitialResolution;
    nsconfig.set_outputdir(output_path);

    const int ninshells = bconfig(NINSHELLS, bco);
    const int nshells   = bconfig(NSHELLS,   bco);
    nsconfig(NINSHELLS) = 0;
    nsconfig(NSHELLS)   = 0;

    std::forward<NsSeedSolve>(ns_seed_solve)(nsconfig, output_path, bconfig, bco);
    mpi_barrier_if_initialized();

    copy_params_to_binary(bconfig, bco, nsconfig);
    bconfig.set(NINSHELLS, bco) = ninshells;
    bconfig.set(NSHELLS,   bco) = nshells;

    return nsconfig.config_outputdir() + nsconfig.config_filename();
}

// Solve an isolated BH seed for binary component `bco`. As with the NS helper,
// the theory-specific driver call (and BIN_BOOST enablement) is supplied by
// `bh_seed_solve`, invoked as bh_seed_solve(bhconfig, output_path, bconfig, bco).
template <typename config_t, typename BhSeedSolve>
std::string solve_BH_from_binary(config_t& bconfig, NODES bco, BhSeedSolve&& bh_seed_solve)
{
    std::string output_path = get_cos_path();
    fs::create_directories(output_path);

    kadath_config<BCO_BH_INFO> bhconfig;
    bhconfig.set_defaults();

    copy_params_from_binary(bhconfig, bconfig, bco);
    bhconfig.seq_setting(INIT_RES) = Kadath::kDefaultInitialResolution;
    // Do NOT pin the seed filename here (mirror solve_NS_from_binary). Pinning
    // "initbh" made the driver's "does a seed already exist?" gate resolve to the
    // bare, mass-agnostic data/convergence/initbh.dat, so a leftover scratch seed
    // from a prior binary run (any mass) was reused instead of rebuilt — the BH
    // grid then kept the old mass while the constraint targeted the new MCH. With
    // the default (empty) name the gate misses the stale file and the driver
    // rebuilds the seed from the current MCH, then names it "initbh" internally.
    bhconfig.set_outputdir(output_path);

    const int nshells = bconfig(NSHELLS, bco);
    bhconfig(NSHELLS) = 0;

    std::forward<BhSeedSolve>(bh_seed_solve)(bhconfig, output_path, bconfig, bco);
    mpi_barrier_if_initialized();

    copy_params_to_binary(bconfig, bco, bhconfig);
    bconfig.set(NSHELLS, bco) = nshells;

    mpi_barrier_if_initialized();
    return bhconfig.config_outputdir() + bhconfig.config_filename();
}

// Eggleton (1983) Roche-lobe radius as a fraction R_L/a of the orbital
// separation, with q = M_this / M_companion. Accurate to ~1% over all q.
inline double eggleton_roche_fraction(double q)
{
    const double q23 = std::cbrt(q * q);
    return 0.49 * q23 / (0.6 * q23 + std::log1p(std::cbrt(q)));
}

inline void check_dist(double dist, BinaryComponent c1, BinaryComponent c2,
                       double overlap_safety)
{
    const double total_mass = c1.mass + c2.mass;
    const bool   have_radii = (c1.radius > 0. && c2.radius > 0.);

    // Roche / mass-shedding onset: a_shed = R / (R_L/a). A star sheds when it
    // fills its Roche lobe; take the larger onset (the star that disrupts first).
    double a_shed = 0.;
    if (c1.deformable && c1.radius > 0. && c2.mass > 0.)
        a_shed = std::max(a_shed, c1.radius / eggleton_roche_fraction(c1.mass / c2.mass));
    if (c2.deformable && c2.radius > 0. && c1.mass > 0.)
        a_shed = std::max(a_shed, c2.radius / eggleton_roche_fraction(c2.mass / c1.mass));

    // (1) HARD floor: the adapted domains / excision spheres must not overlap.
    //     Radius-based (EOS-aware) when available, else the mass-scale fallback.
    const double overlap_floor = have_radii ? overlap_safety * (c1.radius + c2.radius)
                                            : 2.5 * total_mass;
    if (dist <= overlap_floor) {
        const double recommend = std::max(8. * total_mass, (a_shed > 0. ? 1.5 * a_shed : 0.));
        std::cerr << "Separation " << dist << " <= overlap floor " << overlap_floor
                  << (have_radii ? " (safety x (R1+R2))" : " (2.5 x total mass; radii unset)")
                  << ". Components would interpenetrate; use >= " << recommend
                  << " for a reasonable result.\n";
        KADATH_THROW("Stage failed");
    }

    // (2) SOFT floor: warn (never throw) inside the mass-shedding regime.
    if (a_shed > 0. && dist < a_shed) {
        std::cerr << "[warn] Separation " << dist << " is below the mass-shedding (Roche) "
                  << "onset " << a_shed << ". Configuration is physical but near the "
                     "sequence endpoint; expect slower / harder convergence.\n";
    }
}

inline void check_dist(double dist, double M1, double M2, double garbage_factor)
{
    const double total_mass  = M1 + M2;
    const double garbage_dist = garbage_factor * total_mass;
    if (dist <= garbage_dist) {
        std::cerr << "Separation " << dist << " <= garbage threshold " << garbage_dist
                  << ". Use >= " << 8. * total_mass << " for a reasonable result.\n";
        KADATH_THROW("Stage failed");
    }
}


} // namespace Kadath
