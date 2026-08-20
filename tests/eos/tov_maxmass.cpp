// Standalone TOV M(rho_c) / M-R probe for a cold EOS (Cold_Table or Cold_PWPoly).
//
// Sweeps central rest-mass density rho_c over a log grid, solves the TOV system
// at each point, prints M_ADM(rho_c), baryon mass, areal & isotropic radius, and
// reports the stable-branch maximum mass. Use it to (a) confirm a target mass is
// supported by an EOS and (b) read off the TOV-consistent (rho_c, Mb, R) seed for
// an XCTS neutron-star solve.
//
//   tov_maxmass <eosfile> [interp_pts] [rho_lo] [rho_hi] [N]
//
// eosfile resolves like the solver: bare name -> $HOME_CELEPHAIS/data/eos/<name>
// (or the compile-time CELEPHAIS_DATA_DIR fallback). Radius columns are in km
// (code-unit radius * G Msun / c^2 = * 1.4769994423016508).
#include "Hydro/EOS.hh"
#include "Hydro/Margherita/tov.hh"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>

using namespace Kadath::Margherita;

namespace {
constexpr double kKmPerCodeLength = 1.4769994423016508; // G*Msun/c^2 in km
}

int main(int argc, char** argv) {
    const std::string eosfile = (argc > 1) ? argv[1] : "ZLA_a20_B160_g0.lorene";
    const int interp_pts      = (argc > 2) ? std::atoi(argv[2]) : 2000;

    EOS<Cold_Table, eos_var_t::PRESSURE>::init(eosfile, 0.0, interp_pts);

    const double rho_lo = (argc > 3) ? std::atof(argv[3]) : 5.0e-4;
    const double rho_hi = (argc > 4) ? std::atof(argv[4])
                                     : std::min(2.0e-2, Cold_Table::rhomax * 0.999);
    const int N         = (argc > 5) ? std::atoi(argv[5]) : 240;

    std::printf("# EOS=%s interp_pts=%d rhomin=%.6e rhomax=%.6e\n",
                eosfile.c_str(), interp_pts, Cold_Table::rhomin, Cold_Table::rhomax);
    std::printf("# %-14s %-14s %-14s %-12s %-12s\n",
                "rho_c", "M_ADM[Msun]", "Mb[Msun]", "R_areal[km]", "R_iso[km]");

    MargheritaTOV<Cold_Table> tov;
    double best_m = 0.0, best_rho = 0.0, best_mb = 0.0, best_r = 0.0;
    for (int i = 0; i <= N; ++i) {
        const double rho_c = rho_lo * std::pow(rho_hi / rho_lo, double(i) / N); // log sweep
        tov.solve(rho_c, /*finish_isotropic_fields=*/true);
        const double r_areal = tov.arealr * kKmPerCodeLength;
        const double r_iso   = tov.radius * kKmPerCodeLength;
        std::printf("%14.6e %14.8f %14.8f %12.5f %12.5f\n",
                    rho_c, tov.mass, tov.baryon_mass, r_areal, r_iso);
        if (tov.mass > best_m) {
            best_m = tov.mass; best_rho = rho_c; best_mb = tov.baryon_mass; best_r = r_areal;
        }
    }
    std::printf("# MAX_MASS M_ADM=%.6f Mb=%.6f rho_c=%.6e R_areal=%.5f km\n",
                best_m, best_mb, best_rho, best_r);
    return 0;
}
