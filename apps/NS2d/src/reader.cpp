#include "For_Kadath/Kadath_point_h/kadath_adapted_polar.hpp"
#include "For_Kadath/Kadath_point_h/kadath_adapted.hpp"
#include "For_Kadath/Config/config_bco.hpp"
#include "For_Kadath/Utilities/Exporters/bco_geometry.hpp"
#include "Hydro/EOS.hh"
#include "Apps/Diagnostics/configured_eos.hpp"
#include "For_Kadath/Utilities/Exporters/coord_fields.hpp"
#include "For_Kadath/IO/be_file_source.hpp"
#include "For_Kadath/IO/be_file_sink.hpp"
#include "mpi.h"
#include "Apps/Startup/solver_startup.hpp"

using namespace Kadath;

template<class eos_t, typename config_t>
void reader_2d(config_t bconfig);

constexpr double M2km = 1.4769994423016508;
constexpr double M2Hz = 2.029739818539300e+05 / 2. / M_PI; 

int main(int argc, char **argv) {
  KadathApps::init_mpi(argc, argv);

  return KadathApps::guarded_run([&] {
  if(argc < 2) {
    KADATH_THROW("Usage: ./reader /<path>/<ID base name>.toml|.dat  e.g. ./reader converged.NS.9.toml");
  }

  const std::string ifilename = KadathApps::toml_config_path_from_reader_input(argv[1]);
  kadath_config<BCO_NS_INFO> bconfig{ifilename};

  const double h_cut = bconfig.eos<double>(HCUT);
  const std::string eos_file = bconfig.eos<std::string>(EOSFILE);
  const std::string eos_type = bconfig.eos<std::string>(EOSTYPE);

  if(eos_type == "Cold_PWPoly") {
    using eos_t = Kadath::Margherita::Cold_PWPoly;
    EOS<eos_t, eos_var_t::PRESSURE>::init(eos_file, h_cut);
    
    if(bconfig(DIM) == 2) reader_2d<eos_t>(bconfig);
  } else if(eos_type == "Cold_Table") {
    using eos_t = Kadath::Margherita::Cold_Table;
    KadathApps::init_configured_cold_table(bconfig);
    
    if(bconfig(DIM) == 2) reader_2d<eos_t>(bconfig);
  } else {
    KADATH_THROW("Unknown EOSTYPE.");
  }
  MPI_Finalize();
  });
}

template<class eos_t, typename config_t>
void reader_2d(config_t bconfig) {
  const std::string spacein = bconfig.space_filename();
  
  BeFileSource ff1(spacein);
  Space_polar_adapted space(ff1);
  Scalar conf(space, ff1);
  Scalar lapse(space, ff1);
  Scalar shift(space, ff1);
  Scalar metQ(space, ff1);
  Scalar logh(space, ff1);
  Scalar Omg(space, ff1);

  int ndom = space.get_nbr_domains();
  const int adapted_outer = space.ADAPTED_OUTER;

  double loghc = bco_utils::get_boundary_val(0, logh, INNER_BC);
  double hc = std::exp(loghc);
  double nc = EOS<eos_t, eos_var_t::DENSITY>::get(hc);
  double pc = EOS<eos_t, eos_var_t::PRESSURE>::get(hc);

  auto [rmin, rmax] = bco_utils::get_rmin_rmax(space, adapted_outer);
  double rin1 = bco_utils::get_radius(space.get_domain(0), OUTER_BC);

  System_of_eqs syst(space, 0, ndom - 1);

  syst.add_cst("4piG", 4.0 * M_PI);
  syst.add_cst("ome", bconfig(OMEGA));
  syst.add_cst("cstA", bconfig(CSTA));
  syst.add_cst("P", conf);
  syst.add_cst("N", lapse);
  syst.add_cst("brsint", shift);
  syst.add_cst("Q", metQ);
  syst.add_cst("Omg", Omg);
  syst.add_cst("H", logh);

  syst.add_def("bet = divrsint(brsint)");

  if(std::isnan(bconfig.set(BVELY))) bconfig.set(BVELY) = 0.;
  
  syst.add_def(ndom - 1, "intJ = multrsint(multrsint(dr(bet))) / 4 / 4piG");
  syst.add_def(ndom - 1, "intMadm = -dr(P) * 2 / 4piG");
  syst.add_def(ndom - 1, "intMk = dr(N)  / 4piG");
  syst.add_def("metrho = log(N/P)");

  Param p;
  syst.add_ope("eps", &EOS<eos_t, eos_var_t::EPSILON>::action, &p);
  syst.add_ope("pres", &EOS<eos_t, eos_var_t::PRESSURE>::action, &p);
  syst.add_ope("rho", &EOS<eos_t, eos_var_t::DENSITY>::action, &p);

  for(int d = 0; d < ndom; d++) {
    if (d <= adapted_outer) {
      syst.add_def(d, "h = exp(H)");
      syst.add_def(d, "U = multrsint(P^2 / N * (Omg + bet))"); // amplitude of spatial velocity
      syst.add_def(d, "Usq = U*U");
      syst.add_def(d, "Wsquare = 1 / (1 - Usq)");
      syst.add_def(d, "W = sqrt(Wsquare)");
      syst.add_def(d, "intMb = W * rho(h) * exp(2 * Q) * P^6 * 4piG / 2");
      syst.add_def(d, "firstint = H + log(N) - log(W) - cstA^2 * (Omg-ome)^2 / 2.");
    }
  }
  Val_domain integJ    (syst.give_val_def("intJ"   )()(ndom - 1));
  Val_domain integMadm (syst.give_val_def("intMadm")()(ndom - 1));
  Val_domain integMk   (syst.give_val_def("intMk"  )()(ndom - 1));

  double J    = space.get_domain(ndom - 1)->integ(integJ   , OUTER_BC);
  double Madm = space.get_domain(ndom - 1)->integ(integMadm, OUTER_BC);
  double Mk   = space.get_domain(ndom - 1)->integ(integMk  , OUTER_BC);
  double Mbar = 0.0;
  for (int d = 0; d <= adapted_outer; ++d) {
    Mbar += syst.give_val_def("intMb")()(d).integ_volume();
  }
  Point P(2);
  double central_euler = syst.give_val_def("firstint")().val_point(P);
  double central_omg = Omg.val_point(P);

  auto this_domain = Omg(adapted_outer).get_domain();
  Index pos(this_domain->get_nbr_points());
  pos.set(0) = this_domain->get_nbr_points()(0) - 1;
  pos.set(1) = this_domain->get_nbr_points()(1) - 1;
  double equator_omg = Omg(adapted_outer)(pos);
  double psi_eq = conf(adapted_outer)(pos);

  double AR = rmax * psi_eq * psi_eq;

  #define FORMAT std::setw(25) << std::right << std::setprecision(5) << std::fixed << std::showpos
  
  auto print_shells = [&](int dom_min, int dom_max) {
    int cnt = 1;
    for(int i = dom_min; i < dom_max; ++i) {
      std::string shell{"SHELL" + std::to_string(cnt) + " = "};
      std::cout << FORMAT << shell << bco_utils::get_radius(space.get_domain(i), OUTER_BC) << "\n";
      cnt++;
    }
  };

  auto res_r = space.get_domain(0)->get_nbr_points()(0);
  auto res_t = space.get_domain(0)->get_nbr_points()(1);

  std::cout << FORMAT << "RES = " << "[" << res_r << "," << res_t << "]\n"
            << FORMAT << "Coord R_IN = " << rin1 << "\n"
            << FORMAT << "Coord R = " << rmax << " [Axial ratio: " << rmin/rmax << "]\n";
  print_shells(2, ndom-2);
  std::cout << FORMAT << "Coord R_OUT = " << bco_utils::get_radius(space.get_domain(ndom-2), OUTER_BC) << "\n\n"
            << FORMAT << "Areal R = " << AR << " [" << AR * M2km << "km]\n"
            << FORMAT << "Baryonic Mass = " << Mbar << "\n"
            << FORMAT << "ADM Mass = " << Madm << " [" << bconfig(MADM) << "]\n"
            << FORMAT << "ANG Momentum = " << J << "\n"
            << FORMAT << "Chi = " << J / Madm / Madm << " [" << bconfig(CHI) << "]\n"
            << FORMAT << "Central Omega = " << central_omg << " [" << bconfig(OMEGA)*M2Hz << "Hz]\n"
            << FORMAT << "Equator Omega = " << equator_omg << " [" << equator_omg*M2Hz << "Hz]\n"  
            << FORMAT << std::scientific << "Central Density = " << nc << "\n"
            << FORMAT << std::scientific << "Central log(h) = " << loghc << "\n"
            << FORMAT << std::scientific << "Central Pressure = " << pc << "\n"
            << FORMAT << std::scientific << "Central Euler Constant = " << central_euler << "\n\n"
            << FORMAT << "Mk = " << Mk << std::scientific << ", Diff: " << 2. * fabs(Madm-Mk)/(Madm+Mk) << "\n";
}
