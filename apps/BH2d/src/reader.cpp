#include "For_Kadath/Utilities/Exporters/coord_fields.hpp"
#include "For_Kadath/Space/adapted_bh_polar.hpp"
#include "For_Kadath/IO/be_file_source.hpp"
#include "For_Kadath/IO/be_file_sink.hpp"
#include "mpi.h"
#include "Apps/Startup/solver_startup.hpp"
#include "For_Kadath/Config/config_bco.hpp"

#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

using namespace Kadath;

template <typename Space_t, typename... Scalars>
void write_data(const std::string& filename, const Space_t& space, const Scalars&... fields) {
  std::ofstream file(filename);
  file << std::scientific << std::setprecision(16);

  const int ndom = space.get_nbr_domains();
  for (int d = space.HOMOTHETIC_INNER; d < ndom; ++d) {
    const auto* dom = space.get_domain(d);
    if (dom->get_ndim() < 2) continue;
    const auto& nbr_pts = dom->get_nbr_points();
    int npts_r = nbr_pts(0);
    int npts_t = nbr_pts(1);
    int r_start = (d == space.HOMOTHETIC_INNER) ? 0 : 1;
    int r_end = (d == ndom - 1) ? npts_r - 1 : npts_r;
    for (int i = r_start; i < r_end; ++i) {
      for (int j = 0; j < npts_t; ++j) {
        Index pos(nbr_pts);
        pos.set_start();
        pos.set(0) = i;
        pos.set(1) = j;

        double r = dom->get_radius()(pos);
        double z = dom->get_cart(1)(pos);
        double rho = dom->get_cart(2)(pos);
        double theta = std::atan2(rho, z);

        file << r << " " << theta;
        ((file << " " << fields(d)(pos)), ...);
        file << "\n";
      }
    }
  }
  std::cout << "Wrote " << filename << std::endl;
}

template <typename Space_t>
void write_equator_profile(const std::string& filename, const Space_t& space,
                           const std::vector<std::pair<std::string, Scalar>>& fields) {
  std::ofstream file(filename);
  file << std::scientific << std::setprecision(16);

  // Header
  file << "# r";
  for (const auto& [name, _] : fields) {
    file << " " << name;
  }
  file << "\n";

  const int ndom = space.get_nbr_domains();
  for (int d = space.HOMOTHETIC_INNER; d < ndom; ++d) {
    const auto* dom = space.get_domain(d);
    if (dom->get_ndim() < 2) continue;
    const auto& nbr_pts = dom->get_nbr_points();
    int npts_r = nbr_pts(0);
    int npts_t = nbr_pts(1);
    int j_eq = npts_t - 1;  // Equator index (theta = pi/2)
    int r_start = (d == space.HOMOTHETIC_INNER) ? 0 : 1;
    int r_end = (d == ndom - 1) ? npts_r - 1 : npts_r;

    for (int i = r_start; i < r_end; ++i) {
      Index pos(nbr_pts);
      pos.set_start();
      pos.set(0) = i;
      pos.set(1) = j_eq;

      double r = dom->get_radius()(pos);
      file << r;
      for (const auto& [name, field] : fields) {
        file << " " << field(d)(pos);
      }
      file << "\n";
    }
  }
  std::cout << "Wrote " << filename << std::endl;
}

template <typename Space_t>
void write_coef(const std::string& filename, const Space_t& space,
                const std::vector<std::pair<std::string, const Scalar*>>& fields) {
  std::ofstream file(filename);
  file << std::scientific << std::setprecision(16);

  const int ndom = space.get_nbr_domains();
  
  for (int d = space.HOMOTHETIC_INNER; d < ndom; ++d) {
    const auto* dom = space.get_domain(d);
    if (dom->get_ndim() < 2) continue;
    
    file << "\n# ==========================================\n";
    file << "# Domain " << d << "\n";
    file << "# ==========================================\n";

    // Write radius mapping
    const auto& nbr_pts = dom->get_nbr_points();
    file << "# Radius mapping (index -> radius)\n";
    for (int i = 0; i < nbr_pts(0); ++i) {
      Index pos(nbr_pts);
      pos.set_start();
      pos.set(0) = i;
      pos.set(1) = 0;
      double r = dom->get_radius()(pos);
      file << i << " " << r << "\n";
    }
    file << "\n";

    // Write coefficients for all fields in this domain
    for (const auto& [name, field] : fields) {
      if (field == nullptr) continue;

      const Val_domain& vd = (*field)(d);
      vd.coef();
      Array<double> cf = vd.get_coef();
      const auto& nbr_coef = cf.get_dimensions();

      file << "# Field: " << name << "\n";
      
      if (nbr_coef.get_ndim() == 2) {
        file << "# radius theta indices, coefficients\n";
        for (int i = 0; i < nbr_coef(0); ++i) {
          for (int j = 0; j < nbr_coef(1); ++j) {
            Index pos(nbr_coef);
            pos.set_start();
            pos.set(0) = i;
            pos.set(1) = j;
            file << i << " " << j << " " << cf(pos) << "\n";
          }
        }
      } else if (nbr_coef.get_ndim() == 3) {
        file << "# i j k value\n";
        for (int i = 0; i < nbr_coef(0); ++i) {
          for (int j = 0; j < nbr_coef(1); ++j) {
            for (int k = 0; k < nbr_coef(2); ++k) {
              Index pos(nbr_coef);
              pos.set_start();
              pos.set(0) = i;
              pos.set(1) = j;
              pos.set(2) = k;
              file << i << " " << j << " " << k << " " << cf(pos) << "\n";
            }
          }
        }
      }
      file << "\n";
    }
  }
  std::cout << "Wrote " << filename << std::endl;
}

int main(int argc, char **argv) {
  KadathApps::init_mpi(argc, argv);

  const int rc = KadathApps::guarded_run([&] {
  if (argc < 2) {
    KADATH_THROW("Usage: ./reader /<path>/<ID base name>.toml|.dat  e.g. ./reader converged_BH.toml");
  }

  const std::string ifilename = KadathApps::toml_config_path_from_reader_input(argv[1]);
  using config_t = kadath_config<BCO_BH_INFO>;
  config_t bconfig{ifilename};

  const std::string spacein = bconfig.space_filename();

  BeFileSource ff1(spacein);

  Space_adapted_bh_polar space(ff1);
  Scalar lapse(space, ff1);
  Scalar metA(space, ff1);
  Scalar metB(space, ff1);
  Scalar beta(space, ff1);
  Scalar varscal(space, ff1);
  beta.affect_parameters();
  beta.set_parameters()->set_m_quant() = 1;
  beta.std_base();

  varscal.affect_parameters();
  varscal.set_parameters()->set_m_quant() = 1;
  varscal.std_base();

  std::string header(22,'#');
  const int ndom = space.get_nbr_domains();
  /*
  for (int d = space.HOMOTHETIC_INNER; d < ndom; ++d) {
    const Dim_array nbr_pts = space.get_domain(d)->get_nbr_points();
    if (nbr_pts.get_ndim() < 2) {
      continue;
    }
    const int ntheta = nbr_pts(1);
    const int j_eq = ntheta - 1;
    std::cout << "metB (domain " << d << "):" << std::endl;
    for (int i = 0; i < nbr_pts(0); ++i) {
      Index pos_eq(nbr_pts);
      Index pos_po(nbr_pts);
      pos_eq.set(0) = i;
      pos_eq.set(1) = j_eq;
      pos_po.set(0) = i;
      pos_po.set(1) = 0;
      std::cout << " i ="<< std::setw(2) << i
                << " r =" << std::scientific << std::setprecision(3) << std::setw(12) << space.get_domain(d)->get_radius()(pos_eq)
                << "  [lapse, A, B, bet] =" 
                << std::setw(12) << lapse.set_domain(d)(pos_eq) 
                << std::setw(12) << metA.set_domain(d)(pos_eq) 
                << std::setw(12) << metB.set_domain(d)(pos_eq)
                << std::setw(12) << beta.set_domain(d)(pos_eq)
                << std::setw(12) << varscal.set_domain(d)(pos_eq)
                << std::endl;
    }
  }
  */
  System_of_eqs syst(space, 2, ndom - 1);
  Scalar oneField (space) ; oneField = 1. ; oneField.std_base();
  syst.add_cst("ones", oneField);
  syst.add_cst("4piG", 4.0 * M_PI);
  syst.add_cst("N"   , lapse);
  syst.add_cst("A"   , metA);
  syst.add_cst("B"   , metB);
  syst.add_cst("shift", beta);
  syst.add_cst("sphi", varscal);
  syst.add_cst("ome" , bconfig(OMEGA));
  syst.add_cst("mu"  , bconfig.template gravity<double>(GRAV_MASS_SCAL));

  syst.add_def("winding= ones");
  syst.add_def("bet = divrsint(shift)");
  syst.add_def("wmb2= (ome*ones+bet)^2");

  syst.add_def("gradN_i = grad(N)");
  syst.add_def("gradB_i = grad(B)");
  syst.add_def("gradbet_i = grad(bet)");
  syst.add_def("gradsphi_i = grad(sphi)");

  syst.add_def("DADA    = scal( grad(A), grad(A)  )");
  syst.add_def("DNDlnB  = scal( gradN  , gradB  ) / B");
  syst.add_def("DbetDbet= scal( gradbet, gradbet)");
  syst.add_def("DbetDN  = scal( gradbet, gradN  )");
  syst.add_def("DbetDlnB= scal( gradbet, gradB  ) / B");
  syst.add_def("DsphDsph= scal( gradsphi, gradsphi )");

  syst.add_def("wmb2= (ome*ones + bet*winding)^2");
  syst.add_def("aziscal = N^2 / B^2 * divrsint(sphi) * divrsint(sphi)");

  syst.add_def("NNJ = 2 * winding * N * (ome*ones + bet*winding) * sphi^2 ");
  syst.add_def("NNE =    (  wmb2 * sphi^2 + aziscal) + N^2 * ( DsphDsph / A^2 +   mu^2 * sphi^2 )");
  syst.add_def("NNS =    (3*wmb2 * sphi^2 - aziscal) - N^2 * ( DsphDsph / A^2 + 3*mu^2 * sphi^2 )");
  syst.add_def("NNSphSph = (wmb2 * sphi^2 + aziscal) - N^2 * ( DsphDsph / A^2 +   mu^2 * sphi^2 )");

  syst.add_def("THETA = dr(A)/A + dr(B)/B + divr(2*ones)");


  syst.add_def("lapNterms = N * DNDlnB - B^2 * multrsint(multrsint(DbetDbet)) / 2");
  syst.add_def("sourceNterms = 4piG * A^2 * (NNE + NNS)");

  syst.add_def("lapAterms = N * A * lap2(N) - N * N * DADA / A - 3 * A * B^2 * multrsint(multrsint(DbetDbet)) / 4");
  syst.add_def("sourceAterms = 2 * 4piG * A^3 * NNSphSph");
  
  syst.add_def("lapShifts = - multrsint(DbetDN) + 3 * N * multrsint(DbetDlnB)");
  syst.add_def("sourceShifts = divrsint( 4 * 4piG * A^2 * NNJ ) / B^2");

  syst.add_def("sourceBterms = 2 * 4piG * A^2 * B * (NNS - NNSphSph)");

  syst.add_def("NmdivincB2 = N^2 * winding^2 * divrsint(sphi) * divrsint(ones-A^2/B^2)");
  syst.add_def("lapscalar = A^2 * wmb2 * sphi + NmdivincB2 "
               "+ N * scal( gradsphi, gradN ) + N^2 * scal( gradsphi, gradB ) / B ");
  syst.add_def("sourcescalar = A^2 * mu^2 * N^2 * sphi");


  syst.add_def("eqN  = N * lap(N)     + lapNterms ");
  syst.add_def("eqA  = N*N * lap2(A)  + lapAterms ");
  syst.add_def("eqB  = N * (2 * lap(N*B) - lap2(N*B))");
  syst.add_def("eqbet= N * lap(shift) + lapShifts ");
  syst.add_def("eqphi= N*N * lap(sphi) + lapscalar ");


  syst.add_def(ndom - 1, "intJk = multrsint(multrsint(dr(bet))) / 4 / 4piG");
  syst.add_def(ndom - 1, "intMadm = -dr(A^2+B^2) / 4 / 4piG");
  syst.add_def(ndom - 1, "intMk = dr(N) / 4piG");
  syst.add_def("intMsq= A * B / 4 / 4piG");


  syst.add_def("check1 = N*N * lap(sphi)");
  syst.add_def("check2 = A^2 * wmb2 * sphi");
  syst.add_def("check3 = NmdivincB2");
  syst.add_def("check4 = N^2 * winding^2 * sphi * (ones-A^2/B^2)");

  Scalar theta_val = syst.give_val_def("THETA");
  Val_domain integMsq (syst.give_val_def("intMsq" )()(2));
  Val_domain integJk  (syst.give_val_def("intJk"  )()(ndom-1));
  Val_domain integMadm(syst.give_val_def("intMadm")()(ndom-1));
  Val_domain integMk  (syst.give_val_def("intMk"  )()(ndom-1));

  
  double Mirrsq = space.get_domain(2       )->integ(integMsq , INNER_BC);
  double J      = space.get_domain(ndom - 1)->integ(integJk  , OUTER_BC);
  double Madm   = space.get_domain(ndom - 1)->integ(integMadm, OUTER_BC);
  double Mk     = space.get_domain(ndom - 1)->integ(integMk  , OUTER_BC);

  double Mirr = std::sqrt(Mirrsq);
  double Mch = Mirr * std::sqrt(1+J*J/Mirrsq/Mirrsq/4);

  // Find maximum of |sphi|
  double sphi_max = 0.0;
  for (int d = space.HOMOTHETIC_INNER; d < ndom; ++d) {
    const Dim_array nbr_pts = space.get_domain(d)->get_nbr_points();
    if (nbr_pts.get_ndim() < 2) continue;
    Index pos(nbr_pts);
    for (int i = 0; i < nbr_pts(0); ++i) {
      pos.set(0) = i;
      for (int j = 0; j < nbr_pts(1); ++j) {
        pos.set(1) = j;
        double val = std::fabs(varscal(d)(pos));
        if (val > sphi_max) sphi_max = val;
      }
    }
  }

  #define FORMAT std::setw(13) << std::left << std::showpos
  std::ios_base::fmtflags f(std::cout.flags());
  std::cout << header+" BH "+header+'\n'
            << FORMAT << "Mirr: " << Mirr << std::endl
            << FORMAT << "J Komar: " << J << std::endl
            << FORMAT << "Mch: " << Mch << std::endl
            << FORMAT << "M adm: " << Madm << std::endl
            << FORMAT << "Virial err: " << (Mk-Madm) / (Mk+Madm) << std::endl
            << FORMAT << "max|sphi|: " << sphi_max << std::endl
            << header+"####"+header+'\n' << std::endl;
  std::cout.flags(f);
  #undef FORMAT

  write_data("profile.dat", space, lapse, metA, metB, beta, varscal);
  write_coef("coef.dat", space,
             {{"lapse", &lapse}, {"metA", &metA}, {"metB", &metB},
              {"beta", &beta}, {"varscal", &varscal}});

  // Write equatorial radial profile of source terms and equations
  Scalar sourceN = syst.give_val_def("sourceNterms");
  Scalar sourceA = syst.give_val_def("sourceAterms");
  Scalar sourceS = syst.give_val_def("sourceShifts");
  Scalar sourceB = syst.give_val_def("sourceBterms");
  Scalar sourceP = syst.give_val_def("sourcescalar");
  Scalar eqN     = syst.give_val_def("eqN");
  Scalar eqA     = syst.give_val_def("eqA");
  Scalar eqB     = syst.give_val_def("eqB");
  Scalar eqbet   = syst.give_val_def("eqbet");
  Scalar eqscal  = syst.give_val_def("eqphi");
  Scalar check1  = syst.give_val_def("check1");
  Scalar check2  = syst.give_val_def("check2");
  Scalar check3  = syst.give_val_def("check3");
  Scalar check4  = syst.give_val_def("check4");
  write_equator_profile("source.dat", space,
                        {{"sourceN", sourceN}, {"sourceA", sourceA},
                         {"sourceS", sourceS}, {"sourceB", sourceB},  {"sourceP", sourceP},
                         {"eqN", eqN}, {"eqA", eqA},
                         {"eqB", eqB}, {"eqbet", eqbet}, {"eqphi", eqscal},
                         {"check1",check1},
                         {"check2",check2},
                         {"check3",check3},
                         {"check4",check4}});
  });
  MPI_Finalize();
  return rc;
}
