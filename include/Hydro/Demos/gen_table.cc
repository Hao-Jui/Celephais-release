/*
 * Modifications (Celephais):
 *   2026-06-16  Modified for the Celephais tree; see
 *               PATCHES-KADATH-UPSTREAM.md and LICENSE_SOURCE_AUDIT.tsv.
 */

#include <iostream>
#include <cmath>
#include "../margherita/pwp/cold_pwpoly_implementation.hh"
#include "../margherita/margherita.hh"

#include <iomanip>
#include <fstream>
#include <cassert>
#include <vector>
#define STANDALONE
//code repurposed and updated based on setup_polytrope.cc

using namespace Kadath;
using namespace Kadath::Margherita;
static const std::array<std::string, 3>
    var_names{"n_B [fm^{-3}]", "e [g/cm^3]", "p [dyn/cm^2]"};

template<typename eos>
static double h_cold__rho(double rho) {
  typename eos::error_t err;

  double eps_cold = 0.;
  double pressure = eos::press_cold_eps_cold__rho(eps_cold, rho, err);
  double h = 1. + eps_cold + pressure / rho;

  return h;
}
static void write_table(std::ostream &file, auto& table) {
  using namespace Margherita_constants;
  std::string whitespace{"    "};

  // Print header
  file << "#" << std::endl;
  file << "#" << std::endl;
  file << "#" << std::endl;
  file << "#" << std::endl;
  file << "#" << std::endl;
  //file << Hot_Slice::lintp.size() << std::endl;
  file << "42" << std::endl;
  file << "#" << std::endl;
  file << "# index";

  for (int i = 0; i < var_names.size(); ++i) file << whitespace << var_names[i];

  file << std::endl;
  file << "#" << std::endl;

  file << std::setiosflags(std::ios::scientific) << std::setprecision(16);
  int idx = 0;
  // Write table
  for (auto& row : table) {
    double n = row[0] / RHOGF / mnuc_cgs / cm3_to_fm3;
    double e = (row[1] + 1) * n * mnuc_cgs * cm3_to_fm3;
    double p = row[2] / PRESSGF;
    std::vector<double> data {n, e, p};
    file << idx;
    for (auto& out : data) {
      file << whitespace << out;
    }
    file << std::endl;
    ++idx;
  }
}
int main() {
  using eos = Kadath::Margherita::Cold_PWPoly;
  eos::num_pieces = 4;
  assert(eos::num_pieces <= eos::max_num_pieces);

  eos::gamma_tab[0] = 1.3569199999999999;
  eos::gamma_tab[1] = 3.0049999999999999;
  eos::gamma_tab[2] = 2.988;
  eos::gamma_tab[3] = 2.851;
 
  eos::rho_tab[0] = 0.;
  eos::rho_tab[1] = 2.367364e-04;
  eos::rho_tab[2] = 8.114561e-04;
  eos::rho_tab[3] = 1.619068e-03;
 
  eos::k_tab[0] = 8.948446e-02;

  eos::rhomin = 1e-19;
  eos::rhomax = 1.0;

  double rho_unit = 1.0;
  double K_unit = 1.0;
  eos::P_tab[0]   = 0.; 
  eos::eps_tab[0] = 0.;
  eos::h_tab[0] = 1.;

  // Setup piecewise polytrope
  for (int i = 1; i < eos::num_pieces; ++i) {
    eos::k_tab[i] =
      eos::k_tab[i - 1] *
      pow(eos::rho_tab[i],
          eos::gamma_tab[i - 1] - eos::gamma_tab[i]);
      
    eos::P_tab[i] =
      eos::k_tab[i] *
      pow(eos::rho_tab[i], eos::gamma_tab[i]);
    
    eos::eps_tab[i] =
          eos::eps_tab[i - 1] +
          eos::k_tab[i - 1] *
              pow(eos::rho_tab[i], eos::gamma_tab[i - 1] - 1.0) /
              (eos::gamma_tab[i - 1] - 1.0) -
          eos::k_tab[i] *
              pow(eos::rho_tab[i], eos::gamma_tab[i] - 1.0) /
              (eos::gamma_tab[i] - 1.0);
    double eps = eos::eps_tab[i] + eos::P_tab[i] / eos::rho_tab[i] / (eos::gamma_tab[i] - 1.0);
    eos::h_tab[i] = 1. + eps + eos::P_tab[i] / eos::rho_tab[i];
  }
  //std::cout << h_cold__rho<eos>(1.28e-3) << endl;
  const int s = eos::num_pieces - 1;
  const double gam_mo = (eos::gamma_tab[s] - 1.);
  const double ai   = eos::eps_tab[s-1] - eos::k_tab[s]/gam_mo * pow(eos::rho_tab[s-1], gam_mo);
  
  const double emax = ai + eos::k_tab[s] / gam_mo * eos::rhomax;
  eos::error_t err;
  double emax2 = 0; 
  const double pmax = eos::press_cold_eps_cold__rho(emax2, eos::rhomax, err);
  const double hmax = 1. + emax + pmax / eos::rhomax;
  const double hmax2 = 1. + emax2 + pmax / eos::rhomax;
  std::cout << pmax << '\n' 
  << emax << ", " << emax2 << '\n'
  << hmax << ", " << hmax2 << '\n';
  // Output polytrope
  for (int nn = 0; nn < eos::num_pieces; ++nn) {
    printf("nn= %d : K=%.15e , rho= %.15e, gamma= %.15e, eps= %.15e, P=.%15e, h=.%15e \n", nn,
        eos::k_tab[nn], eos::rho_tab[nn],
        eos::gamma_tab[nn], eos::eps_tab[nn],
        eos::P_tab[nn], eos::h_tab[nn]);
  }
//  double delta = 1.75; //80
//double delta = 1.045; //1000
  double delta = 1.6; //100
  //double delta = 1.022; //2000
  std::vector<std::vector<double>> table(100);
  for(auto& row:table)
    row.resize(3);
  
  int i = 0;
  for(auto& row : table) {
    row[0] = (1+ pow(delta, i)) * eos::rhomin;
    row[2] = eos::press_cold_eps_cold__rho(row[1], row[0], err);
//    row[3] = (row[0] == 0.) ? 1. : 1. + row[1] + row[2] / row[0];
//    std::string ws{"     "};
/*    std::cout << i << ws << std::setw(10) << row[0]
              << ws << std::setw(10) << row[1]
              << ws << std::setw(10) << row[2]
              << ws << std::setw(10) << row[3] << std::endl;
  */  i++;
  }
  i = 0;
/*  for(auto& row : table) {
    double h = row[3];
    double eps, r, p;
    r = eos::rho__h_cold(h, err);
    p = eos::press_cold_eps_cold__rho(eps, r, err);
    eps = h - 1. - p / r;
    std::string ws{"     "};
    std::cout << i << ws << std::setw(10) << r << ws  << std::setw(10) << row[0] << ws  << std::setw(10) << abs(row[0]-r) 
              << ws << std::setw(10) << eps << ws << std::setw(10) << row[1] << ws << std::setw(10) << abs(row[1]-eps) 
              << ws << std::setw(10) << p << ws << std::setw(10) << row[2] << ws << std::setw(10) << abs(row[2]-p)  
              << ws << std::setw(10) << h << std::endl;
    //f2 << '\t' << row[0] << '\t' << row[1] << '\t' << row[2] << '\t' << h << std::endl;
    i++;
  }
*/
  //f2.close();
  (table.back())[0] = eos::rhomax;
  (table.back())[2] = eos::press_cold_eps_cold__rho((table.back())[1], (table.back())[0], err);

    
  std::ofstream file("pwpoly.lorene");
  write_table(file, table);
  file.close();

  return EXIT_SUCCESS;
}
