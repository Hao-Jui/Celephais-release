/*
 * =====================================================================================
 *
 *       Filename:  lorene_io.cpp
 *
 *    Description:  Read Lorene Table
 *
 *        Version:  1.0
 *        Created:  01/05/2017 23:21:18
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Elias Roland Most (ERM), most@fias.uni-frankfurt.de
 *   Organization:  Goethe University Frankfurt
 *
 * =====================================================================================
 */

/*
 * Modifications (Celephais):
 *   2026-06-16  Modified for the Celephais tree; see
 *               PATCHES-KADATH-UPSTREAM.md and LICENSE_SOURCE_AUDIT.tsv.
 */

#include <array>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "../margherita.hh"

/*
static const std::array<std::string, Hot_Slice::v_index::NUM_VARS + 1>
    var_names{{"n_B [fm^{-3}]", "e [g/cm^3]", "p [dyn/cm^2]", "Y_e", "T [MeV]",
               "s [kB/m_B]", "c_s^2 [c]"}};

static const std::array<double, Hot_Slice::v_index::NUM_VARS + 1> conv{
    {MC::RHOGF * MC::mnuc_cgs * MC::cm3_to_fm3, MC::RHOGF, MC::PRESSGF, 1., 1.,
     1., 1.}};
*/
inline std::array<std::vector<double>, 3> Lorene_Table(const std::string &filename, const double h_cut,
                                                      const double mnuc_cgs = 0.0) {
  namespace constants = Margherita_constants;

  // Nuclear mass unit used to convert the table's baryon number density n
  // (fm^-3) to rest-mass density rho. Configurable via the [eos] mnuc_cgs key so
  // it can match the convention the table was generated with; a non-positive
  // value (default / unset / NaN from an older config) falls back to the
  // built-in Margherita constant.
  const double mnuc = (mnuc_cgs > 0.0) ? mnuc_cgs : constants::mnuc_cgs;

  std::ifstream file(filename);
  if (!file.is_open()) {
    throw std::runtime_error("failed to open " + filename);
  }
  // std::cout << std::setiosflags(std::ios::scientific) <<
  // std::setprecision(16);

  // Create vectors
  std::array<std::vector<double>, 3> vectors;

  // Skip lines
  constexpr double skip_lines = 9; //8;
  std::string line;
  for (int i = 0; i < skip_lines; i++) {
    std::getline(file, line);
  }

  while (std::getline(file, line)) {
    double n, e, p, dummy;
    file >> dummy >> n >> e >> p;

    double rho = n * constants::RHOGF * mnuc * constants::cm3_to_fm3;
    double eps = e / n / mnuc / constants::cm3_to_fm3 - 1.;
    double press = p * constants::PRESSGF;

    double h = 1. + eps + press / rho;
    if(h >= h_cut) {
      vectors[0].push_back(rho);
      vectors[1].push_back(eps);
      vectors[2].push_back(press);
    }
  }
  // Last entry is eof duplication, so remove
  vectors[0].pop_back();
  vectors[1].pop_back();
  vectors[2].pop_back();

  if (vectors[0].size() != vectors[1].size() ||
      vectors[0].size() != vectors[2].size()) {
    throw std::runtime_error("Lorene table columns have inconsistent lengths");
  }

  return vectors;
}

/*
template <int begin = 0, int end = Hot_Slice::v_index::NUM_VARS>
static void write_table(std::ostream &file) {
  std::string whitespace{"    "};

  // Print header
  file << "#" << std::endl;
  file << "#" << std::endl;
  file << "#" << std::endl;
  file << "#" << std::endl;
  file << "#" << std::endl;
  file << Hot_Slice::lintp.size() << std::endl;
  file << "#" << std::endl;
  file << "# index";

  for (int i = begin; i <= end; ++i) file << whitespace << var_names[i];

  file << std::endl;
  file << "#" << std::endl;

  file << std::setiosflags(std::ios::scientific) << std::setprecision(16);

  // auto rhoL = exp(Hot_Slice::lintp[0]);
  // typename Hot_Slice::error_t error;
  // auto interp = Hot_Slice::get_extra_quantities(rhoL, error);
  // const auto press_0 = exp(interp[Hot_Slice::v_index::PRESS]);

  // Write table
  for (int nn = 0; nn < Hot_Slice::lintp.size(); ++nn) {
    auto rhoL = exp(Hot_Slice::lintp[nn]);
    typename Hot_Slice::error_t error;
    auto interp = Hot_Slice::get_extra_quantities(rhoL, error);

    // We don't output eps but rho(1+eps), so need to modify
    interp[Hot_Slice::v_index::EPS] =
        rhoL * (1. + exp(interp[Hot_Slice::v_index::EPS]));
    interp[Hot_Slice::v_index::TEMP] = exp(interp[Hot_Slice::v_index::TEMP]);
    interp[Hot_Slice::v_index::PRESS] = exp(interp[Hot_Slice::v_index::PRESS]);

    file << 42;
    if (begin == 0) file << whitespace << rhoL / conv[0];

    for (int i = std::max(1, begin); i <= end; ++i) {
      file << whitespace << interp[i - 1] / conv[i];
    }
    file << std::endl;
  }
}*/
