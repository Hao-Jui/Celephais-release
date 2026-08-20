//
// This file is part of Margherita, the light-weight EOS framework
//
//  Copyright (C) 2017, Elias Roland Most
//                      <emost@th.physik.uni-frankfurt.de>
//
//  This program is free software: you can redistribute it and/or modify
//  it under the terms of the GNU General Public License as published by
//  the Free Software Foundation, either version 3 of the License, or
//  (at your option) any later version.
//
//  This program is distributed in the hope that it will be useful,
//  but WITHOUT ANY WARRANTY; without even the implied warranty of
//  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//  GNU General Public License for more details.
//
//  You should have received a copy of the GNU General Public License
//  along with this program.  If not, see <http://www.gnu.org/licenses/>.

/*
 * Modifications (Celephais):
 *   2026-06-16  Modified for the Celephais tree; see
 *               PATCHES-KADATH-UPSTREAM.md and LICENSE_SOURCE_AUDIT.tsv.
 */

/********************************
 * Setup piecewise polytropes
 *
 * Written in June 2016 by Elias R. Most
 * <emost@th.physik.uni-frankfurt.de>
 *
 * Revised in May 2021 by Samuel D. Tootle
 * <tootle@itp.uni-frankfurt.de>
 *
 * -revision <28.May.21> 
 *  Updated to be purely standalone
 *
 * We do things here as in Whisky_Exp
 * (See Takami et al. https://arxiv.org/pdf/1412.3240v2.pdf)
 * (also see Read et al. https://arxiv.org/pdf/0812.2163v1.pdf)
 ********************************/

#include "cold_pwpoly.hh"
#include "cold_pwpoly_implementation.hh"

#include "../margherita.hh"
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace Kadath {
namespace Margherita {

inline std::string read_polytrope(std::string fname) {
  std::ifstream f(fname);

  if(!f.is_open()){
    throw std::runtime_error("File: " + fname + " cannot be opened");
  }
  
  //string descriptor to ignore  
  std::string descr;

  //lambda to ignore leading comments and blank lines 
  //up to the next value to extract
  auto skip_comments = [&]() {
    auto peek_c = f.peek();
    while(peek_c == '#' || peek_c == '\n') {
      std::getline(f, descr, '\n');
      peek_c = f.peek();
    }
  };
  
  auto skip_and_grab =  [&](auto& val) {
    skip_comments();  
    f >> descr >> val;
  };    
  
  skip_and_grab(Cold_PWPoly::num_pieces);
  if (Cold_PWPoly::num_pieces > Cold_PWPoly::max_num_pieces) {
    throw std::runtime_error("Polytrope has more pieces than Cold_PWPoly supports");
  }
  
  skip_and_grab(Cold_PWPoly::rhomin);
  skip_and_grab(Cold_PWPoly::rhomax);
  skip_and_grab(Cold_PWPoly::k_tab[0]);
  skip_and_grab(Cold_PWPoly::P_tab[0]);

  auto read_tab = [&](auto& ary) {
    skip_comments();
    f >> descr;
    //Can't use foreach since array is static length
    for(int i = 0; i < Cold_PWPoly::num_pieces; ++i){
      if(!(f >> ary[i])) {
        throw std::runtime_error("Not enough vars in " + descr);
      }
    }
  };

  read_tab(Cold_PWPoly::gamma_tab);
  read_tab(Cold_PWPoly::rho_tab);
	
	std::string units;
  skip_and_grab(units);
	return units;
}

void Margherita_setup_polytrope(std::string polytrope_file) {
  namespace constants = Margherita_constants;
  auto Units = read_polytrope(polytrope_file);
	const double gam0m1 = Cold_PWPoly::gamma_tab[0] - 1.0;
	double rho_unit = 1.;
	double K_unit = 1.;

	if (Units != "geometrised") {
    if (Units == "cgs") {
      rho_unit = 1.0 * constants::RHOGF ;
      K_unit = pow(constants::INVRHOGF, gam0m1) / constants::c2_cgs;
    } else {
      if (Units == "cgs_cgs_over_c2") {
        rho_unit = 1.0 * constants::RHOGF;
        K_unit = pow(constants::INVRHOGF, gam0m1);
      } else {
        throw std::runtime_error("Unit system, " + Units + ", not recognised");
      }
    }
  }

  //eps_tab == continuity coefficients on the bounds between
  //pieces.  The first is always 0.
  Cold_PWPoly::eps_tab[0] = 0.0;

  //Unit conversion
  Cold_PWPoly::k_tab[0] *= K_unit;
  for (int i = 0; i < Cold_PWPoly::num_pieces; ++i)
    Cold_PWPoly::rho_tab[i] = rho_unit * Cold_PWPoly::rho_tab[i];

  // Setup piecewise polytrope
  for (int i = 1; i < Cold_PWPoly::num_pieces; ++i) {
    // Consistency check
    if (Cold_PWPoly::rho_tab[i] <= Cold_PWPoly::rho_tab[i-1]) {
      throw std::runtime_error("rho_tab must increase monotonically");
    }
  	const double gam_im1 = Cold_PWPoly::gamma_tab[i] - 1.0;

    Cold_PWPoly::k_tab[i] =
        Cold_PWPoly::k_tab[i - 1] *
        pow(Cold_PWPoly::rho_tab[i],
            Cold_PWPoly::gamma_tab[i - 1] - Cold_PWPoly::gamma_tab[i]);

    Cold_PWPoly::eps_tab[i] =
        Cold_PWPoly::eps_tab[i - 1] +
        Cold_PWPoly::k_tab[i - 1] *
            pow(Cold_PWPoly::rho_tab[i], Cold_PWPoly::gamma_tab[i - 1] - 1.0) /
            (Cold_PWPoly::gamma_tab[i - 1] - 1.0) -
        Cold_PWPoly::k_tab[i] *
            pow(Cold_PWPoly::rho_tab[i], gam_im1) /  (gam_im1);
    Cold_PWPoly::P_tab[i] =
        Cold_PWPoly::k_tab[i] *
        pow(Cold_PWPoly::rho_tab[i], Cold_PWPoly::gamma_tab[i]);

    double eps = Cold_PWPoly::eps_tab[i] + 
        Cold_PWPoly::P_tab[i] / Cold_PWPoly::rho_tab[i] / gam_im1;
    
    Cold_PWPoly::h_tab[i] = 1. + eps + Cold_PWPoly::P_tab[i] / Cold_PWPoly::rho_tab[i];
  }
  #ifdef DEBUG
  for (int nn = 0; nn < Cold_PWPoly::num_pieces; ++nn) {
    printf("nn= %d : K=%.15e , rho= %.15e, gamma= %.15e, eps= %.15e, P=.%15e, h=.%15e \n", nn,
        Cold_PWPoly::k_tab[nn], Cold_PWPoly::rho_tab[nn],
        Cold_PWPoly::gamma_tab[nn], Cold_PWPoly::eps_tab[nn],
        Cold_PWPoly::P_tab[nn], Cold_PWPoly::h_tab[nn]);
  }
  #endif

  return;
}
}
}
