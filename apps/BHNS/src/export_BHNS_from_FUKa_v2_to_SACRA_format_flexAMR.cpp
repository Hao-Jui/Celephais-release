#include <string>
#include "For_Kadath/Array/exceptions.hpp"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <filesystem>
#include "mpi.h"
#include "For_Kadath/Kadath_point_h/kadath.hpp"
#include "Hydro/EOS.hh"
#include "Apps/Diagnostics/configured_eos.hpp"
#include "For_Kadath/Utilities/Exporters/coord_fields.hpp"
#include "For_Kadath/Config/config_binary.hpp"
#include "For_Kadath/Utilities/exporter_utilities.hpp"
#include "For_Kadath/IO/be_file_source.hpp"
#include "For_Kadath/IO/be_file_sink.hpp"
#include <cmath>



namespace fs = std::filesystem;
using namespace Kadath;
using namespace export_utils;

string format_eWpDe3(double val, int Width, int Decim) ;
string format_iW(int val, int Width) ;
Point point_from_shifted_spherical(double r, double theta, double phi, double shift_x, double shift_y) ;



int main(int argc, char** argv) {


    // initialize MPI
    int rc = MPI_Init(&argc, &argv);
    if (rc!=MPI_SUCCESS)
    {
      cerr << "Error starting MPI" << endl;
      MPI_Abort(MPI_COMM_WORLD, rc);
    }
    int rank(0), nb_procs(0);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nb_procs);


    // USE INFORMATION
    if(argc < 3)
    {
        std::cerr << "Usage: ./export_BHNS_from_FUKa_v2_to_SACRA_format /<path>/<str: BNS ID basename>.toml ./<output_path>/" << std::endl;
        KADATH_THROW("e.g. ./export_BHNS_from_FUKa_v2_to_SACRA_format converged.QUASI_EQUIL.9.toml ./exported_levels");
    }
    // ACCESS CONFIGURATION FILE (.toml)
    std::string input_file_path = argv[1];
    kadath_config<BIN_INFO> bconfig(input_file_path);
    // GET BINARY DATA FILE NAME (.dat)
    std::string kadath_filename = bconfig.space_filename();
    // CHECK EXISTENCE OF BINARY DATA FILE
    if(!fs::exists(kadath_filename))
    {
        std::ostringstream oss;
        oss << "File: " << kadath_filename << " not found.\n\n";
        KADATH_THROW(oss.str());
    }





    // PARAMETERS OF OUTPUT INITIAL DATA
    double const time(0) ; //initial time of simulation
    int const depth_max1(10) ; //maximum depth for object 1; for NS
    int const depth_max2(9) ; //maximum depth for object 2: for BH
    bool const is_ns_obj1(true) ; //true: object 1 is NS, false: object 1 is BH
    int const flvmax(4) ; // fixed levels range from 0 to flvmax and are centered on total center of mass #5
    
    int const depth_max(depth_max1) ; 
    int const lvmax(flvmax + (depth_max1 - flvmax) + (depth_max2 - flvmax)) ;
    int const lvmax0(lvmax+1) ;

    // In exple below (not to scale because grids at child leved should exactly be half size of parent level), depth_max=3, flvmax=1
    // dep0: __________________________lv0
    // dep1:  _____________lv1
    // dep2:  ____lv2  ____lv4
    // dep3:   __lv3    __lv5


    int const jd_org(-42) ;  // -63 (N62stg) (odd)  , -63 (N63nstg) (odd)
    int const ju_org(41) ;   //  62 (N62stg) (even) ,  63 (N63nstg) (odd)
    int const kd_org(jd_org) ;
    int const ku_org(ju_org) ;
    int const ld_org(0) ;
    int const lu_org(ju_org) ;
    int const total_nb_lg(lu_org - ld_org + 13) ; // = lu_org+6 - (ld_org-6) + 1
    if (nb_procs > lvmax0*total_nb_lg)
    {
      cerr << "Should call the programm with a number of procs less than or at best equal to lvmax0*total_nb_lg = " << lvmax0*total_nb_lg << "." << endl;
      MPI_Abort(MPI_COMM_WORLD, rc);
    }


    bool const is_cell_centered(true) ; //false
    int cell_cent_shift(0) ;
    if(is_cell_centered)
    {
        cell_cent_shift = 1 ;
    }


    double ts_dx[lvmax0] {-1000} ;
    double ts_dy[lvmax0] {-1000} ;
    double ts_dz[lvmax0] {-1000} ;

    
    //l37-38 FUKa...BNS/setup.cpp: // conversion from solar mass to km  ////ie M2km = solar mass/planck mass  * planck length (in km) = solar_mass * G/c^2
    double const M2km = 1.4769994423016508;

    //set grid resolution in the unit of M_sun for depth_max=depth_max1 (not depth_max2)    
    //ts_dx[lvmax] = bconfig(DIST)*pow(2,flvmax - depth_max + 3)/(ju_org-jd_org + cell_cent_shift) ;
    //ts_dx[lvmax] = bconfig(DIST)/(16*(ju_org + 1)) ;
    //ts_dx[lvmax] = 0.1/M2km
    ts_dx[lvmax] = 2.0*(13.2/1.48*1.2)/(ju_org-jd_org + cell_cent_shift) ; // SFHo_1.8, R_ns=11.5km, 1.48km/M_sun, L_box/R_ns=1.1

    
    cout << setprecision(15) ;
    if(rank==0)
    {
        cout << "ts_dx[lvmax] = " << ts_dx[lvmax] << endl ;
    }
    ts_dy[lvmax] = ts_dx[lvmax] ;
    ts_dz[lvmax] = ts_dx[lvmax] ;






    // ACCESS BINARY DATA FILE
    BeFileSource fin(kadath_filename);
    // EXTRACT SPACE (DOMAIN DECOMPOSTION) AND FIELDS
    Space_bhns space(fin);
    Scalar conf (space, fin);
    Scalar lapse (space, fin);
    Vector shift_con (space, fin);
    Scalar logh (space, fin);
    Scalar velocity_potential (space, fin); // called phi in Papenfort paper and FUKa code

    if(shift_con.get_valence() != 1)
    {
        cerr << "Wrong valence for shift_con: shift_con.get_valence() = " << shift_con.get_valence() << endl ;
        return -1 ;
    }
    if(shift_con.get_index_type() != CON)
    {
        cerr << "Wrong variance for shift_con: shift_con.get_index_type() = " << shift_con.get_index_type() << " COV=-1, CON=+1 (l20-21 tensor.hpp)"<< endl ;
        return -1 ;
    }





    int ndoms(space.get_nbr_domains());

    Base_tensor basis(shift_con.get_basis());
    Metric_flat fmet(space, basis);

    System_of_eqs syst(space, 0, ndoms - 1);
    fmet.set_system(syst, "f");




            // EXTRACT LOG OF CONFORMAL FACTOR
    syst.add_cst("P", conf);
    syst.add_def("logP = log(P)") ;
    Scalar log_conf(syst.give_val_def("logP")) ;



            // EXTRACT CONFORMAL (EVOLUTION SCALING) EXTRINSIC CURVATURE
    syst.add_cst("N", lapse);
    syst.add_cst("bet", shift_con);
    syst.add_def("AKevolScalCov_ij = (D_i bet_j + D_j bet_i - 2. / 3.* D^k bet_k * f_ij) /2. / N"); //// evolution scaling \tilde{A}_ij = \Psi^{-4} A = \Psi^{-6} \hat{A} used by SACRA
    Tensor AK_evolScalCov(syst.give_val_def("AKevolScalCov")) ;

    if(AK_evolScalCov.get_valence() != 2)
    {
        cerr << "Wrong valence for AK_evolScalCov: AK_evolScalCov.get_valence() = " << AK_evolScalCov.get_valence() << endl ;
        return -1 ;
    }
    if(AK_evolScalCov.get_index_type(0) != COV || AK_evolScalCov.get_index_type(1) != COV)
    {
        cerr << "Wrong variances for AK_evolScalCov: AK_evolScalCov.get_index_type(0) = " << AK_evolScalCov.get_index_type(0) << " COV=-1, CON=+1 (l20-21 tensor.hpp)"<< endl ;
        cerr << "Wrong variances for AK_evolScalCov: AK_evolScalCov.get_index_type(1) = " << AK_evolScalCov.get_index_type(1) << " COV=-1, CON=+1 (l20-21 tensor.hpp)"<< endl ;
        return -1 ;
        // IT WOULD ACTUALLY NOT BE PROBLEM IF AK_evolScalCov WERE CONTRAVARIANT AS ITS INDICES ARE RULED BY THE MINKOWSKI CONFORMAL METRIC
    }


    syst.add_def("wa = 1/P^2");
    Scalar wa(syst.give_val_def("wa")) ;
    syst.add_def("wa2AKevolScalCov_ij = wa^2 * AKevolScalCov_ij");
    Tensor wa2AK_evolScalCov(syst.give_val_def("wa2AKevolScalCov")) ;




    // EXTRACT COVARIANT SPATIAL COMPONENTS OF 4-VELOCITY
    Index center_pos (space.get_domain(space.NS)->get_nbr_points()) ;
    double xm = space.get_domain(space.NS)->get_cart(1)(center_pos);
    double xp = space.get_domain(space.BH)->get_cart(1)(center_pos);
    double xo  = space.get_domain(ndoms-1)->get_cart(1)(center_pos);


    CoordFields<Space_bhns> cfields(space);

    Vector global_rot = cfields.rot_z();
    Vector star_rot = cfields.rot_z(xm);
    Vector ez = cfields.e_cart(3);
    Vector ey = cfields.e_cart(2);
    Vector ex = cfields.e_cart(1);
    Vector esurf = cfields.e_rad();

    syst.add_cst("omes1", bconfig(OMEGA,BCO1));

    syst.add_cst("Mg", global_rot);
    syst.add_cst("xaxis", bconfig(COM));
    syst.add_cst("yaxis", bconfig(COMY));

    syst.add_cst("ex", ex);
    syst.add_cst("ey", ey);

    syst.add_cst("m1", star_rot);

    syst.add_def ("Morb^i = Mg^i + xaxis * ey^i + yaxis * ex^i");

    for(int d = space.NS; d <= space.ADAPTEDNS; ++d)
    {
        syst.add_def  (d, "s^i  = omes1 * m1^i");
    }

    syst.add_cst("velocPot", velocity_potential);
    for(int d = 0; d < ndoms; ++d)
    {
        if(d <= space.ADAPTEDNS)
        {
            syst.add_def(d, "eta_i = D_i velocPot + P^4 * s_i");
        }
        else
        {
            syst.add_def(d, "eta_i = D_i velocPot");
        }
    }

    syst.add_cst("H", logh);
    syst.add_def("h = exp(H)");
    syst.add_def("spatCpntfourVelCov_i = eta_i/h") ;
    Vector velocity_cov(syst.give_val_def("spatCpntfourVelCov")) ;

    if(velocity_cov.get_valence() != 1)
    {
        cerr << "Wrong valence for velocity_cov: velocity_cov.get_valence() = " << velocity_cov.get_valence() << endl ;
        return -1 ;
    }
    if(velocity_cov.get_index_type() != COV)
    {
        cerr << "Wrong variance for velocity_cov: velocity_cov.get_index_type() = " << velocity_cov.get_index_type() << " COV=-1, CON=+1 (l20-21 tensor.hpp)"<< endl ;
        return -1 ;
    }








            // EXTRACT DENSITY
    // get const EOS information - used for initializing EOS later
    const double h_cut = bconfig.eos<double>(HC, BCO1);
    const std::string eos_file = bconfig.eos<std::string>(EOSFILE, BCO1);
    const std::string eos_type = bconfig.eos<std::string>(EOSTYPE, BCO1);

    Param p;
    // add EOS user defined OPEs based on EOS type
    if(eos_type == "Cold_Table")
    {
        using namespace Kadath::Margherita;
        using eos_t = Kadath::Margherita::Cold_Table;

        KadathApps::init_configured_cold_table(bconfig, BCO1);
        syst.add_ope("eps", &EOS<eos_t, eos_var_t::EPSILON>::action, &p);
        syst.add_ope("press", &EOS<eos_t, eos_var_t::PRESSURE>::action, &p);
        syst.add_ope("rho", &EOS<eos_t, eos_var_t::DENSITY>::action, &p);
    }

    if(eos_type == "Cold_PWPoly")
    {
        using namespace Kadath::Margherita;
        using eos_t = Kadath::Margherita::Cold_PWPoly;

        EOS<eos_t, eos_var_t::PRESSURE>::init(eos_file, h_cut);

        syst.add_ope("eps", &EOS<eos_t, eos_var_t::EPSILON>::action, &p);
        syst.add_ope("press", &EOS<eos_t, eos_var_t::PRESSURE>::action, &p);
        syst.add_ope("rho", &EOS<eos_t, eos_var_t::DENSITY>::action, &p);
    } // end adding EOS OPEs
    syst.add_def("rho = rho(h)");
    Scalar density(syst.give_val_def("rho")) ;


    // NEED TO DO THIS BELOW BECAUSE DENSITY IS COMPUTED BY INTERPOLATION BASED ON H, AND THOUGH H IS EXACTLY 0 OUTSIDE NS, MEANING DENSITY SHOULD BE AS WELL, INTERPOLATION DOES NOT LEAD DENSITY TO BE EXACTLY ZERO AND EVEN NOT SO SMALL (UP TO 10^-9, WHICH NOT SO GOOD, SO BETTER TO FORCE IT TO ZERO)
    for (int d(space.ADAPTEDNS+1); d<ndoms; d++) {
        density.set_domain(d).annule_hard() ;
    }


    const Scalar& ALPHA = lapse ; // lapse
    const Scalar& BX = shift_con(1) ; // shift_x in CARTESIAN_BASIS
    const Scalar& BY = shift_con(2) ; // shift_y
    const Scalar& BZ = shift_con(3) ; // shift_z
    const Scalar& phitmp = log_conf ; // wa = exp(-2 phitmp) ; gmet = wa^-2 conformal_g, where det(conformal_g) = 1 so that wa = det(gmet)^(-1/6) : cf eq. (16) (in SACRA paper) which reajusts for this to hold
    Scalar EK(space) ; // Extrinsic "K"urvature trace
    EK.annule_hard() ;
    EK.std_base() ;
    Scalar GXX(EK) ; GXX = 1 ;
    Scalar GYY(EK) ; GYY = 1 ;
    Scalar GZZ(EK) ; GZZ = 1 ;
    Scalar GXY(EK) ;
    Scalar GXZ(EK) ;
    Scalar GYZ(EK) ;

    const Scalar& AKXX = AK_evolScalCov(1,1) ; // traceless "A" "k"urvature_xx in CARTESIAN_BASIS
    const Scalar& AKYY = AK_evolScalCov(2,2) ; // traceless "A" "k"urvature_yy
    const Scalar& AKZZ = AK_evolScalCov(3,3) ; // traceless "A" "k"urvature_zz
    const Scalar& AKXY = AK_evolScalCov(1,2) ; // traceless "A" "k"urvature_xy
    const Scalar& AKXZ = AK_evolScalCov(1,3) ; // traceless "A" "k"urvature_xz
    const Scalar& AKYZ = AK_evolScalCov(2,3) ; // traceless "A" "k"urvature_yz
    const Scalar& rho = density ; // density cf eq () paper SACRA
    const Scalar& ux = velocity_cov(1) ; // velocity_x cf eq () paper SACRA
    const Scalar& uy = velocity_cov(2) ; // velocity_y cf eq () paper SACRA
    const Scalar& uz = velocity_cov(3) ; // velocity_z cf eq () paper SACRA



    const Scalar& wa2AKXX = wa2AK_evolScalCov(1,1) ;
    const Scalar& wa2AKYY = wa2AK_evolScalCov(2,2) ;
    const Scalar& wa2AKZZ = wa2AK_evolScalCov(3,3) ;
    const Scalar& wa2AKXY = wa2AK_evolScalCov(1,2) ;
    const Scalar& wa2AKXZ = wa2AK_evolScalCov(1,3) ;
    const Scalar& wa2AKYZ = wa2AK_evolScalCov(2,3) ;


    std::vector<const Scalar*> field({&ALPHA, &BX, &BY, &BZ, &phitmp, &EK, &AKXX, &AKYY, &AKZZ, &AKXY, &AKXZ, &AKYZ, &rho, &ux, &uy, &uz}) ;
    std::vector<std::string> field_name({"ALPHA", "BX", "BY", "BZ", "phitmp", "EK", "AKXX", "AKYY", "AKZZ", "AKXY", "AKXZ", "AKYZ", "rho", "ux", "uy", "uz"}) ;
    std::vector<const Scalar*> field_for_extrap({&ALPHA, &BX, &BY, &BZ, &phitmp, &EK, &wa2AKXX, &wa2AKYY, &wa2AKZZ, &wa2AKXY, &wa2AKXZ, &wa2AKYZ, &rho, &ux, &uy, &uz}) ; //ONLY THE wa2AK ARE USEFUL







    // PATHS AND NAMES OF OUTPUT FILES
    std::string output_dir_path = argv[2];
    int const output_dir_path_length(output_dir_path.length()) ;
    if(output_dir_path[output_dir_path_length-1] != '/')
    {
        output_dir_path += '/' ;
    }
    if(rank==0)
    {
        cout << "output_dir_path = " << output_dir_path << endl ;
    }


    int const input_file_path_length(input_file_path.length()) ;
    int index_char(input_file_path_length) ;
    while(index_char > 0 && input_file_path[index_char - 1] != '/') //input_file_path.length() is >= 1 since argv[1] not empty
    {
        index_char-- ;
    }

    string partial_output_file_name = input_file_path.substr(index_char, input_file_path_length-5) + "_FUKaID2SACRA_dp" + to_string(depth_max) + "_flv" + to_string(flvmax) + "_ju" + to_string(ju_org) ;
    if(is_cell_centered)
    {
        partial_output_file_name += "_cell-centered" ;
    }
    string partial_output_file_path = output_dir_path + partial_output_file_name ;


    string output_file_path("unaffected") ;
    ofstream fout ;
    int const doubWid(17) ;
    int const doubDec(8) ;
    int const Digit(12) ;
    if(rank==0)
    {
        // output file
        output_file_path = partial_output_file_path + "_top_info.d" ;

        fout.open(output_file_path, ios::out | ios::trunc) ;

        //l190: read(401,"(6X,e17.8e3)") time
        fout << " time=" << format_eWpDe3(time, doubWid, doubDec) << endl ;


        //l203: read(401,"(/ 7X,i2)") lvmax0
        fout << "AMR grid setting" << endl ;
        fout << "lvmax0=" << format_iW(lvmax0, 2) << endl ;
    }



    //l213: do lv=0,lvmax
    int current_depth(-1000) ;
    int p_lv(-1000) ;
    double ts_xd[lvmax0] {0} ;
    double ts_xu[lvmax0] {0} ;
    double ts_yd[lvmax0] {0} ;
    double ts_yu[lvmax0] {0} ;
    double ts_zd[lvmax0] {0} ;
    double ts_zu[lvmax0] {0} ;
    int ts_jc_amr[lvmax0] {-1000} ;
    int ts_kc_amr[lvmax0] {-1000} ;
    int ts_lc_amr[lvmax0] {-1000} ;
    string str_mvpm("unaffected") ;
    // EXTRACT COORDINATES OF THE CENTERS OF THE NS AND THE BH: CF space_bhns.cpp and includes
    Point center_NS(space.get_domain(space.NS)->get_center()) ;
    Point center_BH(space.get_domain(space.BH)->get_center()) ;
    double yp = space.get_domain(space.BH)->get_cart(2)(center_pos);
    double COMx(-bconfig(COM)) ;
    center_NS.set(1) -= COMx ;
    center_BH.set(1) -= COMx ;
    double COMy(-bconfig(COMY)) ;
    center_NS.set(2) -= COMy ;
    center_BH.set(2) -= COMy ;
    Point center_auxi(center_NS) ;
    if(!is_ns_obj1)
    {
      center_auxi = center_BH ; 
    }
    Point origin_coord(3) ;
    origin_coord.set(1) = 0 ;
    origin_coord.set(2) = 0 ;
    origin_coord.set(3) = 0 ;
    double positive_ratio_step(-1) ;
    for(int lv(0) ; lv <= lvmax ; lv++)
    {
        if(lv <= depth_max)
        {
            current_depth = lv ;
        }
        else
        {
            current_depth = lv - (depth_max - flvmax) ;
        }
        ts_dx[lv] = ts_dx[lvmax]*pow(2, depth_max - current_depth) ;
        ts_dy[lv] = ts_dy[lvmax]*pow(2, depth_max - current_depth) ;
        ts_dz[lv] = ts_dz[lvmax]*pow(2, depth_max - current_depth) ;


        if(lv == depth_max + 1)
        {
            p_lv = flvmax ;
        }
        else
        {
            p_lv = lv - 1 ;
        }


        if(lv <= flvmax) // grid centered on origin (0,0,0)
        {
            ts_xd[lv] = (jd_org + 0.5*cell_cent_shift)*ts_dx[lv] ;
            ts_xu[lv] = (ju_org + 0.5*cell_cent_shift)*ts_dx[lv] ;
            ts_yd[lv] = (kd_org + 0.5*cell_cent_shift)*ts_dy[lv] ;
            ts_yu[lv] = (ku_org + 0.5*cell_cent_shift)*ts_dy[lv] ;
            ts_zd[lv] = (ld_org + 0.5*cell_cent_shift)*ts_dz[lv] ;
            ts_zu[lv] = (lu_org + 0.5*cell_cent_shift)*ts_dz[lv] ;
            ts_jc_amr[lv] = 0 ;
            ts_kc_amr[lv] = 0 ;
            ts_lc_amr[lv] = 0 ;
            str_mvpm = "N" ;
        }
        else
        {
            // if(lv > flvmax && lv <= depth_max) // grid centered on grid point of parent's grid closest to center_NS
            // {
                // center_auxi = center_NS ;
            // }
            // else if(lv > depth_max && lv <= lvmax) // grid centered on grid point of parent's grid closest to center_BH
            // {
                // center_auxi = center_BH ;
            // }
            // else
            // {
                // cerr << "Inexistent level" << endl ;
                // return -1;
            // }

            if(lv == depth_max + 1 )
            {
                center_auxi = center_BH ; // center_auxi is initialized to center_NS, so switch its value to center_BH when reaching lv=depth_max+1
	        if(!is_ns_obj1)
		{
		  center_auxi = center_NS ; // center_auxi is initialized to center_BH, so switch its value to center_NS when reaching lv=depth_max+1
		}
                origin_coord.set(1) = 0 ; // origin_coord reset to zero since parent level is flvmax
                origin_coord.set(2) = 0 ;
                origin_coord.set(3) = 0 ;
                if(rank==0)
                {
                    cout << "Rank " << rank << ": lv = " << lv << " so switched center_auxi to center_BH and set origin_coord back to zero" << endl ;
                }
            }


            positive_ratio_step = abs(center_auxi(1) - origin_coord(1) )/ts_dx[p_lv] ;   //at this point, origin_coord still is coordinates of origin of parent level p_lv
            if(rank==0)
            {
                cout << "Rank " << rank << ": center_auxi(1) = " << center_auxi(1) << ", origin_coord(1) = " << origin_coord(1) << ", ts_dx[p_lv] = " << ts_dx[p_lv] << ", so positive_ratio_step = " << positive_ratio_step << ", so " << flush ;
            }
            if( positive_ratio_step - floor(positive_ratio_step) < 0.5 )
            {
                ts_jc_amr[lv] = floor(positive_ratio_step) ;
            }
            else
            {
                ts_jc_amr[lv] = floor(positive_ratio_step) + 1 ;
            }

            if(center_auxi(1) < origin_coord(1) )
            {
                ts_jc_amr[lv] = -ts_jc_amr[lv] ;
            }
            origin_coord.set(1) += ts_jc_amr[lv]*ts_dx[p_lv] ; // update origin_coord to coordinate of origin of current level lv
            // BLOCK ABOVE IS ALSO CORRECT IF CELL-CENTERED
            ts_xd[lv] = origin_coord(1) + (jd_org + 0.5*cell_cent_shift)*ts_dx[lv] ;
            ts_xu[lv] = origin_coord(1) + (ju_org + 0.5*cell_cent_shift)*ts_dx[lv] ;





            positive_ratio_step = abs(center_auxi(2) - origin_coord(2) )/ts_dy[p_lv] ;
            if( positive_ratio_step - floor(positive_ratio_step) < 0.5 )
            {
                ts_kc_amr[lv] = floor(positive_ratio_step) ;
            }
            else
            {
                ts_kc_amr[lv] = floor(positive_ratio_step) + 1 ;
            }

            if(center_auxi(2) < origin_coord(2) )
            {
                ts_kc_amr[lv] = -ts_kc_amr[lv] ;
            }
            origin_coord.set(2) += ts_kc_amr[lv]*ts_dy[p_lv] ; // update origin_coord to coordinate of origin of current level lv
            ts_yd[lv] = origin_coord(2) + (kd_org + 0.5*cell_cent_shift)*ts_dy[lv] ;
            ts_yu[lv] = origin_coord(2) + (ku_org + 0.5*cell_cent_shift)*ts_dy[lv] ;


            positive_ratio_step = abs(center_auxi(3) - origin_coord(3) )/ts_dz[p_lv] ;
            if( positive_ratio_step - floor(positive_ratio_step) < 0.5 )
            {
                ts_lc_amr[lv] = floor(positive_ratio_step) ;
            }
            else
            {
                ts_lc_amr[lv] = floor(positive_ratio_step) + 1 ;
            }

            if(center_auxi(3) < origin_coord(3) )
            {
                ts_lc_amr[lv] = -ts_lc_amr[lv] ;
            }
            origin_coord.set(3) += ts_lc_amr[lv]*ts_dz[p_lv] ; // update origin_coord to coordinate of origin of current level lv
            ts_zd[lv] = origin_coord(3) + (ld_org + 0.5*cell_cent_shift)*ts_dz[lv] ;
            ts_zu[lv] = origin_coord(3) + (lu_org + 0.5*cell_cent_shift)*ts_dz[lv] ;


            str_mvpm = "Y" ;
        }

        if(rank==0)
        {
            cout << "Rank " << rank << " writing top info for level " << lv << " in output_file_path = " << output_file_path << endl << endl << endl ;

            //l220: read(401,"(4X,i3,8X,i3,7X,i3)") lv_t,ts%lv_depth,ts%lv_parent
            fout << " lv=" << format_iW(lv, 3) << "  depth=" << format_iW(current_depth, 3) << "  p_lv=" << format_iW(p_lv, 3) << endl ;
            //l221: if( trim(adjustl(Digit)) == "8" )then
            if(Digit == 8)
            {
                //l222: read(401,"(5X,2e17.8e3,2X,2i7)") ts%xd,ts%xu,jd_org,ju_org
                fout << "   x:" << format_eWpDe3(ts_xd[lv], 17, 8) << format_eWpDe3(ts_xu[lv], 17, 8) << "  " << format_iW(jd_org, 7) << format_iW(ju_org, 7) << endl ;
                //l223: read(401,"(5X,2e17.8e3,2X,2i7)") ts%yd,ts%yu,kd_org,ku_org
                fout << "   y:" << format_eWpDe3(ts_yd[lv], 17, 8) << format_eWpDe3(ts_yu[lv], 17, 8) << "  " << format_iW(kd_org, 7) << format_iW(ku_org, 7) << endl ;
                //l224: read(401,"(5X,2e17.8e3,2X,2i7)") ts%zd,ts%zu,ld_org,lu_org
                fout << "   z:" << format_eWpDe3(ts_zd[lv], 17, 8) << format_eWpDe3(ts_zu[lv], 17, 8) << "  " << format_iW(ld_org, 7) << format_iW(lu_org, 7) << endl ;
            }
            //l225: else if( trim(adjustl(Digit)) == "12" )then
            else if(Digit == 12)
            {
                //l226: read(401,"(5X,2e21.12e3,2X,2i7)") ts%xd,ts%xu,jd_org,ju_org
                fout << "   x:" << format_eWpDe3(ts_xd[lv], 21, 12) << format_eWpDe3(ts_xu[lv], 21, 12) << "  " << format_iW(jd_org, 7) << format_iW(ju_org, 7) << endl ;
                //l227: read(401,"(5X,2e21.12e3,2X,2i7)") ts%yd,ts%yu,kd_org,ku_org
                fout << "   y:" << format_eWpDe3(ts_yd[lv], 21, 12) << format_eWpDe3(ts_yu[lv], 21, 12) << "  " << format_iW(kd_org, 7) << format_iW(ku_org, 7) << endl ;
                //l228: read(401,"(5X,2e21.12e3,2X,2i7)") ts%zd,ts%zu,ld_org,lu_org
                fout << "   z:" << format_eWpDe3(ts_zd[lv], 21, 12) << format_eWpDe3(ts_zu[lv], 21, 12) << "  " << format_iW(ld_org, 7) << format_iW(lu_org, 7) << endl ;
            }
            //l229: endif

            //l230: read(401,"(8X,i3,8X,i3,8X,i3)") ts%jc_amr,ts%kc_amr,ts%lc_amr
            fout << " jc_amr=" << format_iW(ts_jc_amr[lv], 3) << " kc_amr=" << format_iW(ts_kc_amr[lv], 3) << " lc_amr=" << format_iW(ts_lc_amr[lv], 3) << endl ;
            //l231: read(401,"(6X,a1)") ts%mvpm
            fout << " mvpm=" + str_mvpm << endl ;
        }
    } //l412: end do


    if(rank==0)
    {

        //l663: read(401,"(11X)")
        fout << "metric data" << endl ;
        fout.close() ;
    }




    int remain(-1000), quotient(-1000) ;
    int remain_lg(-1000), quotient_lg(-1000) ;
    int first_level(-1), last_level(-1) ;
    int first_lg(-1000), last_lg(-1000) ;
    int smallest_rank_on_same_level(-1000) ;


    if(nb_procs <= lvmax0)
    {
        first_lg = ld_org-6 ;
        last_lg = lu_org+6 ;

        // USE remain AND quotient TO DISPATCH THE RIGHT LEVELS TO PROC rank
        remain = lvmax0%nb_procs ;
        quotient = floor(lvmax0/nb_procs) ; // = (lvmax0-remain)/nb_procs
        // THE RANKS FROM 0 TO remain-1 WRITE DATA FOR quotient+1 LEVELS, AND THE REMAINING RANKS FOR quotient LEVELS
        if(rank < remain)
        {
            first_level = rank*(quotient+1) ;
            last_level = first_level + quotient ; // = (rank+1)*(quotient+1) - 1
        }
        else
        {
            first_level = remain*(quotient+1) + (rank-remain)*quotient ;
            last_level = first_level + quotient - 1 ; // = remain*(quotient+1) + (rank-remain+1)*quotient - 1
        }
    }
    else
    {
        // IN THIS CASE, USE remain AND quotient TO FIND THE ONLY LEVEL THAT PROC rank HAS TO CONSIDER
        remain = nb_procs%lvmax0 ;
        quotient = floor(nb_procs/lvmax0) ; // = (nb_procs-remain)/lvmax0
        // EACH LEVEL FROM 0 TO remain-1 IS WRITTEN BY quotient+1 PROCS, AND EACH REMAINING LEVEL BY quotient PROCS
        if(rank < remain*(quotient+1))
        {
            first_level = floor(rank/(quotient + 1)) ;
            smallest_rank_on_same_level = first_level*(quotient + 1) ;

            // NOW USE remain_lg AND quotient_lg TO DISPATCH THE RIGHT RANGE OF lg THAT PROC rank HAS TO CONSIDER
            remain_lg = total_nb_lg%(quotient + 1) ;
            quotient_lg = floor(total_nb_lg/(quotient + 1)) ;
        }
        else
        {
            first_level = remain + floor( (rank-remain*(quotient+1) )/quotient );
            smallest_rank_on_same_level = remain*(quotient+1) + (first_level-remain)*quotient ;

            // NOW USE remain_lg AND quotient_lg TO DISPATCH THE RIGHT RANGE OF lg THAT PROC rank HAS TO CONSIDER
            remain_lg = total_nb_lg%quotient ;
            quotient_lg = floor(total_nb_lg/quotient) ;
        }
        last_level = first_level ;


        // RANKS FROM smallest_rank_on_same_level TO smallest_rank_on_same_level+remain_lg-1 WRITE A RANGE OF quotient_lg+1 VALUES OF lg, AND THE REMAINING RANKS ONLY WRITE A RANGE OF quotient_lg VALUES OF lg
        if(rank - smallest_rank_on_same_level < remain_lg)
        {
            first_lg = ld_org - 6 + (rank - smallest_rank_on_same_level) * (quotient_lg+1) ;
            last_lg = first_lg + quotient_lg ;
        }
        else
        {
            first_lg = ld_org - 6 + remain_lg * (quotient_lg+1) + (rank - smallest_rank_on_same_level - remain_lg) * quotient_lg ;
            last_lg = first_lg + quotient_lg - 1 ;
        }
    }






    Point current_point(3) ;



    // setup for filling in BH with junk // BASED ON export_bhns.cpp
    double xxp(-1) ;
    double yyp(-1) ;
    double r_plus(-1) ;

    Index I2(space.get_domain(space.BH+2)->get_radius().get_conf().get_dimensions());
    double rbh(space.get_domain(space.BH+2)->get_radius()(I2));

    double theta(-1) ;
    double phi_angle(-1) ;

    int interp_order(8) ; // 8th order is suggested by Tootle (cf mail July 13th 2022)
    double interpolation_offset(1e-6) ; // no offset is suggested by Tootle (cf mail July 13th 2022); we put a very small one for safety
    double delta_r_rel(0.3) ; // relative spacing of 0.3 is suggested by Tootle (cf mail July 13th 2022)
    std::vector<double> r_points_1side(interp_order);
    for (int j = 0; j < interp_order; j++)
    {
        r_points_1side[j] = (1. + interpolation_offset) * (1. + j * delta_r_rel) * rbh;
    }

    // ALTERNATIVE METHOD THAT USES BOTH SIDES OF HORIZON TO MAKE FIELDS SMOOTHER (LESS STIFF) NEAR CENTER BLACK HOLE
    std::vector<double> r_points_2sides(interp_order);
    std::vector<double> r_points_2sides_center(interp_order+1);
    int half_interp_order(4) ;
    for (int j(0); j < half_interp_order ; j++) {
        r_points_2sides[j]                          =   (1. + interpolation_offset) * (1. + j * delta_r_rel) * rbh; // IT DOES NOT MATTER TO STORE THEM IN ANY CERTAIN ORDER
        r_points_2sides[j+half_interp_order]        = - (1. + interpolation_offset) * (1. + j * delta_r_rel) * rbh; // IT DOES NOT MATTER TO STORE THEM IN ANY CERTAIN ORDER
        r_points_2sides_center[j]                   =   (1. + interpolation_offset) * (1. + j * delta_r_rel) * rbh; // IT DOES NOT MATTER TO STORE THEM IN ANY CERTAIN ORDER
        r_points_2sides_center[j+half_interp_order] = - (1. + interpolation_offset) * (1. + j * delta_r_rel) * rbh; // IT DOES NOT MATTER TO STORE THEM IN ANY CERTAIN ORDER
    }
    r_points_2sides_center[interp_order] = 0 ;

    std::vector<double> vals(interp_order);
    std::vector<double> vals_center(interp_order+1);

    string extrap_type("all1side") ; //or "all2sides", "all2sides_through_wa_wa2AK", or "all2sides_through_wa_wa2AK_0center"

    double current_wa(0) ;





















    //l665: do lv=0,lvmax
    // sequential version: for(int lv(0) ; lv <= lvmax ; lv++)
    for(int lv(first_level) ; lv <= last_level ; lv++)
    {
        // output file
        output_file_path = partial_output_file_path + "_fields_data_lv" + to_string(lv) + "_lg" + to_string(first_lg) + "to" + to_string(last_lg) + ".d" ;
        fout.open(output_file_path, ios::out | ios::trunc) ;



        //l697: do lg=ld_org-6,lu_org+6
        //sequential version: for(int lg(ld_org-6) ; lg <= lu_org+6 ; lg++) // "g"lobal indices on current level
        for(int lg(first_lg) ; lg <= last_lg ; lg++) // "g"lobal indices on current level
        {
             //l693: do l=ld5g,lu5g
                //l694: ts%z(l)=ts%zd +ts%dz*dble(l-ldg)
             //l695: end do
            cout << "Rank " << rank << " currently writing fields data at level " << lv << " in output_file_path = " << output_file_path << " at lg = " << lg << ", noting that this rank writes data for level " << first_level << " from lg = " << first_lg << " to lg = " << last_lg << ", both included." << endl ;
            current_point.set(3) = ts_zd[lv] + ts_dz[lv]*(lg-ld_org) ; // used to be "ts_zd[lv] + ts_dz[lv]*(lg-ld_org + 0.5*cell_cent_shift)" when ts_zd did not take cell_cent_shift into account

            //l698: do kg=kd_org-6,ku_org+6
            for(int kg(kd_org-6) ; kg <= ku_org+6 ; kg++) {
                 //l690: do k=kd5g,ku5g
                    //l691: ts%y(k)=ts%yd +ts%dy*dble(k-kdg)
                 //l692: end do
                current_point.set(2) = ts_yd[lv] + ts_dy[lv]*(kg-kd_org) + COMy ; // "+ COMy" TO HAVE ACTUAL CENTER OF MASS AT CENTER OF SACRA GRID (IF ts_yd[lv] + ts_dy[lv]*(kg-kd_org)=0, WE ACTUALLY EXPORT VALUE AT (COM, COMY,0) IN FUKA CONFIG .dat)
                // Above used to be "ts_yd[lv] + ts_dy[lv]*(kg-kd_org + 0.5*cell_cent_shift) + COMy" when ts_yd did not take cell_cent_shift into account

                //l699: do jg=jd_org-6,ju_org+6
                for(int jg(jd_org-6) ; jg <= ju_org+6 ; jg++)
                {
                     //l687: do j=jd5g,ju5g
                        //l688: ts%x(j)=ts%xd +ts%dx*dble(j-jdg)
                     //l689: end do
                    current_point.set(1) = ts_xd[lv] + ts_dx[lv]*(jg-jd_org) + COMx ;
                    // Above used to be "ts_xd[lv] + ts_dx[lv]*(jg-jd_org + 0.5*cell_cent_shift) + COMx" when ts_xd did not take cell_cent_shift into account

                    //l700: !--- global to local  // local indices for each proc
                    //l701: j = - ( myrankj - jproc/2 )*inode + jg + 1 ;
                    //l702: k = - ( myrankk - kproc/2 )*inode + kg + 1 ;
                    //l703: l = - ( myrankl - lproc/2 )*inode + lg + 1 ;
                    //l704: if( j >= jd5 .and. j <= ju5 .and. &
                      //l705: k >= kd5 .and. k <= ku5 .and. &
                      //l706: l >= ld5 .and. l <= lu5 )then // EACH PROC WILL AFFECT THE VALUES READ BELOW TO ITS OWN FIELDS ONLY IF THEY CORRESPOND TO THEIR GRID, OTHERWISE THEY ARE READ BUT AFFECTED EACH TIME TO hoge (TRASH VARIABLE)

                        //l711: if( trim(adjustl(Comp)) == "BHBH")then
                        //~ if(Comp == "BHBH")
                        //~ {


                                    // CF prg6_... TO EXPORT BBH



                            //~ //l712: !  You must set "enforced_const-ed" initial data
                            //~ //l713: read(401,"(22e17.8e3)") ALPHA(l,k,j), &  !  1
                            //~ //l714: &                  BX(l,k,j),BY(l,k,j),BZ(l,k,j), &  !  2-4
                            //~ //l715: &                  phitmp, &  !  5
                            //~ //l716: &                  GXX(l,k,j),GYY(l,k,j),GZZ(l,k,j), &  !  6-8
                            //~ //l717: &                  GXY(l,k,j),GXZ(l,k,j),GYZ(l,k,j), &  !  9-11
                            //~ //l718: &                  EK(l,k,j), &  !  12
                            //~ //l719: &                  AKXX(l,k,j),AKYY(l,k,j),AKZZ(l,k,j), &  !  13-15
                            //~ //l720: &                  AKXY(l,k,j),AKXZ(l,k,j),AKYZ(l,k,j), &  !  16-18
                            //~ //l721: &                  rho,ux,uy,uz  !  19-22


                        //~ }
                        //~ //l736: else if( trim(adjustl(Comp)) == "NSNS" .or. trim(adjustl(Comp)) == "BHNS" .or. trim(adjustl(Comp)) == "NSWD" )then
                        //~ else if(Comp == "NSNS" || Comp == "BHNS" || Comp == "NSWD")
                        //~ {



                            xxp = current_point(1) - xp ;
                            yyp = current_point(2) - yp ;

                            //Radius measurement centered on the BH
                            r_plus = sqrt(xxp * xxp + yyp * yyp + current_point(3) * current_point(3)) ;


                            //l737: !  You must set "enforced_const-ed" initial data
                            if(r_plus > (1. + interpolation_offset) * rbh)
                            {
                                //~ //l738: read(401,"(16e17.8e3)") ALPHA(l,k,j), &  !  1
                                //~ fout << format_eWpDe3(ALPHA.val_point(current_point), doubWid, doubDec) ;
                                //~ //l739: &                  BX(l,k,j),BY(l,k,j),BZ(l,k,j), &  !  2-4
                                //~ fout << format_eWpDe3(BX.val_point(current_point), doubWid, doubDec) ;
                                //~ fout << format_eWpDe3(BY.val_point(current_point), doubWid, doubDec) ;
                                //~ fout << format_eWpDe3(BZ.val_point(current_point), doubWid, doubDec) ;
                                //~ //l740: &                  phitmp, &  !  5
                                //~ fout << format_eWpDe3(phitmp.val_point(current_point), doubWid, doubDec) ;
                                //~ //l741: &                  EK(l,k,j), &  !  6
                                //~ fout << format_eWpDe3(EK.val_point(current_point), doubWid, doubDec) ;
                                //~ //l742: &                  AKXX(l,k,j),AKYY(l,k,j),AKZZ(l,k,j), &  !  7-9
                                //~ fout << format_eWpDe3(AKXX.val_point(current_point), doubWid, doubDec) ;
                                //~ fout << format_eWpDe3(AKYY.val_point(current_point), doubWid, doubDec) ;
                                //~ fout << format_eWpDe3(AKZZ.val_point(current_point), doubWid, doubDec) ;
                                //~ //l743: &                  AKXY(l,k,j),AKXZ(l,k,j),AKYZ(l,k,j), &  !  10-12
                                //~ fout << format_eWpDe3(AKXY.val_point(current_point), doubWid, doubDec) ;
                                //~ fout << format_eWpDe3(AKXZ.val_point(current_point), doubWid, doubDec) ;
                                //~ fout << format_eWpDe3(AKYZ.val_point(current_point), doubWid, doubDec) ;
                                //~ //l744: &                  rho,ux,uy,uz  !  13-16
                                //~ fout << format_eWpDe3(rho.val_point(current_point), doubWid, doubDec) ;
                                //~ fout << format_eWpDe3(ux.val_point(current_point), doubWid, doubDec) ;
                                //~ fout << format_eWpDe3(uy.val_point(current_point), doubWid, doubDec) ;
                                //~ fout << format_eWpDe3(uz.val_point(current_point), doubWid, doubDec) ;
                                //~ fout << endl ;

                                for (int ff(0) ; ff < field.size() ; ff++) {
                                    fout << format_eWpDe3(field[ff]->val_point(current_point), doubWid, doubDec) ;
                                }
                                fout << endl ;
                            }
                            else // FILLING INSIDE HORIZON WITH JUNK, ADAPTED FROM export_bhns.cpp
                            {
                                if(r_plus == 0) r_plus = 1e-14;
                                theta = std::acos(current_point(3) / r_plus);
                                if(xxp == 0) xxp = 1e-14;
                                phi_angle = std::atan2(yyp, xxp); // atan2 is needed here





                                for (int ff(0) ; ff < field.size() ; ff++) {
                                    if(extrap_type=="all1side") {
                                        for (int j(0) ; j < interp_order; j++) {
                                            vals[j] = field[ff]->val_point(point_from_shifted_spherical(r_points_1side[j], theta, phi_angle, xp, yp));
                                        }
                                        fout << format_eWpDe3(lagrange_gen_k(interp_order, r_plus, r_points_1side.data(), vals.data()), doubWid, doubDec) ; // lagrange_gen_k IMPLEMENTED IN export_utils.cpp
                                    } else if (extrap_type=="all2sides") {
                                        for (int j(0) ; j < interp_order; j++) {
                                            vals[j] = field[ff]->val_point(point_from_shifted_spherical(r_points_2sides[j], theta, phi_angle, xp, yp));
                                        }
                                        fout << format_eWpDe3(lagrange_gen_k(interp_order, r_plus, r_points_2sides.data(), vals.data()), doubWid, doubDec) ; // lagrange_gen_k IMPLEMENTED IN export_utils.cpp
                                    } else if (extrap_type=="all2sides_through_wa_wa2AK") {
                                        if(field_name[ff]=="phitmp") {
                                            for (int j(0) ; j < interp_order; j++) {
                                                vals[j] = wa.val_point(point_from_shifted_spherical(r_points_2sides[j], theta, phi_angle, xp, yp));
                                            }
                                            current_wa = lagrange_gen_k(interp_order, r_plus, r_points_2sides.data(), vals.data())  ;
                                            if(current_wa < 1e-14) current_wa = 1e-14 ; // threshold before reaching 0 and negative values for log below
                                            fout << format_eWpDe3(-0.5*log(current_wa), doubWid, doubDec) ;
                                        } else if(field_name[ff].substr(0,2)=="AK") {
                                            for (int j(0) ; j < interp_order; j++) {
                                                vals[j] = field_for_extrap[ff]->val_point(point_from_shifted_spherical(r_points_2sides[j], theta, phi_angle, xp, yp));
                                            }
                                            // USING current_wa FROM EARLIER IN THE LOOP (USING FACT THAT phitmp IS TREATED BEFORE THE AK'S)
                                            fout << format_eWpDe3(lagrange_gen_k(interp_order, r_plus, r_points_2sides.data(), vals.data())    /    (current_wa*current_wa), doubWid, doubDec)   ;
                                        } else {
                                            for (int j(0) ; j < interp_order; j++) {
                                                vals[j] = field[ff]->val_point(point_from_shifted_spherical(r_points_2sides[j], theta, phi_angle, xp, yp));
                                            }
                                            fout << format_eWpDe3(lagrange_gen_k(interp_order, r_plus, r_points_2sides.data(), vals.data()), doubWid, doubDec) ; // lagrange_gen_k IMPLEMENTED IN export_utils.cpp
                                        }
                                    } else if (extrap_type=="all2sides_through_wa_wa2AK_0center") {
                                        if(field_name[ff]=="phitmp") {
                                            for (int j(0) ; j < interp_order; j++) {
                                                vals_center[j] = wa.val_point(point_from_shifted_spherical(r_points_2sides_center[j], theta, phi_angle, xp, yp));
                                            }
                                            vals_center[interp_order] = 0 ;
                                            current_wa = lagrange_gen_k(interp_order+1, r_plus, r_points_2sides_center.data(), vals_center.data()) ;
                                            if(current_wa < 1e-14) current_wa = 1e-14 ; // threshold before reaching 0 and negative values for log below
                                            fout << format_eWpDe3(-0.5*log(current_wa), doubWid, doubDec) ;
                                        } else if(field_name[ff].substr(0,2)=="AK") {
                                            for (int j(0) ; j < interp_order; j++) {
                                                vals[j] = field_for_extrap[ff]->val_point(point_from_shifted_spherical(r_points_2sides[j], theta, phi_angle, xp, yp));
                                            }
                                            // USING current_wa FROM EARLIER IN THE LOOP (USING FACT THAT phitmp IS TREATED BEFORE THE AK'S)
                                            fout << format_eWpDe3(lagrange_gen_k(interp_order, r_plus, r_points_2sides.data(), vals.data())    /    (current_wa*current_wa), doubWid, doubDec)   ;
                                        } else {
                                            for (int j(0) ; j < interp_order; j++) {
                                                vals[j] = field[ff]->val_point(point_from_shifted_spherical(r_points_2sides[j], theta, phi_angle, xp, yp));
                                            }
                                            fout << format_eWpDe3(lagrange_gen_k(interp_order, r_plus, r_points_2sides.data(), vals.data()), doubWid, doubDec) ; // lagrange_gen_k IMPLEMENTED IN export_utils.cpp
                                        }
                                    } else {
                                        cerr << "Unexpected extrap_type" << endl ;
                                        MPI_Abort(MPI_COMM_WORLD, rc);
                                    }
                                }
                                fout << endl ;
                            }
                        //~ }
                        //~ //l759: else if( trim(adjustl(Comp)) == "BH" )then
                        //~ else if(Comp == "BH")
                        //~ {
                            //~ if(rank==0)
                            //~ {
                                //~ cout << "Comp = \"BH\", no field to write." << endl ;
                            //~ }
                        //~ }
                        //l859: endif


                    //l942: else // IE IF CURRENT POINT DOES NOT BELONG TO THE GRID OF CURRENT PROC, JUST READ THE VALUES BY AFFECTING THEM TO THE TRASH VARIABLE hoge
                        //l943: if( trim(adjustl(Comp)) == "BHBH" .or. trim(adjustl(Comp)) == "NSNS" .or. trim(adjustl(Comp)) == "BHNS" .or. trim(adjustl(Comp)) == "NSWD" )&
                            //l944: read(401,"(16e17.8e3)") hoge,hoge,hoge,hoge,hoge,&
                                                    //l945: hoge,hoge,hoge,hoge,hoge,&
                                                    //l946: hoge,hoge,hoge,hoge,hoge,&
                                                    //l947: hoge
                    //l948: endif // of l704


                } //l949: end do // jg
            } //l950: end do // kg
        } //l951: end do // lg

        fout.close() ;

    } //l953: end do // lv





















    if(rank==0)
    {
        // output file
        output_file_path = partial_output_file_path + "_bottom_info.d" ;

        fout.open(output_file_path, ios::out | ios::trunc) ;


	if(is_ns_obj1)
	{
        //l982: read(401,"(7X)")  //should be 9X?
        fout << "BH data" << endl ; // write BH instead of Comp + " data" even if NSNS....
        //l983:   read(401,"(5X,4e17.8e3,i3)") &
        //l984:       BH_1%ahmass, BH_1%x_punc,BH_1%y_punc,BH_1%z_punc, BH_1%lvt
        double BH_1_ahmass(bconfig(MADM    , BCO1)) ;
        double BH_1_x_punc(center_NS(1)) ;
        double BH_1_y_punc(center_NS(2)) ;
        double BH_1_z_punc(center_NS(3)) ;
        double BH_1_lvt(depth_max) ;
        fout << "BH_1:" << format_eWpDe3(BH_1_ahmass, doubWid, doubDec) << format_eWpDe3(BH_1_x_punc, doubWid, doubDec) << format_eWpDe3(BH_1_y_punc, doubWid, doubDec) ;
        fout << format_eWpDe3(BH_1_z_punc, doubWid, doubDec) << format_iW(BH_1_lvt, 3) << endl ;
        //l985:   read(401,"(5X,4e17.8e3,i3)") &
        //l986:       BH_2%ahmass, BH_2%x_punc,BH_2%y_punc,BH_2%z_punc, BH_2%lvt
        double BH_2_ahmass(bconfig(MCH    , BCO2)) ;
        double BH_2_x_punc(center_BH(1)) ;
        double BH_2_y_punc(center_BH(2)) ;
        double BH_2_z_punc(center_BH(3)) ;
        double BH_2_lvt(lvmax) ;
        fout << "BH_2:" << format_eWpDe3(BH_2_ahmass, doubWid, doubDec) << format_eWpDe3(BH_2_x_punc, doubWid, doubDec) << format_eWpDe3(BH_2_y_punc, doubWid, doubDec) ;
        fout << format_eWpDe3(BH_2_z_punc, doubWid, doubDec) << format_iW(BH_2_lvt, 3) << endl ;
	}
	else
	{
	//l982: read(401,"(7X)")  //should be 9X?
        fout << "BH data" << endl ; // write BH instead of Comp + " data" even if NSNS....
        //l983:   read(401,"(5X,4e17.8e3,i3)") &
        //l984:       BH_1%ahmass, BH_1%x_punc,BH_1%y_punc,BH_1%z_punc, BH_1%lvt
        double BH_1_ahmass(bconfig(MCH    , BCO2)) ;
        double BH_1_x_punc(center_BH(1)) ;          
        double BH_1_y_punc(center_BH(2)) ;          
        double BH_1_z_punc(center_BH(3)) ;          
        double BH_1_lvt(depth_max) ;
        fout << "BH_1:" << format_eWpDe3(BH_1_ahmass, doubWid, doubDec) << format_eWpDe3(BH_1_x_punc, doubWid, doubDec) << format_eWpDe3(BH_1_y_punc, doubWid, doubDec) ;
        fout << format_eWpDe3(BH_1_z_punc, doubWid, doubDec) << format_iW(BH_1_lvt, 3) << endl ;
        //l985:   read(401,"(5X,4e17.8e3,i3)") &
        //l986:       BH_2%ahmass, BH_2%x_punc,BH_2%y_punc,BH_2%z_punc, BH_2%lvt
        double BH_2_ahmass(bconfig(MADM    , BCO1)) ;
        double BH_2_x_punc(center_NS(1)) ;	     
        double BH_2_y_punc(center_NS(2)) ;	     
        double BH_2_z_punc(center_NS(3)) ;	     
        double BH_2_lvt(lvmax) ;
        fout << "BH_2:" << format_eWpDe3(BH_2_ahmass, doubWid, doubDec) << format_eWpDe3(BH_2_x_punc, doubWid, doubDec) << format_eWpDe3(BH_2_y_punc, doubWid, doubDec) ;
        fout << format_eWpDe3(BH_2_z_punc, doubWid, doubDec) << format_iW(BH_2_lvt, 3) << endl ;  
	}
        // close(401)
        fout.close() ;




        // bash FILE TO EXECUTE TO CONCATENATE
        output_file_path = output_dir_path + "concatenate_data_files_" + partial_output_file_name + ".sh" ;

        fout.open(output_file_path, ios::out | ios::trunc) ;

        fout << "cat " + partial_output_file_name + "_top_info.d \\" << endl ;

        if(nb_procs <= lvmax0)
        {
            for(int lv(0) ; lv <= lvmax ; lv++)
            {
                fout << partial_output_file_name + "_fields_data_lv" + to_string(lv) + "_lg" + to_string(first_lg) + "to" + to_string(last_lg) + ".d \\" << endl ; // IN THIS CASE, first_lg = ld_org-6 AND last_lg = lu_org+6 FOR ALL rank, INCLUDING rank 0
            }
        }
        else
        {
            for(int lv(0) ; lv <= remain-1 ; lv++)
            {
                remain_lg = total_nb_lg%(quotient + 1) ;
                quotient_lg = floor(total_nb_lg/(quotient + 1)) ;
                for(int proc_on_lv(0) ; proc_on_lv <= remain_lg-1 ; proc_on_lv++)
                {
                    first_lg = ld_org - 6 + proc_on_lv * (quotient_lg+1) ;
                    last_lg = first_lg + quotient_lg ; ;
                    fout << partial_output_file_name + "_fields_data_lv" + to_string(lv) + "_lg" + to_string(first_lg) + "to" + to_string(last_lg) + ".d \\" << endl ;
                }
                for(int proc_on_lv(remain_lg) ; proc_on_lv <= quotient ; proc_on_lv++)
                {
                    first_lg = ld_org - 6 + remain_lg * (quotient_lg+1) + (proc_on_lv - remain_lg) * quotient_lg ;
                    last_lg = first_lg + quotient_lg - 1 ;
                    fout << partial_output_file_name + "_fields_data_lv" + to_string(lv) + "_lg" + to_string(first_lg) + "to" + to_string(last_lg) + ".d \\" << endl ;
                }
            }
            for(int lv(remain) ; lv <= lvmax ; lv++)
            {
                remain_lg = total_nb_lg%quotient ;
                quotient_lg = floor(total_nb_lg/quotient) ;
                for(int proc_on_lv(0) ; proc_on_lv <= remain_lg-1 ; proc_on_lv++)
                {
                    first_lg = ld_org - 6 + proc_on_lv * (quotient_lg+1) ;
                    last_lg = first_lg + quotient_lg ; ;
                    fout << partial_output_file_name + "_fields_data_lv" + to_string(lv) + "_lg" + to_string(first_lg) + "to" + to_string(last_lg) + ".d \\" << endl ;
                }
                for(int proc_on_lv(remain_lg) ; proc_on_lv <= quotient - 1 ; proc_on_lv++)
                {
                    first_lg = ld_org - 6 + remain_lg * (quotient_lg+1) + (proc_on_lv - remain_lg) * quotient_lg ;
                    last_lg = first_lg + quotient_lg - 1 ;
                    fout << partial_output_file_name + "_fields_data_lv" + to_string(lv) + "_lg" + to_string(first_lg) + "to" + to_string(last_lg) + ".d \\" << endl ;
                }
            }
        }


        fout << partial_output_file_name + "_bottom_info.d \\" << endl ;
        fout << "> concatenated_" + partial_output_file_name + ".d" << endl ;
        fout.close() ;
    }


    MPI_Barrier(MPI_COMM_WORLD) ;

    if(rank==0)
    {
        cout << endl << endl << endl << "All data written in the separate files in folder " + output_dir_path + "." << endl ;
        cout << "In this folder, execute chmod 744 concatenate_data_files_" + partial_output_file_name + ".sh" << endl ;
        cout << "Finally exectute ./concatenate_data_files_" + partial_output_file_name + ".sh" << endl ;
    }

    MPI_Finalize() ;
    return EXIT_SUCCESS ;
}




string format_eWpDe3(double val, int Width, int Decim) // " " + "-/another space if positif" + "1 digit before point in scientific notation" + "." + "Decim decimals" + "E" + "+/-" + "3 exponent digits" =
{
    if(!(  (Width==17 && Decim==8) || (Width==21 && Decim==12)  ))
    {
        cerr << "Unexpected format to write double for SACRA" << endl ;
        return "Unexpected format to write double for SACRA";
    }

    stringstream stream_val ;
    stream_val << scientific ;
    stream_val << uppercase ;
    stream_val << setprecision(Decim) ;

    stream_val << val ;
    string string_val (stream_val.str()) ;
    int length_string_val (string_val.length()) ;

    if(val < 0)
    {
        if(length_string_val == Width - 2)
        {
            return " " + string_val.substr(0, Width - 4) + "0" + string_val.substr(Width - 4) ;
        }
        else if(length_string_val == Width - 3)
        {
            return " " + string_val.substr(0, Width - 4) + "00" + string_val.substr(Width - 4) ;
        }
        else if(length_string_val == Width - 1)
        {
            return " " + string_val ;
        }
        else
        {
            cerr << "Unexpected length of the string representing a negative double" ;
            return "Unexpected length of the string representing a negative double" ;
        }
    }
    else
    {
        if(length_string_val == Width - 3)
        {
            return "  " + string_val.substr(0, Width - 5) + "0" + string_val.substr(Width - 5) ;
        }
        else if(length_string_val == Width - 4)
        {
            return "  " + string_val.substr(0, Width - 5) + "00" + string_val.substr(Width - 5) ;
        }
        else if(length_string_val == Width - 2)
        {
            return "  " + string_val ;
        }
        else
        {
            cerr << "Unexpected length of the string representing a positive double" ;
            return "Unexpected length of the string representing a positive double" ;
        }
    }

    cerr << "Unexpected situation; should have left the function earlier." ;
    return "Unexpected situation; should have left the function earlier.";
}



string format_iW(int val, int Width)
{
    stringstream stream_val ;
    stream_val << right ;
    stream_val << setfill(' ') ;
    stream_val << setw(Width) ;
    stream_val << val ;
    return stream_val.str() ;
}



Point point_from_shifted_spherical(double r, double theta, double phi, double shift_x, double shift_y) // MODIFICATION OF point_spherical IN export_utils.cpp
{
    Point abs_coords(3);

    abs_coords.set(1) = r * std::sin(theta) * std::cos(phi) + shift_x;
    abs_coords.set(2) = r * std::sin(theta) * std::sin(phi) + shift_y;
    abs_coords.set(3) = r * std::cos(theta);

    return abs_coords;
}
