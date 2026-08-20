#pragma once

#include "For_Kadath/Kadath_point_h/kadath.hpp"
#include "For_Kadath/Array/exceptions.hpp"
#include <math.h>
#include <sstream>
#include "For_Kadath/Config/config_bco.hpp"
#include "bh2d_solver_utils.hpp"
#include "For_Kadath/Utilities/Exporters/bco_geometry.hpp"
#include "Apps/Bco_utils/bco_io.hpp"
#include "Apps/Bco_utils/bco_regrid.hpp"
#include "coord_fields_2d.hpp"
#include "For_Kadath/IO/be_file_source.hpp"
#include "For_Kadath/IO/be_file_sink.hpp"

namespace Kadath {

namespace
{
    // Safe import that returns 0 for out-of-bounds points instead of asserting
    // Uses strict tolerance (1e-13) matching absol_to_num's internal check
    void import_safe(Scalar& dest, const Scalar& src)
    {
        dest.set_in_conf();
        dest.allocate_conf();

        const int ndom_src = src.get_nbr_domains();
        const int ndim = dest.get_ndim();
        const double strict_tol = 1e-13; // Same as absol_to_num's default

        for (int dd = 0; dd < dest.get_nbr_domains(); ++dd) {
            const Domain* dom = dest.get_space().get_domain(dd);
            Index pos(dom->get_nbr_points());
            Point xx(ndim);

            do {
                for (int i = 1; i <= ndim; ++i)
                    xx.set(i) = dom->get_cart(i)(pos);

                // Skip outermost radial points in last domain
                if (dd == dest.get_nbr_domains() - 1 && pos(0) == dom->get_nbr_points()(0) - 1) {
                    continue;
                }

                // Find domain containing this point with STRICT tolerance
                // This matches what absol_to_num will check internally
                int found_dom = -1;
                for (int l = ndom_src - 1; l >= 0; --l) {
                    if (src.get_domain(l)->is_in(xx, strict_tol)) {
                        found_dom = l;
                        break;
                    }
                }

                if (found_dom >= 0) {
                    // Point is safely inside - evaluate
                    dest.set_domain(dd).set(pos) = src.val_point(xx);
                } else {
                    // Point not in any source domain with strict tolerance - use 0
                    dest.set_domain(dd).set(pos) = 0.0;
                }
            } while (pos.inc());
        }
    }
} // anonymous namespace

template <typename config_t> int bh_2d_msqi_regrid(config_t& bconfig, const int new_res, std::string outputfile)
{
    std::string in_spacefile = bconfig.space_filename();

    if (!fs::exists(in_spacefile)) {
        std::ostringstream oss;
        oss << "File: " << in_spacefile << " not found.\n\n";
        KADATH_THROW(oss.str());
    }

    using space_t = Space_adapted_bh_polar;
    BeFileSource ff1(in_spacefile);
    space_t old_space(ff1);
    Scalar old_lapse(old_space, ff1);
    Scalar old_metA(old_space, ff1);
    Scalar old_metB(old_space, ff1);
    Scalar old_beta(old_space, ff1);
    Scalar old_varscal(old_space, ff1);

    if ((new_res % 2) == 0) {
        KADATH_THROW("New Resolution is invalid.  Must be odd (9,11,13,etc)");
    }

    std::cout << "Resolution of old space: " << old_space.get_domain(0)->get_nbr_points()(0) << " (r), "
              << old_space.get_domain(0)->get_nbr_points()(1) << " (theta)" << std::endl;

    bconfig.set(BCO_RES) = new_res;

    bconfig.set_filename(outputfile);
    setup_co<BH>(bconfig, true);

    auto new_spacefile = bconfig.space_filename();
    BeFileSource ff2(new_spacefile);
    space_t space(ff2);
    Scalar lapse(space, ff2);
    Scalar metA(space, ff2);
    Scalar metB(space, ff2);
    Scalar beta(space, ff2);
    Scalar varscal(space, ff2);

    const Dim_array res = space.get_domain(0)->get_nbr_points();
    std::cout << "Resolution of new space: " << res(0) << " (r), " << res(1) << " (theta)" << std::endl;

    bco_utils::print_bounds_from_space(old_space);
    bco_utils::print_bounds_from_space(space);

    // needed in some cases, since the interpolation can go crazy
    const int old_outer = old_space.HOMOTHETIC_OUTER;
    const int old_inner = old_space.HOMOTHETIC_INNER;
    const Domain_polar_shell_outer_homothetic* old_outer_homothetic =
        dynamic_cast<const Domain_polar_shell_outer_homothetic*>(old_space.get_domain(old_outer));
    if (old_outer_homothetic == nullptr) {
        std::cerr << "Expected Domain_polar_shell_outer_homothetic for old domain 1.\n";
        KADATH_THROW("Aborted");
    }

    bco_utils::update_adapted_field(old_lapse, old_inner, old_outer, old_outer_homothetic, OUTER_BC);
    bco_utils::update_adapted_field(old_metA, old_inner, old_outer, old_outer_homothetic, OUTER_BC);
    bco_utils::update_adapted_field(old_metB, old_inner, old_outer, old_outer_homothetic, OUTER_BC);

    import_safe(lapse, old_lapse);
    import_safe(metA, old_metA);
    import_safe(metB, old_metB);
    import_safe(beta, old_beta);
    import_safe(varscal, old_varscal);

    for (int d = 0; d < space.HOMOTHETIC_INNER; ++d) {
        lapse.set_domain(d).annule_hard();
        metA.set_domain(d).annule_hard();
        metB.set_domain(d).annule_hard();
        beta.set_domain(d).annule_hard();
        varscal.set_domain(d).annule_hard();
    }

    int ndoms(space.get_nbr_domains());
    const Dim_array nbr_pts = space.get_domain(ndoms - 1)->get_nbr_points();
    const int nr = nbr_pts(0);
    Index pos(nbr_pts);
    for (int j = 0; j < nbr_pts(1); ++j) {
        pos.set(0) = nr - 1;
        pos.set(1) = j;
        metA.set_domain(ndoms - 1).set(pos) = 1.;
        metB.set_domain(ndoms - 1).set(pos) = 1.;
        varscal.set_domain(ndoms - 1).set(pos) = 0.;
    }
    /*
    CoordFields2D<space_t> coord_fields(space);
    Scalar r = coord_fields.radius();

    for (int d = 2; d < ndoms; ++d) {
      const Dim_array nbr_pts = space.get_domain(d)->get_nbr_points();
      if (nbr_pts.get_ndim() < 2) {
        continue;
      }
      std::cout << "Imported fields along equator (domain " << d << "):" << std::endl;
      for (int i = 0; i < nbr_pts(0); ++i) {
        Index pos(nbr_pts);
        pos.set(0) = i;
        pos.set(1) = 1;
        std::cout << "  r=" << std::setw(10) << r(d)(pos)
                  << std::setw(12) << lapse.set_domain(d)(pos)
                  << std::setw(12) << metA.set_domain(d)(pos)
                  << std::setw(12) << metB.set_domain(d)(pos)
                  << std::setw(12) << beta.set_domain(d)(pos)
                  << std::setw(12) << varscal.set_domain(d)(pos)
                  << std::endl;
      }
    }
    */
    lapse.std_base();
    metA.std_base();
    metB.std_base();
    beta.std_base();
    varscal.std_base();

    bco_utils::save_to_file(space, bconfig, lapse, metA, metB, beta, varscal);
    return EXIT_SUCCESS;
}

} // namespace Kadath
