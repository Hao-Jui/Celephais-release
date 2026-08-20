/*
 * 2D Coordinate field utilities for polar KADATH solvers
 * For use with Space_adapted_bh_polar and similar 2D axisymmetric spaces
 * Uses SPHERICAL_BASIS: component 1 = r, component 2 = theta
 * Note: Domain 0 (nucleus) is skipped for base setting since no equations are added there
 */

#pragma once
#include "For_Kadath/Kadath_point_h/kadath.hpp"

namespace Kadath
{

    template <typename space_t> class CoordFields2D
    {
        space_t const& space;
        Base_tensor basis;

        // Set standard base for domains 1 to ndom-1, skip nucleus (domain 0)
        void set_base_outer_domains(Scalar& f) const
        {
            const int ndom = space.get_nbr_domains();
            for (int d = 1; d < ndom; d++) {
                f.set_domain(d).std_base();
            }
        }

        void set_base_outer_domains(Vector& v) const
        {
            const int ndom = space.get_nbr_domains();
            for (int d = 1; d < ndom; d++) {
                v.set(1).set_domain(d).std_base();
                v.set(2).set_domain(d).std_base();
            }
        }

      public:
        CoordFields2D(space_t const& space) : space(space), basis(space, CARTESIAN_BASIS) {}

        // r field using domain radius accessor (works in compact domain).
        Scalar radius() const
        {
            const int ndom = space.get_nbr_domains();
            Scalar r(space);
            for (int d = 0; d < ndom; d++) {
                const auto* dom = space.get_domain(d);
                r.set_domain(d) = dom->get_radius();
            }
            r.std_base();
            return r;
        }

        // 1/r field using div_r(1) - proper handling in compact domain
        Scalar inv_r() const
        {
            const int ndom = space.get_nbr_domains();
            Scalar one(space);
            one = 1.;
            set_base_outer_domains(one);

            Scalar invr(space);
            for (int d = 1; d < ndom; d++) { // skip nucleus
                invr.set_domain(d) = space.get_domain(d)->div_r(one(d));
            }
            // Set nucleus to zero or some placeholder (won't be used in equations)
            invr.set_domain(0) = 0.;
            set_base_outer_domains(invr);
            return invr;
        }

        // expmr = exp(-mass_scal * r) * r^2.
        Scalar overexpmr_field(double mass_scal) const
        {
            Scalar r = radius();
            Scalar overexpmr = exp(-mass_scal * r);
            overexpmr.std_base();

            const int last_dom = space.get_nbr_domains() - 1;
            const Dim_array nbr_pts = space.get_domain(last_dom)->get_nbr_points();
            Index pos(nbr_pts);
            for (int j = 0; j < nbr_pts(1); ++j) {
                pos.set(0) = nbr_pts(0) - 1;
                pos.set(1) = j;
                overexpmr.set_domain(last_dom).set(pos) = 0.0;
            }
            overexpmr.std_base();

            return overexpmr;
        }

        // gradr_i = -r^2 * grad(1/r) = r̂ (unit radial vector)
        // At infinity (outermost points), set to (1, 0) in Cartesian basis
        Vector gradr_i() const
        {
            Scalar r = radius();
            Scalar invr = inv_r();
            Vector v = -r * r * invr.grad();

            // Set to unit radial vector at outermost boundary
            const int last_dom = space.get_nbr_domains() - 1;
            const Dim_array nbr_pts = space.get_domain(last_dom)->get_nbr_points();
            Index pos(nbr_pts);
            for (int j = 0; j < nbr_pts(1); ++j) {
                pos.set(0) = nbr_pts(0) - 1;
                pos.set(1) = j;
                v.set(1).set_domain(last_dom).set(pos) = 1.0;
                v.set(2).set_domain(last_dom).set(pos) = 0.0;
            }
            return v;
        }
    };

} // namespace Kadath
