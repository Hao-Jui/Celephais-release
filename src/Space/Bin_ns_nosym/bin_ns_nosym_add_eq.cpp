/*
    Copyright 2017 Philippe Grandclement

    This file is part of Kadath.

    Kadath is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    Kadath is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with Kadath.  If not, see <http://www.gnu.org/licenses/>.
*/

#include "For_Kadath/Array/headcpp.hpp"
#include "For_Kadath/Array/exceptions.hpp"
#include "For_Kadath/Space/bin_ns_nosym.hpp"
#include "For_Kadath/Utilities/utilities.hpp"
#include "For_Kadath/Array/point.hpp"
#include "For_Kadath/Scalar/scalar.hpp"
#include "For_Kadath/System_of_eqs/system_of_eqs.hpp"
#include "For_Kadath/Utilities/name_tools.hpp"
#include "../Shared/binary_co_add_eq_common.hpp"

namespace Kadath
{

    void Space_bin_ns_nosym::add_bc_sphere_one(System_of_eqs& sys, const char* name, int nused, Array<int>** pused)
    {
        sys.add_eq_bc(ADAPTED1 + 1, INNER_BC, name, nused, pused);
    }

    void Space_bin_ns_nosym::add_bc_sphere_two(System_of_eqs& sys, const char* name, int nused, Array<int>** pused)
    {
        sys.add_eq_bc(ADAPTED2 + 1, INNER_BC, name, nused, pused);
    }

    void Space_bin_ns_nosym::add_bc_outer(System_of_eqs& sys, const char* name, int nused, Array<int>** pused)
    {
        sys.add_eq_bc(OUTER + n_shells_outer + 5, OUTER_BC, name, nused, pused);
    }

    // Star-side block. The post-adapted outer shells come AFTER the adapted pair
    // (ADAPTED+2 .. ADAPTED+1+n_shells), IDENTICAL to the sym Space_bin_ns —
    // construction (gen_ns_domains / read_adapted_star_domains, and the index
    // arithmetic NS2 = ADAPTED1 + 2 + n_shells1) places them there; there is no
    // nosym/sym ordering difference for this matching topology.
    void Space_bin_ns_nosym::add_eq(System_of_eqs& sys, const char* eq, const char* rac, const char* rac_der, int nused,
                              Array<int>** pused)
    {
        // A fixed-radius spherical<->spherical seam between domain `inner` (outer
        // face) and `inner+1` (inner face). When the two share their angular grid
        // (ntheta, nphi) the historical conforming tau-matching is used -- this is
        // bit-identical to the pre-p-refinement path, so a uniform-resolution BNS
        // is unchanged. Only when the angular counts differ (per-domain angular
        // refinement) is the seam split across the two faces with import matching,
        // which is coefficient-counted (tau space, like the bulk) and so keeps the
        // system square across the ntheta jump.
        const auto same_angular = [&](int a, int b) {
            const Dim_array& pa = get_domain(a)->get_nbr_points();
            const Dim_array& pb = get_domain(b)->get_nbr_points();
            return pa(1) == pb(1) && pa(2) == pb(2);
        };
        const auto match_radial_seam = [&](int inner) {
            const int outer = inner + 1;
            if (same_angular(inner, outer)) {
                sys.add_eq_matching(inner, OUTER_BC, rac, nused, pused, rac);
                sys.add_eq_matching(inner, OUTER_BC, rac_der, nused, pused, rac);
            } else {
                sys.add_eq_matching_import(inner, OUTER_BC, rac, nused, pused, rac, true);
                sys.add_eq_matching_import(outer, INNER_BC, rac_der, nused, pused, rac, true);
            }
        };

        // First NS — nucleus<->adapted-outer seam (fixed radius rin).
        sys.add_eq_inside(NS1, eq, nused, pused, rac);
        match_radial_seam(NS1);

        sys.add_eq_inside(ADAPTED1, eq, nused, pused, rac);
        // ADAPTED1.OUTER <-> (ADAPTED1+1).INNER is the deformable star surface:
        // it stays conforming because both adapted domains share the same surface
        // object (and hence the same ntheta/nphi by construction).
        sys.add_eq_matching(ADAPTED1, OUTER_BC, rac, nused, pused, rac);
        sys.add_eq_matching(ADAPTED1, OUTER_BC, rac_der, nused, pused, rac);
        sys.add_eq_inside(ADAPTED1 + 1, eq, nused, pused, rac);

        // Post-adapted outer shells (fixed radius), shells AFTER the adapted pair.
        for (int i = 0; i < n_shells1; ++i) {
            match_radial_seam(ADAPTED1 + 1 + i);
            sys.add_eq_inside(ADAPTED1 + 2 + i, eq, nused, pused, rac);
        }

        // Matching with bispheric (last star domain before the bispheric region).
        sys.add_eq_matching_import(ADAPTED1 + 1 + n_shells1, OUTER_BC, rac, nused, pused, rac, true);
        sys.add_eq_matching_import(OUTER, INNER_BC, rac_der, nused, pused, rac, true);
        sys.add_eq_matching_import(OUTER + 1, INNER_BC, rac_der, nused, pused, rac, true);

        // Second NS — same pattern as star 1.
        sys.add_eq_inside(NS2, eq, nused, pused, rac);
        match_radial_seam(NS2);

        sys.add_eq_inside(ADAPTED2, eq, nused, pused, rac);
        sys.add_eq_matching(ADAPTED2, OUTER_BC, rac, nused, pused, rac);
        sys.add_eq_matching(ADAPTED2, OUTER_BC, rac_der, nused, pused, rac);
        sys.add_eq_inside(ADAPTED2 + 1, eq, nused, pused, rac);

        for (int i = 0; i < n_shells2; ++i) {
            match_radial_seam(ADAPTED2 + 1 + i);
            sys.add_eq_inside(ADAPTED2 + 2 + i, eq, nused, pused, rac);
        }

        // Matching with bispheric :
        sys.add_eq_matching_import(ADAPTED2 + 1 + n_shells2, OUTER_BC, rac, nused, pused, rac, true);
        sys.add_eq_matching_import(OUTER + 3, INNER_BC, rac_der, nused, pused, rac, true);
        sys.add_eq_matching_import(OUTER + 4, INNER_BC, rac_der, nused, pused, rac, true);

        BinaryCoSpaceEquations::add_bispheric_and_outer_wiring(
            *this, sys, eq, rac, rac_der, nused, pused, true);
    }

    void Space_bin_ns_nosym::add_eq_nozec(System_of_eqs& sys, const char* eq, const char* rac, const char* rac_der, int nused,
                                    Array<int>** pused)
    {
        BinaryCoSpaceEquations::bin_ns_add_eq_nozec(*this, sys, eq, rac, rac_der, nused, pused);
    }

    void Space_bin_ns_nosym::add_eq_nozec(System_of_eqs& sys, const char* eq, const char* rac, const char* rac_der,
                                    const List_comp& used)
    {
        add_eq_nozec(sys, eq, rac, rac_der, used.get_ncomp(), used.get_pcomp());
    }

    // Nosym-specific (star blocks indexed through NS + n_shells; the sym class
    // uses the ADAPTED-based layout on purpose).
    void Space_bin_ns_nosym::add_eq_noshell(System_of_eqs& sys, const char* eq, const char* rac, const char* rac_der,
                                      int nused, Array<int>** pused)
    {

        // First NS
        sys.add_eq_inside(NS1, eq, nused, pused, rac);
        sys.add_eq_matching(NS1, OUTER_BC, rac, nused, pused, rac);
        sys.add_eq_matching(NS1, OUTER_BC, rac_der, nused, pused, rac);

        for (int i = 0; i < n_shells1; ++i) {
            sys.add_eq_inside(NS1 + 1 + i, eq, nused, pused, rac);
            sys.add_eq_matching(NS1 + 1 + i, OUTER_BC, rac, nused, pused, rac);
            sys.add_eq_matching(NS1 + 1 + i, OUTER_BC, rac_der, nused, pused, rac);
        }

        sys.add_eq_inside(NS1 + n_shells1 + 1, eq, nused, pused, rac);
        sys.add_eq_matching(NS1 + n_shells1 + 1, OUTER_BC, rac, nused, pused, rac);
        sys.add_eq_matching(NS1 + n_shells1 + 1, OUTER_BC, rac_der, nused, pused, rac);
        sys.add_eq_inside(NS1 + n_shells1 + 2, eq, nused, pused, rac);

        // Matching with bispheric :
        sys.add_eq_matching_import(NS1 + n_shells1 + 2, OUTER_BC, rac, nused, pused, rac, true);
        sys.add_eq_matching_import(OUTER, INNER_BC, rac_der, nused, pused, rac, true);
        sys.add_eq_matching_import(OUTER + 1, INNER_BC, rac_der, nused, pused, rac, true);

        // Second NS :
        sys.add_eq_inside(NS2, eq, nused, pused, rac);
        sys.add_eq_matching(NS2, OUTER_BC, rac, nused, pused, rac);
        sys.add_eq_matching(NS2, OUTER_BC, rac_der, nused, pused, rac);

        for (int i = 0; i < n_shells2; ++i) {
            sys.add_eq_inside(NS2 + 1 + i, eq, nused, pused, rac);
            sys.add_eq_matching(NS2 + 1 + i, OUTER_BC, rac, nused, pused, rac);
            sys.add_eq_matching(NS2 + 1 + i, OUTER_BC, rac_der, nused, pused, rac);
        }

        sys.add_eq_inside(NS2 + n_shells2 + 1, eq, nused, pused, rac);
        sys.add_eq_matching(NS2 + n_shells2 + 1, OUTER_BC, rac, nused, pused, rac);
        sys.add_eq_matching(NS2 + n_shells2 + 1, OUTER_BC, rac_der, nused, pused, rac);
        sys.add_eq_inside(NS2 + n_shells2 + 2, eq, nused, pused, rac);

        // Matching with bispheric :
        sys.add_eq_matching_import(NS2 + n_shells2 + 2, OUTER_BC, rac, nused, pused, rac, true);
        sys.add_eq_matching_import(OUTER + 3, INNER_BC, rac_der, nused, pused, rac, true);
        sys.add_eq_matching_import(OUTER + 4, INNER_BC, rac_der, nused, pused, rac, true);

        // Chi first
        sys.add_eq_inside(OUTER, eq, nused, pused, rac);
        sys.add_eq_matching(OUTER, CHI_ONE_BC, rac, nused, pused, rac);
        sys.add_eq_matching(OUTER, CHI_ONE_BC, rac_der, nused, pused, rac);

        // Rect :
        sys.add_eq_inside(OUTER + 1, eq, nused, pused, rac);
        sys.add_eq_matching(OUTER + 1, ETA_PLUS_BC, rac, nused, pused, rac);
        sys.add_eq_matching(OUTER + 1, ETA_PLUS_BC, rac_der, nused, pused, rac);

        // Eta first
        sys.add_eq_inside(OUTER + 2, eq, nused, pused, rac);
        sys.add_eq_matching(OUTER + 2, ETA_PLUS_BC, rac, nused, pused, rac);
        sys.add_eq_matching(OUTER + 2, ETA_PLUS_BC, rac_der, nused, pused, rac);

        // Rect
        sys.add_eq_inside(OUTER + 3, eq, nused, pused, rac);
        sys.add_eq_matching(OUTER + 3, CHI_ONE_BC, rac, nused, pused, rac);
        sys.add_eq_matching(OUTER + 3, CHI_ONE_BC, rac_der, nused, pused, rac);

        // chi first :
        sys.add_eq_inside(OUTER + 4, eq, nused, pused, rac);
    }

    void Space_bin_ns_nosym::add_eq_noshell(System_of_eqs& sys, const char* eq, const char* rac, const char* rac_der,
                                      const List_comp& used)
    {
        add_eq_noshell(sys, eq, rac, rac_der, used.get_ncomp(), used.get_pcomp());
    }

    void Space_bin_ns_nosym::add_eq_int_inf(System_of_eqs& sys, const char* nom)
    {
        BinaryCoSpaceEquations::add_eq_int_inf<Domain_compact_nosym>(*this, sys, nom);
    }

    void Space_bin_ns_nosym::add_eq_int_volume(System_of_eqs& sys, int dmin, int dmax, const char* nom)
    {
        BinaryCoSpaceEquations::add_eq_int_volume(sys, dmin, dmax, nom);
    }

    void Space_bin_ns_nosym::add_eq_ori_one(System_of_eqs& sys, const char* name)
    {
        BinaryCoSpaceEquations::add_eq_val_origin(*this, sys, NS1, name);
    }

    void Space_bin_ns_nosym::add_eq_ori_two(System_of_eqs& sys, const char* name)
    {
        BinaryCoSpaceEquations::add_eq_val_origin(*this, sys, NS2, name);
    }

    void Space_bin_ns_nosym::add_eq_int_outer_sphere_one(System_of_eqs& sys, const char* nom)
    {
        BinaryCoSpaceEquations::add_eq_int_bc(sys, ADAPTED1 + 1, OUTER_BC, nom);
    }

    void Space_bin_ns_nosym::add_eq_int_outer_sphere_two(System_of_eqs& sys, const char* nom)
    {
        BinaryCoSpaceEquations::add_eq_int_bc(sys, ADAPTED2 + 1, OUTER_BC, nom);
    }

} // namespace Kadath
