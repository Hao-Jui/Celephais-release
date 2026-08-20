#pragma once
/*
    Shared bodies of the binary compact-object Space add_eq* member functions.

    The sym/nosym space pairs (Space_bin_ns / Space_bin_ns_nosym,
    Space_bhns / Space_bhns_nosym, Space_adapted_bh / Space_adapted_bh_nosym)
    carried byte-identical equation-registration bodies differing only in the
    class name and the compactified-domain type used for the sanity cast
    (Domain_compact vs Domain_compact_nosym). Each member function is now a
    thin wrapper around one body here, so an equation-wiring fix is a single
    edit instead of two-to-six.

    Genuinely divergent bodies stay in their class .cpp: Space_bin_ns vs
    Space_bin_ns_nosym differ on purpose in add_eq and the main add_eq_noshell
    overload (nosym processes the inner shells BEFORE the adapted pair and
    matches the bispheric region at ADAPTED + 1 — physics of the asymmetric
    layout, not drift).

    Statement-order micro-variants of the integral registrations are kept as
    separate entry points (record-first vs parse-first) so every class keeps
    its exact pre-existing behavior, including the state left behind on the
    throwing path.

    Private header (lives next to the Space sources, mirrored after
    src/System_of_eqs's coloring_internals.hpp pattern); consumed only by the
    six *_add_eq.cpp translation units.
*/

#include "For_Kadath/Array/exceptions.hpp"
#include "For_Kadath/Array/point.hpp"
#include "For_Kadath/Space/space.hpp"
#include "For_Kadath/System_of_eqs/system_of_eqs.hpp"
#include "For_Kadath/Utilities/name_tools.hpp"

#include <tuple>

namespace Kadath
{

    struct BinaryCoSpaceEquations {

        // Integral equation at infinity, Bin_ns / Bhns statement order: record in
        // eq_int_list first, then check the compactified domain.
        template <typename compact_domain_t>
        static void add_eq_int_inf(const Space& space, System_of_eqs& sys, const char* nom)
        {
            const int nbr_domains = space.get_nbr_domains();
            sys.eq_int_list.push_back(std::make_tuple(nom, nbr_domains - 1, OUTER_BC));

            // Check the last domain is of the right type :
            const compact_domain_t* pcomp = dynamic_cast<const compact_domain_t*>(space.get_domain(nbr_domains - 1));
            if (pcomp == nullptr) {
                KADATH_THROW("add_eq_int_inf requires a compactified domain");
            }
            int dom = nbr_domains - 1;

            add_eq_int_inf_construct(sys, dom, nom);
        }

        // Integral equation at infinity, Adapted_bh statement order: check the
        // compactified domain first, then record.
        template <typename compact_domain_t>
        static void add_eq_int_inf_check_first(const Space& space, System_of_eqs& sys, const char* nom)
        {
            const int nbr_domains = space.get_nbr_domains();

            // Check the last domain is of the right type :
            const compact_domain_t* pcomp = dynamic_cast<const compact_domain_t*>(space.get_domain(nbr_domains - 1));
            if (pcomp == nullptr) {
                KADATH_THROW("add_eq_int_inf requires a compactified domain");
            }
            int dom = nbr_domains - 1;
            sys.eq_int_list.push_back(std::make_tuple(nom, dom, OUTER_BC));

            add_eq_int_inf_construct(sys, dom, nom);
        }

        static void add_eq_int_volume(System_of_eqs& sys, int dmin, int dmax, const char* nom)
        {
            sys.eq_int_list.push_back(std::make_tuple(nom, dmin, -1));

            // Get the lhs and rhs
            char p1[LMAX];
            char p2[LMAX];

            bool indic = sys.is_ope_bin(nom, p1, p2, '=');
            if (!indic) {
                KADATH_THROW("= needed for equations");
            } else {
                // Construction of the equation
                int nz = dmax - dmin + 1;
                sys.ensure_eq_int_slot(); sys.eq_int[sys.neq_int].reset(new Eq_int(nz + 1));

                // Affectation of the intregrale parts
                for (int d = 0; d < nz; d++)
                    sys.eq_int[sys.neq_int]->set_part(d, sys.give_ope(d + dmin, p1));
                // Affectation of the second member (constant value)
                sys.eq_int[sys.neq_int]->set_part(nz, new Ope_minus(&sys, sys.give_ope(dmin, p2)));
                sys.neq_int++;
            }
            sys.nbr_conditions = -1;
        }

        // Pointwise equation at the origin collocation point of domain dom
        // (Bin_ns add_eq_ori_one/two, Bhns add_eq_ori_NS).
        static void add_eq_val_origin(const Space& space, System_of_eqs& sys, int dom, const char* name)
        {
            Index pos(space.get_domain(dom)->get_nbr_points());
            char auxi[LMAX];
            trim_spaces(auxi, name);
            sys.add_eq_val(dom, auxi, pos);
        }

        // Single-domain boundary integral, record-first order (Bin_ns
        // add_eq_int_outer_sphere_one/two, Bhns add_eq_int_outer_NS).
        static void add_eq_int_bc(System_of_eqs& sys, int dom, int bc, const char* nom)
        {
            sys.eq_int_list.push_back(std::make_tuple(nom, dom, bc));

            // Get the lhs and rhs
            char p1[LMAX];
            char p2[LMAX];
            bool indic = sys.is_ope_bin(nom, p1, p2, '=');
            if (!indic) {
                KADATH_THROW("= needed for equations");
            } else {

                // Verif lhs = 0 ?
                indic = ((p2[0] == '0') && (p2[1] == ' ') && (p2[2] == '\0')) ? true : false;

                // Construction of the equation
                sys.ensure_eq_int_slot(); sys.eq_int[sys.neq_int].reset(new Eq_int(1));

                // Affectation :
                // no lhs :
                if (indic) {
                    sys.eq_int[sys.neq_int]->set_part(0, sys.give_ope(dom, p1, bc));
                } else {
                    sys.eq_int[sys.neq_int]->set_part(0, new Ope_sub(&sys, sys.give_ope(dom, p1, bc),
                                                                     sys.give_ope(dom, p2, bc)));
                }
            }
            sys.neq_int++;
            sys.nbr_conditions = -1;
        }

        // Single-domain boundary integral, parse-first order (Bhns add_eq_int_BH,
        // Adapted_bh add_eq_int_bh): nothing is recorded when '=' is missing.
        static void add_eq_int_bc_parse_first(System_of_eqs& sys, int dom, int bc, const char* nom)
        {
            // Get the lhs and rhs
            char p1[LMAX];
            char p2[LMAX];
            bool indic = sys.is_ope_bin(nom, p1, p2, '=');
            if (!indic) {
                KADATH_THROW("= needed for equations");
            } else {
                sys.eq_int_list.push_back(std::make_tuple(nom, dom, bc));

                // Verif lhs = 0 ?
                indic = ((p2[0] == '0') && (p2[1] == ' ') && (p2[2] == '\0')) ? true : false;

                // Construction of the equation
                sys.ensure_eq_int_slot(); sys.eq_int[sys.neq_int].reset(new Eq_int(1));

                // Affectation :
                // no lhs :
                if (indic) {
                    sys.eq_int[sys.neq_int]->set_part(0, sys.give_ope(dom, p1, bc));
                } else {
                    sys.eq_int[sys.neq_int]->set_part(0, new Ope_sub(&sys, sys.give_ope(dom, p1, bc),
                                                                     sys.give_ope(dom, p2, bc)));
                }
            }
            sys.neq_int++;
            sys.nbr_conditions = -1;
        }

        // Zero mode (j, k) of name at infinity (Adapted_bh add_eq_zero_mode_inf).
        static void add_eq_zero_mode_inf(const Space& space, System_of_eqs& sys, const char* name, int j, int k)
        {
            Index pos_cf(space.get_domain(space.get_nbr_domains() - 1)->get_nbr_coefs());
            pos_cf.set(1) = j;
            pos_cf.set(2) = k;
            double value = 0.;
            char auxi[LMAX];
            trim_spaces(auxi, name);
            sys.add_eq_mode(space.get_nbr_domains() - 1, OUTER_BC, auxi, pos_cf, value);
        }

        // Bulk + matching wiring shared by the Bhns pair (the Bin_ns pair diverges
        // here on purpose and keeps per-class bodies).
        template <typename space_t>
        static void bhns_add_eq(const space_t& space, System_of_eqs& sys, const char* eq, const char* rac,
                                const char* rac_der, int nused, Array<int>** pused)
        {
            const auto same_angular = [&](int a, int b) {
                const Dim_array& pa = space.get_domain(a)->get_nbr_points();
                const Dim_array& pb = space.get_domain(b)->get_nbr_points();
                return pa(1) == pb(1) && pa(2) == pb(2);
            };
            const auto match_radial_seam = [&](int inner) {
                const int outer = inner + 1;
                if (same_angular(inner, outer)) {
                    sys.add_eq_matching(inner, OUTER_BC, rac, nused, pused, rac);
                    sys.add_eq_matching(inner, OUTER_BC, rac_der, nused, pused, rac);
                } else {
                    sys.add_eq_matching_import(inner, OUTER_BC, rac, nused, pused, rac);
                    sys.add_eq_matching_import(outer, INNER_BC, rac_der, nused, pused, rac);
                }
            };

            // NS
            sys.add_eq_inside(space.NS, eq, nused, pused, rac);
            match_radial_seam(space.NS);

            sys.add_eq_inside(space.ADAPTEDNS, eq, nused, pused, rac);
            sys.add_eq_matching(space.ADAPTEDNS, OUTER_BC, rac, nused, pused, rac);
            sys.add_eq_matching(space.ADAPTEDNS, OUTER_BC, rac_der, nused, pused, rac);
            sys.add_eq_inside(space.ADAPTEDNS + 1, eq, nused, pused, rac);

            for (int i = 0; i < space.n_shells1; ++i) {
                match_radial_seam(space.ADAPTEDNS + 1 + i);
                sys.add_eq_inside(space.ADAPTEDNS + 2 + i, eq, nused, pused, rac);
            }

            // Matching with bispheric :
            sys.add_eq_matching_import(space.ADAPTEDNS + 1 + space.n_shells1, OUTER_BC, rac, nused, pused, rac);
            sys.add_eq_matching_import(space.OUTER, INNER_BC, rac_der, nused, pused, rac);
            sys.add_eq_matching_import(space.OUTER + 1, INNER_BC, rac_der, nused, pused, rac);

            // BH :
            sys.add_eq_inside(space.ADAPTEDBH + 1, eq, nused, pused, rac);
            for (int i = 0; i < space.n_shells2; ++i) {
                match_radial_seam(space.ADAPTEDBH + 1 + i);
                sys.add_eq_inside(space.ADAPTEDBH + 2 + i, eq, nused, pused, rac);
            }

            // Matching with bispheric :
            sys.add_eq_matching_import(space.ADAPTEDBH + 1 + space.n_shells2, OUTER_BC, rac, nused, pused, rac);
            sys.add_eq_matching_import(space.OUTER + 3, INNER_BC, rac_der, nused, pused, rac);
            sys.add_eq_matching_import(space.OUTER + 4, INNER_BC, rac_der, nused, pused, rac);

            add_bispheric_and_outer_wiring(space, sys, eq, rac, rac_der, nused, pused);
        }

        // Bispheric block + outer-shell / compactified wiring shared verbatim by
        // the Bin_ns and Bhns add_eq tails (chi-first / rect / eta-first / rect /
        // chi-first, outer matching, optional shells, compactified domain).
        template <typename space_t>
        static void add_bispheric_and_outer_wiring(const space_t& space, System_of_eqs& sys, const char* eq,
                                                   const char* rac, const char* rac_der, int nused,
                                                   Array<int>** pused,
                                                   bool reflection_parity_preserving = false)
        {
            // Chi first
            sys.add_eq_inside(space.OUTER, eq, nused, pused, rac);
            sys.add_eq_matching(space.OUTER, CHI_ONE_BC, rac, nused, pused, rac);
            sys.add_eq_matching(space.OUTER, CHI_ONE_BC, rac_der, nused, pused, rac);

            // Rect :
            sys.add_eq_inside(space.OUTER + 1, eq, nused, pused, rac);
            sys.add_eq_matching(space.OUTER + 1, ETA_PLUS_BC, rac, nused, pused, rac);
            sys.add_eq_matching(space.OUTER + 1, ETA_PLUS_BC, rac_der, nused, pused, rac);

            // Eta first
            sys.add_eq_inside(space.OUTER + 2, eq, nused, pused, rac);
            sys.add_eq_matching(space.OUTER + 2, ETA_PLUS_BC, rac, nused, pused, rac);
            sys.add_eq_matching(space.OUTER + 2, ETA_PLUS_BC, rac_der, nused, pused, rac);

            // Rect
            sys.add_eq_inside(space.OUTER + 3, eq, nused, pused, rac);
            sys.add_eq_matching(space.OUTER + 3, CHI_ONE_BC, rac, nused, pused, rac);
            sys.add_eq_matching(space.OUTER + 3, CHI_ONE_BC, rac_der, nused, pused, rac);

            // chi first :
            sys.add_eq_inside(space.OUTER + 4, eq, nused, pused, rac);

            // Matching outer domain :
            for (int d = space.OUTER; d <= space.OUTER + 4; d++)
                sys.add_eq_matching_import(d, OUTER_BC, rac, nused, pused, rac,
                                           reflection_parity_preserving);

            // Matching for first shell or compactified domain
            sys.add_eq_matching_import(space.OUTER + 5, INNER_BC, rac_der,
                                       nused, pused, rac,
                                       reflection_parity_preserving);

            // Optional spherical shells between bi-spherical and compactified domain
            for (int d = 0; d < space.n_shells_outer; d++) {
                sys.add_eq_inside(space.OUTER + 5 + d, eq, nused, pused, rac);
                sys.add_eq_matching(space.OUTER + 5 + d, OUTER_BC, rac, nused, pused, rac);
                sys.add_eq_matching(space.OUTER + 5 + d, OUTER_BC, rac_der, nused, pused, rac);
            }

            // Compactified domain
            sys.add_eq_inside(space.OUTER + 5 + space.n_shells_outer, eq, nused, pused, rac);
        }

        // Bin_ns pair add_eq_nozec main overload (hardcoded shell-free layout).
        template <typename space_t>
        static void bin_ns_add_eq_nozec(const space_t& space, System_of_eqs& sys, const char* eq, const char* rac,
                                        const char* rac_der, int nused, Array<int>** pused)
        {
            // FIXME not fixed yet

            // First NS
            sys.add_eq_inside(0, eq, nused, pused, rac);
            sys.add_eq_matching(0, OUTER_BC, rac, nused, pused, rac);
            sys.add_eq_matching(0, OUTER_BC, rac_der, nused, pused, rac);
            sys.add_eq_inside(1, eq, nused, pused, rac);
            sys.add_eq_matching(1, OUTER_BC, rac, nused, pused, rac);
            sys.add_eq_matching(1, OUTER_BC, rac_der, nused, pused, rac);
            sys.add_eq_inside(2, eq, nused, pused, rac);

            // Matching with bispheric :
            sys.add_eq_matching_import(2, OUTER_BC, rac, nused, pused, rac);
            sys.add_eq_matching_import(6, INNER_BC, rac_der, nused, pused, rac);
            sys.add_eq_matching_import(7, INNER_BC, rac_der, nused, pused, rac);

            // Second NS :
            sys.add_eq_inside(3, eq, nused, pused, rac);
            sys.add_eq_matching(3, OUTER_BC, rac, nused, pused, rac);
            sys.add_eq_matching(3, OUTER_BC, rac_der, nused, pused, rac);
            sys.add_eq_inside(4, eq, nused, pused, rac);
            sys.add_eq_matching(4, OUTER_BC, rac, nused, pused, rac);
            sys.add_eq_matching(4, OUTER_BC, rac_der, nused, pused, rac);
            sys.add_eq_inside(5, eq, nused, pused, rac);

            // Matching with bispheric :
            sys.add_eq_matching_import(5, OUTER_BC, rac, nused, pused, rac);
            sys.add_eq_matching_import(9, INNER_BC, rac_der, nused, pused, rac);
            sys.add_eq_matching_import(10, INNER_BC, rac_der, nused, pused, rac);

            // Chi first
            sys.add_eq_inside(6, eq, nused, pused, rac);
            sys.add_eq_matching(6, CHI_ONE_BC, rac, nused, pused, rac);
            sys.add_eq_matching(6, CHI_ONE_BC, rac_der, nused, pused, rac);

            // Rect :
            sys.add_eq_inside(7, eq, nused, pused, rac);
            sys.add_eq_matching(7, ETA_PLUS_BC, rac, nused, pused, rac);
            sys.add_eq_matching(7, ETA_PLUS_BC, rac_der, nused, pused, rac);

            // Eta first
            sys.add_eq_inside(8, eq, nused, pused, rac);
            sys.add_eq_matching(8, ETA_PLUS_BC, rac, nused, pused, rac);
            sys.add_eq_matching(8, ETA_PLUS_BC, rac_der, nused, pused, rac);

            // Rect
            sys.add_eq_inside(9, eq, nused, pused, rac);
            sys.add_eq_matching(9, CHI_ONE_BC, rac, nused, pused, rac);
            sys.add_eq_matching(9, CHI_ONE_BC, rac_der, nused, pused, rac);

            // chi first :
            sys.add_eq_inside(10, eq, nused, pused, rac);

            if (space.n_shells_outer != 0) {
                // Matching outer domain :
                for (int d = 6; d <= 10; d++)
                    sys.add_eq_matching_import(d, OUTER_BC, rac, nused, pused, rac);
                sys.add_eq_matching_import(11, INNER_BC, rac_der, nused, pused, rac);

                for (int d = 0; d < space.n_shells_outer; d++) {
                    sys.add_eq_inside(11 + d, eq, nused, pused, rac);
                    if (d != space.n_shells_outer - 1) {
                        sys.add_eq_matching(11 + d, OUTER_BC, rac, nused, pused, rac);
                        sys.add_eq_matching(11 + d, OUTER_BC, rac_der, nused, pused, rac);
                    }
                }
            }
        }

        // Adapted_bh pair add_eq body (single BH, shells from domain 2 to the
        // compactified domain).
        static void adapted_bh_add_eq(System_of_eqs& sys, const char* eq, const char* rac, const char* rac_der,
                                      int nused, Array<int>** pused)
        {
            for (int dom = 2; dom < sys.get_dom_max(); ++dom) {
                sys.add_eq_inside(dom, eq, nused, pused, rac);
                sys.add_eq_matching(dom, OUTER_BC, rac, nused, pused, rac);
                sys.add_eq_matching(dom, OUTER_BC, rac_der, nused, pused, rac);
            }

            // Compactified domain
            sys.add_eq_inside(sys.get_dom_max(), eq, nused, pused, rac);
        }

      private:
        // Common lhs/rhs parse + Eq_int construction of the at-infinity integral.
        static void add_eq_int_inf_construct(System_of_eqs& sys, int dom, const char* nom)
        {
            // Get the lhs and rhs
            char p1[LMAX];
            char p2[LMAX];
            bool indic = sys.is_ope_bin(nom, p1, p2, '=');
            if (!indic) {
                KADATH_THROW("= needed for equations");
            } else {
                // Verif lhs = 0 ?
                indic = ((p2[0] == '0') && (p2[1] == ' ') && (p2[2] == '\0')) ? true : false;

                // Construction of the equation
                sys.ensure_eq_int_slot(); sys.eq_int[sys.neq_int].reset(new Eq_int(1));

                // Affectation :
                // no lhs :
                if (indic)
                    sys.eq_int[sys.neq_int]->set_part(0, sys.give_ope(dom, p1, OUTER_BC));

                else
                    sys.eq_int[sys.neq_int]->set_part(
                        0, new Ope_sub(&sys, sys.give_ope(dom, p1, OUTER_BC), sys.give_ope(dom, p2, OUTER_BC)));
                sys.neq_int++;
            }
            sys.nbr_conditions = -1;
        }
    };

} // namespace Kadath
