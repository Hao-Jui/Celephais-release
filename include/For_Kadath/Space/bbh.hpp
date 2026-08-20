/*
    Copyright 2020 Philippe Grandclement

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

#pragma once

#include "For_Kadath/Domain/adapted.hpp"
#include "For_Kadath/Domain/bispheric.hpp"
#include "For_Kadath/IO/binary_sink.hpp"
#include "For_Kadath/IO/binary_source.hpp"

namespace Kadath
{

    /**
     * @class Space_bbh
     * @brief Multi-domain spectral decomposition for binary black hole spacetimes in full general relativity
     *
     * @details
     * Space_bbh provides a sophisticated domain decomposition specifically designed for
     * numerical simulations of binary black hole (BBH) systems in Einstein's theory of
     * general relativity.
     *
     * ## Domain Decomposition Strategy
     *
     * The space uses a hybrid coordinate approach combining:
     * - **Spherical domains**: Near each black hole for horizon tracking
     * - **Bispherical domains**: In the intermediate region exploiting two-center geometry
     * - **Compactified shell**: Extending to spatial infinity
     *
     * ### Coordinate Systems and Domain Layout
     *
     * Layout summary:
     * - BH1 adapted shells (domains 0-1)
     * - BH2 adapted shells (domains 2-3)
     * - Bispherical bridge (domains 4-8, chi/eta regions)
     * - Outer compactified shell (domain 9)
     *
     * ## Domain-by-Domain Description
     *  * ```
     *           BH1                           BH2
     *            ○                             ○
     *       [Adapted]                     [Adapted]
     *      Shells 1-2                    Shells 3-4
     *           ↓                             ↓
     *     ┌─────────────────────────────────────────┐
     *     │         Bispherical Domains             │
     *     │  [χ-first]  [rect]  [η]  [rect]  [χ]   │  Domains 5-9
     *     └─────────────────────────────────────────┘
     *                       ↓
     *              ┌──────────────────┐
     *              │  Outer Shell     │  Domain 10
     *              │  (Compactified)  │
     *              └──────────────────┘
     * ```
     * The 10 domains are organized as follows (domain indices 0-9):
     *
     * **Black Hole 1 Region (Domains 0-1)**:
     * 1. **Domain_shell_outer_adapted** (outer): Spherical shell with fixed inner boundary
     *    and adaptable outer boundary conforming to the first horizon
     * 2. **Domain_shell_inner_adapted** (inner): Spherical shell with adaptable inner
     *    boundary tracking the first horizon and fixed outer boundary
     *
     * **Black Hole 2 Region (Domains 2-3)**:
     * 3. **Domain_shell_outer_adapted**: Mirror of domain 0 for second hole
     * 4. **Domain_shell_inner_adapted**: Mirror of domain 1 for second hole
     *
     * **Bispherical Intermediate Region (Domains 4-8)**:
     * 5. **Domain_bispheric_chi_first**: Near first hole, chi coordinate emphasized
     * 6. **Domain_bispheric_rect**: Rectangular-like region near first hole
     * 7. **Domain_bispheric_eta_first**: Central region between holes, eta coordinate emphasized
     * 8. **Domain_bispheric_rect**: Rectangular-like region near second hole
     * 9. **Domain_bispheric_chi_first**: Near second hole, chi coordinate emphasized
     *
     * **Asymptotic Region (Domain 9)**:
     * 10. **Domain_shell**: Large spherical shell centered at origin, compactified to
     *     include spatial infinity. Used for wave extraction and boundary conditions.
     *
     * ## Bispherical Coordinates
     *
     * The bispherical coordinate system (chi, eta, varphi) is defined by:
     * x = a * sinh(eta) / (cosh(eta) - cos(chi)) * cos(varphi)
     * y = a * sinh(eta) / (cosh(eta) - cos(chi)) * sin(varphi)
     * z = a * sin(chi) / (cosh(eta) - cos(chi))
     *
     * where a = d/2 is half the coordinate separation. This system naturally
     * describes two focal spheres at z = +/- a.
     *
     * **Advantages**:
     * - Constant-chi surfaces are spheres (perfect for spherical horizons)
     * - Constant-eta surfaces are spheres orthogonal to chi-surfaces
     * - Naturally handles the two-center geometry
     * - Singularities located away from physical regions
     *
     * ## Adaptive Horizon Tracking
     *
     * The inner/outer adapted shells use spectral representations of the horizon shape:
     *
     *
     * r_H(theta, varphi) = sum_{l,m} a_lm Y_lm(theta, varphi)
     *
     *
     * where Y_lm are spherical harmonics. During evolution:
     * - Coefficients a_lm are solution variables (degrees of freedom)
     * - Horizon shape determined by apparent horizon conditions
     * - Domains continuously adjust to track moving, deforming horizons
     *
     * ## Typical Use Case: BBH Initial Data
     *
     * @code
     * // Binary black hole at separation d = 10M
     * double separation = 10.0;
     * double mass1 = 0.5;        // Mass parameter ~ radius
     * double mass2 = 0.5;
     * double r_bisph = 8.0;      // Outer bispherical boundary
     * double r_infinity = 50.0;  // Compactified infinity
     * int resolution = 9;        // Collocation points per dimension
     *
     * Space_bbh bbh_space(CHEB_TYPE, separation, mass1, mass2,
     *                     r_bisph, r_infinity, resolution);
     *
     * // Solve Einstein constraints with adaptive horizons
     * System_of_eqs constraints(bbh_space);
     * // Add conformal flatness, maximal slicing, etc.
     * constraints.add_eq(...);
     * bbh_space.add_matching(constraints, "N", list_N);
     * bbh_space.add_matching(constraints, "conf", list_conf);
     * // ... solve system ...
     * @endcode
     *
     * ## Matching Conditions
     *
     * Spectral matching enforces C-infinity continuity at all interfaces:
     * - Field values match exactly
     * - Normal derivatives match exactly
     * - All higher derivatives match (spectral convergence)
     *
     * This is crucial for maintaining constraint satisfaction (Hamiltonian and
     * momentum constraints) throughout the evolution.
     *
     * @note This class handles initial data construction; time evolution requires
     *       additional infrastructure for adaptive mesh refinement and excision.
     *
     * @warning Adaptive shells increase system size - each horizon adds ~100-1000 unknowns
     *          depending on angular resolution.
     *
     * @see Domain_shell_inner_adapted for horizon tracking details
     * @see Domain_bispheric_chi_first for bispherical coordinate implementation
     * @see Space for base class interface
     *
     * @ingroup domain
     */
    class Space_bbh : public Space
    {

      public:
        // =========================================================================
        // Constructors and Destructor
        // =========================================================================

        /**
         * @brief Constructs binary black hole space with specified geometry
         *
         * @param[in] ttype  Spectral basis type (typically CHEB_TYPE or LEG_TYPE)
         * @param[in] dist   Coordinate separation between black hole centers (in code units)
         * @param[in] rbh1   Initial radius of first black hole horizon (spherical approximation)
         * @param[in] rbh2   Initial radius of second black hole horizon (spherical approximation)
         * @param[in] rbi    Outer radius of bispherical domain region
         * @param[in] rext   Outer radius of compactified domain (effective infinity)
         * @param[in] nr     Number of collocation points per dimension in each domain
         *
         * @details
         * **Parameter Guidelines**:
         *
         * - **dist**: Separation should satisfy d > r1 + r2 + 2*delta where
         *   delta approx 0.5M minimum clearance to avoid domain overlap.
         *   Typical range: d = 6M (close) to d = 20M (wide).
         *
         * - **rbh1, rbh2**: Initial horizon radii. For Schwarzschild, r_H = 2M.
         *   For spinning holes, estimate from irreducible mass. These are starting
         *   guesses; actual shapes determined by horizon-finding.
         *
         * - **rbi**: Transition radius from bispherical to outer shell. Should satisfy:
         *   r_bi > sqrt((d/2)^2 + r_max^2) where r_max = max(r1, r2).
         *   Typical: r_bi approx 1.5d to 2d.
         *
         * - **rext**: Outer boundary location.
         *
         * - **nr**: Resolution per dimension.
         *
         * **Domain Construction**:
         *
         * The constructor creates 10 domains in this specific order:
         *
         * | Index | Type | Description | Coordinates |
         * |-------|------|-------------|-------------|
         * | 0 | Domain_shell_outer_adapted | BH1 outer shell    | Spherical (r,theta,phi) |
         * | 1 | Domain_shell_inner_adapted | BH1 inner shell    | Spherical (r,theta,phi) |
         * | 2 | Domain_shell_outer_adapted | BH2 outer shell    | Spherical (r,theta,phi) |
         * | 3 | Domain_shell_inner_adapted | BH2 inner shell    | Spherical (r,theta,phi) |
         * | 4 | Domain_bispheric_chi_first | Near BH1           | Bispherical (chi,eta,phi) |
         * | 5 | Domain_bispheric_rect      | Transition BH1     | Bispherical (chi,eta,phi) |
         * | 6 | Domain_bispheric_eta_first | Between holes      | Bispherical (chi,eta,phi) |
         * | 7 | Domain_bispheric_rect      | Transition BH2     | Bispherical (chi,eta,phi) |
         * | 8 | Domain_bispheric_chi_first | Near BH2           | Bispherical (chi,eta,phi) |
         * | 9 | Domain_shell               | Outer compactified | Spherical (r,theta,phi) |
         *
         * **Memory Requirements**:
         *
         * For typical parameters (nr=11, complex scalar field):
         * - Per domain: ~(11^3) x 8 bytes x 10 components approx 100 KB
         * - Total: ~1-10 MB depending on field complexity
         * - Adaptive unknowns: ~500-2000 additional per horizon
         *
         * @post All 10 domains created and connected
         * @post Horizons initialized as spherical (will adapt during solve)
         * @post Domain matching relationships established
         *
         * @throws std::invalid_argument if geometric constraints violated
         * @throws std::bad_alloc if memory allocation fails
         *
         * @note Construction may take several seconds for large resolutions due to
         *       collocation point computation and connection mapping
         *
         * Example:
         * @code
         * // Equal-mass non-spinning BBH at moderate separation
         * Space_bbh bbh(CHEB_TYPE,
         *               10.0,    // separation = 10M
         *               1.0,     // BH1 radius ~ 2M for M=0.5
         *               1.0,     // BH2 radius ~ 2M for M=0.5
         *               15.0,    // bispherical outer boundary
         *               100.0,   // compactified infinity
         *               11);     // 11^3 points per domain
         * @endcode
         */
        Space_bbh(int ttype, double dist, double rbh1, double rbh2, double rbi, double rext, int nr);

        Space_bbh(BinarySource& source); ///< Modern API.

        /**
         * @brief Destructor - deallocates all domain objects
         *
         * @post All domains destroyed
         * @post Memory freed
         *
         * @note Automatically called; manual invocation unnecessary
         */
        ~Space_bbh() override;

      public:
        // =========================================================================
        // Domain Matching (Spectral Continuity Enforcement)
        // =========================================================================

        /**
         * @brief Adds spectral matching conditions across all bispherical domain interfaces
         *
         * @param[in,out] syst  System of equations to augment
         * @param[in]     rac   String describing the matching condition equation
         * @param[in]     list  List of tensor components to match
         *
         * @details
         * **Matching Philosophy**:
         *
         * Spectral matching enforces that fields are C-infinity smooth across domain boundaries
         * by requiring both the field and its normal derivative to agree at interfaces.
         * For a second-order PDE (like Einstein constraints), this guarantees continuity
         * of the solution and its first derivatives, with higher derivatives following
         * from smoothness of spectral representation.
         *
         * **Equations Added**:
         *
         * At each interface between bispherical domains (5 interfaces: 4<->5, 5<->6, 6<->7, 7<->8):
         * 1. **Value matching**: [f]_interface = 0
         * 2. **Derivative matching**: [partial_n f]_interface = 0
         *
         * where [.] denotes the jump across the interface and partial_n
         * is the normal derivative.
         *
         * **When to use**:
         * - After adding bulk equations to the system
         * - For each field that should be smooth across domains
         * - Typically called once per field during system setup
         *
         * **Component Selection**:
         *
         * The List_comp specifies which tensor components to match:
         * @code
         * // Match all 3 components of shift vector
         * List_comp shift_comps(3, 1);
         * for (int component = 0; component < 3; ++component)
         *     shift_comps.set(component)->set(0) = component + 1;
         * space.add_matching(system, "bet", shift_comps);
         * @endcode
         *
         * @pre System must contain field definitions referenced by rac
         * @post Matching equations added for specified components at all internal boundaries
         *
         * @note Does NOT add matching at spherical shell interfaces - handle separately
         * @note For first-order systems, may need additional matching conditions
         *
         * @see add_matching(System_of_eqs&, const char*, int, Array<int>**) for index-based version
         */
        void add_matching(System_of_eqs& syst, const char* rac, const List_comp& list);

        /**
         * @brief Adds spectral matching conditions with explicit component indexing
         *
         * @param[in,out] syst   System of equations to augment
         * @param[in]     rac    String describing the matching condition equation
         * @param[in]     nused  Number of components to match (-1 = all components)
         * @param[in]     pused  Pointer to array of component indices to match (nullptr if nused=-1)
         *
         * @details
         * **Flexible Component Selection**:
         *
         * This overload provides fine-grained control over which tensor components
         * participate in matching conditions. Useful for:
         * - Gauge conditions that don't require full matching
         * - Exploiting symmetries (e.g., axisymmetry reduces components)
         * - Debugging (match subset of components)
         *
         * **Usage Patterns**:
         *
         * *Match all components*:
         * @code
         * space.add_matching(system, "conf", -1, nullptr);
         * // Matches all components of conformal factor
         * @endcode
         *
         * *Match specific components*:
         * @code
         * Array<int>* indices[3];
         * for(int i = 0; i < 3; i++) {
         *     indices[i] = new Array<int>(1);
         *     (*indices[i])(0) = i;  // Match components 0, 1, 2
         * }
         * space.add_matching(system, "shift", 3, indices);
         * // Cleanup
         * for(int i = 0; i < 3; i++) delete indices[i];
         * @endcode
         *
         * *Match only radial component*:
         * @code
         * Array<int>* r_comp = new Array<int>(1);
         * (*r_comp)(0) = 0;
         * Array<int>* comp_array[1] = {r_comp};
         * space.add_matching(system, "velocity", 1, comp_array);
         * delete r_comp;
         * @endcode
         *
         * **Performance Note**:
         *
         * Matching conditions are boundary integrals, computationally cheaper than
         * bulk equations. Adding matching for all components is usually acceptable.
         *
         * @pre If pused != nullptr, must point to valid array of nused Array<int>* pointers
         * @post Matching equations added for specified components
         *
         * @warning Caller responsible for memory management of pused arrays
         * @note Default behavior (nused=-1) matches all components of tensor field
         */
        void add_matching(System_of_eqs& syst, const char* rac, int nused = -1, Array<int>** pused = nullptr);

      public:
        // =========================================================================
        // Persistence and I/O
        // =========================================================================

        void save(BinarySink& sink) const override; ///< Modern API.

      public:
        // =========================================================================
        // Variable Domain Interface (Adaptive Horizon Management)
        // =========================================================================

        /**
         * @brief Counts total degrees of freedom from adaptive horizons
         *
         * @return Sum of spectral coefficients describing both horizon shapes
         *
         * @details
         * Each adaptive shell contributes unknowns equal to the number of spherical
         * harmonic coefficients used to represent the horizon shape:
         *
         *
         * N_unknowns = sum_{l=0..l_max} (2l + 1) = (l_max + 1)^2
         *
         *
         * For two horizons: N_total = 2(l_max + 1)^2
         *
         * Typical values:
         * - l_max = 4: 50 unknowns per horizon (100 total)
         * - l_max = 6: 98 unknowns per horizon (196 total)
         * - l_max = 8: 162 unknowns per horizon (324 total)
         *
         * **Use case**: System assembly to allocate solution vector
         * @code
         * int total_unknowns = field_unknowns + space.nbr_unknowns_from_variable_domains();
         * Array<double> solution(total_unknowns);
         * @endcode
         */
        int nbr_unknowns_from_variable_domains() const override;

        /**
         * @brief Maps solution vector components to horizon shape parameters
         *
         * @param[in,out] pos        Current position in solution vector (updated on return)
         * @param[in]     start_idx  Starting index for this call
         * @param[out]    affected   Array indicating which domains affected by each unknown
         *
         * @details
         * Establishes correspondence between global solution vector entries and
         * specific spherical harmonic coefficients of horizon shapes. Called during
         * system assembly to build Jacobian matrix structure.
         *
         * **Indexing Convention**:
         * - First block: BH1 horizon coefficients (a_lm^(1))
         * - Second block: BH2 horizon coefficients (a_lm^(2))
         *
         * @note Order within each block: m = -l, ..., +l for l = 0, 1, 2, ...
         */
        void affecte_coef_to_variable_domains(int& pos, int start_idx, Array<int>& affected) const override;

        /**
         * @brief Computes horizon shape derivatives from solution vector
         *
         * @param[in]     solution  Global solution vector
         * @param[in,out] pos       Current reading position (updated on return)
         *
         * @details
         * Extracts horizon shape parameters and computes spatial derivatives needed
         * for evaluating normal vectors, surface elements, and geometric quantities
         * at the horizons. Essential for:
         * - Apparent horizon conditions
         * - Surface integrals (area, angular momentum)
         * - Boundary condition enforcement
         *
         * **Computed quantities**:
         * - partial_theta r_H(theta, varphi)
         * - partial_varphi r_H(theta, varphi)
         * - Unit normal: n = nabla r_H / |nabla r_H|
         */
        void xx_to_ders_variable_domains(const Array<double>& solution, int& pos) const override;

        /**
         * @brief Updates horizon shapes from solution vector
         *
         * @param[in,out] sys       Equation system (may update cached quantities)
         * @param[in]     solution  Global solution vector with updated horizon parameters
         * @param[in,out] pos       Current reading position (updated on return)
         *
         * @details
         * Core method of Newton iteration for horizon adaptation. Reads spectral
         * coefficients from solution vector and reconstructs horizon surfaces:
         *
         *
         * r_H(theta, varphi) = sum_{l,m} a_lm Y_lm(theta, varphi)
         *
         *
         * Then updates:
         * 1. Domain geometric mappings
         * 2. Collocation point positions
         * 3. Derivative operators (affected by coordinate transformation)
         * 4. Cached normal vectors and surface elements
         *
         * **Convergence behavior**:
         * - Horizon shape converges simultaneously with field equations
         * - Typically requires 5-15 Newton iterations for 1e-10 accuracy
         * - Quadratic convergence near solution
         *
         * @post All adaptive domains updated with new horizon geometries
         * @post Geometric quantities recomputed
         *
         * @note Expensive operation - involves spectral transformations and derivative updates
         */
        void xx_to_vars_variable_domains(System_of_eqs* sys, const Array<double>& solution, int& pos) const override;

      public:
        // =========================================================================
        // Domain Connectivity (Non-Standard Matching)
        // =========================================================================

        /**
         * @brief Returns matching indices for non-standard domain interfaces
         *
         * @param[in] dom    Domain index
         * @param[in] bound  Boundary index within that domain
         * @return Array of matching collocation point indices in adjacent domain
         *
         * @details
         * **Standard vs. Non-Standard Matching**:
         *
         * - **Standard**: Domains with identical coordinate structure on shared boundary
         *   (e.g., two spherical shells). Index correspondence is trivial: index[i] = i.
         *
         * - **Non-standard**: Domains with different coordinate systems on shared boundary
         *   (e.g., spherical domain meeting bispherical domain). Requires non-trivial
         *   mapping between collocation point sets.
         *
         * **Use case**: Enforcing spectral matching across coordinate system transitions
         *
         * In Space_bbh, non-standard interfaces occur at:
         * - Spherical adapted shells <-> Bispherical domains
         * - Bispherical domains <-> Outer compactified shell
         *
         * **Index Mapping**:
         *
         * For boundary with N collocation points in domain `dom`, returns array of
         * size N containing indices of corresponding points in adjacent domain.
         *
         * Example:
         * @code
         * // Get matching for boundary 1 of domain 3 (BH2 inner shell)
         * Array<int> match_indices = space.get_indices_matching_non_std(3, 1);
         *
         * // Use for interpolation or matching condition
         * for(int i = 0; i < match_indices.get_size(0); i++) {
         *     int neighbor_idx = match_indices(i);
         *     // Enforce field(dom=3, bound=1, i) == field(neighbor_dom, neighbor_idx)
         * }
         * @endcode
         *
         * @pre dom and bound must specify a valid non-standard interface
         * @return Array with one entry per boundary collocation point
         *
         * @note For standard interfaces, this method may not be called (trivial mapping)
         * @note Matching indices precomputed during space construction for efficiency
         */
        Array<int> get_indices_matching_non_std(int dom, int bound) const override;
    };

} // namespace Kadath
