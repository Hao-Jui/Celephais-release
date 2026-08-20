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

#pragma once

// Standard and Kadath headers
#include "For_Kadath/Array/headcpp.hpp"
#include "For_Kadath/Space/space.hpp"
#include "For_Kadath/Array/memory.hpp"

// =============================================================================
// Tensorial Basis Type Identifiers
// =============================================================================
#define CARTESIAN_BASIS 0
#define SPHERICAL_BASIS 1

/**
 * @def MTZ_BASIS
 * @brief Modified Tolman-Zerilli orthonormal basis for negative-curvature surfaces
 *
 * @details
 * Specialized orthonormal basis adapted to spaces where constant-radius surfaces
 * have negative intrinsic curvature (hyperbolic geometry). Named after the
 * Tolman-Zerilli formalism for black hole perturbation theory.
 *
 * **Geometric context**:
 * In certain spacetimes (particularly those describing hyperbolic geometries or
 * black hole interiors), surfaces of constant radius are not spherical but have
 * negative Gaussian curvature. The MTZ basis is specifically adapted to maintain
 * orthonormality while respecting this hyperbolic structure.
 *
 * **Use cases**:
 * - Black hole interior regions
 * - Hyperbolic spatial slices
 * - Anti-de Sitter (AdS) spacetimes
 * - Certain cosmological models with negative spatial curvature
 *
 * **Properties**:
 * - Maintains orthonormality: g_ij = delta_ij
 * - Adapted to hyperbolic geometry of constant-radius surfaces
 * - Reduces to spherical basis in flat space limit
 *
 * @note Less commonly used than Cartesian or spherical bases
 * @note Requires specialized transformation rules for domain matching
 */
#define MTZ_BASIS 2

/** @} */ // end of basis_types group

namespace Kadath
{

    // =============================================================================
    // Base_tensor Class
    // =============================================================================

    /**
     * @class Base_tensor
     * @brief Manages tensorial basis choice across multi-domain decompositions
     *
     * @details
     * In multi-domain spectral methods, different regions of space may naturally use
     * different coordinate systems (e.g., spherical near a black hole, Cartesian at
     * infinity). Tensor fields must be represented in appropriate bases that respect
     * each domain's coordinate structure.
     *
     * Base_tensor stores the basis choice for each domain in a Space, enabling:
     * - Proper tensor transformations across domain boundaries
     * - Consistent representation within each coordinate system
     * - Efficient basis-dependent differential operators
     * - Correct matching conditions at interfaces
     *
     * ## Multi-Domain Basis Management
     *
     * Consider a binary black hole spacetime (e.g. Space_bin_bh):
     * - **Nucleus domains** (near each hole): SPHERICAL_BASIS
     * - **Shell domains** (intermediate): SPHERICAL_BASIS
     * - **Compactified domain** (infinity): CARTESIAN_BASIS or SPHERICAL_BASIS
     *
     * When a tensor is evaluated across domains, Base_tensor ensures:
     * 1. Components are interpreted in the correct local basis
     * 2. Basis transformations applied at domain interfaces
     * 3. Differential operators use basis-appropriate forms
     *
     * ## Basis Independence of Physics
     *
     * Physical tensors (metric, stress-energy, etc.) are basis-independent geometric
     * objects. However, their **component representations** depend on basis choice:
     *
     * T = T^{ij} e_i  tensor  e_j = T'^{kl} e'_k  tensor  e'_l
     *
     * Base_tensor tracks which basis {e_i} is being used, allowing
     * correct interpretation and transformation of component arrays.
     *
     * ## Orthonormal Bases
     *
     * All basis types (CARTESIAN, SPHERICAL, MTZ) are **orthonormal**:
     * e_i  dot  e_j = delta_ij
     *
     * This simplifies:
     * - Raising/lowering indices (no metric needed)
     * - Inner products (standard Euclidean formula)
     * - Norm computations
     *
     * For non-orthonormal (coordinate) bases, use the metric tensor explicitly.
     *
     * ## Implementation Notes
     *
     * - Basis stored as simple integer per domain (CARTESIAN_BASIS, SPHERICAL_BASIS, MTZ_BASIS)
     * - Lightweight: only one integer per domain (~10s of bytes for typical problems)
     * - Basis transformations handled by Tensor class, not Base_tensor
     * - Compatible with checkpoint/restart via save/load
     *
     * @note Base_tensor is typically created once and shared by many Tensor objects
     * @warning Changing basis after Tensor construction requires explicit transformation
     *
     * @see Tensor for component storage and operations
     * @see Domain for coordinate system definitions
     *
     * @ingroup fields
     */
    class Base_tensor : public MemoryMappable
    {

        // =========================================================================
        // Data Members
        // =========================================================================

      protected:
        /**
         * @brief Reference to the associated spatial decomposition
         * @details Needed to determine number of domains and validate domain indices
         */
        const Space& space;

        /**
         * @brief Basis type for each domain
         * @details Array of integers, one per domain in the Space:
         * - basis[i] = CARTESIAN_BASIS, SPHERICAL_BASIS, or MTZ_BASIS
         * - Size equals space.get_nbr_domains()
         *
         * Access via set_basis() / get_basis() rather than directly for safety
         */
        Array<int> basis;

      public:
        // =========================================================================
        // Constructors and Destructor
        // =========================================================================

        /**
         * @brief Default constructor - creates uninitialized basis
         *
         * @param[in] spa  Space defining the multi-domain decomposition
         *
         * @post Basis array allocated but values undefined
         * @note Must call set_basis() for each domain before use
         * @warning Using uninitialized basis leads to undefined behavior
         *
         * **Use case**: When basis will be set programmatically or loaded from file
         *
         * Example:
         * @code
         * Base_tensor bt(my_space);
         * for(int d = 0; d < my_space.get_nbr_domains(); ++d) {
         *     bt.set_basis(d) = determine_basis_for_domain(d);
         * }
         * @endcode
         */
        explicit Base_tensor(const Space& spa);

        /**
         * @brief Uniform basis constructor - same basis in all domains
         *
         * @param[in] spa  Space defining domain decomposition
         * @param[in] bb   Basis type (CARTESIAN_BASIS, SPHERICAL_BASIS, or MTZ_BASIS)
         *
         * @post All domains use basis bb
         *
         * **Use case**: Homogeneous problems using single coordinate system throughout
         *
         * Example:
         * @code
         * // Globally Cartesian flat spacetime
         * Base_tensor cart_basis(space, CARTESIAN_BASIS);
         *
         * // Single black hole in spherical coordinates everywhere
         * Base_tensor spher_basis(space, SPHERICAL_BASIS);
         * @endcode
         */
        Base_tensor(const Space& spa, int bb);

        /**
         * @brief Copy constructor - creates independent copy
         *
         * @param[in] source  Base_tensor to copy
         *
         * @post Deep copy created with same Space reference and basis values
         * @note Modifying copy does not affect original
         */
        Base_tensor(const Base_tensor& source);

        Base_tensor(const Space& spa, BinarySource& source); ///< Modern API.

        /**
         * @brief Destructor
         * @post Basis array deallocated, Space reference invalidated
         */
        virtual ~Base_tensor();

      public:
        // =========================================================================
        // Basis Access and Modification
        // =========================================================================

        /**
         * @brief Returns reference to basis in specified domain (read/write)
         *
         * @param[in] domain  Domain index (0 ≤ domain < num_domains)
         * @return Reference to basis integer for modification
         *
         * @warning No bounds checking in release builds
         * @note Use for setting basis: bt.set_basis(3) = SPHERICAL_BASIS
         *
         * Example:
         * @code
         * Base_tensor bt(space);
         * // Inner domains: spherical
         * for(int d = 0; d < 5; ++d)
         *     bt.set_basis(d) = SPHERICAL_BASIS;
         * // Outer domain: Cartesian
         * bt.set_basis(5) = CARTESIAN_BASIS;
         * @endcode
         */
        int& set_basis(int domain);

        /**
         * @brief Returns basis type for specified domain (read-only)
         *
         * @param[in] domain  Domain index
         * @return Basis type (CARTESIAN_BASIS, SPHERICAL_BASIS, or MTZ_BASIS)
         *
         * @warning No bounds checking in release builds
         * @note Use for querying basis: if(bt.get_basis(3) == SPHERICAL_BASIS) {...}
         *
         * Example:
         * @code
         * for(int d = 0; d < space.get_nbr_domains(); ++d) {
         *     int b = bt.get_basis(d);
         *     if(b == SPHERICAL_BASIS)
         *         apply_spherical_operator(d);
         *     else if(b == CARTESIAN_BASIS)
         *         apply_cartesian_operator(d);
         * }
         * @endcode
         */
        int get_basis(int domain) const;

        /**
         * @brief Assignment operator - copies basis configuration
         *
         * @param[in] source  Base_tensor to copy from
         *
         * @pre Both objects must reference the same Space
         * @post This object's basis matches source
         *
         * @throws std::invalid_argument if Space references differ
         *
         * @note Only copies basis values, not Space reference (that's immutable)
         */
        void operator=(const Base_tensor& source);

        /**
         * @brief Returns the associated Space (read-only)
         * @return Const reference to Space
         * @note Space is immutable after construction
         */
        const Space& get_space() const { return space; }

        void save(BinarySink& sink) const; ///< Modern API.

        // =========================================================================
        // Friend Functions - Comparison and I/O
        // =========================================================================

        /**
         * @brief Stream insertion operator for display
         *
         * @param[in,out] os  Output stream
         * @param[in]     bt  Base_tensor to display
         * @return Reference to output stream (for chaining)
         *
         * @note Output format: lists basis type for each domain
         *
         * Example output:
         * @code
         * Domain 0: SPHERICAL_BASIS
         * Domain 1: SPHERICAL_BASIS
         * Domain 2: CARTESIAN_BASIS
         * @endcode
         */
        friend ostream& operator<<(ostream& os, const Base_tensor& bt);

        /**
         * @brief Tests equality of two bases
         *
         * @param[in] bt1  First Base_tensor
         * @param[in] bt2  Second Base_tensor
         * @return true if bases identical in all domains
         * @return false otherwise
         *
         * @pre Both must reference the same Space
         *
         * @note Compares basis values domain-by-domain
         *
         * **Use case**: Validating basis compatibility before tensor operations
         *
         * Example:
         * @code
         * if(tensor1.get_basis() == tensor2.get_basis()) {
         *     // Safe to add tensors - bases match
         *     Tensor sum = tensor1 + tensor2;
         * } else {
         *     // Need basis transformation first
         *     Tensor transformed = tensor2.change_basis(tensor1.get_basis());
         *     Tensor sum = tensor1 + transformed;
         * }
         * @endcode
         */
        friend bool operator==(const Base_tensor& bt1, const Base_tensor& bt2);

        /**
         * @brief Tests inequality of two bases
         *
         * @param[in] bt1  First Base_tensor
         * @param[in] bt2  Second Base_tensor
         * @return true if bases differ in any domain
         * @return false if bases identical
         *
         * @note Logical negation of operator==
         */
        friend bool operator!=(const Base_tensor& bt1, const Base_tensor& bt2);
    };

    // =============================================================================
    // Utility Functions
    // =============================================================================

    /**
     * @brief Returns human-readable name for basis type
     *
     * @param[in] basis_id  Basis identifier (CARTESIAN_BASIS, etc.)
     * @return String name ("Cartesian", "Spherical", "MTZ", or "Unknown")
     *
     * @note Useful for debugging and user-facing output
     *
     * Example:
     * @code
     * std::cout << "Using " << basis_name(bt.get_basis(0)) << " basis in domain 0\n";
     * // Output: "Using Spherical basis in domain 0"
     * @endcode
     */
    inline const char* basis_name(int basis_id)
    {
        switch (basis_id) {
            case CARTESIAN_BASIS:
                return "Cartesian";
            case SPHERICAL_BASIS:
                return "Spherical";
            case MTZ_BASIS:
                return "MTZ";
            default:
                return "Unknown";
        }
    }

} // namespace Kadath
