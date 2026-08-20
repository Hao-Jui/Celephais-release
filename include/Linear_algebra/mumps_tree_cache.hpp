/*
    Copyright 2026 Kadath contributors

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

#include <filesystem>
#include <span>
#include <vector>

namespace Kadath
{

    /// Non-owning input to the persistent MUMPS analysis-pair cache.
    ///
    /// Both permutations use MUMPS' one-based conventions. UNS_PERM maps
    /// matched position i to the original column stored at that position;
    /// SYM_PERM maps each variable to its pivot position. pattern_nnz records
    /// provenance only: error-scaled sparse assembly may change the entry count
    /// between preconditioner refreshes without invalidating either bijection.
    struct MumpsTreeCacheView {
        int dimension = 0;
        long long pattern_nnz = 0;
        bool matching_applied = false;
        std::span<const int> column_permutation_1based;
        std::span<const int> symmetric_permutation_1based;
    };

    /// Owned cache returned only after the file passes path, size, schema,
    /// archive-hash, expected-dimension, and permutation validation.
    struct MumpsTreeCache {
        int dimension = 0;
        long long pattern_nnz = 0;
        bool matching_applied = false;
        std::vector<int> column_permutation_1based;
        std::vector<int> symmetric_permutation_1based;
    };

    /// Validate and atomically publish a new `.mumpstree` file without replacing
    /// any existing directory entry, including a symbolic link.
    void write_mumps_tree_cache(const std::filesystem::path& path, const MumpsTreeCacheView& cache);

    /// Read and validate a regular, non-symbolic-link `.mumpstree` file.
    /// A dimension mismatch is rejected from the fixed-size header before the
    /// permutation arrays are allocated.
    MumpsTreeCache read_mumps_tree_cache(const std::filesystem::path& path, int expected_dimension);

    /// Remove a regular `.mumpstree` file without following symbolic links.
    /// Returns false when the path does not exist and true after a durable delete.
    bool remove_mumps_tree_cache(const std::filesystem::path& path);

    /// Compose current one-based COO column indices with a stored UNS_PERM.
    /// If matched position i contains original column uns_perm[i-1], each
    /// original COO column j moves to the inverse-UNS matched position.
    std::vector<int> compose_mumps_tree_column_indices_1based(
        std::span<const int> column_indices_1based,
        std::span<const int> column_permutation_1based,
        int dimension);

    /// Track the live solve-stage cache so converged-save cleanup can also
    /// remove a cache whose natural filename depends on mutable parameters.
    void set_active_mumps_tree_cache_path(const std::filesystem::path& path);
    std::filesystem::path active_mumps_tree_cache_path();
    void clear_active_mumps_tree_cache_path() noexcept;

} // namespace Kadath
