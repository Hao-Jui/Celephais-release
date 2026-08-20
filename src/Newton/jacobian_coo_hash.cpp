#include "Linear_algebra/jacobian_coo_hash.hpp"

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <stdexcept>
#include <tuple>
#include <vector>

namespace Kadath
{
    namespace
    {

        std::uint64_t double_bits(double value)
        {
            std::uint64_t bits = 0;
            static_assert(sizeof(bits) == sizeof(value));
            std::memcpy(&bits, &value, sizeof(bits));
            return bits;
        }

        void hash_bytes(std::uint64_t& hash, const void* data, std::size_t bytes)
        {
            constexpr std::uint64_t fnv_prime = 1099511628211ULL;
            const auto* raw = static_cast<const unsigned char*>(data);
            for (std::size_t i = 0; i < bytes; ++i) {
                hash ^= raw[i];
                hash *= fnv_prime;
            }
        }

        template <typename T> void hash_scalar(std::uint64_t& hash, const T& value)
        {
            hash_bytes(hash, &value, sizeof(T));
        }

        void validate_inputs(long long nnz, const int* irn, const int* jcn, const double* values)
        {
            if (nnz < 0)
                throw std::invalid_argument("Jacobian COO hash requires nnz >= 0");
            if (nnz > 0 && (irn == nullptr || jcn == nullptr || values == nullptr))
                throw std::invalid_argument("Jacobian COO hash requires non-null entry arrays");
        }

        std::uint64_t hash_header(int n, long long nnz, double drop_tol)
        {
            std::uint64_t hash = 1469598103934665603ULL;
            hash_scalar(hash, n);
            hash_scalar(hash, nnz);
            hash_scalar(hash, drop_tol);
            return hash;
        }

        struct CanonicalEntry {
            int row = 0;
            int column = 0;
            std::uint64_t value_bits = 0;

            bool operator<(const CanonicalEntry& other) const
            {
                return std::tie(column, row, value_bits) < std::tie(other.column, other.row, other.value_bits);
            }
        };

    } // namespace

    std::uint64_t jacobian_coo_bit_hash(int n, long long nnz, double drop_tol, const int* irn, const int* jcn,
                                        const double* values)
    {
        validate_inputs(nnz, irn, jcn, values);
        std::uint64_t hash = hash_header(n, nnz, drop_tol);
        for (long long entry = 0; entry < nnz; ++entry) {
            const auto index = static_cast<std::size_t>(entry);
            hash_scalar(hash, irn[index]);
            hash_scalar(hash, jcn[index]);
            hash_scalar(hash, double_bits(values[index]));
        }
        return hash;
    }

    std::uint64_t canonical_jacobian_coo_bit_hash(int n, long long nnz, double drop_tol, const int* irn, const int* jcn,
                                                  const double* values)
    {
        validate_inputs(nnz, irn, jcn, values);
        std::vector<CanonicalEntry> canonical_entries;
        canonical_entries.reserve(static_cast<std::size_t>(nnz));
        for (long long entry = 0; entry < nnz; ++entry) {
            const auto index = static_cast<std::size_t>(entry);
            canonical_entries.push_back({irn[index], jcn[index], double_bits(values[index])});
        }
        std::sort(canonical_entries.begin(), canonical_entries.end());
        for (std::size_t index = 1; index < canonical_entries.size(); ++index) {
            const CanonicalEntry& previous = canonical_entries[index - 1];
            const CanonicalEntry& current = canonical_entries[index];
            if (previous.row == current.row && previous.column == current.column) {
                throw std::invalid_argument(
                    "Canonical Jacobian COO hash requires unique row/column coordinates");
            }
        }

        std::uint64_t hash = hash_header(n, nnz, drop_tol);
        for (const CanonicalEntry& entry : canonical_entries) {
            hash_scalar(hash, entry.row);
            hash_scalar(hash, entry.column);
            hash_scalar(hash, entry.value_bits);
        }
        return hash;
    }

} // namespace Kadath
