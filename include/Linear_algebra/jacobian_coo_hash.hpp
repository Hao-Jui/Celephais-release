#pragma once

#include <cstdint>

namespace Kadath
{

    std::uint64_t jacobian_coo_bit_hash(int n, long long nnz, double drop_tol, const int* irn, const int* jcn,
                                        const double* values);

    // Hash a canonical (column,row,value-bits)-sorted copy of the COO entries.
    // Unlike jacobian_coo_bit_hash, this is invariant to MPI rank-major assembly
    // order. It remains bit-sensitive to floating-point values and rejects
    // duplicate coordinates, whose summation can remain order-sensitive.
    std::uint64_t canonical_jacobian_coo_bit_hash(int n, long long nnz, double drop_tol, const int* irn, const int* jcn,
                                                  const double* values);

} // namespace Kadath
