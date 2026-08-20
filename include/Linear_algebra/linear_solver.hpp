#pragma once
#include "For_Kadath/Array/exceptions.hpp"

namespace Kadath
{

// Linear-solver failures derive from KadathError so guarded_run catches
// them on the same path as other domain-failure throws.
class LinearSolverError : public KadathError
{
  public:
    using KadathError::KadathError;
};

// Abstract interface for a sparse direct linear solver.
// Lifecycle:
//   1. set_pattern(n, nnz, irn, jcn) — establish sparsity once per Newton iteration
//   2. factor(values)                — numerical factorization with COO values
//   3. solve(rhs_inout)              — overwrite rhs with x where A x = rhs
//   4. reset()                       — release internal state, ready for next set_pattern
//
// Implementations dispatch to MUMPS (production) or Mock (tests). MPI lifetime
// management lives inside the concrete adapter — the interface is rank-agnostic.
class LinearSolver
{
  public:
    virtual ~LinearSolver() = default;

    virtual void set_pattern(int n, long long nnz, const int* irn, const int* jcn) = 0;
    virtual void factor(const double* values) = 0;
    virtual void solve(double* rhs_inout) = 0;
    virtual void reset() = 0;
};

} // namespace Kadath
