# patch_metis_idx64.cmake -- widen the bundled METIS index type to 64-bit.
#
# Run as: cmake -DMETIS_HEADER=<path/to/copied/metis.h> -P patch_metis_idx64.cmake
#
# Changes the copied METIS header's "#define IDXTYPEWIDTH 32" to 64 so libmetis.a
# is built with a 64-bit idx_t. The bundled MUMPS analysis phase then calls the
# MUMPS_METIS_*_64 wrappers (guarded by "#if (IDXTYPEWIDTH == 64)" in
# mumps_metis64.c) and can order matrices with more than 2^31 nonzeros --
# fixing MUMPS INFOG(1)=-51 "integer overflow in ordering" at high resolution
# (e.g. res15 BNS/BHNS, nnz ~ 2e9). This is the ordering only: MUMPS_INT stays
# 32-bit (matrix order N never approaches 2^31, and NNZ is already 64-bit in
# MUMPS's default mixed-mode), and BLAS/LAPACK/ScaLAPACK/MPI stay LP64 -- the
# factorization never sees a METIS idx_t. Patches the BUILD COPY of the header,
# so third_party/metis stays pristine.
if(NOT DEFINED METIS_HEADER)
    message(FATAL_ERROR "patch_metis_idx64: METIS_HEADER must be set (-DMETIS_HEADER=...)")
endif()
if(NOT EXISTS "${METIS_HEADER}")
    message(FATAL_ERROR "patch_metis_idx64: header not found: ${METIS_HEADER}")
endif()
# Target width: 64 by default (high-res Linux: order >2^31 nonzeros). Callers may
# request 32 (-DMETIS_IDXWIDTH=32) where the 64-bit GKlib path corrupts the heap
# under a strict allocator (macOS Darwin hardened malloc traps in gk_free during
# nested-dissection refinement; glibc tolerates it). 32-bit caps the ordering at
# 2^31 nonzeros -- fine for the smaller problems run on such hosts.
if(NOT DEFINED METIS_IDXWIDTH)
    set(METIS_IDXWIDTH 64)
endif()
if(NOT METIS_IDXWIDTH MATCHES "^(32|64)$")
    message(FATAL_ERROR "patch_metis_idx64: METIS_IDXWIDTH must be 32 or 64 (got '${METIS_IDXWIDTH}')")
endif()

file(READ "${METIS_HEADER}" _metis_h)
if(_metis_h MATCHES "#define[ \t]+IDXTYPEWIDTH[ \t]+${METIS_IDXWIDTH}([^0-9]|$)")
    message(STATUS "METIS: IDXTYPEWIDTH already ${METIS_IDXWIDTH} in ${METIS_HEADER}")
elseif(_metis_h MATCHES "#define[ \t]+IDXTYPEWIDTH[ \t]+(32|64)")
    string(REGEX REPLACE
        "#define([ \t]+)IDXTYPEWIDTH([ \t]+)(32|64)"
        "#define\\1IDXTYPEWIDTH\\2${METIS_IDXWIDTH}"
        _metis_h "${_metis_h}")
    file(WRITE "${METIS_HEADER}" "${_metis_h}")
    message(STATUS "METIS: patched ${METIS_HEADER} to IDXTYPEWIDTH=${METIS_IDXWIDTH}")
else()
    message(FATAL_ERROR
        "patch_metis_idx64: could not find '#define IDXTYPEWIDTH 32|64' in ${METIS_HEADER}; "
        "the vendored METIS header layout changed -- update this patch.")
endif()
