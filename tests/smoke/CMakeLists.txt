# Tier 2: Smoke tests — infrastructure-only sanity checks.
# Purpose: verify binary launches, input parses, domains construct, MPI initializes.
# Does NOT test Newton convergence, Jacobian assembly, or numerical correctness.
#   smoke_bh2d     — 2D spherical domain, dense solver path              (~60s)
#   smoke_bh2d_xcts — 2D spherical XCTS domain                           (~60s)
#   smoke_ns2d     — 2D spherical domain + EOS loading                   (~30s)
#   smoke_ns2d_xcts — 2D spherical XCTS domain + EOS loading              (~30s)
#   smoke_ns_nosym — 3D nosym spherical domain + EOS loading             (~60s)
#   smoke_bhns     — 3D bispheric domain, BH excision + NS, mixed physics (~120s)

add_test(NAME smoke_bh2d
    COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/test_bh2d_smoke.sh
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
)
set_tests_properties(smoke_bh2d PROPERTIES
    LABELS "smoke"
    TIMEOUT 120
    ENVIRONMENT "HOME_CELEPHAIS=${PROJECT_SOURCE_DIR};ASAN_OPTIONS=detect_leaks=0:abort_on_error=1:halt_on_error=1;UBSAN_OPTIONS=print_stacktrace=1:halt_on_error=1"
)

add_test(NAME smoke_bh2d_xcts
    COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/test_bh2d_xcts_smoke.sh
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
)
set_tests_properties(smoke_bh2d_xcts PROPERTIES
    LABELS "smoke"
    TIMEOUT 120
    ENVIRONMENT "HOME_CELEPHAIS=${PROJECT_SOURCE_DIR};ASAN_OPTIONS=detect_leaks=0:abort_on_error=1:halt_on_error=1;UBSAN_OPTIONS=print_stacktrace=1:halt_on_error=1"
)

add_test(NAME smoke_ns2d
    COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/test_ns2d_smoke.sh
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
)
set_tests_properties(smoke_ns2d PROPERTIES
    LABELS "smoke"
    TIMEOUT 90
    ENVIRONMENT "HOME_CELEPHAIS=${PROJECT_SOURCE_DIR};ASAN_OPTIONS=detect_leaks=0:abort_on_error=1:halt_on_error=1;UBSAN_OPTIONS=print_stacktrace=1:halt_on_error=1"
)

add_test(NAME smoke_ns2d_xcts
    COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/test_ns2d_xcts_smoke.sh
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
)

set_tests_properties(smoke_ns2d_xcts PROPERTIES
    LABELS "smoke"
    TIMEOUT 90
    ENVIRONMENT "HOME_CELEPHAIS=${PROJECT_SOURCE_DIR};ASAN_OPTIONS=detect_leaks=0:abort_on_error=1:halt_on_error=1;UBSAN_OPTIONS=print_stacktrace=1:halt_on_error=1"
)

add_test(NAME smoke_ns_nosym
    COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/test_ns_nosym_smoke.sh
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
)
set_tests_properties(smoke_ns_nosym PROPERTIES
    LABELS "smoke"
    TIMEOUT 120
    ENVIRONMENT "HOME_CELEPHAIS=${PROJECT_SOURCE_DIR};ASAN_OPTIONS=detect_leaks=0:abort_on_error=1:halt_on_error=1;UBSAN_OPTIONS=print_stacktrace=1:halt_on_error=1"
)

add_test(NAME smoke_bhns
    COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/test_bhns_smoke.sh
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
)
set_tests_properties(smoke_bhns PROPERTIES
    LABELS "smoke"
    TIMEOUT 300
    ENVIRONMENT "HOME_CELEPHAIS=${PROJECT_SOURCE_DIR};ASAN_OPTIONS=detect_leaks=0:abort_on_error=1:halt_on_error=1;UBSAN_OPTIONS=print_stacktrace=1:halt_on_error=1"
)
