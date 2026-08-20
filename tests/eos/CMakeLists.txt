# eos_hygiene — cold EOS table validation diagnostic (manual, run on demand).
#
# Built under BUILD_TESTING but deliberately NOT registered as a ctest: it is a
# correctness probe, not an infra smoke check, and it hard-FAILs on tables that
# go superluminal (cs2 > 1) in their high-density tail. Both shipped demo tables
# (include/Hydro/Demos/{mpa1.polytrope,togashi.lorene}) are acausal above the
# physical NS density range and so fail the full-range C5 check by design —
# registering them as gating tests would redden the default suite for no defect.
#
# Run on demand:
#   cmake --build build --target eos_hygiene
#   ./build/tests/eos/eos_hygiene <path/to/table.{lorene,polytrope}> [--n N] [--rtol TOL]
# Exit 0 = all hard checks pass, 1 = a check failed, 2 = bad args / unknown extension.
add_executable(eos_hygiene eos_hygiene.cpp)
target_link_libraries(eos_hygiene PRIVATE celephais_solver)
target_compile_options(eos_hygiene PRIVATE
    -Wall -Wextra -Wno-unused-parameter -Woverloaded-virtual
)
target_include_directories(eos_hygiene PRIVATE ${CELEPHAIS_ROOT_DIR}/include)

# tov_maxmass — TOV M(rho_c) / M-R probe for a cold EOS (manual, run on demand).
#
# Sweeps central density, reports the stable-branch maximum mass and the
# TOV-consistent (rho_c, Mb, R) seed for a target mass. Like eos_hygiene it is a
# diagnostic, not a ctest. Run on demand:
#   cmake --build build --target tov_maxmass
#   ./build/tests/eos/tov_maxmass <table.{lorene,polytrope}> [interp_pts] [rho_lo] [rho_hi] [N]
add_executable(tov_maxmass tov_maxmass.cpp)
target_link_libraries(tov_maxmass PRIVATE celephais_solver)
target_compile_options(tov_maxmass PRIVATE
    -Wall -Wextra -Wno-unused-parameter -Woverloaded-virtual
)
target_include_directories(tov_maxmass PRIVATE ${CELEPHAIS_ROOT_DIR}/include)
