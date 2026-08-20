set(APP_NAME BHNS)

kadath_add_exec(bhns_solve src/wrapper.cpp)
set_target_properties(bhns_solve PROPERTIES OUTPUT_NAME solve)

kadath_add_bhns_eos_reader(
    bhns_reader
    "For_Kadath/Kadath_point_h/kadath_bhns.hpp"
    "Kadath::Space_bhns"
    "Usage: ./reader /<path>/<str: BHNS ID basename>.toml|.dat  e.g. ./reader converged.TOTAL_BC.9.toml"
)
set_target_properties(bhns_reader PROPERTIES OUTPUT_NAME reader)

kadath_add_binary_xz_snapshot(
    bhns_xz_snapshot
    "For_Kadath/Kadath_point_h/kadath_bhns.hpp"
    "Kadath::Space_bhns"
    "KadathApps::GrTraits"
)
set_target_properties(bhns_xz_snapshot PROPERTIES OUTPUT_NAME xz_snapshot)

kadath_add_binary_xy_snapshot(
    bhns_xy_snapshot
    "For_Kadath/Kadath_point_h/kadath_bhns.hpp"
    "Kadath::Space_bhns"
    "KadathApps::GrTraits"
)
set_target_properties(bhns_xy_snapshot PROPERTIES OUTPUT_NAME xy_snapshot)

kadath_add_binary_coeff_reader(
    bhns_coeff_read
    "For_Kadath/Kadath_point_h/kadath_bhns.hpp"
    "Kadath::Space_bhns"
    "KadathApps::GrTraits"
)
set_target_properties(bhns_coeff_read PROPERTIES OUTPUT_NAME coeff_read)

kadath_add_exec(bhns_2sacra src/2sacra.cpp)
set_target_properties(bhns_2sacra PROPERTIES OUTPUT_NAME 2sacra)
