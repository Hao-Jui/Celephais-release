set(APP_NAME BNS)

kadath_add_exec(bns_solve src/wrapper.cpp)
set_target_properties(bns_solve PROPERTIES OUTPUT_NAME solve)

kadath_add_binary_eos_reader(
    bns_reader
    "For_Kadath/Kadath_point_h/kadath_bin_ns.hpp"
    "Kadath::Space_bin_ns"
    "KadathApps::GrFormalism"
    "Usage: ./reader /<path>/<str: BNS ID basename>.toml|.dat  e.g. ./reader converged.TOTAL_BC.9.toml"
)
set_target_properties(bns_reader PROPERTIES OUTPUT_NAME reader)

kadath_add_binary_xz_snapshot(
    bns_xz_snapshot
    "For_Kadath/Kadath_point_h/kadath_bin_ns.hpp"
    "Kadath::Space_bin_ns"
    "KadathApps::GrTraits"
)
set_target_properties(bns_xz_snapshot PROPERTIES OUTPUT_NAME xz_snapshot)

kadath_add_binary_xy_snapshot(
    bns_xy_snapshot
    "For_Kadath/Kadath_point_h/kadath_bin_ns.hpp"
    "Kadath::Space_bin_ns"
    "KadathApps::GrTraits"
)
set_target_properties(bns_xy_snapshot PROPERTIES OUTPUT_NAME xy_snapshot)

kadath_add_binary_coeff_reader(
    bns_coeff_read
    "For_Kadath/Kadath_point_h/kadath_bin_ns.hpp"
    "Kadath::Space_bin_ns"
    "KadathApps::GrTraits"
)
set_target_properties(bns_coeff_read PROPERTIES OUTPUT_NAME coeff_read)

kadath_add_exec(bns_2sacra ${CMAKE_CURRENT_SOURCE_DIR}/../shared/sacra_export_bns.cpp)
set_target_properties(bns_2sacra PROPERTIES OUTPUT_NAME 2sacra)
