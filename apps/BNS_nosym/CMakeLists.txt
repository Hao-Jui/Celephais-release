set(APP_NAME BNS_nosym)

kadath_add_exec(bns_nosym_solve src/wrapper.cpp)
set_target_properties(bns_nosym_solve PROPERTIES OUTPUT_NAME solve)

kadath_add_binary_eos_reader(
    bns_nosym_reader
    "For_Kadath/Kadath_point_h/kadath_bin_ns_nosym.hpp"
    "Kadath::Space_bin_ns_nosym"
    "KadathApps::GrFormalism"
    "Usage: ./reader /<path>/<str: BNS ID basename>.toml|.dat  e.g. ./reader converged.TOTAL_BC.9.toml"
)
set_target_properties(bns_nosym_reader PROPERTIES OUTPUT_NAME reader)

kadath_add_binary_xz_snapshot(
    bns_nosym_xz_snapshot
    "For_Kadath/Kadath_point_h/kadath_bin_ns_nosym.hpp"
    "Kadath::Space_bin_ns_nosym"
    "KadathApps::GrTraits"
)
set_target_properties(bns_nosym_xz_snapshot PROPERTIES OUTPUT_NAME xz_snapshot)

kadath_add_binary_xy_snapshot(
    bns_nosym_xy_snapshot
    "For_Kadath/Kadath_point_h/kadath_bin_ns_nosym.hpp"
    "Kadath::Space_bin_ns_nosym"
    "KadathApps::GrTraits"
)
set_target_properties(bns_nosym_xy_snapshot PROPERTIES OUTPUT_NAME xy_snapshot)

kadath_add_binary_coeff_reader(
    bns_nosym_coeff_read
    "For_Kadath/Kadath_point_h/kadath_bin_ns_nosym.hpp"
    "Kadath::Space_bin_ns_nosym"
    "KadathApps::GrTraits"
)
set_target_properties(bns_nosym_coeff_read PROPERTIES OUTPUT_NAME coeff_read)

kadath_add_exec(bns_nosym_2sacra src/2sacra.cpp)
set_target_properties(bns_nosym_2sacra PROPERTIES OUTPUT_NAME 2sacra)

