set(APP_NAME NS)

kadath_add_exec(ns_solve src/wrapper.cpp)
set_target_properties(ns_solve PROPERTIES OUTPUT_NAME solve)

kadath_add_single_eos_reader(
    ns_reader
    "For_Kadath/Kadath_point_h/kadath_adapted.hpp"
    "Kadath::Space_spheric_adapted"
    "KadathApps::GrFormalism"
    "Usage: ./reader /<path>/<ID base name>.toml|.dat  e.g. ./reader converged.NS.9.toml"
)
set_target_properties(ns_reader PROPERTIES OUTPUT_NAME reader)

kadath_add_ns_xz_snapshot(
    ns_xz_snapshot
    "For_Kadath/Kadath_point_h/kadath_adapted.hpp"
    "Kadath::Space_spheric_adapted"
    "KadathApps::GrTraits"
)
set_target_properties(ns_xz_snapshot PROPERTIES OUTPUT_NAME xz_snapshot)

kadath_add_ns_xy_snapshot(
    ns_xy_snapshot
    "For_Kadath/Kadath_point_h/kadath_adapted.hpp"
    "Kadath::Space_spheric_adapted"
    "KadathApps::GrTraits"
)
set_target_properties(ns_xy_snapshot PROPERTIES OUTPUT_NAME xy_snapshot)

kadath_add_ns_coeff_reader(
    ns_coeff_read
    "For_Kadath/Kadath_point_h/kadath_adapted.hpp"
    "Kadath::Space_spheric_adapted"
    "KadathApps::GrTraits"
)
set_target_properties(ns_coeff_read PROPERTIES OUTPUT_NAME coeff_read)

kadath_add_exec(ns_2sacra ${CMAKE_CURRENT_SOURCE_DIR}/../shared/sacra_export_ns.cpp)
set_target_properties(ns_2sacra PROPERTIES OUTPUT_NAME 2sacra)
