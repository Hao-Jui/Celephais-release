set(APP_NAME NS_nosym)

kadath_add_exec(ns_nosym_solve src/wrapper.cpp)
set_target_properties(ns_nosym_solve PROPERTIES OUTPUT_NAME solve)

kadath_add_single_eos_reader(
    ns_nosym_reader
    "For_Kadath/Domain/spheric_adapted_nosym.hpp"
    "Kadath::Space_spheric_adapted_nosym"
    "KadathApps::GrFormalism"
    "Usage: ./reader /<path>/<ID base name>.toml|.dat  e.g. ./reader converged.NS_nosym.9.toml"
)
set_target_properties(ns_nosym_reader PROPERTIES OUTPUT_NAME reader)

kadath_add_ns_xz_snapshot(
    ns_nosym_xz_snapshot
    "For_Kadath/Domain/spheric_adapted_nosym.hpp"
    "Kadath::Space_spheric_adapted_nosym"
    "KadathApps::GrTraits"
)
set_target_properties(ns_nosym_xz_snapshot PROPERTIES OUTPUT_NAME xz_snapshot)

kadath_add_ns_xy_snapshot(
    ns_nosym_xy_snapshot
    "For_Kadath/Domain/spheric_adapted_nosym.hpp"
    "Kadath::Space_spheric_adapted_nosym"
    "KadathApps::GrTraits"
)
set_target_properties(ns_nosym_xy_snapshot PROPERTIES OUTPUT_NAME xy_snapshot)

kadath_add_ns_coeff_reader(
    ns_nosym_coeff_read
    "For_Kadath/Domain/spheric_adapted_nosym.hpp"
    "Kadath::Space_spheric_adapted_nosym"
    "KadathApps::GrTraits"
)
set_target_properties(ns_nosym_coeff_read PROPERTIES OUTPUT_NAME coeff_read)
