set(APP_NAME NS2d_xcts)

kadath_add_exec(ns2d_xcts_solve src/wrapper.cpp)
set_target_properties(ns2d_xcts_solve PROPERTIES OUTPUT_NAME solve)

kadath_add_2d_eos_reader(
    ns2d_xcts_reader
    "For_Kadath/Kadath_point_h/kadath_adapted_polar.hpp"
    "Kadath::Space_polar_adapted"
    "KadathApps::GrFormalism"
    "Usage: ./reader /<path>/<ID base name>.toml|.dat  e.g. ./reader converged_NS_2D.toml"
)
set_target_properties(ns2d_xcts_reader PROPERTIES OUTPUT_NAME reader)

kadath_add_ns2d_x_profile(
    ns2d_xcts_x_profile
    "For_Kadath/Kadath_point_h/kadath_adapted_polar.hpp"
    "Kadath::Space_polar_adapted"
    "KadathApps::Ns2dXctsGrProfileFields"
)
set_target_properties(ns2d_xcts_x_profile PROPERTIES OUTPUT_NAME x_proflle)
