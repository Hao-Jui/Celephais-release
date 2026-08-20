set(APP_NAME NS2d)

kadath_add_exec(ns2d_solve src/wrapper.cpp)
set_target_properties(ns2d_solve PROPERTIES OUTPUT_NAME solve)

kadath_add_exec(ns2d_reader src/reader.cpp)
set_target_properties(ns2d_reader PROPERTIES OUTPUT_NAME reader)

kadath_add_exec(ns2d_2sacra ${CMAKE_CURRENT_SOURCE_DIR}/../shared/sacra_export_ns.cpp)
set_target_properties(ns2d_2sacra PROPERTIES OUTPUT_NAME 2sacra)

kadath_add_ns2d_x_profile(
    ns2d_x_profile
    "For_Kadath/Kadath_point_h/kadath_adapted_polar.hpp"
    "Kadath::Space_polar_adapted"
    "KadathApps::Ns2dMsqiProfileFields"
)
set_target_properties(ns2d_x_profile PROPERTIES OUTPUT_NAME x_proflle)
