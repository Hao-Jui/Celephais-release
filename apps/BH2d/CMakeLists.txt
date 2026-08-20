set(APP_NAME BH2d)

kadath_add_exec(bh2d_solve src/wrapper.cpp)
set_target_properties(bh2d_solve PROPERTIES OUTPUT_NAME solve)

kadath_add_exec(bh2d_reader src/reader.cpp)
set_target_properties(bh2d_reader PROPERTIES OUTPUT_NAME reader)

kadath_add_bh2d_x_profile(
    bh2d_x_profile
    "For_Kadath/Space/adapted_bh_polar.hpp"
    "Kadath::Space_adapted_bh_polar"
    "KadathApps::Bh2dMsqiProfileFields"
)
set_target_properties(bh2d_x_profile PROPERTIES OUTPUT_NAME x_proflle)
