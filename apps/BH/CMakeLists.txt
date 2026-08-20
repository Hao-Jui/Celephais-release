set(APP_NAME BH)

kadath_add_exec(bh_solve src/wrapper.cpp)
set_target_properties(bh_solve PROPERTIES OUTPUT_NAME solve)

kadath_add_single_bh_reader(
    bh_reader
    "For_Kadath/Kadath_point_h/kadath_adapted_bh.hpp"
    "Kadath::Space_adapted_bh"
    "Usage: ./reader /<path>/<ID base name>.toml|.dat  e.g. ./reader converged.BH.9.toml"
)
set_target_properties(bh_reader PROPERTIES OUTPUT_NAME reader)
