if(NOT DEFINED SOURCE_DIR)
    message(FATAL_ERROR "SOURCE_DIR is required")
endif()

# The boosted-3D BHNS setup was hoisted out of the per-variant
# GR/BHNS_XCTS{,_nosym}/solver_imp.ipp bodies into the shared
# bhns_setup_boosted_3d_impl in PreBinary/bhns_boosted_setup.hpp (both
# variants now delegate to it). The NS-side bispheric bounds call therefore
# lives in that shared impl and in the regrid path; this guard tracks those.
set(BHNS_RBISPH_FILES
    "include/Apps/Formalism/Shared/PreBinary/bhns_boosted_setup.hpp"
    "include/Apps/Formalism/Shared/Regrid/bhns_regrid.hpp"
)

foreach(path IN LISTS BHNS_RBISPH_FILES)
    set(full_path "${SOURCE_DIR}/${path}")
    file(READ "${full_path}" content)

    string(FIND "${content}" "make_binary_NS_bounds(bconfig, BCO1)" modern_found)
    if(modern_found EQUAL -1)
        message(FATAL_ERROR
            "${path} must build the BHNS NS-side bispheric radius through "
            "make_binary_NS_bounds(bconfig, BCO1)"
        )
    endif()

    string(FIND "${content}" "make_NS_bounds(bconfig, BCO1)" legacy_found)
    if(NOT legacy_found EQUAL -1)
        message(FATAL_ERROR
            "${path} still uses legacy make_NS_bounds(bconfig, BCO1); "
            "BHNS must honor [binary] rbisph_fill like BNS"
        )
    endif()
endforeach()
