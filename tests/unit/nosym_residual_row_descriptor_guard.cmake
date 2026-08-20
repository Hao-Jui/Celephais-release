if(NOT DEFINED SOURCE_DIR)
    message(FATAL_ERROR "SOURCE_DIR is required")
endif()

file(READ
    "${SOURCE_DIR}/include/Apps/Formalism/GR/NS_3D_XCTS_nosym/norot_stage.ipp"
    NOROT_STAGE
)
file(READ
    "${SOURCE_DIR}/include/Apps/Formalism/Shared/Stages/bns_stages_common.ipp"
    BNS_STAGES
)

foreach(REQUIRED_FRAGMENT IN ITEMS
        "syst.add_eq_first_integral(0, 1, \"firstint\", \"H - Hc\", true);"
        "space.add_eq_int_volume(syst, 2, \"integvolume(intMb) = Mb\");\n        syst.set_last_eq_int_reflection_sector(+1);"
        "space.add_eq_int_inf(syst, \"integ(intMadm) = Madm\");\n        syst.set_last_eq_int_reflection_sector(+1);")
    string(FIND "${NOROT_STAGE}" "${REQUIRED_FRAGMENT}" FOUND_AT)
    if(FOUND_AT EQUAL -1)
        message(FATAL_ERROR
            "NS no-sym NOROT residual-row metadata missing: ${REQUIRED_FRAGMENT}"
        )
    endif()
endforeach()

foreach(REQUIRED_FRAGMENT IN ITEMS
        "solver.space.add_eq_int_inf(syst, \"integ(intPx) = 0\");\n        // Px closes yaxis through the x-directed shift term.\n        syst.set_last_eq_int_reflection_sector(-1);"
        "solver.space.add_eq_int_inf(syst, \"integ(intPy) = 0\");\n        // Py closes xaxis through the y-directed shift term.\n        syst.set_last_eq_int_reflection_sector(+1);")
    string(FIND "${BNS_STAGES}" "${REQUIRED_FRAGMENT}" FOUND_AT)
    if(FOUND_AT EQUAL -1)
        message(FATAL_ERROR
            "BNS no-sym eccentric residual-row metadata missing: ${REQUIRED_FRAGMENT}"
        )
    endif()
endforeach()

string(REGEX MATCHALL
    "V\\^i \\* D_i H = 0\", -1, nullptr, \"phi\""
    PROJECTED_PHI_OWNER_CALLS
    "${BNS_STAGES}"
)
list(LENGTH PROJECTED_PHI_OWNER_CALLS PROJECTED_PHI_OWNER_COUNT)
if(NOT PROJECTED_PHI_OWNER_COUNT EQUAL 4)
    message(FATAL_ERROR
        "BNS projected velocity-potential BCs must all carry owner phi; found ${PROJECTED_PHI_OWNER_COUNT}/4"
    )
endif()

message(STATUS "No-sym residual-row descriptor stage metadata guard passed")
