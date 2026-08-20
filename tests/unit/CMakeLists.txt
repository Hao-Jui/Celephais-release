include(FetchContent)
FetchContent_Declare(Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG v3.7.1
)
FetchContent_MakeAvailable(Catch2)

add_executable(celephais_unit_tests
    test_dim_array.cpp
    test_array.cpp
    test_matrice.cpp
    test_legendre.cpp
    test_ope_der_1d.cpp
    test_ope_action_operand.cpp
    test_memory_pool.cpp
    test_name_tools.cpp
    test_eos.cpp
    test_tov.cpp
    test_bh_seed.cpp
    test_trumpet.cpp
    test_configurator.cpp
    test_space_spheric.cpp
    test_point_base.cpp
    test_index.cpp
    test_jacobian_column_engine.cpp
    test_jacobian_assembled_coo.cpp
    test_jacobian_coo_hash.cpp
    test_captured_linear_system.cpp
    test_mumps_tree_cache.cpp
    test_mumps_analysis_hints.cpp
    test_mumps_prolongation.cpp
    test_jacobian_group_planner.cpp
    test_jacobian_task_claims.cpp
    test_jacobian_parity_mask.cpp
    test_matching_lane_paths.cpp
    test_import_plan_cache.cpp
    test_column_taxonomy.cpp
    test_column_map.cpp
    test_tagged_jacobian_metadata.cpp
    test_interface_partition.cpp
    test_column_coloring.cpp
    test_sparse_jacobian_compare.cpp
    test_seeded_jacobian.cpp
    test_val_domain.cpp
    test_bns_hp_indicators.cpp
    test_term_eq_derivative_lanes.cpp
    test_levi_civita.cpp
    test_spectral_integration.cpp
    test_spherical_bessel.cpp
    test_coef_precomp.cpp
    test_summation_strided_gather.cpp
    test_transform_line_offsets.cpp
    test_ope_1d_strided.cpp
    test_coef_line_length_contract.cpp
    test_coef_strided_line.cpp
    test_solver_runtime_config.cpp
    test_system_of_eqs_growth.cpp
    test_system_of_eqs_double_der_seeding.cpp
    test_do_jx_jacobian_parity.cpp
    test_system_of_eqs_row_classification.cpp
    test_bns_workflow_amr.cpp
    test_resume_ladder.cpp
    test_bns_per_domain_p_refinement.cpp
    test_spheric_angular_p_refinement.cpp
    test_ns_bounds_rbisph.cpp
    test_system_of_eqs_string_view_api.cpp
    test_equation_parser.cpp
    test_kadath_error.cpp
    test_solver_startup.cpp
    test_ns2d_xcts_lifting.cpp
    test_pn_algebraic_orbital_params.cpp
    test_ns_binary_boost_kinematics.cpp
    test_domain_polar_failures.cpp
    test_domain_bispheric_failures.cpp
    test_param_failures.cpp
    test_domain_shell_symphi_failures.cpp
    test_domain_polar_periodic_failures.cpp
    test_domain_critic_failures.cpp
    test_system_of_eqs_failures.cpp
    test_linear_solver.cpp
    test_krylov_solver.cpp
    test_krylov_gmres.cpp
    test_domain_spheric_periodic_failures.cpp
    test_domain_spheric_time_failures.cpp
    test_domain_oned_failures.cpp
    test_domain_critic_outer_failures.cpp
    test_binary_sink.cpp
    test_be_file_sink.cpp
    test_dim_array_binary.cpp
    test_array_binary.cpp
    test_base_spectral_binary.cpp
    test_base_tensor_binary.cpp
    test_domain_critic_inner_binary.cpp
    test_domain_critic_outer_binary.cpp
    test_domain_oned_binary.cpp
    test_domain_spheric_time_binary.cpp
    test_domain_spheric_periodic_binary.cpp
    test_domain_fourD_periodic_binary.cpp
    test_domain_polar_periodic_binary.cpp
    test_point_binary.cpp
    test_domain_polar_binary.cpp
    test_domain_spheric_binary.cpp
    test_domain_spheric_symphi_binary.cpp
    test_domain_bispheric_binary.cpp
    test_domain_bispheric_nosym_gradient.cpp
    test_domain_bispheric_nosym_der_normal_translation.cpp
    test_domain_homothetic_binary.cpp
    test_space_binary.cpp
    test_bns_mesh.cpp
    test_bns_field_transfer_batch.cpp
    test_bin_ns_nosym_matching.cpp
    test_bin_ns_nosym_dof_parity.cpp
    test_spheric_nosym_smoke.cpp
    test_spheric_nosym_newton_parity.cpp
    test_spheric_nosym_anti_base_setters.cpp
    test_spheric_nosym_integ_volume.cpp
    test_domain_bispheric_integ_volume.cpp
    test_spheric_adapted_nosym_smoke.cpp
    test_spheric_adapted_nosym_base_setters.cpp
    test_spheric_adapted_nosym_newton_parity.cpp
    test_spheric_adapted_nosym_surface_dofs.cpp
    test_spheric_adapted_nosym_square_system.cpp
)

# celephais_solver propagates the solution archive and full MUMPS link closure.
target_link_libraries(celephais_unit_tests PRIVATE
    celephais_solver
    Catch2::Catch2WithMain
)
target_include_directories(celephais_unit_tests PRIVATE
    ${CMAKE_BINARY_DIR}/generated
)
if(CELEPHAIS_ENABLE_FFTW_ORACLE)
    target_compile_definitions(celephais_unit_tests PRIVATE CELEPHAIS_ENABLE_FFTW_ORACLE)
endif()

# MPI — needed because src/Newton was compiled with MPI
find_package(MPI REQUIRED)
target_link_libraries(celephais_unit_tests PRIVATE MPI::MPI_CXX)

list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)
include(Catch)
catch_discover_tests(celephais_unit_tests PROPERTIES
    LABELS "unit"
    ENVIRONMENT "HOME_CELEPHAIS=${CELEPHAIS_ROOT_DIR}"
)

add_test(NAME fft_backend_explicit_selector
    COMMAND celephais_unit_tests "[fft-backend-policy]" --reporter compact)
set_tests_properties(fft_backend_explicit_selector PROPERTIES
    LABELS "unit;fft-backend-policy"
    ENVIRONMENT
        "HOME_CELEPHAIS=${CELEPHAIS_ROOT_DIR};CELEPHAIS_FFT_BACKEND=fftw"
)

add_test(NAME captured_linear_system_mpi_guard
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2
        $<TARGET_FILE:celephais_unit_tests>
        "[captured-linear-system][capture-request-mpi]"
        --reporter compact
)
set_tests_properties(captured_linear_system_mpi_guard PROPERTIES
    LABELS "unit"
    PROCESSORS 2
    TIMEOUT 120
    ENVIRONMENT "HOME_CELEPHAIS=${CELEPHAIS_ROOT_DIR}"
)

add_test(NAME mumps_analysis_metadata_mpi_guard
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2
        $<TARGET_FILE:celephais_unit_tests>
        "[linear_solver][mumps-analysis-metadata]"
        --reporter compact
)
set_tests_properties(mumps_analysis_metadata_mpi_guard PROPERTIES
    LABELS "unit"
    PROCESSORS 2
    TIMEOUT 120
    ENVIRONMENT "HOME_CELEPHAIS=${CELEPHAIS_ROOT_DIR}"
)

add_test(NAME memory_mapper_phase_profile
    COMMAND celephais_unit_tests "[memory-phase-profile]"
)
set_tests_properties(memory_mapper_phase_profile PROPERTIES
    LABELS "unit"
    ENVIRONMENT
        "HOME_CELEPHAIS=${CELEPHAIS_ROOT_DIR};MEMORY_MAPPER_PHASE_PROFILE=1"
)

add_test(NAME test_solver_runtime_config
    COMMAND celephais_unit_tests "[solver-runtime-config]"
)
set_tests_properties(test_solver_runtime_config PROPERTIES
    LABELS "unit"
    ENVIRONMENT "HOME_CELEPHAIS=${CELEPHAIS_ROOT_DIR}"
)

add_test(NAME solver_runtime_config_guard
    COMMAND ${CMAKE_COMMAND}
        -DSOURCE_DIR=${PROJECT_SOURCE_DIR}
        -P ${CMAKE_CURRENT_SOURCE_DIR}/solver_runtime_config_guard.cmake
)
set_tests_properties(solver_runtime_config_guard PROPERTIES
    LABELS "unit"
)

add_test(NAME runtime_env_solver_guard
    COMMAND ${CMAKE_COMMAND}
        -DSOURCE_DIR=${PROJECT_SOURCE_DIR}
        -P ${CMAKE_CURRENT_SOURCE_DIR}/runtime_env_solver_guard.cmake
)
set_tests_properties(runtime_env_solver_guard PROPERTIES
    LABELS "unit"
)

add_test(NAME config_layer_guard
    COMMAND ${CMAKE_COMMAND}
        -DSOURCE_DIR=${PROJECT_SOURCE_DIR}
        -P ${CMAKE_CURRENT_SOURCE_DIR}/config_layer_guard.cmake
)
set_tests_properties(config_layer_guard PROPERTIES
    LABELS "unit"
)

add_test(NAME include_surface_guard
    COMMAND ${CMAKE_COMMAND}
        -DSOURCE_DIR=${PROJECT_SOURCE_DIR}
        -P ${CMAKE_CURRENT_SOURCE_DIR}/include_surface_guard.cmake
)
set_tests_properties(include_surface_guard PROPERTIES
    LABELS "unit"
)

add_test(NAME regression_baseline_input_guard
    COMMAND ${CMAKE_COMMAND}
        -DSOURCE_DIR=${PROJECT_SOURCE_DIR}
        -P ${CMAKE_CURRENT_SOURCE_DIR}/regression_baseline_input_guard.cmake
)
set_tests_properties(regression_baseline_input_guard PROPERTIES
    LABELS "unit"
)

add_test(NAME docs_directory_guard
    COMMAND ${CMAKE_COMMAND}
        -DSOURCE_DIR=${PROJECT_SOURCE_DIR}
        -P ${CMAKE_CURRENT_SOURCE_DIR}/docs_directory_guard.cmake
)
set_tests_properties(docs_directory_guard PROPERTIES
    LABELS "unit"
)


add_test(NAME nosym_residual_row_descriptor_guard
    COMMAND ${CMAKE_COMMAND}
        -DSOURCE_DIR=${PROJECT_SOURCE_DIR}
        -P ${CMAKE_CURRENT_SOURCE_DIR}/nosym_residual_row_descriptor_guard.cmake
)
set_tests_properties(nosym_residual_row_descriptor_guard PROPERTIES
    LABELS "unit"
)

add_test(NAME name_tools_modern_guard
    COMMAND ${CMAKE_COMMAND}
        -DSOURCE_DIR=${PROJECT_SOURCE_DIR}
        -P ${CMAKE_CURRENT_SOURCE_DIR}/name_tools_modern_guard.cmake
)
set_tests_properties(name_tools_modern_guard PROPERTIES
    LABELS "unit"
)

add_test(NAME onboarding_tools_guard
    COMMAND ${CMAKE_COMMAND}
        -DSOURCE_DIR=${PROJECT_SOURCE_DIR}
        -P ${CMAKE_CURRENT_SOURCE_DIR}/onboarding_tools_guard.cmake
)
set_tests_properties(onboarding_tools_guard PROPERTIES
    LABELS "unit"
)

add_test(NAME bhns_rbisph_bounds_guard
    COMMAND ${CMAKE_COMMAND}
        -DSOURCE_DIR=${PROJECT_SOURCE_DIR}
        -P ${CMAKE_CURRENT_SOURCE_DIR}/bhns_rbisph_bounds_guard.cmake
)
set_tests_properties(bhns_rbisph_bounds_guard PROPERTIES
    LABELS "unit"
)
