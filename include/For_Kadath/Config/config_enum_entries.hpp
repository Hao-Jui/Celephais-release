/*
 * This file is part of the KADATH library.
 *
 * Central entry lists for Configurator enums and TOML keys. Each list preserves
 * enum order; the optional string is the serialized TOML key. Empty keys mark
 * internal or retired enum slots that remain array indexes but are not emitted
 * as runtime config keys.
 *
 * X-macro convention: each KADATH_*_ENTRIES(APPLY, CTX) macro expands to a
 * sequence of APPLY(CTX, enumerator, key) invocations. CTX is an opaque token
 * forwarded to the per-entry callback so the same entry list can drive enum
 * generation, count checks, and typed entry-array construction without
 * duplicating the entry sequence.
 */

#pragma once

#define KADATH_BIN_PARAM_ENTRIES(APPLY, CTX) \
    APPLY(CTX, DIST, "distance")             \
    APPLY(CTX, DDIST, "d_dist")              \
    APPLY(CTX, BIN_RES, "res")               \
    APPLY(CTX, GOMEGA, "global_omega")       \
    APPLY(CTX, COM, "com")                   \
    APPLY(CTX, COMY, "comy")                 \
    APPLY(CTX, REXT, "rext")                 \
    APPLY(CTX, Q, "q")                       \
    APPLY(CTX, ADOT, "adot")                 \
    APPLY(CTX, ECC_OMEGA, "ecc_omega")       \
    APPLY(CTX, OUTER_SHELLS, "outer_shells") \
    APPLY(CTX, COM_VELZ, "com_velz")         \
    APPLY(CTX, RBISPH_FILL, "rbisph_fill")

#define KADATH_TRI_PARAM_ENTRIES(APPLY, CTX)     \
    APPLY(CTX, PARENT_DIST, "parent_distance")   \
    APPLY(CTX, CHILD_DIST, "child_distance")     \
    APPLY(CTX, CHILD_REXT, "child_outer_radius") \
    APPLY(CTX, TRI_RES, "res")                   \
    APPLY(CTX, TRI_GOMEGA, "global_omega")       \
    APPLY(CTX, TRI_REXT, "rext")                 \
    APPLY(CTX, TRI_OUTER_SHELLS, "outer_shells")

#define KADATH_BCO_PARAM_ENTRIES(APPLY, CTX) \
    APPLY(CTX, RIN, "rin")                   \
    APPLY(CTX, BCO_RES, "res")               \
    APPLY(CTX, RMID, "rmid")                 \
    APPLY(CTX, FIXED_R, "fixed_r")           \
    APPLY(CTX, ROUT, "rout")                 \
    APPLY(CTX, NSHELLS, "nshells")           \
    APPLY(CTX, OMEGA, "omega")               \
    APPLY(CTX, CHI, "chi")                   \
    APPLY(CTX, MIRR, "mirr")                 \
    APPLY(CTX, MCH, "mch")                   \
    APPLY(CTX, MB, "mb")                     \
    APPLY(CTX, NC, "nc")                     \
    APPLY(CTX, HC, "hc")                     \
    APPLY(CTX, FIXED_LAPSE, "fixed_lapse")   \
    APPLY(CTX, MADM, "madm")                 \
    APPLY(CTX, QLMADM, "ql_madm")            \
    APPLY(CTX, DIM, "dim")                   \
    APPLY(CTX, USE_TOV1D, "use_tov1d")              \
    APPLY(CTX, TRUMPET_BH_SEED, "trumpet_bh_seed")  \
    APPLY(CTX, FIXED_BCOMEGA, "fixed_omega")        \
    APPLY(CTX, DEG, "deg")                          \
    APPLY(CTX, BVELX, "velx")                       \
    APPLY(CTX, BVELY, "vely")                       \
    APPLY(CTX, DECAY, "decay_limit")                \
    APPLY(CTX, KERR_CHI, "kerr_chi")                \
    APPLY(CTX, KERR_MCH, "kerr_mch")                \
    APPLY(CTX, NINSHELLS, "n_inner_shells")

#define KADATH_DIFFROT_PARAM_ENTRIES(APPLY, CTX) \
    APPLY(CTX, DIFF_LAW, "rot_law")              \
    APPLY(CTX, CSTA, "cstA")                     \
    APPLY(CTX, CSTP, "cstp")                     \
    APPLY(CTX, LAM1, "lam1")                     \
    APPLY(CTX, LAM2, "lam2")

#define KADATH_GRAVITY_ENTRIES(APPLY, CTX) \
    APPLY(CTX, GRAV_THEORY, "theory")      \
    APPLY(CTX, GRAV_MASS_SCAL, "massScal") \
    APPLY(CTX, GRAV_CSTB, "cstB")          \
    APPLY(CTX, GRAV_RETIRED_SCALAR_OFFSET, "") \
    APPLY(CTX, GRAV_ALPHA_GB, "alphaGB")   \
    APPLY(CTX, GRAV_COUPLING_TYPE, "coupling")

#define KADATH_EOS_PARAM_ENTRIES(APPLY, CTX) \
    APPLY(CTX, EOSTYPE, "eostype")           \
    APPLY(CTX, EOSFILE, "eosfile")           \
    APPLY(CTX, HCUT, "h_cut")                \
    APPLY(CTX, INTERP_PTS, "interpolation_pts") \
    APPLY(CTX, MNUC_CGS, "mnuc_cgs")

#define KADATH_NODE_ENTRIES(APPLY, CTX)         \
    APPLY(CTX, BCO1, "")                        \
    APPLY(CTX, BCO2, "")                        \
    APPLY(CTX, BCO3, "")                        \
    APPLY(CTX, BINARY, "")                      \
    APPLY(CTX, BH, "bh")                        \
    APPLY(CTX, NS, "ns")                        \
    APPLY(CTX, FIELDS, "fields")                \
    APPLY(CTX, STAGES, "stages")                \
    APPLY(CTX, SCONTROLS, "sequence_controls")  \
    APPLY(CTX, SSETTINGS, "sequence_settings")   \
    APPLY(CTX, AMR, "adaptive_mesh_refinement")

#define KADATH_BCO_FIELD_ENTRIES(APPLY, CTX) \
    APPLY(CTX, CONF, "conf")                 \
    APPLY(CTX, LAPSE, "lapse")               \
    APPLY(CTX, SHIFT, "shift")               \
    APPLY(CTX, ENTH, "enth")                 \
    APPLY(CTX, LOGH, "logh")                 \
    APPLY(CTX, Omg, "Omg")                   \
    APPLY(CTX, NDENS, "ndens")               \
    APPLY(CTX, PHI, "phi")                   \
    APPLY(CTX, SWEIGHT, "sweight")           \
    APPLY(CTX, NU, "nu")                     \
    APPLY(CTX, INCA, "incA")                 \
    APPLY(CTX, BIGA, "bigA")                 \
    APPLY(CTX, NP, "np")                     \
    APPLY(CTX, KS_METRIC, "ks_metric")       \
    APPLY(CTX, KS_LAPSE, "ks_lapse")         \
    APPLY(CTX, KS_K, "ks_k")

/*
 * Stage names are per-physics (the old TOTAL/TOTAL_BC pair meant a different
 * solve in every app). Enum order encodes solve order within each app:
 *   NS:      NOROT < UNIROT < BIN_BOOST
 *   BH:      DIRICHLET_LAPSE < VON_NEUMANN < BIN_BOOST
 *   KSBH:    TRUMPET < BIN_BOOST
 *   binary:  workflow order is QUASI_EQUIL -> FORCE_BALANCE -> ECC_RED
 *            (kept in Apps/Formalism/Shared/binary_driver_common.hpp, not enum order)
 * Legacy v1 keys (norot_bc/total/total_bc) are REJECTED at read time
 * (app-ambiguous); migrate datasets once with
 * developer/migrate_stage_filenames.sh.
 */
#define KADATH_STAGE_ENTRIES(APPLY, CTX)             \
    APPLY(CTX, FIXED_OMEGA, "fixed_omega")           \
    APPLY(CTX, NOROT, "norot")                       \
    APPLY(CTX, COROT_EQUAL, "corot_equal")           \
    APPLY(CTX, DIRICHLET_LAPSE, "dirichlet_lapse")   \
    APPLY(CTX, FORCE_BALANCE, "force_balance")       \
    APPLY(CTX, UNIROT, "uniform_rot")                \
    APPLY(CTX, VON_NEUMANN, "von_neumann")           \
    APPLY(CTX, TRUMPET, "trumpet")                   \
    APPLY(CTX, QUASI_EQUIL, "quasi_equilibrium")     \
    APPLY(CTX, TOTAL_FIXED_COM, "total_fixed_com")   \
    APPLY(CTX, TESTING, "testing")                   \
    APPLY(CTX, GRAV, "grav")                         \
    APPLY(CTX, VEL_POT_ONLY, "vel_pot_only")         \
    APPLY(CTX, ECC_RED, "ecc_red")                   \
    APPLY(CTX, BIN_BOOST, "binary_boost")

#define KADATH_CONTROL_ENTRIES(APPLY, CTX)           \
    APPLY(CTX, USE_PN, "use_pn")                     \
    APPLY(CTX, CHECKPOINT, "checkpoint")             \
    APPLY(CTX, MB_FIXING, "fixed_mb")                \
    APPLY(CTX, COROT_BIN, "corot_binary")            \
    APPLY(CTX, FIXED_GOMEGA, "fixed_bin_omega")      \
    APPLY(CTX, USE_FIXED_LAPSE, "fixed_lapse")       \
    APPLY(CTX, REGRID, "initial_regrid")             \
    APPLY(CTX, SCALAR_CONTROL, "scalar_control")

#define KADATH_SEQ_SETTING_ENTRIES(APPLY, CTX)       \
    APPLY(CTX, PREC, "solver_precision")             \
    APPLY(CTX, MAX_ITER, "solver_max_iterations")    \
    APPLY(CTX, INIT_RES, "initial_resolution")       \
    APPLY(CTX, FINAL_CHI, "")

#define KADATH_AMR_SETTING_ENTRIES(APPLY, CTX)             \
    APPLY(CTX, AMR_ENABLED, "enabled")                     \
    APPLY(CTX, AMR_MAX_CYCLES, "max_cycles")               \
    APPLY(CTX, AMR_TAIL_WIDTH, "tail_width")               \
    APPLY(CTX, AMR_L2_TAIL_THRESHOLD, "l2_tail_threshold") \
    APPLY(CTX, AMR_LINF_TAIL_THRESHOLD, "linf_tail_threshold") \
    APPLY(CTX, AMR_MODE, "mode")                           \
    APPLY(CTX, AMR_NORM_FLOOR, "norm_floor")               \
    APPLY(CTX, AMR_MAX_NR, "max_nr")                       \
    APPLY(CTX, AMR_MAX_SHELLS, "max_shells")               \
    APPLY(CTX, AMR_REFINE_BISPHERIC, "refine_bispheric")   \
    APPLY(CTX, AMR_H_GAIN_MIN, "h_gain_min")               \
    APPLY(CTX, AMR_H_STALL_CYCLES, "h_stall_cycles")       \
    APPLY(CTX, AMR_H_SATURATION_LINF, "h_saturation_linf") \
    APPLY(CTX, AMR_H_ANGULAR_DOMINANCE, "h_angular_dominance") \
    APPLY(CTX, AMR_P_DOF_GROWTH_LIMIT, "p_dof_growth_limit") \
    APPLY(CTX, AMR_P_FALLBACK_DOF_GROWTH_LIMIT, "p_fallback_dof_growth_limit") \
    APPLY(CTX, AMR_P_FALLBACK_MAX_CANDIDATES, "p_fallback_max_candidates")

/*
 * Aggregate macro: invoke
 *   X(EnumName, NumSym, EntriesMacro, ArrayName, DefaultFnName, HasDefault)
 * once per Configurator enum. Drives header-side enum declarations and cpp-side
 * entry-array + default-map-accessor generation from a single registry.
 *
 * HasDefault == 1 emits a default `MFoo()` accessor returning the full
 * key->enum map for that enum. HasDefault == 0 skips that emission (e.g. NODES
 * only exposes curated subset accessors). When HasDefault == 0, the DefaultFnName
 * slot is unused; pass a placeholder identifier.
 */
#define KADATH_CONFIG_ENUMS(X)                                                                                          \
    X(BIN_PARAMS,    NUM_BPARAMS,         KADATH_BIN_PARAM_ENTRIES,     binary_parameter_entries,         MBIN_PARAMS,     1) \
    X(TRI_PARAMS,    NUM_TRI_PARAMS,      KADATH_TRI_PARAM_ENTRIES,     three_body_parameter_entries,     MTRI_PARAMS,     1) \
    X(BCO_PARAMS,    NUM_BCO_PARAMS,      KADATH_BCO_PARAM_ENTRIES,     compact_object_parameter_entries, MBCO_PARAMS,     1) \
    X(DIFFROT_PARAMS,NUM_DIFFROT_PARAMS,  KADATH_DIFFROT_PARAM_ENTRIES, differential_rotation_entries,    MDIFFROT_PARAMS, 1) \
    X(EOS_PARAMS,    NUM_EOS_PARAMS,      KADATH_EOS_PARAM_ENTRIES,     equation_of_state_entries,        MEOS_PARAMS,     1) \
    X(GRAVITY_PARAMS,NUM_GRAVITY_PARAMS,  KADATH_GRAVITY_ENTRIES,       gravity_parameter_entries,        MGRAVITY_PARAMS, 1) \
    X(NODES,         NUM_NODES,           KADATH_NODE_ENTRIES,          node_entries,                     NODES_UNUSED,    0) \
    X(BCO_FIELDS,    NUM_BCO_FIELDS,      KADATH_BCO_FIELD_ENTRIES,     field_entries,                    MBCO_FIELDS,     1) \
    X(STAGE,         NUM_STAGES,          KADATH_STAGE_ENTRIES,         stage_entries,                    MSTAGE,          1) \
    X(CONTROLS,      NUM_CONTROLS,        KADATH_CONTROL_ENTRIES,       control_entries,                  MCONTROLS,       1) \
    X(SEQ_SETTINGS,  NUM_SEQ_SETTINGS,    KADATH_SEQ_SETTING_ENTRIES,   sequence_setting_entries,         MSEQ_SETTINGS,   1) \
    X(AMR_SETTINGS,  NUM_AMR_SETTINGS,    KADATH_AMR_SETTING_ENTRIES,   amr_setting_entries,              MAMR_SETTINGS,   1)
