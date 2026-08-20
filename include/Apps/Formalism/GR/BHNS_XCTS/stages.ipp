/*
 * Copyright 2022
 * This file is part of the KADATH library and published under
 * https://arxiv.org/abs/2103.09911
 *
 * Author:
 * Samuel D. Tootle <tootle@itp.uni-frankfurt.de>
 * L. Jens Papenfort <papenfort@th.physik.uni-frankfurt.de>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

/*
 * Modifications (Celephais):
 *   2026-06-16  Modified for the Celephais tree; see
 *               PATCHES-KADATH-UPSTREAM.md and LICENSE_SOURCE_AUDIT.tsv.
 *   2026-06-18  Stage bodies hoisted to Apps/Formalism/Shared/Stages/bhns_stages_common.ipp;
 *               these are now thin wrappers supplying the (sym) eqbet definition.
 */

#include "Apps/Formalism/Shared/Stages/bhns_stages_common.ipp"

#include <string>

namespace Kadath {

// Symmetric BHNS: the in-star shift equation has no rotational-gauge forcing.
namespace {
inline std::string bhns_sym_matter_eqbet_def(int /*d*/)
{
    return "eqbet^i= delta * D_j D^j bet^i + delta * D^i D_j bet^j / 3. "
           "- 2. * delta * A^ij * D_j Ntilde - 4. * 4piG * N * P^4 * ptilde^i";
}
} // namespace

template <class eos_t, typename config_t, typename space_t>
int bhns_xcts_solver<eos_t, config_t, space_t>::hydrostatic_equilibrium_stage(STAGE stage, OmegaMode omega_mode,
                                                                              const std::string stage_text)
{
    return bhns_run_hydrostatic_equilibrium_stage(
        *this, stage, omega_mode, stage_text, bhns_sym_matter_eqbet_def,
        [](System_of_eqs&, auto&) -> std::string { return {}; },  // no Pz closure under z-reflection symmetry
        [](auto&, System_of_eqs&) {});                            // no Pz constraint
}

template <class eos_t, typename config_t, typename space_t>
int bhns_xcts_solver<eos_t, config_t, space_t>::hydro_rescaling_stages(STAGE stage, OmegaMode omega_mode,
                                                                      std::string stage_text)
{
    return bhns_run_hydro_rescaling_stages(
        *this, stage, omega_mode, stage_text, bhns_sym_matter_eqbet_def,
        [](System_of_eqs&, auto&) -> std::string { return {}; },  // no Pz closure under z-reflection symmetry
        [](auto&, System_of_eqs&) {});                            // no Pz constraint
}

} // namespace Kadath
