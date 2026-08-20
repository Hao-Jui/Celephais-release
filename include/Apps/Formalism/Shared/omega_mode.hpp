/*
 * Copyright 2022
 * This file is part of the KADATH library and published under
 * https://arxiv.org/abs/2103.09911
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
#pragma once

namespace Kadath {

// ---------------------------------------------------------------------------
// Orbital-frequency (Omega) treatment selector for binary XCTS stages.
//
// A binary stage runs either as a fixed-Omega warm-up or as the free-Omega
// quasi-equilibrium pass. The selector is an explicit OmegaMode argument: the
// warm-up decision (`want_warmup`) is computed once by the caller (the workflow
// rung loop) and never mutates the saved config flag `FIXED_GOMEGA`, avoiding
// the historical conflation of user intent with run state.
// ---------------------------------------------------------------------------

// Which orbital-frequency treatment a binary stage runs under.
//   Fixed : the orbital frequency "ome" (and, where an app does so, the
//           x-axis center of mass) is held as a constant (add_cst); the
//           quasi-equilibrium / ADM-momentum integral constraints that would
//           otherwise determine it are NOT added. This is the warm-up pass.
//   Free  : "ome" (and the x-axis) is solved for (add_var) and the
//           quasi-equilibrium / momentum constraints are added. This is the
//           physical quasi-equilibrium pass.
enum class OmegaMode { Fixed, Free };

} // namespace Kadath
