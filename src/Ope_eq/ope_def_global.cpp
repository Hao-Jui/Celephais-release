/*
    Copyright 2017 Philippe Grandclement

    This file is part of Kadath.

    Kadath is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    Kadath is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with Kadath.  If not, see <http://www.gnu.org/licenses/>.
*/

/*
 * Modifications (Celephais):
 *   2026-06-16  Modified for the Celephais tree; see
 *               PATCHES-KADATH-UPSTREAM.md and LICENSE_SOURCE_AUDIT.tsv.
 *   2026-08-06  RAII/span modernization.
 */

#include "For_Kadath/System_of_eqs/system_of_eqs.hpp"
#include "For_Kadath/Array/exceptions.hpp"
#include "For_Kadath/Ope_eq/ope_eq.hpp"
#include "For_Kadath/Term_eq/term_eq.hpp"
#include <algorithm>
#include <array>
namespace Kadath
{
    Ope_def_global::Ope_def_global(const System_of_eqs* zesys, int zedom, const char* name_ope)
        : Ope_eq(zesys, zedom, zesys->get_dom_max() - zesys->get_dom_min() + 1)
    {
        for (int n = 0; n < n_ope; n++)
            parts[n].reset(syst->give_ope(n + syst->get_dom_min(), name_ope));

        res = std::make_unique<Term_eq>(dom, 0., 0.);
        auxi.resize(static_cast<std::size_t>(n_ope));
        compute_res();
    }

    Ope_def_global::~Ope_def_global() = default;

    Term_eq Ope_def_global::action() const
    {
        ScopedOpeActionProfile ope_action_profile_scope(*this);
        /*if (!called) {
            res = new Term_eq(parts[0]->action()) ;
            called = true ;
        }
        return *res ;*/
        KADATH_THROW("Should not call Ope_def_global::action");
    }

    Term_eq* Ope_def_global::get_res()
    {
        return res.get();
    }

    void Ope_def_global::compute_res()
    {
        bool doder = true;
        double val = 0;
        double der = 0;
        int derivative_lane_count = 1;
        std::array<double, Term_eq::max_derivative_lanes - 1> der_lanes{};

        for (int i = 0; i < n_ope; i++) {
            if (auxi[i] == nullptr)
                auxi[i] = std::make_unique<Term_eq>(parts[i]->action());
            else
                *auxi[i] = parts[i]->action();

            val += *auxi[i]->val_d;
            if (auxi[i]->der_d == nullptr)
                doder = false;
            else
                der += *auxi[i]->der_d;
            derivative_lane_count = std::max(derivative_lane_count, auxi[i]->get_derivative_lane_count());
            for (int lane = 1; lane < Term_eq::max_derivative_lanes; ++lane) {
                if (auxi[i]->has_der_d(lane))
                    der_lanes[static_cast<std::size_t>(lane - 1)] += auxi[i]->get_der_d(lane);
            }
        }

        if (!doder)
            *res = Term_eq(dom, val);
        else {
            *res = Term_eq(dom, val, der);
            res->set_derivative_lane_count(derivative_lane_count);
            for (int lane = 1; lane < derivative_lane_count; ++lane)
                res->set_der_d(lane, der_lanes[static_cast<std::size_t>(lane - 1)]);
        }
    }
} // namespace Kadath
