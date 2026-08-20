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

#include "For_Kadath/Ope_eq/ope_eq.hpp"
#include "For_Kadath/Scalar/scalar.hpp"
#include "For_Kadath/Tensor/tensor.hpp"
#include "For_Kadath/System_of_eqs/system_of_eqs.hpp"
#include <memory>
#include <vector>
namespace Kadath
{
    Ope_import::Ope_import(const System_of_eqs* zesys, int dd, int bb, const char* target)
        : Ope_eq(zesys, dd), bound(bb), others(syst->get_space().get_indices_matching_non_std(dd, bb))
    {

        n_ope = others.get_size(1);
        parts.resize(static_cast<std::size_t>(n_ope));
        for (int i = 0; i < n_ope; i++)
            parts[i].reset(syst->give_ope(others(0, i), target, others(1, i)));
    }

    Ope_import::~Ope_import() {}

    Term_eq Ope_import::action() const
    {
        ScopedOpeActionProfile ope_action_profile_scope(*this);

        std::vector<std::unique_ptr<Term_eq>> owned_res(static_cast<std::size_t>(n_ope));
        std::vector<Term_eq*> res(static_cast<std::size_t>(n_ope));
        for (int i = 0; i < n_ope; i++) {
            owned_res[static_cast<std::size_t>(i)] = std::make_unique<Term_eq>(parts[i]->action());
            res[static_cast<std::size_t>(i)] = owned_res[static_cast<std::size_t>(i)].get();
        }

        Term_eq result(syst->get_space().get_domain(dom)->import(dom, bound, n_ope, res.data()));

        // Call the member function from domain
        return result;
    }
} // namespace Kadath
