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
 */

#include "For_Kadath/Ope_eq/ope_eq.hpp"
#include "For_Kadath/Tensor/tensor.hpp"
#include "For_Kadath/System_of_eqs/system_of_eqs.hpp"
#include "For_Kadath/Utilities/name_tools.hpp"
namespace Kadath
{
    // For tensors
    Ope_id::Ope_id(const System_of_eqs* zesys, const Term_eq* tt, int val, char* name, Array<int>* ttype)
        : Ope_eq(zesys, tt->get_dom(), 0), target(tt), valence(val), name_ind(name), type_ind(ttype)
    {
        need_sum = false;
        for (int i = 0; i < valence; i++)
            for (int j = i + 1; j < valence; j++)
                if (name_ind[i] == name_ind[j])
                    need_sum = true;
    }

    // For scalars and double
    Ope_id::Ope_id(const System_of_eqs* zesys, const Term_eq* tt)
        : Ope_eq(zesys, tt->get_dom(), 0), target(tt), valence(0), name_ind(nullptr), type_ind(nullptr), need_sum(false)
    {
        borrowed_action_result_ = target;
    }

    Ope_id::~Ope_id()
    {
        if (name_ind != nullptr)
            delete[] name_ind;
        if (type_ind != nullptr)
            delete type_ind;
    }

    Term_eq Ope_id::action() const
    {
        ScopedOpeActionProfile ope_action_profile_scope(*this);

        Term_eq auxi(*target);
        // First put the names (not for doubles or scalars...)
        if (name_ind != nullptr) {
            for (int i = 0; i < valence; i++)
                auxi.val_t->set_name_ind(i, name_ind[i]);
            auxi.val_t->name_affected = true;
            for (int lane = 0; lane < auxi.get_derivative_lane_count(); ++lane) {
                if (auxi.has_der_t(lane)) {
                    Tensor* derivative = auxi.set_der_t(lane);
                    for (int i = 0; i < valence; i++)
                        derivative->set_name_ind(i, name_ind[i]);
                    derivative->name_affected = true;
                }
            }
        }

        // Manip of the indices if needed :
        for (int i = 0; i < valence; i++)
            if (auxi.val_t->get_index_type(i) != (*type_ind)(i)) {
                // Manipulation using the metric :
                syst->get_met()->manipulate_ind(auxi, i);
            }

        if (!need_sum)
            return auxi;
        else {
            // Doit encore sommer sur certains indices :
            if (auxi.der_t == nullptr)
                return Term_eq(dom, auxi.val_t->do_summation_one_dom(dom));
            else {
                Term_eq summed(dom, auxi.val_t->do_summation_one_dom(dom), auxi.der_t->do_summation_one_dom(dom));
                summed.set_derivative_lane_count(auxi.get_derivative_lane_count());
                for (int lane = 1; lane < auxi.get_derivative_lane_count(); ++lane) {
                    if (auxi.has_der_t(lane))
                        summed.set_der_t(lane, auxi.get_der_t(lane).do_summation_one_dom(dom));
                }
                return summed;
            }
        }
    }

    void Ope_id::collect_vars(std::set<std::string>& vars) const
    {
        if ((syst == nullptr) || (target == nullptr))
            return;
        std::string name;
        if (syst->term_to_var_name(target, name))
            vars.insert(name);
        else if (Ope_def* def = syst->def_from_term(target))
            def->collect_vars(vars);
        else if (Ope_def_global* def_glob = syst->def_glob_from_term(target))
            def_glob->collect_vars(vars);
    }

    void Ope_id::collect_def_targets(std::set<const Term_eq*>& targets) const
    {
        if (target != nullptr)
            targets.insert(target);
    }
} // namespace Kadath
