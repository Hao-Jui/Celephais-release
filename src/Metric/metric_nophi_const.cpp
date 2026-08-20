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

#include <memory>
#include <vector>

#include "For_Kadath/Space/space.hpp"
#include "For_Kadath/Array/exceptions.hpp"
#include "For_Kadath/Tensor/tensor.hpp"
#include "For_Kadath/Metric/metric_nophi.hpp"
#include "For_Kadath/Term_eq/term_eq.hpp"
#include "For_Kadath/Scalar/scalar.hpp"
#include "For_Kadath/System_of_eqs/system_of_eqs.hpp"
#include "For_Kadath/Tensor/metric_tensor.hpp"
#include "For_Kadath/Utilities/name_tools.hpp"
namespace Kadath
{
    Metric_nophi_const::Metric_nophi_const(Metric_tensor& met) : Metric_nophi(met) {}

    Metric_nophi_const::Metric_nophi_const(const Metric_nophi_const& so) : Metric_nophi(*so.p_met) {}

    Metric_nophi_const::~Metric_nophi_const() {}

    void Metric_nophi_const::set_system(System_of_eqs& ss, const char* name_met)
    {

        syst = &ss;

        if (ss.met != nullptr) {
            KADATH_THROW("Metric already set for the system");
        }

        // Position in the system :
        place_syst = ss.ndom * ss.ncst;

        ss.add_cst(nullptr, *p_met);

        ss.attach_metric(*this, name_met);
    }

    void Metric_nophi_const::compute_cov(int dd) const
    {

        int dim = espace.get_ndim();
        if (dim != 3) {
            KADATH_THROW("Function only implemented for dimension 3");
        }

        int place = place_syst + (dd - syst->dom_min);

        // Right storage : simple copy.
        if (type_tensor == COV) {

            if (p_met_cov[dd] == nullptr)
                p_met_cov[dd] = std::make_unique<Term_eq>(*syst->cst[place]);
            else
                *p_met_cov[dd] = Term_eq(*syst->cst[place]);
        } else {
            std::vector<std::unique_ptr<Term_eq>> res(p_met->get_n_comp());

            Scalar val(espace);
            Val_domain cmpval(espace.get_domain(dd));

            Val_domain detval(espace.get_domain(dd));
            detval = (*syst->cst[place]->val_t)(1, 1)(dd) * (*syst->cst[place]->val_t)(2, 2)(dd) *
                         (*syst->cst[place]->val_t)(3, 3)(dd) +
                     (*syst->cst[place]->val_t)(1, 2)(dd) * (*syst->cst[place]->val_t)(2, 3)(dd) *
                         (*syst->cst[place]->val_t)(1, 3)(dd) +
                     (*syst->cst[place]->val_t)(1, 3)(dd) * (*syst->cst[place]->val_t)(1, 2)(dd) *
                         (*syst->cst[place]->val_t)(2, 3)(dd) -
                     (*syst->cst[place]->val_t)(1, 3)(dd) * (*syst->cst[place]->val_t)(1, 3)(dd) *
                         (*syst->cst[place]->val_t)(2, 2)(dd) -
                     (*syst->cst[place]->val_t)(2, 3)(dd) * (*syst->cst[place]->val_t)(2, 3)(dd) *
                         (*syst->cst[place]->val_t)(1, 1)(dd) -
                     (*syst->cst[place]->val_t)(1, 2)(dd) * (*syst->cst[place]->val_t)(1, 2)(dd) *
                         (*syst->cst[place]->val_t)(3, 3)(dd);

            // Compo 1 1
            cmpval = (*syst->cst[place]->val_t)(2, 2)(dd) * (*syst->cst[place]->val_t)(3, 3)(dd) -
                     (*syst->cst[place]->val_t)(2, 3)(dd) * (*syst->cst[place]->val_t)(2, 3)(dd);
            val.set_domain(dd) = cmpval / detval;
            res[0] = std::make_unique<Term_eq>(dd, val);

            // Compo 1 2
            cmpval = (*syst->cst[place]->val_t)(1, 3)(dd) * (*syst->cst[place]->val_t)(2, 3)(dd) -
                     (*syst->cst[place]->val_t)(1, 2)(dd) * (*syst->cst[place]->val_t)(3, 3)(dd);
            val.set_domain(dd) = cmpval / detval;
            res[1] = std::make_unique<Term_eq>(dd, val);

            // Compo 1 3
            cmpval = (*syst->cst[place]->val_t)(1, 2)(dd) * (*syst->cst[place]->val_t)(2, 3)(dd) -
                     (*syst->cst[place]->val_t)(1, 3)(dd) * (*syst->cst[place]->val_t)(2, 2)(dd);
            val.set_domain(dd) = cmpval / detval;
            res[2] = std::make_unique<Term_eq>(dd, val);

            // Compo 2 2
            cmpval = (*syst->cst[place]->val_t)(1, 1)(dd) * (*syst->cst[place]->val_t)(3, 3)(dd) -
                     (*syst->cst[place]->val_t)(1, 3)(dd) * (*syst->cst[place]->val_t)(1, 3)(dd);
            val.set_domain(dd) = cmpval / detval;
            res[3] = std::make_unique<Term_eq>(dd, val);

            // Compo 2 3
            cmpval = (*syst->cst[place]->val_t)(1, 3)(dd) * (*syst->cst[place]->val_t)(1, 2)(dd) -
                     (*syst->cst[place]->val_t)(1, 1)(dd) * (*syst->cst[place]->val_t)(2, 3)(dd);
            val.set_domain(dd) = cmpval / detval;
            res[4] = std::make_unique<Term_eq>(dd, val);

            // Compo 3 3
            cmpval = (*syst->cst[place]->val_t)(1, 1)(dd) * (*syst->cst[place]->val_t)(2, 2)(dd) -
                     (*syst->cst[place]->val_t)(1, 2)(dd) * (*syst->cst[place]->val_t)(1, 2)(dd);
            val.set_domain(dd) = cmpval / detval;
            res[5] = std::make_unique<Term_eq>(dd, val);

            // Value field :
            Metric_tensor resval(espace, COV, basis);
            resval.set(1, 1) = res[0]->get_val_t();
            resval.set(1, 2) = res[1]->get_val_t();
            resval.set(1, 3) = res[2]->get_val_t();
            resval.set(2, 2) = res[3]->get_val_t();
            resval.set(2, 3) = res[4]->get_val_t();
            resval.set(3, 3) = res[5]->get_val_t();

            Metric_tensor zero(espace, COV, basis);
            for (int i = 1; i <= 3; i++)
                for (int j = i; j <= 3; j++)
                    zero.set(i, j) = 0;

            if (p_met_cov[dd] == nullptr)
                p_met_cov[dd] = std::make_unique<Term_eq>(dd, resval, zero);
            else
                *p_met_cov[dd] = Term_eq(dd, resval, zero);
        }
    }

    void Metric_nophi_const::compute_con(int dd) const
    {
        int dim = espace.get_ndim();
        if (dim != 3) {
            KADATH_THROW("Function only implemented for dimension 3");
        }

        int place = place_syst + (dd - syst->dom_min);

        // Right storage : simple copy.
        if (type_tensor == CON) {

            if (p_met_con[dd] == nullptr)
                p_met_con[dd] = std::make_unique<Term_eq>(*syst->cst[place]);
            else
                *p_met_con[dd] = Term_eq(*syst->cst[place]);
        } else {

            std::vector<std::unique_ptr<Term_eq>> res(p_met->get_n_comp());

            Scalar val(espace);
            Val_domain cmpval(espace.get_domain(dd));
            Val_domain detval(espace.get_domain(dd));
            detval = (*syst->cst[place]->val_t)(1, 1)(dd) * (*syst->cst[place]->val_t)(2, 2)(dd) *
                         (*syst->cst[place]->val_t)(3, 3)(dd) +
                     (*syst->cst[place]->val_t)(1, 2)(dd) * (*syst->cst[place]->val_t)(2, 3)(dd) *
                         (*syst->cst[place]->val_t)(1, 3)(dd) +
                     (*syst->cst[place]->val_t)(1, 3)(dd) * (*syst->cst[place]->val_t)(1, 2)(dd) *
                         (*syst->cst[place]->val_t)(2, 3)(dd) -
                     (*syst->cst[place]->val_t)(1, 3)(dd) * (*syst->cst[place]->val_t)(1, 3)(dd) *
                         (*syst->cst[place]->val_t)(2, 2)(dd) -
                     (*syst->cst[place]->val_t)(2, 3)(dd) * (*syst->cst[place]->val_t)(2, 3)(dd) *
                         (*syst->cst[place]->val_t)(1, 1)(dd) -
                     (*syst->cst[place]->val_t)(1, 2)(dd) * (*syst->cst[place]->val_t)(1, 2)(dd) *
                         (*syst->cst[place]->val_t)(3, 3)(dd);

            // Compo 1 1
            cmpval = (*syst->cst[place]->val_t)(2, 2)(dd) * (*syst->cst[place]->val_t)(3, 3)(dd) -
                     (*syst->cst[place]->val_t)(2, 3)(dd) * (*syst->cst[place]->val_t)(2, 3)(dd);
            val.set_domain(dd) = cmpval / detval;
            res[0] = std::make_unique<Term_eq>(dd, val);

            // Compo 1 2
            cmpval = (*syst->cst[place]->val_t)(1, 3)(dd) * (*syst->cst[place]->val_t)(2, 3)(dd) -
                     (*syst->cst[place]->val_t)(1, 2)(dd) * (*syst->cst[place]->val_t)(3, 3)(dd);
            val.set_domain(dd) = cmpval / detval;
            res[1] = std::make_unique<Term_eq>(dd, val);

            // Compo 1 3
            cmpval = (*syst->cst[place]->val_t)(1, 2)(dd) * (*syst->cst[place]->val_t)(2, 3)(dd) -
                     (*syst->cst[place]->val_t)(1, 3)(dd) * (*syst->cst[place]->val_t)(2, 2)(dd);
            val.set_domain(dd) = cmpval / detval;
            res[2] = std::make_unique<Term_eq>(dd, val);

            // Compo 2 2
            cmpval = (*syst->cst[place]->val_t)(1, 1)(dd) * (*syst->cst[place]->val_t)(3, 3)(dd) -
                     (*syst->cst[place]->val_t)(1, 3)(dd) * (*syst->cst[place]->val_t)(1, 3)(dd);
            val.set_domain(dd) = cmpval / detval;
            res[3] = std::make_unique<Term_eq>(dd, val);

            // Compo 2 3
            cmpval = (*syst->cst[place]->val_t)(1, 3)(dd) * (*syst->cst[place]->val_t)(1, 2)(dd) -
                     (*syst->cst[place]->val_t)(1, 1)(dd) * (*syst->cst[place]->val_t)(2, 3)(dd);
            val.set_domain(dd) = cmpval / detval;
            res[4] = std::make_unique<Term_eq>(dd, val);

            // Compo 3 3
            cmpval = (*syst->cst[place]->val_t)(1, 1)(dd) * (*syst->cst[place]->val_t)(2, 2)(dd) -
                     (*syst->cst[place]->val_t)(1, 2)(dd) * (*syst->cst[place]->val_t)(1, 2)(dd);
            val.set_domain(dd) = cmpval / detval;
            res[5] = std::make_unique<Term_eq>(dd, val);

            // Value field :
            Metric_tensor resval(espace, CON, basis);
            resval.set(1, 1) = res[0]->get_val_t();
            resval.set(1, 2) = res[1]->get_val_t();
            resval.set(1, 3) = res[2]->get_val_t();
            resval.set(2, 2) = res[3]->get_val_t();
            resval.set(2, 3) = res[4]->get_val_t();
            resval.set(3, 3) = res[5]->get_val_t();

            Metric_tensor zero(espace, CON, basis);
            for (int i = 1; i <= 3; i++)
                for (int j = i; j <= 3; j++)
                    zero.set(i, j) = 0;

            if (p_met_con[dd] == nullptr)
                p_met_con[dd] = std::make_unique<Term_eq>(dd, resval, zero);
            else
                *p_met_con[dd] = Term_eq(dd, resval, zero);
        }
    }
} // namespace Kadath
