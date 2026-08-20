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

#include "For_Kadath/Space/space.hpp"
#include "For_Kadath/Tensor/tensor.hpp"
#include "For_Kadath/Metric/metric.hpp"
#include "For_Kadath/Term_eq/term_eq.hpp"
#include "For_Kadath/Scalar/scalar.hpp"
#include "For_Kadath/System_of_eqs/system_of_eqs.hpp"
#include "For_Kadath/Tensor/metric_tensor.hpp"
namespace Kadath
{
    Metric_harmonic::Metric_harmonic(Metric_tensor& met) : Metric_general(met)
    {
        type_tensor = met.get_type();
    }

    Metric_harmonic::Metric_harmonic(const Metric_harmonic& so) : Metric_general(so) {}

    Metric_harmonic::~Metric_harmonic() {}

    void Metric_harmonic::compute_det_cov(int dd) const
    {

        // Need that
        if (p_met_cov[dd] == nullptr)
            compute_cov(dd);

        bool doder = (p_met_cov[dd]->der_t == nullptr) ? false : true;

        Scalar res_val(espace);
        Scalar res_der(espace);

        Val_domain cmpval(espace.get_domain(dd));
        cmpval = 0;

        for (int i = 1; i <= espace.get_ndim(); i++)
            for (int j = 1; j <= espace.get_ndim(); j++)
                cmpval += (*p_met_cov[dd]->val_t)(1, 1)(dd) * (*p_met_cov[dd]->val_t)(2, 2)(dd) *
                              (*p_met_cov[dd]->val_t)(3, 3)(dd) +
                          (*p_met_cov[dd]->val_t)(1, 2)(dd) * (*p_met_cov[dd]->val_t)(2, 3)(dd) *
                              (*p_met_cov[dd]->val_t)(1, 3)(dd) +
                          (*p_met_cov[dd]->val_t)(1, 3)(dd) * (*p_met_cov[dd]->val_t)(1, 2)(dd) *
                              (*p_met_cov[dd]->val_t)(2, 3)(dd) -
                          (*p_met_cov[dd]->val_t)(1, 3)(dd) * (*p_met_cov[dd]->val_t)(1, 3)(dd) *
                              (*p_met_cov[dd]->val_t)(2, 2)(dd) -
                          (*p_met_cov[dd]->val_t)(2, 3)(dd) * (*p_met_cov[dd]->val_t)(2, 3)(dd) *
                              (*p_met_cov[dd]->val_t)(1, 1)(dd) -
                          (*p_met_cov[dd]->val_t)(1, 2)(dd) * (*p_met_cov[dd]->val_t)(1, 2)(dd) *
                              (*p_met_cov[dd]->val_t)(3, 3)(dd);
        res_val.set_domain(dd) = cmpval;

        if (doder) {
            Val_domain cmpder(espace.get_domain(dd));
            cmpder = 0;

            for (int i = 1; i <= espace.get_ndim(); i++)
                for (int j = 1; j <= espace.get_ndim(); j++)
                    cmpder += (*p_met_cov[dd]->der_t)(1, 1)(dd) * (*p_met_cov[dd]->val_t)(2, 2)(dd) *
                                  (*p_met_cov[dd]->val_t)(3, 3)(dd) +
                              (*p_met_cov[dd]->der_t)(1, 2)(dd) * (*p_met_cov[dd]->val_t)(2, 3)(dd) *
                                  (*p_met_cov[dd]->val_t)(1, 3)(dd) +
                              (*p_met_cov[dd]->der_t)(1, 3)(dd) * (*p_met_cov[dd]->val_t)(1, 2)(dd) *
                                  (*p_met_cov[dd]->val_t)(2, 3)(dd) -
                              (*p_met_cov[dd]->der_t)(1, 3)(dd) * (*p_met_cov[dd]->val_t)(1, 3)(dd) *
                                  (*p_met_cov[dd]->val_t)(2, 2)(dd) -
                              (*p_met_cov[dd]->der_t)(2, 3)(dd) * (*p_met_cov[dd]->val_t)(2, 3)(dd) *
                                  (*p_met_cov[dd]->val_t)(1, 1)(dd) -
                              (*p_met_cov[dd]->der_t)(1, 2)(dd) * (*p_met_cov[dd]->val_t)(1, 2)(dd) *
                                  (*p_met_cov[dd]->val_t)(3, 3)(dd) +
                              (*p_met_cov[dd]->val_t)(1, 1)(dd) * (*p_met_cov[dd]->der_t)(2, 2)(dd) *
                                  (*p_met_cov[dd]->val_t)(3, 3)(dd) +
                              (*p_met_cov[dd]->val_t)(1, 2)(dd) * (*p_met_cov[dd]->der_t)(2, 3)(dd) *
                                  (*p_met_cov[dd]->val_t)(1, 3)(dd) +
                              (*p_met_cov[dd]->val_t)(1, 3)(dd) * (*p_met_cov[dd]->der_t)(1, 2)(dd) *
                                  (*p_met_cov[dd]->val_t)(2, 3)(dd) -
                              (*p_met_cov[dd]->val_t)(1, 3)(dd) * (*p_met_cov[dd]->der_t)(1, 3)(dd) *
                                  (*p_met_cov[dd]->val_t)(2, 2)(dd) -
                              (*p_met_cov[dd]->val_t)(2, 3)(dd) * (*p_met_cov[dd]->der_t)(2, 3)(dd) *
                                  (*p_met_cov[dd]->val_t)(1, 1)(dd) -
                              (*p_met_cov[dd]->val_t)(1, 2)(dd) * (*p_met_cov[dd]->der_t)(1, 2)(dd) *
                                  (*p_met_cov[dd]->val_t)(3, 3)(dd) +
                              (*p_met_cov[dd]->val_t)(1, 1)(dd) * (*p_met_cov[dd]->val_t)(2, 2)(dd) *
                                  (*p_met_cov[dd]->der_t)(3, 3)(dd) +
                              (*p_met_cov[dd]->val_t)(1, 2)(dd) * (*p_met_cov[dd]->val_t)(2, 3)(dd) *
                                  (*p_met_cov[dd]->der_t)(1, 3)(dd) +
                              (*p_met_cov[dd]->val_t)(1, 3)(dd) * (*p_met_cov[dd]->val_t)(1, 2)(dd) *
                                  (*p_met_cov[dd]->der_t)(2, 3)(dd) -
                              (*p_met_cov[dd]->val_t)(1, 3)(dd) * (*p_met_cov[dd]->val_t)(1, 3)(dd) *
                                  (*p_met_cov[dd]->der_t)(2, 2)(dd) -
                              (*p_met_cov[dd]->val_t)(2, 3)(dd) * (*p_met_cov[dd]->val_t)(2, 3)(dd) *
                                  (*p_met_cov[dd]->der_t)(1, 1)(dd) -
                              (*p_met_cov[dd]->val_t)(1, 2)(dd) * (*p_met_cov[dd]->val_t)(1, 2)(dd) *
                                  (*p_met_cov[dd]->der_t)(3, 3)(dd);
            res_der.set_domain(dd) = cmpder;
        }

        if (!doder) {
            if (p_det_cov[dd] == nullptr)
                p_det_cov[dd] = std::make_unique<Term_eq>(dd, res_val);
            else
                *p_det_cov[dd] = Term_eq(dd, res_val);
        } else {
            if (p_det_cov[dd] == nullptr)
                p_det_cov[dd] = std::make_unique<Term_eq>(dd, res_val, res_der);
            else
                *p_det_cov[dd] = Term_eq(dd, res_val, res_der);
        }
    }

    void Metric_harmonic::compute_ricci_tensor(int dd) const
    {

        // Need christoffels
        if (p_det_cov[dd] == nullptr)
            compute_det_cov(dd);

        if (p_christo[dd] == nullptr)
            compute_christo(dd);
        if (p_dirac[dd] == nullptr)
            compute_dirac(dd);

        // Need christoffels
        if (p_christo[dd] == nullptr)
            compute_christo(dd);

        Tensor res_val(espace, 2, COV, basis);
        Tensor res_der(espace, 2, COV, basis);

        Term_eq der_cov(fmet.derive(COV, ' ', (*p_met_cov[dd])));
        Term_eq der_con(fmet.derive(COV, ' ', (*p_met_con[dd])));
        Term_eq dder_cov(fmet.derive(COV, 'u', der_cov));

        bool doder =
            ((der_cov.der_t == nullptr) || (der_con.der_t == nullptr) || (dder_cov.der_t == nullptr)) ? false : true;

        Index pos(res_val);
        do {
            Val_domain cmpval(espace.get_domain(dd));
            cmpval = 0;

            for (int k = 1; k <= espace.get_ndim(); k++)
                for (int l = 1; l <= espace.get_ndim(); l++) {

                    cmpval +=
                        -0.5 * (*p_met_con[dd]->val_t)(k, l)(dd) *
                            (*dder_cov.val_t)(k, l, pos(0) + 1, pos(1) + 1)(dd)-0.5 *
                            (*der_con.val_t)(pos(1) + 1, k, l)(dd) * (*der_cov.val_t)(k, pos(0) + 1, l)(dd)-0.5 *
                            (*der_con.val_t)(pos(0) + 1, k, l)(dd) * (*der_cov.val_t)(k, pos(1) + 1, l)(dd) +
                        0.5 * (*der_con.val_t)(pos(0) + 1, k, l)(dd) * (*der_cov.val_t)(pos(1) + 1, k, l)(dd) +
                        (*p_christo[dd]->val_t)(pos(0) + 1, pos(1) + 1, k)(dd) * (*p_christo[dd]->val_t)(k, l, l)(dd) -
                        (*p_christo[dd]->val_t)(pos(0) + 1, k, l)(dd) * (*p_christo[dd]->val_t)(pos(1) + 1, l, k)(dd);

                    for (int m = 1; m <= espace.get_ndim(); m++) {
                        cmpval += (*p_met_cov[dd]->val_t)(m, l)(dd) * (*der_con.val_t)(k, k, l)(dd) *
                                      (*p_christo[dd]->val_t)(pos(0) + 1, pos(1) + 1, m)(dd) -
                                  (*p_met_cov[dd]->val_t)(m, l)(dd) * (*der_con.val_t)(pos(1) + 1, k, l)(dd) *
                                      (*p_christo[dd]->val_t)(pos(0) + 1, k, m)(dd);
                    }
                }
            res_val.set(pos).set_domain(dd) = cmpval;

            if (doder) {
                Val_domain cmpder(espace.get_domain(dd));
                cmpder = 0;

                for (int k = 1; k <= espace.get_ndim(); k++)
                    for (int l = 1; l <= espace.get_ndim(); l++) {

                        cmpder +=
                            -0.5 * (*p_met_con[dd]->der_t)(k, l)(dd) *
                                (*dder_cov.val_t)(k, l, pos(0) + 1, pos(1) + 1)(dd)-0.5 *
                                (*p_met_con[dd]->val_t)(k, l)(dd) *
                                (*dder_cov.der_t)(k, l, pos(0) + 1, pos(1) + 1)(dd)-0.5 *
                                (*der_con.der_t)(pos(1) + 1, k, l)(dd) * (*der_cov.val_t)(k, pos(0) + 1, l)(dd)-0.5 *
                                (*der_con.val_t)(pos(1) + 1, k, l)(dd) * (*der_cov.der_t)(k, pos(0) + 1, l)(dd)-0.5 *
                                (*der_con.der_t)(pos(0) + 1, k, l)(dd) * (*der_cov.val_t)(k, pos(1) + 1, l)(dd)-0.5 *
                                (*der_con.val_t)(pos(0) + 1, k, l)(dd) * (*der_cov.der_t)(k, pos(1) + 1, l)(dd) +
                            0.5 * (*der_con.der_t)(pos(0) + 1, k, l)(dd) * (*der_cov.val_t)(pos(1) + 1, k, l)(dd) +
                            0.5 * (*der_con.val_t)(pos(0) + 1, k, l)(dd) * (*der_cov.der_t)(pos(1) + 1, k, l)(dd) +
                            (*p_christo[dd]->der_t)(pos(0) + 1, pos(1) + 1, k)(dd) *
                                (*p_christo[dd]->val_t)(k, l, l)(dd) +
                            (*p_christo[dd]->val_t)(pos(0) + 1, pos(1) + 1, k)(dd) *
                                (*p_christo[dd]->der_t)(k, l, l)(dd) -
                            (*p_christo[dd]->der_t)(pos(0) + 1, k, l)(dd) *
                                (*p_christo[dd]->val_t)(pos(1) + 1, l, k)(dd);
                        -(*p_christo[dd]->val_t)(pos(0) + 1, k, l)(dd) * (*p_christo[dd]->der_t)(pos(1) + 1, l, k)(dd);

                        for (int m = 1; m <= espace.get_ndim(); m++) {

                            cmpder += (*p_met_cov[dd]->der_t)(m, l)(dd) * (*der_con.val_t)(k, k, l)(dd) *
                                          (*p_christo[dd]->val_t)(pos(0) + 1, pos(1) + 1, m)(dd) +
                                      (*p_met_cov[dd]->val_t)(m, l)(dd) * (*der_con.der_t)(k, k, l)(dd) *
                                          (*p_christo[dd]->val_t)(pos(0) + 1, pos(1) + 1, m)(dd) +
                                      (*p_met_cov[dd]->val_t)(m, l)(dd) * (*der_con.val_t)(k, k, l)(dd) *
                                          (*p_christo[dd]->der_t)(pos(0) + 1, pos(1) + 1, m)(dd) -
                                      (*p_met_cov[dd]->der_t)(m, l)(dd) * (*der_con.val_t)(pos(1) + 1, k, l)(dd) *
                                          (*p_christo[dd]->val_t)(pos(0) + 1, k, m)(dd) -
                                      (*p_met_cov[dd]->val_t)(m, l)(dd) * (*der_con.der_t)(pos(1) + 1, k, l)(dd) *
                                          (*p_christo[dd]->val_t)(pos(0) + 1, k, m)(dd) -
                                      (*p_met_cov[dd]->val_t)(m, l)(dd) * (*der_con.val_t)(pos(1) + 1, k, l)(dd) *
                                          (*p_christo[dd]->der_t)(pos(0) + 1, k, m)(dd);
                        }
                    }

                res_der.set(pos).set_domain(dd) = cmpder;
            }
        }

        while (pos.inc());

        if (!doder) {
            if (p_ricci_tensor[dd] == nullptr)
                p_ricci_tensor[dd] = std::make_unique<Term_eq>(dd, res_val);
            else
                *p_ricci_tensor[dd] = Term_eq(dd, res_val);
        } else {
            if (p_ricci_tensor[dd] == nullptr)
                p_ricci_tensor[dd] = std::make_unique<Term_eq>(dd, res_val, res_der);
            else
                *p_ricci_tensor[dd] = Term_eq(dd, res_val, res_der);
        }
    }

    void Metric_harmonic::compute_ricci_scalar(int dd) const
    {

        // Need that
        if (p_met_con[dd] == nullptr)
            compute_con(dd);
        if (p_ricci_tensor[dd] == nullptr)
            compute_ricci_tensor(dd);

        bool doder = ((p_met_con[dd]->der_t == nullptr) || (p_ricci_tensor[dd]->der_t == nullptr)) ? false : true;
        Scalar res_val(espace);
        Scalar res_der(espace);

        Val_domain cmpval(espace.get_domain(dd));
        cmpval = 0;

        for (int i = 1; i <= espace.get_ndim(); i++)
            for (int j = 1; j <= espace.get_ndim(); j++)
                cmpval += (*p_met_con[dd]->val_t)(i, j)(dd) * (*p_ricci_tensor[dd]->val_t)(i, j)(dd);
        res_val.set_domain(dd) = cmpval;

        if (doder) {
            Val_domain cmpder(espace.get_domain(dd));
            cmpder = 0;

            for (int i = 1; i <= espace.get_ndim(); i++)
                for (int j = 1; j <= espace.get_ndim(); j++)
                    cmpder += (*p_met_con[dd]->val_t)(i, j)(dd) * (*p_ricci_tensor[dd]->der_t)(i, j)(dd) +
                              (*p_met_con[dd]->der_t)(i, j)(dd) * (*p_ricci_tensor[dd]->val_t)(i, j)(dd);
            res_der.set_domain(dd) = cmpder;
        }

        if (!doder) {
            if (p_ricci_scalar[dd] == nullptr)
                p_ricci_scalar[dd] = std::make_unique<Term_eq>(dd, res_val);
            else
                *p_ricci_scalar[dd] = Term_eq(dd, res_val);
        } else {
            if (p_ricci_scalar[dd] == nullptr)
                p_ricci_scalar[dd] = std::make_unique<Term_eq>(dd, res_val, res_der);
            else
                *p_ricci_scalar[dd] = Term_eq(dd, res_val, res_der);
        }
    }
} // namespace Kadath
