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
    Metric_dirac::Metric_dirac(Metric_tensor& met) : Metric_conf(met)
    {
        type_tensor = met.get_type();
    }

    Metric_dirac::Metric_dirac(const Metric_dirac& so) : Metric_conf(so) {}

    Metric_dirac::~Metric_dirac() {}

    void Metric_dirac::compute_ricci_tensor(int dd) const
    {

        // Need christoffels
        if (p_christo[dd] == nullptr)
            compute_christo(dd);

        // Get the conformal factor
        /*	Tensor res_val (espace, 2, CON, basis) ;
            Tensor res_der (espace, 2, CON, basis) ;


            Term_eq der_cov (fmet.derive (COV, ' ', (*p_met_cov[dd]))) ;
            Term_eq der_con (fmet.derive (COV, ' ', (*p_met_con[dd]))) ;
            Term_eq dder_con (fmet.derive (COV, ' ', der_con)) ;
            bool doder = ((der_cov.der_t==nullptr) || (der_con.der_t==nullptr) || (dder_con.der_t==nullptr)) ? false :
           true ;

            Index pos (res_val) ;
            do {
                Val_domain cmpval (espace.get_domain(dd)) ;
                cmpval = 0 ;

                for (int k=1 ; k<=espace.get_ndim() ; k++)
                    for (int l=1 ; l<=espace.get_ndim() ; l++) {

                    cmpval += 0.5 * ((*p_met_con[dd]->val_t)(k,l)(dd)*(*dder_con.val_t)(k,l, pos(0)+1, pos(1)+1)(dd)
                     - (*der_con.val_t)(l, pos(0)+1, k)(dd) * (*der_con.val_t)(k, pos(1)+1, l)(dd)) ;
                     for (int m=1 ; m<=espace.get_ndim() ; m++)
                        for (int n=1 ; n<espace.get_ndim() ; n++)
                          cmpval += 0.5 * (
            -(*p_met_cov[dd]->val_t)(k,l)(dd)*(*p_met_con[dd]->val_t)(m,n)(dd)*(*der_con.val_t)(m, pos(0)+1,
           k)(dd)*(*der_con.val_t)(n, pos(1)+1, l)(dd)
            +(*p_met_cov[dd]->val_t)(m,l)(dd)*(*p_met_con[dd]->val_t)(pos(0)+1,k)(dd)*(*der_con.val_t)(k, m,
           n)(dd)*(*der_con.val_t)(n, pos(1)+1, l)(dd)
            +(*p_met_cov[dd]->val_t)(k,n)(dd)*(*p_met_con[dd]->val_t)(pos(1)+1,l)(dd)*(*der_con.val_t)(l, m,
           n)(dd)*(*der_con.val_t)(m, pos(0)+1, k)(dd)
            +0.5*(*p_met_cov[dd]->val_t)(pos(0)+1,k)(dd)*(*p_met_con[dd]->val_t)(pos(1)+1,l)(dd)*(*der_cov.val_t)(k, m,
           n)(dd)*(*der_con.val_t)(l,m,n)(dd)) ;

                    }

                res_val.set(pos).set_domain(dd) = cmpval ;

                if (doder) {
                  Val_domain cmpder (espace.get_domain(dd)) ;
                  cmpder = 0 ;

                for (int k=1 ; k<=espace.get_ndim() ; k++)
                    for (int l=1 ; l<=espace.get_ndim() ; l++) {

                    cmpder += 0.5 * ((*p_met_con[dd]->der_t)(k,l)(dd)*(*dder_con.val_t)(k,l, pos(0)+1, pos(1)+1)(dd)
                    +(*p_met_con[dd]->val_t)(k,l)(dd)*(*dder_con.der_t)(k,l, pos(0)+1, pos(1)+1)(dd)
                     - (*der_con.der_t)(l, pos(0)+1, k)(dd) * (*der_con.val_t)(k, pos(1)+1, l)(dd)
                     - (*der_con.val_t)(l, pos(0)+1, k)(dd) * (*der_con.der_t)(k, pos(1)+1, l)(dd))    ;
                     for (int m=1 ; m<=espace.get_ndim() ; m++)
                        for (int n=1 ; n<espace.get_ndim() ; n++)
                          cmpder += 0.5 * (
            -(*p_met_cov[dd]->der_t)(k,l)(dd)*(*p_met_con[dd]->val_t)(m,n)(dd)*(*der_con.val_t)(m, pos(0)+1,
           k)(dd)*(*der_con.val_t)(n, pos(1)+1, l)(dd)
            -(*p_met_cov[dd]->val_t)(k,l)(dd)*(*p_met_con[dd]->der_t)(m,n)(dd)*(*der_con.val_t)(m, pos(0)+1,
           k)(dd)*(*der_con.val_t)(n, pos(1)+1, l)(dd)
            -(*p_met_cov[dd]->val_t)(k,l)(dd)*(*p_met_con[dd]->val_t)(m,n)(dd)*(*der_con.der_t)(m, pos(0)+1,
           k)(dd)*(*der_con.val_t)(n, pos(1)+1, l)(dd)
            -(*p_met_cov[dd]->val_t)(k,l)(dd)*(*p_met_con[dd]->val_t)(m,n)(dd)*(*der_con.val_t)(m, pos(0)+1,
           k)(dd)*(*der_con.der_t)(n, pos(1)+1, l)(dd)
            +(*p_met_cov[dd]->der_t)(m,l)(dd)*(*p_met_con[dd]->val_t)(pos(0)+1,k)(dd)*(*der_con.val_t)(k, m,
           n)(dd)*(*der_con.val_t)(n, pos(1)+1, l)(dd)
            +(*p_met_cov[dd]->val_t)(m,l)(dd)*(*p_met_con[dd]->der_t)(pos(0)+1,k)(dd)*(*der_con.val_t)(k, m,
           n)(dd)*(*der_con.val_t)(n, pos(1)+1, l)(dd)
            +(*p_met_cov[dd]->val_t)(m,l)(dd)*(*p_met_con[dd]->val_t)(pos(0)+1,k)(dd)*(*der_con.der_t)(k, m,
           n)(dd)*(*der_con.val_t)(n, pos(1)+1, l)(dd)
            +(*p_met_cov[dd]->val_t)(m,l)(dd)*(*p_met_con[dd]->val_t)(pos(0)+1,k)(dd)*(*der_con.val_t)(k, m,
           n)(dd)*(*der_con.der_t)(n, pos(1)+1, l)(dd)
            +(*p_met_cov[dd]->der_t)(k,n)(dd)*(*p_met_con[dd]->val_t)(pos(1)+1,l)(dd)*(*der_con.val_t)(l, m,
           n)(dd)*(*der_con.val_t)(m, pos(0)+1, k)(dd)
            +(*p_met_cov[dd]->val_t)(k,n)(dd)*(*p_met_con[dd]->der_t)(pos(1)+1,l)(dd)*(*der_con.val_t)(l, m,
           n)(dd)*(*der_con.val_t)(m, pos(0)+1, k)(dd)
            +(*p_met_cov[dd]->val_t)(k,n)(dd)*(*p_met_con[dd]->val_t)(pos(1)+1,l)(dd)*(*der_con.der_t)(l, m,
           n)(dd)*(*der_con.val_t)(m, pos(0)+1, k)(dd)
            +(*p_met_cov[dd]->val_t)(k,n)(dd)*(*p_met_con[dd]->val_t)(pos(1)+1,l)(dd)*(*der_con.val_t)(l, m,
           n)(dd)*(*der_con.der_t)(m, pos(0)+1, k)(dd)
            +0.5*(*p_met_cov[dd]->der_t)(pos(0)+1,k)(dd)*(*p_met_con[dd]->val_t)(pos(1)+1,l)(dd)*(*der_cov.val_t)(k, m,
           n)(dd)*(*der_con.val_t)(l,m,n)(dd)
            +0.5*(*p_met_cov[dd]->val_t)(pos(0)+1,k)(dd)*(*p_met_con[dd]->der_t)(pos(1)+1,l)(dd)*(*der_cov.val_t)(k, m,
           n)(dd)*(*der_con.val_t)(l,m,n)(dd)
            +0.5*(*p_met_cov[dd]->val_t)(pos(0)+1,k)(dd)*(*p_met_con[dd]->val_t)(pos(1)+1,l)(dd)*(*der_cov.der_t)(k, m,
           n)(dd)*(*der_con.val_t)(l,m,n)(dd)
            +0.5*(*p_met_cov[dd]->val_t)(pos(0)+1,k)(dd)*(*p_met_con[dd]->val_t)(pos(1)+1,l)(dd)*(*der_cov.val_t)(k, m,
           n)(dd)*(*der_con.der_t)(l,m,n)(dd) ) ;


                    }

                res_der.set(pos).set_domain(dd) = cmpder ;
                }
            }

            while (pos.inc()) ;

        */

        Tensor res_val(espace, 2, COV, basis);
        Tensor res_der(espace, 2, COV, basis);

        Term_eq der_cov(fmet.derive(COV, ' ', (*p_met_cov[dd])));
        Term_eq dder_cov(fmet.derive(COV, ' ', der_cov));
        Term_eq der_con(fmet.derive(COV, ' ', (*p_met_con[dd])));
        bool doder =
            ((der_cov.der_t == nullptr) || (der_con.der_t == nullptr) || (dder_cov.der_t == nullptr)) ? false : true;

        Index pos(res_val);
        do {
            Val_domain cmpval(espace.get_domain(dd));
            cmpval = 0;

            for (int k = 1; k <= espace.get_ndim(); k++)
                for (int l = 1; l <= espace.get_ndim(); l++) {

                    cmpval +=
                        -0.5 *
                            ((*p_met_con[dd]->val_t)(k, l)(dd) * (*dder_cov.val_t)(k, l, pos(0) + 1, pos(1) + 1)(dd) +
                             (*der_con.val_t)(pos(0) + 1, k, l)(dd) * (*der_cov.val_t)(k, pos(1) + 1, l)(dd) +
                             (*der_con.val_t)(pos(1) + 1, k, l)(dd) * (*der_cov.val_t)(k, pos(0) + 1, l)(dd)) -
                        (*p_christo[dd]->val_t)(pos(0) + 1, l, k)(dd) * (*p_christo[dd]->val_t)(pos(1) + 1, k, l)(dd);
                }

            res_val.set(pos).set_domain(dd) = cmpval;

            if (doder) {
                Val_domain cmpder(espace.get_domain(dd));
                cmpder = 0;

                for (int k = 1; k <= espace.get_ndim(); k++)
                    for (int l = 1; l <= espace.get_ndim(); l++) {

                        cmpder +=
                            -0.5 * ((*p_met_con[dd]->der_t)(k, l)(dd) *
                                        (*dder_cov.val_t)(k, l, pos(0) + 1, pos(1) + 1)(dd) +
                                    (*p_met_con[dd]->val_t)(k, l)(dd) *
                                        (*dder_cov.der_t)(k, l, pos(0) + 1, pos(1) + 1)(dd) +
                                    (*der_con.der_t)(pos(0) + 1, k, l)(dd) * (*der_cov.val_t)(k, pos(1) + 1, l)(dd) +
                                    (*der_con.val_t)(pos(0) + 1, k, l)(dd) * (*der_cov.der_t)(k, pos(1) + 1, l)(dd) +
                                    (*der_con.der_t)(pos(1) + 1, k, l)(dd) * (*der_cov.val_t)(k, pos(0) + 1, l)(dd) +
                                    (*der_con.val_t)(pos(1) + 1, k, l)(dd) * (*der_cov.der_t)(k, pos(0) + 1, l)(dd)) -
                            (*p_christo[dd]->der_t)(pos(0) + 1, l, k)(dd) *
                                (*p_christo[dd]->val_t)(pos(1) + 1, k, l)(dd) -
                            (*p_christo[dd]->val_t)(pos(0) + 1, l, k)(dd) *
                                (*p_christo[dd]->der_t)(pos(1) + 1, k, l)(dd);
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

    void Metric_dirac::compute_ricci_scalar(int dd) const
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
