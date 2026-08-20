/*
    Copyright 2018 Philippe Grandclement

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
#include "For_Kadath/Array/exceptions.hpp"
#include "For_Kadath/Tensor/tensor.hpp"
#include "For_Kadath/Metric/metric.hpp"
#include "For_Kadath/Term_eq/term_eq.hpp"
#include "For_Kadath/Scalar/scalar.hpp"
#include "For_Kadath/System_of_eqs/system_of_eqs.hpp"
#include "For_Kadath/Tensor/metric_tensor.hpp"
#include "For_Kadath/Utilities/name_tools.hpp"
namespace Kadath
{

    Metric_cfc::Metric_cfc(Scalar& conf, const Base_tensor& bb)
        : Metric(conf.get_space()), p_conf(&conf), basis(bb), fmet(conf.get_space(), basis)
    {
        type_tensor = COV;
    }

    Metric_cfc::Metric_cfc(const Metric_cfc& so)
        : Metric(so), p_conf(so.p_conf), basis(so.basis), fmet(so.fmet), place_syst(so.place_syst)
    {
    }

    Metric_cfc::~Metric_cfc() {}

    int Metric_cfc::give_type(int dd) const
    {
        return basis.get_basis(dd);
    }

    void Metric_cfc::compute_cov(int dd) const
    {

        int dim = espace.get_ndim();
        if (dim != 3) {
            KADATH_THROW("Function only implemented for dimension 3");
        }

        int place = place_syst + (dd - syst->dom_min);
        // Need to work component by components...
        bool doder = (syst->term[place]->der_t == nullptr) ? false : true;

        Scalar val(espace);
        Scalar der(espace);

        // CFC
        val.set_domain(dd) = pow((*syst->term[place]->val_t)()(dd), 4);
        val.std_base();
        if (doder) {
            der.set_domain(dd) = 4 * pow((*syst->term[place]->val_t)()(dd), 3) * (*syst->term[place]->der_t)()(dd);
            der.std_base();
        }

        // Value field :
        Metric_tensor resval(espace, COV, basis);
        resval.set(1, 1) = val;
        resval.set(1, 2) = 0;
        resval.set(1, 3) = 0;
        resval.set(2, 2) = val;
        resval.set(2, 3) = 0;
        resval.set(3, 3) = val;

        if (!doder) {
            if (p_met_cov[dd] == nullptr)
                p_met_cov[dd] = std::make_unique<Term_eq>(dd, resval);
            else
                *p_met_cov[dd] = Term_eq(dd, resval);
        } else {
            // Der field :
            Metric_tensor resder(espace, COV, basis);
            resder.set(1, 1) = der;
            resder.set(1, 2) = 0;
            resder.set(1, 3) = 0;
            resder.set(2, 2) = der;
            resder.set(2, 3) = 0;
            resder.set(3, 3) = der;

            if (p_met_cov[dd] == nullptr)
                p_met_cov[dd] = std::make_unique<Term_eq>(dd, resval, resder);
            else
                *p_met_cov[dd] = Term_eq(dd, resval, resder);
        }
    }

    void Metric_cfc::compute_con(int dd) const
    {

        int dim = espace.get_ndim();
        if (dim != 3) {
            KADATH_THROW("Function only implemented for dimension 3");
        }

        int place = place_syst + (dd - syst->dom_min);
        // Need to work component by components...
        bool doder = (syst->term[place]->der_t == nullptr) ? false : true;

        Scalar val(espace);
        Scalar der(espace);

        // CFC
        val.set_domain(dd) = pow((*syst->term[place]->val_t)()(dd), -4);
        val.std_base();
        if (doder) {
            der.set_domain(dd) = -4 * pow((*syst->term[place]->val_t)()(dd), -5) * (*syst->term[place]->der_t)()(dd);
            der.std_base();
        }

        // Value field :
        Metric_tensor resval(espace, COV, basis);
        resval.set(1, 1) = val;
        resval.set(1, 2) = 0;
        resval.set(1, 3) = 0;
        resval.set(2, 2) = val;
        resval.set(2, 3) = 0;
        resval.set(3, 3) = val;

        if (!doder) {
            if (p_met_con[dd] == nullptr)
                p_met_con[dd] = std::make_unique<Term_eq>(dd, resval);
            else
                *p_met_con[dd] = Term_eq(dd, resval);
        } else {
            // Der field :
            Metric_tensor resder(espace, COV, basis);
            resder.set(1, 1) = der;
            resder.set(1, 2) = 0;
            resder.set(1, 3) = 0;
            resder.set(2, 2) = der;
            resder.set(2, 3) = 0;
            resder.set(3, 3) = der;

            if (p_met_con[dd] == nullptr)
                p_met_con[dd] = std::make_unique<Term_eq>(dd, resval, resder);
            else
                *p_met_con[dd] = Term_eq(dd, resval, resder);
        }
    }

    void Metric_cfc::compute_christo(int dd) const
    {
        // Need both representation of the metric)
        if (type_tensor == CON) {
            if (p_met_con[dd] == nullptr)
                compute_con(dd);
            if (p_met_cov[dd] == nullptr)
                compute_cov(dd);
        } else {
            if (p_met_cov[dd] == nullptr)
                compute_cov(dd);
            if (p_met_con[dd] == nullptr)
                compute_con(dd);
        }

        Array<int> type_ind(3);
        type_ind.set(0) = COV;
        type_ind.set(1) = COV;
        type_ind.set(2) = CON;
        Tensor res_val(espace, 3, type_ind, basis);
        Tensor res_der(espace, 3, type_ind, basis);

        Term_eq flat_der(fmet.derive(COV, ' ', (*p_met_cov[dd])));
        bool doder = (flat_der.der_t == nullptr) ? false : true;

        Index pos(res_val);
        do {

            Val_domain cmpval(espace.get_domain(dd));
            cmpval = 0;
            for (int l = 1; l <= espace.get_ndim(); l++)
                cmpval += 0.5 * (*p_met_con[dd]->val_t)(pos(2) + 1, l)(dd) *
                          ((*flat_der.val_t)(pos(0) + 1, pos(1) + 1, l)(dd) +
                           (*flat_der.val_t)(pos(1) + 1, pos(0) + 1, l)(dd) -
                           (*flat_der.val_t)(l, pos(0) + 1, pos(1) + 1)(dd));

            res_val.set(pos).set_domain(dd) = cmpval;

            if (doder) {
                Val_domain cmpder(espace.get_domain(dd));
                cmpder = 0;
                for (int l = 1; l <= espace.get_ndim(); l++)
                    cmpder += 0.5 * (*p_met_con[dd]->der_t)(pos(2) + 1, l)(dd) *
                                  ((*flat_der.val_t)(pos(0) + 1, pos(1) + 1, l)(dd) +
                                   (*flat_der.val_t)(pos(1) + 1, pos(0) + 1, l)(dd) -
                                   (*flat_der.val_t)(l, pos(0) + 1, pos(1) + 1)(dd)) +
                              0.5 * (*p_met_con[dd]->val_t)(pos(2) + 1, l)(dd) *
                                  ((*flat_der.der_t)(pos(0) + 1, pos(1) + 1, l)(dd) +
                                   (*flat_der.der_t)(pos(1) + 1, pos(0) + 1, l)(dd) -
                                   (*flat_der.der_t)(l, pos(0) + 1, pos(1) + 1)(dd));

                res_der.set(pos).set_domain(dd) = cmpder;
            }
        } while (pos.inc());

        if (!doder) {
            if (p_christo[dd] == nullptr)
                p_christo[dd] = std::make_unique<Term_eq>(dd, res_val);
            else
                *p_christo[dd] = Term_eq(dd, res_val);
        } else {
            if (p_christo[dd] == nullptr)
                p_christo[dd] = std::make_unique<Term_eq>(dd, res_val, res_der);
            else
                *p_christo[dd] = Term_eq(dd, res_val, res_der);
        }
    }

    void Metric_cfc::compute_riemann(int dd) const
    {

        // Need christoffels
        if (p_christo[dd] == nullptr)
            compute_christo(dd);

        Array<int> indices(4);
        indices.set(0) = CON;
        indices.set(1) = COV;
        indices.set(2) = COV;
        indices.set(3) = COV;
        Tensor res_val(espace, 4, indices, basis);
        Tensor res_der(espace, 4, indices, basis);

        Term_eq flat_der(fmet.derive(COV, ' ', (*p_christo[dd])));
        bool doder = (flat_der.der_t == nullptr) ? false : true;

        Index pos(res_val);
        do {

            Val_domain cmpval((*flat_der.val_t)(pos(2) + 1, pos(1) + 1, pos(3) + 1, pos(0) + 1)(dd) -
                              (*flat_der.val_t)(pos(3) + 1, pos(1) + 1, pos(2) + 1, pos(0) + 1)(dd));
            for (int m = 1; m <= espace.get_ndim(); m++) {
                cmpval += (*p_christo[dd]->val_t)(pos(2) + 1, m, pos(0) + 1)(dd) *
                              (*p_christo[dd]->val_t)(pos(1) + 1, pos(3) + 1, m)(dd) -
                          (*p_christo[dd]->val_t)(pos(3) + 1, m, pos(0) + 1)(dd) *
                              (*p_christo[dd]->val_t)(pos(1) + 1, pos(2) + 1, m)(dd);
            }
            res_val.set(pos).set_domain(dd) = cmpval;
            if (doder) {
                Val_domain cmpder((*flat_der.der_t)(pos(2) + 1, pos(1) + 1, pos(3) + 1, pos(0) + 1)(dd) -
                                  (*flat_der.der_t)(pos(3) + 1, pos(1) + 1, pos(2) + 1, pos(0) + 1)(dd));
                for (int m = 1; m <= espace.get_ndim(); m++) {
                    cmpder += (*p_christo[dd]->der_t)(pos(2) + 1, m, pos(0) + 1)(dd) *
                                  (*p_christo[dd]->val_t)(pos(1) + 1, pos(3) + 1, m)(dd) +
                              (*p_christo[dd]->val_t)(pos(2) + 1, m, pos(0) + 1)(dd) *
                                  (*p_christo[dd]->der_t)(pos(1) + 1, pos(3) + 1, m)(dd) -
                              (*p_christo[dd]->der_t)(pos(3) + 1, m, pos(0) + 1)(dd) *
                                  (*p_christo[dd]->val_t)(pos(1) + 1, pos(2) + 1, m)(dd) -
                              (*p_christo[dd]->val_t)(pos(3) + 1, m, pos(0) + 1)(dd) *
                                  (*p_christo[dd]->der_t)(pos(1) + 1, pos(2) + 1, m)(dd);
                }
                res_der.set(pos).set_domain(dd) = cmpder;
            }
        } while (pos.inc());

        if (!doder) {
            if (p_riemann[dd] == nullptr)
                p_riemann[dd] = std::make_unique<Term_eq>(dd, res_val);
            else
                *p_riemann[dd] = Term_eq(dd, res_val);
        } else {
            if (p_riemann[dd] == nullptr)
                p_riemann[dd] = std::make_unique<Term_eq>(dd, res_val, res_der);
            else
                *p_riemann[dd] = Term_eq(dd, res_val, res_der);
        }
    }

    void Metric_cfc::compute_ricci_tensor(int dd) const
    {
        // Need christoffels
        if (p_christo[dd] == nullptr)
            compute_christo(dd);

        Array<int> indices(2);
        indices.set(0) = COV;
        indices.set(1) = COV;
        Tensor res_val(espace, 2, indices, basis);
        Tensor res_der(espace, 2, indices, basis);

        Term_eq flat_der(fmet.derive(COV, ' ', (*p_christo[dd])));
        bool doder = (flat_der.der_t == nullptr) ? false : true;

        Index pos(res_val);
        do {

            Val_domain cmpval(espace.get_domain(dd));
            cmpval = 0;

            for (int l = 1; l <= espace.get_ndim(); l++) {
                cmpval += (*flat_der.val_t)(l, pos(1) + 1, pos(0) + 1, l)(dd) -
                          (*flat_der.val_t)(pos(1) + 1, pos(0) + 1, l, l)(dd);
                for (int m = 1; m <= espace.get_ndim(); m++) {
                    cmpval +=
                        (*p_christo[dd]->val_t)(l, m, l)(dd) * (*p_christo[dd]->val_t)(pos(0) + 1, pos(1) + 1, m)(dd) -
                        (*p_christo[dd]->val_t)(pos(0) + 1, l, m)(dd) * (*p_christo[dd]->val_t)(pos(1) + 1, m, l)(dd);
                }
            }

            res_val.set(pos).set_domain(dd) = cmpval;
            if (doder) {
                Val_domain cmpder(espace.get_domain(dd));
                cmpder = 0;

                for (int l = 1; l <= espace.get_ndim(); l++) {
                    cmpder += (*flat_der.der_t)(l, pos(1) + 1, pos(0) + 1, l)(dd) -
                              (*flat_der.der_t)(pos(1) + 1, pos(0) + 1, l, l)(dd);
                    for (int m = 1; m <= espace.get_ndim(); m++) {
                        cmpder += (*p_christo[dd]->der_t)(l, m, l)(dd) *
                                      (*p_christo[dd]->val_t)(pos(0) + 1, pos(1) + 1, m)(dd) +
                                  (*p_christo[dd]->val_t)(l, m, l)(dd) *
                                      (*p_christo[dd]->der_t)(pos(0) + 1, pos(1) + 1, m)(dd) -
                                  (*p_christo[dd]->der_t)(pos(0) + 1, l, m)(dd) *
                                      (*p_christo[dd]->val_t)(pos(1) + 1, m, l)(dd) -
                                  (*p_christo[dd]->val_t)(pos(0) + 1, l, m)(dd) *
                                      (*p_christo[dd]->der_t)(pos(1) + 1, m, l)(dd);
                    }
                }
                res_der.set(pos).set_domain(dd) = cmpder;
            }
        } while (pos.inc());

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

    Term_eq Metric_cfc::derive_flat(int type_der, char ind_der, const Term_eq& so) const
    {

        int dd = so.get_dom();

        if (p_met_con[dd] == nullptr)
            compute_con(dd);
        if (p_met_cov[dd] == nullptr)
            compute_cov(dd);

        // The partial derivative part
        Term_eq res(fmet.derive_with_other(type_der, ind_der, so, this));
        return res;
    }

    Term_eq Metric_cfc::derive(int type_der, char ind_der, const Term_eq& so) const
    {

        int dd = so.get_dom();

        if (p_christo[dd] == nullptr)
            compute_christo(dd);
        // The partial derivative part
        Term_eq res(fmet.derive_with_other(type_der, ind_der, so, this));

        // Add the part containing the Christoffel :
        // Must find a name for summation on Christofel :
        bool found = false;
        int start = 97;
        do {
            bool same = false;
            if (ind_der == char(start))
                same = true;
            for (int i = 0; i < so.val_t->get_valence(); i++)
                if (so.val_t->get_name_ind()[i] == char(start))
                    same = true;
            if (!same)
                found = true;
            else
                start++;
        } while ((!found) && (start < 123));
        if (!found) {
            KADATH_THROW("Trouble with indices in derive (you are not using tensors of order > 24, are you ?)");
        }
        char name_sum = char(start);

        bool doder = ((so.der_t == nullptr) || (p_christo[dd]->der_t == nullptr)) ? false : true;

        // loop on the components of so :
        for (int cmp = 0; cmp < so.val_t->get_valence(); cmp++) {

            int genre_indice = so.val_t->get_index_type(cmp);

            // Affecte names on the Christoffels :
            p_christo[dd]->val_t->set_name_affected();
            p_christo[dd]->val_t->set_name_ind(0, ind_der);
            if (genre_indice == COV) {
                p_christo[dd]->val_t->set_name_ind(1, so.val_t->get_name_ind()[cmp]);
                p_christo[dd]->val_t->set_name_ind(2, name_sum);
            } else {
                p_christo[dd]->val_t->set_name_ind(2, so.val_t->get_name_ind()[cmp]);
                p_christo[dd]->val_t->set_name_ind(1, name_sum);
            }

            if (doder) {
                p_christo[dd]->der_t->set_name_affected();
                p_christo[dd]->der_t->set_name_ind(0, ind_der);
                if (genre_indice == COV) {
                    p_christo[dd]->der_t->set_name_ind(1, so.der_t->get_name_ind()[cmp]);
                    p_christo[dd]->der_t->set_name_ind(2, name_sum);
                } else {
                    p_christo[dd]->der_t->set_name_ind(2, so.der_t->get_name_ind()[cmp]);
                    p_christo[dd]->der_t->set_name_ind(1, name_sum);
                }
            }

            Term_eq auxi_christ(*p_christo[dd]);

            if (type_der == CON)
                manipulate_ind(auxi_christ, 0);

            // Check if one inner summation is needed :
            bool need_sum = false;
            char* ind = p_christo[dd]->val_t->get_name_ind();
            if ((ind[0] == ind[2]) || (ind[1] == ind[2]) || (ind[0] == ind[1]))
                need_sum = true;

            Term_eq* christ;
            if (need_sum)
                if (!doder)
                    christ = new Term_eq(dd, auxi_christ.val_t->do_summation_one_dom(dd));
                else
                    christ = new Term_eq(dd, auxi_christ.val_t->do_summation_one_dom(dd),
                                         auxi_christ.der_t->do_summation_one_dom(dd));
            else
                christ = new Term_eq(auxi_christ);

            // Affecte names on the field :
            Term_eq copie(so);
            copie.val_t->set_name_affected();
            for (int i = 0; i < so.val_t->get_valence(); i++)
                if (i != cmp)
                    copie.val_t->set_name_ind(i, so.val_t->get_name_ind()[i]);
                else
                    copie.val_t->set_name_ind(i, name_sum);
            if (doder) {
                copie.der_t->set_name_affected();
                for (int i = 0; i < so.der_t->get_valence(); i++)
                    if (i != cmp)
                        copie.der_t->set_name_ind(i, so.der_t->get_name_ind()[i]);
                    else
                        copie.der_t->set_name_ind(i, name_sum);
            }

            Term_eq part_christo((*christ) * copie);
            delete christ;

            if (genre_indice == CON)
                res = res + part_christo;
            else
                res = res - part_christo;
        }
        return res;
    }

    void Metric_cfc::set_system(System_of_eqs& ss, const char* name_met)
    {

        syst = &ss;

        // Position in the system :
        place_syst = ss.ndom * ss.nvar;

        //  unknown for the system (no name, the name is in the metric already)
        ss.add_var(nullptr, *p_conf);

        if (ss.met != nullptr) {
            KADATH_THROW("Metric already set for the system");
        }

        ss.attach_metric(*this, name_met);
    }
} // namespace Kadath
