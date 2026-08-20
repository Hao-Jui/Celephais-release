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
#include "For_Kadath/Array/exceptions.hpp"
#include "For_Kadath/Tensor/tensor.hpp"
#include "For_Kadath/Metric/metric.hpp"
#include "For_Kadath/Term_eq/term_eq.hpp"
#include "For_Kadath/Scalar/scalar.hpp"
#include "For_Kadath/System_of_eqs/system_of_eqs.hpp"

#include <array>
namespace Kadath
{
    namespace
    {
        Term_eq make_summed_metric_term(int domain, const Term_eq& source)
        {
            auto derivative_ready_for_summation = [&source](int lane) {
                Tensor derivative(source.get_der_t(lane));
                if (!derivative.is_name_affected() && source.get_val_t().is_name_affected()) {
                    derivative.set_name_affected();
                    for (int index = 0; index < derivative.get_valence(); index++)
                        derivative.set_name_ind(index, source.get_val_t().get_name_ind()[index]);
                }
                return derivative;
            };

            Tensor summed_value(source.get_val_t().do_summation_one_dom(domain));
            if (source.get_p_der_t() == nullptr)
                return Term_eq(domain, summed_value);

            Tensor primary_derivative(derivative_ready_for_summation(0));
            Tensor summed_derivative(primary_derivative.do_summation_one_dom(domain));
            Term_eq result(domain, summed_value, summed_derivative);
            result.set_derivative_lane_count(source.get_derivative_lane_count());
            for (int lane = 1; lane < source.get_derivative_lane_count(); ++lane) {
                if (source.has_der_t(lane)) {
                    Tensor derivative(derivative_ready_for_summation(lane));
                    result.set_der_t(lane, derivative.do_summation_one_dom(domain));
                }
            }
            return result;
        }
    }

    Metric::Metric(const Space& sp) : espace(sp), syst(nullptr), type_tensor(0)
    {
        p_met_cov.resize(espace.get_nbr_domains());
        p_met_con.resize(espace.get_nbr_domains());
        p_christo.resize(espace.get_nbr_domains());
        p_riemann.resize(espace.get_nbr_domains());
        p_ricci_tensor.resize(espace.get_nbr_domains());
        p_ricci_scalar.resize(espace.get_nbr_domains());
        p_dirac.resize(espace.get_nbr_domains());
        p_det_cov.resize(espace.get_nbr_domains());
        for (int i = 0; i < espace.get_nbr_domains(); i++) {
            p_met_con[i] = nullptr;
            p_met_cov[i] = nullptr;
            p_christo[i] = nullptr;
            p_riemann[i] = nullptr;
            p_ricci_tensor[i] = nullptr;
            p_ricci_scalar[i] = nullptr;
            p_dirac[i] = nullptr;
            p_det_cov[i] = nullptr;
        }
    }

    Metric::Metric(const Metric& so) : espace(so.espace), syst(so.syst), type_tensor(so.type_tensor)
    {

        p_met_cov.resize(espace.get_nbr_domains());
        p_met_con.resize(espace.get_nbr_domains());
        p_christo.resize(espace.get_nbr_domains());
        p_riemann.resize(espace.get_nbr_domains());
        p_ricci_tensor.resize(espace.get_nbr_domains());
        p_ricci_scalar.resize(espace.get_nbr_domains());
        p_dirac.resize(espace.get_nbr_domains());
        p_det_cov.resize(espace.get_nbr_domains());
        for (int i = 0; i < espace.get_nbr_domains(); i++) {
            if (so.p_met_cov[i] != nullptr)
                p_met_cov[i] = std::make_unique<Term_eq>(*so.p_met_cov[i]);
            if (so.p_met_con[i] != nullptr)
                p_met_con[i] = std::make_unique<Term_eq>(*so.p_met_con[i]);
            if (so.p_christo[i] != nullptr)
                p_christo[i] = std::make_unique<Term_eq>(*so.p_christo[i]);
            if (so.p_riemann[i] != nullptr)
                p_riemann[i] = std::make_unique<Term_eq>(*so.p_riemann[i]);
            if (so.p_ricci_tensor[i] != nullptr)
                p_ricci_tensor[i] = std::make_unique<Term_eq>(*so.p_ricci_tensor[i]);
            if (so.p_ricci_scalar[i] != nullptr)
                p_ricci_scalar[i] = std::make_unique<Term_eq>(*so.p_ricci_scalar[i]);
            if (so.p_dirac[i] != nullptr)
                p_dirac[i] = std::make_unique<Term_eq>(*so.p_dirac[i]);
            if (so.p_det_cov[i] != nullptr)
                p_det_cov[i] = std::make_unique<Term_eq>(*so.p_det_cov[i]);
        }
    }

    Metric::~Metric() = default;

    void Metric::update(int i)
    {

        if (type_tensor == CON) {
            if (p_met_con[i] != nullptr)
                compute_con(i);
            if (p_met_cov[i] != nullptr)
                compute_cov(i);
        } else {
            if (p_met_cov[i] != nullptr)
                compute_cov(i);
            if (p_met_con[i] != nullptr)
                compute_con(i);
        }
        if (p_det_cov[i] != nullptr)
            compute_det_cov(i);
        if (p_christo[i] != nullptr)
            compute_christo(i);
        if (p_riemann[i] != nullptr)
            compute_riemann(i);
        if (p_dirac[i] != nullptr)
            compute_dirac(i);
        if (p_ricci_tensor[i] != nullptr)
            compute_ricci_tensor(i);
        if (p_ricci_scalar[i] != nullptr)
            compute_ricci_scalar(i);
    }

    void Metric::update()
    {
        for (int i = 0; i < espace.get_nbr_domains(); i++)
            update(i);
    }

    const Term_eq* Metric::give_term(int dd, int type) const
    {
        assert((dd >= 0) && (dd < espace.get_nbr_domains()));
        switch (type) {
            case COV:
                if (p_met_cov[dd] == nullptr)
                    compute_cov(dd);
                return p_met_cov[dd].get();
            case CON:
                if (p_met_con[dd] == nullptr)
                    compute_con(dd);
                return p_met_con[dd].get();
            default:
                KADATH_THROW("Unknown type of indice in Metric::give_term");
        }
    }

    int Metric::give_type(int dd) const
    {
        if (p_met_cov[dd] == nullptr)
            compute_cov(dd);
        return (p_met_cov[dd]->val_t->get_basis().get_basis(dd));
    }

    const Term_eq* Metric::give_christo(int dd) const
    {
        assert((dd >= 0) && (dd < espace.get_nbr_domains()));
        if (p_christo[dd] == nullptr)
            compute_christo(dd);
        return p_christo[dd].get();
    }

    const Term_eq* Metric::give_riemann(int dd) const
    {
        assert((dd >= 0) && (dd < espace.get_nbr_domains()));
        if (p_riemann[dd] == nullptr)
            compute_riemann(dd);
        return p_riemann[dd].get();
    }

    const Term_eq* Metric::give_ricci_tensor(int dd) const
    {
        assert((dd >= 0) && (dd < espace.get_nbr_domains()));
        if (p_ricci_tensor[dd] == nullptr)
            compute_ricci_tensor(dd);
        return p_ricci_tensor[dd].get();
    }

    const Term_eq* Metric::give_ricci_scalar(int dd) const
    {
        assert((dd >= 0) && (dd < espace.get_nbr_domains()));
        if (p_ricci_scalar[dd] == nullptr)
            compute_ricci_scalar(dd);
        return p_ricci_scalar[dd].get();
    }

    const Term_eq* Metric::give_dirac(int dd) const
    {
        assert((dd >= 0) && (dd < espace.get_nbr_domains()));
        if (p_dirac[dd] == nullptr)
            compute_dirac(dd);
        return p_dirac[dd].get();
    }

    const Term_eq* Metric::give_det_cov(int dd) const
    {
        assert((dd >= 0) && (dd < espace.get_nbr_domains()));
        if (p_det_cov[dd] == nullptr)
            compute_det_cov(dd);
        return p_det_cov[dd].get();
    }

    const Metric* Metric::get_background() const
    {
        KADATH_THROW("No background metric defined");
    }

    Term_eq Metric::derive_partial(int type_der, char ind_der, const Term_eq& so) const
    {

        int dom = so.get_dom();

        if (type_tensor == CON) {
            if (p_met_con[dom] == nullptr)
                compute_con(dom);
            if (p_met_cov[dom] == nullptr)
                compute_cov(dom);
        } else {
            if (p_met_cov[dom] == nullptr)
                compute_cov(dom);
            if (p_met_con[dom] == nullptr)
                compute_con(dom);
        }

        // so must be tensor :
        if (so.get_type_data() != TERM_T) {
            KADATH_THROW("Metric::derive partial only defined for tensor data");
        }

        // Check that the triad is good :
        if ((so.val_t->get_valence() > 0) && (so.val_t->get_basis().get_basis(dom) != CARTESIAN_BASIS)) {
            KADATH_THROW("Metric::derive_partial only defined for tensor on cartesian triad");
        }

        bool doder = ((so.der_t == nullptr) || (p_met_cov[dom]->der_t == nullptr) || (p_met_cov[dom]->der_t == nullptr))
                         ? false
                         : true;

        assert((type_der == COV) || (type_der == CON));
        int val_res = so.val_t->get_valence() + 1;

        bool doname = true;
        if (so.val_t->get_valence() > 0)
            if (!so.val_t->is_name_affected())
                doname = false;

        Array<int> type_ind(val_res);
        type_ind.set(0) = COV;
        for (int i = 1; i < val_res; i++)
            type_ind.set(i) = so.val_t->get_index_type(i - 1);

        // Need for summation ?
        bool need_sum = false;
        if (doname)
            for (int i = 1; i < val_res; i++)
                if (ind_der == so.val_t->get_name_ind()[i - 1])
                    need_sum = true;

        // Tensor for val
        Tensor auxi_val(espace, val_res, type_ind, so.val_t->get_basis());
        // Set the names of the indices :
        if (doname) {
            auxi_val.set_name_affected();
            auxi_val.set_name_ind(0, ind_der);
            for (int i = 1; i < val_res; i++)
                auxi_val.set_name_ind(i, so.val_t->get_name_ind()[i - 1]);
        }

        // Loop on the components :
        Index pos_auxi(auxi_val);
        Index pos_so(*so.val_t);
        do {
            for (int i = 0; i < val_res - 1; i++)
                pos_so.set(i) = pos_auxi(i + 1);
            auxi_val.set(pos_auxi).set_domain(dom) = (*so.val_t)(pos_so)(dom).der_abs(pos_auxi(0) + 1);
        } while (pos_auxi.inc());

        if (!doder) {
            // No need for derivative :
            Term_eq auxi(dom, auxi_val);
            // If derive contravariant : manipulate first indice :
            if (type_der == CON)
                manipulate_ind(auxi, 0);

            if (!need_sum)
                return auxi;
            else
                return make_summed_metric_term(dom, auxi);
        } else {
            auto build_derivative = [&](const Tensor& source_derivative) {
                Tensor auxi_der(espace, val_res, type_ind, source_derivative.get_basis());
                // Set the names of the indices :
                auxi_der.set_name_affected();
                auxi_der.set_name_ind(0, ind_der);
                for (int i = 1; i < val_res; i++)
                    auxi_der.set_name_ind(i, source_derivative.get_name_ind()[i - 1]);

                // Loop on the components :
                Index pos_auxi_der(auxi_der);
                do {
                    for (int i = 0; i < val_res - 1; i++)
                        pos_so.set(i) = pos_auxi_der(i + 1);
                    auxi_der.set(pos_auxi_der).set_domain(dom) =
                        source_derivative(pos_so)(dom).der_abs(pos_auxi_der(0) + 1);
                } while (pos_auxi_der.inc());
                return auxi_der;
            };

            // Need for derivative :
            Term_eq auxi(dom, auxi_val, build_derivative(*so.der_t));
            auxi.set_derivative_lane_count(so.get_derivative_lane_count());
            for (int lane = 1; lane < so.get_derivative_lane_count(); ++lane) {
                if (so.has_der_t(lane))
                    auxi.set_der_t(lane, build_derivative(so.get_der_t(lane)));
            }
            // If derive contravariant : manipulate first indice :
            if (type_der == CON)
                manipulate_ind(auxi, 0);

            if (!need_sum)
                return auxi;
            else
                return make_summed_metric_term(dom, auxi);
        }
    }

    Term_eq Metric::derive(int type_der, char ind_der, const Term_eq& so) const
    {

        // The partial derivative part
        Term_eq res(derive_partial(type_der, ind_der, so));

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

        int dd = so.get_dom();
        if (p_christo[dd] == nullptr)
            compute_christo(dd);
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
                christ = new Term_eq(make_summed_metric_term(dd, auxi_christ));
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

    Term_eq Metric::derive_flat(int, char, const Term_eq&) const
    {
        KADATH_THROW("derive_flat not implemented for this type of metric (yet)");
    }

    void Metric::compute_christo(int dd) const
    {

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

        // Get the conformal factor
        bool doder = ((p_met_con[dd]->der_t == nullptr) || (p_met_cov[dd]->der_t == nullptr)) ? false : true;

        Array<int> type_ind(3);
        type_ind.set(0) = COV;
        type_ind.set(1) = COV;
        type_ind.set(2) = CON;
        Tensor res_val(espace, 3, type_ind, p_met_con[dd]->val_t->get_basis());
        Tensor res_der(espace, 3, type_ind, p_met_con[dd]->val_t->get_basis());

        res_val = 0;
        res_der = 0;

        Index pos(res_val);
        do {
            Val_domain cmpval(espace.get_domain(dd));
            cmpval = 0;

            for (int l = 1; l <= espace.get_ndim(); l++)
                cmpval += 0.5 * ((*p_met_con[dd]->val_t)(pos(2) + 1, l)(dd) *
                                 ((*p_met_cov[dd]->val_t)(pos(1) + 1, l)(dd).der_abs(pos(0) + 1) +
                                  (*p_met_cov[dd]->val_t)(pos(0) + 1, l)(dd).der_abs(pos(1) + 1) -
                                  (*p_met_cov[dd]->val_t)(pos(1) + 1, pos(0) + 1)(dd).der_abs(l)));

            res_val.set(pos).set_domain(dd) = cmpval;

            if (doder) {
                Val_domain cmpder(espace.get_domain(dd));
                cmpder = 0;

                for (int l = 1; l <= espace.get_ndim(); l++)
                    cmpder += 0.5 * ((*p_met_con[dd]->der_t)(pos(2) + 1, l)(dd) *
                                     ((*p_met_cov[dd]->val_t)(pos(1) + 1, l)(dd).der_abs(pos(0) + 1) +
                                      (*p_met_cov[dd]->val_t)(pos(0) + 1, l)(dd).der_abs(pos(1) + 1) -
                                      (*p_met_cov[dd]->val_t)(pos(1) + 1, pos(0) + 1)(dd).der_abs(l))) +
                              0.5 * ((*p_met_con[dd]->val_t)(pos(2) + 1, l)(dd) *
                                     ((*p_met_cov[dd]->der_t)(pos(1) + 1, l)(dd).der_abs(pos(0) + 1) +
                                      (*p_met_cov[dd]->der_t)(pos(0) + 1, l)(dd).der_abs(pos(1) + 1) -
                                      (*p_met_cov[dd]->der_t)(pos(1) + 1, pos(0) + 1)(dd).der_abs(l)));

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

    void Metric::compute_riemann(int dd) const
    {
        // Need christoffels
        if (p_christo[dd] == nullptr)
            compute_christo(dd);

        // Get the conformal factor
        bool doder = ((p_met_con[dd]->der_t == nullptr) || (p_met_cov[dd]->der_t == nullptr)) ? false : true;
        Array<int> indices(4);
        indices.set(0) = CON;
        indices.set(1) = COV;
        indices.set(2) = COV;
        indices.set(3) = COV;
        Tensor res_val(espace, 4, indices, p_met_con[dd]->val_t->get_basis());
        Tensor res_der(espace, 4, indices, p_met_con[dd]->val_t->get_basis());

        Index pos(res_val);
        do {
            Val_domain cmpval(espace.get_domain(dd));

            cmpval = (*p_christo[dd]->val_t)(pos(1) + 1, pos(3) + 1, pos(0) + 1)(dd).der_abs(pos(2) + 1) -
                     (*p_christo[dd]->val_t)(pos(1) + 1, pos(2) + 1, pos(0) + 1)(dd).der_abs(pos(3) + 1);

            for (int m = 1; m <= espace.get_ndim(); m++)
                cmpval += (*p_christo[dd]->val_t)(pos(2) + 1, m, pos(0) + 1)(dd) *
                              (*p_christo[dd]->val_t)(pos(1) + 1, pos(3) + 1, m)(dd) -
                          (*p_christo[dd]->val_t)(pos(3) + 1, m, pos(0) + 1)(dd) *
                              (*p_christo[dd]->val_t)(pos(1) + 1, pos(2) + 1, m)(dd);
            res_val.set(pos).set_domain(dd) = cmpval;

            if (doder) {
                Val_domain cmpder(espace.get_domain(dd));

                cmpder = (*p_christo[dd]->der_t)(pos(1) + 1, pos(3) + 1, pos(0) + 1)(dd).der_abs(pos(2) + 1) -
                         (*p_christo[dd]->der_t)(pos(1) + 1, pos(2) + 1, pos(0) + 1)(dd).der_abs(pos(3) + 1);

                for (int m = 1; m <= espace.get_ndim(); m++)
                    cmpder += (*p_christo[dd]->der_t)(pos(2) + 1, m, pos(0) + 1)(dd) *
                                  (*p_christo[dd]->val_t)(pos(1) + 1, pos(3) + 1, m)(dd) +
                              (*p_christo[dd]->val_t)(pos(2) + 1, m, pos(0) + 1)(dd) *
                                  (*p_christo[dd]->der_t)(pos(1) + 1, pos(3) + 1, m)(dd) -
                              (*p_christo[dd]->der_t)(pos(3) + 1, m, pos(0) + 1)(dd) *
                                  (*p_christo[dd]->val_t)(pos(1) + 1, pos(2) + 1, m)(dd) -
                              (*p_christo[dd]->val_t)(pos(3) + 1, m, pos(0) + 1)(dd) *
                                  (*p_christo[dd]->der_t)(pos(1) + 1, pos(2) + 1, m)(dd);

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

    void Metric::compute_ricci_tensor(int dd) const
    {
        // Need christoffels
        if (p_christo[dd] == nullptr)
            compute_christo(dd);

        // Get the conformal factor
        bool doder = ((p_met_con[dd]->der_t == nullptr) || (p_met_cov[dd]->der_t == nullptr)) ? false : true;
        Tensor res_val(espace, 2, COV, p_met_con[dd]->val_t->get_basis());
        Tensor res_der(espace, 2, COV, p_met_con[dd]->val_t->get_basis());

        Index pos(res_val);
        do {
            Val_domain cmpval(espace.get_domain(dd));
            cmpval = 0;

            for (int k = 1; k <= espace.get_ndim(); k++) {
                cmpval += (*p_christo[dd]->val_t)(pos(0) + 1, pos(1) + 1, k)(dd).der_abs(k) -
                          (*p_christo[dd]->val_t)(pos(1) + 1, k, k)(dd).der_abs(pos(0) + 1);
                for (int l = 1; l <= espace.get_ndim(); l++)
                    cmpval +=
                        (*p_christo[dd]->val_t)(k, l, l)(dd) * (*p_christo[dd]->val_t)(pos(0) + 1, pos(1) + 1, k)(dd) -
                        (*p_christo[dd]->val_t)(pos(0) + 1, k, l)(dd) * (*p_christo[dd]->val_t)(pos(1) + 1, l, k)(dd);
            }

            res_val.set(pos).set_domain(dd) = cmpval;

            if (doder) {
                Val_domain cmpder(espace.get_domain(dd));
                cmpder = 0;

                for (int k = 1; k <= espace.get_ndim(); k++) {
                    cmpder += (*p_christo[dd]->der_t)(pos(0) + 1, pos(1) + 1, k)(dd).der_abs(k) -
                              (*p_christo[dd]->der_t)(pos(1) + 1, k, k)(dd).der_abs(pos(0) + 1);
                    for (int l = 1; l <= espace.get_ndim(); l++)
                        cmpder += (*p_christo[dd]->der_t)(k, l, l)(dd) *
                                      (*p_christo[dd]->val_t)(pos(0) + 1, pos(1) + 1, k)(dd) +
                                  (*p_christo[dd]->val_t)(k, l, l)(dd) *
                                      (*p_christo[dd]->der_t)(pos(0) + 1, pos(1) + 1, k)(dd) -
                                  (*p_christo[dd]->der_t)(pos(0) + 1, k, l)(dd) *
                                      (*p_christo[dd]->val_t)(pos(1) + 1, l, k)(dd) -
                                  (*p_christo[dd]->val_t)(pos(0) + 1, k, l)(dd) *
                                      (*p_christo[dd]->der_t)(pos(1) + 1, l, k)(dd);
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

    void Metric::compute_ricci_scalar(int dd) const
    {

        // Need that
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

    void Metric::manipulate_ind(Term_eq& so, int ind) const
    {

        int dd = so.get_dom();
        int valence = so.get_val_t().get_valence();
        int type_start = so.get_val_t().get_index_type(ind);

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

        Term_eq& metric_term = (type_start == COV) ? *p_met_con[dd] : *p_met_cov[dd];

        bool doder = (metric_term.der_t != nullptr) && (so.der_t != nullptr);
        const int lane_count = (metric_term.get_derivative_lane_count() > so.get_derivative_lane_count())
                                   ? metric_term.get_derivative_lane_count()
                                   : so.get_derivative_lane_count();

        Array<int> type_res(valence);
        for (int i = 0; i < valence; i++)
            type_res.set(i) = (i == ind) ? -so.get_val_t().get_index_type(i) : so.get_val_t().get_index_type(i);

        Tensor val_res(espace, valence, type_res, metric_term.val_t->get_basis());
        Tensor val_der(espace, valence, type_res, metric_term.val_t->get_basis());
        std::array<Tensor*, Term_eq::max_derivative_lanes - 1> val_der_lanes{nullptr, nullptr, nullptr};
        for (int lane = 1; lane < lane_count; ++lane) {
            if (metric_term.has_der_t(lane) || so.has_der_t(lane))
                val_der_lanes[static_cast<std::size_t>(lane - 1)] =
                    new Tensor(espace, valence, type_res, metric_term.val_t->get_basis());
        }

        auto fill_derivative = [&](Tensor& target, const Tensor* metric_derivative, const Tensor* source_derivative) {
            Index derivative_position(target);
            do {
                Val_domain cmpder(espace.get_domain(dd));
                cmpder = 0;
                for (int k = 0; k < espace.get_ndim(); k++) {
                    Index copie(derivative_position);
                    copie.set(ind) = k;
                    if (metric_derivative != nullptr)
                        cmpder +=
                            (*metric_derivative)(derivative_position(ind) + 1, k + 1)(dd) * (*so.val_t)(copie)(dd);
                    if (source_derivative != nullptr)
                        cmpder += (*metric_term.val_t)(derivative_position(ind) + 1, k + 1)(dd) *
                                  (*source_derivative)(copie)(dd);
                }
                target.set(derivative_position).set_domain(dd) = cmpder;
            } while (derivative_position.inc());
        };

        Index pos(val_res);
        do {
            Val_domain cmpval(espace.get_domain(dd));
            cmpval = 0;

            for (int k = 0; k < espace.get_ndim(); k++) {
                Index copie(pos);
                copie.set(ind) = k;
                cmpval += (*metric_term.val_t)(pos(ind) + 1, k + 1)(dd) * (*so.val_t)(copie)(dd);
            }

            val_res.set(pos).set_domain(dd) = cmpval;
        } while (pos.inc());

        if (doder)
            fill_derivative(val_der, metric_term.der_t, so.der_t);
        for (int lane = 1; lane < lane_count; ++lane) {
            Tensor* lane_target = val_der_lanes[static_cast<std::size_t>(lane - 1)];
            if (lane_target != nullptr)
                fill_derivative(*lane_target, metric_term.get_p_der_t(lane), so.get_p_der_t(lane));
        }

        // Put the names :
        if (so.get_val_t().is_name_affected()) {
            val_res.set_name_affected();
            for (int i = 0; i < valence; i++)
                val_res.set_name_ind(i, so.val_t->get_name_ind()[i]);
        }

        if ((doder) && (so.get_der_t().is_name_affected())) {
            val_der.set_name_affected();
            for (int i = 0; i < valence; i++)
                val_der.set_name_ind(i, so.der_t->get_name_ind()[i]);
        }
        for (int lane = 1; lane < lane_count; ++lane) {
            Tensor* lane_target = val_der_lanes[static_cast<std::size_t>(lane - 1)];
            if (lane_target != nullptr && so.has_der_t(lane) && so.get_der_t(lane).is_name_affected()) {
                lane_target->set_name_affected();
                for (int i = 0; i < valence; i++)
                    lane_target->set_name_ind(i, so.get_der_t(lane).get_name_ind()[i]);
            }
        }

        so.set_val_t()->set_index_type(ind) *= -1;
        if (so.set_der_t() != nullptr)
            so.set_der_t()->set_index_type(ind) *= -1;

        delete so.val_t;
        so.val_t = new Tensor(val_res);
        delete so.der_t;
        so.der_t = (doder) ? new Tensor(val_der) : nullptr;
        so.set_derivative_lane_count(lane_count);
        for (int lane = 1; lane < lane_count; ++lane) {
            const std::size_t slot = static_cast<std::size_t>(lane - 1);
            Tensor*& lane_ptr = val_der_lanes[slot];
            if (lane_ptr != nullptr) {
                so.set_der_t(lane, *lane_ptr);
                delete lane_ptr;
                lane_ptr = nullptr;
            } else {
                so.clear_der(lane);
            }
        }
    }

    void Metric::compute_cov(int) const
    {
        KADATH_THROW("compute_cov not implemented for this type of metric");
    }

    void Metric::compute_con(int) const
    {
        KADATH_THROW("compute_con not implemented for this type of metric");
    }

    void Metric::compute_dirac(int) const
    {
        KADATH_THROW("Compute Dirac not implemented for this type of metric");
    }

    void Metric::compute_det_cov(int) const
    {
        KADATH_THROW("Compute determinant of covariant not implemented for this type of metric");
    }
} // namespace Kadath
