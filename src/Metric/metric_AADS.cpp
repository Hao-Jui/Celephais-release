/*
    Copyright 2017 Grégoire Martinon

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
 *   2026-08-07  Replaced GSL permutation handling with the C++ standard library.
 */

#include <algorithm>
#include <numeric>

#include "For_Kadath/Metric/metric_AADS.hpp"
#include "For_Kadath/Array/exceptions.hpp"
namespace Kadath
{
    Metric_AADS::Metric_AADS(const Space& space, Metric_tensor& hmet)
        : Metric(space), m_doder(false), m_ads(space), p_hmet(&hmet)
    {
        int dim(space.get_ndim());
        assert(dim == 3);

        p_basis = new Base_tensor(space, CARTESIAN_BASIS);
        p_der_eps = new Vector(space, COV, *p_basis);

        type_tensor = COV; // assume covariance
        assert(hmet.get_type() == COV);
        m_nd = espace.get_nbr_domains();               // discard infinity (last domain)
        m_L = espace.get_domain(m_nd - 1)->get_rmax(); // AdS length

        p_h.resize(m_nd);
        p_k.resize(m_nd);

        for (int d(0); d < m_nd; ++d)
            for (int i(1); i <= dim; ++i)
                p_der_eps->set(i).set_domain(d) = -4.0 * espace.get_domain(d)->get_cart(i) / m_L / m_L /
                                                  pow(1.0 + m_ads.m_rho2(d), 2); // derivative of eps
    }

    Metric_AADS::Metric_AADS(const Metric_AADS& so)
        : Metric(so), m_nd(so.m_nd), m_L(so.m_L), m_doder(so.m_doder), m_ads(so.m_ads)
    {
        p_h.resize(m_nd);
        p_k.resize(m_nd);
        for (int d(0); d < espace.get_nbr_domains(); ++d) {
            if (so.p_h[d] != nullptr)
                p_h[d] = std::make_unique<Term_eq>(*so.p_h[d]);
            if (so.p_k[d] != nullptr)
                p_k[d] = std::make_unique<Term_eq>(*so.p_k[d]);
        }
        p_basis = new Base_tensor(*so.p_basis);
        p_der_eps = new Vector(*so.p_der_eps);
        p_hmet = new Metric_tensor(*so.p_hmet);
    }

    Metric_AADS::~Metric_AADS()
    {
        delete p_basis;
        delete p_der_eps;
    }

    void Metric_AADS::update(int d)
    {
        assert(type_tensor == COV);
        if (p_met_cov[d] != nullptr)
            compute_cov(d);
        if (p_met_con[d] != nullptr)
            compute_con(d);
        if (p_christo[d] != nullptr)
            compute_christo(d);
        if (p_ricci_tensor[d] != nullptr)
            compute_ricci_tensor(d);
        if (p_ricci_scalar[d] != nullptr)
            compute_ricci_scalar(d);
    }

    void Metric_AADS::update()
    {
        for (int d(0); d < m_nd; ++d)
            update(d);
    }

    void Metric_AADS::compute_cov(int d) const // compute covariant metric in domain d
    {                                          // eps^2*gama_ij is stored
        if (m_ads.p_met_cov[d] == nullptr)
            m_ads.compute_cov(d);
        if (syst != nullptr) {
            int place(
                m_place_syst +
                (d - syst->dom_min)); // where is the unknown eps^2*h Term_eq in the d-th domain in system of equations
            if (p_h[d] == nullptr)
                p_h[d] = std::make_unique<Term_eq>(*syst->term[place]);
            else
                *p_h[d] = *syst->term[place];

            if (p_met_cov[d] == nullptr)
                p_met_cov[d] = std::make_unique<Term_eq>(*m_ads.p_met_cov[d] + *p_h[d]);
            else
                *p_met_cov[d] = *m_ads.p_met_cov[d] + *p_h[d];
        } else {
            if (p_met_cov[d] == nullptr)
                p_met_cov[d] = std::make_unique<Term_eq>(d, *m_ads.give_term(d, COV)->val_t + *p_hmet);
            else
                *p_met_cov[d] = Term_eq(d, *m_ads.give_term(d, COV)->val_t + *p_hmet);
        }
        // at this point, metric is multiplied by epsilon^2
        m_doder = (p_met_cov[d]->der_t != nullptr);
    }

    void Metric_AADS::compute_con(int d) const // compute contravariant metric in domain d
    {
        if (p_met_cov[d] == nullptr)
            compute_cov(d); // need the covariant metric
        if (m_ads.p_met_con[d] == nullptr)
            m_ads.compute_con(d);   // need the background contravariant metric
        int dim(espace.get_ndim()); // dimension
        assert(dim == 3);           // what follows only valid for 3D but is easily generalisable

        std::vector<std::unique_ptr<Term_eq>> res(dim * (dim + 1) / 2); // 6 components to compute...
        Scalar val(espace); // val and cmpval are more or less the same intermediate, but with different types
        Scalar der(espace);
        Val_domain cmpval(espace.get_domain(d));
        Val_domain cmpder(espace.get_domain(d));

        Val_domain detval(espace.get_domain(d)); // let's compute the determinant...
        Val_domain detder(espace.get_domain(d)); // ... and its automatic derivative
        detval = 0.0;
        detder = 0.0;

        // compute determinant as a sum over permutations
        double signature;
        std::vector<int> permutation(dim);
        std::iota(permutation.begin(), permutation.end(), 0);
        do {
            signature = sign(permutation);
            int p1(permutation[0] + 1); // tensor indices are one-based
            int p2(permutation[1] + 1);
            int p3(permutation[2] + 1);
            detval += signature * (*p_met_cov[d]->val_t)(p1, 1)(d) * (*p_met_cov[d]->val_t)(p2, 2)(d) *
                      (*p_met_cov[d]->val_t)(p3, 3)(d); // sum on permutations
            if (m_doder) {
                detder += signature * ((*p_met_cov[d]->der_t)(p1, 1)(d) * (*p_met_cov[d]->val_t)(p2, 2)(d) *
                                           (*p_met_cov[d]->val_t)(p3, 3)(d) +
                                       (*p_met_cov[d]->val_t)(p1, 1)(d) * (*p_met_cov[d]->der_t)(p2, 2)(d) *
                                           (*p_met_cov[d]->val_t)(p3, 3)(d) +
                                       (*p_met_cov[d]->val_t)(p1, 1)(d) * (*p_met_cov[d]->val_t)(p2, 2)(d) *
                                           (*p_met_cov[d]->der_t)(p3, 3)(d));
            }
        } while (std::next_permutation(permutation.begin(), permutation.end()));

        // compute the transposed comatrix
        for (int i(1); i <= dim; ++i) // sum over components, only the upper triangular part
        {
            for (int j(i); j <= dim; ++j) {
                cmpval = 0.0; // initialise to 0 before each computation
                cmpder = 0.0; // initialise to 0 before each computation
                permutation.resize(dim - 1);
                std::iota(permutation.begin(), permutation.end(), 0);
                do {
                    signature = sign(permutation);
                    int p1(permutation[0] + 1);
                    int p2(permutation[1] + 1);
                    vector<int> index1, index2;    // manage the indices of the transposed comatrix
                    index1 = ind_com(i, j, p1, 1); // we need to play a bit with indices to get transpose(COM(metric))
                    index2 = ind_com(i, j, p2, 2);
                    cmpval += pow(-1.0, i + j) * signature * (*p_met_cov[d]->val_t)(index1[0], index1[1])(d) *
                              (*p_met_cov[d]->val_t)(index2[0], index2[1])(d); // sum on permutations
                    if (m_doder)
                        cmpder += pow(-1.0, i + j) * signature * (*p_met_cov[d]->der_t)(index1[0], index1[1])(d) *
                                      (*p_met_cov[d]->val_t)(index2[0], index2[1])(d) +
                                  pow(-1.0, i + j) * signature * (*p_met_cov[d]->val_t)(index1[0], index1[1])(d) *
                                      (*p_met_cov[d]->der_t)(index2[0], index2[1])(d);
                } while (std::next_permutation(permutation.begin(), permutation.end()));
                val.set_domain(d) = cmpval / detval; // now we have the component of the inverse matrix
                if (m_doder) {
                    der.set_domain(d) = cmpder / detval - cmpval * detder / (detval * detval);
                    res[j - 1 + dim * (i - 1) - i * (i - 1) / 2] =
                        std::make_unique<Term_eq>(d, val, der); // just save upper triangular components in a 1D array of Term_eq (the
                                                  // indice will run throug 0, 1, 2, 3, 4, 5 with i, j)
                } else
                    res[j - 1 + dim * (i - 1) - i * (i - 1) / 2] = std::make_unique<Term_eq>(d, val);
            }
        }

        // Value field :
        Metric_tensor resval(espace, CON, *p_basis);
        Metric_tensor resder(espace, CON, *p_basis);
        for (int i(1); i <= dim; ++i)
            for (int j(i); j <= dim; ++j)
                resval.set(i, j) = res[j - 1 + dim * (i - 1) - i * (i - 1) / 2]->get_val_t();

        // Der field
        if (!m_doder) {
            if (p_met_con[d] == nullptr)
                p_met_con[d] = std::make_unique<Term_eq>(d, resval);
            else
                *p_met_con[d] = Term_eq(d, resval);
        } else {
            for (int i(1); i <= dim; ++i)
                for (int j(i); j <= dim; ++j)
                    resder.set(i, j) = res[j - 1 + dim * (i - 1) - i * (i - 1) / 2]->get_der_t();

            if (p_met_con[d] == nullptr)
                p_met_con[d] = std::make_unique<Term_eq>(d, resval, resder);
            else
                *p_met_con[d] = Term_eq(d, resval, resder);
        }

        // k^ij/eps^2
        resval = *p_met_con[d]->val_t - *m_ads.give_term(d, CON)->val_t;
        if (m_doder)
            resder = *p_met_con[d]->der_t - *m_ads.give_term(d, CON)->der_t;
        if (!m_doder) {
            if (p_k[d] == nullptr)
                p_k[d] = std::make_unique<Term_eq>(d, resval);
            else
                *p_k[d] = Term_eq(d, resval);
        } else {
            if (p_k[d] == nullptr)
                p_k[d] = std::make_unique<Term_eq>(d, resval, resder);
            else
                *p_k[d] = Term_eq(d, resval, resder);
        }
    }

    void Metric_AADS::manipulate_ind(Term_eq& so,
                                     int ind) const // rise or lower indice ind on term_eq via the metric (in place)
    {
        int d(so.get_dom()); // domain
        int dim(espace.get_ndim());
        int valence(so.get_val_t().get_valence());          // number of indices
        int type_start(so.get_val_t().get_index_type(ind)); // contravariant or covariant

        if (p_met_con[d] == nullptr and type_start == COV)
            compute_con(d); // make sure that metric is computed
        if (p_met_cov[d] == nullptr and type_start == CON)
            compute_cov(d);
        bool doder(so.der_t != nullptr and m_doder); // compute variations only if tensor need to
        Array<int> type_res(valence);                // CON or COV
        for (int i(0); i < valence; ++i)
            type_res.set(i) = (i == ind) ? -so.get_val_t().get_index_type(i)
                                         : so.get_val_t().get_index_type(i); // change the COV/CON of ind
        Tensor val_res(espace, valence, type_res, *p_basis);                 // tensor
        Tensor val_der(espace, valence, type_res, *p_basis);                 // its variation
        Val_domain cmpval(espace.get_domain(d));                             // component value
        Val_domain cmpder(espace.get_domain(d));
        Index pos(val_res);
        do // for all components
        {
            cmpval = 0.0;
            cmpder = 0.0;
            Index copie(pos);
            for (int k(0); k < dim; ++k) // loop on mute indice
            {
                copie.set(ind) = k; // loop on the indice to rise/lower
                if (type_start == COV)
                    cmpval += (*p_met_con[d]->val_t)(pos(ind) + 1, k + 1)(d) * (*so.val_t)(copie)(d); // g^ik T_jk
                else
                    cmpval += (*p_met_cov[d]->val_t)(pos(ind) + 1, k + 1)(d) * (*so.val_t)(copie)(d); // g_ik T^k_j
            }
            val_res.set(pos).set_domain(d) = cmpval; // assign cmpval to right component

            if (doder) // if variation of tensor needs to be computed
            {
                for (int k(0); k < dim; ++k) {
                    copie.set(ind) = k;
                    if (type_start == COV)
                        cmpder += (*p_met_con[d]->val_t)(pos(ind) + 1, k + 1)(d) * (*so.der_t)(copie)(d) +
                                  (*p_met_con[d]->der_t)(pos(ind) + 1, k + 1)(d) *
                                      (*so.val_t)(copie)(d); // recall that variations of background metric are not zero

                    else
                        cmpder += (*p_met_cov[d]->val_t)(pos(ind) + 1, k + 1)(d) * (*so.der_t)(copie)(d) +
                                  (*p_met_cov[d]->der_t)(pos(ind) + 1, k + 1)(d) * (*so.val_t)(copie)(d);
                }
                val_der.set(pos).set_domain(d) = cmpder; // assign cmpder to the component of the variation
            }
        } while (pos.inc());

        // Put the names :
        if (so.get_val_t().is_name_affected()) {
            val_res.set_name_affected();
            for (int i(0); i < valence; ++i)
                val_res.set_name_ind(i, so.val_t->get_name_ind()[i]); // result have same indice names as source
            if (doder) {
                val_der.set_name_affected();
                for (int i(0); i < valence; ++i)
                    val_der.set_name_ind(i, so.val_t->get_name_ind()[i]);
            }
        }

        delete so.val_t;
        so.val_t = new Tensor(val_res); // in place substitution
        delete so.der_t;
        so.der_t = doder ? new Tensor(val_der) : nullptr;

        // multiplication/division by eps^2
        if (type_start == CON)
            so = m_ads.div_eps(so, 2);
        else if (type_start == COV)
            so = m_ads.mul_eps(so, 2);
    }

    void Metric_AADS::compute_christo(
        int d) const // WARNING : christo will store the DIFFERENCE between background Christoffel and actual one
    {
        if (p_met_cov[d] == nullptr)
            compute_cov(d);
        if (p_met_con[d] == nullptr)
            compute_con(d);
        if (m_ads.p_christo[d] == nullptr)
            m_ads.compute_christo(d);

        int dim(espace.get_ndim());
        Array<int> type_ind(3);
        type_ind.set(0) = COV;
        type_ind.set(1) = COV;
        type_ind.set(2) = CON;                         // position of indices
        Tensor res_val(espace, 3, type_ind, *p_basis); // tensor to affect to the value of *p_christo
        Tensor res_der(espace, 3, type_ind, *p_basis); // tensor to affect to the variation of *p_christo
        res_val = 0.0;
        res_der = 0.0;
        Index pos(res_val);
        Val_domain cmpval(espace.get_domain(d)); // components of the tensor
        Val_domain cmpder(espace.get_domain(d));
        do {
            cmpval = 0.0;
            for (int l(1); l <= dim; ++l) {
                cmpval += 0.5 * m_ads.m_eps(d) *
                              (*p_met_con[d]->val_t)(pos(2) + 1, l)(d) // formule (7.30) 3+1 Eric Gourgoulhon
                              * ((*p_h[d]->val_t)(pos(1) + 1, l)(d).der_abs(pos(0) + 1) +
                                 (*p_h[d]->val_t)(pos(0) + 1, l)(d).der_abs(pos(1) + 1) -
                                 (*p_h[d]->val_t)(pos(0) + 1, pos(1) + 1)(d).der_abs(l)) -
                          (*p_met_con[d]->val_t)(pos(2) + 1, l)(d) *
                              ((*p_h[d]->val_t)(pos(1) + 1, l)(d) * (*p_der_eps)(pos(0) + 1)(d) +
                               (*p_h[d]->val_t)(pos(0) + 1, l)(d) * (*p_der_eps)(pos(1) + 1)(d) -
                               (*p_h[d]->val_t)(pos(0) + 1, pos(1) + 1)(d) * (*p_der_eps)(l)(d));
                for (int m(1); m <= dim; ++m)
                    cmpval -= (*p_met_con[d]->val_t)(pos(2) + 1, l)(d) *
                              (*m_ads.p_christo[d]->val_t)(pos(0) + 1, pos(1) + 1, m)(d) * (*p_h[d]->val_t)(l, m)(d);
            }
            res_val.set(pos).set_domain(d) = cmpval;

            if (m_doder) {
                cmpder = 0.0;
                for (int l(1); l <= dim; ++l) {
                    cmpder += 0.5 * m_ads.m_eps(d) *
                                  (*p_met_con[d]->der_t)(pos(2) + 1, l)(d) // formule (7.30) 3+1 Eric Gourgoulhon
                                  * ((*p_h[d]->val_t)(pos(1) + 1, l)(d).der_abs(pos(0) + 1) +
                                     (*p_h[d]->val_t)(pos(0) + 1, l)(d).der_abs(pos(1) + 1) -
                                     (*p_h[d]->val_t)(pos(0) + 1, pos(1) + 1)(d).der_abs(l)) +
                              0.5 * m_ads.m_eps(d) *
                                  (*p_met_con[d]->val_t)(pos(2) + 1, l)(d) // formule (7.30) 3+1 Eric Gourgoulhon
                                  * ((*p_h[d]->der_t)(pos(1) + 1, l)(d).der_abs(pos(0) + 1) +
                                     (*p_h[d]->der_t)(pos(0) + 1, l)(d).der_abs(pos(1) + 1) -
                                     (*p_h[d]->der_t)(pos(0) + 1, pos(1) + 1)(d).der_abs(l)) -
                              (*p_met_con[d]->der_t)(pos(2) + 1, l)(d) *
                                  ((*p_h[d]->val_t)(pos(0) + 1, l)(d) * (*p_der_eps)(pos(1) + 1)(d) +
                                   (*p_h[d]->val_t)(pos(1) + 1, l)(d) * (*p_der_eps)(pos(0) + 1)(d) -
                                   (*p_h[d]->val_t)(pos(0) + 1, pos(1) + 1)(d) * (*p_der_eps)(l)(d)) -
                              (*p_met_con[d]->val_t)(pos(2) + 1, l)(d) *
                                  ((*p_h[d]->der_t)(pos(0) + 1, l)(d) * (*p_der_eps)(pos(1) + 1)(d) +
                                   (*p_h[d]->der_t)(pos(1) + 1, l)(d) * (*p_der_eps)(pos(0) + 1)(d) -
                                   (*p_h[d]->der_t)(pos(0) + 1, pos(1) + 1)(d) * (*p_der_eps)(l)(d));
                    for (int m(1); m <= dim; ++m)
                        cmpder -=
                            (*p_met_con[d]->der_t)(pos(2) + 1, l)(d) *
                                (*m_ads.p_christo[d]->val_t)(pos(0) + 1, pos(1) + 1, m)(d) * (*p_h[d]->val_t)(l, m)(d) +
                            (*p_met_con[d]->val_t)(pos(2) + 1, l)(d) *
                                (*m_ads.p_christo[d]->val_t)(pos(0) + 1, pos(1) + 1, m)(d) * (*p_h[d]->der_t)(l, m)(d);
                }
                res_der.set(pos).set_domain(d) = cmpder;
            }
        } while (pos.inc());

        if (!m_doder) {
            if (p_christo[d] == nullptr)
                p_christo[d] = std::make_unique<Term_eq>(d, res_val);
            else
                *p_christo[d] = Term_eq(d, res_val);
        } else {
            if (p_christo[d] == nullptr)
                p_christo[d] = std::make_unique<Term_eq>(d, res_val, res_der);
            else
                *p_christo[d] = Term_eq(d, res_val, res_der);
        }
    }

    Term_eq Metric_AADS::derive_simple(const Term_eq& so)
        const // covariant derivative which does not affect indices and assume derivative indice is covariant
    {
        assert(espace.get_ndim() == 3);
        bool doder(so.der_t != nullptr and m_doder);
        int d(so.get_dom()); // domain
        int so_val(so.val_t->get_valence());
        Term_eq res(m_ads.derive_simple(so)); // The background covariant derivative part
        Tensor gam_res_arg(espace, res.val_t->get_valence(), res.val_t->get_index_type(),
                           *p_basis); // just to initialise a term_eq gam
        gam_res_arg = 0.0;
        Term_eq gam_res(d, gam_res_arg);
        if (doder)
            gam_res.set_der_zero();
        if (p_christo[d] == nullptr)
            compute_christo(d); // of course you need Christoffel
        Index posgamres(*gam_res.val_t);
        Index poschristo(*p_christo[d]->val_t);
        Index posso(*so.val_t);
        int genre_indice; // COV or CON
        for (int cmp(0); cmp < so_val;
             ++cmp) // loop on the components of so, compute one Christoffel term on each cycle
        {
            posgamres.set_start();                        // reinitialize the gam_res indice position
            genre_indice = so.val_t->get_index_type(cmp); // COV or CON
            do // loop on the components of gam_res, compute all components of the Chritoffel term under consideration
            {
                if (genre_indice == CON) {
                    poschristo.set(0) = posgamres(0);       // indice of D
                    poschristo.set(2) = posgamres(cmp + 1); // e.g. first indice of so is the second one of gam_res...
                                                            // poschristo.set(1) sumation, see below
                    for (int i(0); i < so_val; ++i)
                        if (i != cmp)
                            posso.set(i) = posgamres(i + 1); // e.g. first indice of so is the second one of gam_res...
                    for (int mute(0); mute < 3; ++mute)      // case sumation
                    {
                        poschristo.set(1) = mute;
                        posso.set(cmp) = mute;
                        gam_res.val_t->set(posgamres).set_domain(d) +=
                            (*p_christo[d]->val_t)(poschristo)(d) * (*so.val_t)(posso)(d);
                        if (doder)
                            gam_res.der_t->set(posgamres).set_domain(d) +=
                                (*p_christo[d]->val_t)(poschristo)(d) * (*so.der_t)(posso)(d) +
                                (*p_christo[d]->der_t)(poschristo)(d) * (*so.val_t)(posso)(d);
                    }
                }
                if (genre_indice == COV) {
                    poschristo.set(0) = posgamres(0);       // indice of D
                    poschristo.set(1) = posgamres(cmp + 1); // poschristo.set(2) summation, see below
                    for (int i(0); i < so_val; ++i)
                        if (i != cmp)
                            posso.set(i) = posgamres(i + 1); // e.g. first indice of so is the second one of gam_res...
                    for (int mute(0); mute < 3; ++mute)      // case sumation
                    {
                        poschristo.set(2) = mute;
                        posso.set(cmp) = mute;
                        gam_res.val_t->set(posgamres).set_domain(d) -=
                            (*p_christo[d]->val_t)(poschristo)(d) * (*so.val_t)(posso)(d);
                        if (doder)
                            gam_res.der_t->set(posgamres).set_domain(d) -=
                                (*p_christo[d]->val_t)(poschristo)(d) * (*so.der_t)(posso)(d) -
                                (*p_christo[d]->der_t)(poschristo)(d) * (*so.val_t)(posso)(d);
                    }
                }
            } while (posgamres.inc());
        }
        return res + m_ads.div_eps(gam_res, 1);
    }

    Term_eq Metric_AADS::derive(int type_der, char ind_der, const Term_eq& so) const // covariant derivative
    {
        int d(so.get_dom()); // domain
        int so_val(so.val_t->get_valence());
        bool doder(so.der_t != nullptr and m_doder);
        if (p_christo[d] == nullptr)
            compute_christo(d); // of course you need Christoffel
        Term_eq res(
            derive_simple(so)); // The background simple derivative part (no indice names affected, no inner summation)
        if (type_der == CON)
            manipulate_ind(res, 0); // manipulate indice if you want D^i

        // Set names and check inner summ
        bool inner_sum(false);
        if (so.val_t->is_name_affected()) {
            res.val_t->set_name_affected();
            res.val_t->set_name_ind(0, ind_der);
            for (int i(0); i < so_val; ++i) {
                res.val_t->set_name_ind(
                    i + 1,
                    so.val_t->get_name_ind()[i]); // put the names of ressult with name of D and names of tensor indices
                if (so.val_t->get_name_ind()[i] == ind_der)
                    inner_sum = true;
            }
            if (doder) {
                res.der_t->set_name_affected();
                res.der_t->set_name_ind(0, ind_der);
                for (int i(0); i < so_val; ++i)
                    res.der_t->set_name_ind(i + 1, so.val_t->get_name_ind()[i]);
            }
        } else if (so_val == 0) // manage the scalar case
        {
            res.val_t->set_name_affected();
            res.val_t->set_name_ind(0, ind_der);
            if (doder) {
                res.der_t->set_name_affected();
                res.der_t->set_name_ind(0, ind_der);
            }
        }
        if (inner_sum) {
            if (!doder)
                return Term_eq(d, res.val_t->do_summation_one_dom(d));
            else
                return Term_eq(d, res.val_t->do_summation_one_dom(d), res.der_t->do_summation_one_dom(d));
        } else
            return res;
    }

    void Metric_AADS::compute_ricci_tensor(int d) const // epsilon^2 *(Ricci - RicciB)
    {
        if (p_christo[d] == nullptr)
            compute_christo(d);
        if (m_ads.p_christo[d] == nullptr)
            m_ads.compute_christo(d);
        int dim(espace.get_ndim());
        Tensor res_val(espace, 2, COV, *p_basis); // the result to put into Ricci
        Tensor res_der(espace, 2, COV, *p_basis);
        Index pos(res_val);
        Val_domain cmpval(espace.get_domain(d)); // each component
        Val_domain cmpder(espace.get_domain(d));
        do {
            cmpval = 0.0;
            for (int k(1); k <= dim; ++k) // formula (7.40) 3+1 Eric Gourgoulhon
            {
                cmpval += m_ads.m_eps(d) * ((*p_christo[d]->val_t)(pos(0) + 1, pos(1) + 1, k)(d).der_abs(k) -
                                            (*p_christo[d]->val_t)(pos(1) + 1, k, k)(d).der_abs(pos(0) + 1)) -
                          (*p_christo[d]->val_t)(pos(0) + 1, pos(1) + 1, k)(d) * (*p_der_eps)(k)(d) +
                          (*p_christo[d]->val_t)(pos(1) + 1, k, k)(d) * (*p_der_eps)(pos(0) + 1)(d);
                for (int l(1); l <= dim; ++l)
                    cmpval +=
                        (*p_christo[d]->val_t)(k, l, l)(d) * (*p_christo[d]->val_t)(pos(0) + 1, pos(1) + 1, k)(d) -
                        (*p_christo[d]->val_t)(pos(0) + 1, l, k)(d) * (*p_christo[d]->val_t)(pos(1) + 1, k, l)(d) +
                        (*m_ads.p_christo[d]->val_t)(k, l, k)(d) *
                            (*p_christo[d]->val_t)(pos(0) + 1, pos(1) + 1, l)(d) -
                        (*m_ads.p_christo[d]->val_t)(k, pos(1) + 1, l)(d) *
                            (*p_christo[d]->val_t)(pos(0) + 1, l, k)(d) -
                        (*m_ads.p_christo[d]->val_t)(pos(0) + 1, l, k)(d) *
                            (*p_christo[d]->val_t)(pos(1) + 1, k, l)(d) +
                        (*m_ads.p_christo[d]->val_t)(pos(0) + 1, pos(1) + 1, l)(d) * (*p_christo[d]->val_t)(l, k, k)(d);
            }
            res_val.set(pos).set_domain(d) = cmpval;

            if (m_doder) { // variation
                cmpder = 0.0;
                for (int k(1); k <= dim; ++k) {
                    cmpder += m_ads.m_eps(d) * ((*p_christo[d]->der_t)(pos(0) + 1, pos(1) + 1, k)(d).der_abs(k) -
                                                (*p_christo[d]->der_t)(pos(1) + 1, k, k)(d).der_abs(pos(0) + 1)) -
                              (*p_christo[d]->der_t)(pos(0) + 1, pos(1) + 1, k)(d) * (*p_der_eps)(k)(d) +
                              (*p_christo[d]->der_t)(pos(1) + 1, k, k)(d) * (*p_der_eps)(pos(0) + 1)(d);
                    for (int l(1); l <= dim; ++l)
                        cmpder +=
                            (*p_christo[d]->der_t)(k, l, l)(d) * (*p_christo[d]->val_t)(pos(0) + 1, pos(1) + 1, k)(d) +
                            (*p_christo[d]->val_t)(k, l, l)(d) * (*p_christo[d]->der_t)(pos(0) + 1, pos(1) + 1, k)(d) -
                            (*p_christo[d]->der_t)(pos(0) + 1, l, k)(d) * (*p_christo[d]->val_t)(pos(1) + 1, k, l)(d) -
                            (*p_christo[d]->val_t)(pos(0) + 1, l, k)(d) * (*p_christo[d]->der_t)(pos(1) + 1, k, l)(d) +
                            (*m_ads.p_christo[d]->val_t)(k, l, k)(d) *
                                (*p_christo[d]->der_t)(pos(0) + 1, pos(1) + 1, l)(d) -
                            (*m_ads.p_christo[d]->val_t)(k, pos(1) + 1, l)(d) *
                                (*p_christo[d]->der_t)(pos(0) + 1, l, k)(d) -
                            (*m_ads.p_christo[d]->val_t)(pos(0) + 1, l, k)(d) *
                                (*p_christo[d]->der_t)(pos(1) + 1, k, l)(d) +
                            (*m_ads.p_christo[d]->val_t)(pos(0) + 1, pos(1) + 1, l)(d) *
                                (*p_christo[d]->der_t)(l, k, k)(d);
                }
                res_der.set(pos).set_domain(d) = cmpder;
            }
        } while (pos.inc());

        if (!m_doder)
            if (p_ricci_tensor[d] == nullptr)
                p_ricci_tensor[d] = std::make_unique<Term_eq>(d, res_val);
            else
                *p_ricci_tensor[d] = Term_eq(d, res_val);
        else if (p_ricci_tensor[d] == nullptr)
            p_ricci_tensor[d] = std::make_unique<Term_eq>(d, res_val, res_der);
        else
            *p_ricci_tensor[d] = Term_eq(d, res_val, res_der);
    }

    void Metric_AADS::compute_ricci_scalar(int d) const // O(1) in last domain
    {
        int dim(espace.get_ndim());
        if (p_met_cov[d] == nullptr)
            compute_cov(d);
        if (p_met_con[d] == nullptr)
            compute_con(d);
        if (p_ricci_tensor[d] == nullptr)
            compute_ricci_tensor(d);
        if (m_ads.p_ricci_tensor[d] == nullptr)
            m_ads.compute_ricci_tensor(d);
        Scalar res_val(espace);
        Scalar res_der(espace);
        Val_domain cmpval(espace.get_domain(d));
        Val_domain cmpder(espace.get_domain(d));
        cmpval = 0.0;
        cmpder = 0.0;
        for (int i(1); i <= dim; ++i)
            for (int j(1); j <= dim; ++j)
                cmpval += (*p_met_con[d]->val_t)(i, j)(d) * (*p_ricci_tensor[d]->val_t)(i, j)(d) +
                          (*p_k[d]->val_t)(i, j)(d) * (*m_ads.p_ricci_tensor[d]->val_t)(i, j)(d);
        res_val.set_domain(d) = cmpval;

        if (m_doder) {
            for (int i(1); i <= dim; ++i)
                for (int j(1); j <= dim; ++j)
                    cmpder += (*p_met_con[d]->val_t)(i, j)(d) * ((*p_ricci_tensor[d]->der_t)(i, j)(d)) +
                              (*p_met_con[d]->der_t)(i, j)(d) * (*p_ricci_tensor[d]->val_t)(i, j)(d) +
                              (*p_k[d]->der_t)(i, j)(d) * (*m_ads.p_ricci_tensor[d]->val_t)(i, j)(d);
            res_der.set_domain(d) = cmpder;
        }

        if (!m_doder) {
            if (p_ricci_scalar[d] == nullptr)
                p_ricci_scalar[d] = std::make_unique<Term_eq>(d, res_val);
            else
                *p_ricci_scalar[d] = Term_eq(d, res_val);
        } else {
            if (p_ricci_scalar[d] == nullptr)
                p_ricci_scalar[d] = std::make_unique<Term_eq>(d, res_val, res_der);
            else
                *p_ricci_scalar[d] = Term_eq(d, res_val, res_der);
        }
    }

    void Metric_AADS::set_system(System_of_eqs& ss, const char* name_met, const char* name_hmet)
    {
        syst = &ss;
        m_place_syst = ss.ndom * ss.nvar; // position in the system
        ss.add_var(name_hmet, *p_hmet);   // unknown for the system (no name, the name is in the metric already)
        if (ss.met != nullptr) {
            KADATH_THROW("Metric already set for the system");
        }
        ss.attach_metric(*this, name_met);
    }

    void Metric_AADS::set_system(System_of_eqs& ss, const char* name_met, const char* name_hmet,
                                 const char* name_back_cov, const char* name_back_con, const char* name_back_ricci)
    {
        syst = &ss;
        m_ads.init_system(ss, name_back_cov, name_back_con,
                          name_back_ricci); // put background metric as a cst in the system
        m_place_syst = ss.ndom * ss.nvar;   // position in the system
        ss.add_var(name_hmet, *p_hmet);     // unknown for the system (no name, the name is in the metric already)
        if (ss.met != nullptr) {
            KADATH_THROW("Metric already set for the system");
        }
        ss.attach_metric(*this, name_met);
    }

    void Metric_AADS::set_system(System_of_eqs& ss, const char* name_met, const char* name_hmet,
                                 const char* name_back_cov, const char* name_back_con, const char* name_back_gam,
                                 const char* name_back_ricci)
    {
        syst = &ss;
        m_ads.init_system(ss, name_back_cov, name_back_con, name_back_gam,
                          name_back_ricci); // put background metric as a cst in the system
        m_place_syst = ss.ndom * ss.nvar;   // position in the system
        ss.add_var(name_hmet, *p_hmet);     // unknown for the system (no name, the name is in the metric already)
        if (ss.met != nullptr) {
            KADATH_THROW("Metric already set for the system");
        }
        ss.attach_metric(*this, name_met);
    }

    const Metric* Metric_AADS::get_background() const
    {
        return &m_ads;
    }
} // namespace Kadath
