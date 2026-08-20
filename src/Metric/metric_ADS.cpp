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
 */

#include "For_Kadath/Metric/metric_AADS.hpp"
#include "For_Kadath/Array/exceptions.hpp"
namespace Kadath
{
    Metric_ADS::Metric_ADS(const Space& space) : Metric(space), m_rho2(space), m_eps(space)
    {
        type_tensor = COV; // assume covariance
        m_nd = espace.get_nbr_domains();
        m_L =
            espace.get_domain(m_nd - 1)->get_rmax(); // radius at the last collocation point of the last physical domain
        for (int d(0); d < m_nd; ++d) {
            m_rho2.set_domain(d) = pow(espace.get_domain(d)->get_radius() / m_L, 2);
            m_eps.set_domain(d) = (1.0 - m_rho2(d)) / (1.0 + m_rho2(d)); // conformal factor
        }
        p_basis = new Base_tensor(space, CARTESIAN_BASIS);
    }

    Metric_ADS::Metric_ADS(const Metric_ADS& so)
        : Metric(so), m_nd(so.m_nd), m_L(so.m_L), m_rho2(so.m_rho2), m_eps(so.m_eps)
    {
        p_basis = new Base_tensor(*so.p_basis);
    }

    Metric_ADS::~Metric_ADS()
    {
        delete p_basis;
    }

    void Metric_ADS::update(int d)
    {
        assert(type_tensor == COV);
        if (p_met_cov[d] != nullptr)
            compute_cov(d);
        if (p_met_con[d] != nullptr)
            compute_con(d);
        if (p_christo[d] != nullptr)
            compute_christo(d);
        if (p_riemann[d] != nullptr)
            compute_riemann(d);
        if (p_ricci_tensor[d] != nullptr)
            compute_ricci_tensor(d);
        if (p_ricci_scalar[d] != nullptr)
            compute_ricci_scalar(d);
    }

    void Metric_ADS::update()
    {
        for (int d(0); d < m_nd; ++d)
            update(d);
    }

    void Metric_ADS::compute_cov(int d) const
    {
        // epsilon^2*gama_ij is stored
        int dim(espace.get_ndim());    // dimension
        if (p_met_cov[d] == nullptr) { // if covariant metric not computed yet
            Metric_tensor res(espace, COV, *p_basis);
            for (int i(1); i <= dim; ++i) // build metric
            {
                for (int j(i); j <= dim; ++j) {
                    if (i == j)
                        res.set(i, j).set_domain(d) = 4.0 / pow(1.0 + m_rho2(d), 2);
                    else
                        res.set(i, j).set_domain(d) = 0.0;
                }
            }
            p_met_cov[d] = std::make_unique<Term_eq>(d, res); // assign metric
            p_met_cov[d]->set_der_zero();       // don't compute variations
        }
    }

    void Metric_ADS::compute_con(int d) const // compute contravariant metric in domain d
    {                                         // gama^ij / eps^2 is stored
        // Standard value everywhere
        int dim(espace.get_ndim()); // dimension
        if (p_met_con[d] == nullptr) {
            Metric_tensor res(espace, CON, *p_basis);
            for (int i(1); i <= dim; ++i) {
                for (int j(i); j <= dim; ++j) {
                    if (i == j)
                        res.set(i, j).set_domain(d) = 0.25 * pow(1.0 + m_rho2(d), 2);
                    else
                        res.set(i, j).set_domain(d) = 0.0;
                }
            }
            p_met_con[d] = std::make_unique<Term_eq>(d, res); // assign
            p_met_con[d]->set_der_zero();
        }
    }

    void Metric_ADS::manipulate_ind(Term_eq& so,
                                    int ind) const // rise or lower indice  ind on term_eq via the metric (in place)
    {
        int d(so.get_dom());                                // domain
        int dim(espace.get_ndim());                         // dimension
        int valence(so.get_val_t().get_valence());          // number of indices
        int type_start(so.get_val_t().get_index_type(ind)); // Contravariant or covariant
        if (p_met_con[d] == nullptr and type_start == COV)
            compute_con(d); // make sure that metric is computed
        if (p_met_cov[d] == nullptr and type_start == CON)
            compute_cov(d);

        bool doder(so.der_t != nullptr); // compute variations only if tensor need to
        Array<int> type_res(valence);    // CON or COV
        for (int i(0); i < valence; ++i)
            type_res.set(i) = (i == ind) ? -so.get_val_t().get_index_type(i)
                                         : so.get_val_t().get_index_type(i); // change the COV/CON of ind

        Tensor val_res(espace, valence, type_res, *p_basis); // tensor
        Tensor val_der(espace, valence, type_res, *p_basis); // its variation

        Val_domain cmpval(espace.get_domain(d)); // component value
        Val_domain cmpder(espace.get_domain(d)); // if variation of tensor need be computed
        Index pos(val_res);
        do { // for all indices
            cmpval = 0.0;
            cmpder = 0.0;
            Index copie(pos);
            for (int k(0); k < dim; ++k) {
                copie.set(ind) = k; // loop on the indice to rise/lower
                if (type_start == COV)
                    cmpval += (*p_met_con[d]->val_t)(pos(ind) + 1, k + 1)(d) * (*so.val_t)(copie)(d); // g^ik T_jk
                else
                    cmpval += (*p_met_cov[d]->val_t)(pos(ind) + 1, k + 1)(d) * (*so.val_t)(copie)(d); // g_ik T^k_j
            }
            val_res.set(pos).set_domain(d) = cmpval; // assign cmpval to right component

            if (doder) {
                for (int k(0); k < dim; ++k) {
                    copie.set(ind) = k;
                    if (type_start == COV)
                        cmpder += (*p_met_con[d]->val_t)(pos(ind) + 1, k + 1)(d) *
                                  (*so.der_t)(copie)(d); // recall that variations of background metric are zero
                    else
                        cmpder += (*p_met_cov[d]->val_t)(pos(ind) + 1, k + 1)(d) * (*so.der_t)(copie)(d);
                }
                val_der.set(pos).set_domain(d) = cmpder; // assign cmpder to the component of the variation
            }
        } while (pos.inc());

        // Put the names :
        if (so.get_val_t().is_name_affected()) {
            val_res.set_name_affected();
            for (int i(0); i < valence; ++i)
                val_res.set_name_ind(i, so.val_t->get_name_ind()[i]); // result have same indices names as source
        }

        if ((doder) and (so.get_der_t().is_name_affected())) {
            val_der.set_name_affected();
            for (int i(0); i < valence; ++i)
                val_der.set_name_ind(i, so.der_t->get_name_ind()[i]);
        }

        delete so.val_t;
        so.val_t = new Tensor(val_res); // in place substitution
        delete so.der_t;
        so.der_t = doder ? new Tensor(val_der) : nullptr;

        // multiplication/division by eps^2
        if (type_start == CON)
            so = div_eps(so, 2);
        else if (type_start == COV)
            so = mul_eps(so, 2);
    }

    void Metric_ADS::compute_christo(int d) const // compute Christoffel symbols in domain d
    {                                             // eps*Gam_ij^k is stored
        if (p_christo[d] == nullptr) {
            Array<int> type_ind(3);
            type_ind.set(0) = COV;
            type_ind.set(1) = COV;
            type_ind.set(2) = CON;                         // position of indices
            Tensor res_val(espace, 3, type_ind, *p_basis); // template of christoffel
            res_val = 0.0;

            Val_domain cmpval(espace.get_domain(d));
            cmpval = 2.0 * espace.get_domain(d)->get_cart(1) / m_L / m_L / (1.0 + m_rho2(d));
            res_val.set(1, 1, 1).set_domain(d) = cmpval;
            res_val.set(1, 2, 2).set_domain(d) = cmpval;
            res_val.set(2, 1, 2).set_domain(d) = cmpval;
            res_val.set(1, 3, 3).set_domain(d) = cmpval;
            res_val.set(3, 1, 3).set_domain(d) = cmpval;
            res_val.set(2, 2, 1).set_domain(d) = -cmpval;
            res_val.set(3, 3, 1).set_domain(d) = -cmpval;

            cmpval = 2.0 * espace.get_domain(d)->get_cart(2) / m_L / m_L / (1.0 + m_rho2(d));
            res_val.set(2, 2, 2).set_domain(d) = cmpval;
            res_val.set(2, 1, 1).set_domain(d) = cmpval;
            res_val.set(1, 2, 1).set_domain(d) = cmpval;
            res_val.set(2, 3, 3).set_domain(d) = cmpval;
            res_val.set(3, 2, 3).set_domain(d) = cmpval;
            res_val.set(1, 1, 2).set_domain(d) = -cmpval;
            res_val.set(3, 3, 2).set_domain(d) = -cmpval;

            cmpval = 2.0 * espace.get_domain(d)->get_cart(3) / m_L / m_L / (1.0 + m_rho2(d));
            res_val.set(3, 3, 3).set_domain(d) = cmpval;
            res_val.set(3, 1, 1).set_domain(d) = cmpval;
            res_val.set(1, 3, 1).set_domain(d) = cmpval;
            res_val.set(3, 2, 2).set_domain(d) = cmpval;
            res_val.set(2, 3, 2).set_domain(d) = cmpval;
            res_val.set(1, 1, 3).set_domain(d) = -cmpval;
            res_val.set(2, 2, 3).set_domain(d) = -cmpval;

            p_christo[d] = std::make_unique<Term_eq>(d, res_val);
            p_christo[d]->set_der_zero(); // don't compute the variations
        }
    }

    Term_eq Metric_ADS::derive_simple(const Term_eq& so)
        const // covariant derivative which does not affect indices and assume derivative indice is covariant
    {
        assert(espace.get_ndim() == 3);
        bool doder(so.der_t != nullptr);
        int d(so.get_dom()); // domain
        int so_val(so.val_t->get_valence());
        Term_eq res(espace.get_domain(d)->partial_cart(so)); // The partial derivative part

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
                                (*p_christo[d]->val_t)(poschristo)(d) *
                                (*so.der_t)(posso)(d); // no variation of christo
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
                                (*p_christo[d]->val_t)(poschristo)(d) *
                                (*so.der_t)(posso)(d); // no variation of christo
                    }
                }
            } while (posgamres.inc());
        }

        // division by eps
        if (so_val >= 1)
            return res + div_eps(gam_res, 1);
        return res;
    }

    Term_eq Metric_ADS::derive(int type_der, char ind_der, const Term_eq& so)
        const // covariant derivative, can only be applied to quantities going at least as O(epsilon)
    {
        int d(so.get_dom()); // domain
        int so_val(so.val_t->get_valence());
        bool doder(so.der_t != nullptr);
        if (p_christo[d] == nullptr)
            compute_christo(d);         // of course you need Christoffel
        Term_eq res(derive_simple(so)); // The simple derivative part (no indice names affected, no inner summation)
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

    void Metric_ADS::compute_riemann(int d) const // eps^2*Riemann is stored
    {
        if (p_riemann[d] == nullptr) {
            if (p_met_cov[d] == nullptr)
                compute_cov(d); // Maximally symmetric need the metric (COV version)
            Array<int> indices(4);
            indices.set(0) = CON;
            indices.set(1) = COV;
            indices.set(2) = COV;
            indices.set(3) = COV;
            Tensor res_val(espace, 4, indices, *p_basis);
            Index pos(res_val);
            Val_domain cmpval(espace.get_domain(d));
            do {
                cmpval = 0.0;
                if (pos(0) == pos(2))
                    cmpval += (*p_met_cov[d]->val_t)(pos(1) + 1, pos(3) + 1)(d);
                if (pos(0) == pos(3))
                    cmpval -= (*p_met_cov[d]->val_t)(pos(1) + 1, pos(2) + 1)(d);
                res_val.set(pos).set_domain(d) = -cmpval / (m_L * m_L);
            } while (pos.inc());
            p_riemann[d] = std::make_unique<Term_eq>(d, res_val);
            p_riemann[d]->set_der_zero();
        }
    }

    void Metric_ADS::compute_ricci_tensor(int d) const // In the last domain epsilon^2 Ricci is stored
    {
        if (p_ricci_tensor[d] == nullptr) {
            if (p_met_cov[d] == nullptr)
                compute_cov(d); // Maximally symmetric need the metric (COV version)
            p_ricci_tensor[d] = std::make_unique<Term_eq>(d, -2.0 * (*p_met_cov[d]->val_t) / (m_L * m_L));
            p_ricci_tensor[d]->set_der_zero();
        }
    }

    void Metric_ADS::compute_ricci_scalar(int d) const
    {
        if (p_ricci_scalar[d] == nullptr) {
            Scalar val(espace);
            val = -6.0 / (m_L * m_L);
            p_ricci_scalar[d] = std::make_unique<Term_eq>(d, val);
            p_ricci_scalar[d]->set_der_zero();
        }
    }

    void Metric_ADS::set_system(System_of_eqs& ss, const char* name)
    {
        syst = &ss;
        if (syst->met != nullptr) {
            KADATH_THROW("Metric already set for the system");
        }
        ss.attach_metric(*this, name);
    }

    void Metric_ADS::init_system(System_of_eqs& ss, const char* name_back_cov, const char* name_back_con,
                                 const char* name_back_ricci)
    {
        syst = &ss;                                   // Not to be used alone
        for (int d(ss.dom_min); d <= ss.dom_max; ++d) // make sure that metric is computed
        {
            if (p_met_cov[d] == nullptr)
                compute_cov(d);
            if (p_met_con[d] == nullptr)
                compute_con(d);
            if (p_ricci_tensor[d] == nullptr)
                compute_ricci_tensor(d);
        }

        int dim(espace.get_ndim());
        Metric_tensor gamaBcov(espace, COV, *p_basis);
        Metric_tensor gamaBcon(espace, CON, *p_basis);
        gamaBcov = 0.0;
        gamaBcon = 0.0;
        for (int i(1); i <= dim; ++i) {
            for (int j(i); j <= dim; ++j) {
                for (int d(ss.dom_min); d <= ss.dom_max; ++d) // give tensors from Term_eq values
                {
                    gamaBcov.set(i, j).set_domain(d) = (*p_met_cov[d]->val_t)(i, j)(d);
                    gamaBcon.set(i, j).set_domain(d) = (*p_met_con[d]->val_t)(i, j)(d);
                }
            }
        }

        Tensor RicciB(espace, 2, COV, *p_basis); // background Ricci
        RicciB = 0.0;
        for (int i(1); i <= dim; ++i)
            for (int j(1); j <= dim; ++j)
                for (int d(ss.dom_min); d <= ss.dom_max; ++d) // give tensors from Term_eq values
                    RicciB.set(i, j).set_domain(d) = (*p_ricci_tensor[d]->val_t)(i, j)(d);

        ss.add_cst(name_back_cov, gamaBcov);
        ss.add_cst(name_back_con, gamaBcon);
        ss.add_cst(name_back_ricci, RicciB);
    }

    void Metric_ADS::init_system(System_of_eqs& ss, const char* name_back_cov, const char* name_back_con,
                                 const char* name_back_gam, const char* name_back_ricci)
    {
        syst = &ss;                                   // Not to be used alone
        for (int d(ss.dom_min); d <= ss.dom_max; ++d) // make sure that metric is computed
        {
            if (p_met_cov[d] == nullptr)
                compute_cov(d);
            if (p_met_con[d] == nullptr)
                compute_con(d);
            if (p_christo[d] == nullptr)
                compute_christo(d);
            if (p_ricci_tensor[d] == nullptr)
                compute_ricci_tensor(d);
        }

        int dim(espace.get_ndim());
        Metric_tensor gamaBcov(espace, COV, *p_basis);
        Metric_tensor gamaBcon(espace, CON, *p_basis);
        gamaBcov = 0.0;
        gamaBcon = 0.0;
        for (int i(1); i <= dim; ++i)
            for (int j(i); j <= dim; ++j)
                for (int d(ss.dom_min); d <= ss.dom_max; ++d) // give tensors from Term_eq values
                {
                    gamaBcov.set(i, j).set_domain(d) = (*p_met_cov[d]->val_t)(i, j)(d);
                    gamaBcon.set(i, j).set_domain(d) = (*p_met_con[d]->val_t)(i, j)(d);
                }

        Array<int> ind_type(3);
        ind_type.set(0) = COV;
        ind_type.set(1) = COV;
        ind_type.set(2) = CON;
        Tensor ChristoB(espace, 3, ind_type, *p_basis);
        ChristoB = 0.0;
        for (int i(1); i <= dim; ++i)
            for (int j(1); j <= dim; ++j)
                for (int k(1); k <= dim; ++k)
                    for (int d(ss.dom_min); d <= ss.dom_max; ++d) // give tensors from Term_eq values
                        ChristoB.set(i, j, k).set_domain(d) = (*p_christo[d]->val_t)(i, j, k)(d);

        Tensor RicciB(espace, 2, COV, *p_basis); // background Ricci
        RicciB = 0.0;
        for (int i(1); i <= dim; ++i)
            for (int j(1); j <= dim; ++j)
                for (int d(ss.dom_min); d <= ss.dom_max; ++d) // give tensors from Term_eq values
                    RicciB.set(i, j).set_domain(d) = (*p_ricci_tensor[d]->val_t)(i, j)(d);

        ss.add_cst(name_back_cov, gamaBcov);
        ss.add_cst(name_back_con, gamaBcon);
        ss.add_cst(name_back_gam, ChristoB);
        ss.add_cst(name_back_ricci, RicciB);
    }

    Term_eq Metric_ADS::div_eps(Term_eq& so, int n) const
    {
        int d(so.get_dom());
        bool doder(so.der_t != nullptr);
        Term_eq res(so);
        for (int i(1); i <= n; ++i) {
            Index pos(*res.val_t);
            do {
                (*res.val_t).set(pos).set_domain(d) *= (1.0 + m_rho2(d)); // multiplication by 1 + rho^2
                if (doder)
                    (*res.der_t).set(pos).set_domain(d) *= (1.0 + m_rho2(d));
                if (m_nd != 1) {
                    if (d < m_nd - 1) {
                        (*res.val_t).set(pos).set_domain(d) /= (1.0 - m_rho2(d));
                        if (doder)
                            (*res.der_t).set(pos).set_domain(d) /= (1.0 - m_rho2(d));
                    }
                    if (d == m_nd - 1) {
                        Val_domain us1prsL(1.0 / (1.0 + espace.get_domain(d)->get_radius() / m_L));
                        (*res.val_t).set(pos).set_domain(d) =
                            espace.get_domain(d)->div_1mrsL((*res.val_t)(pos)(d)) * us1prsL;
                        if (doder)
                            (*res.der_t).set(pos).set_domain(d) =
                                espace.get_domain(d)->div_1mrsL((*res.der_t)(pos)(d)) * us1prsL;
                    }
                }
            } while (pos.inc());
            if (m_nd == 1)
                res = div_1mx2(res); // division by 1 - rho^2
        }
        return res;
    }

    Term_eq Metric_ADS::mul_eps(Term_eq& so, int n) const
    {
        int d(so.get_dom());
        bool doder(so.der_t != nullptr);
        Term_eq res(so);
        for (int i(1); i <= n; ++i) {
            Index pos(*res.val_t);
            do {
                (*res.val_t).set(pos).set_domain(d) *= (1.0 - m_rho2(d)) / (1.0 + m_rho2(d));
                if (doder)
                    (*res.der_t).set(pos).set_domain(d) *= (1.0 - m_rho2(d)) / (1.0 + m_rho2(d));
            } while (pos.inc());
        }
        return res;
    }
} // namespace Kadath
