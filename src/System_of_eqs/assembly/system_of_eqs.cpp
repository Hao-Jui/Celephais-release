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

#include "For_Kadath/Space/space.hpp"
#include "For_Kadath/Array/exceptions.hpp"

#include <memory>
#include <sstream>
#include "For_Kadath/System_of_eqs/system_of_eqs.hpp"
#include "For_Kadath/Utilities/name_tools.hpp"
#include "For_Kadath/Scalar/scalar.hpp"
#include <string_view>

namespace Kadath
{
namespace {
// Whitespace-normalize a name into an owned std::string. Same algorithm as
// trim_spaces() (collapse runs of spaces, append a single trailing space) but:
//   - input is std::string_view (length-bounded, may be non-NUL-terminated)
//   - output is std::string sized to input — no LMAX cap
//   - empty/whitespace-only input yields empty output
std::string normalize_name(std::string_view in)
{
    std::string out;
    out.reserve(in.size() + 1);
    bool inspace = !in.empty() && in.front() == ' ';
    bool wrote = false;
    for (char c : in) {
        const bool isspace = (c == ' ');
        if (!isspace || !inspace) {
            out.push_back(c);
            wrote = true;
        }
        inspace = isspace;
    }
    if (wrote && out.back() != ' ') out.push_back(' ');
    return out;
}

bool reusable_snapshot_tensor(const Tensor& target, const Tensor& source)
{
    if (&target.get_space() != &source.get_space() ||
        target.get_valence() != source.get_valence() ||
        target.get_n_comp() != source.get_n_comp() ||
        target.get_ndim() != source.get_ndim() ||
        target.is_name_affected() != source.is_name_affected() ||
        target.get_parameters() != nullptr || source.get_parameters() != nullptr)
        return false;

    for (int index = 0; index < target.get_valence(); ++index) {
        if (target.get_index_type(index) != source.get_index_type(index))
            return false;
        if (target.is_name_affected() &&
            target.get_name_ind()[index] != source.get_name_ind()[index])
            return false;
    }
    return true;
}

void copy_snapshot_tensor_domains(Tensor& target, const Tensor& source)
{
    const int domains = source.get_space().get_nbr_domains();
    for (int domain = 0; domain < domains; ++domain) {
        affecte_one_dom(domain, &target, &source);
        target.set_basis(domain) = source.get_basis().get_basis(domain);
    }
}

void copy_snapshot_tensor_domain(int domain, Tensor& target, const Tensor& source)
{
    affecte_one_dom(domain, &target, &source);
    target.set_basis(domain) = source.get_basis().get_basis(domain);
}

// Resolution-keyed MUMPS ICNTL(14) seed. The analysis estimate that ICNTL(14)
// must pad degrades with spectral resolution (res13 floor <= 100, res17
// FORCE_BALANCE needed 1125), while a flat high seed over-reserves factor RAM
// at scale. Keyed on the largest per-direction point count over the domains.
int mumps_icntl14_seed_for_space(const Space& sp)
{
    int max_points = 0;
    for (int d = 0; d < sp.get_nbr_domains(); ++d) {
        const Dim_array points = sp.get_domain(d)->get_nbr_points();
        for (int i = 0; i < points.get_ndim(); ++i) {
            max_points = std::max(max_points, points(i));
        }
    }
    if (max_points <= 7) return 50;
    if (max_points <= 9) return 100;
    if (max_points <= 11) return 200;
    if (max_points <= 13) return 400;
    if (max_points <= 15) return 800;
    if (max_points <= 17) return 1600;
    return 2000;
}
}  // namespace
    // Constructor
    System_of_eqs::System_of_eqs(const Space& sp)
        : espace(sp), dom_min(0), dom_max(espace.get_nbr_domains() - 1), ndom(dom_max - dom_min + 1), nvar_double(0),
          nvar(0), nterm_double(0), nterm(0), ncst(0), nterm_cst(0), ncst_hard(0), ndef(0), ndef_glob(0),
          nopeuser(0), nopeuser_bin(0), met(nullptr), neq_int(0), neq(0), nbr_unknowns(0), nbr_conditions(-1),
          jac_col_engine_(*this)
    {
        nbr_unknowns = sp.nbr_unknowns_from_variable_domains();
        mumps_icntl14_seed = mumps_icntl14_seed_for_space(sp);
        mumps_runtime_state.icntl14 = mumps_icntl14_seed;
    }

    // same between two bounds
    System_of_eqs::System_of_eqs(const Space& sp, int dmin, int dmax)
        : espace(sp), dom_min(dmin), dom_max(dmax), ndom(dom_max - dom_min + 1), nvar_double(0), nvar(0),
          nterm_double(0), nterm(0), ncst(0), nterm_cst(0), ncst_hard(0), ndef(0), ndef_glob(0),
          nopeuser(0), nopeuser_bin(0), met(nullptr), neq_int(0), neq(0), nbr_unknowns(0), nbr_conditions(-1),
          jac_col_engine_(*this)
    {
        nbr_unknowns = sp.nbr_unknowns_from_variable_domains();
        mumps_icntl14_seed = mumps_icntl14_seed_for_space(sp);
        mumps_runtime_state.icntl14 = mumps_icntl14_seed;
    }

    // In only one domain
    System_of_eqs::System_of_eqs(const Space& sp, int dd)
        : espace(sp), dom_min(dd), dom_max(dd), ndom(1), nvar_double(0), nvar(0), nterm_double(0), nterm(0),
          ncst(0), nterm_cst(0), ncst_hard(0), ndef(0), ndef_glob(0), nopeuser(0), nopeuser_bin(0),
          met(nullptr), neq_int(0), neq(0), nbr_unknowns(0), nbr_conditions(-1),
          jac_col_engine_(*this)
    {
        nbr_unknowns = sp.nbr_unknowns_from_variable_domains();
        mumps_icntl14_seed = mumps_icntl14_seed_for_space(sp);
        mumps_runtime_state.icntl14 = mumps_icntl14_seed;
    }

    System_of_eqs::System_of_eqs(const System_of_eqs& so) : espace(so.espace), jac_col_engine_(*this)
    {
        KADATH_THROW("Copy constructor not implemented for System_of_eqs");
    }

    System_of_eqs::~System_of_eqs()
    {
        // Operand scratch can contain Tensors that reference this system's
        // Space. Release them before the external Space may leave scope.
        ope_action_detail::release_operand_scratch(this);
    }

    void System_of_eqs::attach_metric(Metric& metric, const char* name)
    {
        if (met != nullptr) {
            KADATH_THROW("Metric already set for the system");
        }

        char trimmed[LMAX];
        trim_spaces(trimmed, name);
        met = &metric;
        name_met = trimmed;
    }

    const Metric* System_of_eqs::get_met() const
    {
        if (met == nullptr) {
            KADATH_THROW("Undefined metric in System_of_eqs");
        }
        return met;
    }

    Term_eq* System_of_eqs::give_term_double(int which, int dd) const
    {

        assert((which >= 0) && (which < nvar_double));
        assert((dd >= dom_min) && (dd <= dom_max));

        int pos = which * ndom + (dd - dom_min);
        return term_double[pos].get();
    }

    Term_eq* System_of_eqs::give_term(int which, int dd) const
    {

        assert((which >= 0) && (which < nvar));
        assert((dd >= dom_min) && (dd <= dom_max));

        int pos = which * ndom + (dd - dom_min);
        return term[pos].get();
    }

    Term_eq* System_of_eqs::give_cst(int which, int dd) const
    {

        assert((which >= 0) && (which < ncst));
        assert((dd >= dom_min) && (dd <= dom_max));
        int pos = which * ndom + (dd - dom_min);
        return cst[pos].get();
    }

    Term_eq* System_of_eqs::give_cst_hard(double xx, int dd) const
    {
        // Check if it has already been stored :
        for (int i = 0; i < ncst_hard; i++)
            if ((fabs(val_cst_hard[i] - xx) < PRECISION) && (cst_hard[i]->get_dom() == dd))
                return cst_hard[i].get();

        // Term not found, one needs to put it
        cst_hard.emplace_back(std::make_unique<Term_eq>(dd, xx));
        cst_hard.back()->set_der_zero();
        val_cst_hard.push_back(xx);
        ncst_hard++;
        return cst_hard.back().get();
    }

    Ope_def* System_of_eqs::give_def(int which) const
    {

        assert((which >= 0) && (which < ndef));
        return def[which].get();
    }

    Ope_def_global* System_of_eqs::give_def_glob(int which) const
    {

        assert((which >= 0) && (which < ndef_glob));
        return def_glob[which].get();
    }

    bool System_of_eqs::term_to_var_name(const Term_eq* term_in, std::string& name) const
    {
        if (term_in == nullptr)
            return false;

        for (int vv = 0; vv < nvar_double; vv++) {
            for (int dd = dom_min; dd <= dom_max; dd++) {
                int pos = vv * ndom + (dd - dom_min);
                if (term_double[pos].get() == term_in) {
                    name = names_var_double[vv];
                    return true;
                }
            }
        }

        for (int vv = 0; vv < nvar; vv++) {
            for (int dd = dom_min; dd <= dom_max; dd++) {
                int pos = vv * ndom + (dd - dom_min);
                if (term[pos].get() == term_in) {
                    name = names_var[vv];
                    return true;
                }
            }
        }

        return false;
    }

    Ope_def* System_of_eqs::def_from_term(const Term_eq* term_in) const
    {
        if (term_in == nullptr)
            return nullptr;
        for (int i = 0; i < ndef; i++) {
            if (def[i] != nullptr && def[i]->get_res() == term_in)
                return def[i].get();
        }
        return nullptr;
    }

    Ope_def_global* System_of_eqs::def_glob_from_term(const Term_eq* term_in) const
    {
        if (term_in == nullptr)
            return nullptr;
        for (int i = 0; i < ndef_glob; i++) {
            if (def_glob[i] != nullptr && def_glob[i]->get_res() == term_in)
                return def_glob[i].get();
        }
        return nullptr;
    }

    void System_of_eqs::vars_to_terms()
    {

        for (int d = dom_min; d <= dom_max; d++)
            espace.get_domain(d)->vars_to_terms();

        for (int vv = 0; vv < nvar_double; vv++)
            for (int dd = dom_min; dd <= dom_max; dd++)
                give_term_double(vv, dd)->set_val_d(*var_double[vv]);

        for (int vv = 0; vv < nvar; vv++)
            for (int dd = dom_min; dd <= dom_max; dd++)
                give_term(vv, dd)->set_val_t(*var[vv]);
    }

    // Add an uknown (numeric)
    void System_of_eqs::add_var(std::string_view name, double& vv)
    {
        if (name.find('_') != std::string_view::npos || name.find('^') != std::string_view::npos) {
            KADATH_THROW("No indices allowed in names of variables");
        }

        var_double.push_back(&vv);
        names_var_double.push_back(normalize_name(name));

        for (int dd = dom_min; dd <= dom_max; dd++) {
            term_double.push_back(std::make_unique<Term_eq>(dd, vv));
            assoc_var_double.push_back(nvar_double);
            nterm_double++;
        }
        nvar_double++;

        nbr_unknowns++;
    }

    // Add an uknown (tensor / field)
    void System_of_eqs::add_var(std::string_view name, Tensor& tt)
    {
        if (name.find('_') != std::string_view::npos || name.find('^') != std::string_view::npos) {
            KADATH_THROW("No indices allowed in names of variables");
        }
        names_var.push_back(normalize_name(name));

        assert(&tt.espace == &espace);
        var.push_back(&tt);
        for (int dd = dom_min; dd <= dom_max; dd++) {
            term.push_back(std::make_unique<Term_eq>(dd, tt));
            assoc_var.push_back(nvar);
            nterm++;
        }

        nvar++;

        for (int dd = dom_min; dd <= dom_max; dd++)
            nbr_unknowns += espace.get_domain(dd)->nbr_unknowns(tt, dd);
    }

    // Anonymous tensor variable: stored under the legacy "$" placeholder.
    void System_of_eqs::add_var(std::nullptr_t, Tensor& tt)
    {
        add_var(std::string_view{"$"}, tt);
    }

    // Add a constant (tensor / field)
    void System_of_eqs::add_cst(std::string_view name, const Tensor& so)
    {
        if (name.find('_') != std::string_view::npos || name.find('^') != std::string_view::npos) {
            KADATH_THROW("No indices allowed in names of constants");
        }
        names_cst.push_back(normalize_name(name));
        ncst++;

        for (int dd = dom_min; dd <= dom_max; dd++) {
            cst.push_back(std::make_unique<Term_eq>(dd, so));
            cst.back()->set_der_zero();
            nterm_cst++;
        }
    }

    // Anonymous tensor constant: stored under the legacy "$" placeholder.
    void System_of_eqs::add_cst(std::nullptr_t, const Tensor& so)
    {
        add_cst(std::string_view{"$"}, so);
    }

    // Add a constant (numeric)
    void System_of_eqs::add_cst(std::string_view name, double xx)
    {
        names_cst.push_back(normalize_name(name));
        ncst++;

        for (int dd = dom_min; dd <= dom_max; dd++) {
            cst.push_back(std::make_unique<Term_eq>(dd, xx));
            cst.back()->set_der_zero();
            nterm_cst++;
        }
    }

    void System_of_eqs::add_ope(std::string_view name, Term_eq (*pope)(const Term_eq&, Param*), Param* par)
    {
        names_opeuser.push_back(normalize_name(name));
        opeuser.push_back(pope);
        paruser.push_back(par);
        nopeuser++;
    }

    void System_of_eqs::add_ope(std::string_view name, Term_eq (*pope)(const Term_eq&, const Term_eq&, Param*), Param* par)
    {
        names_opeuser_bin.push_back(normalize_name(name));
        opeuser_bin.push_back(pope);
        paruser_bin.push_back(par);
        nopeuser_bin++;
    }

    // Add a definition. Definition body parsing still uses LMAX-bounded char buffers
    // (is_ope_bin / get_util / is_tensor are out of scope for this migration).
    void System_of_eqs::add_def(int dom, std::string_view name)
    {
        // Materialize a NUL-terminated owning copy for the legacy parsers.
        std::string nom{name};

        // Récupère le = :
        char p1[LMAX];
        char p2[LMAX];
        bool indic = is_ope_bin(nom.c_str(), p1, p2, '=');
        if (!indic) {
            KADATH_THROW("= needed for definitions");
        }

        // lhs is name of definition :
        char name_def[LMAX];
        get_util(name_def, p1);
        names_def.emplace_back(name_def);

        int valence;
        char* indices = nullptr;
        Array<int>* ttype = nullptr;
        [[maybe_unused]] bool auxi = is_tensor(p1, names_def.back().c_str(), valence, indices, ttype);
        assert(auxi);

        std::unique_ptr<char[]> owned_indices(indices);
        std::unique_ptr<Array<int>> owned_ttype(ttype);
        def.emplace_back(
            std::make_unique<Ope_def>(this, give_ope(dom, p2), valence, owned_indices.get(), owned_ttype.get()));
        ndef++;
    }

    void System_of_eqs::add_def(std::string_view name)
    {
        for (int dd = dom_min; dd <= dom_max; dd++)
            add_def(dd, name);
    }

    // Add a definition involving all the domains (i.e. integral).
    void System_of_eqs::add_def_global(int dom, std::string_view name)
    {
        std::string nom{name};

        // Récupère le = :
        char p1[LMAX];
        char p2[LMAX];
        bool indic = is_ope_bin(nom.c_str(), p1, p2, '=');
        if (!indic) {
            KADATH_THROW("= needed for definitions");
        }

        // lhs is name of definition :
        char name_def[LMAX];
        get_util(name_def, p1);
        names_def_glob.emplace_back(name_def);
        def_glob.emplace_back(std::make_unique<Ope_def_global>(this, dom, p2));
        ndef_glob++;
    }

    void System_of_eqs::add_def_global(std::string_view name)
    {
        for (int dd = dom_min; dd <= dom_max; dd++)
            add_def_global(dd, name);
    }

    void System_of_eqs::xx_to_ders(const Array<double>& xx, const std::vector<char>* term_filter)
    {

        assert(xx.get_ndim() == 1);
        assert(xx.get_size(0) == nbr_unknowns);
        assert(term_filter == nullptr ||
               static_cast<int>(term_filter->size()) == nterm);

        int conte = 0;
        int pos_term = 0;

        espace.xx_to_ders_variable_domains(xx, conte);

        for (int i = 0; i < nvar_double; i++) {
            for (int dd = dom_min; dd <= dom_max; dd++) {
                term_double[pos_term]->set_der_d(xx(conte));
                pos_term++;
            }
            conte++;
        }

        for (int i = 0; i < nterm; i++) {

            // Per-rank preamble closure (DO_JX_TERM_CLOSURE): skip the
            // tensor allocation + affecte_tau + set_der_t for terms whose
            // derivative no owned row of this rank reads, advancing conte by
            // the term's xx-block stride (= Domain::nbr_unknowns(var, dom),
            // the same count add_var used to size the block). The skipped
            // term keeps its previous derivative, which this rank never
            // reads.
            if (term_filter != nullptr && !(*term_filter)[i]) {
                conte += do_jx_mpi_.term_xx_stride[i];
                continue;
            }

            // Check for symetric tensor :
            const Metric_tensor* pmet = dynamic_cast<const Metric_tensor*>(term[i]->val_t);
            if (pmet == nullptr) {
                Tensor auxi(term[i]->get_val_t(), false);

                int dom = term[i]->get_dom();
                espace.get_domain(dom)->affecte_tau(auxi, dom, xx, conte);

                term[i]->set_der_t(auxi);
            } else {
                Metric_tensor auxi(*pmet, false);

                int dom = term[i]->get_dom();
                espace.get_domain(dom)->affecte_tau(auxi, dom, xx, conte);
                term[i]->set_der_t(auxi);
            }
        }
        // Stride-contract tripwire (debug builds): a filtered pass must land
        // exactly on nbr_unknowns, mixing real consumption (kept terms) with
        // the precomputed strides (skipped terms).
        assert(term_filter == nullptr || conte == nbr_unknowns);
    }

    void System_of_eqs::xx_to_vars(const Array<double>& xx, int& conte)
    {
        // Unknowns are about to change: any forwarded residual is stale
        // (mirrors xx_to_vars_delta / restore_state — every unknown mutator
        // must invalidate the single-shot forwarded-residual slot).
        forwarded_residual_.reset();

        assert(xx.get_ndim() == 1);
        assert(xx.get_size(0) == nbr_unknowns);

        for (int i = 0; i < nvar_double; i++) {
            *var_double[i] = xx(conte);
            conte++;
        }

        for (int i = 0; i < nvar; i++)
            for (int dom = dom_min; dom <= dom_max; dom++) {
                espace.get_domain(dom)->affecte_tau(*var[i], dom, xx, conte);
            }
    }

    void System_of_eqs::xx_to_vars_delta(const Array<double>& xx, int& conte)
    {
        // Unknowns are about to change: any forwarded residual is stale.
        forwarded_residual_.reset();

        assert(xx.get_ndim() == 1);
        assert(xx.get_size(0) == nbr_unknowns);

        for (int i = 0; i < nvar_double; i++) {
            *var_double[i] -= xx(conte);
            conte++;
        }

        for (int i = 0; i < nvar; i++) {
            // Only the Tensor layout is needed before affecte_tau populates the
            // correction. Avoid copying every live field/domain buffer into a
            // temporary that is immediately overwritten by annule_hard().
            Tensor delta(*var[i], false);
            if (var[i]->is_name_affected()) {
                delta.set_name_affected();
                for (int index = 0; index < var[i]->get_valence(); ++index)
                    delta.set_name_ind(index, var[i]->get_name_ind()[index]);
            }
            delta.annule_hard();
            for (int dom = dom_min; dom <= dom_max; dom++) {
                espace.get_domain(dom)->affecte_tau(delta, dom, xx, conte);
            }
            if (var[i]->get_valence() == 0 && var[i]->get_n_comp() == 1) {
                // The NS2d unknown fields are scalar-shaped. For one component,
                // Tensor::operator-= follows the same Scalar subtraction path as
                // operator- and commits via Scalar's move assignment, removing
                // operator-'s Tensor copy and the subsequent copy back.
                *var[i] -= delta;
            } else {
                // Preserve the legacy temporary-based exception behavior for a
                // general tensor: an allocation failure in a later component
                // must not leave earlier components updated in place.
                *var[i] = *var[i] - delta;
            }
        }
    }

    void System_of_eqs::snapshot_state_into(State_snapshot& snapshot) const
    {
        snapshot.scalars.resize(static_cast<std::size_t>(nvar_double));
        for (int i = 0; i < nvar_double; i++)
            snapshot.scalars[static_cast<std::size_t>(i)] = *var_double[i];

        bool reuse_fields = snapshot.fields.size() == static_cast<std::size_t>(nvar);
        for (int i = 0; reuse_fields && i < nvar; ++i)
            reuse_fields = reusable_snapshot_tensor(
                snapshot.fields[static_cast<std::size_t>(i)], *var[i]);
        if (!reuse_fields) {
            snapshot.fields.clear();
            snapshot.fields.reserve(static_cast<std::size_t>(nvar));
            for (int i = 0; i < nvar; ++i)
                snapshot.fields.emplace_back(*var[i]);
        } else {
            for (int i = 0; i < nvar; ++i)
                copy_snapshot_tensor_domains(
                    snapshot.fields[static_cast<std::size_t>(i)], *var[i]);
        }

        // Only TERM_T constants carry tensorial values that the moving surface
        // remaps (update_constante); numeric (TERM_D) constants are untouched.
        std::size_t cst_count = 0;
        for (int i = 0; i < nterm_cst; ++i)
            if (cst[i] && cst[i]->get_type_data() == TERM_T)
                ++cst_count;

        bool reuse_cst_terms = snapshot.cst_terms.size() == cst_count;
        std::size_t cst_snapshot_index = 0;
        for (int i = 0; reuse_cst_terms && i < nterm_cst; ++i) {
            if (!cst[i] || cst[i]->get_type_data() != TERM_T)
                continue;
            const auto& entry = snapshot.cst_terms[cst_snapshot_index++];
            reuse_cst_terms = entry.first == i &&
                reusable_snapshot_tensor(entry.second, cst[i]->get_val_t());
        }
        if (!reuse_cst_terms) {
            snapshot.cst_terms.clear();
            snapshot.cst_terms.reserve(cst_count);
            for (int i = 0; i < nterm_cst; ++i)
                if (cst[i] && cst[i]->get_type_data() == TERM_T)
                    snapshot.cst_terms.emplace_back(i, cst[i]->get_val_t());
        } else {
            for (auto& entry : snapshot.cst_terms) {
                const Term_eq& term = *cst[entry.first];
                // A TERM_T constant owns one sparse active-domain shell. Do
                // not materialize its deliberately absent domain storage.
                copy_snapshot_tensor_domain(term.get_dom(), entry.second,
                                            term.get_val_t());
            }
        }

        std::size_t mapping_index = 0;
        for (int dom = dom_min; dom <= dom_max; ++dom)
            espace.get_domain(dom)->snapshot_mapping_into(snapshot.radii, mapping_index);
        if (mapping_index < snapshot.radii.size())
            snapshot.radii.erase(snapshot.radii.begin() + mapping_index,
                                 snapshot.radii.end());
    }

    System_of_eqs::State_snapshot System_of_eqs::snapshot_state() const
    {
        State_snapshot snapshot;
        snapshot_state_into(snapshot);
        return snapshot;
    }

    void System_of_eqs::restore_state(const State_snapshot& snapshot)
    {
        // Unknowns are about to change: any forwarded residual is stale.
        forwarded_residual_.reset();
        for (int i = 0; i < nvar_double; i++)
            *var_double[i] = snapshot.scalars[i];

        for (int i = 0; i < nvar; i++) {
            const Tensor& field = snapshot.fields[static_cast<std::size_t>(i)];
            if (reusable_snapshot_tensor(*var[i], field))
                copy_snapshot_tensor_domains(*var[i], field);
            else
                *var[i] = field;
        }

        for (const auto& entry : snapshot.cst_terms) {
            Term_eq& term = *cst[entry.first];
            Tensor& value = *(term.set_val_t());
            if (reusable_snapshot_tensor(value, entry.second))
                copy_snapshot_tensor_domain(term.get_dom(), value, entry.second);
            else
                value = entry.second;
        }

        // Restore adapted-domain mappings last so the cache rebuild in
        // set_mapping observes the restored field/constant values.
        std::size_t idx = 0;
        for (int dom = dom_min; dom <= dom_max; dom++)
            espace.get_domain(dom)->restore_mapping(snapshot.radii, idx);
    }

    void System_of_eqs::ensure_def_values_current(const char* name) const
    {
        // Lazy rank-local repair after a partitioned value pass
        // (SEC_MEMBER_MPI): sec_member_partitioned_tail recomputes only
        // this rank's def closure, so defs outside it keep stale values until
        // the next full pass. Recompute the requested name's stale instances
        // plus their transitive chains here, ascending index first (a
        // referenced def always has a lower index than its referrer), so any
        // give_val_def reader sees current values. Pure local computation —
        // no collective, safe for rank-0-only callers. The term values these
        // defs read were refreshed by vars_to_terms in the partitioned pass's
        // replicated head, so a local compute_res reproduces the full-pass
        // result exactly.
        auto& part = do_jx_mpi_;
        if (!part.val_active || part.def_val_current.empty() ||
            static_cast<int>(part.def_val_current.size()) != ndef ||
            static_cast<int>(part.def_closed.size()) != ndef)
            return;

        std::vector<int> stale;
        for (int i = 0; i < ndef; i++) {
            if (strcmp(names_def[i].c_str(), name) != 0)
                continue;
            for (const int j : part.def_closed[i])
                if (!part.def_val_current[j])
                    stale.push_back(j);
            if (!part.def_val_current[i])
                stale.push_back(i);
        }
        if (stale.empty())
            return;
        std::sort(stale.begin(), stale.end());
        stale.erase(std::unique(stale.begin(), stale.end()), stale.end());
        for (const int j : stale) {
            def[j]->compute_res();
            part.def_val_current[j] = 1;
        }
    }

    Tensor System_of_eqs::give_val_def(const char* so) const
    {

        char name[LMAX];
        trim_spaces(name, so);
        ensure_def_values_current(name);

        bool found = false;
        int valence = -1;
        Array<int>* index = nullptr;
        Base_tensor* basis = nullptr;

        for (int i = 0; i < ndef; i++)
            if (!found) {
                if (strcmp(names_def[i].c_str(), name) == 0) {
                    found = true;
                    Tensor auxi(give_def(i)->get_res()->get_val_t());
                    valence = auxi.get_valence();
                    index = new Array<int>(auxi.get_index_type());
                    basis = new Base_tensor(auxi.get_basis());
                }
            }

        if (!found) {
            std::string def(so);
            std::ostringstream oss;
            oss << "Definition " << def << " not found" << endl;
            KADATH_THROW(oss.str());
        }

        Tensor res(espace, valence, *index, *basis);
        res = 0;

        delete index;
        delete basis;

        for (int i = 0; i < ndef; i++)
            if (strcmp(names_def[i].c_str(), name) == 0) {
                int zedom = give_def(i)->get_res()->get_dom();
                Tensor auxi(give_def(i)->get_res()->get_val_t());
                for (int n = 0; n < res.get_n_comp(); n++) {
                    Array<int> ind(res.indices(n));
                    res.set(ind).set_domain(zedom) = auxi(ind)(zedom);
                }
                res.set_basis(zedom) = auxi.get_basis().get_basis(zedom);
            }

        return res;
    }

    const Val_domain& System_of_eqs::give_val_def_scalar_domain(const char* so, int domain) const
    {
        char name[LMAX];
        trim_spaces(name, so);
        ensure_def_values_current(name);

        int match = -1;
        for (int i = 0; i < ndef; ++i) {
            if (strcmp(names_def[i].c_str(), name) == 0 &&
                give_def(i)->get_res()->get_dom() == domain)
                match = i;
        }

        if (match < 0) {
            std::ostringstream oss;
            oss << "Definition " << name << " not found on domain " << domain << endl;
            KADATH_THROW(oss.str());
        }

        const Tensor& value = give_def(match)->get_res()->get_val_t();
        if (value.get_valence() != 0) {
            std::ostringstream oss;
            oss << "Definition " << name << " is not scalar" << endl;
            KADATH_THROW(oss.str());
        }

        return value()(domain);
    }

    void System_of_eqs::print_system_structure(std::ostream& os) const
    {
        if (nbr_conditions == -1) {
            const_cast<System_of_eqs*>(this)->sec_member();
        }

        os << "--------------------------------------------------------------------------------" << std::endl;
        os << "System of Equations Structure" << std::endl;
        os << "--------------------------------------------------------------------------------" << std::endl;

        os << "Global Scalar Unknowns (" << nvar_double << "):" << std::endl;
        for (int i = 0; i < nvar_double; ++i) {
            os << "  " << i << ": " << (!names_var_double[i].empty() ? names_var_double[i].c_str() : "unnamed")
               << std::endl;
        }

        int n_var_dom = espace.nbr_unknowns_from_variable_domains();
        if (n_var_dom > 0) {
            os << "Variable Domain Unknowns: " << n_var_dom << " coefficients" << std::endl;
        }
        os << std::endl;

        for (int d = dom_min; d <= dom_max; ++d) {
            os << "Domain " << d << ":" << std::endl;

            // Unknowns
            os << "  Unknown Fields:" << std::endl;
            bool has_fields = false;
            for (int i = 0; i < nvar; ++i) {
                int n_coeffs = espace.get_domain(d)->nbr_unknowns(*var[i], d);
                if (n_coeffs > 0) {
                    os << "    " << (!names_var[i].empty() ? names_var[i].c_str() : "unnamed") << " (" << n_coeffs << " coeffs)"
                       << std::endl;
                    has_fields = true;
                }
            }
            if (!has_fields)
                os << "    (none)" << std::endl;

            // Equations
            os << "  Equations:" << std::endl;
            bool has_eqs = false;

            // Integral equations
            for (size_t i = 0; i < eq_int_list.size(); ++i) {
                const auto& entry = eq_int_list[i];
                if (std::get<1>(entry) == d) {
                    os << "    [Integral] " << std::get<0>(entry);
                    if (std::get<2>(entry) != -1)
                        os << " [BC: " << std::get<2>(entry) << "]";
                    os << " (1 condition)";
                    os << std::endl;
                    has_eqs = true;
                }
            }

            // Standard equations
            for (size_t i = 0; i < eq_list.size(); ++i) {
                const auto& entry = eq_list[i];
                if (std::get<1>(entry) == d) {
                    os << "    " << std::get<0>(entry);
                    if (std::get<2>(entry) != -1)
                        os << " [BC: " << std::get<2>(entry) << "]";
                    if (i < static_cast<size_t>(neq)) {
                        os << " (" << eq[i]->get_n_cond_tot() << " conditions)";
                    }
                    os << std::endl;
                    has_eqs = true;
                }
            }
            if (!has_eqs)
                os << "    (none)" << std::endl;
            os << std::endl;
        }
        os << "--------------------------------------------------------------------------------" << std::endl;
    }

    Term_eq** System_of_eqs::results_shadow_view()
    {
        // Grow `results` to fit the total number of slots Equation::apply will write
        // (sum of n_ope across all registered equations). Equation registration is
        // structural, so the slot count only changes when `neq` changes.
        if (results_shadow_equation_count != neq) {
            std::size_t total = 0;
            for (const auto& e : eq) {
                if (e) total += static_cast<std::size_t>(e->n_ope);
            }
            results_shadow_slot_count = total;
            results_shadow_equation_count = neq;
        }
        const std::size_t total = results_shadow_slot_count;
        if (results.size() < total) results.resize(total);
        if (results_shadow.size() != results.size()) {
            results_shadow.resize(results.size());
            for (std::size_t i = 0; i < results.size(); ++i)
                results_shadow[i] = results[i].get();
        }
        return results_shadow.data();
    }

    void System_of_eqs::results_shadow_sync()
    {
        for (std::size_t i = 0; i < results.size(); ++i) {
            // Equation::apply() may allocate new Term_eq into results_shadow[i] when the slot
            // was nullptr. Transfer ownership back into the owning unique_ptr.
            if (results_shadow[i] != results[i].get())
                results[i].reset(results_shadow[i]);
        }
    }

} // namespace Kadath
