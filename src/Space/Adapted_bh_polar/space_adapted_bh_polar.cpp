// Domain layout: nucleus (0), outer homothetic (1), inner homothetic (2), then optional shells from bounds[2..], then
// compactified.

#include "For_Kadath/Array/headcpp.hpp"
#include "For_Kadath/Space/adapted_bh_polar.hpp"
#include "For_Kadath/Domain/polar.hpp"
#include "For_Kadath/Utilities/utilities.hpp"
#include "For_Kadath/Array/point.hpp"
#include "For_Kadath/Scalar/scalar.hpp"
#include "For_Kadath/System_of_eqs/system_of_eqs.hpp"
#include "For_Kadath/Utilities/name_tools.hpp"

namespace Kadath
{
    Space_adapted_bh_polar::Space_adapted_bh_polar(int ttype, const Point& center, const Dim_array& res,
                                                   const Array<double>& bounds)
    {

        assert(bounds.get_ndim() == 1);

        ndim = 2;

        n_shells = bounds.get_size(0) - 3;
        HOMOTHETIC_OUTER = 1;
        HOMOTHETIC_INNER = 2;

        nbr_domains = bounds.get_size(0) + 1;
        assert(nbr_domains >= 4);
        type_base = ttype;
        domains = new Domain*[nbr_domains];

        domains[0] = new Domain_polar_nucleus(0, ttype, bounds(0), center, res);
        domains[HOMOTHETIC_OUTER] =
            new Domain_polar_shell_outer_homothetic(*this, HOMOTHETIC_OUTER, ttype, bounds(0), bounds(1), center, res);
        domains[HOMOTHETIC_INNER] =
            new Domain_polar_shell_inner_homothetic(*this, HOMOTHETIC_INNER, ttype, bounds(1), bounds(2), center, res);

        for (int i = 0; i < n_shells; ++i) {
            const int dom = 3 + i;
            domains[dom] = new Domain_polar_shell(dom, ttype, bounds(2 + i), bounds(2 + i + 1), center, res);
        }

        domains[nbr_domains - 1] =
            new Domain_polar_compact(nbr_domains - 1, ttype, bounds(nbr_domains - 2), center, res);

        const Domain_polar_shell_outer_homothetic* pshell_outer =
            dynamic_cast<const Domain_polar_shell_outer_homothetic*>(domains[HOMOTHETIC_OUTER]);
        pshell_outer->vars_to_terms();
        const Domain_polar_shell_inner_homothetic* pshell_inner =
            dynamic_cast<const Domain_polar_shell_inner_homothetic*>(domains[HOMOTHETIC_INNER]);
        pshell_inner->vars_to_terms();
    }

    Space_adapted_bh_polar::Space_adapted_bh_polar(int ttype, const Point& center, const Dim_array& res,
                                                   const std::vector<double>& bounds)
    {

        ndim = 2;

        n_shells = bounds.size() - 3;
        HOMOTHETIC_OUTER = 1;
        HOMOTHETIC_INNER = 2;

        nbr_domains = bounds.size() + 1;
        assert(nbr_domains >= 4);
        type_base = ttype;
        domains = new Domain*[nbr_domains];

        domains[0] = new Domain_polar_nucleus(0, ttype, bounds[0], center, res);
        domains[HOMOTHETIC_OUTER] =
            new Domain_polar_shell_outer_homothetic(*this, HOMOTHETIC_OUTER, ttype, bounds[0], bounds[1], center, res);
        domains[HOMOTHETIC_INNER] =
            new Domain_polar_shell_inner_homothetic(*this, HOMOTHETIC_INNER, ttype, bounds[1], bounds[2], center, res);

        for (int i = 0; i < n_shells; ++i) {
            const int dom = 3 + i;
            domains[dom] = new Domain_polar_shell(dom, ttype, bounds[2 + i], bounds[2 + i + 1], center, res);
        }

        domains[nbr_domains - 1] =
            new Domain_polar_compact(nbr_domains - 1, ttype, bounds[nbr_domains - 2], center, res);

        const Domain_polar_shell_outer_homothetic* pshell_outer =
            dynamic_cast<const Domain_polar_shell_outer_homothetic*>(domains[HOMOTHETIC_OUTER]);
        pshell_outer->vars_to_terms();
        const Domain_polar_shell_inner_homothetic* pshell_inner =
            dynamic_cast<const Domain_polar_shell_inner_homothetic*>(domains[HOMOTHETIC_INNER]);
        pshell_inner->vars_to_terms();
    }

    Space_adapted_bh_polar::Space_adapted_bh_polar(const Space_adapted_bh_polar& sp)
    {
        nbr_domains = sp.nbr_domains;
        ndim = sp.ndim;
        type_base = sp.type_base;

        HOMOTHETIC_OUTER = 1;
        HOMOTHETIC_INNER = 2;

        domains = new Domain*[nbr_domains];

        const Domain_polar_nucleus* d_nuc = dynamic_cast<const Domain_polar_nucleus*>(sp.get_domain(0));
        domains[0] = new Domain_polar_nucleus(*d_nuc, true);

        const Domain_polar_shell_outer_homothetic* pouter =
            dynamic_cast<const Domain_polar_shell_outer_homothetic*>(sp.get_domain(1));
        domains[1] = new Domain_polar_shell_outer_homothetic(*this, *pouter);
        const Domain_polar_shell_inner_homothetic* pinner =
            dynamic_cast<const Domain_polar_shell_inner_homothetic*>(sp.get_domain(2));
        domains[2] = new Domain_polar_shell_inner_homothetic(*this, *pinner);

        for (int i = 3; i < nbr_domains - 1; i++) {
            const Domain_polar_shell* d_shell = dynamic_cast<const Domain_polar_shell*>(sp.get_domain(i));
            domains[i] = new Domain_polar_shell(*d_shell, true);
        }

        const Domain_polar_compact* d_compact =
            dynamic_cast<const Domain_polar_compact*>(sp.get_domain(nbr_domains - 1));
        domains[nbr_domains - 1] = new Domain_polar_compact(*d_compact, true);

        const Domain_polar_shell_outer_homothetic* pouter_1 =
            dynamic_cast<const Domain_polar_shell_outer_homothetic*>(domains[1]);
        pouter_1->vars_to_terms();
        pouter_1->update();
        const Domain_polar_shell_inner_homothetic* pinner_1 =
            dynamic_cast<const Domain_polar_shell_inner_homothetic*>(domains[2]);
        pinner_1->vars_to_terms();
        pinner_1->update();
    }

    Space_adapted_bh_polar::Space_adapted_bh_polar(BinarySource& source)
    {
        nbr_domains = source.read<int>();
        ndim = source.read<int>();
        n_shells = source.read<int>();
        type_base = source.read<int>();

        HOMOTHETIC_OUTER = 1;
        HOMOTHETIC_INNER = 2;

        domains = new Domain*[nbr_domains];
        domains[0] = new Domain_polar_nucleus(0, source);
        domains[HOMOTHETIC_OUTER] = new Domain_polar_shell_outer_homothetic(*this, HOMOTHETIC_OUTER, source);
        domains[HOMOTHETIC_INNER] = new Domain_polar_shell_inner_homothetic(*this, HOMOTHETIC_INNER, source);
        for (int i = 0; i < n_shells; ++i) {
            const int dom = 3 + i;
            domains[dom] = new Domain_polar_shell(dom, source);
        }

        domains[nbr_domains - 1] = new Domain_polar_compact(nbr_domains - 1, source);

        const Domain_polar_shell_outer_homothetic* pshell_outer =
            dynamic_cast<const Domain_polar_shell_outer_homothetic*>(domains[HOMOTHETIC_OUTER]);
        pshell_outer->vars_to_terms();
        const Domain_polar_shell_inner_homothetic* pshell_inner =
            dynamic_cast<const Domain_polar_shell_inner_homothetic*>(domains[HOMOTHETIC_INNER]);
        pshell_inner->vars_to_terms();
    }

    Space_adapted_bh_polar::~Space_adapted_bh_polar()
    {
        const Domain_polar_shell_outer_homothetic* pshell_outer =
            dynamic_cast<const Domain_polar_shell_outer_homothetic*>(domains[HOMOTHETIC_OUTER]);
        pshell_outer->del_deriv();
        const Domain_polar_shell_inner_homothetic* pshell_inner =
            dynamic_cast<const Domain_polar_shell_inner_homothetic*>(domains[HOMOTHETIC_INNER]);
        pshell_inner->del_deriv();
    }

    void Space_adapted_bh_polar::save(BinarySink& sink) const
    {
        sink.write<int>(nbr_domains);
        sink.write<int>(ndim);
        sink.write<int>(n_shells);
        sink.write<int>(type_base);
        for (int i = 0; i < nbr_domains; i++)
            domains[i]->save(sink);
    }

    int Space_adapted_bh_polar::nbr_unknowns_from_variable_domains() const
    {
        return domains[HOMOTHETIC_OUTER]->nbr_unknowns_from_adapted();
    }

    void Space_adapted_bh_polar::affecte_coef_to_variable_domains(int& pos, int cc, Array<int>& zedoms) const
    {
        bool found_outer = false;
        int old_pos = pos;
        domains[HOMOTHETIC_OUTER]->affecte_coef(pos, cc, found_outer);
        pos = old_pos;
        bool found_inner = false;
        domains[HOMOTHETIC_INNER]->affecte_coef(pos, cc, found_inner);
        assert(found_outer == found_inner);
        if (found_outer) {
            zedoms.set(0) = HOMOTHETIC_OUTER;
            zedoms.set(1) = HOMOTHETIC_INNER;
        } else {
            zedoms.set(0) = -1;
            zedoms.set(1) = -1;
        }
    }

    void Space_adapted_bh_polar::xx_to_ders_variable_domains(const Array<double>& xx, int& conte) const
    {
        int old_conte = conte;
        domains[HOMOTHETIC_OUTER]->xx_to_ders_from_adapted(xx, conte);
        conte = old_conte;
        domains[HOMOTHETIC_INNER]->xx_to_ders_from_adapted(xx, conte);
    }

    void Space_adapted_bh_polar::xx_to_vars_variable_domains(System_of_eqs* sys, const Array<double>& xx,
                                                             int& pos) const
    {

        int old_pos = pos;
        Val_domain cor_outer(domains[HOMOTHETIC_OUTER]);
        domains[HOMOTHETIC_OUTER]->xx_to_vars_from_adapted(cor_outer, xx, pos);
        pos = old_pos;
        Val_domain cor_inner(domains[HOMOTHETIC_INNER]);
        domains[HOMOTHETIC_INNER]->xx_to_vars_from_adapted(cor_inner, xx, pos);

        for (int i = 0; i < sys->nvar; i++) {
            for (int n = 0; n < sys->var[i]->get_n_comp(); n++) {
                Scalar res(*this);
                domains[HOMOTHETIC_OUTER]->update_variable(cor_outer, *sys->var[i]->cmp[n], res);
                domains[HOMOTHETIC_INNER]->update_variable(cor_inner, *sys->var[i]->cmp[n], res);

                sys->var[i]->cmp[n]->set_domain(HOMOTHETIC_OUTER) = res(HOMOTHETIC_OUTER);
                sys->var[i]->cmp[n]->set_domain(HOMOTHETIC_INNER) = res(HOMOTHETIC_INNER);
            }
        }

        for (int i = 0; i < sys->ncst; i++)
            if (sys->cst[i * (sys->dom_max - sys->dom_min + 1)]->get_type_data() == TERM_T)
                for (int n = 0; n < sys->cst[i * (sys->dom_max - sys->dom_min + 1)]->val_t->get_n_comp(); n++) {

                    Scalar so(*this);
                    so = 0;
                    so.set_domain(HOMOTHETIC_OUTER) =
                        (*sys->cst[i * (sys->dom_max - sys->dom_min + 1) + HOMOTHETIC_OUTER - sys->dom_min]
                              ->val_t->cmp[n])(HOMOTHETIC_OUTER);
                    so.set_domain(HOMOTHETIC_INNER) =
                        (*sys->cst[i * (sys->dom_max - sys->dom_min + 1) + HOMOTHETIC_INNER - sys->dom_min]
                              ->val_t->cmp[n])(HOMOTHETIC_INNER);

                    Scalar res(*this);
                    res = 0;
                    domains[HOMOTHETIC_OUTER]->update_constante(cor_outer, so, res);
                    domains[HOMOTHETIC_INNER]->update_constante(cor_inner, so, res);

                    sys->cst[i * (sys->dom_max - sys->dom_min + 1) + HOMOTHETIC_OUTER - sys->dom_min]
                        ->val_t->cmp[n]
                        ->set_domain(HOMOTHETIC_OUTER) = res(HOMOTHETIC_OUTER);
                    sys->cst[i * (sys->dom_max - sys->dom_min + 1) + HOMOTHETIC_INNER - sys->dom_min]
                        ->val_t->cmp[n]
                        ->set_domain(HOMOTHETIC_INNER) = res(HOMOTHETIC_INNER);
                }

        domains[HOMOTHETIC_OUTER]->update_mapping(cor_outer);
        domains[HOMOTHETIC_INNER]->update_mapping(cor_inner);
    }

    void Space_adapted_bh_polar::add_eq_ori(System_of_eqs& sys, const char* name)
    {
        Index pos(domains[0]->get_nbr_points());
        char auxi[LMAX];
        trim_spaces(auxi, name);
        sys.add_eq_val(0, auxi, pos);
    }

} // namespace Kadath
