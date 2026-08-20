#include "For_Kadath/Array/headcpp.hpp"
#include "For_Kadath/Array/exceptions.hpp"
#include "For_Kadath/Space/trumpet.hpp"
#include "For_Kadath/System_of_eqs/system_of_eqs.hpp"

namespace Kadath
{
    Space_trumpet::Space_trumpet(int ttype, const Point& center, const Dim_array& res, const Array<double>& bounds)
        : Space_spheric_adapted(ttype, center, res, bounds)
    {
    }

    Space_trumpet::Space_trumpet(int ttype, const Point& center, const Dim_array& res,
                                 const std::vector<double>& bounds)
        : Space_spheric_adapted(ttype, center, res, bounds)
    {
    }

    Space_trumpet::Space_trumpet(BinarySource& source) : Space_spheric_adapted(source) {}

    void Space_trumpet::xx_to_vars_variable_domains(System_of_eqs* sys, const Array<double>& xx, int& pos) const
    {
        int old_pos = pos;
        Val_domain cor_outer(domains[1]);
        domains[1]->xx_to_vars_from_adapted(cor_outer, xx, pos);
        pos = old_pos;
        Val_domain cor_inner(domains[2]);
        domains[2]->xx_to_vars_from_adapted(cor_inner, xx, pos);

        for (int i = 0; i < sys->nvar; ++i) {
            for (int n = 0; n < sys->var[i]->get_n_comp(); ++n) {
                Scalar res(*this);
                domains[1]->update_variable(cor_outer, *sys->var[i]->cmp[n], res);
                domains[2]->update_variable(cor_inner, *sys->var[i]->cmp[n], res);

                sys->var[i]->cmp[n]->set_domain(1) = res(1);
                sys->var[i]->cmp[n]->set_domain(2) = res(2);
            }
        }

        const int span = sys->dom_max - sys->dom_min + 1;
        for (int i = 0; i < sys->ncst; ++i) {
            const int base = i * span;
            if (sys->cst[base]->get_type_data() != TERM_T)
                continue;

            for (int n = 0; n < sys->cst[base]->val_t->get_n_comp(); ++n) {
                Scalar so(*this);
                so = 0;
                for (int d = 1; d <= 2; ++d) {
                    if (d < sys->dom_min || d > sys->dom_max)
                        continue;
                    so.set_domain(d) = (*sys->cst[base + d - sys->dom_min]->val_t->cmp[n])(d);
                }

                Scalar res(*this);
                res = 0;
                domains[1]->update_constante(cor_outer, so, res);
                domains[2]->update_constante(cor_inner, so, res);

                for (int d = 1; d <= 2; ++d) {
                    if (d < sys->dom_min || d > sys->dom_max)
                        continue;
                    sys->cst[base + d - sys->dom_min]->val_t->cmp[n]->set_domain(d) = res(d);
                }
            }
        }

        domains[1]->update_mapping(cor_outer);
        domains[2]->update_mapping(cor_inner);
    }

    void Space_trumpet::add_bc_horizon(System_of_eqs& syst, const char* eq, int nused, Array<int>** pused)
    {
        syst.add_eq_bc(2, INNER_BC, eq, nused, pused);
    }

    void Space_trumpet::add_bc_inf(System_of_eqs& syst, const char* eq, int nused, Array<int>** pused)
    {
        syst.add_eq_bc(get_nbr_domains() - 1, OUTER_BC, eq, nused, pused);
    }

    void Space_trumpet::add_eq_int_horizon(System_of_eqs& syst, const char* eq)
    {
        add_eq_int(syst, 2, INNER_BC, eq);
    }
} // namespace Kadath
