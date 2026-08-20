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

#include "For_Kadath/Array/headcpp.hpp"
#include "For_Kadath/Domain/adapted_polar.hpp"
#include "For_Kadath/Domain/polar.hpp"
#include "For_Kadath/Utilities/utilities.hpp"
#include "For_Kadath/Array/point.hpp"
#include "For_Kadath/Scalar/scalar.hpp"
#include "For_Kadath/System_of_eqs/system_of_eqs.hpp"
#include "For_Kadath/Utilities/name_tools.hpp"

namespace Kadath
{
    Space_polar_bilateral_adapted::Space_polar_bilateral_adapted(int ttype, const Point& center, const Dim_array& res,
                                                                 const Array<double>& bounds)
    {

        // Verif :
        assert(bounds.get_ndim() == 1);

        ndim = 2;

        n_shells = bounds.get_size(0) - 2;
        ADAPTED_BILATERAL = n_shells + 1;

        nbr_domains = bounds.get_size(0) + 1;
        assert(nbr_domains >= 3);
        type_base = ttype;
        domains = new Domain*[nbr_domains];

        // Nucleus
        domains[0] = new Domain_polar_nucleus(0, ttype, bounds(0), center, res);
        for (int i = 1; i < ADAPTED_BILATERAL; i++)
            domains[i] = new Domain_polar_shell(i, ttype, bounds(i - 1), bounds(i), center, res);

        domains[ADAPTED_BILATERAL] = new Domain_polar_shell_bilateral_adapted(
            *this, ADAPTED_BILATERAL, ttype, bounds(ADAPTED_BILATERAL - 1), bounds(ADAPTED_BILATERAL), center, res);

        domains[nbr_domains - 1] =
            new Domain_polar_compact(nbr_domains - 1, ttype, bounds(nbr_domains - 2), center, res);

        const Domain_polar_shell_bilateral_adapted* pshell =
            dynamic_cast<const Domain_polar_shell_bilateral_adapted*>(domains[ADAPTED_BILATERAL]);
        pshell->vars_to_terms();
    }

    Space_polar_bilateral_adapted::Space_polar_bilateral_adapted(int ttype, const Point& center, const Dim_array& res,
                                                                 const std::vector<double>& bounds)
    {

        ndim = 2;

        n_shells = bounds.size() - 2;
        ADAPTED_BILATERAL = n_shells + 1;

        nbr_domains = bounds.size() + 1;
        assert(nbr_domains >= 3);
        type_base = ttype;
        domains = new Domain*[nbr_domains];

        // Nucleus
        domains[0] = new Domain_polar_nucleus(0, ttype, bounds[0], center, res);
        for (int i = 1; i < ADAPTED_BILATERAL; i++)
            domains[i] = new Domain_polar_shell(i, ttype, bounds[i - 1], bounds[i], center, res);

        domains[ADAPTED_BILATERAL] = new Domain_polar_shell_bilateral_adapted(
            *this, ADAPTED_BILATERAL, ttype, bounds[ADAPTED_BILATERAL - 1], bounds[ADAPTED_BILATERAL], center, res);

        domains[nbr_domains - 1] =
            new Domain_polar_compact(nbr_domains - 1, ttype, bounds[nbr_domains - 2], center, res);

        const Domain_polar_shell_bilateral_adapted* pshell =
            dynamic_cast<const Domain_polar_shell_bilateral_adapted*>(domains[ADAPTED_BILATERAL]);
        pshell->vars_to_terms();
    }

    Space_polar_bilateral_adapted::Space_polar_bilateral_adapted(const Space_polar_bilateral_adapted& sp)
    {
        // Copy Space values
        nbr_domains = sp.nbr_domains;
        ndim = sp.ndim;
        type_base = sp.type_base;
        n_shells = sp.n_shells;
        ADAPTED_BILATERAL = sp.ADAPTED_BILATERAL;

        // Initialize Domain array
        domains = new Domain*[nbr_domains];

        // Copy nucleus
        const Domain_polar_nucleus* d_nuc = dynamic_cast<const Domain_polar_nucleus*>(sp.get_domain(0));
        domains[0] = new Domain_polar_nucleus(*d_nuc, true);

        const Domain_polar_shell_bilateral_adapted* pbilat =
            dynamic_cast<const Domain_polar_shell_bilateral_adapted*>(sp.get_domain(ADAPTED_BILATERAL));
        domains[ADAPTED_BILATERAL] = new Domain_polar_shell_bilateral_adapted(*this, *pbilat);

        for (int i = 1; i < ADAPTED_BILATERAL; i++) {
            const Domain_polar_shell* d_shell = dynamic_cast<const Domain_polar_shell*>(sp.get_domain(i));
            domains[i] = new Domain_polar_shell(*d_shell, true);
        }
        // Compactified
        const Domain_polar_compact* d_compact =
            dynamic_cast<const Domain_polar_compact*>(sp.get_domain(nbr_domains - 1));
        domains[nbr_domains - 1] = new Domain_polar_compact(*d_compact, true);

        const Domain_polar_shell_bilateral_adapted* pbilat_1 =
            dynamic_cast<const Domain_polar_shell_bilateral_adapted*>(domains[ADAPTED_BILATERAL]);
        pbilat_1->vars_to_terms();
        pbilat_1->update();
    }

    Space_polar_bilateral_adapted::Space_polar_bilateral_adapted(BinarySource& source)
    {
        nbr_domains = source.read<int>();
        ndim = source.read<int>();
        n_shells = source.read<int>();
        type_base = source.read<int>();

        ADAPTED_BILATERAL = n_shells + 1;

        domains = new Domain*[nbr_domains];
        domains[0] = new Domain_polar_nucleus(0, source);
        for (int i = 1; i < ADAPTED_BILATERAL; i++)
            domains[i] = new Domain_polar_shell(i, source);

        domains[ADAPTED_BILATERAL] = new Domain_polar_shell_bilateral_adapted(*this, ADAPTED_BILATERAL, source);
        domains[nbr_domains - 1] = new Domain_polar_compact(nbr_domains - 1, source);

        const Domain_polar_shell_bilateral_adapted* pshell =
            dynamic_cast<const Domain_polar_shell_bilateral_adapted*>(domains[ADAPTED_BILATERAL]);
        pshell->vars_to_terms();
    }

    Space_polar_bilateral_adapted::~Space_polar_bilateral_adapted()
    {
        const Domain_polar_shell_bilateral_adapted* pshell =
            dynamic_cast<const Domain_polar_shell_bilateral_adapted*>(domains[ADAPTED_BILATERAL]);
        pshell->del_deriv();
    }

    void Space_polar_bilateral_adapted::save(BinarySink& sink) const
    {
        sink.write<int>(nbr_domains);
        sink.write<int>(ndim);
        sink.write<int>(n_shells);
        sink.write<int>(type_base);
        for (int i = 0; i < nbr_domains; i++)
            domains[i]->save(sink);
    }

    int Space_polar_bilateral_adapted::nbr_unknowns_from_variable_domains() const
    {

        return domains[ADAPTED_BILATERAL]->nbr_unknowns_from_adapted();
    }

    void Space_polar_bilateral_adapted::affecte_coef_to_variable_domains(int& pos, int cc, Array<int>& zedoms) const
    {

        bool found = false;
        int old_pos = pos;
        domains[ADAPTED_BILATERAL]->affecte_coef(pos, cc, found);
        pos = old_pos;
        if (found) {
            zedoms.set(0) = ADAPTED_BILATERAL;
            zedoms.set(1) = -1;
        } else {
            zedoms.set(0) = -1;
            zedoms.set(1) = -1;
        }
    }

    void Space_polar_bilateral_adapted::xx_to_ders_variable_domains(const Array<double>& xx, int& conte) const
    {

        domains[ADAPTED_BILATERAL]->xx_to_ders_from_adapted(xx, conte);
    }

    void Space_polar_bilateral_adapted::xx_to_vars_variable_domains(System_of_eqs* sys, const Array<double>& xx,
                                                                    int& pos) const
    {

        Domain_polar_shell_bilateral_adapted* pbilat =
            dynamic_cast<Domain_polar_shell_bilateral_adapted*>(domains[ADAPTED_BILATERAL]);

        Val_domain cor_inner(domains[ADAPTED_BILATERAL]);
        Val_domain cor_outer(domains[ADAPTED_BILATERAL]);
        pbilat->xx_to_vars_from_adapted(cor_inner, cor_outer, xx, pos);

        // Now update the variables :
        for (int i = 0; i < sys->nvar; i++) {
            for (int n = 0; n < sys->var[i]->get_n_comp(); n++) {
                Scalar res(*this);
                pbilat->update_variable(cor_inner, cor_outer, *sys->var[i]->cmp[n], res);
                sys->var[i]->cmp[n]->set_domain(ADAPTED_BILATERAL) = res(ADAPTED_BILATERAL);
            }
        }

        // Now the constants :
        for (int i = 0; i < sys->ncst; i++)
            if (sys->cst[i * (sys->dom_max - sys->dom_min + 1)]->get_type_data() == TERM_T)
                for (int n = 0; n < sys->cst[i * (sys->dom_max - sys->dom_min + 1)]->val_t->get_n_comp(); n++) {

                    Scalar so(*this);
                    so = 0;
                    so.set_domain(ADAPTED_BILATERAL) =
                        (*sys->cst[i * (sys->dom_max - sys->dom_min + 1) + ADAPTED_BILATERAL - sys->dom_min]
                              ->val_t->cmp[n])(ADAPTED_BILATERAL);

                    Scalar res(*this);
                    res = 0;
                    pbilat->update_constante(cor_inner, cor_outer, so, res);

                    sys->cst[i * (sys->dom_max - sys->dom_min + 1) + ADAPTED_BILATERAL - sys->dom_min]
                        ->val_t->cmp[n]
                        ->set_domain(ADAPTED_BILATERAL) = res(ADAPTED_BILATERAL);
                }

        // Update the mapping :
        pbilat->update_mapping(cor_inner, cor_outer);
    }

    void Space_polar_bilateral_adapted::add_eq_ori(System_of_eqs& sys, const char* name)
    {

        Index pos(domains[0]->get_nbr_points());
        char auxi[LMAX];
        trim_spaces(auxi, name);
        sys.add_eq_val(0, auxi, pos);
    }
} // namespace Kadath
