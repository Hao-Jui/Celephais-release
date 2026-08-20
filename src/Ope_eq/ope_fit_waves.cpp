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

#include "For_Kadath/Ope_eq/ope_eq.hpp"
#include "For_Kadath/Array/exceptions.hpp"
#include "For_Kadath/Scalar/scalar.hpp"
#include "For_Kadath/Param/param.hpp"
#include "For_Kadath/System_of_eqs/system_of_eqs.hpp"
#include "For_Kadath/Domain/spheric.hpp"
namespace Kadath
{
    Array<double> mat_leg_even(int, int);
    Array<double> mat_leg_odd(int, int);

    Term_eq one(const Space& space, int mm, int ll, const Term_eq& omega, const Param& parfit)
    {

        int dom = omega.get_dom();

        Scalar valr(space);
        Scalar derr(space);
        valr.set_domain(dom) = 1;
        derr.set_domain(dom) = 0;

        return Term_eq(dom, valr, derr);
    }

    Term_eq fjl(const Space& space, int mm, int ll, const Term_eq& omega, const Param& parfit)
    {

        double rmatch = parfit.get_double(0);
        int dom = omega.get_dom();

        Scalar valr(space);
        Scalar derr(space);
        valr.set_domain(dom) = space.get_domain(dom)->get_radius();
        valr.std_base();
        derr.set_domain(dom) = 0;
        derr.std_base();
        Term_eq rr(dom, valr, derr);
        valr.set_domain(dom) = rmatch;
        valr.std_base();
        Term_eq rm(dom, valr, derr);

        if (mm == 0) {
            Term_eq res(pow(rm / rr, ll + 1));
            return res;
        } else {

            Term_eq res(bessel_jl(mm * rr * omega, ll) / bessel_jl(mm * rm * omega, ll));
            return res;
        }
    }

    Term_eq fyl(const Space& space, int mm, int ll, const Term_eq& omega, const Param& parfit)
    {

        double rmatch = parfit.get_double(0);
        int dom = omega.get_dom();

        Scalar valr(space);
        Scalar derr(space);
        valr.set_domain(dom) = space.get_domain(dom)->get_radius();
        valr.std_base();
        derr.set_domain(dom) = 0;
        derr.std_base();
        Term_eq rr(dom, valr, derr);
        valr.set_domain(dom) = rmatch;
        valr.std_base();
        Term_eq rm(dom, valr, derr);

        if (mm == 0) {
            return (pow(rm / rr, (ll + 1)));
        } else {
            return (bessel_yl(mm * rr * omega, ll) / bessel_yl(mm * rm * omega, ll));
        }
    }

    Ope_fit_waves::Ope_fit_waves(const System_of_eqs* zesys, Ope_eq* target, Ope_eq* omega)
        : Ope_eq(zesys, target->get_dom(), 2)
    {
        parts[0].reset(target);
        parts[1].reset(omega);
    }

    Ope_fit_waves::~Ope_fit_waves() {}

    Term_eq Ope_fit_waves::action() const
    {
        ScopedOpeActionProfile ope_action_profile_scope(*this);

        Term_eq target(parts[0]->action());
        Term_eq omega(parts[1]->action());

        const Domain_shell* pshell = dynamic_cast<const Domain_shell*>(syst->get_space().get_domain(dom));
        if (pshell == nullptr) {
            KADATH_THROW("Ope_fit_waves called in bad domain");
        }

        return pshell->bc_waves(target, omega);

        /*	bool doder = ((target.der_t==nullptr) || (omega.der_d==nullptr)) ? false : true ;

            // Check if so is a tensor
            if (target.type_data != TERM_T) {
                KADATH_THROW("Ope_fit_waves::action only defined for a tensor");
            }
            int valence = target.val_t->get_valence() ;
            if ((valence !=0) && (target.val_t->get_triad()!=target.val_t->get_space().get_bvect_cart())) {
                KADATH_THROW("Ope_fit_waves::action only defined for Cartesian tensorial basis of decompostion");
            }

            double rmatch = syst->get_space().get_domain(dom)->get_rmax() ;
            Param parfit ;
            parfit.add_double (rmatch, 0) ;
            const Domain* zedom = syst->get_space().get_domain(dom) ;

            Array<double> peven (mat_leg_even(zedom->get_nbr_coefs()(1), zedom->get_nbr_coefs()(2))) ;
            Array<double> podd (mat_leg_odd(zedom->get_nbr_coefs()(1), zedom->get_nbr_coefs()(2))) ;

            Term_eq res (target) ;

            // Computation component by components...
            for (int i=0 ; i<target.val_t->get_n_comp() ; i++) {
                Array<int> ind (target.val_t->indices(i)) ;
                // Sym or not sym ?
                int sym = 1 ;
                for (int n=0 ; n<ind.get_size(0) ; n++)
                  if (ind(n)==3)
                    sym *= -1 ;
                if (sym==1) {
                    // Sym case
                    Term_eq res_cmp (zedom->harmonics_sym(target(ind), omega, OUTER_BC, fyl, parfit, peven)) ;
                    res.val_t->set(ind) = (*res_cmp.val_t)() ;
                    if (doder) {
                    res.der_t->set(ind) = (*res_cmp.der_t)() ;
                    }
                }
                else {
                    // Asym case
                    Term_eq res_cmp (zedom->harmonics_asym(target(ind), omega, OUTER_BC, fyl, parfit, podd)) ;
                     res.val_t->set(ind) = (*res_cmp.val_t)() ;
                    if (doder) {
                    res.der_t->set(ind) = (*res_cmp.der_t)() ;
                    }
                }

                if ((*target.val_t)(ind)(dom).check_if_zero())
                  res.val_t->set(ind).set_domain(dom).set_zero() ;
                else
                  res.val_t->set(ind).set_domain(dom).set_base() = (*target.val_t)(ind)(dom).get_base() ;
                if (doder) {
                  if ((*target.der_t)(ind)(dom).check_if_zero())
                      res.der_t->set(ind).set_domain(dom).set_zero() ;
                  else
                  res.der_t->set(ind).set_domain(dom).set_base() = (*target.der_t)(ind)(dom).get_base() ;
                }
            }

            return res ;*/
    }

} // namespace Kadath
