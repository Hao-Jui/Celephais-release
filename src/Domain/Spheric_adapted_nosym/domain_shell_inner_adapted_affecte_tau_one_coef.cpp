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
 */

#include "For_Kadath/Array/headcpp.hpp"
#include "For_Kadath/Array/exceptions.hpp"
#include "For_Kadath/Domain/spheric_adapted_nosym.hpp"
#include "For_Kadath/Domain/spheric_nosym_regularization.hpp"
#include "For_Kadath/Array/point.hpp"
#include "For_Kadath/Array/array.hpp"
#include "For_Kadath/Scalar/scalar.hpp"
#include "For_Kadath/Tensor/tensor.hpp"

namespace Kadath
{
    void Domain_shell_inner_adapted_nosym::affecte_tau_one_coef_val_domain(Val_domain& so, int mlim, int cc, int& conte) const
    {

        int kmin = 2 * mlim + 2;

        so.is_zero = false;
        so.allocate_coef();
        *so.cf = 0.;
        Index pos_cf(nbr_coefs);

        bool found = false;

        // True values
        // Loop on phi :
        for (int k = 0; k < nbr_coefs(2) - 1; k++)
            if (k != 1) {
                pos_cf.set(2) = k;
                // Loop on theta
                int baset = (*so.get_base().bases_1d[1])(k);
                for (int j = 0; j < nbr_coefs(1); j++) {
                    pos_cf.set(1) = j;
                    bool true_tet = true;
                    true_tet = detail::spheric_nosym_true_theta_coef(baset, j, k, kmin, nbr_coefs(1));

                    if (true_tet)
                        for (int i = 0; i < nbr_coefs(0); i++) {
                            pos_cf.set(0) = i;
                            if (conte == cc) {
                                so.cf->set(pos_cf) = 1;
                                found = true;
                                if (detail::spheric_nosym_uses_theta_galerkin(baset, k, kmin)) {
                                    pos_cf.set(1) = detail::spheric_nosym_theta_anchor(baset, j);
                                    so.cf->set(pos_cf) = -detail::spheric_nosym_basis_anchor_weight(baset, j);
                                }
                            } else {
                                so.cf->set(pos_cf) = 0.;
                            }
                            conte++;
                        }
                }
            }
        // If not found put to zero :
        if (!found)
            so.set_zero();
    }

    void Domain_shell_inner_adapted_nosym::affecte_tau_one_coef(Tensor& tt, int dom, int cc, int& pos_cf) const
    {

        // Check right domain
        assert(tt.get_space().get_domain(dom) == this);

        int val = tt.get_valence();
        switch (val) {
            case 0:
                affecte_tau_one_coef_val_domain(tt.set().set_domain(dom), 0, cc, pos_cf);
                break;
            case 1: {
                bool found = false;
                // Cartesian basis
                if (tt.get_basis().get_basis(dom) == CARTESIAN_BASIS) {
                    affecte_tau_one_coef_val_domain(tt.set(1).set_domain(dom), 0, cc, pos_cf);
                    affecte_tau_one_coef_val_domain(tt.set(2).set_domain(dom), 0, cc, pos_cf);
                    affecte_tau_one_coef_val_domain(tt.set(3).set_domain(dom), 0, cc, pos_cf);
                    found = true;
                }
                // Spherical coordinates
                if (tt.get_basis().get_basis(dom) == SPHERICAL_BASIS) {
                    affecte_tau_one_coef_val_domain(tt.set(1).set_domain(dom), 0, cc, pos_cf);
                    affecte_tau_one_coef_val_domain(tt.set(2).set_domain(dom), 1, cc, pos_cf);
                    affecte_tau_one_coef_val_domain(tt.set(3).set_domain(dom), 1, cc, pos_cf);
                    found = true;
                }
                if (!found) {
                    KADATH_THROW("Unknown type of vector Domain_shell_inner_adapted_nosym::affecte_tau_one_coef");
                }
            } break;
            case 2: {
                bool found = false;
                // Cartesian basis and symetric
                if ((tt.get_basis().get_basis(dom) == CARTESIAN_BASIS) && (tt.get_n_comp() == 6)) {
                    affecte_tau_one_coef_val_domain(tt.set(1, 1).set_domain(dom), 0, cc, pos_cf);
                    affecte_tau_one_coef_val_domain(tt.set(1, 2).set_domain(dom), 0, cc, pos_cf);
                    affecte_tau_one_coef_val_domain(tt.set(1, 3).set_domain(dom), 0, cc, pos_cf);
                    affecte_tau_one_coef_val_domain(tt.set(2, 2).set_domain(dom), 0, cc, pos_cf);
                    affecte_tau_one_coef_val_domain(tt.set(2, 3).set_domain(dom), 0, cc, pos_cf);
                    affecte_tau_one_coef_val_domain(tt.set(3, 3).set_domain(dom), 0, cc, pos_cf);
                    found = true;
                }
                // Cartesian basis and not symetric
                if ((tt.get_basis().get_basis(dom) == CARTESIAN_BASIS) && (tt.get_n_comp() == 9)) {
                    affecte_tau_one_coef_val_domain(tt.set(1, 1).set_domain(dom), 0, cc, pos_cf);
                    affecte_tau_one_coef_val_domain(tt.set(1, 2).set_domain(dom), 0, cc, pos_cf);
                    affecte_tau_one_coef_val_domain(tt.set(1, 3).set_domain(dom), 0, cc, pos_cf);
                    affecte_tau_one_coef_val_domain(tt.set(2, 1).set_domain(dom), 0, cc, pos_cf);
                    affecte_tau_one_coef_val_domain(tt.set(2, 2).set_domain(dom), 0, cc, pos_cf);
                    affecte_tau_one_coef_val_domain(tt.set(2, 3).set_domain(dom), 0, cc, pos_cf);
                    affecte_tau_one_coef_val_domain(tt.set(3, 1).set_domain(dom), 0, cc, pos_cf);
                    affecte_tau_one_coef_val_domain(tt.set(3, 2).set_domain(dom), 0, cc, pos_cf);
                    affecte_tau_one_coef_val_domain(tt.set(3, 3).set_domain(dom), 0, cc, pos_cf);
                    found = true;
                }
                // Spherical coordinates and symetric
                if ((tt.get_basis().get_basis(dom) == SPHERICAL_BASIS) && (tt.get_n_comp() == 6)) {
                    affecte_tau_one_coef_val_domain(tt.set(1, 1).set_domain(dom), 0, cc, pos_cf);
                    affecte_tau_one_coef_val_domain(tt.set(1, 2).set_domain(dom), 1, cc, pos_cf);
                    affecte_tau_one_coef_val_domain(tt.set(1, 3).set_domain(dom), 1, cc, pos_cf);
                    affecte_tau_one_coef_val_domain(tt.set(2, 2).set_domain(dom), 2, cc, pos_cf);
                    affecte_tau_one_coef_val_domain(tt.set(2, 3).set_domain(dom), 2, cc, pos_cf);
                    affecte_tau_one_coef_val_domain(tt.set(3, 3).set_domain(dom), 2, cc, pos_cf);
                    found = true;
                }
                // Spherical coordinates and not symetric
                if ((tt.get_basis().get_basis(dom) == SPHERICAL_BASIS) && (tt.get_n_comp() == 9)) {
                    affecte_tau_one_coef_val_domain(tt.set(1, 1).set_domain(dom), 0, cc, pos_cf);
                    affecte_tau_one_coef_val_domain(tt.set(1, 2).set_domain(dom), 1, cc, pos_cf);
                    affecte_tau_one_coef_val_domain(tt.set(1, 3).set_domain(dom), 1, cc, pos_cf);
                    affecte_tau_one_coef_val_domain(tt.set(2, 1).set_domain(dom), 1, cc, pos_cf);
                    affecte_tau_one_coef_val_domain(tt.set(2, 2).set_domain(dom), 2, cc, pos_cf);
                    affecte_tau_one_coef_val_domain(tt.set(2, 3).set_domain(dom), 2, cc, pos_cf);
                    affecte_tau_one_coef_val_domain(tt.set(3, 1).set_domain(dom), 1, cc, pos_cf);
                    affecte_tau_one_coef_val_domain(tt.set(3, 2).set_domain(dom), 2, cc, pos_cf);
                    affecte_tau_one_coef_val_domain(tt.set(3, 3).set_domain(dom), 2, cc, pos_cf);
                    found = true;
                }
                if (!found) {
                    KADATH_THROW("Unknown type of 2-tensor Domain_shell_inner_adapted_nosym::affecte_tau_one_coef");
                }
            } break;
            default:
                cerr << "Valence " << val << " not implemented in Domain_shell_inner_adapted_nosym::affecte_tau" << endl;
                break;
        }
    }

    bool Domain_shell_inner_adapted_nosym::describe_tau_seed_block(
        const Tensor& tt, int dom, std::vector<TauSeedDescriptor>& descriptors) const
    {
        descriptors.clear();
        if (tt.get_space().get_domain(dom) != this)
            return false;
        const int valence = tt.get_valence();
        if (valence < 0 || valence > 2 ||
            (valence == 1 && tt.get_n_comp() != 3) ||
            (valence == 2 && tt.get_n_comp() != 6 && tt.get_n_comp() != 9)) {
            return false;
        }
        const int tensor_basis = tt.get_basis().get_basis(dom);
        if (valence > 0 && tensor_basis != CARTESIAN_BASIS &&
            tensor_basis != SPHERICAL_BASIS) {
            return false;
        }

        for (int component = 0; component < tt.get_n_comp(); ++component) {
            const Array<int> index(tt.indices(component));
            int mlim = 0;
            if (tensor_basis == SPHERICAL_BASIS) {
                for (int slot = 0; slot < valence; ++slot)
                    if (index(slot) != 1)
                        ++mlim;
            }
            detail::append_spheric_shell_tau_seed_component(
                nbr_coefs, tt(index)(dom), component, mlim, descriptors);
        }
        return descriptors.size() ==
            static_cast<std::size_t>(nbr_unknowns(tt, dom));
    }
} // namespace Kadath
