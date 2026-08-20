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

#include "For_Kadath/Domain/spheric.hpp"
#include "For_Kadath/Scalar/scalar.hpp"
#include "For_Kadath/Tensor/tensor.hpp"
namespace Kadath
{
    namespace
    {
        void append_shell_tau_seed_component(
            const Dim_array& nbr_coefs, const Val_domain& field, int component,
            int mlim, std::vector<TauSeedDescriptor>& descriptors)
        {
            const int kmin = 2 * mlim + 2;
            const std::size_t phi_size = static_cast<std::size_t>(nbr_coefs(2));
            const std::size_t radial_stride =
                static_cast<std::size_t>(nbr_coefs(1)) * phi_size;

            for (int k = 0; k < nbr_coefs(2) - 1; ++k) {
                if (k == 1)
                    continue;
                const int baset = (*field.get_base().get_base_1d(1))(k);
                for (int j = 0; j < nbr_coefs(1); ++j) {
                    bool true_theta = true;
                    switch (baset) {
                        case COS_EVEN:
                            true_theta = (j != 0) || (k < kmin);
                            break;
                        case COS_ODD:
                            true_theta = j != nbr_coefs(1) - 1 &&
                                         ((j != 0) || (k < kmin));
                            break;
                        case SIN_EVEN:
                            true_theta = j != 0 && j != nbr_coefs(1) - 1 &&
                                         ((j != 1) || (k < kmin + 2));
                            break;
                        case SIN_ODD:
                            true_theta = j != nbr_coefs(1) - 1 &&
                                         ((j != 0) || (k < kmin + 2));
                            break;
                        default:
                            KADATH_THROW(
                                "Unknown theta basis in Domain_shell::describe_tau_seed_block");
                    }
                    if (!true_theta)
                        continue;

                    for (int i = 0; i < nbr_coefs(0); ++i) {
                        TauSeedDescriptor descriptor;
                        descriptor.component = component;
                        descriptor.writes[0] = {
                            static_cast<std::size_t>(i) * radial_stride +
                                static_cast<std::size_t>(j) * phi_size +
                                static_cast<std::size_t>(k),
                            1.};
                        descriptor.write_count = 1;

                        if ((baset == COS_EVEN || baset == COS_ODD) && k >= kmin) {
                            descriptor.writes[1] = {
                                static_cast<std::size_t>(i) * radial_stride +
                                    static_cast<std::size_t>(k),
                                -1.};
                            descriptor.write_count = 2;
                        } else if (baset == SIN_EVEN && k >= kmin + 2) {
                            descriptor.writes[1] = {
                                static_cast<std::size_t>(i) * radial_stride + phi_size +
                                    static_cast<std::size_t>(k),
                                -double(j)};
                            descriptor.write_count = 2;
                        } else if (baset == SIN_ODD && k >= kmin + 2) {
                            descriptor.writes[1] = {
                                static_cast<std::size_t>(i) * radial_stride +
                                    static_cast<std::size_t>(k),
                                -double(2 * j + 1)};
                            descriptor.write_count = 2;
                        }
                        descriptors.push_back(descriptor);
                    }
                }
            }
        }
    } // namespace

    void Domain_shell::affecte_tau_one_coef_val_domain(Val_domain& so, int mlim, int cc, int& conte) const
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
                    switch (baset) {
                        case COS_EVEN:
                            if ((j == 0) && (k >= kmin))
                                true_tet = false;
                            break;
                        case COS_ODD:
                            if ((j == nbr_coefs(1) - 1) || ((j == 0) && (k >= kmin)))
                                true_tet = false;
                            break;
                        case SIN_EVEN:
                            if (((j == 1) && (k >= kmin + 2)) || (j == 0) || (j == nbr_coefs(1) - 1))
                                true_tet = false;
                            break;
                        case SIN_ODD:
                            if (((j == 0) && (k >= kmin + 2)) || (j == nbr_coefs(1) - 1))
                                true_tet = false;
                            break;
                        default:
                            KADATH_THROW("Unknow theta basis in Domain_shell::affecte_one_coef_val_domain");
                    }

                    if (true_tet)
                        for (int i = 0; i < nbr_coefs(0); i++) {
                            pos_cf.set(0) = i;
                            if (conte == cc) {
                                so.cf->set(pos_cf) = 1;
                                found = true;
                                // regularity ??
                                if ((baset == COS_EVEN) || (baset == COS_ODD))
                                    if (k >= kmin) {
                                        pos_cf.set(1) = 0;
                                        so.cf->set(pos_cf) = -1;
                                    }

                                if (baset == SIN_EVEN)
                                    if (k >= kmin + 2) {
                                        pos_cf.set(1) = 1;
                                        so.cf->set(pos_cf) = -j;
                                    }
                                if (baset == SIN_ODD)
                                    if (k >= kmin + 2) {
                                        pos_cf.set(1) = 0;
                                        so.cf->set(pos_cf) = -(2 * j + 1);
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

    void Domain_shell::affecte_tau_one_coef(Tensor& tt, int dom, int cc, int& pos_cf) const
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
                // MTZ coordinates
                if (tt.get_basis().get_basis(dom) == MTZ_BASIS) {
                    affecte_tau_one_coef_val_domain(tt.set(1).set_domain(dom), 0, cc, pos_cf);
                    affecte_tau_one_coef_val_domain(tt.set(2).set_domain(dom), 1, cc, pos_cf);
                    affecte_tau_one_coef_val_domain(tt.set(3).set_domain(dom), 1, cc, pos_cf);
                    found = true;
                }
                if (!found) {
                    KADATH_THROW("Unknown type of vector Domain_shell::affecte_tau_one_coef");
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
                // MTZ coordinates and symetric
                if ((tt.get_basis().get_basis(dom) == MTZ_BASIS) && (tt.get_n_comp() == 6)) {
                    affecte_tau_one_coef_val_domain(tt.set(1, 1).set_domain(dom), 0, cc, pos_cf);
                    affecte_tau_one_coef_val_domain(tt.set(1, 2).set_domain(dom), 1, cc, pos_cf);
                    affecte_tau_one_coef_val_domain(tt.set(1, 3).set_domain(dom), 1, cc, pos_cf);
                    affecte_tau_one_coef_val_domain(tt.set(2, 2).set_domain(dom), 2, cc, pos_cf);
                    affecte_tau_one_coef_val_domain(tt.set(2, 3).set_domain(dom), 2, cc, pos_cf);
                    affecte_tau_one_coef_val_domain(tt.set(3, 3).set_domain(dom), 2, cc, pos_cf);
                    found = true;
                }
                // MTZ coordinates and not symetric
                if ((tt.get_basis().get_basis(dom) == MTZ_BASIS) && (tt.get_n_comp() == 9)) {
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
                    KADATH_THROW("Unknown type of 2-tensor Domain_shell::affecte_tau_one_coef");
                }
            } break;
            default:
                cerr << "Valence " << val << " not implemented in Domain_shell::affecte_tau" << endl;
                break;
        }
    }

    bool Domain_shell::describe_tau_seed_block(
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

        int tensor_basis = CARTESIAN_BASIS;
        if (valence > 0) {
            tensor_basis = tt.get_basis().get_basis(dom);
            if (tensor_basis != CARTESIAN_BASIS &&
                tensor_basis != SPHERICAL_BASIS && tensor_basis != MTZ_BASIS) {
                return false;
            }
        }

        for (int component = 0; component < tt.get_n_comp(); ++component) {
            const Array<int> index(tt.indices(component));
            int mlim = 0;
            if (tensor_basis != CARTESIAN_BASIS) {
                for (int rank = 0; rank < valence; ++rank) {
                    if (index(rank) != 1)
                        ++mlim;
                }
            }
            append_shell_tau_seed_component(
                nbr_coefs, tt(index)(dom), component, mlim, descriptors);
        }
        return descriptors.size() ==
            static_cast<std::size_t>(nbr_unknowns(tt, dom));
    }
} // namespace Kadath
