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
        bool nucleus_radial_weights(int basis, int radial_index,
                                    bool even_radial, double& radial_weight,
                                    double& normalization)
        {
            switch (basis) {
                case CHEB_EVEN:
                    if (!even_radial)
                        return false;
                    normalization = pow(-1, radial_index);
                    radial_weight = -normalization;
                    return true;
                case LEG_EVEN:
                    if (!even_radial)
                        return false;
                    normalization = 1.;
                    for (int t = 0; t < radial_index; ++t)
                        normalization *=
                            -double(2 * t + 1) / double(2 * t + 2);
                    radial_weight = -normalization;
                    return true;
                case CHEB_ODD:
                    if (even_radial)
                        return false;
                    normalization =
                        pow(-1, radial_index) * (2 * radial_index + 1);
                    radial_weight = -normalization;
                    return true;
                case LEG_ODD:
                    if (even_radial)
                        return false;
                    normalization = 1.;
                    for (int t = 0; t < radial_index; ++t)
                        normalization *=
                            -double(2 * t + 3) / double(2 * t + 2);
                    radial_weight = -normalization;
                    return true;
                default:
                    return false;
            }
        }

        bool append_nucleus_tau_seed_component(
            const Dim_array& nbr_coefs, const Val_domain& field, int component,
            std::vector<TauSeedDescriptor>& descriptors)
        {
            constexpr int kmin = 2;
            constexpr int llim = 0;
            const std::size_t phi_size =
                static_cast<std::size_t>(nbr_coefs(2));
            const std::size_t radial_stride =
                static_cast<std::size_t>(nbr_coefs(1)) * phi_size;

            for (int k = 0; k < nbr_coefs(2) - 1; ++k) {
                if (k == 1)
                    continue;
                const int theta_basis =
                    (*field.get_base().get_base_1d(1))(k);
                if (theta_basis != COS_EVEN && theta_basis != COS_ODD &&
                    theta_basis != SIN_EVEN && theta_basis != SIN_ODD) {
                    return false;
                }

                for (int j = 0; j < nbr_coefs(1); ++j) {
                    const int radial_basis =
                        (*field.get_base().get_base_1d(0))(j, k);
                    const bool even_radial =
                        theta_basis == COS_EVEN || theta_basis == SIN_EVEN;
                    double radial_weight = 0.;
                    double normalization = 0.;
                    if (!nucleus_radial_weights(
                            radial_basis, 0, even_radial, radial_weight,
                            normalization)) {
                        return false;
                    }

                    for (int i = 0; i < nbr_coefs(0); ++i) {
                        const int lquant =
                            (theta_basis == COS_EVEN ||
                             theta_basis == SIN_EVEN)
                                ? 2 * j
                                : 2 * j + 1;
                        bool include = false;
                        bool radial_galerkin = false;
                        bool double_galerkin = false;
                        int theta_anchor = 0;
                        double theta_weight = 0.;

                        switch (theta_basis) {
                            case COS_EVEN:
                                if (k < kmin && lquant <= llim) {
                                    include = true;
                                } else if (k < kmin) {
                                    include = i != 0;
                                    radial_galerkin = include;
                                } else {
                                    include = j != 0 && i != 0;
                                    double_galerkin = include;
                                    theta_weight = -1.;
                                }
                                break;
                            case COS_ODD:
                                if (j == nbr_coefs(1) - 1 ||
                                    i == nbr_coefs(0) - 1) {
                                    break;
                                }
                                if (k < kmin && lquant <= llim + 1) {
                                    include = true;
                                } else if (k < kmin) {
                                    include = i != 0;
                                    radial_galerkin = include;
                                } else {
                                    include = j != 0 && i != 0;
                                    double_galerkin = include;
                                    theta_weight = -1.;
                                }
                                break;
                            case SIN_EVEN:
                                if (j == 0 || j == nbr_coefs(1) - 1)
                                    break;
                                if (k < kmin + 2 && lquant <= llim) {
                                    include = true;
                                } else if (k < kmin + 2) {
                                    include = i != 0;
                                    radial_galerkin = include;
                                } else {
                                    include = j != 1 && i != 0;
                                    double_galerkin = include;
                                    theta_anchor = 1;
                                    theta_weight = -double(j);
                                }
                                break;
                            case SIN_ODD:
                                if (j == nbr_coefs(1) - 1 ||
                                    i == nbr_coefs(0) - 1) {
                                    break;
                                }
                                if (k < kmin + 2 && lquant <= llim + 1) {
                                    include = true;
                                } else if (k < kmin + 2) {
                                    include = i != 0;
                                    radial_galerkin = include;
                                } else {
                                    include = j != 0 && i != 0;
                                    double_galerkin = include;
                                    theta_weight = -double(2 * j + 1);
                                }
                                break;
                        }
                        if (!include)
                            continue;

                        const std::size_t row_offset =
                            static_cast<std::size_t>(j) * phi_size +
                            static_cast<std::size_t>(k);
                        TauSeedDescriptor descriptor;
                        descriptor.component = component;
                        descriptor.writes[0] = {
                            static_cast<std::size_t>(i) * radial_stride +
                                row_offset,
                            1.};
                        descriptor.write_count = 1;

                        if (radial_galerkin || double_galerkin) {
                            if (!nucleus_radial_weights(
                                    radial_basis, i, even_radial,
                                    radial_weight, normalization)) {
                                return false;
                            }
                            descriptor.writes[1] = {
                                row_offset, radial_weight};
                            descriptor.write_count = 2;
                        }
                        if (double_galerkin) {
                            const std::size_t anchor_offset =
                                static_cast<std::size_t>(theta_anchor) * phi_size +
                                static_cast<std::size_t>(k);
                            descriptor.writes[2] = {
                                static_cast<std::size_t>(i) * radial_stride +
                                    anchor_offset,
                                theta_weight};
                            descriptor.writes[3] = {
                                anchor_offset,
                                normalization * -theta_weight};
                            descriptor.write_count = 4;
                        }
                        descriptors.push_back(descriptor);
                    }
                }
            }
            return true;
        }
    } // namespace

    void Domain_nucleus::affecte_tau_one_coef_val_domain_vr(Val_domain& so, int cc, int& conte) const
    {

        so.allocate_coef();
        *so.cf = 0.;
        Index pos_cf(nbr_coefs);

        // Positions of the Galerkin basis
        Index pos_gal_t(nbr_coefs);
        Index pos_gal_r(nbr_coefs);
        Index pos_gal_rt(nbr_coefs);
        double fact_t, fact_r, fact_rt;

        bool found = false;

        // Case k=0 ; j<=1
        {
            pos_cf.set(2) = 0;
            [[maybe_unused]] int baset = (*so.get_base().bases_1d[1])(0);
            assert(baset == COS_EVEN);
            for (int j = 0; j < 2; j++) {
                [[maybe_unused]] int baser = (*so.get_base().bases_1d[0])(j, 0);
                pos_cf.set(1) = j;
                assert((baser == CHEB_ODD) || (baser == LEG_ODD));
                for (int i = 0; i < nbr_coefs(0) - 1; i++) {
                    pos_cf.set(0) = i;
                    if (conte == cc) {
                        found = true;
                        so.cf->set(pos_cf) = 1.;
                    }
                    conte++;
                }
            }
        }

        // Case k==0 ; j>1
        {
            pos_cf.set(2) = 0;
            [[maybe_unused]] int baset = (*so.get_base().bases_1d[1])(0);
            assert(baset == COS_EVEN);
            for (int j = 2; j < nbr_coefs(1); j++) {
                [[maybe_unused]] int baser = (*so.get_base().bases_1d[0])(j, 0);
                pos_cf.set(1) = j;
                assert((baser == CHEB_ODD) || (baser == LEG_ODD));
                for (int i = 1; i < nbr_coefs(0) - 1; i++) {
                    pos_cf.set(0) = i;
                    pos_gal_r = pos_cf;
                    pos_gal_r.set(0) = 0;
                    switch (baser) {
                        case CHEB_ODD:
                            fact_r = -(2 * i + 1) * pow(-1, i);
                            break;
                        case LEG_ODD: {
                            fact_r = -1.;
                            for (int t = 0; t < i; t++)
                                fact_r *= -double(2 * t + 3) / double(2 * t + 2);
                        } break;
                        default:
                            KADATH_THROW("Strange base in Domain_nucleus::affecte_tau_val_domain_cart");
                    }
                    if (conte == cc) {
                        found = true;
                        so.cf->set(pos_cf) = 1;
                        so.cf->set(pos_gal_r) += fact_r;
                    }
                    conte++;
                }
            }
        }

        // Next ones
        for (int k = 2; k < nbr_coefs(2) - 1; k++) {

            int mquant = (k % 2 == 0) ? int(k / 2) : int((k - 1) / 2);
            pos_cf.set(2) = k;
            [[maybe_unused]] int baset = (*so.get_base().bases_1d[1])(k);

            if (mquant % 2 == 0) {
                assert(baset = COS_EVEN);
                for (int j = 1; j < nbr_coefs(1); j++) {
                    [[maybe_unused]] int baser = (*so.get_base().bases_1d[0])(j, k);
                    pos_cf.set(1) = j;
                    assert((baser == CHEB_ODD) || (baser == LEG_ODD));
                    for (int i = 1; i < nbr_coefs(0) - 1; i++) {
                        pos_cf.set(0) = i;
                        pos_gal_r = pos_cf;
                        pos_gal_r.set(0) = 0;
                        pos_gal_t = pos_cf;
                        pos_gal_t.set(1) = 0;
                        pos_gal_rt = pos_cf;
                        pos_gal_rt.set(0) = 0;
                        pos_gal_rt.set(1) = 0;
                        switch (baser) {
                            case CHEB_ODD:
                                fact_r = -pow(-1, i) * (2 * i + 1);
                                fact_t = -1.;
                                fact_rt = pow(-1, i) * (2 * i + 1);
                                break;
                            case LEG_ODD: {
                                double l0 = 1;
                                for (int t = 0; t < i; t++)
                                    l0 *= -double(2 * t + 3) / double(2 * t + 2);
                                fact_r = -l0;
                                fact_t = -1.;
                                fact_rt = l0;
                            } break;
                            default:
                                KADATH_THROW("Strange base in Domain_nucleus::affecte_tau_val_domain_cart");
                        }

                        if (conte == cc) {
                            found = true;
                            so.cf->set(pos_cf) = 1.;
                            so.cf->set(pos_gal_r) += fact_r;
                            so.cf->set(pos_gal_t) += fact_t;
                            so.cf->set(pos_gal_rt) += fact_rt;
                        }
                        conte++;
                    }
                }
            }

            if (mquant % 2 == 1) {
                assert(baset = SIN_ODD);
                for (int j = 0; j < nbr_coefs(1) - 1; j++) {
                    [[maybe_unused]] int baser = (*so.get_base().bases_1d[0])(j, k);
                    pos_cf.set(1) = j;
                    assert((baser == CHEB_EVEN) || (baser == LEG_EVEN));
                    for (int i = 1; i < nbr_coefs(0); i++) {
                        pos_cf.set(0) = i;
                        pos_gal_r = pos_cf;
                        pos_gal_r.set(0) = 0;
                        switch (baser) {
                            case CHEB_EVEN:
                                fact_r = -pow(-1, i);
                                break;
                            case LEG_EVEN: {
                                fact_r = -1.;
                                for (int t = 0; t < i; t++)
                                    fact_r *= -double(2 * t + 1) / double(2 * t + 2);
                            } break;
                            default:
                                KADATH_THROW("Strange base in Domain_nucleus::affecte_tau_val_domain_vr");
                        }
                        if (conte == cc) {
                            found = true;
                            so.cf->set(pos_cf) = 1.;
                            so.cf->set(pos_gal_r) += fact_r;
                        }
                        conte++;
                    }
                }
            }
        }

        // If not found put to zero :
        if (!found)
            so.set_zero();
    }

    void Domain_nucleus::affecte_tau_one_coef_val_domain_vt(Val_domain& so, int cc, int& conte) const
    {

        bool found = false;
        so.allocate_coef();
        *so.cf = 0.;
        Index pos_cf(nbr_coefs);

        // Positions of the Galerkin basis
        Index pos_gal_t(nbr_coefs);
        Index pos_gal_r(nbr_coefs);
        Index pos_gal_rt(nbr_coefs);
        double fact_t, fact_r, fact_rt;

        // Case k=0
        {
            pos_cf.set(2) = 0;
            [[maybe_unused]] int baset = (*so.get_base().bases_1d[1])(0);
            assert(baset == SIN_EVEN);
            for (int j = 1; j < nbr_coefs(1) - 1; j++) {
                [[maybe_unused]] int baser = (*so.get_base().bases_1d[0])(j, 0);
                pos_cf.set(1) = j;
                assert((baser == CHEB_ODD) || (baser == LEG_ODD));
                for (int i = 1; i < nbr_coefs(0) - 1; i++) {
                    pos_cf.set(0) = i;
                    pos_gal_r = pos_cf;
                    pos_gal_r.set(0) = 0;
                    switch (baser) {
                        case CHEB_ODD:
                            fact_r = -(2 * i + 1) * pow(-1, i);
                            break;
                        case LEG_ODD: {
                            fact_r = -1.;
                            for (int t = 0; t < i; t++)
                                fact_r *= -double(2 * t + 3) / double(2 * t + 2);
                        } break;
                        default:
                            KADATH_THROW("Strange base in Domain_nucleus::affecte_tau_val_domain_vt");
                    }
                    if (conte == cc) {
                        found = true;
                        so.cf->set(pos_cf) = 1.;
                        so.cf->set(pos_gal_r) += fact_r;
                    }
                    conte++;
                }
            }
        }

        // Case k==2 ; j==0
        if (nbr_coefs(2) - 1 > 2) {
            pos_cf.set(2) = 2;
            [[maybe_unused]] int baset = (*so.get_base().bases_1d[1])(2);
            assert(baset == COS_ODD);
            [[maybe_unused]] int baser = (*so.get_base().bases_1d[0])(0, 2);
            pos_cf.set(1) = 0;
            assert((baser == CHEB_EVEN) || (baser == LEG_EVEN));
            for (int i = 0; i < nbr_coefs(0); i++) {
                pos_cf.set(0) = i;
                if (conte == cc) {
                    found = true;
                    so.cf->set(pos_cf) = 1.;
                }
                conte++;
            }
        }

        // Case k==2 ; j!=0
        if (nbr_coefs(2) - 1 > 2) {
            pos_cf.set(2) = 2;
            [[maybe_unused]] int baset = (*so.get_base().bases_1d[1])(2);
            assert(baset = COS_ODD);
            for (int j = 1; j < nbr_coefs(1) - 1; j++) {
                [[maybe_unused]] int baser = (*so.get_base().bases_1d[0])(j, 2);
                pos_cf.set(1) = j;
                assert((baser == CHEB_EVEN) || (baser == LEG_EVEN));
                for (int i = 1; i < nbr_coefs(0); i++) {
                    pos_cf.set(0) = i;
                    pos_gal_r = pos_cf;
                    pos_gal_r.set(0) = 0;
                    switch (baser) {
                        case CHEB_EVEN:
                            fact_r = -pow(-1, i);
                            break;
                        case LEG_EVEN: {
                            fact_r = -1.;
                            for (int t = 0; t < i; t++)
                                fact_r *= -double(2 * t + 1) / double(2 * t + 2);
                        } break;
                        default:
                            KADATH_THROW("Strange base in Domain_nucleus::affecte_tau_val_domain_vt");
                    }
                    if (conte == cc) {
                        found = true;
                        so.cf->set(pos_cf) = 1;
                        so.cf->set(pos_gal_r) += fact_r;
                    }
                    conte++;
                }
            }
        }

        // Case k==3 ; j==0
        if (nbr_coefs(2) - 1 > 3) {
            pos_cf.set(2) = 3;
            [[maybe_unused]] int baset = (*so.get_base().bases_1d[1])(3);
            assert(baset == COS_ODD);
            [[maybe_unused]] int baser = (*so.get_base().bases_1d[0])(0, 3);
            pos_cf.set(1) = 0;
            assert((baser == CHEB_EVEN) || (baser == LEG_EVEN));
            for (int i = 0; i < nbr_coefs(0); i++) {
                pos_cf.set(0) = i;
                if (conte == cc) {
                    found = true;
                    so.cf->set(pos_cf) = 1.;
                }
                conte++;
            }
        }

        // Case k==3 ; j!=0
        if (nbr_coefs(2) - 1 > 3) {
            pos_cf.set(2) = 3;
            [[maybe_unused]] int baset = (*so.get_base().bases_1d[1])(3);
            assert(baset = COS_ODD);
            for (int j = 1; j < nbr_coefs(1) - 1; j++) {
                [[maybe_unused]] int baser = (*so.get_base().bases_1d[0])(j, 3);
                pos_cf.set(1) = j;
                assert((baser == CHEB_EVEN) || (baser == LEG_EVEN));
                for (int i = 1; i < nbr_coefs(0); i++) {
                    pos_cf.set(0) = i;
                    pos_gal_r = pos_cf;
                    pos_gal_r.set(0) = 0;
                    switch (baser) {
                        case CHEB_EVEN:
                            fact_r = -pow(-1, i);
                            break;
                        case LEG_EVEN: {
                            fact_r = -1.;
                            for (int t = 0; t < i; t++)
                                fact_r *= -double(2 * t + 1) / double(2 * t + 2);
                        } break;
                        default:
                            KADATH_THROW("Strange base in Domain_nucleus::affecte_tau_val_domain_vt");
                    }
                    if (conte == cc) {
                        found = true;
                        so.cf->set(pos_cf) = 1;
                        so.cf->set(pos_gal_r) += fact_r;
                    }
                    conte++;
                }
            }
        }

        // Next ones
        for (int k = 4; k < nbr_coefs(2) - 1; k++) {

            int mquant = (k % 2 == 0) ? int(k / 2) : int((k - 1) / 2);
            pos_cf.set(2) = k;
            [[maybe_unused]] int baset = (*so.get_base().bases_1d[1])(k);

            if (mquant % 2 == 0) {
                assert(baset = SIN_EVEN);
                for (int j = 1; j < nbr_coefs(1) - 1; j++) {
                    [[maybe_unused]] int baser = (*so.get_base().bases_1d[0])(j, k);
                    pos_cf.set(1) = j;
                    assert((baser == CHEB_ODD) || (baser == LEG_ODD));
                    for (int i = 1; i < nbr_coefs(0) - 1; i++) {
                        pos_cf.set(0) = i;
                        pos_gal_r = pos_cf;
                        pos_gal_r.set(0) = 0;
                        switch (baser) {
                            case CHEB_ODD:
                                fact_r = -(2 * i + 1) * pow(-1, i);
                                break;
                            case LEG_ODD: {
                                fact_r = -1.;
                                for (int t = 0; t < i; t++)
                                    fact_r *= -double(2 * t + 3) / double(2 * t + 2);
                            } break;
                            default:
                                KADATH_THROW("Strange base in Domain_nucleus::affecte_tau_val_domain_vt");
                        }
                        if (conte == cc) {
                            found = true;
                            so.cf->set(pos_cf) = 1;
                            so.cf->set(pos_gal_r) += fact_r;
                        }
                        conte++;
                    }
                }
            }

            if (mquant % 2 == 1) {
                assert(baset = COS_ODD);
                for (int j = 1; j < nbr_coefs(1) - 1; j++) {
                    [[maybe_unused]] int baser = (*so.get_base().bases_1d[0])(j, k);
                    pos_cf.set(1) = j;
                    assert((baser == CHEB_EVEN) || (baser == LEG_EVEN));
                    for (int i = 1; i < nbr_coefs(0); i++) {
                        pos_cf.set(0) = i;
                        pos_gal_r = pos_cf;
                        pos_gal_r.set(0) = 0;
                        pos_gal_t = pos_cf;
                        pos_gal_t.set(1) = 0;
                        pos_gal_rt = pos_cf;
                        pos_gal_rt.set(0) = 0;
                        pos_gal_rt.set(1) = 0;
                        switch (baser) {
                            case CHEB_EVEN:
                                fact_r = -pow(-1, i);
                                fact_t = -1.;
                                fact_rt = pow(-1, i);
                                break;
                            case LEG_EVEN: {
                                double l0 = 1;
                                for (int t = 0; t < i; t++)
                                    l0 *= -double(2 * t + 1) / double(2 * t + 2);
                                fact_r = -l0;
                                fact_t = -1.;
                                fact_rt = l0;
                            } break;
                            default:
                                KADATH_THROW("Strange base in Domain_nucleus::affecte_tau_val_domain_vt");
                        }
                        if (conte == cc) {
                            found = true;
                            so.cf->set(pos_cf) = 1;
                            so.cf->set(pos_gal_r) += fact_r;
                            so.cf->set(pos_gal_t) += fact_t;
                            so.cf->set(pos_gal_rt) += fact_rt;
                        }
                        conte++;
                    }
                }
            }
        }

        // If not found put to zero :
        if (!found)
            so.set_zero();
    }

    void Domain_nucleus::affecte_tau_one_coef_val_domain_vp(Val_domain& so, int cc, int& conte) const
    {

        bool found = false;
        so.allocate_coef();
        *so.cf = 0.;
        Index pos_cf(nbr_coefs);

        // Positions of the Galerkin basis
        Index pos_gal_t(nbr_coefs);
        Index pos_gal_r(nbr_coefs);
        Index pos_gal_rt(nbr_coefs);
        double fact_t, fact_r, fact_rt;

        // k = 0 ; j==0
        {
            pos_cf.set(2) = 0;
            [[maybe_unused]] int baset = (*so.get_base().bases_1d[1])(0);
            assert(baset == SIN_ODD);
            [[maybe_unused]] int baser = (*so.get_base().bases_1d[0])(0, 0);
            pos_cf.set(1) = 0;
            assert((baser == CHEB_ODD) || (baser == LEG_ODD));
            for (int i = 0; i < nbr_coefs(0) - 1; i++) {
                pos_cf.set(0) = i;
                if (conte == cc) {
                    found = true;
                    so.cf->set(pos_cf) = 1;
                }
                conte++;
            }
        }

        // Case k==0 ; j!=0
        {
            pos_cf.set(2) = 0;
            [[maybe_unused]] int baset = (*so.get_base().bases_1d[1])(0);
            assert(baset = SIN_ODD);
            for (int j = 1; j < nbr_coefs(1) - 1; j++) {
                [[maybe_unused]] int baser = (*so.get_base().bases_1d[0])(j, 0);
                pos_cf.set(1) = j;
                assert((baser == CHEB_ODD) || (baser == LEG_ODD));
                for (int i = 1; i < nbr_coefs(0) - 1; i++) {
                    pos_cf.set(0) = i;
                    pos_gal_r = pos_cf;
                    pos_gal_r.set(0) = 0;
                    switch (baser) {
                        case CHEB_ODD:
                            fact_r = -(2 * i + 1) * pow(-1, i);
                            break;
                        case LEG_ODD: {
                            fact_r = -1.;
                            for (int t = 0; t < i; t++)
                                fact_r *= -double(2 * t + 3) / double(2 * t + 2);
                        } break;
                        default:
                            KADATH_THROW("Strange base in Domain_nucleus::affecte_tau_val_domain_vp");
                    }
                    if (conte == cc) {
                        found = true;
                        so.cf->set(pos_cf) = 1;
                        so.cf->set(pos_gal_r) += fact_r;
                    }
                    conte++;
                }
            }
        }

        // Case k==2 ; j==0
        if (nbr_coefs(2) - 1 > 2) {
            pos_cf.set(2) = 2;
            [[maybe_unused]] int baset = (*so.get_base().bases_1d[1])(2);
            assert(baset == COS_EVEN);
            [[maybe_unused]] int baser = (*so.get_base().bases_1d[0])(0, 2);
            pos_cf.set(1) = 0;
            assert((baser == CHEB_EVEN) || (baser == LEG_EVEN));
            for (int i = 0; i < nbr_coefs(0); i++) {
                pos_cf.set(0) = i;
                if (conte == cc) {
                    found = true;
                    so.cf->set(pos_cf) = 1.;
                }
                conte++;
            }
        }

        // Case k==2 ; j!=0
        if (nbr_coefs(2) - 1 > 2) {
            pos_cf.set(2) = 2;
            [[maybe_unused]] int baset = (*so.get_base().bases_1d[1])(2);
            assert(baset = COS_EVEN);
            for (int j = 1; j < nbr_coefs(1); j++) {
                [[maybe_unused]] int baser = (*so.get_base().bases_1d[0])(j, 2);
                pos_cf.set(1) = j;
                assert((baser == CHEB_EVEN) || (baser == LEG_EVEN));
                for (int i = 1; i < nbr_coefs(0); i++) {
                    pos_cf.set(0) = i;
                    pos_gal_r = pos_cf;
                    pos_gal_r.set(0) = 0;
                    switch (baser) {
                        case CHEB_EVEN:
                            fact_r = -pow(-1, i);
                            break;
                        case LEG_EVEN: {
                            fact_r = -1.;
                            for (int t = 0; t < i; t++)
                                fact_r *= -double(2 * t + 1) / double(2 * t + 2);
                        } break;
                        default:
                            KADATH_THROW("Strange base in Domain_nucleus::affecte_tau_val_domain_vp");
                    }

                    if (conte == cc) {
                        found = true;
                        so.cf->set(pos_cf) = 1;
                        so.cf->set(pos_gal_r) += fact_r;
                    }
                    conte++;
                }
            }
        }

        // Case k==3 ; j==0
        if (nbr_coefs(2) - 1 > 3) {
            pos_cf.set(2) = 3;
            [[maybe_unused]] int baset = (*so.get_base().bases_1d[1])(3);
            assert(baset == COS_EVEN);
            [[maybe_unused]] int baser = (*so.get_base().bases_1d[0])(0, 3);
            pos_cf.set(1) = 0;
            assert((baser == CHEB_EVEN) || (baser == LEG_EVEN));
            for (int i = 0; i < nbr_coefs(0); i++) {
                pos_cf.set(0) = i;
                if (conte == cc) {
                    found = true;
                    so.cf->set(pos_cf) = 1.;
                }
                conte++;
            }
        }

        // Case k==3 ; j!=0
        if (nbr_coefs(2) - 1 > 3) {
            pos_cf.set(2) = 3;
            [[maybe_unused]] int baset = (*so.get_base().bases_1d[1])(3);
            assert(baset = COS_EVEN);
            for (int j = 1; j < nbr_coefs(1); j++) {
                [[maybe_unused]] int baser = (*so.get_base().bases_1d[0])(j, 3);
                pos_cf.set(1) = j;
                assert((baser == CHEB_EVEN) || (baser == LEG_EVEN));
                for (int i = 1; i < nbr_coefs(0); i++) {
                    pos_cf.set(0) = i;
                    pos_gal_r = pos_cf;
                    pos_gal_r.set(0) = 0;
                    switch (baser) {
                        case CHEB_EVEN:
                            fact_r = -pow(-1, i);
                            break;
                        case LEG_EVEN: {
                            fact_r = -1.;
                            for (int t = 0; t < i; t++)
                                fact_r *= -double(2 * t + 1) / double(2 * t + 2);
                        } break;
                        default:
                            KADATH_THROW("Strange base in Domain_nucleus::affecte_tau_val_domain_vp");
                    }
                    if (conte == cc) {
                        found = true;
                        so.cf->set(pos_cf) = 1;
                        so.cf->set(pos_gal_r) += fact_r;
                    }
                    conte++;
                }
            }
        }

        // k = 4 ; j==0
        if (nbr_coefs(2) - 1 > 4) {
            pos_cf.set(2) = 4;
            [[maybe_unused]] int baset = (*so.get_base().bases_1d[1])(4);
            assert(baset == SIN_ODD);
            [[maybe_unused]] int baser = (*so.get_base().bases_1d[0])(0, 4);
            pos_cf.set(1) = 0;
            assert((baser == CHEB_ODD) || (baser == LEG_ODD));
            for (int i = 0; i < nbr_coefs(0) - 1; i++) {
                pos_cf.set(0) = i;
                if (conte == cc) {
                    found = true;
                    so.cf->set(pos_cf) = 1;
                }
                conte++;
            }
        }

        // Case k==4 ; j!=0
        if (nbr_coefs(2) - 1 > 4) {
            pos_cf.set(2) = 4;
            [[maybe_unused]] int baset = (*so.get_base().bases_1d[1])(4);
            assert(baset = SIN_ODD);
            for (int j = 1; j < nbr_coefs(1) - 1; j++) {
                [[maybe_unused]] int baser = (*so.get_base().bases_1d[0])(j, 4);
                pos_cf.set(1) = j;
                assert((baser == CHEB_ODD) || (baser == LEG_ODD));
                for (int i = 1; i < nbr_coefs(0) - 1; i++) {
                    pos_cf.set(0) = i;
                    pos_gal_r = pos_cf;
                    pos_gal_r.set(0) = 0;
                    switch (baser) {
                        case CHEB_ODD:
                            fact_r = -(2 * i + 1) * pow(-1, i);
                            break;
                        case LEG_ODD: {
                            fact_r = -1.;
                            for (int t = 0; t < i; t++)
                                fact_r *= -double(2 * t + 3) / double(2 * t + 2);
                        } break;
                        default:
                            KADATH_THROW("Strange base in Domain_nucleus::affecte_tau_val_domain_vp");
                    }
                    if (conte == cc) {
                        found = true;
                        so.cf->set(pos_cf) = 1;
                        so.cf->set(pos_gal_r) += fact_r;
                    }
                    conte++;
                }
            }
        }

        // k = 5 ; j==0
        if (nbr_coefs(2) - 1 > 5) {
            pos_cf.set(2) = 5;
            [[maybe_unused]] int baset = (*so.get_base().bases_1d[1])(5);
            assert(baset == SIN_ODD);
            [[maybe_unused]] int baser = (*so.get_base().bases_1d[0])(0, 5);
            pos_cf.set(1) = 0;
            assert((baser == CHEB_ODD) || (baser == LEG_ODD));
            for (int i = 0; i < nbr_coefs(0) - 1; i++) {
                pos_cf.set(0) = i;
                if (conte == cc) {
                    found = true;
                    so.cf->set(pos_cf) = 1;
                }
                conte++;
            }
        }

        // Case k==5 ; j!=0
        if (nbr_coefs(2) - 1 > 5) {
            pos_cf.set(2) = 5;
            [[maybe_unused]] int baset = (*so.get_base().bases_1d[1])(5);
            assert(baset = SIN_ODD);
            for (int j = 1; j < nbr_coefs(1) - 1; j++) {
                [[maybe_unused]] int baser = (*so.get_base().bases_1d[0])(j, 5);
                pos_cf.set(1) = j;
                assert((baser == CHEB_ODD) || (baser == LEG_ODD));
                for (int i = 1; i < nbr_coefs(0) - 1; i++) {
                    pos_cf.set(0) = i;
                    pos_gal_r = pos_cf;
                    pos_gal_r.set(0) = 0;
                    switch (baser) {
                        case CHEB_ODD:
                            fact_r = -(2 * i + 1) * pow(-1, i);
                            break;
                        case LEG_ODD: {
                            fact_r = -1.;
                            for (int t = 0; t < i; t++)
                                fact_r *= -double(2 * t + 3) / double(2 * t + 2);
                        } break;
                        default:
                            KADATH_THROW("Strange base in Domain_nucleus::affecte_tau_val_domain_vp");
                    }
                    if (conte == cc) {
                        found = true;
                        so.cf->set(pos_cf) = 1;
                        so.cf->set(pos_gal_r) += fact_r;
                    }
                    conte++;
                }
            }
        }

        // Next ones
        for (int k = 6; k < nbr_coefs(2) - 1; k++) {

            int mquant = (k % 2 == 0) ? int(k / 2) : int((k - 1) / 2);
            pos_cf.set(2) = k;
            [[maybe_unused]] int baset = (*so.get_base().bases_1d[1])(k);

            if (mquant % 2 == 0) {

                assert(baset = SIN_ODD);
                for (int j = 0; j < nbr_coefs(1) - 1; j++) {
                    [[maybe_unused]] int baser = (*so.get_base().bases_1d[0])(j, 0);
                    pos_cf.set(1) = j;
                    assert((baser == CHEB_ODD) || (baser == LEG_ODD));
                    for (int i = 1; i < nbr_coefs(0) - 1; i++) {
                        pos_cf.set(0) = i;
                        pos_gal_r = pos_cf;
                        pos_gal_r.set(0) = 0;
                        switch (baser) {
                            case CHEB_ODD:
                                fact_r = -(2 * i + 1) * pow(-1, i);
                                break;
                            case LEG_ODD: {
                                fact_r = -1.;
                                for (int t = 0; t < i; t++)
                                    fact_r *= -double(2 * t + 3) / double(2 * t + 2);
                            } break;
                            default:
                                KADATH_THROW("Strange base in Domain_nucleus::affecte_tau_val_domain_vp");
                        }
                        if (conte == cc) {
                            found = true;
                            so.cf->set(pos_cf) = 1;
                            so.cf->set(pos_gal_r) += fact_r;
                        }
                        conte++;
                    }
                }
            }

            if (mquant % 2 == 1) {
                assert(baset = COS_EVEN);
                for (int j = 1; j < nbr_coefs(1); j++) {
                    [[maybe_unused]] int baser = (*so.get_base().bases_1d[0])(j, k);
                    pos_cf.set(1) = j;
                    assert((baser == CHEB_EVEN) || (baser == LEG_EVEN));
                    for (int i = 1; i < nbr_coefs(0); i++) {
                        pos_cf.set(0) = i;
                        pos_gal_r = pos_cf;
                        pos_gal_r.set(0) = 0;
                        pos_gal_t = pos_cf;
                        pos_gal_t.set(1) = 0;
                        pos_gal_rt = pos_cf;
                        pos_gal_rt.set(0) = 0;
                        pos_gal_rt.set(1) = 0;
                        switch (baser) {
                            case CHEB_EVEN:
                                fact_r = -pow(-1, i);
                                fact_t = -1.;
                                fact_rt = pow(-1, i);
                                break;
                            case LEG_EVEN: {
                                double l0 = 1;
                                for (int t = 0; t < i; t++)
                                    l0 *= -double(2 * t + 1) / double(2 * t + 2);
                                fact_r = -l0;
                                fact_t = -1.;
                                fact_rt = l0;
                            } break;
                            default:
                                KADATH_THROW("Strange base in Domain_nucleus::affecte_tau_val_domain_vp");
                        }
                        if (conte == cc) {
                            found = true;
                            so.cf->set(pos_cf) = 1;
                            so.cf->set(pos_gal_r) += fact_r;
                            so.cf->set(pos_gal_t) += fact_t;
                            so.cf->set(pos_gal_rt) += fact_rt;
                        }
                        conte++;
                    }
                }
            }
        }

        // If not found put to zero :
        if (!found)
            so.set_zero();
    }

    void Domain_nucleus::affecte_tau_one_coef_val_domain(Val_domain& so, int mlim, int llim, int cc, int& conte) const
    {

        int kmin = 2 * mlim + 2;
        int lquant;

        so.is_zero = false;
        so.allocate_coef();
        *so.cf = 0.;
        Index pos_cf(nbr_coefs);

        bool found = false;

        // Positions of the Galerkin basis
        Index pos_gal_t(nbr_coefs);
        Index pos_gal_r(nbr_coefs);
        Index pos_gal_rt(nbr_coefs);
        double fact_t, fact_r, fact_rt;

        // Loop on phi :
        for (int k = 0; k < nbr_coefs(2) - 1; k++)
            if (k != 1) {
                pos_cf.set(2) = k;
                // Loop on theta
                [[maybe_unused]] int baset = (*so.get_base().bases_1d[1])(k);
                for (int j = 0; j < nbr_coefs(1); j++) {
                    [[maybe_unused]] int baser = (*so.get_base().bases_1d[0])(j, k);
                    pos_cf.set(1) = j;
                    // Loop on r :
                    for (int i = 0; i < nbr_coefs(0); i++) {
                        pos_cf.set(0) = i;
                        switch (baset) {
                            case COS_EVEN:
                                lquant = 2 * j;
                                // No galerkin :
                                if ((k < kmin) && (lquant <= llim)) {
                                    if (conte == cc) {
                                        found = true;
                                        so.cf->set(pos_cf) = 1.;
                                    }
                                    conte++;
                                } else if (k < kmin) {
                                    if (i != 0) {
                                        if (conte == cc) {
                                            found = true;
                                            // Galerkin base in r only
                                            pos_gal_r = pos_cf;
                                            pos_gal_r.set(0) = 0;
                                            switch (baser) {
                                                case CHEB_EVEN:
                                                    fact_r = -pow(-1, i);
                                                    break;
                                                case LEG_EVEN: {
                                                    fact_r = -1.;
                                                    for (int t = 0; t < i; t++)
                                                        fact_r *= -double(2 * t + 1) / double(2 * t + 2);
                                                } break;
                                                default:
                                                    KADATH_THROW("Strange base in Domain_nucleus::affecte_one_coef_val_domain");
                                            }
                                            so.cf->set(pos_cf) = 1;
                                            so.cf->set(pos_gal_r) += fact_r;
                                        }
                                        conte++;
                                    }
                                } else if ((j != 0) && (i != 0)) {
                                    if (conte == cc) {
                                        found = true;
                                        // Need to use two_dimensional Galerkin basis (aouch !)
                                        pos_gal_r = pos_cf;
                                        pos_gal_r.set(0) = 0;
                                        pos_gal_t = pos_cf;
                                        pos_gal_t.set(1) = 0;
                                        pos_gal_rt = pos_cf;
                                        pos_gal_rt.set(0) = 0;
                                        pos_gal_rt.set(1) = 0;
                                        switch (baser) {
                                            case CHEB_EVEN:
                                                fact_r = -pow(-1, i);
                                                fact_t = -1.;
                                                fact_rt = pow(-1, i);
                                                break;
                                            case LEG_EVEN: {
                                                double l0 = 1;
                                                for (int t = 0; t < i; t++)
                                                    l0 *= -double(2 * t + 1) / double(2 * t + 2);
                                                fact_r = -l0;
                                                fact_t = -1.;
                                                fact_rt = l0;
                                            } break;
                                            default:
                                                KADATH_THROW("Strange base in Domain_nucleus::affecte_one_coef_val_domain");
                                        }
                                        so.cf->set(pos_cf) = 1.;
                                        so.cf->set(pos_gal_r) = fact_r;
                                        so.cf->set(pos_gal_t) = fact_t;
                                        so.cf->set(pos_gal_rt) = fact_rt;
                                    }
                                    conte++;
                                }
                                break;
                            case COS_ODD:
                                lquant = 2 * j + 1;
                                if ((j != nbr_coefs(1) - 1) && (i != nbr_coefs(0) - 1)) {
                                    if ((k < kmin) && (lquant <= llim + 1)) {
                                        if (conte == cc) {
                                            found = true;
                                            so.cf->set(pos_cf) = 1.;
                                        }
                                        conte++;
                                    } else {
                                        if ((k < kmin) && (i != 0)) {
                                            if (conte == cc) {
                                                found = true;
                                                pos_gal_r = pos_cf;
                                                pos_gal_r.set(0) = 0;
                                                switch (baser) {
                                                    case CHEB_ODD:
                                                        fact_r = -(2 * i + 1) * pow(-1, i);
                                                        break;
                                                    case LEG_ODD: {
                                                        fact_r = -1.;
                                                        for (int t = 0; t < i; t++)
                                                            fact_r *= -double(2 * t + 3) / double(2 * t + 2);
                                                    } break;
                                                    default:
                                                        KADATH_THROW("Strange base in " "Domain_nucleus::affecte_one_coef_val_domain");
                                                }

                                                so.cf->set(pos_cf) = 1.;
                                                so.cf->set(pos_gal_r) = fact_r;
                                            }
                                            conte++;
                                        } else if ((j != 0) && (i != 0)) {
                                            if (conte == cc) {
                                                found = true;
                                                // Need to use two_dimensional Galerkin basis (aouch !)
                                                pos_gal_r = pos_cf;
                                                pos_gal_r.set(0) = 0;
                                                pos_gal_t = pos_cf;
                                                pos_gal_t.set(1) = 0;
                                                pos_gal_rt = pos_cf;
                                                pos_gal_rt.set(0) = 0;
                                                pos_gal_rt.set(1) = 0;
                                                switch (baser) {
                                                    case CHEB_ODD:
                                                        fact_r = -pow(-1, i) * (2 * i + 1);
                                                        fact_t = -1.;
                                                        fact_rt = pow(-1, i) * (2 * i + 1);
                                                        break;
                                                    case LEG_ODD: {
                                                        double l0 = 1;
                                                        for (int t = 0; t < i; t++)
                                                            l0 *= -double(2 * t + 3) / double(2 * t + 2);
                                                        fact_r = -l0;
                                                        fact_t = -1.;
                                                        fact_rt = l0;
                                                    } break;
                                                    default:
                                                        KADATH_THROW("Strange base in " "Domain_nucleus::affecte_one_coef_val_domain");
                                                }
                                                so.cf->set(pos_cf) = 1.;
                                                so.cf->set(pos_gal_r) = fact_r;
                                                so.cf->set(pos_gal_t) = fact_t;
                                                so.cf->set(pos_gal_rt) = fact_rt;
                                            }
                                            conte++;
                                        }
                                    }
                                }
                                break;
                            case SIN_EVEN:
                                lquant = 2 * j;
                                if ((j != 0) && (j != nbr_coefs(1) - 1)) {
                                    if ((k < kmin + 2) && (lquant <= llim)) {
                                        if (conte == cc) {
                                            found = true;
                                            so.cf->set(pos_cf) = 1.;
                                        }
                                        conte++;
                                    } else {
                                        if ((k < kmin + 2) && (i != 0)) {
                                            // Galerkin base in r only
                                            if (conte == cc) {
                                                found = true;
                                                pos_gal_r = pos_cf;
                                                pos_gal_r.set(0) = 0;
                                                switch (baser) {
                                                    case CHEB_EVEN:
                                                        fact_r = -pow(-1, i);
                                                        break;
                                                    case LEG_EVEN: {
                                                        fact_r = -1.;
                                                        for (int t = 0; t < i; t++)
                                                            fact_r *= -double(2 * t + 1) / double(2 * t + 2);
                                                    } break;
                                                    default:
                                                        KADATH_THROW("Strange base in " "Domain_nucleus_::affecte_one_coef_val_domain");
                                                }
                                                so.cf->set(pos_cf) = 1.;
                                                so.cf->set(pos_gal_r) = fact_r;
                                            }
                                            conte++;
                                        } else {
                                            // Double Galerkin
                                            if ((j != 1) && (i != 0)) {

                                                if (conte == cc) {
                                                    found = true;
                                                    // Need to use two_dimensional Galerkin basis (aouch !)
                                                    pos_gal_r = pos_cf;
                                                    pos_gal_r.set(0) = 0;
                                                    pos_gal_t = pos_cf;
                                                    pos_gal_t.set(1) = 1;
                                                    pos_gal_rt = pos_cf;
                                                    pos_gal_rt.set(0) = 0;
                                                    pos_gal_rt.set(1) = 1;
                                                    switch (baser) {
                                                        case CHEB_EVEN:
                                                            fact_r = -pow(-1, i);
                                                            fact_t = -j;
                                                            fact_rt = pow(-1, i) * j;
                                                            break;
                                                        case LEG_EVEN: {
                                                            double l0 = 1;
                                                            for (int t = 0; t < i; t++)
                                                                l0 *= -double(2 * t + 1) / double(2 * t + 2);
                                                            fact_r = -l0;
                                                            fact_t = -j;
                                                            fact_rt = l0 * j;
                                                        } break;
                                                        default:
                                                            KADATH_THROW("Strange base in " "Domain_nucleus::affecte_one_domain_val_domain");
                                                    }
                                                    so.cf->set(pos_cf) = 1;
                                                    so.cf->set(pos_gal_r) = fact_r;
                                                    so.cf->set(pos_gal_t) = fact_t;
                                                    so.cf->set(pos_gal_rt) = fact_rt;
                                                }
                                                conte++;
                                            }
                                        }
                                    }
                                }
                                break;
                            case SIN_ODD:
                                lquant = 2 * j + 1;
                                if ((j != nbr_coefs(1) - 1) && (i != nbr_coefs(0) - 1)) {
                                    if ((k < kmin + 2) && (lquant <= llim + 1)) {
                                        if (conte == cc) {
                                            found = true;
                                            so.cf->set(pos_cf) = 1.;
                                        }
                                        conte++;
                                    } else {
                                        if ((k < kmin + 2) && (i != 0)) {
                                            if (conte == cc) {
                                                found = true;
                                                pos_gal_r = pos_cf;
                                                pos_gal_r.set(0) = 0;
                                                switch (baser) {
                                                    case CHEB_ODD:
                                                        fact_r = -(2 * i + 1) * pow(-1, i);
                                                        break;
                                                    case LEG_ODD: {
                                                        fact_r = -1.;
                                                        for (int t = 0; t < i; t++)
                                                            fact_r *= -double(2 * t + 3) / double(2 * t + 2);
                                                    } break;
                                                    default:
                                                        KADATH_THROW("Strange base in " "Domain_nucleus::affecte_one_coef_val_domain");
                                                }

                                                so.cf->set(pos_cf) = 1.;
                                                so.cf->set(pos_gal_r) = fact_r;
                                            }
                                            conte++;
                                        } else if ((j != 0) && (i != 0)) {
                                            if (conte == cc) {
                                                found = true;
                                                // Need to use two_dimensional Galerkin basis (aouch !)
                                                pos_gal_r = pos_cf;
                                                pos_gal_r.set(0) = 0;
                                                pos_gal_t = pos_cf;
                                                pos_gal_t.set(1) = 0;
                                                pos_gal_rt = pos_cf;
                                                pos_gal_rt.set(0) = 0;
                                                pos_gal_rt.set(1) = 0;
                                                switch (baser) {
                                                    case CHEB_ODD:
                                                        fact_r = -pow(-1, i) * (2 * i + 1);
                                                        fact_t = -(2 * j + 1);
                                                        fact_rt = pow(-1, i) * (2 * i + 1) * (2 * j + 1);
                                                        break;
                                                    case LEG_ODD: {
                                                        double l0 = 1;
                                                        for (int t = 0; t < i; t++)
                                                            l0 *= -double(2 * t + 3) / double(2 * t + 2);
                                                        fact_r = -l0;
                                                        fact_t = -(2 * j + 1);
                                                        fact_rt = l0 * (2 * j + 1);
                                                    } break;
                                                    default:
                                                        KADATH_THROW("Strange base in " "Domain_nucleus::affecte_one_coef_val_domain");
                                                }
                                                so.cf->set(pos_cf) = 1.;
                                                so.cf->set(pos_gal_r) = fact_r;
                                                so.cf->set(pos_gal_t) = fact_t;
                                                so.cf->set(pos_gal_rt) = fact_rt;
                                            }
                                            conte++;
                                        }
                                    }
                                }
                                break;
                            default:
                                KADATH_THROW("Unknow theta basis in Domain_nucleus::affecte_coef_val_domain");
                        }
                    }
                }
            }
        // If not found put to zero :
        if (!found)
            so.set_zero();
    }

    void Domain_nucleus::affecte_tau_one_coef(Tensor& tt, int dom, int cc, int& pos_cf) const
    {

        // Check right domain
        assert(tt.get_space().get_domain(dom) == this);

        int val = tt.get_valence();
        switch (val) {
            case 0:
                affecte_tau_one_coef_val_domain(tt.set().set_domain(dom), 0, 0, cc, pos_cf);
                break;
            case 1: {
                bool found = false;
                // Cartesian basis
                if (tt.get_basis().get_basis(dom) == CARTESIAN_BASIS) {
                    affecte_tau_one_coef_val_domain(tt.set(1).set_domain(dom), 0, 0, cc, pos_cf);
                    affecte_tau_one_coef_val_domain(tt.set(2).set_domain(dom), 0, 0, cc, pos_cf);
                    affecte_tau_one_coef_val_domain(tt.set(3).set_domain(dom), 0, 0, cc, pos_cf);
                    found = true;
                }
                // Spherical coordinates
                if (tt.get_basis().get_basis(dom) == SPHERICAL_BASIS) {
                    affecte_tau_one_coef_val_domain_vr(tt.set(1).set_domain(dom), cc, pos_cf);
                    affecte_tau_one_coef_val_domain_vt(tt.set(2).set_domain(dom), cc, pos_cf);
                    affecte_tau_one_coef_val_domain_vp(tt.set(3).set_domain(dom), cc, pos_cf);
                    found = true;
                }
                if (!found) {
                    KADATH_THROW("Unknown type of vector Domain_nucleus::affecte_tau_one_coef");
                }
            } break;
            case 2: {
                bool found = false;
                // Cartesian basis and symetric
                if ((tt.get_basis().get_basis(dom) == CARTESIAN_BASIS) && (tt.get_n_comp() == 6)) {
                    affecte_tau_one_coef_val_domain(tt.set(1, 1).set_domain(dom), 0, 0, cc, pos_cf);
                    affecte_tau_one_coef_val_domain(tt.set(1, 2).set_domain(dom), 0, 0, cc, pos_cf);
                    affecte_tau_one_coef_val_domain(tt.set(1, 3).set_domain(dom), 0, 0, cc, pos_cf);
                    affecte_tau_one_coef_val_domain(tt.set(2, 2).set_domain(dom), 0, 0, cc, pos_cf);
                    affecte_tau_one_coef_val_domain(tt.set(2, 3).set_domain(dom), 0, 0, cc, pos_cf);
                    affecte_tau_one_coef_val_domain(tt.set(3, 3).set_domain(dom), 0, 0, cc, pos_cf);
                    found = true;
                }
                // Cartesian basis and not symetric
                if ((tt.get_basis().get_basis(dom) == CARTESIAN_BASIS) && (tt.get_n_comp() == 9)) {
                    affecte_tau_one_coef_val_domain(tt.set(1, 1).set_domain(dom), 0, 0, cc, pos_cf);
                    affecte_tau_one_coef_val_domain(tt.set(1, 2).set_domain(dom), 0, 0, cc, pos_cf);
                    affecte_tau_one_coef_val_domain(tt.set(1, 3).set_domain(dom), 0, 0, cc, pos_cf);
                    affecte_tau_one_coef_val_domain(tt.set(2, 1).set_domain(dom), 0, 0, cc, pos_cf);
                    affecte_tau_one_coef_val_domain(tt.set(2, 2).set_domain(dom), 0, 0, cc, pos_cf);
                    affecte_tau_one_coef_val_domain(tt.set(2, 3).set_domain(dom), 0, 0, cc, pos_cf);
                    affecte_tau_one_coef_val_domain(tt.set(3, 1).set_domain(dom), 0, 0, cc, pos_cf);
                    affecte_tau_one_coef_val_domain(tt.set(3, 2).set_domain(dom), 0, 0, cc, pos_cf);
                    affecte_tau_one_coef_val_domain(tt.set(3, 3).set_domain(dom), 0, 0, cc, pos_cf);
                    found = true;
                }
                if (!found) {
                    KADATH_THROW("Unknown type of 2-tensor Domain_nucleus::affecte_tau_one_coef");
                }
            } break;
            default:
                cerr << "Valence " << val << " not implemented in Domain_nucleus::affecte_tau" << endl;
                break;
        }
    }

    bool Domain_nucleus::describe_tau_seed_block(
        const Tensor& tt, int dom,
        std::vector<TauSeedDescriptor>& descriptors) const
    {
        descriptors.clear();
        if (tt.get_space().get_domain(dom) != this)
            return false;

        const int valence = tt.get_valence();
        if (valence < 0 || valence > 2 ||
            (valence == 1 && tt.get_n_comp() != 3) ||
            (valence == 2 && tt.get_n_comp() != 6 &&
             tt.get_n_comp() != 9)) {
            return false;
        }
        if (valence > 0 &&
            tt.get_basis().get_basis(dom) != CARTESIAN_BASIS) {
            return false;
        }

        for (int component = 0; component < tt.get_n_comp(); ++component) {
            const Array<int> index(tt.indices(component));
            if (!append_nucleus_tau_seed_component(
                    nbr_coefs, tt(index)(dom), component, descriptors)) {
                descriptors.clear();
                return false;
            }
        }
        if (descriptors.size() !=
            static_cast<std::size_t>(nbr_unknowns(tt, dom))) {
            descriptors.clear();
            return false;
        }
        return true;
    }
} // namespace Kadath
