#pragma once

#include "For_Kadath/Base_spectral/base_spectral.hpp"
#include "For_Kadath/Array/exceptions.hpp"
#include "For_Kadath/Space/space.hpp"
#include "For_Kadath/Val_domain/val_domain.hpp"

namespace Kadath
{
namespace detail
{
    // The no-sym spherical domains store the full-theta COS/SIN basis in one
    // coefficient array. Regularity still follows the z-symmetric/antisymmetric
    // Galerkin rules separately on the even and odd theta subsequences.
    inline bool spheric_nosym_true_theta_coef(int baset, int j, int k, int kmin, int ntheta)
    {
        if (j == ntheta - 1)
            return false;

        switch (baset) {
            case COS:
                return (k < kmin) || ((j != 0) && (j != 1));
            case SIN:
                if (j == 0)
                    return false;
                return (k < kmin + 2) || ((j != 1) && (j != 2));
            default:
                KADATH_THROW("Unknown theta basis in spheric_nosym_true_theta_coef");
        }
    }

    inline bool spheric_nosym_uses_theta_galerkin(int baset, int k, int kmin)
    {
        switch (baset) {
            case COS:
                return k >= kmin;
            case SIN:
                return k >= kmin + 2;
            default:
                KADATH_THROW("Unknown theta basis in spheric_nosym_uses_theta_galerkin");
        }
    }

    inline int spheric_nosym_theta_anchor(int baset, int j)
    {
        switch (baset) {
            case COS:
                return (j % 2 == 0) ? 0 : 1;
            case SIN:
                return (j % 2 == 0) ? 2 : 1;
            default:
                KADATH_THROW("Unknown theta basis in spheric_nosym_theta_anchor");
        }
    }

    inline double spheric_nosym_basis_anchor_weight(int baset, int j)
    {
        switch (baset) {
            case COS:
                return 1.;
            case SIN:
                return (j % 2 == 0) ? 0.5 * j : double(j);
            default:
                KADATH_THROW("Unknown theta basis in spheric_nosym_basis_anchor_weight");
        }
    }

    inline double spheric_nosym_export_anchor_weight(int baset, int j)
    {
        switch (baset) {
            case COS:
                return (j % 2 == 0) ? 2. : 1.;
            case SIN:
                return (j % 2 == 0) ? 0.5 * j : double(j);
            default:
                KADATH_THROW("Unknown theta basis in spheric_nosym_export_anchor_weight");
        }
    }

    inline void append_spheric_shell_tau_seed_component(
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
                if (!spheric_nosym_true_theta_coef(
                        baset, j, k, kmin, nbr_coefs(1))) {
                    continue;
                }
                for (int i = 0; i < nbr_coefs(0); ++i) {
                    TauSeedDescriptor descriptor;
                    descriptor.component = component;
                    descriptor.writes[0] = {
                        static_cast<std::size_t>(i) * radial_stride +
                            static_cast<std::size_t>(j) * phi_size +
                            static_cast<std::size_t>(k),
                        1.0};
                    descriptor.write_count = 1;
                    if (spheric_nosym_uses_theta_galerkin(baset, k, kmin)) {
                        descriptor.writes[1] = {
                            static_cast<std::size_t>(i) * radial_stride +
                                static_cast<std::size_t>(
                                    spheric_nosym_theta_anchor(baset, j)) * phi_size +
                                static_cast<std::size_t>(k),
                            -spheric_nosym_basis_anchor_weight(baset, j)};
                        descriptor.write_count = 2;
                    }
                    descriptors.push_back(descriptor);
                }
            }
        }
    }
} // namespace detail
} // namespace Kadath
