#pragma once

#include "For_Kadath/Val_domain/der_abs_lane_batch.hpp"
#include "For_Kadath/Val_domain/val_domain.hpp"

#include <algorithm>
#include <array>
#include <optional>
#include <utility>

namespace Kadath::bispheric_nosym_detail
{
    inline void do_der_abs_from_der_var_lanes(
        DerAbsLaneBatch& batch,
        const Val_domain& detadx,
        const Val_domain& detady,
        const Val_domain& detadz,
        const Val_domain& dchidx,
        const Val_domain& dchidy,
        const Val_domain& dchidz,
        const Val_domain& dphidy,
        const Val_domain& dphidz,
        int eta_axis,
        int chi_axis)
    {
        constexpr std::size_t tile_width = 4;
        for (std::size_t tile_begin = 0; tile_begin < batch.lane_count(); tile_begin += tile_width) {
            const std::size_t tile_size = std::min(tile_width, batch.lane_count() - tile_begin);
            std::array<std::optional<Val_domain>, tile_width> out_x;
            std::array<std::optional<Val_domain>, tile_width> auchi_y;
            std::array<std::optional<Val_domain>, tile_width> part_y_phi;
            std::array<std::optional<Val_domain>, tile_width> out_y;
            std::array<std::optional<Val_domain>, tile_width> auchi_z;
            std::array<std::optional<Val_domain>, tile_width> part_z_phi;
            std::array<std::optional<Val_domain>, tile_width> out_z;

            for (std::size_t offset = 0; offset < tile_size; ++offset)
                out_x[offset].emplace(batch.der_var(tile_begin + offset, eta_axis) * detadx
                                      + batch.der_var(tile_begin + offset, chi_axis) * dchidx);
            for (std::size_t offset = 0; offset < tile_size; ++offset)
                auchi_y[offset].emplace(batch.der_var(tile_begin + offset, 2) * dphidy);
            for (std::size_t offset = 0; offset < tile_size; ++offset) {
                part_y_phi[offset].emplace(auchi_y[offset]->div_sin_chi());
                auchi_y[offset].reset();
            }
            for (std::size_t offset = 0; offset < tile_size; ++offset) {
                out_y[offset].emplace(batch.der_var(tile_begin + offset, eta_axis) * detady
                                      + batch.der_var(tile_begin + offset, chi_axis) * dchidy
                                      + *part_y_phi[offset]);
                part_y_phi[offset].reset();
            }
            for (std::size_t offset = 0; offset < tile_size; ++offset)
                auchi_z[offset].emplace(batch.der_var(tile_begin + offset, 2) * dphidz);
            for (std::size_t offset = 0; offset < tile_size; ++offset) {
                part_z_phi[offset].emplace(auchi_z[offset]->div_sin_chi());
                auchi_z[offset].reset();
            }
            for (std::size_t offset = 0; offset < tile_size; ++offset) {
                out_z[offset].emplace(batch.der_var(tile_begin + offset, eta_axis) * detadz
                                      + batch.der_var(tile_begin + offset, chi_axis) * dchidz
                                      + *part_z_phi[offset]);
                part_z_phi[offset].reset();
            }

            for (std::size_t offset = 0; offset < tile_size; ++offset) {
                batch.set_der_abs(tile_begin + offset, 0, std::move(*out_x[offset]));
                batch.set_der_abs(tile_begin + offset, 1, std::move(*out_y[offset]));
                batch.set_der_abs(tile_begin + offset, 2, std::move(*out_z[offset]));
            }
        }
    }
} // namespace Kadath::bispheric_nosym_detail
