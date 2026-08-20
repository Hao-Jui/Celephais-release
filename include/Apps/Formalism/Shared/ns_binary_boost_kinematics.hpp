#pragma once

#include "For_Kadath/Config/config_enums.hpp"

#include <stdexcept>

namespace Kadath::ns_binary_boost
{
    struct CartesianVector {
        double x;
        double y;
        double z;
    };

    // The isolated NS coordinates are centred on the star, while the binary
    // coordinates place BCO1/BCO2 at -DIST/2 and +DIST/2 on the x axis.
    inline CartesianVector component_center(double separation, NODES bco)
    {
        switch (bco) {
            case BCO1:
                return {-separation / 2., 0., 0.};
            case BCO2:
                return {separation / 2., 0., 0.};
            default:
                throw std::invalid_argument("NS binary boost requires BCO1 or BCO2");
        }
    }

    // Constant term needed when the global helical generator is evaluated in
    // star-centred coordinates:
    //
    //   Omega zhat x (C + r) + Omega xaxis ey + zvel ez
    // = Omega zhat x r + local_translation(C).
    inline CartesianVector local_translation(CartesianVector center, double omega, double xaxis, double zvel)
    {
        return {
            -omega * center.y,
            omega * (center.x + xaxis),
            zvel,
        };
    }

    inline CartesianVector rotation_about_z(CartesianVector position)
    {
        return {-position.y, position.x, 0.};
    }
} // namespace Kadath::ns_binary_boost
