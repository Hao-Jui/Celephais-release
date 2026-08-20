// Axisymmetric NS_2D_XCTS -> 3D XCTS lifting support.
#pragma once

#include "For_Kadath/Kadath_point_h/kadath_adapted.hpp"
#include "For_Kadath/Kadath_point_h/kadath_spheric_homothetic.hpp"
#include "For_Kadath/Domain/spheric_adapted_nosym.hpp"
#include "For_Kadath/Array/exceptions.hpp"
#include "For_Kadath/Config/config_bco.hpp"
#include "For_Kadath/IO/be_file_source.hpp"
#include "Apps/Bco_utils/bco_io.hpp"
#include "Apps/Bco_utils/bco_regrid.hpp"
#include "Apps/Bco_utils/ns_bounds.hpp"
#include "Apps/Formalism/Shared/NS_2D_XCTS/scalar_import_batch.hpp"
#include "Apps/Policy/app_resolution.hpp"
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <sstream>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace Kadath {

enum class ns_2d_xcts_lift_field_layout
{
    gr
};

template <typename target_space_t, ns_2d_xcts_lift_field_layout field_layout = ns_2d_xcts_lift_field_layout::gr,
          typename config_t>
int ns_2d_xcts_lift_to_3d_as(config_t& bconfig, const int new_res, std::string outputfile,
                              bool use_config_vars = true, double tilt_angle = 0.);

template <typename config_t>
// Preserve the source radial split by default; rebuilding bounds from surface
// extrema introduces a regrid-level Mb residue.
int ns_2d_xcts_lift_to_3d(config_t& bconfig, const int new_res, std::string outputfile,
                           bool use_config_vars = true);

template <typename config_t>
int ns_2d_xcts_lift_to_3d(config_t& bconfig, std::string outputfile, bool use_config_vars = true)
{
    return ns_2d_xcts_lift_to_3d(bconfig, static_cast<int>(bconfig(BCO_RES)), std::move(outputfile),
                                 use_config_vars);
}

namespace ns_2d_xcts_lifting_detail {

Point cylindrical_point_from_3d(const Domain& domain, const Index& pos);

Point rotated_cylindrical_point_from_3d(const Domain& domain, const Index& pos,
                                        const Point& source_center, double tilt_angle);

void axisymmetric_import(Scalar& target, const Scalar& source);

void rotated_axisymmetric_import(Scalar& target, const Scalar& source,
                                 const Point& source_center, double tilt_angle);

void lift_azimuthal_shift(Vector& shift, const Scalar& brsint);

void lift_rotated_azimuthal_shift(Vector& shift, const Scalar& brsint,
                                  const Point& center, double tilt_angle);

} // namespace ns_2d_xcts_lifting_detail

} // namespace Kadath

#include "lifting.ipp"
