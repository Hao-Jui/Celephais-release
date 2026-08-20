#pragma once

namespace Kadath {
namespace ns_2d_xcts_lifting_detail {

template <typename target_space_t>
struct lift_target_traits;

template <>
struct lift_target_traits<Space_spheric_adapted> {
    using outer_adapted = Domain_shell_outer_adapted;
    using inner_adapted = Domain_shell_inner_adapted;
    static Dim_array resolution(const int radial_resolution) { return spheric_res(radial_resolution); }
};

template <>
struct lift_target_traits<Space_spheric_homothetic> {
    using outer_adapted = Domain_shell_outer_homothetic;
    using inner_adapted = Domain_shell_inner_homothetic;
    static Dim_array resolution(const int radial_resolution) { return spheric_res(radial_resolution); }
};

template <>
struct lift_target_traits<Space_spheric_adapted_nosym> {
    using outer_adapted = Domain_shell_outer_adapted_nosym;
    using inner_adapted = Domain_shell_inner_adapted_nosym;
    static Dim_array resolution(const int radial_resolution) { return spheric_res_nosym(radial_resolution); }
};

inline Point cylindrical_point_from_3d(const Domain& domain, const Index& pos)
{
    const double x = domain.get_cart(1)(pos);
    const double y = domain.get_cart(2)(pos);
    const double z = domain.get_cart(3)(pos);

    Point point(2);
    point.set(1) = std::sqrt(x * x + y * y);
    point.set(2) = z;
    return point;
}

inline Point rotated_cylindrical_point_from_3d(const Domain& domain, const Index& pos,
                                               const Point& source_center, const double tilt_angle)
{
    const double x = domain.get_cart(1)(pos) - source_center(1);
    const double y = domain.get_cart(2)(pos);
    const double z = domain.get_cart(3)(pos) - source_center(2);
    const double cosine = std::cos(tilt_angle);
    const double sine = std::sin(tilt_angle);

    // Pull back the target point with R_y(-tilt).  The 2D source axis is z',
    // while the target spin axis is n = (sin(tilt), 0, cos(tilt)).
    const double source_x = cosine * x - sine * z;
    const double source_z = sine * x + cosine * z;

    Point point(2);
    point.set(1) = source_center(1) + std::sqrt(source_x * source_x + y * y);
    point.set(2) = source_center(2) + source_z;
    return point;
}

inline void axisymmetric_import(Scalar& target, const Scalar& source)
{
    if (target.get_ndim() != 3 || source.get_ndim() != 2)
        KADATH_THROW("axisymmetric_import requires a 3D target and a 2D source");

    target.set_in_conf();
    target.allocate_conf();

    const int ndom = target.get_nbr_domains();
    for (int dom = 0; dom < ndom; ++dom) {
        const Domain& target_domain = *target.get_space().get_domain(dom);
        const Dim_array& points = target_domain.get_nbr_points();
        Index pos(points);

        do {
            if (dom == ndom - 1 && pos(0) == points(0) - 1)
                continue;

            const Point source_point = cylindrical_point_from_3d(target_domain, pos);
            target.set_domain(dom).set(pos) = source.val_point(source_point);
        } while (pos.inc());
    }
}

inline void rotated_axisymmetric_import(Scalar& target, const Scalar& source,
                                        const Point& source_center, const double tilt_angle)
{
    if (target.get_ndim() != 3 || source.get_ndim() != 2)
        KADATH_THROW("rotated_axisymmetric_import requires a 3D target and a 2D source");

    target.set_in_conf();
    target.allocate_conf();

    const int ndom = target.get_nbr_domains();
    for (int dom = 0; dom < ndom; ++dom) {
        const Domain& target_domain = *target.get_space().get_domain(dom);
        const Dim_array& points = target_domain.get_nbr_points();
        Index pos(points);

        do {
            if (dom == ndom - 1 && pos(0) == points(0) - 1)
                continue;

            const Point source_point =
                rotated_cylindrical_point_from_3d(target_domain, pos, source_center, tilt_angle);
            target.set_domain(dom).set(pos) = source.val_point(source_point);
        } while (pos.inc());
    }
}

inline void lift_azimuthal_shift(Vector& shift, const Scalar& brsint)
{
    if (shift.get_ndim() != 3 || brsint.get_ndim() != 3)
        KADATH_THROW("lift_azimuthal_shift requires 3D fields");

    // The 2D solver stores brsint = r sin(theta) beta^phi. In Cartesian
    // components, beta^x = -brsint sin(phi), beta^y = brsint cos(phi).
    shift.set(1) = -brsint.mult_sin_phi();
    shift.set(2) = brsint.mult_cos_phi();
    shift.set(3).annule_hard();
    shift.std_base();
}

inline void lift_rotated_azimuthal_shift(Vector& shift, const Scalar& brsint,
                                         const Point& center, const double tilt_angle)
{
    if (shift.get_ndim() != 3 || brsint.get_ndim() != 3)
        KADATH_THROW("lift_rotated_azimuthal_shift requires 3D fields");

    const double cosine = std::cos(tilt_angle);
    const double sine = std::sin(tilt_angle);
    for (int component = 1; component <= 3; ++component) {
        shift.set(component).annule_hard();
        shift.set(component).set_in_conf();
        shift.set(component).allocate_conf();
    }

    const int ndom = shift.get_space().get_nbr_domains();
    for (int dom = 0; dom < ndom; ++dom) {
        const Domain& domain = *shift.get_space().get_domain(dom);
        const Dim_array& points = domain.get_nbr_points();
        Index pos(points);
        do {
            if (dom == ndom - 1 && pos(0) == points(0) - 1)
                continue;

            const double x = domain.get_cart(1)(pos) - center(1);
            const double y = domain.get_cart(2)(pos);
            const double z = domain.get_cart(3)(pos) - center(3);
            const double source_x = cosine * x - sine * z;
            const double rho = std::sqrt(source_x * source_x + y * y);
            if (rho <= 1.e-14)
                continue;

            const double beta_phi_radius = brsint(dom)(pos) / rho;
            // beta = brsint * e_phi' = brsint/rho * (n x r),
            // n = (sin(tilt), 0, cos(tilt)).
            shift.set(1).set_domain(dom).set(pos) = -cosine * y * beta_phi_radius;
            shift.set(2).set_domain(dom).set(pos) = source_x * beta_phi_radius;
            shift.set(3).set_domain(dom).set(pos) = sine * y * beta_phi_radius;
        } while (pos.inc());
    }
    shift.std_base();
}

template <typename adapted_t>
inline void interp_rotated_adapted_mapping(const adapted_t* new_shell, const int old_outer_adapted_dom,
                                           const Scalar& old_radius_field, const Point& source_center,
                                           const double tilt_angle)
{
    Val_domain new_mapping = new_shell->get_radius();
    const Domain* old_shell = old_radius_field.get_space().get_domain(old_outer_adapted_dom);
    const double inner_radius = bco_utils::get_radius(old_shell, INNER_BC);
    const double cosine = std::cos(tilt_angle);
    const double sine = std::sin(tilt_angle);
    const Point target_center = new_shell->get_center();

    Index pos(new_shell->get_nbr_points());
    do {
        double x = new_shell->get_cart(1)(pos) - target_center(1);
        double y = new_shell->get_cart(2)(pos) - target_center(2);
        double z = new_shell->get_cart(3)(pos) - target_center(3);
        const double radius = std::sqrt(x * x + y * y + z * z);
        if (!std::isfinite(radius) || !std::isfinite(inner_radius) || radius <= 0. || inner_radius <= 0.)
            KADATH_THROW("interp_rotated_adapted_mapping encountered an invalid radial normalization");

        x *= inner_radius / radius;
        y *= inner_radius / radius;
        z *= inner_radius / radius;

        const double source_x = cosine * x - sine * z;
        const double source_z = sine * x + cosine * z;
        Point source_point(2);
        source_point.set(1) = source_center(1) + std::sqrt(source_x * source_x + y * y);
        source_point.set(2) = source_center(2) + source_z;

        const double mapped_radius = old_radius_field.val_point(source_point);
        if (!std::isfinite(mapped_radius))
            KADATH_THROW("interp_rotated_adapted_mapping imported a non-finite surface radius");
        new_mapping.set(pos) = mapped_radius;
    } while (pos.inc());

    new_mapping.std_base();
    new_shell->set_mapping(new_mapping);
}

} // namespace ns_2d_xcts_lifting_detail

template <typename target_space_t, ns_2d_xcts_lift_field_layout field_layout, typename config_t>
int ns_2d_xcts_lift_to_3d_as(config_t& bconfig, const int new_res, std::string outputfile,
                             bool use_config_vars, const double tilt_angle)
{
    using namespace ns_2d_xcts_lifting_detail;
    using target_traits = lift_target_traits<target_space_t>;

    if (!std::isfinite(tilt_angle))
        KADATH_THROW("ns_2d_xcts_lift_to_3d requires a finite tilt angle");

    const std::string in_spacefile = bconfig.space_filename();
    if (!std::filesystem::exists(in_spacefile)) {
        std::ostringstream oss;
        oss << "File: " << in_spacefile << " not found.\n\n";
        KADATH_THROW(oss.str());
    }

    validate_resolution(new_res);

    BeFileSource ff1(in_spacefile);
    Space_polar_adapted old_space(ff1);
    if (old_space.get_ndim() != 2)
        KADATH_THROW("ns_2d_xcts_lift_to_3d expects a Space_polar_adapted 2D source");

    Scalar old_conf(old_space, ff1);
    Scalar old_lapse(old_space, ff1);
    Scalar old_shift(old_space, ff1);
    Scalar old_logh(old_space, ff1);
    Scalar old_Omg(old_space, ff1);
    (void)old_Omg;
    std::unique_ptr<Scalar> old_scalar;
    std::unique_ptr<Scalar> old_sweight;
    if constexpr (field_layout != ns_2d_xcts_lift_field_layout::gr)
        old_scalar = std::make_unique<Scalar>(old_space, ff1);

    std::cout << "Resolution of 2D source space: "
              << old_space.get_domain(0)->get_nbr_points()(0) << " (r), "
              << old_space.get_domain(0)->get_nbr_points()(1) << " (theta)" << std::endl;

    const Domain_polar_shell_outer_adapted* old_outer_adapted =
        dynamic_cast<const Domain_polar_shell_outer_adapted*>(old_space.get_domain(1));
    if (old_outer_adapted == nullptr)
        KADATH_THROW("ns_2d_xcts_lift_to_3d could not identify the 2D outer adapted shell");

    Scalar old_space_radius(old_space);
    old_space_radius = 0.;
    for (int dom = 0; dom < old_space.get_nbr_domains(); ++dom)
        old_space_radius.set_domain(dom) = old_space.get_domain(dom)->get_radius();
    old_space_radius.set_domain(1) = old_outer_adapted->get_outer_radius();
    old_space_radius.std_base();

    auto [r_min, r_max] = bco_utils::get_rmin_rmax(old_space, 1);
    if (!use_config_vars) {
        bconfig.set(RIN) = 0.5 * r_min;
        bconfig.set(RMID) = r_min;
        bconfig.set(ROUT) = 1.5 * r_max;
    }

    bconfig.set(DIM) = 3;
    bconfig.set(BCO_RES) = new_res;
    bconfig.set_filename(outputfile);

    const int type_coloc = old_space.get_type_base();
    const Dim_array res = target_traits::resolution(new_res);

    Point old_center = old_space.get_domain(0)->get_center();
    Point center(3);
    center.set(1) = old_center(1);
    center.set(2) = 0.;
    center.set(3) = old_center(2);

    std::vector<double> bounds = bco_utils::make_NS_bounds(bconfig);
    bco_utils::print_bounds("3D lift bounds", bounds);

    target_space_t space(type_coloc, center, res, bounds);
    Base_tensor basis(space, CARTESIAN_BASIS);

    const typename target_traits::outer_adapted* new_outer_adapted =
        dynamic_cast<const typename target_traits::outer_adapted*>(space.get_domain(1));
    const typename target_traits::inner_adapted* new_inner_adapted =
        dynamic_cast<const typename target_traits::inner_adapted*>(space.get_domain(2));
    if (new_outer_adapted == nullptr || new_inner_adapted == nullptr)
        KADATH_THROW("ns_2d_xcts_lift_to_3d could not identify the 3D adapted shells");

    const bool aligned = std::abs(tilt_angle) <= 1.e-15;
    if (aligned) {
        bco_utils::interp_adapted_mapping(new_outer_adapted, 1, old_space_radius);
        bco_utils::interp_adapted_mapping(new_inner_adapted, 1, old_space_radius);
    } else {
        interp_rotated_adapted_mapping(new_outer_adapted, 1, old_space_radius, old_center, tilt_angle);
        interp_rotated_adapted_mapping(new_inner_adapted, 1, old_space_radius, old_center, tilt_angle);
    }

    Scalar conf(space);
    conf = 1.;
    conf.std_base();

    Scalar lapse(space);
    lapse = 1.;
    lapse.std_base();

    Scalar logh(space);
    logh.annule_hard();
    logh.std_base();

    Scalar brsint(space);
    brsint.annule_hard();
    // brsint is the physical azimuthal component r sin(theta) beta^phi before
    // it is projected into Cartesian components.
    if constexpr (std::is_same_v<target_space_t, Space_spheric_adapted_nosym>)
        brsint.std_base();
    else
        brsint.std_base_p_spher();

    std::unique_ptr<Scalar> scalar;
    std::unique_ptr<Scalar> sweight;
    auto make_lift_target = [&]() {
        auto target = std::make_unique<Scalar>(space);
        target->annule_hard();
        target->std_base();
        return target;
    };
    if (old_scalar)
        scalar = make_lift_target();
    if (old_sweight)
        sweight = make_lift_target();

    std::vector<ns_2d_xcts_import::scalar_import_field> lift_fields{
        ns_2d_xcts_import::import_field(conf, old_conf),
        ns_2d_xcts_import::import_field(lapse, old_lapse),
        ns_2d_xcts_import::import_field(logh, old_logh),
        ns_2d_xcts_import::import_field(brsint, old_shift),
    };
    if (old_scalar)
        lift_fields.push_back(ns_2d_xcts_import::import_field(*scalar, *old_scalar));
    if (old_sweight)
        lift_fields.push_back(ns_2d_xcts_import::import_field(*sweight, *old_sweight));

    const double tilt_cosine = std::cos(tilt_angle);
    const double tilt_sine = std::sin(tilt_angle);
    ns_2d_xcts_import::import_scalar_batch(
        lift_fields,
        [&](const std::span<const Val_domain> coordinates, const Index& pos) {
            const double target_x = coordinates[0](pos);
            const double target_y = coordinates[1](pos);
            const double target_z = coordinates[2](pos);

            Point source_point(2);
            if (aligned) {
                source_point.set(1) = std::sqrt(target_x * target_x + target_y * target_y);
                source_point.set(2) = target_z;
                return source_point;
            }

            const double x = target_x - old_center(1);
            const double z = target_z - old_center(2);
            const double source_x = tilt_cosine * x - tilt_sine * z;
            const double source_z = tilt_sine * x + tilt_cosine * z;
            source_point.set(1) = old_center(1) + std::sqrt(source_x * source_x + target_y * target_y);
            source_point.set(2) = old_center(2) + source_z;
            return source_point;
        });

    conf.std_base();
    lapse.std_base();
    logh.std_base();
    if constexpr (std::is_same_v<target_space_t, Space_spheric_adapted_nosym>)
        brsint.std_base();
    else
        brsint.std_base_p_spher();

    Vector shift(space, CON, basis);
    for (int i = 1; i <= 3; ++i)
        shift.set(i).annule_hard();
    if (aligned)
        lift_azimuthal_shift(shift, brsint);
    else
        lift_rotated_azimuthal_shift(shift, brsint, center, tilt_angle);

    if (scalar)
        scalar->std_base();
    if (sweight)
        sweight->std_base();

    std::cout << "Resolution of lifted 3D space: " << res(0) << " (r), " << res(1) << " (theta), "
              << res(2) << " (phi)" << std::endl;

    if constexpr (field_layout == ns_2d_xcts_lift_field_layout::gr) {
        bco_utils::save_to_file(space, bconfig, conf, lapse, shift, logh);
    }

    return EXIT_SUCCESS;
}

template <typename config_t>
int ns_2d_xcts_lift_to_3d(config_t& bconfig, const int new_res, std::string outputfile, bool use_config_vars)
{
    return ns_2d_xcts_lift_to_3d_as<Space_spheric_adapted>(
        bconfig, new_res, std::move(outputfile), use_config_vars);
}

} // namespace Kadath
