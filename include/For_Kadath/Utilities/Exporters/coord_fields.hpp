/*
 * This file is part of the KADATH library.
 * Copyright (C) 2019, Samuel Tootle
 *                     <tootle@th.physik.uni-frankfurt.de>
 * Copyright (C) 2019, Ludwig Jens Papenfort
 *                      <papenfort@th.physik.uni-frankfurt.de>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

/*
 * Modifications (Celephais):
 *   2026-06-16  Modified for the Celephais tree; see
 *               PATCHES-KADATH-UPSTREAM.md and LICENSE_SOURCE_AUDIT.tsv.
 *   2026-08-06  RAII/span modernization.
 */

#pragma once
#include "For_Kadath/Kadath_point_h/kadath.hpp"
#include "For_Kadath/Config/config_enums.hpp"
#include <memory>
#include <optional>
using namespace Kadath;

/**
 * \addtogroup domain
 * @{
 */

constexpr int NUM_VECTORS_V = 11;
enum class coord_vector { GLOBAL_ROT, BCO1_ROTx, BCO1_ROTz, BCO2_ROTx, BCO2_ROTz, EX, EY, EZ, S_BCO1, S_BCO2, S_INF };
constexpr int NUM_SCALARS_V = 2;
enum class coord_scalar { R_BCO1 = 0, R_BCO2 };

using vec_ary_t = std::array<std::optional<Vector>, NUM_VECTORS_V>;
using scalar_ary_t = std::array<std::optional<Scalar>, NUM_SCALARS_V>;

struct coord_field_binding_t
{
    explicit coord_field_binding_t(System_of_eqs& syst_in) : syst(&syst_in)
    {
        vector_constants.fill(-1);
        scalar_constants.fill(-1);
    }

    System_of_eqs* syst;
    std::array<int, NUM_VECTORS_V> vector_constants;
    std::array<int, NUM_SCALARS_V> scalar_constants;
};

/**
 * gen_cv_names()
 * in order to make sure fields are updated during each solver iteration,
 * we need to maintain a catalogue of the field names.  They must be in
 * the system of equations with the same name!
 * We generate this using a function such that the array used in the update functions
 * is const
 *
 * @return array of pre-defined names that correspond to enum coord_vector
 */
inline std::array<std::string, NUM_VECTORS_V> gen_cv_names()
{
    std::array<std::string, NUM_VECTORS_V> cv_names;
    cv_names[to_int(coord_vector::GLOBAL_ROT)] = "mg^i";
    cv_names[to_int(coord_vector::BCO1_ROTx)] = "mmx^i";
    cv_names[to_int(coord_vector::BCO1_ROTz)] = "mmz^i";
    cv_names[to_int(coord_vector::BCO2_ROTx)] = "mpx^i";
    cv_names[to_int(coord_vector::BCO2_ROTz)] = "mpz^i";
    cv_names[to_int(coord_vector::EX)] = "ex^i";
    cv_names[to_int(coord_vector::EY)] = "ey^i";
    cv_names[to_int(coord_vector::EZ)] = "ez^i";
    cv_names[to_int(coord_vector::S_BCO1)] = "sm^i";
    cv_names[to_int(coord_vector::S_BCO2)] = "sp^i";
    cv_names[to_int(coord_vector::S_INF)] = "einf^i";
    return cv_names;
}
const std::array<std::string, NUM_VECTORS_V> cv_names = gen_cv_names();

/**
 * gen_cs_names()
 * same as gen_cv_names only for Scalar fields
 *
 * return cs_names array of predefined names that correspond to enum coord_scalar
 */
inline std::array<std::string, NUM_SCALARS_V> gen_cs_names()
{
    std::array<std::string, NUM_SCALARS_V> cs_names;
    cs_names[to_int(coord_scalar::R_BCO1)] = "rm";
    cs_names[to_int(coord_scalar::R_BCO2)] = "rp";
    return cs_names;
}
const std::array<std::string, NUM_SCALARS_V> cs_names = gen_cs_names();

// Forward declarations
template <typename space_t> class CoordFields;

/**
 * update_fields
 *
 * this function updates the constant fields used in the generation of
 * various ID solvers.  Since the fields are a function of the cartesian basis,
 * they must be updated every iteration as the coordinates change.  However, since
 * they are a "constant" in the solver, they must be updated manually.  This includes
 * the fields themselves and their value within the system of equation. See
 * update_field()
 *
 * @tparam space_t type of computation space
 * @param [input] cf_generator: CoordFields object for a given space
 * @param [input] coord_vectors: ref to array of pointers to vector fields
 * @param [input] coord_scalars: ref to array of pointers to scalar fields
 * @param [input] xo: origin of the global fields
 * @param [input] xc1: origin of BCO1
 * @param [input] xc2: origin of BCO2
 * @param [input] syst optional pointer to the system of equations.
 */
template <typename space_t>
void update_fields(CoordFields<space_t> const& cf_generator, vec_ary_t& coord_vectors, scalar_ary_t& coord_scalars,
                   const double xo, const double xc1, const double xc2, System_of_eqs* syst = nullptr);

// Helper function to avoid declaring unnecessary coord_scalar arrays
template <typename space_t>
void update_fields(CoordFields<space_t> const& cf_generator, vec_ary_t& coord_vectors, scalar_ary_t&& coord_scalars,
                   const double xo, const double xc1, const double xc2, System_of_eqs* syst = nullptr);

template <typename space_t>
void update_fields(CoordFields<space_t> const& cf_generator, vec_ary_t& coord_vectors, scalar_ary_t& coord_scalars,
                   const double xo, const double xc1, const double xc2, const coord_field_binding_t& binding);

template <typename space_t>
void update_fields(CoordFields<space_t> const& cf_generator, vec_ary_t& coord_vectors, scalar_ary_t&& coord_scalars,
                   const double xo, const double xc1, const double xc2, const coord_field_binding_t& binding);
/**
 * update_fields_co
 *
 * this function is an abbreviated version of updates_fields
 * in order to update only the fields required for an isolated compact object
 *
 * @tparam space_t type of computation space
 * @param [input] cf_generator: CoordFields object for a given space
 * @param [input] coord_vectors: ref to array of coordinate vector field pointers
 * @param [input] coord_scalars: pointer to the needed radius scalar field
 * @param [input] xo: origin of the global fields
 * @param [input] syst: optional pointer to the system of equations.
 */
template <typename space_t>
void update_fields_co(CoordFields<space_t> const& cf_generator, vec_ary_t& coord_vectors, scalar_ary_t& coord_scalars,
                      const double xo, System_of_eqs* syst = nullptr);

// Helper function to avoid declaring unnecessary coord_scalar arrays
template <typename space_t>
void update_fields_co(CoordFields<space_t> const& cf_generator, vec_ary_t& coord_vectors, scalar_ary_t&& coord_scalars,
                      const double xo, System_of_eqs* syst = nullptr);

template <typename space_t>
void update_fields_co(CoordFields<space_t> const& cf_generator, vec_ary_t& coord_vectors, scalar_ary_t& coord_scalars,
                      const double xo, const coord_field_binding_t& binding);

template <typename space_t>
void update_fields_co(CoordFields<space_t> const& cf_generator, vec_ary_t& coord_vectors, scalar_ary_t&& coord_scalars,
                      const double xo, const coord_field_binding_t& binding);

/**
 * update_field
 *
 * this function updates a field in the given system of equations
 * based on the statically definded names: see cv_names and cs_names
 *
 * @param [input] syst: optional pointer to the system of equations.
 * @param [input] dom: dom of the field to be updated
 * @param [input] so the value to set 'n' to in the system of equations
 * @param [input] so: value to apply
 */
bool update_field(System_of_eqs& syst, int dom, const char* n, Tensor& so);

coord_field_binding_t bind_coordinate_fields(System_of_eqs& syst, const vec_ary_t& coord_vectors,
                                             const scalar_ary_t& coord_scalars = {});

void update_bound_coordinate_fields(const coord_field_binding_t& binding, vec_ary_t& coord_vectors,
                                    scalar_ary_t& coord_scalars);

void finalize_coordinate_field_refresh(System_of_eqs& syst, vec_ary_t& coord_vectors,
                                       scalar_ary_t& coord_scalars,
                                       const coord_field_binding_t* existing_binding = nullptr);

/**
 * default_binary_vector_ary
 *
 * generates the default array of Vector fields needed for XCTS
 * binary initial data.
 *
 * @tparam space_t type of computation space
 * @param [input] space: numerical spoace
 * @return array of Vector fields
 */
template <class space_t> vec_ary_t default_binary_vector_ary(space_t& space);

/**
 * default_co_vector_ary
 *
 * generates the default array of Vector fields needed for XCTS
 * isolated compact object initial data.
 *
 * @tparam space_t type of computation space
 * @param [input] space: numerical spoace
 * @return array of Vector fields
 */
template <class space_t> vec_ary_t default_co_vector_ary(space_t& space);

template <class space_t>
void activate_coordinate_vector(vec_ary_t& coord_vectors, space_t& space, coord_vector field);

/**
 * @class CoordFields
 *
 * generates and updates necessary coordinate fields
 * @tparam space_t Space type (e.g. Space_bin_bh)
 */
template <typename space_t> class CoordFields
{
  private:
    space_t const& space;      ///< store reference to computational space
    Kadath::Base_tensor basis; ///< store basis

  public:
    CoordFields(space_t const& space) : space(space), basis(space, CARTESIAN_BASIS) {}

    /**
     * CoordFields::cart
     *
     * generate and return vector field of the cartesian coordinates
     *
     * @param [input] shift_x: coordinate shift in the x direction
     * @param [input] shift_y: coordinate shift in the y direction
     * @param [input] shift_z: coordinate shift in the z direction
     * @return Vector field containing the shifted cartesian coordinates
     */
    template <int ind_t = CON> Kadath::Vector cart(double shift_x = 0., double shift_y = 0., double shift_z = 0.) const;

    /**
     * CoordFields::radius
     *
     * generate and return a shifted radius field
     *
     * @param [input] shift_x: coordinate shift in the x direction
     * @param [input] shift_y: coordinate shift in the y direction
     * @param [input] shift_z: coordinate shift in the z direction
     * @return Scalar field containing the shifted radius field
     */
    Kadath::Scalar radius(double shift_x = 0., double shift_y = 0., double shift_z = 0.) const;

    Kadath::Scalar radius_from_cart(const Kadath::Vector& coords) const;

    /**
     * CoordFields::rot_z
     *
     * generate and return a shifted rotation field about the z axis
     *
     * @param [input] shift_x: coordinate shift in the x direction
     * @param [input] shift_y: coordinate shift in the y direction
     * @param [input] shift_z: coordinate shift in the z direction
     * @return Vector field containing the shifted rotation field
     */
    template <int ind_t = CON>
    Kadath::Vector rot_z(double shift_x = 0., double shift_y = 0., double shift_z = 0.) const;

    template <int ind_t = CON> Kadath::Vector rot_z_from_cart(const Kadath::Vector& coords) const;

    /**
     * CoordFields::rot_x
     *
     * generate and return a shifted rotation field about the z axis
     *
     * @param [input] shift_x: coordinate shift in the x direction
     * @param [input] shift_y: coordinate shift in the y direction
     * @param [input] shift_z: coordinate shift in the z direction
     * @return Vector field containing the shifted rotation field
     */

    template <int ind_t = CON>
    Kadath::Vector rot_x(double shift_x = 0., double shift_y = 0., double shift_z = 0.) const;

    template <int ind_t = CON> Kadath::Vector rot_x_from_cart(const Kadath::Vector& coords) const;
    /**
     * CoordFields::e_rad
     *
     * generate and return a shifted, radial pointing, unit vector field
     *
     * @param [input] shift_x: coordinate shift in the x direction
     * @param [input] shift_y: coordinate shift in the y direction
     * @param [input] shift_z: coordinate shift in the z direction
     * @return Vector field containing the shifted unit vector field
     */
    template <int ind_t = CON>
    Kadath::Vector e_rad(double shift_x = 0., double shift_y = 0., double shift_z = 0.) const;

    template <int ind_t = CON>
    Kadath::Vector e_rad_from_cart(const Kadath::Vector& coords, const Kadath::Scalar& radius) const;

    /**
     * CoordFields::e_cart
     *
     * generate and return a unit vector field for a given cartesian coordinate
     *
     * @param [input] dir: coordinate direction (e.g. x = 1, y = 2, z = 3)
     * @return Vector field containing the vector field of the specified coordinate direction
     */
    template <int ind_t = CON> Kadath::Vector e_cart(int dir) const;
};
/**
 * @}
 */

template <typename space_t>
template <int ind_t>
Kadath::Vector CoordFields<space_t>::cart(double shift_x, double shift_y, double shift_z) const
{
    Kadath::Vector cart(space, ind_t, basis);

    int ndom = space.get_nbr_domains();
    const int ndim = space.get_ndim();

    for (int d = 0; d < ndom - 1; d++) {
        cart.set(1).set_domain(d) = space.get_domain(d)->get_cart(1) - shift_x;
        cart.set(2).set_domain(d) = space.get_domain(d)->get_cart(2) - shift_y;
        if (ndim >= 3) {
            cart.set(3).set_domain(d) = space.get_domain(d)->get_cart(3) - shift_z;
        }
    }

    // set outter boundary to surface coordinates (1/r) for compact domain
    // FIXME: these are not coordinate shifted
    cart.set(1).set_domain(ndom - 1) = space.get_domain(ndom - 1)->get_cart_surr(1);
    cart.set(2).set_domain(ndom - 1) = space.get_domain(ndom - 1)->get_cart_surr(2);
    if (ndim >= 3) {
        cart.set(3).set_domain(ndom - 1) = space.get_domain(ndom - 1)->get_cart_surr(3);
    }

    cart.set(1).std_base();
    cart.set(2).std_base();
    if (ndim >= 3) {
        cart.set(3).std_anti_base();
    }

    return cart;
}

template <typename space_t>
Kadath::Scalar CoordFields<space_t>::radius(double shift_x, double shift_y, double shift_z) const
{
    auto coords = this->cart(shift_x, shift_y, shift_z);
    return radius_from_cart(coords);
}

template <typename space_t>
Kadath::Scalar CoordFields<space_t>::radius_from_cart(const Kadath::Vector& coords) const
{
    int ndom = space.get_nbr_domains();
    const int ndim = space.get_ndim();

    Kadath::Scalar r_sq = coords(1) * coords(1) + coords(2) * coords(2);
    if (ndim >= 3) {
        r_sq = r_sq + coords(3) * coords(3);
    }
    r_sq.std_base();

    Kadath::Scalar r = sqrt(r_sq);
    r.std_base();

    // set outter boundary to constant large radius
    Index pos(space.get_domain(ndom - 1)->get_nbr_points());
    const int npts_r = space.get_domain(ndom - 1)->get_nbr_points()(0);
    for (int i = 1; i <= ndim; ++i) {
        const int dom = ndom - 1;
        do {
            if (pos(0) == npts_r - 1) {
                r.set_domain(dom).set(pos) = 1e10;
            }
        } while (pos.inc());
    }

    return r;
}

template <typename space_t>
template <int ind_t>
Kadath::Vector CoordFields<space_t>::rot_z(double shift_x, double shift_y, double shift_z) const
{
    auto coords = this->cart(shift_x, shift_y, shift_z);
    return rot_z_from_cart<ind_t>(coords);
}

template <typename space_t>
template <int ind_t>
Kadath::Vector CoordFields<space_t>::rot_z_from_cart(const Kadath::Vector& coords) const
{
    Kadath::Vector rot_z(space, ind_t, basis);
    rot_z.set(1) = -coords(2);
    rot_z.set(2) = coords(1);
    rot_z.set(3) = 0.;

    rot_z.std_base();
    return rot_z;
}

template <typename space_t>
template <int ind_t>
Kadath::Vector CoordFields<space_t>::rot_x(double shift_x, double shift_y, double shift_z) const
{
    auto coords = this->cart(shift_x, shift_y, shift_z);
    return rot_x_from_cart<ind_t>(coords);
}

template <typename space_t>
template <int ind_t>
Kadath::Vector CoordFields<space_t>::rot_x_from_cart(const Kadath::Vector& coords) const
{
    Kadath::Vector rot_x(space, ind_t, basis);
    rot_x.set(1) = 0.;
    rot_x.set(2) = -coords(3);
    rot_x.set(3) = coords(2);

    rot_x.set(1).std_base();
    rot_x.set(2).std_anti_base();
    rot_x.set(3).std_base();
    return rot_x;
}

template <typename space_t>
template <int ind_t>
Kadath::Vector CoordFields<space_t>::e_rad(double shift_x, double shift_y, double shift_z) const
{
    auto coords = this->cart(shift_x, shift_y, shift_z);
    auto r = this->radius_from_cart(coords);
    return e_rad_from_cart<ind_t>(coords, r);
}

template <typename space_t>
template <int ind_t>
Kadath::Vector CoordFields<space_t>::e_rad_from_cart(const Kadath::Vector& coords,
                                                     const Kadath::Scalar& radius) const
{
    int ndom = space.get_nbr_domains();

    // this can be problematic around the given origin
    Kadath::Vector e_rad(space, ind_t, basis);
    for (int i : {1, 2, 3}) {
        e_rad.set(i) = coords(i) / radius;

        // fix non-finite values likely at (0,0,0)
        // Note: this fix is sufficient for the FUKA codes as
        // we are only concerned with e_rad on the surfaces of compact
        // objects or at infinity.  This fix is to remove NaNs and their
        // impact on the system of equations
        for (int dom = 0; dom < ndom - 1; ++dom) {
            Index pos(space.get_domain(dom)->get_nbr_points());
            do {
                if (!std::isfinite(e_rad(i)(dom)(pos)) && pos(0) == 0) {
                    Index tempos(pos);
                    tempos.set(0) = pos(0) + 1;
                    e_rad.set(i).set_domain(dom).set(pos) = e_rad(i)(dom)(tempos);
                }
            } while (pos.inc());
        }
        // FIXME this is not correct for shifted cartesian coords
        e_rad.set(i).set_domain(ndom - 1) = space.get_domain(ndom - 1)->get_cart_surr(i);
    }

    e_rad.set(1).std_base();
    e_rad.set(2).std_base();
    e_rad.set(3).std_anti_base();

    return e_rad;
}

template <typename space_t> template <int ind_t> Kadath::Vector CoordFields<space_t>::e_cart(int dir) const
{
    Kadath::Vector e_cart(space, ind_t, basis);
    e_cart = 0;
    e_cart.set(dir) = 1.;

    // all components should act like simple scalar fields,
    // so decompose them separately
    e_cart.set(1).std_base();
    e_cart.set(2).std_base();
    e_cart.set(3).std_base();

    return e_cart;
}

inline void assign_coordinate_field(Kadath::Scalar& destination, const Kadath::Scalar& source)
{
    const int ndom = destination.get_space().get_nbr_domains();
    for (int dom = 0; dom < ndom; ++dom)
        destination.set_domain(dom) = source(dom);
}

inline void assign_coordinate_field(Kadath::Vector& destination, const Kadath::Vector& source)
{
    const int ndom = destination.get_space().get_nbr_domains();
    for (int dom = 0; dom < ndom; ++dom) {
        destination.set_basis(dom) = source.get_basis().get_basis(dom);
        for (int component = 1; component <= 3; ++component)
            destination.set(component).set_domain(dom) = source(component)(dom);
    }
}

inline int coordinate_field_constant_index(System_of_eqs& syst, const char* field_name)
{
    std::array<char, LMAX> name{};
    trim_spaces(name.data(), field_name);

    int which = -1;
    int valence = 0;
    char* name_ind = nullptr;
    Array<int>* type_ind = nullptr;
    std::unique_ptr<char[]> owned_name_ind;
    std::unique_ptr<Array<int>> owned_type_ind;
    struct parsed_indices_adopter
    {
        char*& name_ind;
        Array<int>*& type_ind;
        std::unique_ptr<char[]>& owned_name_ind;
        std::unique_ptr<Array<int>>& owned_type_ind;
        ~parsed_indices_adopter()
        {
            owned_name_ind.reset(name_ind);
            owned_type_ind.reset(type_ind);
        }
    } adopter{name_ind, type_ind, owned_name_ind, owned_type_ind};

    return syst.iscst(name.data(), which, valence, name_ind, type_ind) ? which : -1;
}

template <typename space_t>
void update_fields_values(CoordFields<space_t> const& cf_generator, vec_ary_t& coord_vectors,
                          scalar_ary_t& coord_scalars, const double xo, const double xc1, const double xc2,
                          bool refresh_cartesian_units = true)
{
    const bool needs_global_cart = coord_vectors[to_int(coord_vector::GLOBAL_ROT)] ||
                                   coord_vectors[to_int(coord_vector::S_INF)];
    if (needs_global_cart) {
        const auto coords = cf_generator.template cart<CON>(xo);
        if (coord_vectors[to_int(coord_vector::GLOBAL_ROT)])
            assign_coordinate_field(*coord_vectors[to_int(coord_vector::GLOBAL_ROT)],
                                    cf_generator.template rot_z_from_cart<CON>(coords));
        if (coord_vectors[to_int(coord_vector::S_INF)]) {
            const auto radius = cf_generator.radius_from_cart(coords);
            assign_coordinate_field(*coord_vectors[to_int(coord_vector::S_INF)],
                                    cf_generator.template e_rad_from_cart<COV>(coords, radius));
        }
    }

    const bool needs_bco1_cart = coord_vectors[to_int(coord_vector::BCO1_ROTx)] ||
                                 coord_vectors[to_int(coord_vector::BCO1_ROTz)] ||
                                 coord_vectors[to_int(coord_vector::S_BCO1)] ||
                                 coord_scalars[to_int(coord_scalar::R_BCO1)];
    if (needs_bco1_cart) {
        const auto coords = cf_generator.template cart<CON>(xc1);
        if (coord_vectors[to_int(coord_vector::BCO1_ROTx)])
            assign_coordinate_field(*coord_vectors[to_int(coord_vector::BCO1_ROTx)],
                                    cf_generator.template rot_x_from_cart<CON>(coords));
        if (coord_vectors[to_int(coord_vector::BCO1_ROTz)])
            assign_coordinate_field(*coord_vectors[to_int(coord_vector::BCO1_ROTz)],
                                    cf_generator.template rot_z_from_cart<CON>(coords));
        if (coord_vectors[to_int(coord_vector::S_BCO1)] ||
            coord_scalars[to_int(coord_scalar::R_BCO1)]) {
            const auto radius = cf_generator.radius_from_cart(coords);
            if (coord_scalars[to_int(coord_scalar::R_BCO1)])
                assign_coordinate_field(*coord_scalars[to_int(coord_scalar::R_BCO1)], radius);
            if (coord_vectors[to_int(coord_vector::S_BCO1)])
                assign_coordinate_field(*coord_vectors[to_int(coord_vector::S_BCO1)],
                                        cf_generator.template e_rad_from_cart<COV>(coords, radius));
        }
    }

    const bool needs_bco2_cart = coord_vectors[to_int(coord_vector::BCO2_ROTx)] ||
                                 coord_vectors[to_int(coord_vector::BCO2_ROTz)] ||
                                 coord_vectors[to_int(coord_vector::S_BCO2)] ||
                                 coord_scalars[to_int(coord_scalar::R_BCO2)];
    if (needs_bco2_cart) {
        const auto coords = cf_generator.template cart<CON>(xc2);
        if (coord_vectors[to_int(coord_vector::BCO2_ROTx)])
            assign_coordinate_field(*coord_vectors[to_int(coord_vector::BCO2_ROTx)],
                                    cf_generator.template rot_x_from_cart<CON>(coords));
        if (coord_vectors[to_int(coord_vector::BCO2_ROTz)])
            assign_coordinate_field(*coord_vectors[to_int(coord_vector::BCO2_ROTz)],
                                    cf_generator.template rot_z_from_cart<CON>(coords));
        if (coord_vectors[to_int(coord_vector::S_BCO2)] ||
            coord_scalars[to_int(coord_scalar::R_BCO2)]) {
            const auto radius = cf_generator.radius_from_cart(coords);
            if (coord_scalars[to_int(coord_scalar::R_BCO2)])
                assign_coordinate_field(*coord_scalars[to_int(coord_scalar::R_BCO2)], radius);
            if (coord_vectors[to_int(coord_vector::S_BCO2)])
                assign_coordinate_field(*coord_vectors[to_int(coord_vector::S_BCO2)],
                                        cf_generator.template e_rad_from_cart<COV>(coords, radius));
        }
    }

    if (refresh_cartesian_units && coord_vectors[to_int(coord_vector::EX)])
        assign_coordinate_field(*coord_vectors[to_int(coord_vector::EX)], cf_generator.template e_cart<COV>(1));

    if (refresh_cartesian_units && coord_vectors[to_int(coord_vector::EY)])
        assign_coordinate_field(*coord_vectors[to_int(coord_vector::EY)], cf_generator.template e_cart<COV>(2));

    if (refresh_cartesian_units && coord_vectors[to_int(coord_vector::EZ)])
        assign_coordinate_field(*coord_vectors[to_int(coord_vector::EZ)], cf_generator.template e_cart<COV>(3));
}

template <typename space_t>
void update_fields(CoordFields<space_t> const& cf_generator, vec_ary_t& coord_vectors, scalar_ary_t& coord_scalars,
                   const double xo, const double xc1, const double xc2, System_of_eqs* syst)
{
    update_fields_values(cf_generator, coord_vectors, coord_scalars, xo, xc1, xc2);

    if (syst != nullptr) {
        finalize_coordinate_field_refresh(*syst, coord_vectors, coord_scalars);
    }
}

template <typename space_t>
void update_fields(CoordFields<space_t> const& cf_generator, vec_ary_t& coord_vectors, scalar_ary_t&& coord_scalars,
                   const double xo, const double xc1, const double xc2, System_of_eqs* syst)
{
    update_fields(cf_generator, coord_vectors, coord_scalars, xo, xc1, xc2, syst);
}

template <typename space_t>
void update_fields(CoordFields<space_t> const& cf_generator, vec_ary_t& coord_vectors, scalar_ary_t& coord_scalars,
                   const double xo, const double xc1, const double xc2, const coord_field_binding_t& binding)
{
    update_fields_values(cf_generator, coord_vectors, coord_scalars, xo, xc1, xc2,
                         /*refresh_cartesian_units=*/false);
    finalize_coordinate_field_refresh(*binding.syst, coord_vectors, coord_scalars, &binding);
}

template <typename space_t>
void update_fields(CoordFields<space_t> const& cf_generator, vec_ary_t& coord_vectors, scalar_ary_t&& coord_scalars,
                   const double xo, const double xc1, const double xc2, const coord_field_binding_t& binding)
{
    update_fields(cf_generator, coord_vectors, coord_scalars, xo, xc1, xc2, binding);
}

template <typename space_t>
void update_fields_co_values(CoordFields<space_t> const& cf_generator, vec_ary_t& coord_vectors,
                             scalar_ary_t& coord_scalars, const double xo,
                             bool refresh_cartesian_units = true)
{
    const bool needs_origin_cart =
        coord_vectors[to_int(coord_vector::GLOBAL_ROT)] || coord_vectors[to_int(coord_vector::BCO1_ROTx)] ||
        coord_vectors[to_int(coord_vector::BCO1_ROTz)] || coord_vectors[to_int(coord_vector::S_BCO1)] ||
        coord_vectors[to_int(coord_vector::S_INF)] || coord_scalars[to_int(coord_scalar::R_BCO1)];
    if (needs_origin_cart) {
        const auto coords = cf_generator.template cart<CON>(xo);

        if (coord_vectors[to_int(coord_vector::GLOBAL_ROT)] ||
            coord_vectors[to_int(coord_vector::BCO1_ROTz)]) {
            const auto rot_z = cf_generator.template rot_z_from_cart<CON>(coords);
            if (coord_vectors[to_int(coord_vector::GLOBAL_ROT)])
                assign_coordinate_field(*coord_vectors[to_int(coord_vector::GLOBAL_ROT)], rot_z);
            if (coord_vectors[to_int(coord_vector::BCO1_ROTz)])
                assign_coordinate_field(*coord_vectors[to_int(coord_vector::BCO1_ROTz)], rot_z);
        }
        if (coord_vectors[to_int(coord_vector::BCO1_ROTx)]) {
            const auto rot_x = cf_generator.template rot_x_from_cart<CON>(coords);
            assign_coordinate_field(*coord_vectors[to_int(coord_vector::BCO1_ROTx)], rot_x);
        }

        const bool needs_origin_radius = coord_vectors[to_int(coord_vector::S_BCO1)] ||
                                         coord_vectors[to_int(coord_vector::S_INF)] ||
                                         coord_scalars[to_int(coord_scalar::R_BCO1)];
        if (needs_origin_radius) {
            const auto radius = cf_generator.radius_from_cart(coords);
            if (coord_scalars[to_int(coord_scalar::R_BCO1)])
                assign_coordinate_field(*coord_scalars[to_int(coord_scalar::R_BCO1)], radius);
            if (coord_vectors[to_int(coord_vector::S_BCO1)] || coord_vectors[to_int(coord_vector::S_INF)]) {
                const auto e_rad = cf_generator.template e_rad_from_cart<COV>(coords, radius);
                if (coord_vectors[to_int(coord_vector::S_BCO1)])
                    assign_coordinate_field(*coord_vectors[to_int(coord_vector::S_BCO1)], e_rad);
                if (coord_vectors[to_int(coord_vector::S_INF)])
                    assign_coordinate_field(*coord_vectors[to_int(coord_vector::S_INF)], e_rad);
            }
        }
    }

    const bool needs_bco2_cart = coord_vectors[to_int(coord_vector::BCO2_ROTx)] ||
                                 coord_vectors[to_int(coord_vector::BCO2_ROTz)] ||
                                 coord_vectors[to_int(coord_vector::S_BCO2)] ||
                                 coord_scalars[to_int(coord_scalar::R_BCO2)];
    if (needs_bco2_cart) {
        const auto coords = cf_generator.template cart<CON>(0.);
        if (coord_vectors[to_int(coord_vector::BCO2_ROTx)]) {
            const auto rot_x = cf_generator.template rot_x_from_cart<CON>(coords);
            assign_coordinate_field(*coord_vectors[to_int(coord_vector::BCO2_ROTx)], rot_x);
        }
        if (coord_vectors[to_int(coord_vector::BCO2_ROTz)]) {
            const auto rot_z = cf_generator.template rot_z_from_cart<CON>(coords);
            assign_coordinate_field(*coord_vectors[to_int(coord_vector::BCO2_ROTz)], rot_z);
        }
        if (coord_vectors[to_int(coord_vector::S_BCO2)] || coord_scalars[to_int(coord_scalar::R_BCO2)]) {
            const auto radius = cf_generator.radius_from_cart(coords);
            if (coord_scalars[to_int(coord_scalar::R_BCO2)])
                assign_coordinate_field(*coord_scalars[to_int(coord_scalar::R_BCO2)], radius);
            if (coord_vectors[to_int(coord_vector::S_BCO2)]) {
                const auto e_rad = cf_generator.template e_rad_from_cart<COV>(coords, radius);
                assign_coordinate_field(*coord_vectors[to_int(coord_vector::S_BCO2)], e_rad);
            }
        }
    }

    if (refresh_cartesian_units && coord_vectors[to_int(coord_vector::EX)])
        assign_coordinate_field(*coord_vectors[to_int(coord_vector::EX)], cf_generator.template e_cart<COV>(1));
    if (refresh_cartesian_units && coord_vectors[to_int(coord_vector::EY)])
        assign_coordinate_field(*coord_vectors[to_int(coord_vector::EY)], cf_generator.template e_cart<COV>(2));
    if (refresh_cartesian_units && coord_vectors[to_int(coord_vector::EZ)])
        assign_coordinate_field(*coord_vectors[to_int(coord_vector::EZ)], cf_generator.template e_cart<COV>(3));
}

inline void finalize_coordinate_field_refresh(System_of_eqs& syst, vec_ary_t& coord_vectors,
                                              scalar_ary_t& coord_scalars,
                                              const coord_field_binding_t* existing_binding)
{
    if (existing_binding != nullptr) {
        update_bound_coordinate_fields(*existing_binding, coord_vectors, coord_scalars);
    } else {
        const auto binding = bind_coordinate_fields(syst, coord_vectors, coord_scalars);
        update_bound_coordinate_fields(binding, coord_vectors, coord_scalars);
    }
    syst.store_forwarded_residual(syst.sec_member_partitioned());
}

/**
 * @brief Overload for temporary (R-value) scalar arrays.
 * @details This version is called when a temporary object is passed for `coord_scalars`.
 *          The `&&` signifies an "R-value reference", which binds to unnamed, temporary
 *          objects. This allows for convenient calls where no scalar fields are needed.
 *
 * **Example Call**:
 * @code
 * // The `{}` creates a temporary, default-initialized scalar_ary_t.
 * // This is an R-value, so this overload is selected.
 * update_fields_co(cf_gen, vec_ary, {}, 0.0);
 * @endcode
 */
template <typename space_t>
void update_fields_co(CoordFields<space_t> const& cf_generator, vec_ary_t& coord_vectors, scalar_ary_t&& coord_scalars,
                      const double xo, System_of_eqs* syst)
{
    update_fields_co(cf_generator, coord_vectors, coord_scalars, xo, syst);
}

/**
 * @brief Overload for existing (L-value) scalar arrays.
 * @details This version is called when a named variable is passed for `coord_scalars`.
 *          The single `&` signifies an "L-value reference", which binds to objects
 *          that have a persistent memory location.
 *
 * **Example Call**:
 * @code
 * scalar_ary_t my_scalars; // A named variable, i.e., an L-value.
 * update_fields_co(cf_gen, vec_ary, my_scalars, 0.0); // This overload is selected.
 * @endcode
 */
template <typename space_t>
void update_fields_co(CoordFields<space_t> const& cf_generator, vec_ary_t& coord_vectors, scalar_ary_t& coord_scalars,
                      const double xo, System_of_eqs* syst)
{
    update_fields_co_values(cf_generator, coord_vectors, coord_scalars, xo);
    if (syst != nullptr)
        finalize_coordinate_field_refresh(*syst, coord_vectors, coord_scalars);
}

template <typename space_t>
void update_fields_co(CoordFields<space_t> const& cf_generator, vec_ary_t& coord_vectors, scalar_ary_t&& coord_scalars,
                      const double xo, const coord_field_binding_t& binding)
{
    update_fields_co(cf_generator, coord_vectors, coord_scalars, xo, binding);
}

template <typename space_t>
void update_fields_co(CoordFields<space_t> const& cf_generator, vec_ary_t& coord_vectors, scalar_ary_t& coord_scalars,
                      const double xo, const coord_field_binding_t& binding)
{
    update_fields_co_values(cf_generator, coord_vectors, coord_scalars, xo,
                            /*refresh_cartesian_units=*/false);
    finalize_coordinate_field_refresh(*binding.syst, coord_vectors, coord_scalars, &binding);
}

inline bool update_field(System_of_eqs& syst, int dom, const char* n, Tensor& so)
{
    const int which = coordinate_field_constant_index(syst, n);
    const bool found = which >= 0;
    if (found) {
        Term_eq* mm = syst.give_cst(which, dom);
        mm->set_val_t(so);
    }
    return found;
}

inline coord_field_binding_t bind_coordinate_fields(System_of_eqs& syst, const vec_ary_t& coord_vectors,
                                                    const scalar_ary_t& coord_scalars)
{
    coord_field_binding_t binding(syst);
    for (int i = 0; i < NUM_VECTORS_V; ++i) {
        if (coord_vectors[i])
            binding.vector_constants[i] = coordinate_field_constant_index(syst, cv_names[i].c_str());
    }
    for (int i = 0; i < NUM_SCALARS_V; ++i) {
        if (coord_scalars[i])
            binding.scalar_constants[i] = coordinate_field_constant_index(syst, cs_names[i].c_str());
    }
    return binding;
}

inline void update_bound_coordinate_fields(const coord_field_binding_t& binding, vec_ary_t& coord_vectors,
                                           scalar_ary_t& coord_scalars)
{
    auto update = [&](const int which, Tensor& field) {
        if (which < 0)
            return;
        const int ndom = field.get_space().get_nbr_domains();
        for (int dom = 0; dom < ndom; ++dom)
            binding.syst->give_cst(which, dom)->set_val_t(field);
    };

    for (int i = 0; i < NUM_VECTORS_V; ++i) {
        if (coord_vectors[i])
            update(binding.vector_constants[i], *coord_vectors[i]);
    }
    for (int i = 0; i < NUM_SCALARS_V; ++i) {
        if (coord_scalars[i])
            update(binding.scalar_constants[i], *coord_scalars[i]);
    }
}

template <class space_t>
void activate_coordinate_vector(vec_ary_t& coord_vectors, space_t& space, coord_vector field)
{
    Base_tensor basis(space, CARTESIAN_BASIS);
    const bool contravariant = field == coord_vector::GLOBAL_ROT || field == coord_vector::BCO1_ROTx ||
                               field == coord_vector::BCO1_ROTz || field == coord_vector::BCO2_ROTx ||
                               field == coord_vector::BCO2_ROTz;
    coord_vectors[to_int(field)] = Vector(space, contravariant ? CON : COV, basis);
}

template <class space_t> vec_ary_t default_binary_vector_ary(space_t& space)
{
    vec_ary_t coord_vectors{};
    Base_tensor basis(space, CARTESIAN_BASIS);
    for (auto i = 0; i < NUM_VECTORS_V; ++i) {
        switch (i) {
            case to_int(coord_vector::GLOBAL_ROT):
            case to_int(coord_vector::BCO1_ROTx):
            case to_int(coord_vector::BCO1_ROTz):
            case to_int(coord_vector::BCO2_ROTx):
            case to_int(coord_vector::BCO2_ROTz):
                coord_vectors[i] = Vector(space, CON, basis);
                break;
            default:
                coord_vectors[i] = Vector(space, COV, basis);
        }
    }
    return coord_vectors;
}

template <class space_t> vec_ary_t default_co_vector_ary(space_t& space)
{
    vec_ary_t coord_vectors{};
    activate_coordinate_vector(coord_vectors, space, coord_vector::GLOBAL_ROT);
    activate_coordinate_vector(coord_vectors, space, coord_vector::BCO1_ROTx);
    activate_coordinate_vector(coord_vectors, space, coord_vector::BCO1_ROTz);
    activate_coordinate_vector(coord_vectors, space, coord_vector::EX);
    activate_coordinate_vector(coord_vectors, space, coord_vector::EY);
    activate_coordinate_vector(coord_vectors, space, coord_vector::EZ);
    activate_coordinate_vector(coord_vectors, space, coord_vector::S_BCO1);
    activate_coordinate_vector(coord_vectors, space, coord_vector::S_INF);
    return coord_vectors;
}
