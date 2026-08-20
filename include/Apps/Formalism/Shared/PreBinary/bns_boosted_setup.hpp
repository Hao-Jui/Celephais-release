#pragma once

#include "Hydro/EOS.hh"
#include "For_Kadath/Base_tensor/base_tensor.hpp"
#include "For_Kadath/Scalar/scalar.hpp"
#include "For_Kadath/Tensor/vector.hpp"
#include "For_Kadath/Config/config_binary.hpp"
#include "For_Kadath/Config/config_bco.hpp"
#include "For_Kadath/Utilities/Exporters/bco_geometry.hpp"
#include "Apps/Bco_utils/bco_io.hpp"
#include "Apps/Bco_utils/bco_regrid.hpp"
#include "Apps/Bco_utils/ns_bounds.hpp"
#include "Apps/Formalism/Shared/scalar_point_batch.hpp"
#include "For_Kadath/Utilities/PN/orbital_pn_params.hpp"
#include "For_Kadath/IO/be_file_source.hpp"

#include <array>
#include <cmath>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace Kadath {

template <bool WithScalar, typename eos_t, typename NSpaceT, typename BinSpaceT,
          typename NSOuterAdaptedT, typename NSInnerAdaptedT, typename BinOuterAdaptedT,
          typename BinInnerAdaptedT, typename WarnTheoryParams>
inline void bns_setup_boosted_3d_impl(kadath_config<BCO_NS_INFO>& NS1config,
                                      kadath_config<BCO_NS_INFO>& NS2config,
                                      kadath_config<BIN_INFO>& bconfig,
                                      WarnTheoryParams&& warn_theory_params)
{
    using namespace Kadath::Margherita;
    static_assert(!WithScalar,
                  "boosted setup field set is unavailable in this build");

    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // -----------------------------------------------------------------------
    // Read NS1 data
    // -----------------------------------------------------------------------
    std::string nsspacein = NS1config.space_filename();
    BeFileSource ff1(nsspacein);
    NSpaceT spacein1(ff1);
    Scalar confin1(spacein1, ff1);
    Scalar lapsein1(spacein1, ff1);
    Vector shiftin1(spacein1, ff1);
    Scalar loghin1(spacein1, ff1);
    std::unique_ptr<Scalar> phiin1;
    if (NS1config.set_field(PHI) == true)
        phiin1 = std::make_unique<Scalar>(spacein1, ff1);
    std::optional<Scalar> scalar_slot1;
    if constexpr (WithScalar)
        scalar_slot1.emplace(spacein1, ff1);
    int ndomin1 = spacein1.get_nbr_domains();

    NS1config.set(HC) = std::exp(bco_utils::get_boundary_val(0, loghin1, INNER_BC));
    NS1config.set(NC) = EOS<eos_t, eos_var_t::DENSITY>::get(NS1config(HC));
    bco_utils::update_config_NS_radii(spacein1, NS1config, 1);


    // -----------------------------------------------------------------------
    // Read NS2 data
    // -----------------------------------------------------------------------
    nsspacein = NS2config.space_filename();
    BeFileSource ff2(nsspacein);
    NSpaceT spacein2(ff2);
    Scalar confin2(spacein2, ff2);
    Scalar lapsein2(spacein2, ff2);
    Vector shiftin2(spacein2, ff2);
    Scalar loghin2(spacein2, ff2);
    std::unique_ptr<Scalar> phiin2;
    if (NS2config.set_field(PHI) == true)
        phiin2 = std::make_unique<Scalar>(spacein2, ff2);
    std::optional<Scalar> scalar_slot2;
    if constexpr (WithScalar)
        scalar_slot2.emplace(spacein2, ff2);
    int ndomin2 = spacein2.get_nbr_domains();

    NS2config.set(HC) = std::exp(bco_utils::get_boundary_val(0, loghin2, INNER_BC));
    NS2config.set(NC) = EOS<eos_t, eos_var_t::DENSITY>::get(NS2config(HC));
    bco_utils::update_config_NS_radii(spacein2, NS2config, 1);


    // -----------------------------------------------------------------------
    // Update binary config from NS configs
    // -----------------------------------------------------------------------
    for (int i = 0; i < NUM_BCO_PARAMS_V; ++i)
        bconfig.set(i, BCO1) = NS1config.set(i);

    for (int i = 0; i < NUM_BCO_PARAMS_V; ++i)
        bconfig.set(i, BCO2) = NS2config.set(i);

    // -----------------------------------------------------------------------
    // Helper lambdas
    // -----------------------------------------------------------------------
    auto gen_radius_field = [&](auto& spacein, auto& old_space_radius, const int ndomin) {
        const NSOuterAdaptedT* old_outer_adapted =
            dynamic_cast<const NSOuterAdaptedT*>(spacein.get_domain(1));
        for (int i = 0; i < ndomin; ++i)
            if (i != 1)
                old_space_radius.set_domain(i) = spacein.get_domain(i)->get_radius();
        old_space_radius.set_domain(1) = old_outer_adapted->get_outer_radius();
        old_space_radius.std_base();
    };

    auto interp_field = [&](auto& space_arg, int outer_dom, auto& old_phi) {
        const int d = outer_dom;
        const NSInnerAdaptedT* old_inner =
            dynamic_cast<const NSInnerAdaptedT*>(space_arg.get_domain(d + 1));
        bco_utils::update_adapted_field(old_phi, d, d + 1, old_inner, INNER_BC);
    };

    if (NS1config.set_field(PHI) == true)
        interp_field(spacein1, 1, *phiin1);
    if (NS2config.set_field(PHI) == true)
        interp_field(spacein2, 1, *phiin2);

    // Create radius fields for interpolation
    Scalar old_space_radius1(spacein1);
    old_space_radius1.annule_hard();
    gen_radius_field(spacein1, old_space_radius1, ndomin1);

    Scalar old_space_radius2(spacein2);
    old_space_radius2.annule_hard();
    gen_radius_field(spacein2, old_space_radius2, ndomin2);

    // -----------------------------------------------------------------------
    // Setup domain boundaries
    // -----------------------------------------------------------------------
    double r_max_tot = std::max(bconfig(RMID, BCO1), bconfig(RMID, BCO2));
    bconfig.set(ROUT, BCO1) = (bconfig(DIST) / 2. - r_max_tot) / 3. + r_max_tot;
    bconfig.set(ROUT, BCO2) = (bconfig(DIST) / 2. - r_max_tot) / 3. + r_max_tot;

    std::vector<double> out_bounds(1 + bconfig(OUTER_SHELLS));

    if (rank == 0)
        std::cout << "NSHELLS_NS1: " << bconfig(NSHELLS, BCO1) << ", "
                  << "NSHELLS_NS2: " << bconfig(NSHELLS, BCO2) << std::endl;

    for (size_t e = 0; e < out_bounds.size(); ++e)
        out_bounds[e] = bconfig(REXT) * (1. + e * 0.25);

    std::vector<double> NS1_bounds = bco_utils::make_binary_NS_bounds(bconfig, BCO1);
    std::vector<double> NS2_bounds = bco_utils::make_binary_NS_bounds(bconfig, BCO2);

    if (rank == 0) {
        std::cout << "Bounds:" << std::endl;
        bco_utils::print_bounds("Outer", out_bounds);
        bco_utils::print_bounds("NS1",   NS1_bounds);
        bco_utils::print_bounds("NS2",   NS2_bounds);
    }

    // -----------------------------------------------------------------------
    // Create binary space
    // -----------------------------------------------------------------------
    int typer = CHEB_TYPE;
    BinSpaceT space(typer, bconfig(DIST), NS1_bounds, NS2_bounds, out_bounds, bconfig(BIN_RES));
    Base_tensor basis(space, CARTESIAN_BASIS);

    std::array<const BinOuterAdaptedT*, 2> new_outer_adapted{
        dynamic_cast<const BinOuterAdaptedT*>(space.get_domain(space.ADAPTED1)),
        dynamic_cast<const BinOuterAdaptedT*>(space.get_domain(space.ADAPTED2))};

    std::array<const BinInnerAdaptedT*, 2> new_inner_adapted{
        dynamic_cast<const BinInnerAdaptedT*>(space.get_domain(space.ADAPTED1 + 1)),
        dynamic_cast<const BinInnerAdaptedT*>(space.get_domain(space.ADAPTED2 + 1))};

    bco_utils::interp_adapted_mapping(new_inner_adapted[0], 1, old_space_radius1);
    bco_utils::interp_adapted_mapping(new_outer_adapted[0], 1, old_space_radius1);

    bco_utils::interp_adapted_mapping(new_inner_adapted[1], 1, old_space_radius2);
    bco_utils::interp_adapted_mapping(new_outer_adapted[1], 1, old_space_radius2);

    double xc1 = bco_utils::get_center(space, space.NS1);
    double xc2 = bco_utils::get_center(space, space.NS2);

    if (rank == 0)
        std::cout << "xc1: " << xc1 << std::endl << "xc2: " << xc2 << std::endl;

    // -----------------------------------------------------------------------
    // Initialize fields
    // -----------------------------------------------------------------------
    Scalar logh(space);
    logh.annule_hard();

    Scalar conf(space);
    conf.annule_hard();

    Scalar lapse(space);
    lapse.annule_hard();

    Vector shift(space, CON, basis);
    for (int i = 1; i <= 3; i++)
        shift.set(i).annule_hard();

    Scalar phi(space);
    phi.annule_hard();

    std::optional<Scalar> scalar_slot;
    if constexpr (WithScalar) {
        scalar_slot.emplace(space);
        scalar_slot->annule_hard();
    }

    const double ns1_invw4 = bco_utils::set_decay(bconfig, BCO1);
    const double ns2_invw4 = bco_utils::set_decay(bconfig, BCO2);

    if (rank == 0)
        std::cout << "WeightNS1: " << bconfig(DECAY, BCO1) << ", "
                  << "WeightNS2: " << bconfig(DECAY, BCO2) << std::endl;

    // -----------------------------------------------------------------------
    // Interpolate fields from single stars. One target traversal now shares
    // both source-domain lookups and numerical points across the full field set.
    // -----------------------------------------------------------------------
    using bns_field_transfer::two_source_combination;
    using bns_field_transfer::two_source_field;
    std::array<two_source_field, 8> transfer_fields;
    std::size_t transfer_count = 0;
    transfer_fields[transfer_count++] =
        {&conf, &confin1, &confin2, 1., two_source_combination::background, true};
    transfer_fields[transfer_count++] =
        {&lapse, &lapsein1, &lapsein2, 1., two_source_combination::background, true};
    transfer_fields[transfer_count++] =
        {&logh, &loghin1, &loghin2, 0., two_source_combination::weighted_sum, false};
    transfer_fields[transfer_count++] =
        {&phi, phiin1.get(), phiin2.get(), 0.,
         two_source_combination::accumulate_from_zero, false};
    for (int i = 1; i <= 3; ++i) {
        transfer_fields[transfer_count++] =
            {&shift.set(i), &shiftin1.set(i), &shiftin2.set(i), 0.,
             two_source_combination::weighted_sum, false};
    }
    if constexpr (WithScalar) {
        transfer_fields[transfer_count++] =
            {&*scalar_slot, &*scalar_slot1, &*scalar_slot2, 0.,
             two_source_combination::background, true};
    }
    bns_field_transfer::superpose_two_source_batch(
        std::span<const two_source_field>(transfer_fields.data(), transfer_count),
        {xc1, xc2}, {ns1_invw4, ns2_invw4});

    int ndom = space.get_nbr_domains();

    // Zero out fields outside stars
    for (auto i : {space.ADAPTED1 + 1, space.ADAPTED2 + 1}) {
        phi.set_domain(i).annule_hard();
        logh.set_domain(i).annule_hard();
    }
    for (int i = space.OUTER; i < ndom; ++i) {
        phi.set_domain(i).annule_hard();
        logh.set_domain(i).annule_hard();
    }

    // Set spectral bases
    conf.std_base();
    lapse.std_base();
    logh.std_base();
    shift.std_base();
    phi.std_base();
    if constexpr (WithScalar)
        scalar_slot->std_base();

    // Save to file. The SWEIGHT field flag records the layout of THIS file for
    // later warm-start decode; the binary two-centre weight is built from the
    // current binary-TOML massScal.
    {
        bco_utils::save_to_file(space, bconfig, conf, lapse, shift, logh, phi);
    }
}

} // namespace Kadath
