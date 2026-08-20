#pragma once

#include "For_Kadath/Kadath_point_h/kadath_adapted.hpp"
#include "For_Kadath/Kadath_point_h/kadath_spheric_homothetic.hpp"
#include "For_Kadath/IO/be_file_source.hpp"
#include "For_Kadath/Array/exceptions.hpp"
#include "For_Kadath/Config/config_enums.hpp"
#include "Hydro/EOS.hh"

#include <cstdlib>
#include <string>
#include <utility>

namespace Kadath {
namespace ns_xcts {

template <typename eos_t_>
struct eos_type_tag {
    using eos_t = eos_t_;
};

// Common NS checkpoint prefix, members read from the source in disk order.
template <typename space_t>
struct NsCheckpointFieldsFor {
    BeFileSource source;
    space_t space;
    Scalar conf;
    Scalar lapse;
    Vector shift;
    Scalar logh;

    explicit NsCheckpointFieldsFor(const std::string& spacein)
        : source(spacein), space(source), conf(space, source), lapse(space, source),
          shift(space, source), logh(space, source)
    {
    }
};

using NsCheckpointFields = NsCheckpointFieldsFor<Space_spheric_adapted>;

// Initializes the Margherita EOS singleton from bconfig and invokes
// run(eos_type_tag<eos_t>{}) with the selected EOS type.
template <typename config_t, typename run_t>
int dispatch_on_eos(config_t& bconfig, run_t&& run)
{
    const double h_cut = bconfig.template eos<double>(HCUT);
    const std::string eos_file = bconfig.template eos<std::string>(EOSFILE);
    const std::string eos_type = bconfig.template eos<std::string>(EOSTYPE);

    if (eos_type == "Cold_PWPoly") {
        using eos_t = Kadath::Margherita::Cold_PWPoly;
        EOS<eos_t, eos_var_t::PRESSURE>::init(eos_file, h_cut);
        return run(eos_type_tag<eos_t>{});
    } else if (eos_type == "Cold_Table") {
        using eos_t = Kadath::Margherita::Cold_Table;
        const int interp_pts =
            (bconfig.template eos<int>(INTERP_PTS) == 0) ? 2000 : bconfig.template eos<int>(INTERP_PTS);
        EOS<eos_t, eos_var_t::PRESSURE>::init(eos_file, h_cut, interp_pts,
                                             bconfig.template eos<double>(MNUC_CGS));
        return run(eos_type_tag<eos_t>{});
    }
    KADATH_THROW("Unknown EOSTYPE.");
    return EXIT_FAILURE;
}

// Full from-file instantiation: common prefix load, formalism trailing fields,
// EOSTYPE dispatch.
template <typename space_t, typename config_t, typename load_extra_t, typename run_t>
int instantiate_from_file_as(config_t& bconfig, const std::string& spacein, load_extra_t&& load_extra, run_t&& run)
{
    NsCheckpointFieldsFor<space_t> fields(spacein);
    auto extra = load_extra(fields);
    Base_tensor basis(fields.space, CARTESIAN_BASIS);

    return dispatch_on_eos(bconfig, [&](auto tag) { return run(tag, fields, basis, extra); });
}

template <typename config_t, typename load_extra_t, typename run_t>
int instantiate_from_file(config_t& bconfig, const std::string& spacein, load_extra_t&& load_extra, run_t&& run)
{
    return instantiate_from_file_as<Space_spheric_adapted>(
        bconfig, spacein, std::forward<load_extra_t>(load_extra), std::forward<run_t>(run));
}

} // namespace ns_xcts
} // namespace Kadath
