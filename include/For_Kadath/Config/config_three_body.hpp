/*
 * This file is part of the KADATH library.
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

#pragma once
#include "config_bco.hpp"
#include "configurator_toml.hpp"
#include "For_Kadath/Array/exceptions.hpp"
#include <cmath>
#include <memory>
#include <array>
#include <numeric>
#include <sstream>
/**
 * \addtogroup Containers
 * @ingroup Configurator
 * @{*/

/**
 * TRI_INFO
 * Parameter container for the collinear nested three-body system
 * (Space_three_body): a parent bispheric aggregate wrapping one direct body
 * and a child bispheric aggregate of two bodies.
 *
 * Compact-object branch suffixes map to system positions:
 *   <type>1 -> the direct body on the parent level (BCO1)
 *   <type>2 -> first child-aggregate body (BCO2)
 *   <type>3 -> second child-aggregate body (BCO3)
 *
 * Three-body globals (TRI_PARAMS) live in the "three_body" node:
 * parent_distance, child_distance, child_outer_radius, res, rext, ...
 */
class TRI_INFO
{

  private:
    const std::string node_t{"three_body"}; ///< node type
    static constexpr int nbco = 3;

    /**
     * TRI_INFO::init_bco
     * Static switch to determine the type of BCO to point to. Mirrors
     * BIN_INFO::init_bco; extend together with the MBCO map.
     *
     * @param[input]  idx Index of the bco to be stored
     * @param[input]  bco_type String with the bco type (e.g NS, BH)
     */
    void init_bco(const int idx, const std::string bco_type)
    {
        switch (MBCO().at(bco_type)) {
            case NODES::BH:
                BCOS[idx] = std::make_unique<BCO_BH_INFO>();
                break;
            case NODES::NS:
                BCOS[idx] = std::make_unique<BCO_NS_INFO>();
                break;
            default: {
                std::ostringstream oss;
                oss << "Static switch not defined for " << bco_type;
                KADATH_THROW(oss.str());
            }
        }
    }

  protected:
    using Array = std::array<Kadath::ConfigSlot, NUM_TRI_PARAMS_V>;
    using BArray = std::array<std::unique_ptr<BCO_INFO>, nbco>;
    using Map = std::map<std::string, TRI_PARAMS>;

  public:
    BArray BCOS{};                                     ///< Array of pointers to BCOs
    Array tri_params{};                                ///< Array storing three-body parameters
    Map tri_map{MTRI_PARAMS()};                        ///< Map of parameter strings to enum indexes
    std::map<std::string, STAGE> tri_stages{MSTAGE()}; ///< Map of only relevant stages

    /**
     * Default constructor initializing pointers to 0
     * and parameters to NaN.  This is a feature, not a
     * bug.
     */
    TRI_INFO() { tri_params.fill(std::nan("1")); }

    /**
     * Copy constructor. Slightly messy due to handling of pointers to
     * BCO parameter containers.
     */
    TRI_INFO(const TRI_INFO& b) : tri_params{b.tri_params}, tri_stages{b.tri_stages}
    {
        for (int i = 0; i < nbco; ++i) {
            if (b.BCOS[i]) {
                auto b_type = b.BCOS[i]->get_type();
                if (b_type == "bh") {
                    auto child_ptr = dynamic_cast<BCO_BH_INFO*>(b.BCOS[i].get());
                    BCOS[i] = std::make_unique<BCO_BH_INFO>(*child_ptr);
                } else if (b_type == "ns") {
                    auto child_ptr = dynamic_cast<BCO_NS_INFO*>(b.BCOS[i].get());
                    BCOS[i] = std::make_unique<BCO_NS_INFO>(*child_ptr);
                }
            }
        }
    }

    /**
     * Move contructor
     */
    TRI_INFO(TRI_INFO&& b) noexcept : tri_params(std::move(b.tri_params)), tri_stages(std::move(b.tri_stages))
    {
        for (int i = 0; i < nbco; ++i)
            BCOS[i] = std::move(b.BCOS[i]);
    }

    /**
     * Assignment operator
     */
    TRI_INFO& operator=(const TRI_INFO& b)
    {
        if (this == &b)
            return *this;

        TRI_INFO tmp(b);
        *this = std::move(tmp);

        return *this;
    }

    /**
     * Move assignment operator
     */
    TRI_INFO& operator=(TRI_INFO&& b) noexcept
    {
        this->tri_params = std::move(b.tri_params);
        for (int i = 0; i < nbco; ++i)
            this->BCOS[i] = std::move(b.BCOS[i]);
        this->tri_stages = std::move(b.tri_stages);
        return *this;
    }

    /* TRI_INFO::init_three_body
     * initialize the pointers of the BCOs of the three-body system
     *
     * @param[input] bco_types array of strings containing bco types
     */
    void init_three_body(std::array<std::string, nbco> bco_types)
    {
        int idx = 0;
        for (auto& b : bco_types) {
            init_bco(idx, b);
            ++idx;
        }
        set_stage_map(bco_types);
    }

    void set_stage_map(std::array<std::string, nbco> bco_types)
    {
        const bool all_ns = std::all_of(bco_types.begin(), bco_types.end(),
                                        [](const std::string& t) { return t == "ns"; });
        if (all_ns)
            tri_stages = MTRISTAGE();
    }

    /* TRI_INFO::read_params
     * A tree containing all the three-body and bco parameters is received.
     * The parameters related to the three-body globals are stored locally.
     * The BCO types are determined based on the branch nodes
     *
     * @param[input] tri_tree ConfigTree containing all parameters of the system and BCOs
     */
    void read_params(ConfigTree& tri_tree)
    {
        // Read-in strictly three-body related parameters
        read_keys(tri_map, tri_params, tri_tree);

        std::array<std::string, nbco> compact_object_types{};
        std::array<bool, nbco> compact_object_seen{};

        for (const auto& node : tri_tree) {
            if (node.second.empty()) {
                continue;
            }

            const std::string node_name = node.first;
            const std::string compact_object_type = node_name.substr(0, 2);
            if (MBCO().find(compact_object_type) == MBCO().end()) {
                std::ostringstream message;
                message << "Err: " << compact_object_type << " not a recognized node";
                KADATH_THROW(message.str());
            }

            if (node_name.size() < 3 || node_name.back() < '1' || node_name.back() > '0' + nbco) {
                std::ostringstream message;
                message << "Err: compact object branch \"" << node_name
                        << "\" must end in 1, 2 or 3";
                KADATH_THROW(message.str());
            }

            const int compact_object_index = node_name.back() - '1';
            init_bco(compact_object_index, compact_object_type);
            auto compact_object_branch = tri_tree.get_child(node_name);
            BCOS[compact_object_index]->read_params(compact_object_branch);
            compact_object_types[compact_object_index] = compact_object_type;
            compact_object_seen[compact_object_index] = true;
        }

        for (int i = 0; i < nbco; ++i) {
            if (!compact_object_seen[i]) {
                KADATH_THROW("Err: three-body config requires compact object branches ending in 1, 2 and 3");
            }
        }

        set_stage_map(compact_object_types);
    }

    /**
     * TRI_INFO::get_type
     * Returns three-body node type
     *
     * @param[output] node_t node type
     */
    std::string get_type() const { return node_t; }

    /**
     * TRI_INFO::print_me
     * Debug function to print stored node_type
     */
    void print_me() const
    {
        std::cout << node_t << std::endl;
        std::cout << "-----------------------" << std::endl;
    }

    /* TRI_INFO::return_branch
     * Build a tree containing all the three-body and bco parameters.
     * The parameters related to the system globals are built locally.
     * The BCO branches are built by BCO_INFO and added to the tree
     *
     * @param[output] branch ConfigTree containing all parameters of the system and BCOs
     */
    ConfigTree return_branch()
    {
        ConfigTree branch;
        branch.put_child(node_t, build_branch<ConfigTree>(MTRI_PARAMS(), tri_params));
        for (int i = 0; i < nbco; ++i)
            branch.add_child(node_t + "." + BCOS[i]->give_name_string(i + 1), BCOS[i]->return_branch());
        return branch;
    }

    /* TRI_INFO::read_eos_branch / return_eos_branch
     * Every neutron star in the system shares one EOS, stored in a single
     * top-level [eos] block. read_eos_branch fills every compact object's EOS
     * from it (black-hole children are no-ops); return_eos_branch emits the EOS
     * of the first EOS-bearing object (the shared neutron-star EOS).
     */
    void read_eos_branch(ConfigTree& eos_branch)
    {
        for (int i = 0; i < nbco; ++i)
            BCOS[i]->read_eos_branch(eos_branch);
    }
    ConfigTree return_eos_branch()
    {
        for (int i = 0; i < nbco; ++i) {
            ConfigTree eos = BCOS[i]->return_eos_branch();
            if (!eos.empty())
                return eos;
        }
        return ConfigTree{};
    }

    /* TRI_INFO::return_params
     * This function will return the raw arrays, but this is discouraged.
     * It is best to use the overloaded operators to access information
     *
     * @param[output] tri_params Array of three-body parameters
     */
    Array& return_params() { return tri_params; }

    /* TRI_INFO::return_bcos
     * This function will return the raw arrays, but this is discouraged.
     * It is best to use the overloaded operators to access information
     *
     * @param[output] BCOs array of BCO pointers
     */
    BArray& return_bcos() { return BCOS; }

    /* TRI_INFO::get_map
     * Returns three-body parameter map
     *
     * @param[output] tri_map three-body parameter map
     */
    const Map& get_map() const { return tri_map; }

    /* TRI_INFO::get_stage_map
     * Returns three-body parameter stages map
     *
     * @param[output] tri_stages stages map
     */
    const auto& get_stage_map() const { return tri_stages; }

    /* TRI_INFO::get_map
     * Returns BCO parameter map
     *
     * @param[input]  BOCidx BCO pointer index
     * @param[output] bco_map BCO parameter map
     */
    template <typename E> auto& get_map(E BCOidx) const { return BCOS[static_cast<int>(BCOidx)]->get_map(); }
    /* TRI_INFO::operator()
     * This operator returns the requested parameter for a given BCO
     *
     * @param[input]  idx Index of the Paramter of interest - see config_enum
     * @param[input]  BCOidx Index of BCO of interest
     * @param[output] *BCOS[BCOidx])(idx) referene to BCO parameter requested
     */
    template <typename E, typename P> auto& operator()(E idx, P BCOidx)
    {
        return (*BCOS[static_cast<int>(BCOidx)])(idx);
    }

    /* TRI_INFO::operator()
     * This operator returns the requested parameter for the three-body globals
     *
     * @param[input]  idx Index of the Paramter of interest - see config_enum
     * @param[output] tri_params[idx] referene to parameter requested
     */
    template <typename E> auto& operator()(E idx) { return tri_params[static_cast<int>(idx)]; }

    /* TRI_INFO::set_eos_param
     * Returns a reference to a BCO's EOS parameter to set.  Cannot be used
     * to read the value directly since parameters are stored as std::variant.
     *
     * @param[input]  idx Index of the Paramter of interest - see config_enum
     * @param[input]  BCOidx Index of BCO of interest
     * @param[output] eos_param reference to eos parameter to assign
     * @throws std::invalid_argument Throws when dynamic_cast fails
     */
    template <typename E, typename P> auto& set_eos_param(E idx, P BCOidx) const
    {
        if (auto child_ptr = dynamic_cast<BCO_NS_INFO*>(BCOS[static_cast<int>(BCOidx)].get())) {
            return child_ptr->set_eos_param(idx);
        }
        throw std::invalid_argument("\nInvalid EOS Parameter indices for assignment\n");
    }

    /* TRI_INFO::get_eos_param
     * Returns a given EOS parameter
     *
     * @tparam T Parameter indicating the type of the EOS parameter
     * @param[input]  idx Index of the Paramter of interest - see config_enum
     * @param[input]  BCOidx Index of BCO of interest
     * @param[output] eos_param eos parameter value - not assignable.
     * @throws std::invalid_argument Throws when dynamic_cast fails
     */
    template <typename T, typename E, typename P> constexpr T get_eos_param(E idx, P BCOidx) const
    {
        if (auto child_ptr = dynamic_cast<BCO_NS_INFO*>(BCOS[static_cast<int>(BCOidx)].get())) {
            return child_ptr->template get_eos_param<T>(idx);
        }
        throw std::invalid_argument("\nInvalid EOS Parameter indices for reading\n");
    }
    friend std::ostream& operator<<(std::ostream&, const TRI_INFO&);

    /**
     * TRI_INFO::set_defaults
     * Allow the setting of default configurator values for base three-body setup.
     * Geometry defaults are mutually consistent with the converged-TOV outer
     * boundary rout = 2*rmid ~ 14.9 enforced by bco_utils::set_NS_bounds:
     * child_distance > 2*rout, child_outer_radius > child_distance/2 + rout,
     * parent_distance > rout + child_outer_radius, rext > both exposed spheres.
     *
     * @tparam config_t configuration file type
     * @param bconfig reference to configuration file to be modified
     */
    template <typename config_t> void set_defaults(config_t& bconfig)
    {
        bool includes_matter = false;
        // copy Compact Object defaults
        for (auto& bco : {BCO1, BCO2, BCO3}) {
            const auto bcotype = BCOS[to_int(bco)]->get_type();
            if (bcotype == "ns") {
                kadath_config<BCO_NS_INFO> nsconfig;
                nsconfig.set_defaults();
                for (int i = 0; i < NUM_BCO_PARAMS_V; ++i)
                    bconfig.set(i, bco) = nsconfig.set(i);
                for (int i = 0; i < NUM_EOS_PARAMS_V; ++i)
                    bconfig.set_eos(i, bco) = nsconfig.set_eos(i);
                includes_matter = true;
            } else if (bcotype == "bh") {
                kadath_config<BCO_BH_INFO> bhconfig;
                bhconfig.set_defaults();
                for (int i = 0; i < NUM_BCO_PARAMS_V; ++i)
                    bconfig.set(i, bco) = bhconfig.set(i);
            }
        }
        // Set three-body defaults
        bconfig.set(TRI_RES) = 9.;
        bconfig.set(CHILD_DIST) = 34.;
        bconfig.set(CHILD_REXT) = 36.;
        bconfig.set(PARENT_DIST) = 60.;
        bconfig.set(TRI_REXT) = 2. * bconfig(PARENT_DIST);
        bconfig.set(TRI_OUTER_SHELLS) = 0;
        bconfig.set(TRI_GOMEGA) = 0.;

        // Set default fields. No velocity potential: the three-body stage is
        // static (corotating form), so phi is neither solved nor saved.
        bconfig.set_field(SHIFT) = true;
        bconfig.set_field(LAPSE) = true;
        bconfig.set_field(CONF) = true;
        if (includes_matter)
            bconfig.set_field(LOGH) = true;

        bconfig.control(FIXED_GOMEGA) = false;

        bconfig.set_stage(QUASI_EQUIL) = true;

        bconfig.seq_setting(INIT_RES) = 9;
    }
};
/**
 * @}*/

std::ostream& operator<<(std::ostream& out, const TRI_INFO& TRI);
