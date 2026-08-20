/*
 * This file is part of the KADATH library.
 * Author: Samuel Tootle
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
 * BIN_INFO
 * Locally store the parameters related to the binary
 * Store pointers to related BCOs which store their own parameters.
 *
 */
class BIN_INFO
{

  private:
    const std::string node_t{"binary"}; ///< node type

    /**
     * BIN_INFO::init_bco
     * Static switch to determine the type of BCO to point to.  If one intends
     * to implement new BCOs, this needs to be updated along with MBCO map.
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
    using Array = std::array<Kadath::ConfigSlot, NUM_BPARAMS_V>;
    using BArray = std::array<std::unique_ptr<BCO_INFO>, 2>;
    using Map = std::map<std::string, BIN_PARAMS>;

  public:
    BArray BCOS{};                                     ///< Array of pointers to BCOs
    Array bin_params{};                                ///< Array storing Binary parameters
    Map bin_map{MBIN_PARAMS()};                        ///< Map of parameter strings to enum indexes
    std::map<std::string, STAGE> bin_stages{MSTAGE()}; ///< Map of only relevant stages

    /**
     * Default constructor initializing pointers to 0
     * and parameters to NaN.  This is a feature, not a
     * bug.
     */
    BIN_INFO() { bin_params.fill(std::nan("1")); }

    /**
     * Copy constructor. Slightly messy due to handling of pointers to
     * BCO parameter containers.
     */
    BIN_INFO(const BIN_INFO& b) : bin_params{b.bin_params}, bin_stages{b.bin_stages}
    {
        for (int i = 0; i < 2; ++i) {
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
    BIN_INFO(BIN_INFO&& b) noexcept : bin_params(std::move(b.bin_params)), bin_stages(std::move(b.bin_stages))
    {
        for (int i = 0; i < 2; ++i)
            BCOS[i] = std::move(b.BCOS[i]);
    }

    /**
     * Assignment operator
     */
    BIN_INFO& operator=(const BIN_INFO& b)
    {
        if (this == &b)
            return *this;

        BIN_INFO tmp(b);
        *this = std::move(tmp);

        return *this;
    }

    /**
     * Move assignment operator
     */
    BIN_INFO& operator=(BIN_INFO&& b) noexcept
    {
        this->bin_params = std::move(b.bin_params);
        for (int i = 0; i < 2; ++i)
            this->BCOS[i] = std::move(b.BCOS[i]);
        this->bin_stages = std::move(b.bin_stages);
        return *this;
    }

    /* BIN_INFO::init_binary
     * initialize the pointers of the BCOs of the binary
     *
     * @param[input] bco_types vector of strings containing bco types
     */
    void init_binary(std::array<std::string, 2> bco_types)
    {
        int idx = 0;
        for (auto& b : bco_types) {
            init_bco(idx, b);
            ++idx;
        }
        set_stage_map(bco_types);
    }

    void set_stage_map(std::array<std::string, 2> bco_types)
    {
        if (bco_types[0] == bco_types[1]) { // check for BBH or BNS
            if (bco_types[0] == "ns")
                bin_stages = MBNSSTAGE();
            if (bco_types[0] == "bh")
                bin_stages = MBBHSTAGE();
        } else if (bco_types[0] == "ns" && bco_types[1] == "bh")
            bin_stages = MBHNSSTAGE();
    }

    /* BIN_INFO::read_params
     * A tree containing all the binary and bco parameters is received.
     * The parameters related to the binary are stored locally.
     * The BCO types are determined based on the branch nodes
     *
     * @param[input] bin_tree ConfigTree containing all parameters of the binary and BCOs
     */
    void read_params(ConfigTree& bin_tree)
    {
        // Read-in strictly binary related parameters
        read_keys(bin_map, bin_params, bin_tree);

        std::array<std::string, 2> compact_object_types{};
        std::array<bool, 2> compact_object_seen{};

        for (const auto& node : bin_tree) {
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

            if (node_name.size() < 3 || (node_name.back() != '1' && node_name.back() != '2')) {
                std::ostringstream message;
                message << "Err: compact object branch \"" << node_name
                        << "\" must end in 1 or 2";
                KADATH_THROW(message.str());
            }

            const int compact_object_index = node_name.back() - '1';
            init_bco(compact_object_index, compact_object_type);
            auto compact_object_branch = bin_tree.get_child(node_name);
            BCOS[compact_object_index]->read_params(compact_object_branch);
            compact_object_types[compact_object_index] = compact_object_type;
            compact_object_seen[compact_object_index] = true;
        }

        if (!compact_object_seen[0] || !compact_object_seen[1]) {
            KADATH_THROW("Err: binary config requires compact object branches ending in 1 and 2");
        }

        set_stage_map(compact_object_types);
    }

    /**
     * BIN_INFO::get_type
     * Returns binary type
     *
     * @param[output] node_t node type
     */
    std::string get_type() const { return node_t; }

    /**
     * BIN_INFO::print_me
     * Debug function to print stored node_type
     */
    void print_me() const
    {
        std::cout << node_t << std::endl;
        std::cout << "-----------------------" << std::endl;
    }

    /* BIN_INFO::return_branch
     * Build a tree containing all the binary and bco parameters.
     * The parameters related to the binary are built locally.
     * The BCO branches are built by BCO_INFO and added to the tree
     *
     * @param[output] branch ConfigTree containing all parameters of the binary and BCOs
     */
    ConfigTree return_branch()
    {
        ConfigTree branch;
        branch.put_child(node_t, build_branch<ConfigTree>(MBIN_PARAMS(), bin_params));
        branch.add_child(node_t + "." + BCOS[0]->give_name_string(1), BCOS[0]->return_branch());
        branch.add_child(node_t + "." + BCOS[1]->give_name_string(2), BCOS[1]->return_branch());
        return branch;
    }

    /* BIN_INFO::read_eos_branch / return_eos_branch
     * The two stars of a binary always share one EOS, stored in a single
     * top-level [eos] block. read_eos_branch fills every compact object's EOS
     * from that block (black-hole children are no-ops); return_eos_branch emits
     * the EOS of the first EOS-bearing object (i.e. the shared neutron-star EOS).
     */
    void read_eos_branch(ConfigTree& eos_branch)
    {
        for (auto& bco : BCOS)
            bco->read_eos_branch(eos_branch);
    }
    ConfigTree return_eos_branch()
    {
        for (auto& bco : BCOS) {
            ConfigTree eos = bco->return_eos_branch();
            if (!eos.empty())
                return eos;
        }
        return ConfigTree{};
    }

    /* BIN_INFO::return_params
     * This function will return the raw arrays, but this is discouraged.
     * It is best to use the overloaded operators to access information
     *
     * @param[output] bin_params Array of binary parameters
     */
    Array& return_params() { return bin_params; }

    /* BIN_INFO::return_bcos
     * This function will return the raw arrays, but this is discouraged.
     * It is best to use the overloaded operators to access information
     *
     * @param[output] BCOs array of BCO pointers
     */
    BArray& return_bcos() { return BCOS; }

    /* BIN_INFO::get_map
     * Returns binary parameter map
     *
     * @param[output] bin_map binary parameter map
     */
    const Map& get_map() const { return bin_map; }

    /* BIN_INFO::get_stage_map
     * Returns binary parameter stages map
     *
     * @param[output] bin_map binary parameter map
     */
    const auto& get_stage_map() const { return bin_stages; }

    /* BIN_INFO::get_map
     * Returns BCO parameter map
     *
     * @param[input]  BOCidx BCO pointer index
     * @param[output] bco_map binary parameter map
     */
    template <typename E> auto& get_map(E BCOidx) const { return BCOS[static_cast<int>(BCOidx)]->get_map(); }
    /* BIN_INFO::operator()
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

    /* BIN_INFO::operator()
     * This operator returns the requested parameter for the binary
     *
     * @param[input]  idx Index of the Paramter of interest - see config_enum
     * @param[output] bin_params[idx] referene to binary parameter requested
     */
    template <typename E> auto& operator()(E idx) { return bin_params[static_cast<int>(idx)]; }

    /* BIN_INFO::set_eos_param
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

    /* BIN_INFO::get_eos_param
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
    friend std::ostream& operator<<(std::ostream&, const BIN_INFO&);

    /**
     * BIN_INFO::set_defaults
     * Allow the setting of default configurator values for base binary setup
     *
     * @tparam config_t configuration file type
     * @param bconfig reference to configuration file to be modified
     */
    template <typename config_t> void set_defaults(config_t& bconfig)
    {
        bool includes_matter = false;
        int ns_count = 0;
        std::array<double, 2> Ms{};
        // copy Compact Object defaults
        for (auto& bco : {BCO1, BCO2}) {
            const auto bcotype = BCOS[to_int(bco)]->get_type();
            if (bcotype == "ns") {
                ++ns_count;
                kadath_config<BCO_NS_INFO> nsconfig;
                nsconfig.set_defaults();
                for (int i = 0; i < NUM_BCO_PARAMS_V; ++i)
                    bconfig.set(i, bco) = nsconfig.set(i);
                for (int i = 0; i < NUM_EOS_PARAMS_V; ++i)
                    bconfig.set_eos(i, bco) = nsconfig.set_eos(i);
                includes_matter = true;
                Ms[to_int(bco)] = bconfig(MADM, bco);
            } else if (bcotype == "bh") {
                kadath_config<BCO_BH_INFO> bhconfig;
                bhconfig.set_defaults();
                for (int i = 0; i < NUM_BCO_PARAMS_V; ++i)
                    bconfig.set(i, bco) = bhconfig.set(i);
                Ms[to_int(bco)] = bconfig(MCH, bco);
            }
        }
        // Set binary defaults
        bconfig.set(BIN_RES) = 9.;
        bconfig.set(DIST) = 40.;
        bconfig.set(REXT) = 2. * bconfig(DIST);
        bconfig.set(Q) = Ms[1] / Ms[0];
        bconfig.set(COM) = 0.;
        bconfig.set(COMY) = 0.;
        bconfig.set(COM_VELZ) = 0.;
        bconfig.set(OUTER_SHELLS) = 0;
        bconfig.set(GOMEGA) = 0.;
        // Blended binary bispheric radius: r_bisph = RMID + fill*(dist/2 - RMID).
        // With no per-object shells, ROUT is also r_bisph. Nonzero shells
        // subdivide the fixed band between RMID and r_bisph.
        bconfig.set(RBISPH_FILL) = 1. / 3.;

        // Set default fields
        bconfig.set_field(SHIFT) = true;
        bconfig.set_field(LAPSE) = true;
        bconfig.set_field(CONF) = true;
        if (includes_matter) {
            bconfig.set_field(LOGH) = true;
            bconfig.set_field(PHI) = true;
        }

        bconfig.control(FIXED_GOMEGA) = false;
        bconfig.control(USE_PN) = true;

        // Binary production default: fixed-PN-Omega hydro-rescale, free-Omega
        // force-balance, then eccentricity-reduction rescale.
        if (ns_count == 2) {
            bconfig.set_stage(QUASI_EQUIL) = true;
            bconfig.set_stage(FORCE_BALANCE) = true;
            bconfig.set_stage(ECC_RED) = true;
        } else {
            bconfig.set_stage(QUASI_EQUIL) = true;
            bconfig.set_stage(FORCE_BALANCE) = true;
            bconfig.set_stage(ECC_RED) = true;
        }

        bconfig.seq_setting(INIT_RES) = 9;
    }
};
/**
 * @}*/

std::ostream& operator<<(std::ostream& out, const BIN_INFO& BIN);
