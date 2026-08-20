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

#include <cmath>
#include <variant>

#include "configurator_toml.hpp"

/**
 * \addtogroup Containers
 * @ingroup Configurator
 * @{*/

/**
 * BCO_INFO
 * Each BCO retains it's own configuration data.
 * This includes the tree that it can update which is later written to file.
 *
 */
class BCO_INFO
{

  private:
    std::string node_t{"bco"}; ///< string containing node type

  protected:
    using Array = std::array<Kadath::ConfigSlot, NUM_BCO_PARAMS_V>;
    using Map = std::map<std::string, BCO_PARAMS>;

    /**
     * std::map for mapping configuration strings to enum indexes
     */
    const Map bco_map{MBCO_PARAMS()};
    std::map<std::string, STAGE> bco_stages{MSTAGE()};

    /**
     *Array to store parameters
     */
    Array bco_params{};

  public:
    virtual ~BCO_INFO() = default;

    /**
     * Constructor.  Everything is set to nan.
     * this is a feature - not a bug
     */
    BCO_INFO()
    {
        bco_params.fill(std::nan("1"));
    }

    /**
     * BCO_INFO copy constructor
     */
    BCO_INFO(const BCO_INFO& b) : bco_stages{b.bco_stages}, bco_params{b.bco_params} {}

    /**
     * BCO_INFO move constructor
     */
    BCO_INFO(BCO_INFO&& b) noexcept
        : bco_stages(std::move(b.bco_stages)), bco_params(std::move(b.bco_params))
    {
    }

    /**
     * BCO_INFO assignment operator
     */
    BCO_INFO& operator=(const BCO_INFO& b)
    {
        this->bco_params = b.bco_params;
        this->bco_stages = b.bco_stages;
        return *this;
    }

    /**
     * BCO_INFO move assignment operator
     */
    BCO_INFO& operator=(BCO_INFO&& b) noexcept
    {
        this->bco_params = std::move(b.bco_params);
        this->bco_stages = std::move(b.bco_stages);
        return *this;
    }

    /**
     * BCO_INFO::return_branch
     * Build a branch based on non-nan parameters
     *
     * param[output] branch branch containing BCO paramters
     */
    virtual ConfigTree return_branch()
    {
        return build_branch<ConfigTree>(MBCO_PARAMS(), bco_params);
    }

    /**
     * BCO_INFO::read_eos_branch / BCO_INFO::return_eos_branch
     * The EOS lives in a shared top-level [eos] block (read and written by
     * kadath_config), not per-object — both stars of a binary always use the
     * same EOS. Base compact objects (black holes) carry no EOS, so these are
     * no-ops here; BCO_NS_INFO overrides them to read/emit the EOS parameters.
     */
    virtual void read_eos_branch(ConfigTree& /*eos_branch*/) {}
    virtual ConfigTree return_eos_branch() { return ConfigTree{}; }

    /**
     * BCO_INFO::give_name_string
     * Returns BCO type with an added int, 1 or 2.  Needed
     * for config_binary.hpp
     *
     * @param[input]  N integer to concatenate with type
     * @param[output] string concatenated string
     */
    virtual std::string give_name_string(const int& N) { return node_t.data() + std::to_string(N); }

    /**
     * BCO_INFO::print_me
     * Debug function to print stored node_type
     */
    virtual void print_me()
    {
        std::cout << node_t << std::endl;
        std::cout << "-----------------------" << std::endl;
    }

    /**
     * BCO_INFO::read_params
     * Store parameters from an input branch
     *
     * param[input] branch branch containing BCO paramters
     */
    virtual void read_params(ConfigTree& branch)
    {
        read_keys(bco_map, bco_params, branch);
        if (std::isnan(bco_params[to_int(DEG)])) {
            bco_params[to_int(DEG)] = 0.0;
        }
    }

    /**
     * BCO_INFO::get_type
     * Returns BCO type
     *
     * @param[output] node_t node type
     */
    virtual std::string get_type() const { return node_t; }

    /**
     * BCO_INFO::get_map
     * Returns BCO map which contains the mapping from configuration
     * strings to enumerator indexes.  See config_enum.hpp
     *
     * @param[output] bco_map Parameter map
     */
    const Map& get_map() { return bco_map; }

    /* BIN_INFO::get_stage_map
     * Returns binary parameter stages map
     *
     * @param[output] bin_map binary parameter map
     */
    virtual const std::map<std::string, STAGE>& get_stage_map() const { return bco_stages; }

    /**
     * BCO_INFO::operator()
     * Returns a given BCO parameter based on a given parameter
     * index contained in bco_map
     *
     * @param[output] bco_params[idx] requested BCO parameter
     */
    template <typename E> auto& operator()(E idx) { return bco_params[static_cast<int>(idx)]; }

    friend std::ostream& operator<<(std::ostream&, const BCO_INFO&);
    friend void print_config_bco_info_with_label(std::ostream&, const BCO_INFO&, const std::string&);
};

/**
 * BCO_BH_INFO
 * Child class containing parameters for a basic BH.
 * We also store default parameters which is used for an initial setup.
 */

class BCO_BH_INFO : public BCO_INFO
{
  private:
    std::string node_t = "bh"; ///< string containing node type

  public:
    BCO_BH_INFO() : BCO_INFO() { bco_stages = MBHSTAGE(); }
    /**
     * BCO_INFO copy constructor
     */
    BCO_BH_INFO(const BCO_BH_INFO& b) : BCO_INFO(b) {}

    /**
     * BCO_INFO move constructor
     */
    BCO_BH_INFO(BCO_BH_INFO&& b) noexcept : BCO_INFO(std::move(b)) {}

    /**
     * BCO_INFO assignment operator
     */
    BCO_BH_INFO& operator=(BCO_BH_INFO& b)
    {
        this->bco_params = b.bco_params;
        this->bco_stages = b.bco_stages;
        return *this;
    }

    /**
     * BCO_INFO move assignment operator
     */
    BCO_BH_INFO& operator=(const BCO_BH_INFO&& b) noexcept
    {
        this->bco_params = std::move(b.bco_params);
        this->bco_stages = std::move(b.bco_stages);
        return *this;
    }

    /**
     * BCO_BH_INFO::give_name_string
     * Returns BCO type with an added int, 1 or 2.  Needed
     * for config_binary.hpp
     *
     * @param[input]  N integer to concatenate with type
     * @param[output] string concatenated string
     */
    virtual std::string give_name_string(const int& N) override { return node_t.data() + std::to_string(N); }

    /**
     * BCO_BH_INFO::print_me
     * Debug function to print stored node_type
     */
    virtual void print_me() override
    {
        std::cout << node_t << std::endl;
        std::cout << "-----------------------" << std::endl;
    }

    /**
     * BCO_BH_INFO::get_type
     * Returns BCO type
     *
     * @param[output] node_t node type
     */
    virtual std::string get_type() const override { return node_t; }

    /**
     * BCO_BH_INFO::set_defaults
     * Allow the setting of default configurator values for a base BH setup - do not modify
     *
     * @tparam config_t configuration file type
     * @param bconfig reference to configuration file to be modified
     */
    template <typename config_t> void set_defaults(config_t& bconfig)
    {
        // start - set BH properties in config file
        // Resolution of the initial setup
        bconfig.set(BCO_RES) = 9;
        bconfig.set(DIM) = 3;

        // Units of the system - 4 * PI * G

        // Nucleus Radius
        bconfig.set(RIN) = 0.1;

        // Initial guess for BH - radius
        bconfig.set(RMID) = 0.3;

        // Radius of the inner adapted domain outer boundary
        bconfig.set(ROUT) = 5 * bconfig(RMID);

        // Additional shells subdivide the fixed outer band for refined resolution
        bconfig.set(NSHELLS) = 0;

        // Initial dimensionless spin
        bconfig.set(CHI) = 0;
        bconfig.set(DEG) = 0;

        // Initial guess of omega
        bconfig.set(OMEGA) = 0;

        // MIRR and MCH are fixing parameters...this is what the resulting BH will be
        bconfig.set(MIRR) = 1.;
        bconfig.set(MCH) = 1.;

        bconfig.set(BVELX) = 0.;
        bconfig.set(BVELY) = 0.;
        bconfig.set(TRUMPET_BH_SEED) = false;

        /**
         * fixed lapse is used for the DIRICHLET_LAPSE stage only.
         * Once the system is solved using the Neumann lapse condition
         * (VON_NEUMANN stage) this value is updated in the config file in the
         * standard solver */
        bconfig.set(FIXED_LAPSE) = .3;
        // end   - set BH parameters

        // start - set BH stages in config file
        bconfig.set_stage(DIRICHLET_LAPSE) = true;
        bconfig.set_stage(VON_NEUMANN) = true;
        // end   - set BH stages

        // start - set BH fields in config file
        bconfig.set_field(CONF) = true;
        bconfig.set_field(LAPSE) = true;
        bconfig.set_field(SHIFT) = true;
        // end   - set BH fields

        bconfig.seq_setting(INIT_RES) = 9;
    }
};

/**
 * BCO_NS_INFO
 * Child class containing parameters for a NS
 * We also store default parameters which is used for an initial setup.
 *
 * EOSType is a std::variant container that allows the storage of various
 * types while being type safe.  This is extremely important when
 * considering the EOS could be simply a file name for a table, or this
 * can be used to store information relation to a piecewise polytrope EOS.
 * This leaves a lot of room for modification later while only needing to
 * modify config_enums for the added parameters.
 *
 * EOSMap is an additional map that only handles EOS parameters.
 */

class BCO_NS_INFO : public BCO_INFO
{
    using vars_t = std::variant<double, int, std::string>;
    using EOSArray = std::array<vars_t, NUM_EOS_PARAMS_V>;
    using EOSMap = std::map<std::string, EOS_PARAMS>;
    using DIFFROT_ary = std::array<vars_t, NUM_DIFFROT_PARAMS_V>;
    using DIFFROT_map = std::map<std::string, DIFFROT_PARAMS>;

  private:
    std::string node_t{"ns"}; ///< node type

    EOSArray eos_params{}; ///< Array storing EOS parameters
    DIFFROT_ary diffrot_params{};

    // link the naming in the .toml file to the numerical parameters
    const EOSMap eos_map{MEOS_PARAMS()};
    const DIFFROT_map diffrot_map{MDIFFROT_PARAMS()};

  public:
    /**
     * BCO_NS_INFO::BCO_NS_INFO
     * In addition to the parent constructor, we initialize the
     * the EOS array to NaN as well.
     */
    BCO_NS_INFO() : BCO_INFO()
    {
        bco_stages = MNSSTAGE();
        eos_params.fill(std::nan("1"));
        diffrot_params.fill(std::nan("1"));
    }

    /**
     * BCO_NS_INFO copy constructor
     */
    BCO_NS_INFO(const BCO_NS_INFO& b) : BCO_INFO(b), eos_params{b.eos_params}, diffrot_params(b.diffrot_params) {}

    /**
     * BCO_NS_INFO move constructor
     */
    BCO_NS_INFO(BCO_NS_INFO&& b) noexcept
        : BCO_INFO(std::move(b)), eos_params(std::move(b.eos_params)), diffrot_params(std::move(b.diffrot_params))
    {
    }

    /**
     * BCO_NS_INFO assignment operator
     */
    BCO_NS_INFO& operator=(const BCO_NS_INFO& b)
    {
        this->bco_params = b.bco_params;
        this->bco_stages = b.bco_stages;
        this->eos_params = b.eos_params;
        this->diffrot_params = b.diffrot_params;
        return *this;
    }

    /**
     * BCO_NS_INFO move assignment operator
     */
    BCO_NS_INFO& operator=(BCO_NS_INFO&& b) noexcept
    {
        this->bco_params = std::move(b.bco_params);
        this->bco_stages = std::move(b.bco_stages);
        this->eos_params = std::move(b.eos_params);
        this->diffrot_params = std::move(b.diffrot_params);
        return *this;
    }

    /**
     * BCO_NS_INFO::return_eos_params
     * Return a reference to the EOS array.  Mainly for testing.
     * Recommended to use get/set_eos_param for safety
     * param[out] eos_params
     */
    const EOSArray& return_eos_params() const { return eos_params; }
    const DIFFROT_ary& return_diffrot_params() const { return diffrot_params; }
    /**
     * BCO_NS_INFO::return_eos_map
     * Return a reference to the EOS parameter map
     * param[out] eos_params
     */
    const EOSMap& return_eos_map() const { return eos_map; }
    const DIFFROT_map& return_diffrot_map() const { return diffrot_map; }

    /**
     * BCO_NS_INFO::give_name_string
     * Returns BCO type with an added int, 1 or 2.  Needed
     * for config_binary.hpp
     *
     * @param[input]  N integer to concatenate with type
     * @param[output] string concatenated string
     */
    virtual std::string give_name_string(const int& N) override { return node_t.data() + std::to_string(N); }

    /**
     * BCO_NS_INFO::print_me
     * Debug function to print stored node_type
     */
    virtual void print_me() override
    {
        std::cout << node_t << std::endl;
        std::cout << "-----------------------" << std::endl;
    }

    /**
     * BCO_NS_INFO::read_params
     * Reads the mapped parameters from a given branch
     *
     * @param[input] branch input branch to read bco and eos params from
     */
    virtual void read_params(ConfigTree& branch) override
    {
        read_keys(bco_map, bco_params, branch);
        read_keys(eos_map, eos_params, branch);
        read_keys(diffrot_map, diffrot_params, branch);
        if (std::isnan(bco_params[to_int(DEG)])) {
            bco_params[to_int(DEG)] = 0.0;
        }
    }

    /**
     * BCO_NS_INFO::return_branch
     * Builds a single branch from the stored bco and eos parameters
     * @param[output] branch containing bco and eos params
     */
    //
    virtual ConfigTree return_branch() override
    {
        ConfigTree branch(build_branch<ConfigTree>(bco_map, bco_params));

        // EOS is no longer flattened into the per-object branch; it is emitted
        // once into the shared top-level [eos] block (see return_eos_branch).
        ConfigTree diffrot_childs = build_branch<ConfigTree>(diffrot_map, diffrot_params);
        for (auto child2 : diffrot_childs)
            branch.push_back(child2);
        return branch;
    }

    /**
     * BCO_NS_INFO::read_eos_branch / return_eos_branch
     * The EOS is shared across all neutron stars of a system and lives in a
     * single top-level [eos] block. read_eos_branch fills this star's EOS
     * parameters from that block; return_eos_branch emits them back into it.
     */
    void read_eos_branch(ConfigTree& eos_branch) override { read_keys(eos_map, eos_params, eos_branch); }
    ConfigTree return_eos_branch() override { return build_branch<ConfigTree>(eos_map, eos_params); }

    /**
     * BCO_NS_INFO::get_type
     * Returns BCO type
     *
     * @param[output] node_t node type
     */
    virtual std::string get_type() const override { return node_t; }

    /**
     * BCO_NS_INFO::set_eos_param
     * Returns reference to EOS param allowing value assignment
     * Note type is EOSType which is a std::variant
     *
     * @param[output] eos_param[idx]
     */
    template <typename E> auto& set_eos_param(E idx) { return eos_params[static_cast<int>(idx)]; }
    template <typename E> auto& set_diffrot_param(E idx) { return diffrot_params[static_cast<int>(idx)]; }

    /**
     * BCO_NS_INFO::get_eos_param
     * For accessing values from a std::variant, we need to specify
     * the expected type to access it. Returns are value only.
     *
     * @tparam T type of EOS parameter - See EOSType for options
     * @param[output] eos_param[idx]
     */
    template <typename T, typename E> const T get_eos_param(E idx) const
    {
        return std::get<T>(eos_params[static_cast<int>(idx)]);
    }

    template <typename T, typename E> const T get_diffrot_param(E idx) const
    {
        return std::get<T>(diffrot_params[static_cast<int>(idx)]);
    }
    /**
     * BCO_NS_INFO::set_defaults
     * Allow the setting of default configurator values for a base NS setup - do not modify
     *
     * @tparam config_t configuration file type
     * @param bconfig reference to configuration file to be modified
     */
    template <typename config_t> void set_defaults(config_t& bconfig)
    {
        // start - set NS properties in config file
        bconfig.set_eos(EOSFILE) = "dd2.lorene";
        bconfig.set_eos(EOSTYPE) = "Cold_Table";
        bconfig.set_eos(HCUT) = 0.0;
        bconfig.set_eos(INTERP_PTS) = 2000;
        // Nuclear mass unit (g) for the table n->rho conversion; default is the
        // atomic mass unit (= 931.494 MeV/c^2), matching the Margherita constant.
        // See lorene_io.hh.
        bconfig.set_eos(MNUC_CGS) = 1.660539040e-24;
        bconfig.set_diffrot(DIFF_LAW) = "uniform";
        bconfig.set_diffrot(CSTA) = 1;
        bconfig.set_diffrot(CSTP) = 1;
        bconfig.set(HC) = 1.26;
        bconfig.set(NC) = 1.37e-3;

        // Resolution of the initial setup
        bconfig.set(BCO_RES) = 9;
        bconfig.set(DIM) = 3;

        // Units of the system - 4 * PI * G

        // Initial guess for NS - radius
        bconfig.set(RMID) = 6.2;

        // Nucleus Radius
        bconfig.set(RIN) = 0.5 * bconfig(RMID);

        // Radius of the inner adapted domain outer boundary
        bconfig.set(ROUT) = 1.5 * bconfig(RMID);

        // Additional shells between RIN and RMID
        bconfig.set(NINSHELLS) = 0;
        // Additional shells between ROUT and the fixed outer boundary
        bconfig.set(NSHELLS) = 0;

        // Initial dimensionless spin
        bconfig.set(CHI) = 0;
        bconfig.set(DEG) = 0;

        // Initial guess of omega
        bconfig.set(OMEGA) = 0;

        // We initialize based on fixed MADM
        bconfig.set(MADM) = 1.4;
        bconfig.set(QLMADM) = 1.4;
        bconfig.set(MB) = 1.55;
        // end   - set NS parameters

        // start - set NS stages in config file
        bconfig.set_stage(NOROT) = true;
        bconfig.set_stage(UNIROT) = true;
        // end   - set NS stages

        // start - set NS fields in config file
        bconfig.set_field(SHIFT) = true;
        bconfig.set_field(LAPSE) = true;
        bconfig.set_field(CONF) = true;
        bconfig.set_field(LOGH) = true;
        // end   - set NS fields

        bconfig.seq_setting(INIT_RES) = 9;
    }
};

/**
 * BCO_KSBH_INFO
 * Child class containing parameters for a KerrSchild BH.
 * We also store default parameters which is used for an initial setup.
 */

class BCO_KSBH_INFO : public BCO_INFO
{
  private:
    std::string node_t = "bh"; ///< string containing node type

  public:
    BCO_KSBH_INFO() : BCO_INFO() { bco_stages = MKSBHSTAGE(); }
    BCO_KSBH_INFO(const BCO_KSBH_INFO& b) : BCO_INFO(b) {}
    BCO_KSBH_INFO(BCO_KSBH_INFO&& b) noexcept : BCO_INFO(std::move(b)) {}
    BCO_KSBH_INFO& operator=(BCO_KSBH_INFO& b)
    {
        this->bco_params = b.bco_params;
        this->bco_stages = b.bco_stages;
        return *this;
    }
    BCO_KSBH_INFO& operator=(const BCO_KSBH_INFO&& b) noexcept
    {
        this->bco_params = std::move(b.bco_params);
        this->bco_stages = std::move(b.bco_stages);
        return *this;
    }

    /**
     * BCO_KSBH_INFO::give_name_string
     * Returns BCO type with an added int, 1 or 2.  Needed
     * for config_binary.hpp
     *
     * @param[input]  N integer to concatenate with type
     * @param[output] string concatenated string
     */
    virtual std::string give_name_string(const int& N) override { return node_t.data() + std::to_string(N); }

    /**
     * BCO_KSBH_INFO::print_me
     * Debug function to print stored node_type
     */
    virtual void print_me() override
    {
        std::cout << node_t << std::endl;
        std::cout << "-----------------------" << std::endl;
    }

    /**
     * BCO_KSBH_INFO::get_type
     * Returns BCO type
     *
     * @param[output] node_t node type
     */
    virtual std::string get_type() const override { return node_t; }

    /**
     * BCO_KSBH_INFO::set_defaults
     * Allow the setting of default configurator values for a base BH setup - do not modify
     *
     * @tparam config_t configuration file type
     * @param bconfig reference to configuration file to be modified
     */
    template <typename config_t> void set_defaults(config_t& bconfig)
    {
        // start - set BH properties in config file
        // Resolution of the initial setup
        bconfig.set(BCO_RES) = 9;
        bconfig.set(DIM) = 3;

        // Units of the system - 4 * PI * G

        // Nucleus Radius
        bconfig.set(RIN) = 1.;

        // Initial guess for BH - radius
        bconfig.set(RMID) = 2.;

        // Radius of the inner adapted domain outer boundary
        bconfig.set(ROUT) = 2. * bconfig(RMID);

        // Additional shells subdivide the fixed outer band for refined resolution
        bconfig.set(NSHELLS) = 1;

        // Initial dimensionless spin
        bconfig.set(CHI) = 0;
        bconfig.set(DEG) = 0;

        // Initial guess of omega
        bconfig.set(OMEGA) = 0;

        // MIRR and MCH are fixing parameters...this is what the resulting BH will be
        bconfig.set(MIRR) = 1.;
        bconfig.set(MCH) = 1.;
        bconfig.set(KERR_MCH) = 1.;
        bconfig.set(KERR_CHI) = 0.;

        bconfig.set(BVELX) = 0.;
        bconfig.set(BVELY) = 0.;
        // end   - set BH parameters

        // start - set BH stages in config file
        bconfig.set_stage(TRUMPET) = true;
        // end   - set BH stages

        // start - set BH fields in config file
        bconfig.set_field(CONF) = true;
        bconfig.set_field(LAPSE) = true;
        bconfig.set_field(SHIFT) = true;
        bconfig.set_field(KS_K) = true;
        bconfig.set_field(KS_LAPSE) = true;
        bconfig.set_field(KS_METRIC) = true;
        // end   - set BH fields

        bconfig.seq_setting(INIT_RES) = 9;
    }
};
/**
 * @}*/

std::ostream& operator<<(std::ostream& out, const BCO_INFO& BCO);
