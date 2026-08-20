/*
    Copyright 2019 Samuel Tootle

    This file is part of Kadath.

    Kadath is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    Kadath is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with Kadath.  If not, see <http://www.gnu.org/licenses/>.
*/

#include "For_Kadath/Config/config_enums.hpp"

#include <algorithm>
#include <array>
#include <map>
#include <string>
#include <string_view>

namespace
{
    template <typename Enum> struct ConfigEnumEntry
    {
        std::string_view key;
        Enum value;
    };

    template <typename Enum, std::size_t EntryCount>
    std::map<std::string, Enum> make_config_map(const std::array<ConfigEnumEntry<Enum>, EntryCount>& entries)
    {
        std::map<std::string, Enum> result;
        for (const auto& entry : entries) {
            if (!entry.key.empty()) {
                result.emplace(std::string(entry.key), entry.value);
            }
        }
        return result;
    }

    template <typename Enum, std::size_t EntryCount, std::size_t SelectedCount>
    std::map<std::string, Enum>
    make_selected_config_map(const std::array<ConfigEnumEntry<Enum>, EntryCount>& entries,
                             const std::array<Enum, SelectedCount>& selected_entries)
    {
        std::map<std::string, Enum> result;
        for (Enum selected_entry : selected_entries) {
            const auto entry = std::find_if(entries.begin(), entries.end(),
                                           [selected_entry](const auto& candidate) {
                                               return candidate.value == selected_entry;
                                           });
            if (entry != entries.end() && !entry->key.empty()) {
                result.emplace(std::string(entry->key), entry->value);
            }
        }
        return result;
    }

    /*
     * X-macro emit helpers: invoked by KADATH_*_ENTRIES(APPLY, CTX) lists where
     * APPLY receives (CTX, enumerator_name, key_string).
     */
#define KADATH_COUNT_ENTRY(CTX, name, key) +1
#define KADATH_AS_ENTRY_PAIR(EnumName, name, key) ConfigEnumEntry<EnumName>{key, EnumName::name},

    /*
     * Per-enum: declare a static_assert that the entry-count matches the enum
     * sentinel, then materialize a constexpr std::array<ConfigEnumEntry<E>, N>
     * of (key, value) pairs in enum order. EmptyKey entries remain present so
     * array indices line up with enumerator ordinals, but they are filtered
     * out at map construction time.
     */
#define KADATH_DEFINE_ENTRIES_ARRAY(EnumName, NumSym, EntriesMacro, ArrayName, FnName, HasDefault)            \
    static_assert((0 EntriesMacro(KADATH_COUNT_ENTRY, /*ctx*/)) == NumSym##_V,                                 \
                  #EnumName " entry count drift: KADATH_" #EnumName " entry list out of sync with enum body"); \
    static constexpr std::array<ConfigEnumEntry<EnumName>, NumSym##_V> ArrayName = {{                          \
        EntriesMacro(KADATH_AS_ENTRY_PAIR, EnumName)                                                           \
    }};

    KADATH_CONFIG_ENUMS(KADATH_DEFINE_ENTRIES_ARRAY)
#undef KADATH_DEFINE_ENTRIES_ARRAY
} // namespace

/*
 * Default map accessors: one full-map MFoo() per enum that opts in via
 * HasDefault == 1 in the registry. Curated subset accessors (MBNSSTAGE,
 * MBCO_SFIELDS_*, etc.) are hand-written below.
 */
#define KADATH_DEFINE_DEFAULT_FN_1(EnumName, FnName, ArrayName)             \
    const std::map<std::string, EnumName>& FnName()                         \
    {                                                                       \
        static const auto value = make_config_map(ArrayName);               \
        return value;                                                       \
    }
#define KADATH_DEFINE_DEFAULT_FN_0(EnumName, FnName, ArrayName) /* skip */
#define KADATH_DEFINE_DEFAULT_FN_DISPATCH(EnumName, FnName, ArrayName, HasDefault) \
    KADATH_DEFINE_DEFAULT_FN_##HasDefault(EnumName, FnName, ArrayName)
#define KADATH_DEFINE_DEFAULT_FN(EnumName, NumSym, EntriesMacro, ArrayName, FnName, HasDefault) \
    KADATH_DEFINE_DEFAULT_FN_DISPATCH(EnumName, FnName, ArrayName, HasDefault)

KADATH_CONFIG_ENUMS(KADATH_DEFINE_DEFAULT_FN)
#undef KADATH_DEFINE_DEFAULT_FN

/* ---------------------------------------------------------------------- *
 * Curated subset accessors. Each lists the active enumerators by value;
 * make_selected_config_map walks the per-enum entries array and emits only
 * the matching key->value pairs (skipping empty-key entries).
 * ---------------------------------------------------------------------- */

const std::map<std::string, NODES>& M_REQ_NODES()
{
    static constexpr std::array<NODES, 4> required_nodes = {
        NODES::FIELDS,
        NODES::STAGES,
        NODES::SCONTROLS,
        NODES::SSETTINGS,
    };
    static const auto value = make_selected_config_map(node_entries, required_nodes);
    return value;
}

const std::map<std::string, NODES>& MBCO()
{
    static constexpr std::array<NODES, 2> compact_object_nodes = {
        NODES::BH,
        NODES::NS,
    };
    static const auto value = make_selected_config_map(node_entries, compact_object_nodes);
    return value;
}

const std::map<std::string, BCO_FIELDS>& MBCO_SFIELDS_0()
{
    static constexpr std::array<BCO_FIELDS, 3> zero_scalar_fields = {
        BCO_FIELDS::LOGH,
        BCO_FIELDS::Omg,
        BCO_FIELDS::NDENS,
    };
    static const auto value = make_selected_config_map(field_entries, zero_scalar_fields);
    return value;
}

const std::map<std::string, BCO_FIELDS>& MBCO_SFIELDS_1()
{
    static constexpr std::array<BCO_FIELDS, 7> unit_scalar_fields = {
        BCO_FIELDS::CONF, BCO_FIELDS::LAPSE, BCO_FIELDS::ENTH, BCO_FIELDS::NU,
        BCO_FIELDS::INCA, BCO_FIELDS::BIGA,  BCO_FIELDS::NP,
    };
    static const auto value = make_selected_config_map(field_entries, unit_scalar_fields);
    return value;
}

const std::map<std::string, BCO_FIELDS>& MBCO_VFIELDS()
{
    static constexpr std::array<BCO_FIELDS, 2> vector_fields = {
        BCO_FIELDS::SHIFT,
        BCO_FIELDS::PHI,
    };
    static const auto value = make_selected_config_map(field_entries, vector_fields);
    return value;
}

const std::map<std::string, STAGE>& MBNSSTAGE()
{
    static constexpr std::array<STAGE, 3> binary_neutron_star_stages = {
        STAGE::QUASI_EQUIL,
        STAGE::FORCE_BALANCE,
        STAGE::ECC_RED,
    };
    static const auto value = make_selected_config_map(stage_entries, binary_neutron_star_stages);
    return value;
}

const std::map<std::string, STAGE>& MTRISTAGE()
{
    // ThreeBody keeps the binary three-stage map; its current production stage
    // is its QUASI_EQUIL analog.
    static constexpr std::array<STAGE, 3> three_body_stages = {
        STAGE::FORCE_BALANCE,
        STAGE::QUASI_EQUIL,
        STAGE::ECC_RED,
    };
    static const auto value = make_selected_config_map(stage_entries, three_body_stages);
    return value;
}

const std::map<std::string, STAGE>& MBBHSTAGE()
{
    static constexpr std::array<STAGE, 2> binary_black_hole_stages = {
        STAGE::QUASI_EQUIL,
        STAGE::ECC_RED,
    };
    static const auto value = make_selected_config_map(stage_entries, binary_black_hole_stages);
    return value;
}

const std::map<std::string, STAGE>& MBHNSSTAGE()
{
    static constexpr std::array<STAGE, 3> black_hole_neutron_star_stages = {
        STAGE::FORCE_BALANCE,
        STAGE::QUASI_EQUIL,
        STAGE::ECC_RED,
    };
    static const auto value = make_selected_config_map(stage_entries, black_hole_neutron_star_stages);
    return value;
}

const std::map<std::string, STAGE>& MBHSTAGE()
{
    static constexpr std::array<STAGE, 3> black_hole_stages = {
        STAGE::DIRICHLET_LAPSE,
        STAGE::VON_NEUMANN,
        STAGE::BIN_BOOST,
    };
    static const auto value = make_selected_config_map(stage_entries, black_hole_stages);
    return value;
}

const std::map<std::string, STAGE>& MNSSTAGE()
{
    static constexpr std::array<STAGE, 3> neutron_star_stages = {
        STAGE::NOROT,
        STAGE::UNIROT,
        STAGE::BIN_BOOST,
    };
    static const auto value = make_selected_config_map(stage_entries, neutron_star_stages);
    return value;
}

const std::map<std::string, STAGE>& MKSBHSTAGE()
{
    static constexpr std::array<STAGE, 2> kerr_schild_black_hole_stages = {
        STAGE::TRUMPET,
        STAGE::BIN_BOOST,
    };
    static const auto value = make_selected_config_map(stage_entries, kerr_schild_black_hole_stages);
    return value;
}
