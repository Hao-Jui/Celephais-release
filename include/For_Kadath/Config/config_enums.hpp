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

#pragma once
#include "For_Kadath/Config/config_enum_entries.hpp"
#include <map>
#include <string>
#include <type_traits>

/// Convert any enum value to its underlying integer type.
/// Used at array-index sites after enum class migration.
template <typename E> constexpr auto to_int(E e) noexcept
{
    return static_cast<std::underlying_type_t<E>>(e);
}
/**
 * @defgroup Configurator_enums
 * @ingroup Configurator
 * Physically motivated, static enums that allow for flexible reading
 * of Configurator TOML while allowing the ability to add/modify/delete params at a
 * later date without having to modify BIN/BCO classes that read in data. This
 * also allows for calling container locations using physically meaningful names
 * @{
 */

/* X-macro emit helper for enum-value enumeration. */
#define KADATH_ENUM_VALUE_EMIT(CTX, name, key) name,

/**
 * Declare one Configurator enum class together with its sentinel and the
 * matching NUM_*_V constexpr int. EntriesMacro follows the
 * KADATH_*_ENTRIES(APPLY, CTX) X-macro convention from config_enum_entries.hpp.
 */
#define KADATH_DECLARE_CONFIG_ENUM(EnumName, NumSym, EntriesMacro)    \
    enum class EnumName {                                             \
        EntriesMacro(KADATH_ENUM_VALUE_EMIT, /*ctx*/)                 \
        NumSym                                                        \
    };                                                                \
    constexpr int NumSym##_V = to_int(EnumName::NumSym)

#define KADATH_DECLARE_FROM_REGISTRY(EnumName, NumSym, EntriesMacro, ArrayName, FnName, HasDefault) \
    KADATH_DECLARE_CONFIG_ENUM(EnumName, NumSym, EntriesMacro);

KADATH_CONFIG_ENUMS(KADATH_DECLARE_FROM_REGISTRY)
#undef KADATH_DECLARE_FROM_REGISTRY

/**
 * Convenience using-enum declarations so that enumerators can be used
 * unqualified (e.g. RIN, STAGES, CONF) throughout the codebase exactly
 * as they were before the enum class migration.  Files that need strict
 * scoping can simply not include this header and qualify manually.
 */
#if defined(__cpp_using_enum)
#  define KADATH_USING_ENUM_FROM_REGISTRY(EnumName, NumSym, EntriesMacro, ArrayName, FnName, HasDefault) \
       using enum EnumName;
#else
#  define KADATH_ENUM_USING_EMIT(CTX, name, key) inline constexpr auto name = CTX::name;
#  define KADATH_USING_ENUM_FROM_REGISTRY(EnumName, NumSym, EntriesMacro, ArrayName, FnName, HasDefault) \
       EntriesMacro(KADATH_ENUM_USING_EMIT, EnumName)                                                    \
       inline constexpr auto NumSym = EnumName::NumSym;
#endif

KADATH_CONFIG_ENUMS(KADATH_USING_ENUM_FROM_REGISTRY)
#undef KADATH_USING_ENUM_FROM_REGISTRY

/**@{
 * extern definitions of maps containing the maps of strings for each parameter
 * name to the associated enumerator index. Default accessors (one per enum)
 * are declared via the registry; curated subset accessors are declared
 * individually below.
 */
#define KADATH_DECL_FN_1(EnumName, FnName) const std::map<std::string, EnumName>& FnName();
#define KADATH_DECL_FN_0(EnumName, FnName) /* no default accessor */
#define KADATH_DECL_FN_DISPATCH(EnumName, FnName, HasDefault) KADATH_DECL_FN_##HasDefault(EnumName, FnName)
#define KADATH_DECLARE_DEFAULT_MAP_FN(EnumName, NumSym, EntriesMacro, ArrayName, FnName, HasDefault) \
    KADATH_DECL_FN_DISPATCH(EnumName, FnName, HasDefault)

KADATH_CONFIG_ENUMS(KADATH_DECLARE_DEFAULT_MAP_FN)
#undef KADATH_DECLARE_DEFAULT_MAP_FN

/* Curated subset accessors (no default-map analog; NODES has no default). */
const std::map<std::string, NODES>& M_REQ_NODES();
const std::map<std::string, NODES>& MBCO();
const std::map<std::string, BCO_FIELDS>& MBCO_VFIELDS();
const std::map<std::string, BCO_FIELDS>& MBCO_SFIELDS_0();
const std::map<std::string, BCO_FIELDS>& MBCO_SFIELDS_1();
const std::map<std::string, STAGE>& MBNSSTAGE();
const std::map<std::string, STAGE>& MTRISTAGE();
const std::map<std::string, STAGE>& MBHNSSTAGE();
const std::map<std::string, STAGE>& MBBHSTAGE();
const std::map<std::string, STAGE>& MBHSTAGE();
const std::map<std::string, STAGE>& MKSBHSTAGE();
const std::map<std::string, STAGE>& MNSSTAGE();
/**@} end extern group definition*/
/**@} end config_enums group*/
