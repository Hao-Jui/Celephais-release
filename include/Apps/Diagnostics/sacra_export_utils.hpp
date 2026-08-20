#pragma once
// Shared scaffolding of the SACRA exporters (apps/shared/sacra_export_*.cpp):
// AMR level/longitude work dispatch across MPI ranks and the concatenation
// helper for the per-rank output files. Hoisted verbatim from the BNS
// exporter's anonymous namespace; the NS exporter previously inlined the same
// arithmetic.

#include <cmath>
#include <charconv>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace sacra_export {

struct GridOptions {
    int depth_max;
    int flvmax;
    int grid_half_width;
};

[[noreturn]] inline void throw_grid_option_error(const std::string& message, const std::string& usage)
{
    std::ostringstream oss;
    oss << message << "\n" << usage;
    throw std::invalid_argument(oss.str());
}

inline int parse_integer_flag(const std::string& flag, const std::string& value, const std::string& usage)
{
    int result = 0;
    const char* first = value.data();
    const char* last = first + value.size();
    const auto parsed = std::from_chars(first, last, result);
    if (parsed.ec == std::errc{} && parsed.ptr == last)
        return result;

    throw_grid_option_error("Invalid integer for " + flag + ": " + value, usage);
}

inline GridOptions parse_grid_options(int argc, char** argv, int first_option_arg,
                                      GridOptions defaults, const std::string& usage)
{
    GridOptions options = defaults;
    for (int i = first_option_arg; i < argc; ++i) {
        std::string const arg = argv[i];
        auto next_value = [&](const std::string& flag) -> std::string {
            if (i + 1 >= argc)
                throw_grid_option_error("Missing value after " + flag, usage);
            return argv[++i];
        };

        if (arg == "--depth-max") {
            options.depth_max = parse_integer_flag(arg, next_value(arg), usage);
        } else if (arg == "--flvmax") {
            options.flvmax = parse_integer_flag(arg, next_value(arg), usage);
        } else if (arg == "--N") {
            options.grid_half_width = parse_integer_flag(arg, next_value(arg), usage);
        } else if (arg == "-h" || arg == "--help") {
            throw std::invalid_argument(usage);
        } else if (!arg.empty() && arg[0] == '-') {
            throw_grid_option_error("Unknown option: " + arg, usage);
        } else {
            throw_grid_option_error("Unexpected positional argument: " + arg, usage);
        }
    }

    if (options.depth_max < 0)
        throw_grid_option_error("--depth-max must be >= 0", usage);
    if (options.flvmax < 0)
        throw_grid_option_error("--flvmax must be >= 0", usage);
    if (options.flvmax > options.depth_max)
        throw_grid_option_error("--flvmax must be <= --depth-max", usage);
    if (options.grid_half_width <= 0)
        throw_grid_option_error("--N must be > 0", usage);

    return options;
}

// Snap an object's center onto the parent grid spacing (cell-centered AMR).
inline int snap_to_parent_grid(double center, double origin, double parent_dx)
{
    double ratio = std::abs(center - origin) / parent_dx;
    int steps = (ratio - std::floor(ratio) < 0.5)
        ? static_cast<int>(std::floor(ratio))
        : static_cast<int>(std::floor(ratio)) + 1;
    if (center < origin)
        steps = -steps;
    return steps;
}

struct LevelRange {
    int first_level;
    int last_level;
    int first_lg;
    int last_lg;
};

// Distribute the (level, longitude-slab) work across ranks: with
// nb_procs <= lvmax0 each rank takes whole levels (all lg), otherwise each
// rank takes one level and a slab of lg values on it.
inline LevelRange compute_level_range(int rank, int nb_procs, int lvmax0, int total_nb_lg,
                                      int ld_org, int hllc)
{
    LevelRange r{};
    if (nb_procs <= lvmax0) {
        r.first_lg = ld_org - (6 + hllc);
        r.last_lg = -ld_org + (6 + hllc) - 1 + total_nb_lg + ld_org - (6 + hllc);

        int remain = lvmax0 % nb_procs;
        int quotient = lvmax0 / nb_procs;
        if (rank < remain) {
            r.first_level = rank * (quotient + 1);
            r.last_level = r.first_level + quotient;
        } else {
            r.first_level = remain * (quotient + 1) + (rank - remain) * quotient;
            r.last_level = r.first_level + quotient - 1;
        }
        r.first_lg = ld_org - (6 + hllc);
        r.last_lg = ld_org - (6 + hllc) + total_nb_lg - 1;
    } else {
        int remain = nb_procs % lvmax0;
        int quotient = nb_procs / lvmax0;
        int smallest_rank;
        int procs_on_level;

        if (rank < remain * (quotient + 1)) {
            r.first_level = rank / (quotient + 1);
            smallest_rank = r.first_level * (quotient + 1);
            procs_on_level = quotient + 1;
        } else {
            r.first_level = remain + (rank - remain * (quotient + 1)) / quotient;
            smallest_rank = remain * (quotient + 1) + (r.first_level - remain) * quotient;
            procs_on_level = quotient;
        }
        r.last_level = r.first_level;

        int remain_lg = total_nb_lg % procs_on_level;
        int quotient_lg = total_nb_lg / procs_on_level;
        int local_rank = rank - smallest_rank;

        if (local_rank < remain_lg) {
            r.first_lg = ld_org - (6 + hllc) + local_rank * (quotient_lg + 1);
            r.last_lg = r.first_lg + quotient_lg;
        } else {
            r.first_lg = ld_org - (6 + hllc) + remain_lg * (quotient_lg + 1) + (local_rank - remain_lg) * quotient_lg;
            r.last_lg = r.first_lg + quotient_lg - 1;
        }
    }
    return r;
}

// Emit the cat-script lines for one level split across procs_on_level ranks
// (mirrors the lg slabs of compute_level_range).
inline void write_concat_level_files(std::ofstream& fout, const std::string& base,
                                     int lv, int procs_on_level, int total_nb_lg,
                                     int ld_org, int hllc)
{
    int remain_lg = total_nb_lg % procs_on_level;
    int quotient_lg = total_nb_lg / procs_on_level;

    for (int p = 0; p < procs_on_level; ++p) {
        int flg, llg;
        if (p < remain_lg) {
            flg = ld_org - (6 + hllc) + p * (quotient_lg + 1);
            llg = flg + quotient_lg;
        } else {
            flg = ld_org - (6 + hllc) + remain_lg * (quotient_lg + 1) + (p - remain_lg) * quotient_lg;
            llg = flg + quotient_lg - 1;
        }
        fout << base + "_fields_data_lv" + std::to_string(lv) + "_lg" + std::to_string(flg) + "to" + std::to_string(llg) + ".d \\" << "\n";
    }
}

} // namespace sacra_export
