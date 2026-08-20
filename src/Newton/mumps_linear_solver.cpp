#ifdef CELEPHAIS_USE_MUMPS

#include "Linear_algebra/mumps_linear_solver.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#ifdef __APPLE__
#include <mach/mach.h>
#include <sys/sysctl.h>
#endif

namespace
{
constexpr int MUMPS_JOB_INIT = -1;
constexpr int MUMPS_JOB_END = -2;
constexpr int MUMPS_JOB_ANALYZE = 1;
constexpr int MUMPS_JOB_FACTORIZE = 2;
constexpr int MUMPS_JOB_SOLVE = 3;

long long decode_mumps_count(int encoded)
{
    return encoded < 0 ? -1000000LL * static_cast<long long>(encoded)
                       : static_cast<long long>(encoded);
}

// Parse a finite, non-negative double from env. Returns fallback if unset/empty.
// Aborts (writes to stderr + MPI_Abort) on malformed or negative/non-finite input —
// silent fallback would hide a misconfigured numerical knob.
double env_double_nonneg_or_default(const char* name, double fallback)
{
    const char* raw = std::getenv(name);
    if (raw == nullptr || raw[0] == '\0')
        return fallback;
    char* end = nullptr;
    const double value = std::strtod(raw, &end);
    if (end == raw || *end != '\0' || !std::isfinite(value) || value < 0.0) {
        std::cerr << "FATAL: " << name << "='" << raw
                  << "' is not a finite non-negative double\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    return value;
}

bool env_flag_enabled(const char* name)
{
    const char* value = std::getenv(name);
    return value != nullptr && value[0] != '\0' && std::string(value) != "0";
}

// Parse an integer from env. Returns fallback if unset/empty. Aborts on malformed
// input (silent fallback would hide a misconfigured numerical knob). Diagnostic
// harness only (ICNTL(14) reservation-trim bisect).
int env_int_or_default(const char* name, int fallback)
{
    const char* raw = std::getenv(name);
    if (raw == nullptr || raw[0] == '\0')
        return fallback;
    char* end = nullptr;
    const long value = std::strtol(raw, &end, 10);
    if (end == raw || *end != '\0') {
        std::cerr << "FATAL: " << name << "='" << raw << "' is not an integer\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    return static_cast<int>(value);
}

} // namespace

namespace Kadath
{

namespace mumps_memory_detail
{

namespace
{
constexpr std::uint64_t BYTES_PER_MB = 1024ULL * 1024ULL;
constexpr std::uint64_t V1_UNLIMITED_SENTINEL_MIN =
    static_cast<std::uint64_t>(std::numeric_limits<long long>::max()) - 4095ULL;

// Live free-RAM probes are self-consuming and swung from 6.8 to 12.6 GB in a
// measured solve. A process-lifetime minimum is a conservative ceiling across
// stages: it prevents a transient high from re-admitting in-core factorization.
// If this forces late small rungs OOC unnecessarily, the upgrade path is a
// per-nonlinear-solve reset. The solver calls this on rank zero under its
// current single-threaded solver contract.
long long& lowest_node_available_memory_probe_mb()
{
    static long long lowest_mb = std::numeric_limits<long long>::max();
    return lowest_mb;
}

std::string_view trim_ascii_whitespace(std::string_view text)
{
    while (!text.empty() &&
           std::isspace(static_cast<unsigned char>(text.front())) != 0)
        text.remove_prefix(1);
    while (!text.empty() &&
           std::isspace(static_cast<unsigned char>(text.back())) != 0)
        text.remove_suffix(1);
    return text;
}

std::optional<std::uint64_t> parse_unsigned_decimal(std::string_view text)
{
    text = trim_ascii_whitespace(text);
    if (text.empty())
        return std::nullopt;

    std::uint64_t value = 0;
    for (const char character : text) {
        if (character < '0' || character > '9')
            return std::nullopt;
        const auto digit = static_cast<std::uint64_t>(character - '0');
        if (value > (std::numeric_limits<std::uint64_t>::max() - digit) / 10ULL)
            return std::nullopt;
        value = value * 10ULL + digit;
    }
    return value;
}

#ifdef __linux__
std::optional<std::string> read_text_file(const std::filesystem::path& path)
{
    std::ifstream stream(path);
    if (!stream)
        return std::nullopt;
    std::ostringstream contents;
    contents << stream.rdbuf();
    return stream.bad() ? std::nullopt
                        : std::optional<std::string>(contents.str());
}

std::optional<std::string> cgroup_membership_path(
    const std::string& contents, bool v2)
{
    std::istringstream lines(contents);
    std::string line;
    while (std::getline(lines, line)) {
        const auto first_colon = line.find(':');
        const auto second_colon = first_colon == std::string::npos
                                      ? std::string::npos
                                      : line.find(':', first_colon + 1);
        if (second_colon == std::string::npos)
            continue;
        const std::string hierarchy = line.substr(0, first_colon);
        const std::string controllers =
            line.substr(first_colon + 1, second_colon - first_colon - 1);
        if (v2 && hierarchy == "0" && controllers.empty())
            return line.substr(second_colon + 1);
        if (!v2) {
            std::istringstream names(controllers);
            std::string name;
            while (std::getline(names, name, ',')) {
                if (name == "memory")
                    return line.substr(second_colon + 1);
            }
        }
    }
    return std::nullopt;
}

CgroupMemoryHeadroom effective_cgroup_headroom(
    const std::filesystem::path& mount_root,
    const std::string& membership_path,
    bool v2)
{
    std::filesystem::path relative = membership_path;
    if (relative.is_absolute())
        relative = relative.relative_path();
    auto directory = (mount_root / relative).lexically_normal();
    if (directory.native().compare(0, mount_root.native().size(),
                                   mount_root.native()) != 0)
        return {CgroupMemoryStatus::Unreadable, -1};

    CgroupMemoryHeadroom effective{CgroupMemoryStatus::Unlimited, -1};
    for (;;) {
        const auto limit = read_text_file(
            directory / (v2 ? "memory.max" : "memory.limit_in_bytes"));
        const auto usage = read_text_file(
            directory / (v2 ? "memory.current" : "memory.usage_in_bytes"));
        if (!limit || !usage)
            return {CgroupMemoryStatus::Unreadable, -1};
        const auto headroom = parse_cgroup_memory_headroom(*limit, *usage);
        if (headroom.status == CgroupMemoryStatus::Unreadable)
            return headroom;
        if (headroom.status == CgroupMemoryStatus::Limited &&
            (effective.status != CgroupMemoryStatus::Limited ||
             headroom.available_mb < effective.available_mb))
            effective = headroom;
        if (directory == mount_root)
            break;
        directory = directory.parent_path();
    }
    return effective;
}

long long linux_node_available_memory_mb()
{
    const auto meminfo = read_text_file("/proc/meminfo");
    const auto membership = read_text_file("/proc/self/cgroup");
    if (!meminfo || !membership)
        return -1;
    long long available_mb = parse_mem_available_mb(*meminfo);
    if (available_mb < 0)
        return -1;

    // Linux cgroupfs is conventionally mounted at this namespace root. Resolve
    // the process's nested membership below it and walk parents so a Slurm
    // job/step limit above the leaf is included. A nonstandard/unreadable mount
    // returns -1 instead of silently using host MemAvailable.
    const auto v2_path = cgroup_membership_path(*membership, true);
    const auto v1_path = cgroup_membership_path(*membership, false);
    const auto apply = [&](const std::filesystem::path& root,
                           const std::string& path, bool v2) {
        const auto headroom = effective_cgroup_headroom(root, path, v2);
        if (headroom.status == CgroupMemoryStatus::Unreadable)
            return false;
        if (headroom.status == CgroupMemoryStatus::Limited)
            available_mb = std::min(available_mb, headroom.available_mb);
        return true;
    };
    // In a hybrid hierarchy an explicit v1 memory controller is authoritative;
    // otherwise the unified v2 membership owns memory accounting.
    if (v1_path) {
        if (!apply("/sys/fs/cgroup/memory", *v1_path, false))
            return -1;
    } else if (v2_path && !apply("/sys/fs/cgroup", *v2_path, true)) {
        return -1;
    }
    return available_mb;
}
#endif

#ifdef __APPLE__
long long macos_node_available_memory_mb()
{
    std::uint64_t physical_bytes = 0;
    std::size_t physical_size = sizeof(physical_bytes);
    if (sysctlbyname("hw.memsize", &physical_bytes, &physical_size, nullptr, 0) != 0 ||
        physical_size != sizeof(physical_bytes) || physical_bytes == 0)
        return -1;

    const mach_port_t host = mach_host_self();
    vm_size_t page_size = 0;
    vm_statistics64_data_t statistics{};
    mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;
    const kern_return_t page_result = host_page_size(host, &page_size);
    const kern_return_t statistics_result = host_statistics64(
        host, HOST_VM_INFO64, reinterpret_cast<host_info64_t>(&statistics), &count);
    mach_port_deallocate(mach_task_self(), host);
    if (page_result != KERN_SUCCESS || statistics_result != KERN_SUCCESS ||
        page_size == 0)
        return -1;

    const std::uint64_t available_pages =
        static_cast<std::uint64_t>(statistics.free_count) +
        static_cast<std::uint64_t>(statistics.inactive_count) +
        static_cast<std::uint64_t>(statistics.purgeable_count);
    const std::uint64_t page_bytes = static_cast<std::uint64_t>(page_size);
    const std::uint64_t available_bytes =
        available_pages > std::numeric_limits<std::uint64_t>::max() / page_bytes
            ? std::numeric_limits<std::uint64_t>::max()
            : available_pages * page_bytes;
    return static_cast<long long>(
        std::min(available_bytes, physical_bytes) / BYTES_PER_MB);
}
#endif

} // namespace

long long parse_mem_available_mb(const std::string& meminfo)
{
    std::istringstream lines(meminfo);
    std::string line;
    while (std::getline(lines, line)) {
        std::istringstream fields(line);
        std::string key;
        std::string value;
        std::string unit;
        std::string trailing;
        if (!(fields >> key) || key != "MemAvailable:")
            continue;
        if (!(fields >> value >> unit) || unit != "kB" || (fields >> trailing))
            return -1;
        const auto kb = parse_unsigned_decimal(value);
        if (!kb)
            return -1;
        const auto mb = *kb / 1024ULL;
        if (mb > static_cast<std::uint64_t>(std::numeric_limits<long long>::max()))
            return -1;
        return static_cast<long long>(mb);
    }
    return -1;
}

CgroupMemoryHeadroom parse_cgroup_memory_headroom(
    const std::string& limit, const std::string& usage)
{
    const std::string_view trimmed_limit = trim_ascii_whitespace(limit);
    if (trimmed_limit == "max")
        return {CgroupMemoryStatus::Unlimited, -1};

    const auto limit_bytes = parse_unsigned_decimal(trimmed_limit);
    const auto usage_bytes = parse_unsigned_decimal(usage);
    if (!limit_bytes || !usage_bytes)
        return {CgroupMemoryStatus::Unreadable, -1};
    if (*limit_bytes >= V1_UNLIMITED_SENTINEL_MIN)
        return {CgroupMemoryStatus::Unlimited, -1};

    const std::uint64_t headroom_bytes =
        *usage_bytes >= *limit_bytes ? 0ULL : *limit_bytes - *usage_bytes;
    const std::uint64_t headroom_mb = headroom_bytes / BYTES_PER_MB;
    if (headroom_mb >
        static_cast<std::uint64_t>(std::numeric_limits<long long>::max()))
        return {CgroupMemoryStatus::Unreadable, -1};
    return {CgroupMemoryStatus::Limited, static_cast<long long>(headroom_mb)};
}

long long node_available_memory_mb()
{
#ifdef __linux__
    // MemAvailable already accounts for OS/cache pressure. The cgroup walk
    // follows this process's namespace path and includes finite parent limits,
    // such as a Slurm job/step cgroup, before taking the smaller headroom.
    return linux_node_available_memory_mb();
#elif defined(__APPLE__)
    return macos_node_available_memory_mb();
#else
    return -1;
#endif
}

long long ratcheted_node_available_memory_mb(long long probed_mb)
{
    if (probed_mb < 0)
        return -1;
    long long& lowest_mb = lowest_node_available_memory_probe_mb();
    lowest_mb = std::min(lowest_mb, probed_mb);
    return lowest_mb;
}

void reset_node_available_memory_ratchet_for_tests()
{
    lowest_node_available_memory_probe_mb() =
        std::numeric_limits<long long>::max();
}

} // namespace mumps_memory_detail

MumpsPatternSupersetUpdate update_mumps_pattern_superset(
    int n,
    long long nnz,
    const int* irn,
    const int* jcn,
    const double* values,
    double numerical_drop_tol,
    const std::vector<int>& pattern_irn,
    const std::vector<int>& pattern_jcn,
    const std::vector<long long>& pattern_column_offsets,
    std::vector<int>& next_pattern_irn,
    std::vector<int>& next_pattern_jcn,
    std::vector<long long>& next_pattern_column_offsets,
    std::vector<double>& aligned_values)
{
    if (n < 0 || nnz < 0 || !std::isfinite(numerical_drop_tol) ||
        numerical_drop_tol < 0.0) {
        throw std::invalid_argument(
            "update_mumps_pattern_superset: invalid dimension, nnz, or drop tolerance");
    }
    if (nnz > 0 && (irn == nullptr || jcn == nullptr || values == nullptr)) {
        throw std::invalid_argument(
            "update_mumps_pattern_superset: non-empty COO has a null input pointer");
    }

    const std::size_t nnz_size = static_cast<std::size_t>(nnz);
    if (static_cast<long long>(nnz_size) != nnz) {
        throw std::length_error(
            "update_mumps_pattern_superset: nnz does not fit addressable memory");
    }

    // JacobianAssembler assigns each column to exactly one rank, emits that
    // column contiguously, and gathers rank-major. Record those groups without
    // an O(nnz) key/hash side table. A repeated non-contiguous group would make
    // the bounded-memory merge ambiguous, so reject it loudly.
    std::vector<long long> current_begin(static_cast<std::size_t>(n), -1);
    std::vector<long long> current_end(static_cast<std::size_t>(n), -1);
    long long numerical_nnz = 0;
    for (long long k = 0; k < nnz; ++k) {
        const int row = irn[static_cast<std::size_t>(k)];
        const int col = jcn[static_cast<std::size_t>(k)];
        if (row < 1 || row > n || col < 1 || col > n) {
            throw std::out_of_range(
                "update_mumps_pattern_superset: COO coordinate outside [1,n]");
        }
        const std::size_t col_index = static_cast<std::size_t>(col - 1);
        if (current_begin[col_index] < 0) {
            current_begin[col_index] = k;
            current_end[col_index] = k + 1;
        } else if (current_end[col_index] == k) {
            current_end[col_index] = k + 1;
        } else {
            throw std::invalid_argument(
                "update_mumps_pattern_superset: a column has non-contiguous COO groups");
        }
        if (std::abs(values[static_cast<std::size_t>(k)]) > numerical_drop_tol)
            ++numerical_nnz;
    }

    next_pattern_irn.clear();
    next_pattern_jcn.clear();
    next_pattern_column_offsets.clear();

    const bool have_pattern = !pattern_column_offsets.empty();
    if (have_pattern) {
        if (pattern_column_offsets.size() != static_cast<std::size_t>(n + 1) ||
            pattern_irn.size() != pattern_jcn.size() ||
            pattern_column_offsets.back() !=
                static_cast<long long>(pattern_irn.size())) {
            throw std::invalid_argument(
                "update_mumps_pattern_superset: cached pattern invariants are inconsistent");
        }
    } else if (!pattern_irn.empty() || !pattern_jcn.empty()) {
        throw std::invalid_argument(
            "update_mumps_pattern_superset: cached pattern lacks column offsets");
    }

    struct RowValue {
        int row = 0;
        double value = 0.0;
    };
    std::vector<RowValue> sorted_scratch;

    auto prepare_current_column = [&](int col) -> bool {
        const std::size_t col_index = static_cast<std::size_t>(col);
        const long long begin = current_begin[col_index];
        const long long end = current_end[col_index];
        sorted_scratch.clear();
        if (begin < 0)
            return false;
        bool sorted = true;
        for (long long k = begin + 1; k < end; ++k) {
            if (irn[static_cast<std::size_t>(k - 1)] >
                irn[static_cast<std::size_t>(k)]) {
                sorted = false;
                break;
            }
        }
        if (!sorted) {
            sorted_scratch.reserve(static_cast<std::size_t>(end - begin));
            for (long long k = begin; k < end; ++k) {
                sorted_scratch.push_back(RowValue{
                    irn[static_cast<std::size_t>(k)],
                    values[static_cast<std::size_t>(k)]});
            }
            std::stable_sort(
                sorted_scratch.begin(), sorted_scratch.end(),
                [](const RowValue& lhs, const RowValue& rhs) {
                    return lhs.row < rhs.row;
                });
        }
        return sorted;
    };
    auto current_count = [&](int col) -> long long {
        const std::size_t col_index = static_cast<std::size_t>(col);
        return current_begin[col_index] < 0
                   ? 0
                   : current_end[col_index] - current_begin[col_index];
    };
    auto current_row = [&](int col, long long local_index, bool sorted) -> int {
        if (!sorted)
            return sorted_scratch[static_cast<std::size_t>(local_index)].row;
        const long long offset = current_begin[static_cast<std::size_t>(col)];
        return irn[static_cast<std::size_t>(offset + local_index)];
    };
    auto current_value = [&](int col, long long local_index, bool sorted) -> double {
        const double value = !sorted
                                 ? sorted_scratch[static_cast<std::size_t>(local_index)].value
                                 : values[static_cast<std::size_t>(
                                       current_begin[static_cast<std::size_t>(col)] +
                                       local_index)];
        return (std::abs(value) > numerical_drop_tol) ? value : 0.0;
    };
    auto current_is_numerical = [&](int col, long long local_index,
                                    bool sorted) -> bool {
        return current_value(col, local_index, sorted) != 0.0;
    };

    long long new_entries = 0;
    if (!have_pattern) {
        new_entries = nnz;
    } else {
        for (int col = 0; col < n; ++col) {
            const bool sorted = prepare_current_column(col);
            long long ci = 0;
            long long pi = pattern_column_offsets[static_cast<std::size_t>(col)];
            const long long current_size = current_count(col);
            const long long pattern_end =
                pattern_column_offsets[static_cast<std::size_t>(col + 1)];
            while (ci < current_size && pi < pattern_end) {
                const int crow = current_row(col, ci, sorted);
                const int prow = pattern_irn[static_cast<std::size_t>(pi)];
                if (crow == prow) {
                    ++ci;
                    ++pi;
                } else if (crow < prow) {
                    // A newly visible low-threshold candidate that is still
                    // dropped from the numerical Jacobian is already an
                    // explicit zero; it cannot falsify symbolic reuse.
                    if (current_is_numerical(col, ci, sorted))
                        ++new_entries;
                    ++ci;
                } else {
                    ++pi;
                }
            }
            while (ci < current_size) {
                if (current_is_numerical(col, ci, sorted))
                    ++new_entries;
                ++ci;
            }
        }
    }

    const bool pattern_changed = !have_pattern || new_entries > 0;
    if (pattern_changed) {
        std::vector<int> next_irn;
        std::vector<int> next_jcn;
        std::vector<long long> next_offsets(static_cast<std::size_t>(n + 1), 0);
        std::vector<double> next_values;
        const long long next_nnz =
            static_cast<long long>(pattern_irn.size()) + new_entries;
        const std::size_t next_size = static_cast<std::size_t>(next_nnz);
        if (static_cast<long long>(next_size) != next_nnz)
            throw std::length_error(
                "update_mumps_pattern_superset: expanded pattern is not addressable");
        next_irn.reserve(next_size);
        next_jcn.reserve(next_size);
        next_values.reserve(next_size);

        for (int col = 0; col < n; ++col) {
            const bool sorted = prepare_current_column(col);
            long long ci = 0;
            long long pi = have_pattern
                               ? pattern_column_offsets[static_cast<std::size_t>(col)]
                               : 0;
            const long long current_size = current_count(col);
            const long long pattern_end =
                have_pattern
                    ? pattern_column_offsets[static_cast<std::size_t>(col + 1)]
                    : 0;
            while (ci < current_size || pi < pattern_end) {
                const int crow = (ci < current_size)
                                     ? current_row(col, ci, sorted)
                                     : std::numeric_limits<int>::max();
                const int prow = (pi < pattern_end)
                                     ? pattern_irn[static_cast<std::size_t>(pi)]
                                     : std::numeric_limits<int>::max();
                if (crow == prow) {
                    next_irn.push_back(crow);
                    next_jcn.push_back(col + 1);
                    next_values.push_back(current_value(col, ci, sorted));
                    ++ci;
                    ++pi;
                } else if (crow < prow) {
                    // The initial pattern is the complete low-threshold
                    // candidate matrix. On later steps, only a missing
                    // numerically-active coordinate grows that superset.
                    if (!have_pattern || current_is_numerical(col, ci, sorted)) {
                        next_irn.push_back(crow);
                        next_jcn.push_back(col + 1);
                        next_values.push_back(current_value(col, ci, sorted));
                    }
                    ++ci;
                } else {
                    next_irn.push_back(prow);
                    next_jcn.push_back(col + 1);
                    next_values.push_back(0.0);
                    ++pi;
                }
            }
            next_offsets[static_cast<std::size_t>(col + 1)] =
                static_cast<long long>(next_irn.size());
        }
        next_pattern_irn.swap(next_irn);
        next_pattern_jcn.swap(next_jcn);
        next_pattern_column_offsets.swap(next_offsets);
        aligned_values.swap(next_values);
    } else {
        aligned_values.assign(pattern_irn.size(), 0.0);
        for (int col = 0; col < n; ++col) {
            const bool sorted = prepare_current_column(col);
            long long ci = 0;
            long long pi = pattern_column_offsets[static_cast<std::size_t>(col)];
            const long long current_size = current_count(col);
            const long long pattern_end =
                pattern_column_offsets[static_cast<std::size_t>(col + 1)];
            while (ci < current_size && pi < pattern_end) {
                const int crow = current_row(col, ci, sorted);
                const int prow = pattern_irn[static_cast<std::size_t>(pi)];
                if (crow == prow) {
                    aligned_values[static_cast<std::size_t>(pi)] =
                        current_value(col, ci, sorted);
                    ++ci;
                    ++pi;
                } else if (crow < prow) {
                    // This coordinate was absent from the superset, but the
                    // containment pass proved its value is numerically dropped.
                    ++ci;
                } else {
                    // The missing-coordinate pass above proved containment, so
                    // only a cached explicit-zero row may precede current data.
                    ++pi;
                }
            }
            while (ci < current_size &&
                   !current_is_numerical(col, ci, sorted)) {
                // The cached column ended before one or more newly visible
                // low-threshold candidates. They remain numerical zeros and
                // therefore do not invalidate the symbolic pattern.
                ++ci;
            }
            if (ci != current_size) {
                throw std::logic_error(
                    "update_mumps_pattern_superset: containment changed during alignment");
            }
        }
    }

    const long long superset_nnz = pattern_changed
                                       ? static_cast<long long>(next_pattern_irn.size())
                                       : static_cast<long long>(pattern_irn.size());
    return MumpsPatternSupersetUpdate{
        pattern_changed,
        nnz,
        numerical_nnz,
        superset_nnz,
        new_entries,
        superset_nnz - numerical_nnz};
}

std::string mumps_ordering_name(int ordering)
{
    switch (ordering) {
    case 0:
        return "AMD";
    case 1:
        return "user-provided";
    case 2:
        return "AMF";
    case 3:
        return "SCOTCH";
    case 4:
        return "PORD";
    case 5:
        return "METIS";
    case 6:
        return "QAMD";
    case 7:
        return "automatic";
    default:
        return "unknown(" + std::to_string(ordering) + ")";
    }
}

MumpsLinearSolver::MumpsLinearSolver(int n, int ordering, bool out_of_core,
                                     int blr, int icntl14_initial, MPI_Comm comm,
                                     int ranks_per_node, bool memory_capped)
    : MumpsLinearSolver(n, ordering,
                        out_of_core ? MumpsOutOfCoreMode::On
                                    : MumpsOutOfCoreMode::Off,
                        blr, icntl14_initial, comm, ranks_per_node,
                        memory_capped)
{
}

MumpsLinearSolver::MumpsLinearSolver(int n, int ordering,
                                     MumpsOutOfCoreMode out_of_core_mode,
                                     int blr, int icntl14_initial, MPI_Comm comm,
                                     int ranks_per_node, bool memory_capped,
                                     double out_of_core_touch,
                                     double out_of_core_safety,
                                     double out_of_core_budget_mb)
    : world_comm_{comm},
      icntl14_{icntl14_initial},
      ordering_{ordering},
      out_of_core_mode_{out_of_core_mode},
      ooc_icntl22_{out_of_core_mode == MumpsOutOfCoreMode::On ? 1 : 0},
      out_of_core_touch_{out_of_core_touch},
      out_of_core_safety_{out_of_core_safety},
      out_of_core_budget_mb_{out_of_core_budget_mb},
      auto_out_of_core_diagnostic_{&std::cerr},
      blr_icntl35_{blr},
      n_{n}
{
    // ICNTL(14) reservation-trim bisect harness (diagnostic only): a fixed start
    // seed that OVERRIDES the carried-forward ratchet value so every per-step
    // factor restarts from the same tested ICNTL(14). Read once at construction.
    // Unset (<1) leaves the production seed/ratchet untouched. Paired with
    // MUMPS_ICNTL14_NORATCHET (checked in factor_analyzed) so a
    // workspace-too-small (-9/-20) at the tested value FAILS VISIBLY instead of
    // silently climbing -- a clean bisect on the minimum reservation.
    const int icntl14_override = env_int_or_default("MUMPS_ICNTL14", -1);
    if (icntl14_override >= 1) {
        icntl14_ = icntl14_override;
    }

    // Build the MUMPS communicator. ranks_per_node<=0 -> all of comm (original path,
    // bit-identical). Otherwise keep only the first `ranks_per_node` ranks of each
    // physical node (MPI_COMM_TYPE_SHARED) so the factor runs on fewer, fatter ranks
    // while assembly/do_JX/GMRES stay on the full comm. World rank 0 is node-local
    // rank 0 on node 0, so it is always kept -> remains the centralized-COO host and
    // the solve() bcast root.
    if (ranks_per_node <= 0) {
        mumps_comm_ = comm;
        owns_mumps_comm_ = false;
        in_mumps_ = true;
    } else {
        int world_rank = 0;
        MPI_Comm_rank(comm, &world_rank); // NOLINT(bugprone-casting-through-void)
        MPI_Comm node_comm = MPI_COMM_NULL;
        MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, world_rank, MPI_INFO_NULL, &node_comm);
        int node_rank = 0;
        MPI_Comm_rank(node_comm, &node_rank); // NOLINT(bugprone-casting-through-void)
        MPI_Comm_free(&node_comm);
        in_mumps_ = (node_rank < ranks_per_node);
        MPI_Comm_split(comm, in_mumps_ ? 0 : MPI_UNDEFINED, world_rank, &mumps_comm_);
        owns_mumps_comm_ = in_mumps_;  // non-participants get MPI_COMM_NULL (nothing to free)
    }
    comm_ = mumps_comm_;
    ranks_per_node_ = ranks_per_node;
    memory_capped_ = memory_capped;
    if (in_mumps_) {
        int factor_rank = 0;
        MPI_Comm_rank(comm_, &factor_rank); // NOLINT(bugprone-casting-through-void)
        MPI_Comm shared_factor_comm = MPI_COMM_NULL;
        MPI_Comm_split_type(comm_, MPI_COMM_TYPE_SHARED, factor_rank,
                            MPI_INFO_NULL, &shared_factor_comm);
        MPI_Comm_size(shared_factor_comm, &factor_ranks_on_host_node_);
        MPI_Comm_free(&shared_factor_comm);
    }
    const int local_analysis_rank = in_mumps_ ? 1 : 0;
    MPI_Allreduce(&local_analysis_rank, &analysis_rank_count_, 1, MPI_INT,
                  MPI_SUM, world_comm_);
    std::memset(&mumps_, 0, sizeof(mumps_));

    if (in_mumps_) {
        MPI_Comm_rank(comm_, &rank_); // NOLINT(bugprone-casting-through-void)
        mumps_.comm_fortran = MPI_Comm_c2f(comm_);
        mumps_.par = 1;
        mumps_.sym = 0; // PAR=1: host participates as worker (required for np=1)
        run_job(MUMPS_JOB_INIT);
        initialized_ = true;
        mumps_.n = n_;
        apply_icntls();
        if (rank_ == 0 && icntl14_override >= 1) {
            std::cerr << "MUMPS_ICNTL14 override: ICNTL(14) seeded = "
                      << icntl14_ << " (noratchet="
                      << (env_flag_enabled("MUMPS_ICNTL14_NORATCHET") ? 1 : 0)
                      << ")\n";
        }
    } else {
        rank_ = -1;  // not in the MUMPS comm; only joins solve()'s world bcast
    }
}

MumpsLinearSolver::~MumpsLinearSolver()
{
    if (initialized_) {
        run_job(MUMPS_JOB_END);
        initialized_ = false;
    }
    if (owns_mumps_comm_ && mumps_comm_ != MPI_COMM_NULL) {
        MPI_Comm_free(&mumps_comm_); // NOLINT(bugprone-casting-through-void)
        mumps_comm_ = MPI_COMM_NULL;
    }
}

namespace {
// MUMPS reads MUMPS_OOC_TMPDIR when a factorization runs out-of-core.
// Default it to ${HOME_CELEPHAIS}/data/ooc so OOC factor files land beside
// the repository's other runtime data instead of the upstream default
// scratch. An explicit MUMPS_OOC_TMPDIR always wins; without HOME_CELEPHAIS
// or with an uncreatable directory the upstream default stands.
void ensure_out_of_core_tmpdir()
{
    if (std::getenv("MUMPS_OOC_TMPDIR") != nullptr)
        return;
    const char* home = std::getenv("HOME_CELEPHAIS");
    if (home == nullptr || home[0] == '\0')
        return;
    const std::filesystem::path ooc_dir =
        std::filesystem::path(home) / "data" / "ooc";
    std::error_code directory_error;
    std::filesystem::create_directories(ooc_dir, directory_error);
    if (directory_error)
        return;
    setenv("MUMPS_OOC_TMPDIR", ooc_dir.c_str(), 0);
}
} // namespace

void MumpsLinearSolver::apply_icntls()
{
    mumps_.icntl[17] = 0; // centralized assembled input
    // Native MUMPS output is extremely verbose on MPI jobs. Keep it suppressed
    // unless explicitly requested for a MUMPS debugging session.
    const bool native_mumps_verbose = env_flag_enabled("MUMPS_NATIVE_VERBOSE");
    mumps_.icntl[0] = native_mumps_verbose ? 6 : -1;  // error message unit
    mumps_.icntl[1] = native_mumps_verbose ? 6 : -1;  // diagnostic/warning unit
    mumps_.icntl[2] = native_mumps_verbose ? 6 : -1;  // global info unit (rank 0)
    mumps_.icntl[3] = native_mumps_verbose ? 2 : 0;   // verbosity level
    mumps_.icntl[6] = ordering_;
    mumps_.icntl[13] = icntl14_;
    mumps_.icntl[21] = ooc_icntl22_;
    if (ooc_icntl22_ == 1)
        ensure_out_of_core_tmpdir();
    mumps_.icntl[34] = blr_icntl35_;
    // ICNTL(23): cap per-rank working memory (MB) at (free node RAM x 0.8) /
    // factor-ranks-per-node, so an over-budget factor returns a catchable -9/-16
    // instead of being SIGKILLed -- the adaptive loop then raises the rank count.
    // Skipped only when the platform/cgroup probe is unreadable (-1). A real zero
    // headroom still produces the minimum 1 MB cap instead of disabling the guard.
    if (memory_capped_ && ranks_per_node_ > 0) {
        const long long avail_mb =
            mumps_memory_detail::node_available_memory_mb();
        if (avail_mb >= 0) {
            const int cap_mb = static_cast<int>(
                std::max<long long>(1, avail_mb * 8 / 10 / ranks_per_node_));
            mumps_.icntl[22] = cap_mb;  // ICNTL(23)
            if (rank_ == 0) {
                std::cerr << "MUMPS ICNTL(23)=" << cap_mb << "MB/rank (free " << avail_mb
                          << "MB x0.8 / " << ranks_per_node_ << " factor-ranks/node)\n";
            }
        }
    }
    mumps_.icntl[23] = null_pivot_detection_ ? 1 : 0;
    if (null_pivot_detection_) {
        mumps_.cntl[2] = null_pivot_threshold_;
    }

    // CNTL(7): BLR drop tolerance. Default 0.0 = exact BLR (no compression).
    // Only meaningful when BLR is on (ICNTL(35) != 0); silently skipped otherwise
    // so MUMPS_BLR_DROP_TOL acts as no-op when BLR is disabled.
    const double blr_drop_tol =
        env_double_nonneg_or_default("MUMPS_BLR_DROP_TOL", 0.0);
    if (blr_icntl35_ != 0) {
        mumps_.cntl[6] = blr_drop_tol;
        if (rank_ == 0) {
            std::cerr << "MUMPS CNTL(7) BLR drop tolerance = " << blr_drop_tol
                      << " (ICNTL(35)=" << blr_icntl35_ << ")\n";
        }
    }

    // MUMPS 5.9.0 mixed-precision knobs (both default off -> no behaviour change).
    // ICNTL(40): mixed/adaptive BLR precision -- low-rank blocks stored in reduced
    // precision. Only meaningful when BLR is active (ICNTL(35) != 0).
    if (blr_icntl35_ != 0 && env_flag_enabled("MUMPS_BLR_MIXED")) {
        mumps_.icntl[39] = 1;
        if (rank_ == 0) {
            std::cerr << "MUMPS ICNTL(40)=1 (mixed-precision BLR)\n";
        }
    }

    // ICNTL(47): single-precision factorization + solve inside the double-precision
    // instance -- a native mixed-precision factor (no separate smumps build). As a
    // JFNK preconditioner the loss of precision is absorbed by the outer GMRES.
    // Independent of BLR.
    if (env_flag_enabled("MUMPS_SINGLE_FACTOR")) {
        mumps_.icntl[46] = 1;
        if (rank_ == 0) {
            std::cerr << "MUMPS ICNTL(47)=1 (single-precision factorization)\n";
        }
    }

}

void MumpsLinearSolver::set_auto_out_of_core_diagnostic(
    std::ostream* diagnostic, std::string prefix)
{
    if (analyzed_) {
        throw LinearSolverError(
            __FILE__, __LINE__,
            "AUTO OOC diagnostic must be configured before analyze_pattern()");
    }
    auto_out_of_core_diagnostic_ = diagnostic;
    auto_out_of_core_diagnostic_prefix_ = std::move(prefix);
}

void MumpsLinearSolver::prepare_auto_out_of_core_for_analysis()
{
    if (!in_mumps_ || out_of_core_mode_ != MumpsOutOfCoreMode::Auto)
        return;
    // A wrapper can be re-analyzed after a prior Auto decision. Keep JOB=1
    // independent of the earlier result and decide afresh from its INFOG(16).
    ooc_icntl22_ = 0;
    mumps_.icntl[21] = 0;
}

void MumpsLinearSolver::resolve_auto_out_of_core_after_analysis()
{
    if (!in_mumps_ || out_of_core_mode_ != MumpsOutOfCoreMode::Auto)
        return;

    int resolved_icntl22 = 0;
    std::string diagnostic;
    if (rank_ == 0) {
        double node_available_mb = out_of_core_budget_mb_;
        double unratcheted_budget_mb = 0.0;
        bool node_available_budget_ratcheted = false;
        if (out_of_core_budget_mb_ == kMumpsOutOfCoreBudgetUnset) {
            const long long probed_available_mb =
                mumps_memory_detail::node_available_memory_mb();
            if (probed_available_mb < 0) {
                diagnostic =
                    "OOC auto: node memory budget unreadable -> OFF";
            } else {
                const long long ratcheted_available_mb =
                    mumps_memory_detail::ratcheted_node_available_memory_mb(
                        probed_available_mb);
                node_available_mb = static_cast<double>(ratcheted_available_mb);
                node_available_budget_ratcheted =
                    ratcheted_available_mb != probed_available_mb;
                unratcheted_budget_mb =
                    static_cast<double>(probed_available_mb) *
                    out_of_core_safety_;
            }
        }

        if (diagnostic.empty()) {
            const MumpsOutOfCoreDecision decision = decide_mumps_out_of_core(
                static_cast<double>(estimated_factor_memory_mb_),
                mumps_.icntl[13], out_of_core_touch_, out_of_core_safety_,
                factor_ranks_on_host_node_, node_available_mb);
            if (decision.valid) {
                resolved_icntl22 = decision.use_out_of_core ? 1 : 0;
                std::ostringstream line;
                line << "OOC auto: expected "
                     << decision.expected_mb_per_rank << " MB/rank x "
                     << factor_ranks_on_host_node_
                     << " ranks/node vs budget " << decision.budget_mb;
                if (node_available_budget_ratcheted)
                    line << " (ratcheted from " << unratcheted_budget_mb << ")";
                line << " -> "
                     << (decision.use_out_of_core ? "ON" : "OFF");
                diagnostic = line.str();
            } else {
                diagnostic = "OOC auto: invalid policy inputs -> OFF";
            }
        }
    }

    if (MPI_Bcast(&resolved_icntl22, 1, MPI_INT, 0, comm_) != MPI_SUCCESS) {
        throw LinearSolverError(
            __FILE__, __LINE__,
            "MPI_Bcast failed while resolving MUMPS automatic OOC policy");
    }
    ooc_icntl22_ = resolved_icntl22;
    mumps_.icntl[21] = resolved_icntl22;
    if (resolved_icntl22 == 1)
        ensure_out_of_core_tmpdir();
    if (rank_ == 0 && auto_out_of_core_diagnostic_ != nullptr) {
        *auto_out_of_core_diagnostic_
            << auto_out_of_core_diagnostic_prefix_ << diagnostic << '\n';
    }
}

void MumpsLinearSolver::run_job(int job)
{
    if (!in_mumps_)
        return;  // rank outside the MUMPS sub-comm: no factor here, dmumps_c would hang
    mumps_.job = job;
    dmumps_c(&mumps_);
}

void MumpsLinearSolver::set_pattern(int n, long long nnz, const int* irn, const int* jcn)
{
    if ((block_analysis_enabled_ || user_permutation_enabled_ ||
         !solution_column_permutation_1based_.empty()) &&
        n != n_) {
        std::ostringstream oss;
        oss << "MUMPS analysis metadata was configured for n=" << n_
            << " but set_pattern received n=" << n
            << "; clear and reconfigure it first";
        throw LinearSolverError(__FILE__, __LINE__, oss.str());
    }
    mumps_.icntl[17] = 0;       // ICNTL(18)=0: centralized assembled input
    n_ = n;
    nnz_ = nnz;
    pattern_irn_ = (rank_ == 0) ? irn : nullptr;
    pattern_jcn_ = (rank_ == 0) ? jcn : nullptr;
    analyzed_ = false;
    mumps_.n = n;
    if (rank_ == 0) {
        const long long nz_i32_max =
            static_cast<long long>(std::numeric_limits<decltype(mumps_.nz)>::max());
        mumps_.nz = (nnz <= nz_i32_max) ? static_cast<decltype(mumps_.nz)>(nnz) : 0;
        mumps_.nnz = static_cast<decltype(mumps_.nnz)>(nnz);
        mumps_.irn = const_cast<int*>(pattern_irn_);
        mumps_.jcn = const_cast<int*>(pattern_jcn_);
    } else {
        mumps_.nz = 0;
        mumps_.nnz = 0;
        mumps_.irn = nullptr;
        mumps_.jcn = nullptr;
    }
}

void MumpsLinearSolver::validate_block_permutation_compatibility(
    int n,
    const std::vector<MUMPS_INT>& blkptr_1based,
    const std::vector<MUMPS_INT>& blkvar_1based,
    const std::vector<MUMPS_INT>& permutation_1based)
{
    if (blkptr_1based.empty() || permutation_1based.empty())
        return;

    for (std::size_t block = 0; block + 1 < blkptr_1based.size(); ++block) {
        const int begin = static_cast<int>(blkptr_1based[block]) - 1;
        const int end = static_cast<int>(blkptr_1based[block + 1]) - 1;
        int minimum_position = n + 1;
        int maximum_position = 0;
        for (int position = begin; position < end; ++position) {
            const int variable = blkvar_1based.empty()
                                     ? position + 1
                                     : static_cast<int>(
                                           blkvar_1based[static_cast<std::size_t>(position)]);
            const int pivot_position = static_cast<int>(
                permutation_1based[static_cast<std::size_t>(variable - 1)]);
            minimum_position = std::min(minimum_position, pivot_position);
            maximum_position = std::max(maximum_position, pivot_position);
        }
        const int block_size = end - begin;
        if (maximum_position - minimum_position + 1 != block_size) {
            std::ostringstream oss;
            oss << "MUMPS user permutation is incompatible with block "
                << (block + 1) << ": its " << block_size
                << " variables occupy pivot positions " << minimum_position
                << " through " << maximum_position
                << " instead of one consecutive interval";
            throw LinearSolverError(__FILE__, __LINE__, oss.str());
        }
    }
}

void MumpsLinearSolver::enable_block_analysis(
    const std::vector<int>& blkptr_1based,
    const std::vector<int>& blkvar_1based,
    BlockAnalysisMatching matching)
{
    const auto fail = [](const std::string& reason) {
        throw LinearSolverError(
            __FILE__, __LINE__, "MUMPS block analysis: " + reason);
    };

    if (analyzed_)
        fail("must be configured before analyze_pattern()");
    if (block_analysis_enabled_)
        fail("is already enabled; disable it before reconfiguration");
    if (n_ <= 0)
        fail("matrix order must be positive");
    if (blkptr_1based.size() < 2)
        fail("BLKPTR must contain at least one block and its end pointer");
    if (blkptr_1based.size() > static_cast<std::size_t>(n_) + 1)
        fail("BLKPTR defines more blocks than variables");
    if (blkptr_1based.front() != 1) {
        fail("BLKPTR must start at 1 (MUMPS uses 1-based positions)");
    }
    if (static_cast<long long>(blkptr_1based.back()) !=
        static_cast<long long>(n_) + 1) {
        std::ostringstream oss;
        oss << "BLKPTR must end at n+1=" << (static_cast<long long>(n_) + 1)
            << " but ends at " << blkptr_1based.back();
        fail(oss.str());
    }
    for (std::size_t i = 1; i < blkptr_1based.size(); ++i) {
        if (blkptr_1based[i] <= blkptr_1based[i - 1]) {
            std::ostringstream oss;
            oss << "BLKPTR must be strictly increasing; entries " << i
                << " and " << (i + 1) << " are " << blkptr_1based[i - 1]
                << " and " << blkptr_1based[i];
            fail(oss.str());
        }
    }

    if (!blkvar_1based.empty()) {
        if (blkvar_1based.size() != static_cast<std::size_t>(n_)) {
            std::ostringstream oss;
            oss << "BLKVAR must contain exactly n=" << n_
                << " variables but contains " << blkvar_1based.size();
            fail(oss.str());
        }
        std::vector<unsigned char> seen(static_cast<std::size_t>(n_), 0);
        for (std::size_t i = 0; i < blkvar_1based.size(); ++i) {
            const int variable = blkvar_1based[i];
            if (variable < 1 || variable > n_) {
                std::ostringstream oss;
                oss << "BLKVAR entry " << (i + 1) << " is " << variable
                    << ", outside [1," << n_ << ']';
                fail(oss.str());
            }
            unsigned char& count = seen[static_cast<std::size_t>(variable - 1)];
            if (count != 0) {
                std::ostringstream oss;
                oss << "BLKVAR repeats variable " << variable
                    << "; it must be a permutation of [1,n]";
                fail(oss.str());
            }
            count = 1;
        }
    }

    // Vendored MUMPS 5.9 disables maximum-transversal matching (ICNTL(6))
    // during block analysis. It leaves the independent factor-scaling control
    // ICNTL(8) intact. Do not let MUMPS silently drop matching, but preserve
    // scaling verbatim. This wrapper fixes SYM=0, for which ICNTL(12)=1.
    const int current_icntl6 = in_mumps_ ? mumps_.icntl[5] : 7;
    if (matching == BlockAnalysisMatching::Preserve &&
        current_icntl6 != 0) {
        std::ostringstream oss;
        oss << "ICNTL(15)=1 cannot preserve the configured matching "
            << "control (ICNTL(6)=" << current_icntl6
            << "); pass BlockAnalysisMatching::ExplicitlyDisable only for an "
            << "intentional experimental comparison";
        fail(oss.str());
    }

    std::vector<MUMPS_INT> owned_blkptr;
    owned_blkptr.reserve(blkptr_1based.size());
    for (int value : blkptr_1based)
        owned_blkptr.push_back(static_cast<MUMPS_INT>(value));
    std::vector<MUMPS_INT> owned_blkvar;
    owned_blkvar.reserve(blkvar_1based.size());
    for (int value : blkvar_1based)
        owned_blkvar.push_back(static_cast<MUMPS_INT>(value));

    if (user_permutation_enabled_) {
        validate_block_permutation_compatibility(
            n_, owned_blkptr, owned_blkvar, user_permutation_1based_);
    }

    block_ptr_1based_.swap(owned_blkptr);
    block_variables_1based_.swap(owned_blkvar);
    block_analysis_enabled_ = true;
    saved_block_icntl6_ = current_icntl6;
    block_matching_overridden_ =
        matching == BlockAnalysisMatching::ExplicitlyDisable;

    if (in_mumps_) {
        if (block_matching_overridden_) {
            mumps_.icntl[5] = 0;  // ICNTL(6): no zero-free-diagonal matching.
        }
        mumps_.icntl[14] = 1; // ICNTL(15): user-provided block format.
        mumps_.nblk = static_cast<MUMPS_INT>(block_ptr_1based_.size() - 1);
        if (rank_ == 0) {
            mumps_.blkptr = block_ptr_1based_.data();
            mumps_.blkvar = block_variables_1based_.empty()
                                ? nullptr
                                : block_variables_1based_.data();
        } else {
            mumps_.blkptr = nullptr;
            mumps_.blkvar = nullptr;
        }
    }
}

void MumpsLinearSolver::disable_block_analysis()
{
    if (analyzed_) {
        throw LinearSolverError(
            __FILE__, __LINE__,
            "MUMPS block analysis cannot be disabled after analyze_pattern()");
    }
    if (!block_analysis_enabled_)
        return;

    if (in_mumps_) {
        mumps_.icntl[14] = 0;
        mumps_.nblk = 0;
        mumps_.blkptr = nullptr;
        mumps_.blkvar = nullptr;
        if (block_matching_overridden_) {
            mumps_.icntl[5] = saved_block_icntl6_;
        }
    }
    block_ptr_1based_.clear();
    block_variables_1based_.clear();
    block_analysis_enabled_ = false;
    block_matching_overridden_ = false;
}

void MumpsLinearSolver::set_user_permutation_1based(
    const std::vector<int>& permutation_1based)
{
    const auto fail = [](const std::string& reason) {
        throw LinearSolverError(
            __FILE__, __LINE__, "MUMPS user permutation: " + reason);
    };
    if (analyzed_)
        fail("must be configured before analyze_pattern()");
    if (user_permutation_enabled_)
        fail("is already configured; clear it before replacement");
    if (n_ <= 0)
        fail("matrix order must be positive");
    if (permutation_1based.size() != static_cast<std::size_t>(n_)) {
        std::ostringstream oss;
        oss << "must contain exactly n=" << n_ << " pivot positions but contains "
            << permutation_1based.size();
        fail(oss.str());
    }

    std::vector<unsigned char> seen(static_cast<std::size_t>(n_), 0);
    std::vector<MUMPS_INT> owned_permutation;
    owned_permutation.reserve(permutation_1based.size());
    for (std::size_t i = 0; i < permutation_1based.size(); ++i) {
        const int pivot_position = permutation_1based[i];
        if (pivot_position < 1 || pivot_position > n_) {
            std::ostringstream oss;
            oss << "entry " << (i + 1) << " is " << pivot_position
                << ", outside [1," << n_ << ']';
            fail(oss.str());
        }
        unsigned char& count =
            seen[static_cast<std::size_t>(pivot_position - 1)];
        if (count != 0) {
            std::ostringstream oss;
            oss << "repeats pivot position " << pivot_position
                << "; values must be a bijection over [1,n]";
            fail(oss.str());
        }
        count = 1;
        owned_permutation.push_back(static_cast<MUMPS_INT>(pivot_position));
    }

    if (block_analysis_enabled_) {
        validate_block_permutation_compatibility(
            n_, block_ptr_1based_, block_variables_1based_, owned_permutation);
    }

    user_permutation_1based_.swap(owned_permutation);
    user_permutation_enabled_ = true;
    if (in_mumps_) {
        mumps_.icntl[6] = 1; // ICNTL(7): user-provided ordering.
        mumps_.perm_in = rank_ == 0 ? user_permutation_1based_.data() : nullptr;
    }
}

void MumpsLinearSolver::clear_user_permutation()
{
    if (analyzed_) {
        throw LinearSolverError(
            __FILE__, __LINE__,
            "MUMPS user permutation cannot be cleared after analyze_pattern()");
    }
    if (!user_permutation_enabled_)
        return;
    if (in_mumps_) {
        mumps_.icntl[6] = ordering_;
        mumps_.perm_in = nullptr;
    }
    user_permutation_1based_.clear();
    user_permutation_enabled_ = false;
}

void MumpsLinearSolver::set_solution_column_permutation_1based(
    const std::vector<int>& permutation_1based)
{
    const auto fail = [](const std::string& reason) {
        throw LinearSolverError(
            __FILE__, __LINE__, "MUMPS solution column permutation: " + reason);
    };
    if (analyzed_)
        fail("must be configured before analyze_pattern()");
    if (!solution_column_permutation_1based_.empty())
        fail("is already configured; clear it before replacement");
    if (n_ <= 0)
        fail("matrix order must be positive");
    if (permutation_1based.size() != static_cast<std::size_t>(n_)) {
        std::ostringstream oss;
        oss << "must contain exactly n=" << n_ << " original-column indices but contains "
            << permutation_1based.size();
        fail(oss.str());
    }

    std::vector<unsigned char> seen(static_cast<std::size_t>(n_), 0);
    std::vector<int> owned_permutation;
    owned_permutation.reserve(permutation_1based.size());
    for (std::size_t i = 0; i < permutation_1based.size(); ++i) {
        const int original_column = permutation_1based[i];
        if (original_column < 1 || original_column > n_) {
            std::ostringstream oss;
            oss << "entry " << (i + 1) << " is " << original_column
                << ", outside [1," << n_ << ']';
            fail(oss.str());
        }
        unsigned char& count = seen[static_cast<std::size_t>(original_column - 1)];
        if (count != 0) {
            std::ostringstream oss;
            oss << "repeats original column " << original_column
                << "; values must be a bijection over [1,n]";
            fail(oss.str());
        }
        count = 1;
        owned_permutation.push_back(original_column);
    }

    // Allocate before publishing either piece of state. solve() then performs
    // only indexed assignments and a copy before all ranks enter MPI_Bcast.
    std::vector<double> workspace(permutation_1based.size(), 0.0);
    solution_column_permutation_1based_.swap(owned_permutation);
    solution_column_permutation_workspace_.swap(workspace);
}

void MumpsLinearSolver::clear_solution_column_permutation()
{
    if (analyzed_) {
        throw LinearSolverError(
            __FILE__, __LINE__,
            "MUMPS solution column permutation cannot be cleared after analyze_pattern()");
    }
    solution_column_permutation_1based_.clear();
    solution_column_permutation_workspace_.clear();
}

void MumpsLinearSolver::validate_collective_analysis_metadata() const
{
    // JOB=1 is collective only on the factor communicator, but these controls
    // are configured through the wrapper on every caller rank. Gather one
    // fixed-size record on the full caller communicator so a rank excluded from
    // MUMPS cannot silently hold different analysis metadata.
    using FingerprintValue = std::uint64_t;
    constexpr std::size_t field_count = 12;
    constexpr FingerprintValue schema_version = 2;
    constexpr FingerprintValue fnv_offset_basis = 14695981039346656037ULL;
    constexpr FingerprintValue fnv_prime = 1099511628211ULL;

    FingerprintValue first_hash = fnv_offset_basis;
    FingerprintValue second_hash = fnv_offset_basis ^ 0x9e3779b97f4a7c15ULL;
    const auto mix = [&](FingerprintValue value) {
        first_hash ^= value;
        first_hash *= fnv_prime;
        second_hash ^= value + 0x9e3779b97f4a7c15ULL;
        second_hash *= 14029467366897019727ULL;
        second_hash ^= second_hash >> 29U;
    };
    const auto mix_sequence = [&](FingerprintValue tag,
                                  const std::vector<MUMPS_INT>& values) {
        mix(tag);
        mix(static_cast<FingerprintValue>(values.size()));
        for (const MUMPS_INT value : values)
            mix(static_cast<FingerprintValue>(value));
    };

    const FingerprintValue flags =
        (block_analysis_enabled_ ? 1ULL : 0ULL) |
        (block_matching_overridden_ ? 2ULL : 0ULL) |
        (user_permutation_enabled_ ? 4ULL : 0ULL);
    mix(schema_version);
    mix(static_cast<FingerprintValue>(n_));
    mix(flags);
    mix_sequence(1, block_ptr_1based_);
    mix_sequence(2, block_variables_1based_);
    mix_sequence(3, user_permutation_1based_);

    const auto double_bits = [](double value) {
        static_assert(sizeof(double) == sizeof(FingerprintValue));
        FingerprintValue bits = 0;
        std::memcpy(&bits, &value, sizeof(bits));
        return bits;
    };

    const std::array<FingerprintValue, field_count> local_record{{
        schema_version,
        static_cast<FingerprintValue>(n_),
        flags,
        static_cast<FingerprintValue>(block_ptr_1based_.size()),
        static_cast<FingerprintValue>(block_variables_1based_.size()),
        static_cast<FingerprintValue>(user_permutation_1based_.size()),
        static_cast<FingerprintValue>(out_of_core_mode_),
        double_bits(out_of_core_touch_),
        double_bits(out_of_core_safety_),
        double_bits(out_of_core_budget_mb_),
        first_hash,
        second_hash,
    }};

    int world_size = 0;
    if (MPI_Comm_size(world_comm_, &world_size) != MPI_SUCCESS) {
        throw LinearSolverError(
            __FILE__, __LINE__,
            "MPI_Comm_size failed while validating MUMPS analysis metadata");
    }
    std::vector<FingerprintValue> gathered_records(
        static_cast<std::size_t>(world_size) * field_count);
    if (MPI_Allgather(local_record.data(), static_cast<int>(field_count),
                      MPI_UINT64_T, gathered_records.data(),
                      static_cast<int>(field_count), MPI_UINT64_T,
                      world_comm_) != MPI_SUCCESS) {
        throw LinearSolverError(
            __FILE__, __LINE__,
            "MPI_Allgather failed while validating MUMPS analysis metadata");
    }

    int mismatching_rank = -1;
    for (int candidate_rank = 1; candidate_rank < world_size; ++candidate_rank) {
        const std::size_t candidate_offset =
            static_cast<std::size_t>(candidate_rank) * field_count;
        for (std::size_t field = 0; field < field_count; ++field) {
            if (gathered_records[field] !=
                gathered_records[candidate_offset + field]) {
                mismatching_rank = candidate_rank;
                break;
            }
        }
        if (mismatching_rank >= 0)
            break;
    }
    if (mismatching_rank < 0)
        return;

    const auto describe_record = [&](int world_rank) {
        const std::size_t offset =
            static_cast<std::size_t>(world_rank) * field_count;
        const FingerprintValue record_flags = gathered_records[offset + 2];
        const auto double_from_bits = [](FingerprintValue bits) {
            double value = 0.0;
            std::memcpy(&value, &bits, sizeof(value));
            return value;
        };
        const auto mode_name = [](FingerprintValue mode) {
            switch (static_cast<MumpsOutOfCoreMode>(mode)) {
            case MumpsOutOfCoreMode::Off:
                return "Off";
            case MumpsOutOfCoreMode::On:
                return "On";
            case MumpsOutOfCoreMode::Auto:
                return "Auto";
            }
            return "invalid";
        };
        std::ostringstream description;
        description << "rank " << world_rank
                    << " {block_enabled=" << (record_flags & 1ULL)
                    << ", matching_explicitly_disabled="
                    << ((record_flags >> 1U) & 1ULL)
                    << ", user_permutation_enabled="
                    << ((record_flags >> 2U) & 1ULL)
                    << ", BLKPTR_size=" << gathered_records[offset + 3]
                    << ", BLKVAR_size=" << gathered_records[offset + 4]
                    << ", PERM_IN_size=" << gathered_records[offset + 5]
                    << ", ooc_mode=" << mode_name(gathered_records[offset + 6])
                    << ", ooc_touch="
                    << double_from_bits(gathered_records[offset + 7])
                    << ", ooc_safety="
                    << double_from_bits(gathered_records[offset + 8])
                    << ", ooc_budget_mb="
                    << double_from_bits(gathered_records[offset + 9])
                    << ", fingerprint=" << gathered_records[offset + 10]
                    << ':' << gathered_records[offset + 11] << '}';
        return description.str();
    };

    std::ostringstream oss;
    oss << "MUMPS analysis metadata differs across world_comm_ ranks 0 and "
        << mismatching_rank << ": " << describe_record(0) << "; "
        << describe_record(mismatching_rank)
        << "; configure identical block, user-permutation, and OOC policy "
           "controls on every "
           "caller rank before JOB=1";
    throw LinearSolverError(__FILE__, __LINE__, oss.str());
}

void MumpsLinearSolver::copy_analysis_controls(
    std::array<std::int32_t, 60>& icntl,
    std::array<double, 15>& cntl) const
{
    icntl.fill(0);
    cntl.fill(0.0);
    if (!analyzed_) {
        throw LinearSolverError(
            __FILE__, __LINE__,
            "MUMPS controls are unavailable before successful analysis");
    }
    if (!in_mumps_ || rank_ != 0)
        return;
    static_assert(std::extent_v<decltype(DMUMPS_STRUC_C::icntl)> == 60);
    static_assert(std::extent_v<decltype(DMUMPS_STRUC_C::cntl)> == 15);
    for (std::size_t i = 0; i < icntl.size(); ++i)
        icntl[i] = static_cast<std::int32_t>(mumps_.icntl[i]);
    std::copy(std::begin(mumps_.cntl), std::end(mumps_.cntl), cntl.begin());
}

void MumpsLinearSolver::factor(const double* values)
{
    analyze_pattern();
    factor_analyzed(values);
}

void MumpsLinearSolver::analyze_pattern()
{
    validate_collective_analysis_metadata();
    prepare_auto_out_of_core_for_analysis();

    if (rank_ == 0) {
        mumps_.a = nullptr;
        mumps_.irn = const_cast<int*>(pattern_irn_);
        mumps_.jcn = const_cast<int*>(pattern_jcn_);
    } else {
        mumps_.a = nullptr;
        mumps_.irn = nullptr;
        mumps_.jcn = nullptr;
    }

    run_job(MUMPS_JOB_ANALYZE);
    if (mumps_.infog[0] < 0) {
        std::ostringstream oss;
        oss << "MUMPS analysis failed: INFOG(1)=" << mumps_.infog[0]
            << " INFOG(2)=" << mumps_.infog[1] << " n=" << n_ << " nnz=" << nnz_
            << " ICNTL(6)=" << mumps_.icntl[5]
            << " ICNTL(7)=" << mumps_.icntl[6] << " ICNTL(35)=" << mumps_.icntl[34]
            << " ICNTL(14)=" << mumps_.icntl[13]
            << " ICNTL(15)=" << mumps_.icntl[14]
            << " NBLK=" << mumps_.nblk;
        if (mumps_.infog[0] == -51 && mumps_.infog[1] < 0) {
            oss << "; integer overflow in ordering (~"
                << -1000000LL * static_cast<long long>(mumps_.infog[1]) << ")";
        }
        throw LinearSolverError(__FILE__, __LINE__, oss.str());
    }
    analyzed_ = true;
    // Analysis diagnostics must be captured here: later jobs reuse INFOG for
    // factorization statistics. INFOG(16)/(17) are the max-over-ranks and rank-sum
    // working-memory estimates in MB; INFOG(5) is the estimated largest front.
    capture_analysis_diagnostics();
    resolve_auto_out_of_core_after_analysis();
}

void MumpsLinearSolver::release_centralized_coo_input()
{
    pattern_irn_ = nullptr;
    pattern_jcn_ = nullptr;
    mumps_.irn = nullptr;
    mumps_.jcn = nullptr;
    mumps_.a = nullptr;
    analyzed_ = false;
}

void MumpsLinearSolver::release_factor_values_input()
{
    mumps_.a = nullptr;
}

void MumpsLinearSolver::release_pattern_input()
{
    release_centralized_coo_input();
}

void MumpsLinearSolver::factor_analyzed(const double* values)
{
    if (!analyzed_) {
        throw LinearSolverError(
            __FILE__, __LINE__,
            "MUMPS factor_analyzed() called before successful analyze_pattern()");
    }
    if (rank_ == 0) {
        mumps_.a = const_cast<double*>(values);
    } else {
        mumps_.a = nullptr;
    }

    int retry_icntl14 = icntl14_;
    bool factorized = false;
    // Diagnostic no-ratchet mode: with the ICNTL(14) bisect harness we want a
    // workspace-too-small (-9/-20) at the tested value to FAIL VISIBLY (throw)
    // rather than climb, so the smallest ICNTL(14) that factors is unambiguous.
    const bool noratchet = env_flag_enabled("MUMPS_ICNTL14_NORATCHET");

    // Factorize-only retries reuse the analysis, so they need no re-analysis.
    // Every JOB=2 still borrows the same centralized irn/jcn/a arrays, however;
    // the caller may detach and free them only after this method succeeds. Two
    // workspace-relaxable failures are retried by growing ICNTL(14):
    //   INFOG(1)=-9  : internal real workarray too small (delayed pivoting).
    //   INFOG(1)=-20 : internal reception buffer too small for a front message
    //                  (large dense separator front in the parallel factor;
    //                  INFOG(2) = the message bytes). The MUMPS comm buffers are
    //                  sized from the workspace x ICNTL(14), so the same knob grows
    //                  them. (Seen at BNS res15 n=209299: INFOG(2)~277 MB front.)
    // Each retry only re-runs the numerical factorization -- never a restart.
    // Capless geometric climb: grow ICNTL(14) by 50% each retry (500 -> 750 ->
    // 1125 -> ...) and keep going until the factor fits OR the node runs out of
    // memory -- a hard OOM surfaces as a non-(-9/-20) MUMPS code (e.g. -13
    // allocation failure) that ends the loop. No fixed ceiling: on a node with
    // spare RAM this absorbs a bigger per-rank buffer rather than failing -20, so
    // the factor can keep as many ranks as possible at high resolution.
    // kMaxRelaxationRetries is only an infinite-loop guard (500*1.5^40 is
    // astronomical; a hard OOM ends the climb long before it is reached).
    constexpr int kMaxRelaxationRetries = 40;
    int relaxation_retries = 0;
    bool relaxation_growth_overflow = false;
    while (true) {
        run_job(MUMPS_JOB_FACTORIZE);
        if (mumps_.infog[0] != -9 && mumps_.infog[0] != -20) {
            factorized = (mumps_.infog[0] >= 0);
            break;  // success, or a different failure (hard OOM / etc.) -> stop
        }
        if (noratchet) {
            if (rank_ == 0) {
                std::cerr << "MUMPS_ICNTL14_NORATCHET: FACTORIZE returned INFOG(1)="
                          << mumps_.infog[0] << " at ICNTL(14)=" << retry_icntl14
                          << "; not retrying (clean-bisect fail)\n";
            }
            break;  // leave factorized=false -> throws at the tested ICNTL(14)
        }
        if (++relaxation_retries > kMaxRelaxationRetries) {
            break;
        }
        const int increase = retry_icntl14 / 2;
        if (increase > std::numeric_limits<int>::max() - retry_icntl14) {
            relaxation_growth_overflow = true;
            break;
        }
        retry_icntl14 += increase;  // x1.5 (integer)
        mumps_.icntl[13] = retry_icntl14;
        if (rank_ == 0) {
            std::cerr << "MUMPS factorization workspace too small (INFOG(2)="
                      << mumps_.infog[1] << "); retrying FACTORIZE with ICNTL(14)="
                      << retry_icntl14 << '\n';
        }
    }

    if (!factorized || mumps_.infog[0] < 0) {
        std::ostringstream oss;
        oss << "MUMPS factorization failed: INFOG(1)=" << mumps_.infog[0]
            << " INFOG(2)=" << mumps_.infog[1] << " n=" << n_ << " nnz=" << nnz_
            << " ICNTL(7)=" << mumps_.icntl[6] << " ICNTL(35)=" << mumps_.icntl[34]
            << " ICNTL(14)=" << mumps_.icntl[13];
        if (relaxation_growth_overflow) {
            oss << "; ICNTL(14) workspace retry would exceed the signed-int range";
        } else if (mumps_.infog[0] == -51 && mumps_.infog[1] < 0) {
            oss << "; integer overflow in ordering (~"
                << -1000000LL * static_cast<long long>(mumps_.infog[1]) << ")";
        } else if (mumps_.infog[0] == -9 || mumps_.infog[0] == -20) {
            oss << "; workspace/comm-buffer too small even at ICNTL(14)="
                << mumps_.icntl[13]
                << " (workarray = 11x the analysis estimate) — genuine memory "
                << "exhaustion. Reduce MPI ranks, enable out-of-core "
                << "(MUMPS_OOC=1), or use more nodes/memory";
        }
        throw LinearSolverError(__FILE__, __LINE__, oss.str());
    }

    factor_retry_count_ = relaxation_retries;
    capture_factor_diagnostics(retry_icntl14);
    print_infog_trace();
}

void MumpsLinearSolver::capture_analysis_diagnostics()
{
    last_actual_ordering_ = mumps_.infog[6];
    estimated_factor_memory_mb_ = mumps_.infog[15];
    estimated_factor_memory_total_mb_ = mumps_.infog[16];
    estimated_max_front_order_ = mumps_.infog[4];
    estimated_factor_nnz_ = decode_mumps_count(mumps_.infog[19]);
    estimated_factor_real_slots_ = decode_mumps_count(mumps_.infog[2]);
    estimated_factor_integer_slots_ = decode_mumps_count(mumps_.infog[3]);
    estimated_tree_node_count_ = mumps_.infog[5];
    estimated_factor_flops_gflop_ = mumps_.rinfog[0] / 1e9;
}

void MumpsLinearSolver::capture_factor_diagnostics(int successful_icntl14)
{
    successful_factor_icntl14_ = successful_icntl14;
    // Preserve successful ICNTL(14) for next iteration; clamp to [20, 2000]
    // (the resolution-keyed seed reaches 2000 above res 17, and res17
    // FORCE_BALANCE measured a working value of 1125 that the old 1000
    // ceiling could not store).
    icntl14_ = std::max(20, std::min(successful_icntl14, 2000));
    last_icntl14_ = icntl14_;
    last_actual_ordering_ = mumps_.infog[6];
    // INFOG(18) is allocated factor memory on the max-allocation rank;
    // INFOG(21)/(22) are effective factor memory (max-rank / rank sum).
    // INFOG(29)/(9) use MUMPS' negative-millions encoding on 32-bit overflow.
    // INFOG(11) is the largest front *order*, not a memory count. RINFOG(3) is
    // the elimination flop count.
    factor_allocated_memory_mb_ = mumps_.infog[17];
    factor_memory_mb_ = mumps_.infog[20];
    factor_memory_total_mb_ = mumps_.infog[21];
    factor_nnz_ = decode_mumps_count(mumps_.infog[28]);
    factor_real_slots_ = decode_mumps_count(mumps_.infog[8]);
    max_front_order_ = mumps_.infog[10];
    delayed_pivot_count_ = mumps_.infog[12];
    factor_flops_gflop_ = mumps_.rinfog[2] / 1e9;
    // Capture null-pivot diagnostics (only meaningful when ICNTL(24)=1 was
    // active on this factor; INFOG(28) is always populated by MUMPS).
    if (null_pivot_detection_) {
        last_null_pivot_count_ = mumps_.infog[27];
        last_null_pivot_list_1based_.clear();
        if (last_null_pivot_count_ > 0 && mumps_.pivnul_list != nullptr) {
            last_null_pivot_list_1based_.assign(
                mumps_.pivnul_list,
                mumps_.pivnul_list + last_null_pivot_count_);
        }
    } else {
        last_null_pivot_count_ = 0;
        last_null_pivot_list_1based_.clear();
    }
}

void MumpsLinearSolver::print_infog_trace() const
{
    if (rank_ == 0 && env_flag_enabled("MUMPS_INFOG_TRACE")) {
        std::cerr << "MUMPS INFOG trace:"
                  << " n=" << n_
                  << " nnz=" << nnz_
                  << " INFOG(1)=" << mumps_.infog[0]
                  << " INFOG(2)=" << mumps_.infog[1]
                  << " INFOG(3)=" << mumps_.infog[2]
                  << " INFOG(4)=" << mumps_.infog[3]
                  << " INFOG(6)=" << mumps_.infog[5]
                  << " INFOG(7)=" << mumps_.infog[6]
                  << " INFOG(11)=" << mumps_.infog[10]
                  << " INFOG(12)=" << mumps_.infog[11]
                  << " INFOG(13)=" << mumps_.infog[12]
                  << " INFOG(20)=" << mumps_.infog[19]
                  << " INFOG(21)=" << mumps_.infog[20]
                  << " INFOG(22)=" << mumps_.infog[21]
                  << " INFOG(28)=" << mumps_.infog[27]
                  << " INFOG(29)=" << mumps_.infog[28]
                  << " INFOG(35)=" << mumps_.infog[34]
                  << " INFOG(36)=" << mumps_.infog[35]
                  << " INFOG(37)=" << mumps_.infog[36]
                  << " INFO(9)=" << mumps_.info[8]
                  << " INFO(21)=" << mumps_.info[20]
                  << " INFO(22)=" << mumps_.info[21]
                  << " INFO(24)=" << mumps_.info[23]
                  << " INFO(27)=" << mumps_.info[26]
                  << " INFO(28)=" << mumps_.info[27]
                  << " INFO(30)=" << mumps_.info[29]
                  << " INFO(31)=" << mumps_.info[30]
                  << " RINFOG(1)=" << mumps_.rinfog[0]
                  << " RINFOG(2)=" << mumps_.rinfog[1]
                  << " RINFOG(3)=" << mumps_.rinfog[2]
                  << " RINFOG(14)=" << mumps_.rinfog[13]
                  << " ICNTL(6)=" << mumps_.icntl[5]
                  << " ICNTL(7)=" << mumps_.icntl[6]
                  << " ICNTL(8)=" << mumps_.icntl[7]
                  << " ICNTL(11)=" << mumps_.icntl[10]
                  << " ICNTL(13)=" << mumps_.icntl[12]
                  << " ICNTL(14)=" << mumps_.icntl[13]
                  << " ICNTL(15)=" << mumps_.icntl[14]
                  << " ICNTL(24)=" << mumps_.icntl[23]
                  << " ICNTL(35)=" << mumps_.icntl[34]
                  << " NBLK=" << mumps_.nblk
                  << '\n';
    }
}

void MumpsLinearSolver::enable_null_pivot_detection(bool enable, double threshold)
{
    null_pivot_detection_ = enable;
    null_pivot_threshold_ = threshold;
    if (initialized_) {
        mumps_.icntl[23] = enable ? 1 : 0;
        if (enable) {
            mumps_.cntl[2] = threshold;
        }
    }
}

void MumpsLinearSolver::analyze_symmetric_permutation(
    const double* values,
    std::vector<int>& sym_perm_1based)
{
    validate_collective_analysis_metadata();
    prepare_auto_out_of_core_for_analysis();

    if (rank_ == 0) {
        mumps_.a = const_cast<double*>(values);
    } else {
        mumps_.a = nullptr;
    }
    mumps_.icntl[18] = 0; // ICNTL(19): no Schur complement.

    run_job(MUMPS_JOB_ANALYZE);
    if (mumps_.infog[0] < 0) {
        std::ostringstream oss;
        oss << "MUMPS permutation analysis failed: INFOG(1)="
            << mumps_.infog[0] << " INFOG(2)=" << mumps_.infog[1]
            << " n=" << n_ << " nnz=" << nnz_
            << " ICNTL(7)=" << mumps_.icntl[6];
        throw LinearSolverError(__FILE__, __LINE__, oss.str());
    }
    capture_analysis_diagnostics();
    resolve_auto_out_of_core_after_analysis();

    analyzed_ = true;
    copy_symmetric_permutation_1based(sym_perm_1based);
}

void MumpsLinearSolver::copy_symmetric_permutation_1based(
    std::vector<int>& sym_perm_1based) const
{
    if (!analyzed_) {
        throw LinearSolverError(
            __FILE__, __LINE__,
            "MUMPS symmetric permutation requested before successful analysis");
    }
    if (rank_ == 0) {
        if (mumps_.sym_perm == nullptr) {
            throw LinearSolverError(
                __FILE__, __LINE__,
                "MUMPS successful analysis did not return SYM_PERM");
        }
        sym_perm_1based.assign(static_cast<std::size_t>(n_), 0);
        for (int i = 0; i < n_; ++i)
            sym_perm_1based[static_cast<std::size_t>(i)] = mumps_.sym_perm[i];
    } else {
        sym_perm_1based.clear();
    }
}

void MumpsLinearSolver::copy_column_permutation_1based(
    std::vector<int>& uns_perm_1based, bool& matching_applied) const
{
    if (!analyzed_) {
        throw LinearSolverError(
            __FILE__, __LINE__,
            "MUMPS column permutation requested before successful analysis");
    }
    matching_applied = false;
    if (rank_ == 0) {
        uns_perm_1based.assign(static_cast<std::size_t>(n_), 0);
        if (mumps_.uns_perm == nullptr) {
            for (int i = 0; i < n_; ++i)
                uns_perm_1based[static_cast<std::size_t>(i)] = i + 1;
        } else {
            for (int i = 0; i < n_; ++i)
                uns_perm_1based[static_cast<std::size_t>(i)] = mumps_.uns_perm[i];
            matching_applied = true;
        }
    } else {
        uns_perm_1based.clear();
    }
}

void MumpsLinearSolver::extract_schur(
    const double* values,
    const std::vector<int>& listvar_schur_1based,
    std::vector<double>& schur_col_major)
{
    validate_collective_analysis_metadata();

    if (block_analysis_enabled_) {
        throw LinearSolverError(
            __FILE__, __LINE__,
            "MUMPS Schur extraction is incompatible with ICNTL(15) block analysis");
    }
    const int size_schur = static_cast<int>(listvar_schur_1based.size());
    if (size_schur <= 0 || size_schur >= n_) {
        std::ostringstream oss;
        oss << "MUMPS Schur extraction requires 0 < size_schur < n"
            << " size_schur=" << size_schur << " n=" << n_;
        throw LinearSolverError(__FILE__, __LINE__, oss.str());
    }

    std::vector<int> listvar = listvar_schur_1based;
    if (rank_ == 0) {
        for (int value : listvar) {
            if (value <= 0 || value > n_) {
                std::ostringstream oss;
                oss << "MUMPS Schur extraction variable out of range: " << value
                    << " n=" << n_;
                throw LinearSolverError(__FILE__, __LINE__, oss.str());
            }
        }
    }

    // Use the recommended centralized-by-columns Schur path:
    // ICNTL(19)=2, NPROW=NPCOL=1, PAR=1. Schur is incompatible with
    // maximum-transversal/scaling/error-analysis/parallel-analysis, so force
    // those controls off for this diagnostic instance.
    mumps_.icntl[5] = 0;   // ICNTL(6): no maximum transversal/scaling prepass.
    mumps_.icntl[7] = 0;   // ICNTL(8): no scaling.
    mumps_.icntl[10] = 0;  // ICNTL(11): no error analysis.
    mumps_.icntl[18] = 2;  // ICNTL(19): Schur complement, centralized columns.
    mumps_.icntl[27] = 1;  // ICNTL(28): sequential analysis.
    mumps_.size_schur = size_schur;
    mumps_.nprow = 1;
    mumps_.npcol = 1;
    mumps_.mblock = 100;
    mumps_.nblock = 100;
    if (rank_ == 0)
        mumps_.listvar_schur = listvar.data();
    else
        mumps_.listvar_schur = nullptr;

    if (rank_ == 0) {
        mumps_.a = const_cast<double*>(values);
    } else {
        mumps_.a = nullptr;
    }

    int retry_icntl14 = icntl14_;
    constexpr int max_retries = 3;

    prepare_auto_out_of_core_for_analysis();
    run_job(MUMPS_JOB_ANALYZE);
    if (mumps_.infog[0] < 0) {
        std::ostringstream oss;
        oss << "MUMPS Schur analysis failed: INFOG(1)=" << mumps_.infog[0]
            << " INFOG(2)=" << mumps_.infog[1] << " n=" << n_
            << " nnz=" << nnz_ << " size_schur=" << size_schur
            << " ICNTL(7)=" << mumps_.icntl[6];
        throw LinearSolverError(__FILE__, __LINE__, oss.str());
    }
    capture_analysis_diagnostics();
    resolve_auto_out_of_core_after_analysis();

    if (rank_ == 0) {
        mumps_.schur_lld = size_schur;
        schur_col_major.assign(
            static_cast<std::size_t>(size_schur) *
                static_cast<std::size_t>(size_schur),
            0.0);
        mumps_.schur = schur_col_major.data();
    } else {
        schur_col_major.clear();
        mumps_.schur_lld = 0;
        mumps_.schur = nullptr;
    }

    bool factorized = false;
    for (int attempt = 0; attempt <= max_retries; ++attempt) {
        mumps_.icntl[13] = retry_icntl14;
        run_job(MUMPS_JOB_FACTORIZE);
        if ((mumps_.infog[0] == -9 || mumps_.infog[0] == -20) &&
            attempt < max_retries) {
            const long long deficit_w = static_cast<long long>(mumps_.infog[1]);
            const long long min_mw = static_cast<long long>(mumps_.info[14]);
            if (mumps_.infog[0] == -20) {
                retry_icntl14 = static_cast<int>(std::min(std::max(static_cast<long long>(retry_icntl14) * 2, 500LL),
                                                           1000LL));
            } else if (min_mw > 0 && deficit_w > 0) {
                const double allocated_w =
                    static_cast<double>(min_mw) * 1e6 * (1.0 + retry_icntl14 * 0.01);
                const double needed_w = allocated_w + static_cast<double>(deficit_w);
                const double exact_pct =
                    std::ceil((needed_w / (static_cast<double>(min_mw) * 1e6) - 1.0) * 100.0);
                retry_icntl14 = exact_pct >= 980.0 ? 1000 : static_cast<int>(exact_pct) + 20;
            } else {
                retry_icntl14 = static_cast<int>(
                    std::min(static_cast<long long>(retry_icntl14) * 2, 1000LL));
            }
            if (rank_ == 0) {
                std::cerr << "MUMPS Schur factorization retry"
                          << " INFOG(1)=" << mumps_.infog[0]
                          << " INFOG(2)="
                          << mumps_.infog[1]
                          << "); retrying FACTORIZE with ICNTL(14)="
                          << retry_icntl14 << '\n';
            }
            continue;
        }
        factorized = (mumps_.infog[0] >= 0);
        break;
    }

    if (!factorized || mumps_.infog[0] < 0) {
        std::ostringstream oss;
        oss << "MUMPS Schur factorization failed: INFOG(1)=" << mumps_.infog[0]
            << " INFOG(2)=" << mumps_.infog[1] << " n=" << n_
            << " nnz=" << nnz_ << " size_schur=" << size_schur
            << " ICNTL(7)=" << mumps_.icntl[6]
            << " ICNTL(14)=" << mumps_.icntl[13];
        throw LinearSolverError(__FILE__, __LINE__, oss.str());
    }

    capture_factor_diagnostics(retry_icntl14);
}

void MumpsLinearSolver::solve(double* rhs_inout)
{
    if (in_mumps_) {
        const int saved_icntl9 = mumps_.icntl[8];
        double* const saved_rhs = mumps_.rhs;
        const int saved_nrhs = mumps_.nrhs;
        const int saved_lrhs = mumps_.lrhs;
        const auto restore_solve_state = [&]() {
            mumps_.icntl[8] = saved_icntl9;
            mumps_.rhs = saved_rhs;
            mumps_.nrhs = saved_nrhs;
            mumps_.lrhs = saved_lrhs;
        };
        mumps_.icntl[8] = 1; // ICNTL(9): solve A x = b.
        // JOB=3 is synchronous: on the MUMPS host it overwrites rhs_inout with
        // the solution before returning. Borrow the caller's mutable buffer for
        // that interval instead of round-tripping through a rank-0 workspace.
        mumps_.rhs = (rank_ == 0) ? rhs_inout : nullptr;
        mumps_.nrhs = 1;
        mumps_.lrhs = n_;
        try {
            run_job(MUMPS_JOB_SOLVE);
        } catch (...) {
            restore_solve_state();
            throw;
        }
        restore_solve_state();
    }
    if (rank_ == 0 && !solution_column_permutation_1based_.empty()) {
        // JOB=3 returned y for A*P. Restore original unknown order on the MUMPS
        // host before the existing full-communicator bcast so every rank gets x.
        for (std::size_t i = 0; i < solution_column_permutation_1based_.size(); ++i) {
            const std::size_t original_index = static_cast<std::size_t>(
                solution_column_permutation_1based_[i] - 1);
            solution_column_permutation_workspace_[original_index] = rhs_inout[i];
        }
        std::copy(solution_column_permutation_workspace_.begin(),
                  solution_column_permutation_workspace_.end(), rhs_inout);
    }
    // Solution lives on world rank 0 (== MUMPS host). Broadcast over the FULL comm
    // so the GMRES vectors stay consistent on the non-factor ranks too. (When
    // ranks_per_node==0, world_comm_==comm_ and this is the original behavior.)
    MPI_Bcast(rhs_inout, n_, MPI_DOUBLE, 0, world_comm_); // NOLINT(bugprone-casting-through-void)
}

void MumpsLinearSolver::solve_transpose(double* rhs_inout)
{
    if (in_mumps_) {
        const int saved_icntl9 = mumps_.icntl[8];
        double* const saved_rhs = mumps_.rhs;
        const int saved_nrhs = mumps_.nrhs;
        const int saved_lrhs = mumps_.lrhs;
        const auto restore_solve_state = [&]() {
            mumps_.icntl[8] = saved_icntl9;
            mumps_.rhs = saved_rhs;
            mumps_.nrhs = saved_nrhs;
            mumps_.lrhs = saved_lrhs;
        };
        mumps_.icntl[8] = 0; // ICNTL(9): any value other than 1 solves A^T x = b.
        mumps_.rhs = (rank_ == 0) ? rhs_inout : nullptr;
        mumps_.nrhs = 1;
        mumps_.lrhs = n_;
        try {
            run_job(MUMPS_JOB_SOLVE);
        } catch (...) {
            restore_solve_state();
            throw;
        }
        restore_solve_state();
    }
    MPI_Bcast(rhs_inout, n_, MPI_DOUBLE, 0, world_comm_); // NOLINT(bugprone-casting-through-void)
}

void MumpsLinearSolver::reset()
{
    if (initialized_) {
        run_job(MUMPS_JOB_END);
        initialized_ = false;
    }
    pattern_irn_ = nullptr;
    pattern_jcn_ = nullptr;
    analyzed_ = false;
    block_ptr_1based_.clear();
    block_variables_1based_.clear();
    block_analysis_enabled_ = false;
    block_matching_overridden_ = false;
    user_permutation_1based_.clear();
    user_permutation_enabled_ = false;
    solution_column_permutation_1based_.clear();
    solution_column_permutation_workspace_.clear();
}

} // namespace Kadath

#endif // CELEPHAIS_USE_MUMPS
