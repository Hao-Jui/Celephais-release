/*
    Added 2026 Hao-Jui Kuan

    Diagnostic Newton backend (CELEPHAIS_SOLVER=jfnk-schur): a one-shot
    pre-backend gate for the hand-built aggregate-Schur direction. It does NOT
    converge the system. Two phases:

      Phase 1 -- PARTITION CONTRACT GATE (structural, no COO; runs in seconds).
        Build semantic aggregates (BNS default: star1, star2, bispheric exterior,
        asymptotic tail; override with SCHUR_AGGREGATES="0-2;3-5;6-10;11-last").
        Promote rows and columns whose support is aggregate-local into the bulk side, keeping
        GlobalInt / gauge / cross-aggregate coupling on the interface side.
        Fail hard unless: unclassified_rows == 0, unclassified_cols == 0, every
        aggregate block is square, interface rows == interface cols, and
        0 < interface < n. Stop here with SCHUR_PROBE_CONTRACT_ONLY=1 to
        iterate the contract cheaply.

      Phase 2 -- HAND FORMATION PROBE (only if the contract passes).
        S = A_II - sum_g A_IB^(g) (A_BB^(g))^{-1} A_BI^(g). Time, separately:
        per-aggregate dgetrf(A_BB), dgetrs(A_BB^{-1} A_BI), dgemm accumulate
        into S, dgetrf(S). Count cross-aggregate bulk entries (must be 0 for
        the bulk to be block-diagonal). Per-block rcond via SCHUR_PROBE_RCOND
        (default off). Report peak RSS. Compare against the full MUMPS factor.
        SCHUR_PROBE_STRICT=1 makes any numeric gate failure return a
        non-zero process status for automated go/no-go runs.

      SCHUR_PROBE_NULLITY=1 -- RANK-REVEALING nullity probe (replaces the
        formation for this run). Per A_BB block: dgesdd(jobz='N') singular-value
        spectrum -> null_dim at several tolerances, the smallest-sigma tail, and
        the largest tail gap ratio. Clean kernel = large gap + tol-stable
        null_dim; smeared ill-conditioning = no gap + tol-sensitive null_dim.
        dgetrf info is the first zero-pivot index, NOT nullity; THIS measures
        nullity. With SCHUR_PROBE_NULLVEC=1 it also runs stage 2:
        dgesdd(jobz='S') for the near-null right vectors, classified by ENERGY
        (sum of squared components, basis-independent over the null subspace)
        per column class + domain -- a fingerprint of WHERE the kernel lives,
        NOT analytic proof of constant/CKV. Stage 2 is capped by
        SCHUR_PROBE_NULLVEC_MAX (max null_dim, default 64) and
        SCHUR_PROBE_NULLVEC_DIM_MAX (max block m for the heavy full VT,
        default 8000; larger blocks are logged + skipped, no silent truncation).
        SCHUR_PROBE_NULLITY_AGGREGATES selects a comma-separated subset
        by id / gN / label; SCHUR_PROBE_NULLITY_DIM_MAX skips larger
        blocks before dense allocation. With SCHUR_PROBE_PINNING=1,
        stage 2 also applies the omitted interface rows to the near-null basis
        (A_IB V_N) and buckets the pinning energy by GlobalInt, true
        cross-aggregate TauMatch, selector leaks, and residual interface rows.
        Stage 4 (same flag) re-buckets the active rows by (taxonomy, set of
        aggregates their actual COO columns touch) -- endpoints inferred from
        the matrix via columns[c].domain -> domain_to_aggregate, NOT the
        unreliable dom_pair tag -- and prints a self/cross/global energy split,
        a merge-vs-reclaim verdict, and a reclaim candidate list. To measure the
        merged exterior directly, set SCHUR_SPLIT_EXTERIOR_TAIL=0 (3
        aggregates: star1, star2, exterior=bispheric+tail).

      MEASURED VERDICT (res9 BNS, 2026-06-04) -- aggregate / disjoint-block
      Schur is NOT a viable default backend. This code is RETAINED for
      diagnostics only. Three independent kills:
        1. Disjoint blocks are singular and cross/global-pinned. g3
           asymptotic_tail has a CLEAN kernel (null_dim=126, spectral gap
           6.09e7), but Stage-4 numeric-support pinning attributes 100% of it to
           the bispheric<->tail seam (TauMatch {g2,g3}=1.0; self=0, cross=1,
           global=0) -> the tail cannot be deflated in isolation.
        2. Merging g2+g3 removes the seam, but the merged exterior (m=23259) has
           a SMEARED near-null spectrum: null_dim = 605 / 790 / 1192 / 2505 at
           tol = 1e-13 / 1e-11 / 1e-9 / 1e-6, max tail gap only 4.68. No spectral
           gap => no well-defined coarse space => deflation/BDDC is undefined
           (the compactified-exterior decay continuum r^-(l+1) has no gap to cut).
        3. Economics: the dense exterior factor (~180s LU) already dwarfs the
           full sparse MUMPS factor (~2.8s) at res9.
      The earlier "~45% kernel" figure was a misread of dgetrf info (first
      zero-pivot INDEX, not nullity) and is retracted. The only non-disjoint
      Schur that works is the bordered-global-constraint form (bulk = one full
      MUMPS factor, the wpc24 PASS); per-domain / per-aggregate disjoint Schur is
      closed. res15 factor-cost lever stays MUMPS ranks / GPU dense fronts.
      Evidence logs (gitignored, local): build-logs/bns_stage4_g3_*.log,
      build-logs/bns_stage1_merged_exterior_*.log.

      SCHWARZ-DIRICHLET UPDATE (res9 BNS, 2026-06-25) -- the frozen-neighbour
      local operators are full column rank, so the free-block singularity above
      is a boundary-condition artefact for overlapping Schwarz. However, the
      decisive full-Jacobian left/right inverse-iteration oracle coarse test is
      RED: one-level Schwarz-Dirichlet stalls at 120 GMRES iters with rel_resid
      2.73e-5; Petrov oracle coarse spaces from 12 and 20 near-null modes are
      worse (2.60e-4 and 6.82e-4). Conclusion: do not spend on more
      Schwarz plus algebraic-near-null tuning, including GenEO or analytic
      gauge-CKV repairs, for this route. Reopen only for a different
      preconditioner class: e.g. a physics-based exterior coarse space
      (analytic multipole hierarchy) or a domain decomposition that does not
      collapse the compactified exterior into one smeared block. For res19
      memory pressure, prefer MUMPS-OOC / more nodes / alternate sparse-direct
      packages.
*/

#include "mpi.h"
#include "For_Kadath/System_of_eqs/system_of_eqs.hpp"
#include "For_Kadath/Array/array.hpp"
#include "For_Kadath/Array/exceptions.hpp"
#include "For_Kadath/Space/bin_ns.hpp"
#include "For_Kadath/Space/bin_ns_nosym.hpp"
#include "For_Kadath/Utilities/runtime_env.hpp"

#include "newton_norms.hpp"

#include "Linear_algebra/jacobian_assembler.hpp"
#include "Linear_algebra/krylov_solver.hpp"

#include <map>

#ifdef CELEPHAIS_USE_MUMPS
#include "Linear_algebra/mumps_linear_solver.hpp"
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <iostream>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <random>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <sys/resource.h>

namespace Kadath
{
    namespace
    {
        double elapsed_time(std::chrono::time_point<std::chrono::system_clock> const& begin)
        {
            return std::chrono::duration<double>(std::chrono::system_clock::now() - begin).count();
        }

        double peak_rss_gb()
        {
            struct rusage usage;
            getrusage(RUSAGE_SELF, &usage);
#if defined(__APPLE__)
            return static_cast<double>(usage.ru_maxrss) / (1024.0 * 1024.0 * 1024.0);
#else
            return static_cast<double>(usage.ru_maxrss) / (1024.0 * 1024.0);
#endif
        }

        enum class DtnTraceChannel {
            P,
            N,
            Beta,
            Other,
        };

        struct DtnTraceEnergyBucket {
            double p = 0.0;
            double n = 0.0;
            double beta = 0.0;
            double other = 0.0;
            int p_rows = 0;
            int n_rows = 0;
            int beta_rows = 0;
            int other_rows = 0;
        };

        struct DtnTraceOwnerCensus {
            int rows = 0;
            double energy = 0.0;
        };

        const char* dtn_row_taxonomy_name(RowTaxonomy taxonomy)
        {
            switch (taxonomy) {
                case RowTaxonomy::Vol: return "Vol";
                case RowTaxonomy::TauBc: return "TauBc";
                case RowTaxonomy::TauMatch: return "TauMatch";
                case RowTaxonomy::GlobalInt: return "GlobalInt";
                case RowTaxonomy::Unknown: return "Unknown";
            }
            return "Unknown";
        }

        std::string lowercase_ascii(std::string value)
        {
            for (char& ch : value) {
                ch = static_cast<char>(
                    std::tolower(static_cast<unsigned char>(ch)));
            }
            return value;
        }

        std::string trim_ascii_space(std::string value)
        {
            const std::size_t first = value.find_first_not_of(" \t\n\r");
            if (first == std::string::npos)
                return "";
            const std::size_t last = value.find_last_not_of(" \t\n\r");
            return value.substr(first, last - first + 1);
        }

        DtnTraceChannel classify_dtn_trace_channel(const std::string& name)
        {
            const std::string lower = lowercase_ascii(trim_ascii_space(name));
            if (lower == "p" || lower == "conf" || lower == "psi")
                return DtnTraceChannel::P;
            if (lower == "n" || lower == "lapse")
                return DtnTraceChannel::N;
            if (lower == "bet" || lower == "beta" || lower == "shift" ||
                lower.find("bet") == 0)
                return DtnTraceChannel::Beta;
            return DtnTraceChannel::Other;
        }

        DtnTraceChannel choose_counted_trace_channel(int p_count,
                                                     int n_count,
                                                     int beta_count)
        {
            const int best = std::max({p_count, n_count, beta_count});
            if (best <= 0)
                return DtnTraceChannel::Other;
            int ties = 0;
            ties += (p_count == best) ? 1 : 0;
            ties += (n_count == best) ? 1 : 0;
            ties += (beta_count == best) ? 1 : 0;
            if (ties != 1)
                return DtnTraceChannel::Other;
            if (p_count == best)
                return DtnTraceChannel::P;
            if (n_count == best)
                return DtnTraceChannel::N;
            return DtnTraceChannel::Beta;
        }

        DtnTraceChannel infer_dtn_trace_row_channel(
            const RowMetadata& row,
            const TaggedJacobianMetadata& metadata,
            const std::vector<int>& support_columns,
            int gamma_domain)
        {
            DtnTraceChannel channel =
                classify_dtn_trace_channel(row.owner_var_name);
            if (channel != DtnTraceChannel::Other)
                return channel;

            int p_count = 0;
            int n_count = 0;
            int beta_count = 0;
            auto tally = [&](bool compact_only) {
                p_count = 0;
                n_count = 0;
                beta_count = 0;
                for (int col : support_columns) {
                    if (col < 0 || col >= metadata.ncols)
                        continue;
                    const ColumnMetadata& column =
                        metadata.columns[static_cast<std::size_t>(col)];
                    if (compact_only && column.domain != gamma_domain)
                        continue;
                    switch (classify_dtn_trace_channel(column.var_name)) {
                        case DtnTraceChannel::P: ++p_count; break;
                        case DtnTraceChannel::N: ++n_count; break;
                        case DtnTraceChannel::Beta: ++beta_count; break;
                        case DtnTraceChannel::Other: break;
                    }
                }
            };

            tally(/*compact_only=*/true);
            channel = choose_counted_trace_channel(p_count, n_count, beta_count);
            if (channel != DtnTraceChannel::Other)
                return channel;

            tally(/*compact_only=*/false);
            return choose_counted_trace_channel(p_count, n_count, beta_count);
        }

        int infer_dtn_trace_mode_bucket(const RowMetadata& row,
                                        const TaggedJacobianMetadata& metadata,
                                        const std::vector<int>& support_columns,
                                        DtnTraceChannel channel,
                                        int gamma_domain)
        {
            if (row.taxonomy == RowTaxonomy::TauMatch && row.eq_local_row >= 0) {
                int remaining = row.eq_local_row;
                const int component_factor =
                    (channel == DtnTraceChannel::Beta) ? 3 : 1;
                for (int l = 1; l < 512; ++l) {
                    const int row_count = component_factor * (2 * l + 1);
                    if (remaining < row_count)
                        return l;
                    remaining -= row_count;
                }
                return 511;
            }

            if (row.basis_mode >= 0)
                return row.basis_mode;

            int best_mode = -1;
            auto consider = [&](bool compact_only) {
                for (int col : support_columns) {
                    if (col < 0 || col >= metadata.ncols)
                        continue;
                    const ColumnMetadata& column =
                        metadata.columns[static_cast<std::size_t>(col)];
                    if (compact_only && column.domain != gamma_domain)
                        continue;
                    if (classify_dtn_trace_channel(column.var_name) != channel)
                        continue;
                    best_mode = std::max(best_mode, column.basis_mode);
                }
            };

            consider(/*compact_only=*/true);
            if (best_mode >= 0)
                return best_mode;
            consider(/*compact_only=*/false);
            return std::max(0, best_mode);
        }

        void add_dtn_trace_energy(DtnTraceEnergyBucket& bucket,
                                  DtnTraceChannel channel,
                                  double energy)
        {
            switch (channel) {
                case DtnTraceChannel::P:
                    bucket.p += energy;
                    ++bucket.p_rows;
                    break;
                case DtnTraceChannel::N:
                    bucket.n += energy;
                    ++bucket.n_rows;
                    break;
                case DtnTraceChannel::Beta:
                    bucket.beta += energy;
                    ++bucket.beta_rows;
                    break;
                case DtnTraceChannel::Other:
                    bucket.other += energy;
                    ++bucket.other_rows;
                    break;
            }
        }

        double dtn_bucket_channel_energy(const DtnTraceEnergyBucket& bucket,
                                         DtnTraceChannel channel)
        {
            switch (channel) {
                case DtnTraceChannel::P: return bucket.p;
                case DtnTraceChannel::N: return bucket.n;
                case DtnTraceChannel::Beta: return bucket.beta;
                case DtnTraceChannel::Other: return bucket.other;
            }
            return bucket.other;
        }

        int dtn_bucket_channel_rows(const DtnTraceEnergyBucket& bucket,
                                    DtnTraceChannel channel)
        {
            switch (channel) {
                case DtnTraceChannel::P: return bucket.p_rows;
                case DtnTraceChannel::N: return bucket.n_rows;
                case DtnTraceChannel::Beta: return bucket.beta_rows;
                case DtnTraceChannel::Other: return bucket.other_rows;
            }
            return bucket.other_rows;
        }

        double safe_fraction(double numerator, double denominator)
        {
            return (denominator > 0.0) ? numerator / denominator : 0.0;
        }

        double reduction_factor_from_energy(double before, double after)
        {
            if (before <= 0.0)
                return 1.0;
            if (after <= 0.0)
                return std::numeric_limits<double>::infinity();
            return std::sqrt(before / after);
        }

        void print_dtn_trace_energy_line(
            const char* field,
            DtnTraceChannel channel,
            const std::vector<DtnTraceEnergyBucket>& buckets,
            const DtnTraceEnergyBucket& tail)
        {
            double subtotal = 0.0;
            int rows = 0;
            std::cout << "dtn-trace energy field=" << field;
            for (std::size_t l = 0; l < buckets.size(); ++l) {
                const double energy =
                    dtn_bucket_channel_energy(buckets[l], channel);
                subtotal += energy;
                rows += dtn_bucket_channel_rows(buckets[l], channel);
                std::cout << " l" << l << "=" << energy;
            }
            const double tail_energy = dtn_bucket_channel_energy(tail, channel);
            subtotal += tail_energy;
            rows += dtn_bucket_channel_rows(tail, channel);
            std::cout << " tail=" << tail_energy
                      << " rows=" << rows
                      << " total=" << subtotal << "\n";
        }

        void run_dtn_trace_probe(int rank,
                                 int nproc,
                                 int ndom,
                                 int column_count,
                                 double initial_error,
                                 double gamma_radius,
                                 const Array<double>& residual,
                                 const TaggedJacobianMetadata& metadata)
        {
            if (rank != 0)
                return;

            const auto probe_start = std::chrono::system_clock::now();
            const int lmax = std::max(0, env_int_value("XCTS_DTN_TRACE_PC_LMAX", 8));
            const int harmonic_count = (lmax + 1) * (lmax + 1);
            const int trace_channels = 5;
            const int ntrace = trace_channels * harmonic_count;
            const int gamma_domain = std::max(0, ndom - 1);

            std::vector<std::vector<int>> columns_per_row(
                static_cast<std::size_t>(metadata.nrows));
            if (static_cast<int>(metadata.rows_per_column.size()) == metadata.ncols) {
                for (int col = 0; col < metadata.ncols; ++col) {
                    for (int row :
                         metadata.rows_per_column[static_cast<std::size_t>(col)]) {
                        if (row < 0 || row >= metadata.nrows)
                            continue;
                        columns_per_row[static_cast<std::size_t>(row)]
                            .push_back(col);
                    }
                }
            }

            std::vector<DtnTraceEnergyBucket> buckets(
                static_cast<std::size_t>(lmax + 1));
            DtnTraceEnergyBucket tail;
            int tau_match_rows = 0;
            int gamma_trace_rows = 0;
            int unresolved_rows = 0;
            double selected_energy = 0.0;
            double captured_energy = 0.0;
            double scalar_energy = 0.0;
            double beta_energy = 0.0;
            double other_energy = 0.0;
            std::map<std::string, DtnTraceOwnerCensus> exterior_census;
            std::vector<char> trace_dtn_corrected(
                static_cast<std::size_t>(metadata.nrows), 0);

            const int rows_to_scan = std::min(
                metadata.nrows,
                static_cast<int>(residual.get_nbr()));
            for (int row_index = 0; row_index < rows_to_scan; ++row_index) {
                const RowMetadata& row =
                    metadata.rows[static_cast<std::size_t>(row_index)];
                if (row.dom >= gamma_domain - 1 || row.dom_pair >= gamma_domain - 1) {
                    std::ostringstream key;
                    key << "dom=" << row.dom
                        << " dom_pair=" << row.dom_pair
                        << " taxonomy=" << dtn_row_taxonomy_name(row.taxonomy)
                        << " owner="
                        << (trim_ascii_space(row.owner_var_name).empty()
                                ? "__none__"
                                : trim_ascii_space(row.owner_var_name));
                    DtnTraceOwnerCensus& census = exterior_census[key.str()];
                    ++census.rows;
                    const double value = residual(row_index);
                    census.energy += value * value;
                }
                if (row.taxonomy != RowTaxonomy::TauMatch)
                    continue;
                ++tau_match_rows;
                const bool touches_gamma =
                    row.dom == gamma_domain || row.dom_pair == gamma_domain;
                if (!touches_gamma)
                    continue;

                ++gamma_trace_rows;
                const std::vector<int>& support_columns =
                    columns_per_row[static_cast<std::size_t>(row_index)];
                const DtnTraceChannel channel = infer_dtn_trace_row_channel(
                    row, metadata, support_columns, gamma_domain);
                if (channel == DtnTraceChannel::Other)
                    ++unresolved_rows;
                const int mode_bucket = infer_dtn_trace_mode_bucket(
                    row, metadata, support_columns, channel, gamma_domain);
                const double value = residual(row_index);
                const double energy = value * value;
                selected_energy += energy;

                if (mode_bucket >= 0 && mode_bucket <= lmax) {
                    add_dtn_trace_energy(
                        buckets[static_cast<std::size_t>(mode_bucket)],
                        channel, energy);
                    captured_energy += energy;
                    if (channel == DtnTraceChannel::P ||
                        channel == DtnTraceChannel::N) {
                        scalar_energy += energy;
                    } else if (channel == DtnTraceChannel::Beta) {
                        beta_energy += energy;
                    } else {
                        other_energy += energy;
                    }
                    if (channel == DtnTraceChannel::P ||
                        channel == DtnTraceChannel::N ||
                        channel == DtnTraceChannel::Beta) {
                        trace_dtn_corrected[static_cast<std::size_t>(row_index)] = 1;
                    }
                } else {
                    add_dtn_trace_energy(tail, channel, energy);
                }
            }

            const double tail_energy =
                std::max(0.0, selected_energy - captured_energy);
            const double scalar_tail_energy = tail.p + tail.n;
            const double beta_tail_energy = tail.beta;
            const double other_tail_energy = tail.other;
            const double scalar_before_energy = scalar_energy + scalar_tail_energy;
            const double beta_before_energy = beta_energy + beta_tail_energy;
            const double other_before_energy = other_energy + other_tail_energy;
            const double scalar_after_energy = scalar_tail_energy;
            const double beta_after_energy = beta_tail_energy;
            const double other_after_energy = other_before_energy;
            const double total_after_energy =
                scalar_after_energy + beta_after_energy + other_after_energy;
            const double scalar_reduction =
                reduction_factor_from_energy(scalar_before_energy, scalar_after_energy);
            const double beta_reduction =
                reduction_factor_from_energy(beta_before_energy, beta_after_energy);
            const double total_reduction =
                reduction_factor_from_energy(selected_energy, total_after_energy);

            int scalar_rank = 0;
            int beta_rank = 0;
            int beta_expected_full_rows = 0;
            double scalar_z_min = std::numeric_limits<double>::infinity();
            double scalar_z_max = 0.0;
            double beta_z_min = std::numeric_limits<double>::infinity();
            double beta_z_max = 0.0;
            double scalar_alpha_energy = 0.0;
            double beta_alpha_energy = 0.0;
            const double safe_radius =
                gamma_radius > 0.0 ? gamma_radius : 1.0;
            for (int l = 0; l <= lmax; ++l) {
                const DtnTraceEnergyBucket& bucket =
                    buckets[static_cast<std::size_t>(l)];
                const double z_ext = static_cast<double>(l + 1) / safe_radius;
                const int scalar_rows = bucket.p_rows + bucket.n_rows;
                if (scalar_rows > 0) {
                    scalar_z_min = std::min(scalar_z_min, z_ext);
                    scalar_z_max = std::max(scalar_z_max, z_ext);
                    scalar_rank += scalar_rows;
                    scalar_alpha_energy += (bucket.p + bucket.n) / (z_ext * z_ext);
                }
                if (bucket.beta_rows > 0) {
                    beta_z_min = std::min(beta_z_min, z_ext);
                    beta_z_max = std::max(beta_z_max, z_ext);
                    beta_rank += bucket.beta_rows;
                    beta_expected_full_rows += 3 * (2 * l + 1);
                    beta_alpha_energy += bucket.beta / (z_ext * z_ext);
                }
            }
            if (scalar_rank == 0) {
                scalar_z_min = 0.0;
            }
            if (beta_rank == 0) {
                beta_z_min = 0.0;
            }
            const double z_min = std::min(
                scalar_rank > 0 ? scalar_z_min : std::numeric_limits<double>::infinity(),
                beta_rank > 0 ? beta_z_min : std::numeric_limits<double>::infinity());
            const double z_max = std::max(scalar_z_max, beta_z_max);
            const double trace_rcond =
                (z_max > 0.0 && z_min < std::numeric_limits<double>::infinity())
                    ? z_min / z_max
                    : 0.0;

            double full_after_linf = 0.0;
            const int full_rows = static_cast<int>(residual.get_nbr());
            for (int row_index = 0; row_index < full_rows; ++row_index) {
                double value = residual(row_index);
                if (row_index < static_cast<int>(trace_dtn_corrected.size()) &&
                    trace_dtn_corrected[static_cast<std::size_t>(row_index)]) {
                    value = 0.0;
                }
                full_after_linf = std::max(full_after_linf, std::abs(value));
            }

            std::cout << "=== dtn-trace PC probe "
                         "(XCTS_DTN_TRACE_PC_PROBE) ===\n";
            std::cout << "dtn-trace scaffold=1 np=" << nproc
                      << " ndom=" << ndom
                      << " n=" << column_count
                      << " init_error=" << initial_error << "\n";
            std::cout << "dtn-trace projection mode="
                      << "metadata-tau-match-row-order-l-proxy"
                      << " gamma_domain=" << gamma_domain
                      << " tau_match_rows=" << tau_match_rows
                      << " gamma_rows=" << gamma_trace_rows
                      << " unresolved_rows=" << unresolved_rows
                      << "\n";
            int census_lines = 0;
            for (const auto& entry : exterior_census) {
                if (census_lines >= 32) {
                    std::cout << "dtn-trace exterior_row_census truncated=1\n";
                    break;
                }
                std::cout << "dtn-trace exterior_row_census "
                          << entry.first
                          << " rows=" << entry.second.rows
                          << " energy=" << entry.second.energy
                          << "\n";
                ++census_lines;
            }
            print_dtn_trace_energy_line("P", DtnTraceChannel::P, buckets, tail);
            print_dtn_trace_energy_line("N", DtnTraceChannel::N, buckets, tail);
            print_dtn_trace_energy_line("beta", DtnTraceChannel::Beta, buckets, tail);
            print_dtn_trace_energy_line("other", DtnTraceChannel::Other, buckets, tail);
            std::cout << "dtn-trace saturation scalar="
                      << safe_fraction(scalar_energy, scalar_energy +
                                                        tail.p + tail.n)
                      << " beta="
                      << safe_fraction(beta_energy, beta_energy + tail.beta)
                      << " total="
                      << safe_fraction(captured_energy, selected_energy)
                      << " selected_energy=" << selected_energy
                      << " tail_over_Lmax=" << tail_energy
                      << " other_energy=" << other_energy
                      << "\n";
            std::cout << "dtn-trace Lmax=" << lmax
                      << " ntrace=" << ntrace
                      << " scalar_energy=" << scalar_energy
                      << " beta_energy=" << beta_energy
                      << " total_energy=" << captured_energy
                      << "\n";
            std::cout << "dtn-trace scalar_dtn R=" << safe_radius
                      << " corrected_rows=" << scalar_rank
                      << " z_min=" << scalar_z_min
                      << " z_max=" << scalar_z_max
                      << " alpha_energy=" << scalar_alpha_energy
                      << " scalar_before=" << scalar_before_energy
                      << " scalar_after=" << scalar_after_energy
                      << "\n";
            std::cout << "dtn-trace beta_component_dtn R=" << safe_radius
                      << " corrected_rows=" << beta_rank
                      << " z_min=" << beta_z_min
                      << " z_max=" << beta_z_max
                      << " alpha_energy=" << beta_alpha_energy
                      << " beta_before=" << beta_before_energy
                      << " beta_after=" << beta_after_energy
                      << "\n";
            std::cout << "dtn-trace beta_vector_block status=not_implemented"
                      << " componentwise_smoke=1"
                      << " proxy_rows=" << beta_rank
                      << " expected_full_vsh_rows=" << beta_expected_full_rows
                      << " row_coverage="
                      << safe_fraction(static_cast<double>(beta_rank),
                                       static_cast<double>(beta_expected_full_rows))
                      << "\n";
            std::cout << "dtn-trace rank=" << (scalar_rank + beta_rank)
                      << " rcond=" << trace_rcond << "\n";
            std::cout << "dtn-trace gamma_reduction scalar=" << scalar_reduction
                      << " beta=" << beta_reduction
                      << " total=" << total_reduction << "\n";
            std::cout << "dtn-trace full_reduction before=" << initial_error
                      << " after=" << full_after_linf << "\n";
            std::cout << "dtn-trace backend_gate status=closed"
                      << " smoother=disabled backend=not_wired"
                      << " reason=beta_vector_block_not_implemented"
                      << ",full_residual_unchanged"
                      << " production_mumps=preserved\n";
            std::cout << "dtn-trace gmres iters=0 rel=nan"
                      << " setup_s=" << elapsed_time(probe_start)
                      << " apply_s=0"
                      << " rss_mb=" << (peak_rss_gb() * 1024.0)
                      << "\n";
            std::cout << "dtn-trace scaffold: analytic DtN correction is not "
                         "wired as a solver backend; scalar/beta DtN is a "
                         "Gamma-only diagnostic model, beta is componentwise "
                         "only, and l buckets are trace-row order proxies.\n";
            std::cout.flush();
        }

        [[noreturn]] void collective_exit(int exit_code = 0)
        {
            MPI_Barrier(MPI_COMM_WORLD);
            MPI_Finalize();
            std::exit(exit_code);
        }

        struct SchurAggregateBlock {
            int id = -1;
            std::string label;
            std::vector<int> domains;
            std::vector<int> bulk_rows;
            std::vector<int> bulk_cols;
        };

        struct SchurAggregateShape {
            int id = -1;
            int bulk_rows = 0;
            int bulk_cols = 0;
            bool square = false;
        };

        struct SchurAggregateCensus {
            int total_rows = 0;
            int total_cols = 0;
            int bulk_rows = 0;
            int interface_rows = 0;
            int bulk_cols = 0;
            int interface_cols = 0;
            int unclassified_rows = 0;
            int unclassified_cols = 0;

            int rows_vol = 0;
            int rows_tau_bc = 0;
            int rows_tau_match = 0;
            int rows_global_int = 0;
            int rows_unknown = 0;

            int cols_field_unknown = 0;
            int cols_field_interior = 0;
            int cols_field_boundary = 0;
            int cols_field_interior_vol = 0;
            int cols_field_boundary_tau = 0;
            int cols_field_outer_shell_tau = 0;
            int cols_field_matching = 0;
            int cols_field_gauge = 0;
            int cols_var_domain = 0;
            int cols_scalar_global = 0;
            int cols_unknown = 0;

            int bulk_cols_field_interior_vol = 0;
            int bulk_cols_field_boundary_tau = 0;
            int bulk_cols_field_outer_shell_tau = 0;
            int bulk_cols_field_matching = 0;
            int bulk_cols_var_domain = 0;
            int bulk_cols_square_closure = 0;
        };

        struct SchurAggregatePartition {
            std::string source;
            std::vector<int> domain_to_aggregate;
            std::vector<int> row_role;       // 0=bulk, 1=interface, -1=unclassified.
            std::vector<int> row_aggregate;  // aggregate id for bulk rows; -1 otherwise.
            std::vector<int> col_role;       // 0=bulk, 1=interface, -1=unclassified.
            std::vector<int> col_aggregate;  // aggregate id for bulk cols; -1 otherwise.
            std::vector<int> interface_rows;
            std::vector<int> interface_cols;
            std::vector<SchurAggregateBlock> blocks;
            std::vector<SchurAggregateShape> shapes;
            bool all_square = false;
            SchurAggregateCensus census;
        };

        std::string trim_copy(const std::string& input)
        {
            std::size_t begin = 0;
            while (begin < input.size() &&
                   std::isspace(static_cast<unsigned char>(input[begin]))) {
                ++begin;
            }
            std::size_t end = input.size();
            while (end > begin &&
                   std::isspace(static_cast<unsigned char>(input[end - 1]))) {
                --end;
            }
            return input.substr(begin, end - begin);
        }

        std::vector<std::string> split_string(const std::string& input, char delimiter)
        {
            std::vector<std::string> out;
            std::stringstream ss(input);
            std::string item;
            while (std::getline(ss, item, delimiter)) {
                out.push_back(trim_copy(item));
            }
            return out;
        }

        int parse_domain_endpoint(const std::string& token, int last_domain)
        {
            const std::string text = trim_copy(token);
            if (text == "last" || text == "end" || text == "*") {
                return last_domain;
            }
            std::size_t parsed = 0;
            const int value = std::stoi(text, &parsed);
            if (parsed != text.size()) {
                throw std::runtime_error("trailing text in domain endpoint '" + text + "'");
            }
            return value;
        }

        std::string domains_label(const std::vector<int>& domains)
        {
            if (domains.empty()) {
                return "{}";
            }
            std::ostringstream out;
            out << "{";
            int range_begin = domains.front();
            int previous = domains.front();
            bool first_range_output = true;
            auto flush_range = [&](int begin, int end) {
                if (!first_range_output) {
                    out << ",";
                }
                first_range_output = false;
                if (begin == end) {
                    out << begin;
                } else {
                    out << begin << "-" << end;
                }
            };
            for (std::size_t i = 1; i < domains.size(); ++i) {
                const int current = domains[i];
                if (current == previous + 1) {
                    previous = current;
                    continue;
                }
                flush_range(range_begin, previous);
                range_begin = current;
                previous = current;
            }
            flush_range(range_begin, previous);
            out << "}";
            return out.str();
        }

        void add_aggregate_range(std::vector<SchurAggregateBlock>& blocks,
                                 std::vector<int>& domain_to_aggregate,
                                 int first_domain,
                                 int last_domain,
                                 const std::string& label)
        {
            if (first_domain > last_domain) {
                return;
            }
            if (first_domain < 0 ||
                last_domain >= static_cast<int>(domain_to_aggregate.size())) {
                throw std::runtime_error("aggregate range is outside the space domain range");
            }
            SchurAggregateBlock block;
            block.id = static_cast<int>(blocks.size());
            block.label = label;
            for (int domain = first_domain; domain <= last_domain; ++domain) {
                if (domain_to_aggregate[static_cast<std::size_t>(domain)] >= 0) {
                    std::ostringstream msg;
                    msg << "domain " << domain << " appears in more than one Schur aggregate";
                    throw std::runtime_error(msg.str());
                }
                domain_to_aggregate[static_cast<std::size_t>(domain)] = block.id;
                block.domains.push_back(domain);
            }
            blocks.push_back(std::move(block));
        }

        bool parse_manual_aggregates(const char* spec,
                                     int ndom,
                                     std::vector<SchurAggregateBlock>& blocks,
                                     std::vector<int>& domain_to_aggregate,
                                     std::string& source,
                                     std::string& error)
        {
            blocks.clear();
            domain_to_aggregate.assign(static_cast<std::size_t>(ndom), -1);
            source = "manual";
            error.clear();
            try {
                const std::vector<std::string> groups = split_string(spec, ';');
                for (const std::string& group : groups) {
                    if (group.empty()) {
                        continue;
                    }
                    SchurAggregateBlock block;
                    block.id = static_cast<int>(blocks.size());
                    block.label = "manual" + std::to_string(block.id);
                    const std::vector<std::string> terms = split_string(group, ',');
                    for (const std::string& term : terms) {
                        if (term.empty()) {
                            continue;
                        }
                        const std::size_t dash = term.find('-');
                        int first_domain = -1;
                        int last_domain = -1;
                        if (dash == std::string::npos) {
                            first_domain = parse_domain_endpoint(term, ndom - 1);
                            last_domain = first_domain;
                        } else {
                            first_domain =
                                parse_domain_endpoint(term.substr(0, dash), ndom - 1);
                            last_domain =
                                parse_domain_endpoint(term.substr(dash + 1), ndom - 1);
                        }
                        if (first_domain > last_domain) {
                            std::swap(first_domain, last_domain);
                        }
                        if (first_domain < 0 || last_domain >= ndom) {
                            throw std::runtime_error("manual aggregate domain is out of range");
                        }
                        for (int domain = first_domain; domain <= last_domain; ++domain) {
                            if (domain_to_aggregate[static_cast<std::size_t>(domain)] >= 0) {
                                std::ostringstream msg;
                                msg << "domain " << domain
                                    << " appears in more than one manual Schur aggregate";
                                throw std::runtime_error(msg.str());
                            }
                            domain_to_aggregate[static_cast<std::size_t>(domain)] = block.id;
                            block.domains.push_back(domain);
                        }
                    }
                    if (!block.domains.empty()) {
                        std::sort(block.domains.begin(), block.domains.end());
                        blocks.push_back(std::move(block));
                    }
                }
                for (int domain = 0; domain < ndom; ++domain) {
                    if (domain_to_aggregate[static_cast<std::size_t>(domain)] < 0) {
                        std::ostringstream msg;
                        msg << "manual aggregate spec did not cover domain " << domain;
                        throw std::runtime_error(msg.str());
                    }
                }
            } catch (const std::exception& ex) {
                error = ex.what();
                return false;
            }
            return !blocks.empty();
        }

        void add_singleton_aggregates(int ndom,
                                      std::vector<SchurAggregateBlock>& blocks,
                                      std::vector<int>& domain_to_aggregate)
        {
            blocks.clear();
            domain_to_aggregate.assign(static_cast<std::size_t>(ndom), -1);
            for (int domain = 0; domain < ndom; ++domain) {
                add_aggregate_range(blocks, domain_to_aggregate, domain, domain,
                                    "domain" + std::to_string(domain));
            }
        }

        bool build_bns_aggregate_layout(const Space_bin_ns& space,
                                        std::vector<SchurAggregateBlock>& blocks,
                                        std::vector<int>& domain_to_aggregate,
                                        std::string& source,
                                        std::string& error)
        {
            const int ndom = space.get_nbr_domains();
            blocks.clear();
            domain_to_aggregate.assign(static_cast<std::size_t>(ndom), -1);
            source = "Space_bin_ns";
            error.clear();
            try {
                add_aggregate_range(blocks, domain_to_aggregate,
                                    space.NS1, space.ADAPTED1 + 1, "star1");
                add_aggregate_range(blocks, domain_to_aggregate,
                                    space.NS2, space.ADAPTED2 + 1, "star2");
                const bool split_exterior_tail =
                    env_flag_enabled("SCHUR_SPLIT_EXTERIOR_TAIL", true);
                if (split_exterior_tail && space.OUTER + 5 < ndom) {
                    add_aggregate_range(blocks, domain_to_aggregate,
                                        space.OUTER, space.OUTER + 4, "bispheric");
                    add_aggregate_range(blocks, domain_to_aggregate,
                                        space.OUTER + 5, ndom - 1, "asymptotic_tail");
                    source += " split-exterior-semantics";
                } else {
                    add_aggregate_range(blocks, domain_to_aggregate,
                                        space.OUTER, ndom - 1, "exterior");
                    source += " full-exterior";
                }
                for (int domain = 0; domain < ndom; ++domain) {
                    if (domain_to_aggregate[static_cast<std::size_t>(domain)] < 0) {
                        std::ostringstream msg;
                        msg << "BNS aggregate layout did not cover domain " << domain;
                        throw std::runtime_error(msg.str());
                    }
                }
            } catch (const std::exception& ex) {
                error = ex.what();
                return false;
            }
            return true;
        }

        bool build_bns_aggregate_layout(const Space_bin_ns_nosym& space,
                                        std::vector<SchurAggregateBlock>& blocks,
                                        std::vector<int>& domain_to_aggregate,
                                        std::string& source,
                                        std::string& error)
        {
            const int ndom = space.get_nbr_domains();
            blocks.clear();
            domain_to_aggregate.assign(static_cast<std::size_t>(ndom), -1);
            source = "Space_bin_ns_nosym";
            error.clear();
            try {
                add_aggregate_range(blocks, domain_to_aggregate,
                                    space.NS1, space.ADAPTED1 + 1, "star1");
                add_aggregate_range(blocks, domain_to_aggregate,
                                    space.NS2, space.ADAPTED2 + 1, "star2");
                const bool split_exterior_tail =
                    env_flag_enabled("SCHUR_SPLIT_EXTERIOR_TAIL", true);
                if (split_exterior_tail && space.OUTER + 5 < ndom) {
                    add_aggregate_range(blocks, domain_to_aggregate,
                                        space.OUTER, space.OUTER + 4, "bispheric");
                    add_aggregate_range(blocks, domain_to_aggregate,
                                        space.OUTER + 5, ndom - 1, "asymptotic_tail");
                    source += " split-exterior-semantics";
                } else {
                    add_aggregate_range(blocks, domain_to_aggregate,
                                        space.OUTER, ndom - 1, "exterior");
                    source += " full-exterior";
                }
                for (int domain = 0; domain < ndom; ++domain) {
                    if (domain_to_aggregate[static_cast<std::size_t>(domain)] < 0) {
                        std::ostringstream msg;
                        msg << "BNS-nosym aggregate layout did not cover domain " << domain;
                        throw std::runtime_error(msg.str());
                    }
                }
            } catch (const std::exception& ex) {
                error = ex.what();
                return false;
            }
            return true;
        }

        void tally_row_taxonomy(RowTaxonomy taxonomy, SchurAggregateCensus& census)
        {
            switch (taxonomy) {
                case RowTaxonomy::Vol: ++census.rows_vol; break;
                case RowTaxonomy::TauBc: ++census.rows_tau_bc; break;
                case RowTaxonomy::TauMatch: ++census.rows_tau_match; break;
                case RowTaxonomy::GlobalInt: ++census.rows_global_int; break;
                case RowTaxonomy::Unknown: ++census.rows_unknown; break;
            }
        }

        void tally_column_class(ColumnClass column_class, SchurAggregateCensus& census)
        {
            switch (column_class) {
                case ColumnClass::FieldUnknown: ++census.cols_field_unknown; break;
                case ColumnClass::FieldInterior: ++census.cols_field_interior; break;
                case ColumnClass::FieldBoundary: ++census.cols_field_boundary; break;
                case ColumnClass::FieldInteriorVol: ++census.cols_field_interior_vol; break;
                case ColumnClass::FieldBoundaryTau: ++census.cols_field_boundary_tau; break;
                case ColumnClass::FieldOuterShellTau: ++census.cols_field_outer_shell_tau; break;
                case ColumnClass::FieldMatching: ++census.cols_field_matching; break;
                case ColumnClass::FieldGauge: ++census.cols_field_gauge; break;
                case ColumnClass::VarDomain: ++census.cols_var_domain; break;
                case ColumnClass::ScalarGlobal: ++census.cols_scalar_global; break;
                case ColumnClass::Unknown: ++census.cols_unknown; break;
            }
        }

        void tally_bulk_column_class(ColumnClass column_class, SchurAggregateCensus& census)
        {
            switch (column_class) {
                case ColumnClass::FieldInteriorVol:
                    ++census.bulk_cols_field_interior_vol;
                    break;
                case ColumnClass::FieldBoundaryTau:
                    ++census.bulk_cols_field_boundary_tau;
                    break;
                case ColumnClass::FieldOuterShellTau:
                    ++census.bulk_cols_field_outer_shell_tau;
                    break;
                case ColumnClass::FieldMatching:
                    ++census.bulk_cols_field_matching;
                    break;
                case ColumnClass::VarDomain:
                    ++census.bulk_cols_var_domain;
                    break;
                default:
                    break;
            }
        }

        int aggregate_for_domain(const std::vector<int>& domain_to_aggregate, int domain)
        {
            if (domain < 0 || domain >= static_cast<int>(domain_to_aggregate.size())) {
                return -1;
            }
            return domain_to_aggregate[static_cast<std::size_t>(domain)];
        }

        void build_aggregate_partition(const TaggedJacobianMetadata& metadata,
                                       const std::vector<SchurAggregateBlock>& layout_blocks,
                                       const std::vector<int>& domain_to_aggregate,
                                       const std::string& source,
                                       SchurAggregatePartition& out)
        {
            out = SchurAggregatePartition{};
            out.source = source;
            out.domain_to_aggregate = domain_to_aggregate;
            out.blocks = layout_blocks;
            out.census.total_rows = metadata.nrows;
            out.census.total_cols = metadata.ncols;
            out.row_role.assign(static_cast<std::size_t>(metadata.nrows), -1);
            out.row_aggregate.assign(static_cast<std::size_t>(metadata.nrows), -1);
            out.col_role.assign(static_cast<std::size_t>(metadata.ncols), -1);
            out.col_aggregate.assign(static_cast<std::size_t>(metadata.ncols), -1);
            std::vector<std::vector<std::pair<int, int>>> square_closure_candidates(
                out.blocks.size());

            for (int row = 0; row < metadata.nrows; ++row) {
                const RowMetadata& row_meta = metadata.rows[static_cast<std::size_t>(row)];
                tally_row_taxonomy(row_meta.taxonomy, out.census);
                const int row_aggregate =
                    aggregate_for_domain(domain_to_aggregate, row_meta.dom);
                switch (row_meta.taxonomy) {
                    case RowTaxonomy::Vol:
                    case RowTaxonomy::TauBc: {
                        if (row_aggregate < 0) {
                            ++out.census.unclassified_rows;
                            break;
                        }
                        out.row_role[static_cast<std::size_t>(row)] = 0;
                        out.row_aggregate[static_cast<std::size_t>(row)] = row_aggregate;
                        out.blocks[static_cast<std::size_t>(row_aggregate)].bulk_rows.push_back(row);
                        ++out.census.bulk_rows;
                        break;
                    }
                    case RowTaxonomy::TauMatch: {
                        if (row_aggregate < 0) {
                            ++out.census.unclassified_rows;
                            break;
                        }
                        const int other_aggregate =
                            aggregate_for_domain(domain_to_aggregate, row_meta.dom_pair);
                        if (row_meta.dom_pair >= 0 && other_aggregate == row_aggregate) {
                            out.row_role[static_cast<std::size_t>(row)] = 0;
                            out.row_aggregate[static_cast<std::size_t>(row)] = row_aggregate;
                            out.blocks[static_cast<std::size_t>(row_aggregate)]
                                .bulk_rows.push_back(row);
                            ++out.census.bulk_rows;
                        } else {
                            out.row_role[static_cast<std::size_t>(row)] = 1;
                            out.interface_rows.push_back(row);
                            ++out.census.interface_rows;
                        }
                        break;
                    }
                    case RowTaxonomy::GlobalInt: {
                        out.row_role[static_cast<std::size_t>(row)] = 1;
                        out.interface_rows.push_back(row);
                        ++out.census.interface_rows;
                        break;
                    }
                    case RowTaxonomy::Unknown: {
                        ++out.census.unclassified_rows;
                        break;
                    }
                }
            }

            for (int col = 0; col < metadata.ncols; ++col) {
                const ColumnMetadata& col_meta = metadata.columns[static_cast<std::size_t>(col)];
                tally_column_class(col_meta.column_class, out.census);
                const int col_aggregate =
                    aggregate_for_domain(domain_to_aggregate, col_meta.domain);
                bool use_bulk = false;
                bool unclassified = false;
                int closure_priority = -1;

                if (col_meta.column_class == ColumnClass::FieldGauge ||
                    col_meta.column_class == ColumnClass::ScalarGlobal ||
                    col_meta.column_class == ColumnClass::Unknown) {
                    use_bulk = false;
                } else if (col_aggregate < 0) {
                    use_bulk = false;
                } else {
                    bool has_same_aggregate_vol = false;
                    bool has_same_aggregate_tau_bc = false;
                    bool has_same_aggregate_tau_match = false;
                    bool has_same_aggregate_bulk_row = false;
                    bool has_any_tau_match = false;
                    bool has_nonlocal_tau_match = false;
                    bool has_cross_aggregate_bulk_row = false;
                    if (static_cast<int>(metadata.rows_per_column.size()) == metadata.ncols) {
                        for (int row : metadata.rows_per_column[static_cast<std::size_t>(col)]) {
                            if (row < 0 || row >= metadata.nrows) {
                                has_cross_aggregate_bulk_row = true;
                                break;
                            }

                            const RowMetadata& row_meta =
                                metadata.rows[static_cast<std::size_t>(row)];
                            if (row_meta.taxonomy == RowTaxonomy::TauMatch) {
                                has_any_tau_match = true;
                            }
                            if (out.row_role[static_cast<std::size_t>(row)] != 0) {
                                if (row_meta.taxonomy == RowTaxonomy::TauMatch) {
                                    has_nonlocal_tau_match = true;
                                }
                                continue;
                            }

                            const int support_aggregate =
                                out.row_aggregate[static_cast<std::size_t>(row)];
                            if (support_aggregate != col_aggregate) {
                                if (row_meta.taxonomy == RowTaxonomy::TauMatch) {
                                    has_nonlocal_tau_match = true;
                                }
                                has_cross_aggregate_bulk_row = true;
                                break;
                            }

                            has_same_aggregate_bulk_row = true;
                            switch (row_meta.taxonomy) {
                                case RowTaxonomy::Vol:
                                    has_same_aggregate_vol = true;
                                    break;
                                case RowTaxonomy::TauBc:
                                    has_same_aggregate_tau_bc = true;
                                    break;
                                case RowTaxonomy::TauMatch:
                                    has_same_aggregate_tau_match = true;
                                    break;
                                default:
                                    break;
                            }
                        }
                    }

                    switch (col_meta.column_class) {
                        case ColumnClass::FieldInteriorVol:
                        case ColumnClass::FieldInterior:
                        case ColumnClass::FieldBoundary:
                        case ColumnClass::FieldUnknown:
                            use_bulk = has_same_aggregate_vol &&
                                       !has_cross_aggregate_bulk_row;
                            break;
                        case ColumnClass::FieldBoundaryTau:
                        case ColumnClass::FieldOuterShellTau:
                            use_bulk = has_same_aggregate_tau_bc && !has_any_tau_match &&
                                       !has_cross_aggregate_bulk_row;
                            break;
                        case ColumnClass::FieldMatching:
                            use_bulk = has_same_aggregate_tau_match &&
                                       !has_nonlocal_tau_match &&
                                       !has_cross_aggregate_bulk_row;
                            break;
                        case ColumnClass::VarDomain:
                            use_bulk = false;
                            break;
                        default:
                            use_bulk = false;
                            break;
                    }

                    if (!use_bulk && !has_cross_aggregate_bulk_row) {
                        switch (col_meta.column_class) {
                            case ColumnClass::FieldMatching:
                                if (has_same_aggregate_tau_match) {
                                    closure_priority = 10;
                                }
                                break;
                            case ColumnClass::FieldBoundaryTau:
                            case ColumnClass::FieldOuterShellTau:
                                if (has_same_aggregate_tau_bc) {
                                    closure_priority = 20;
                                }
                                break;
                            case ColumnClass::VarDomain:
                                if (has_same_aggregate_bulk_row) {
                                    closure_priority = 30;
                                }
                                break;
                            case ColumnClass::FieldBoundary:
                            case ColumnClass::FieldInterior:
                            case ColumnClass::FieldUnknown:
                                if (has_same_aggregate_bulk_row) {
                                    closure_priority = 40;
                                }
                                break;
                            default:
                                break;
                        }
                    }
                }

                if (unclassified) {
                    ++out.census.unclassified_cols;
                    continue;
                }
                if (use_bulk) {
                    out.col_role[static_cast<std::size_t>(col)] = 0;
                    out.col_aggregate[static_cast<std::size_t>(col)] = col_aggregate;
                    out.blocks[static_cast<std::size_t>(col_aggregate)].bulk_cols.push_back(col);
                    tally_bulk_column_class(col_meta.column_class, out.census);
                    ++out.census.bulk_cols;
                } else {
                    out.col_role[static_cast<std::size_t>(col)] = 1;
                    if (closure_priority >= 0 && col_aggregate >= 0) {
                        square_closure_candidates[static_cast<std::size_t>(col_aggregate)]
                            .push_back({closure_priority, col});
                    }
                }
            }

            for (int aggregate = 0; aggregate < static_cast<int>(out.blocks.size()); ++aggregate) {
                SchurAggregateBlock& block = out.blocks[static_cast<std::size_t>(aggregate)];
                int deficit = static_cast<int>(block.bulk_rows.size()) -
                              static_cast<int>(block.bulk_cols.size());
                if (deficit <= 0) {
                    continue;
                }

                auto& candidates =
                    square_closure_candidates[static_cast<std::size_t>(aggregate)];
                std::sort(candidates.begin(), candidates.end());
                for (const auto& candidate : candidates) {
                    if (deficit <= 0) {
                        break;
                    }
                    const int col = candidate.second;
                    if (out.col_role[static_cast<std::size_t>(col)] != 1) {
                        continue;
                    }
                    const ColumnMetadata& col_meta =
                        metadata.columns[static_cast<std::size_t>(col)];
                    out.col_role[static_cast<std::size_t>(col)] = 0;
                    out.col_aggregate[static_cast<std::size_t>(col)] = aggregate;
                    block.bulk_cols.push_back(col);
                    tally_bulk_column_class(col_meta.column_class, out.census);
                    ++out.census.bulk_cols;
                    ++out.census.bulk_cols_square_closure;
                    --deficit;
                }
            }

            for (int col = 0; col < metadata.ncols; ++col) {
                if (out.col_role[static_cast<std::size_t>(col)] == 1) {
                    out.interface_cols.push_back(col);
                    ++out.census.interface_cols;
                }
            }

            std::sort(out.interface_rows.begin(), out.interface_rows.end());
            std::sort(out.interface_cols.begin(), out.interface_cols.end());
            for (SchurAggregateBlock& block : out.blocks) {
                std::sort(block.bulk_rows.begin(), block.bulk_rows.end());
                std::sort(block.bulk_cols.begin(), block.bulk_cols.end());
            }

            out.all_square = !out.blocks.empty();
            out.shapes.reserve(out.blocks.size());
            for (const SchurAggregateBlock& block : out.blocks) {
                SchurAggregateShape shape;
                shape.id = block.id;
                shape.bulk_rows = static_cast<int>(block.bulk_rows.size());
                shape.bulk_cols = static_cast<int>(block.bulk_cols.size());
                shape.square = (shape.bulk_rows == shape.bulk_cols);
                out.shapes.push_back(shape);
                if (!shape.square) {
                    out.all_square = false;
                }
            }
        }
    } // namespace

#ifdef CELEPHAIS_USE_MUMPS
    // LAPACK/BLAS (Fortran ABI; the build already links these). Read-only
    // dimension args still take int* / double* by the Fortran convention.
    extern "C"
    {
    void dgetrf_(int* m, int* n, double* a, int* lda, int* ipiv, int* info);
    void dgetrs_(char* trans, int* n, int* nrhs, double* a, int* lda, int* ipiv,
                 double* b, int* ldb, int* info);
    void dgemm_(char* transa, char* transb, int* m, int* n, int* k, double* alpha,
                double* a, int* lda, double* b, int* ldb, double* beta,
                double* c, int* ldc);
    void dgesdd_(char* jobz, int* m, int* n, double* a, int* lda, double* s,
                 double* u, int* ldu, double* vt, int* ldvt, double* work,
                 int* lwork, int* iwork, int* info);
    void dgeev_(char* jobvl, char* jobvr, int* n, double* a, int* lda,
                double* wr, double* wi, double* vl, int* ldvl, double* vr,
                int* ldvr, double* work, int* lwork, int* info);
    void dgeqrf_(int* m, int* n, double* a, int* lda, double* tau, double* work,
                 int* lwork, int* info);
    void dormqr_(char* side, char* trans, int* m, int* n, int* k, double* a,
                 int* lda, double* tau, double* c, int* ldc, double* work,
                 int* lwork, int* info);
    void dtrtrs_(char* uplo, char* trans, char* diag, int* n, int* nrhs,
                 double* a, int* lda, double* b, int* ldb, int* info);
    void dgecon_(char* norm, int* n, double* a, int* lda, double* anorm,
                 double* rcond, double* work, int* iwork, int* info);
    }

#endif // CELEPHAIS_USE_MUMPS

    bool System_of_eqs::do_newton_jfnk_schur(double precision,
                                             double& error,
                                             const SolverRuntimeConfig& config)
    {
        set_solver_runtime_config(config);
#ifdef CELEPHAIS_USE_MUMPS
        int rank = 0;
        int nproc = 1;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &nproc);

        Array<double> residual(sec_member());
        error = infinity_norm(residual);
        if (rank == 0) {
            std::cout << "do_newton_jfnk_schur: aggregate-Schur pre-backend gate"
                      << " (np=" << nproc << ", init_error=" << error << ")" << std::endl;
        }
        // Diagnostic probes that analyse one binary Jacobian then exit the
        // process: the frozen-neighbour rank probe and the Schwarz-PC GMRES
        // harness. Both must fire on the binary system and tolerate a converged
        // fixture (their object is structural, not residual-dependent).
        const char* pcoarse_dump_dir = std::getenv("PCOARSE_DUMP");
        const char* pcoarse_probe_dir = std::getenv("PCOARSE_PC_PROBE");
        const bool pcoarse_dump_on =
            pcoarse_dump_dir != nullptr && pcoarse_dump_dir[0] != '\0';
        const bool pcoarse_probe_on =
            pcoarse_probe_dir != nullptr && pcoarse_probe_dir[0] != '\0';
        const bool probe_forces =
            env_flag_enabled("SCHUR_PROBE_DIRICHLET", false) ||
            env_flag_enabled("SCHUR_PC_GMRES", false) ||
            env_flag_enabled("XCTS_DTN_TRACE_PC_PROBE", false) ||
            env_flag_enabled("BORDER_ANALYSIS", false) ||
            pcoarse_dump_on || pcoarse_probe_on;
        // Restrict to the binary bispheric space so the probe fires on the binary
        // Jacobian, not the single-star seed/boost stages that run first; let
        // those continue.
        if (probe_forces &&
            dynamic_cast<const Space_bin_ns*>(&espace) == nullptr &&
            dynamic_cast<const Space_bin_ns_nosym*>(&espace) == nullptr) {
            if (rank == 0)
                std::cout << "do_newton_jfnk_schur: non-binary space; skipping "
                             "diagnostic probe, continuing solve.\n";
            return true;
        }
        // p-coarse fixture alignment. The regrid workflow runs preliminary
        // binary Newton solves (hydro rescaling, and a coarse force-balance solve
        // at the loaded resolution) BEFORE the target fine force-balance solve.
        // The probe must fire ONLY on the force-balance system -- uniquely marked
        // by the Euler first-integral eq_int ("D_i H") -- AND, for the probe run,
        // only at the FINE rung (strictly more unknowns than the dumped coarse
        // system). Every other binary Newton solve is delegated to the real
        // JFNK-MUMPS backend so the stage actually converges and the workflow
        // advances (returning a fake "converged" would spin run_newton_loop).
        if (pcoarse_dump_on || pcoarse_probe_on) {
            bool fire = (eq_int_index_of("D_i H") >= 0);
            if (fire && pcoarse_probe_on) {
                const long long coarse_n =
                    pcoarse_read_coarse_n(pcoarse_probe_dir);
                if (coarse_n > 0 && nbr_unknowns <= static_cast<int>(coarse_n))
                    fire = false; // coarse-rung force-balance solve; keep laddering
            }
            if (!fire) {
                if (rank == 0)
                    std::cout << "do_newton_jfnk_schur: pcoarse delegating stage to "
                                 "jfnk-mumps (n="
                              << nbr_unknowns << ", euler_fi="
                              << (eq_int_index_of("D_i H") >= 0)
                              << "); advancing workflow.\n";
                return do_newton_jfnk_mumps(precision, error, config);
            }
        }
        // A converged binary is still a valid fixture, so the probes force
        // through the converged early-exit.
        if (error < precision && !probe_forces) {
            if (rank == 0)
                std::cerr << "do_newton_jfnk_schur: already converged; nothing to probe.\n";
            return true;
        }

        const int column_count = nbr_unknowns;
        if (static_cast<int>(residual.get_nbr()) != column_count) {
            if (rank == 0)
                std::cerr << "do_newton_jfnk_schur: non-square system m="
                          << residual.get_nbr() << " n=" << column_count
                          << "; cannot probe." << std::endl;
            return false;
        }

        // ====================================================================
        // Phase 1: PARTITION CONTRACT GATE (structural; no COO assembly).
        // ====================================================================
        const auto partition_start = std::chrono::system_clock::now();
        TaggedJacobianMetadata metadata;
        build_tagged_jacobian_metadata(metadata, /*include_row_incidence=*/true);
        if (env_flag_enabled("XCTS_DTN_TRACE_PC_PROBE", false)) {
            const int dtn_ndom = espace.get_nbr_domains();
            // Compactified domains do not expose get_rmax() without noisy
            // diagnostics. The res9 BNS probe fixture has rext=70; override for
            // other fixtures with XCTS_DTN_TRACE_PC_RADIUS.
            const double gamma_radius = env_double_value(
                "XCTS_DTN_TRACE_PC_RADIUS", 70.0);
            run_dtn_trace_probe(rank, nproc, dtn_ndom, column_count, error,
                                gamma_radius, residual, metadata);
            collective_exit(0);
        }

        // ====================================================================
        // Arm 17: bordered-MUMPS ANALYSIS probe (BORDER_ANALYSIS=1).
        // Measures whether pulling the small dense global border (the ~204
        // global-scalar columns col_dom==-1 plus the global/integral constraint
        // rows) out of the sparse elliptic bulk makes the MUMPS factor of the
        // remaining A_FF superlinearly cheaper -- the res19/21 fill lever. Pure
        // analysis economics: NO GMRES, NO preconditioner. Runs raw dmumps_c
        // (MumpsLinearSolver exposes only INFOG(16) of the analysis estimates,
        // not RINFOG(1)/INFOG(3,4,5,7,17)/INFOG(13)). Self-contained, rank-0,
        // self-exiting (mirrors the DTN / pcoarse probe idiom above). np=1.
        // ====================================================================
        if (env_flag_enabled("BORDER_ANALYSIS", false)) {
            // Fire only on the force-balance system (unique Euler first integral
            // "D_i H") so the census matches the documented fixture (204 global
            // cols, n=91866); advance any preliminary binary stage to jfnk-mumps.
            if (eq_int_index_of("D_i H") < 0) {
                if (rank == 0)
                    std::cout << "do_newton_jfnk_schur: BORDER_ANALYSIS "
                                 "delegating non-force-balance binary stage to "
                                 "jfnk-mumps (n="
                              << column_count << "); advancing workflow.\n";
                return do_newton_jfnk_mumps(precision, error, config);
            }

            const int mumps_ordering = config.mumps.ordering;
            const bool mumps_verbose =
                env_flag_enabled("MUMPS_NATIVE_VERBOSE", false);
            const bool do_factor =
                !env_flag_enabled("BORDER_ANALYSIS_NOFACTOR", false);
            const double configured_drop_tol =
                (config.mumps.drop_tol > 0.0) ? config.mumps.drop_tol : 1e-8;
            const double drop_tol = std::max(
                1e-16,
                configured_drop_tol * std::sqrt(std::sqrt(std::max(error, 0.0))));

            const auto border_coo_start = std::chrono::system_clock::now();
            JacobianAssembler border_assembler(*this, MPI_COMM_WORLD);
            AssembledJacobianCoo coo = border_assembler.assemble(drop_tol); // collective
            const double border_coo_seconds = elapsed_time(border_coo_start);
            if (rank != 0) {
                MPI_Finalize();
                std::exit(0);
            }

            const int n_mat = coo.n;
            const long long nnz_full = coo.nnz;
            std::cout << "\n================ ARM 17: BORDERED-MUMPS ANALYSIS PROBE"
                         " ================\n";
            std::cout << "fixture: n=" << n_mat << " nnz=" << nnz_full
                      << " drop_tol=" << drop_tol
                      << " coo_assemble_s=" << border_coo_seconds
                      << " ICNTL(7)ordering=" << mumps_ordering
                      << " do_factor=" << (do_factor ? 1 : 0) << "\n";
            if (nproc != 1) {
                std::cout << "  WARNING: nproc=" << nproc
                          << " (probe designed for np=1; MUMPS runs on MPI_COMM_SELF"
                             " so results stay rank-local but re-run at np=1 for a"
                             " production-matched partition).\n";
            }

            // ---- (1) CENSUS -----------------------------------------------
            std::vector<char> is_global_col(static_cast<std::size_t>(n_mat), 0);
            std::vector<char> is_globalint_row(static_cast<std::size_t>(n_mat), 0);
            long long n_global_by_dom = 0, n_global_by_class = 0;
            long long tax_unknown = 0, tax_vol = 0, tax_taubc = 0,
                      tax_taumatch = 0, tax_globalint = 0;
            for (int c = 0; c < n_mat; ++c) {
                const ColumnMetadata& cm =
                    metadata.columns[static_cast<std::size_t>(c)];
                if (cm.domain == -1) {
                    is_global_col[static_cast<std::size_t>(c)] = 1;
                    ++n_global_by_dom;
                }
                if (cm.column_class == ColumnClass::ScalarGlobal)
                    ++n_global_by_class;
            }
            for (int r = 0; r < n_mat; ++r) {
                switch (metadata.rows[static_cast<std::size_t>(r)].taxonomy) {
                    case RowTaxonomy::Unknown: ++tax_unknown; break;
                    case RowTaxonomy::Vol: ++tax_vol; break;
                    case RowTaxonomy::TauBc: ++tax_taubc; break;
                    case RowTaxonomy::TauMatch: ++tax_taumatch; break;
                    case RowTaxonomy::GlobalInt:
                        ++tax_globalint;
                        is_globalint_row[static_cast<std::size_t>(r)] = 1;
                        break;
                }
            }
            long long nnz_in_global_cols = 0;    // entries whose col is global
            long long nnz_in_globalcol_rows = 0; // entries whose row is at a global-col index
            long long nnz_in_globalint_rows = 0; // entries whose row is GlobalInt
            for (long long k = 0; k < nnz_full; ++k) {
                const int r = coo.irn[static_cast<std::size_t>(k)] - 1;
                const int c = coo.jcn[static_cast<std::size_t>(k)] - 1;
                if (is_global_col[static_cast<std::size_t>(c)]) ++nnz_in_global_cols;
                if (is_global_col[static_cast<std::size_t>(r)]) ++nnz_in_globalcol_rows;
                if (is_globalint_row[static_cast<std::size_t>(r)])
                    ++nnz_in_globalint_rows;
            }
            std::cout << "\n[CENSUS]\n";
            std::cout << "  global scalar COLS (col_dom==-1) = " << n_global_by_dom
                      << "   (by ColumnClass::ScalarGlobal = " << n_global_by_class
                      << ")\n";
            std::cout << "    nnz in global cols             = " << nnz_in_global_cols
                      << "\n";
            std::cout << "  row taxonomy inventory: Vol=" << tax_vol
                      << " TauBc=" << tax_taubc << " TauMatch=" << tax_taumatch
                      << " GlobalInt=" << tax_globalint << " Unknown=" << tax_unknown
                      << "\n";
            std::cout << "    nnz in GlobalInt rows          = "
                      << nnz_in_globalint_rows << "   (|GlobalInt rows|="
                      << tax_globalint << ")\n";
            std::cout << "  nnz in rows @ global-col INDICES (index-symmetric) = "
                      << nnz_in_globalcol_rows
                      << "   (prior fixture measurement: 18193)\n";
            std::cout.flush();

            // ---- raw dmumps_c analysis(+factor+solve) helper --------------
            struct BorderMumpsStats {
                std::string tag;
                bool analysis_ok = false;
                double analyze_seconds = 0.0;
                double est_flops = 0.0;       // RINFOG(1)
                long long est_real = 0;       // INFOG(3)
                long long est_int = 0;        // INFOG(4)
                int max_front = 0;            // INFOG(5)
                int ordering_used = 0;        // INFOG(7)
                int est_ram_mb = 0;           // INFOG(16) per-rank max
                int est_ram_total_mb = 0;     // INFOG(17) total
                bool factor_done = false;
                bool factor_ok = false;
                double factor_seconds = 0.0;
                int infog11 = 0;              // INFOG(11)
                int delayed_pivots = 0;       // INFOG(13)
                int factor_mem_mb = 0;        // INFOG(21) effective, max rank
                int final_icntl14 = 0;
                int factor_retries = 0;
                bool solve_done = false;
                bool solve_ok = false;
                double backward_error = -1.0; // ||A x - b||_inf / ||b||_inf
                int mumps_info1 = 0;          // last INFOG(1) on failure
            };

            auto run_mumps = [&](const char* tag, int n_sq,
                                 std::span<const int> irn,
                                 std::span<const int> jcn,
                                 std::span<const double> aval, bool factor,
                                 bool solve) -> BorderMumpsStats {
                const long long nnz = static_cast<long long>(aval.size());
                BorderMumpsStats st;
                st.tag = tag;
                DMUMPS_STRUC_C id;
                std::memset(&id, 0, sizeof(id));
                id.comm_fortran = MPI_Comm_c2f(MPI_COMM_SELF);
                id.par = 1;
                id.sym = 0;
                id.job = -1;
                dmumps_c(&id); // JOB_INIT
                id.icntl[0] = mumps_verbose ? 6 : -1;  // error unit
                id.icntl[1] = mumps_verbose ? 6 : -1;  // diag/warning unit
                id.icntl[2] = mumps_verbose ? 6 : -1;  // global info unit
                id.icntl[3] = mumps_verbose ? 2 : 0;   // verbosity
                id.icntl[6] = mumps_ordering;          // ICNTL(7): ordering (production)
                id.icntl[17] = 0;                      // ICNTL(18)=0: centralized
                int icntl14 = 200;
                id.icntl[13] = icntl14;                // ICNTL(14): workspace relax %
                id.n = n_sq;
                const long long nz_i32_max =
                    static_cast<long long>(std::numeric_limits<decltype(id.nz)>::max());
                id.nnz = static_cast<decltype(id.nnz)>(nnz);
                id.nz = (nnz <= nz_i32_max) ? static_cast<decltype(id.nz)>(nnz) : 0;
                id.irn = const_cast<int*>(irn.data());
                id.jcn = const_cast<int*>(jcn.data());

                const auto ta = std::chrono::system_clock::now();
                id.job = 1;
                dmumps_c(&id); // ANALYZE
                st.analyze_seconds = elapsed_time(ta);
                st.mumps_info1 = id.infog[0];
                if (id.infog[0] < 0) {
                    id.job = -2;
                    dmumps_c(&id);
                    return st;
                }
                st.analysis_ok = true;
                st.est_flops = id.rinfog[0];        // RINFOG(1)
                st.est_real = id.infog[2];          // INFOG(3)
                st.est_int = id.infog[3];           // INFOG(4)
                st.max_front = id.infog[4];         // INFOG(5)
                st.ordering_used = id.infog[6];     // INFOG(7)
                st.est_ram_mb = id.infog[15];       // INFOG(16)
                st.est_ram_total_mb = id.infog[16]; // INFOG(17)

                if (factor) {
                    st.factor_done = true;
                    id.a = const_cast<double*>(aval.data());
                    const auto tf = std::chrono::system_clock::now();
                    int attempt = 0;
                    for (;;) {
                        id.job = 2;
                        dmumps_c(&id); // FACTORIZE
                        const int e = id.infog[0];
                        if (e >= 0) {
                            st.factor_ok = true;
                            break;
                        }
                        // workspace-relaxable failures: grow ICNTL(14) and retry
                        if ((e == -9 || e == -8 || e == -19 || e == -20) &&
                            attempt < 8) {
                            icntl14 = static_cast<int>(icntl14 * 1.5) + 1;
                            id.icntl[13] = icntl14;
                            ++attempt;
                            continue;
                        }
                        st.factor_ok = false;
                        st.mumps_info1 = e;
                        break;
                    }
                    st.factor_seconds = elapsed_time(tf);
                    st.final_icntl14 = icntl14;
                    st.factor_retries = attempt;
                    if (st.factor_ok) {
                        st.infog11 = id.infog[10];       // INFOG(11)
                        st.delayed_pivots = id.infog[12]; // INFOG(13)
                        st.factor_mem_mb = id.infog[20];  // INFOG(21)
                    }
                }

                if (solve && st.factor_ok) {
                    st.solve_done = true;
                    std::vector<double> b(static_cast<std::size_t>(n_sq));
                    std::vector<double> x(static_cast<std::size_t>(n_sq));
                    std::mt19937_64 gen(12345ULL);
                    std::uniform_real_distribution<double> dist(-1.0, 1.0);
                    for (int i = 0; i < n_sq; ++i) {
                        b[static_cast<std::size_t>(i)] = dist(gen);
                    }
                    x = b; // rhs is overwritten in place with the solution
                    id.nrhs = 1;
                    id.lrhs = n_sq;
                    id.rhs = x.data();
                    id.job = 3;
                    dmumps_c(&id); // SOLVE
                    if (id.infog[0] < 0) {
                        st.solve_ok = false;
                        st.mumps_info1 = id.infog[0];
                    } else {
                        st.solve_ok = true;
                        std::vector<double> res(static_cast<std::size_t>(n_sq), 0.0);
                        for (long long k = 0; k < nnz; ++k) {
                            res[static_cast<std::size_t>(irn[k] - 1)] +=
                                aval[k] * x[static_cast<std::size_t>(jcn[k] - 1)];
                        }
                        double rn = 0.0, bn = 0.0;
                        for (int i = 0; i < n_sq; ++i) {
                            const double d = res[static_cast<std::size_t>(i)] -
                                             b[static_cast<std::size_t>(i)];
                            rn = std::max(rn, std::fabs(d));
                            bn = std::max(bn, std::fabs(b[static_cast<std::size_t>(i)]));
                        }
                        st.backward_error = (bn > 0.0) ? rn / bn : rn;
                    }
                }

                id.job = -2;
                dmumps_c(&id); // JOB_END
                return st;
            };

            auto print_stats = [&](const BorderMumpsStats& s) {
                std::cout << "  [" << s.tag << "]\n";
                if (!s.analysis_ok) {
                    std::cout << "    ANALYSIS FAILED INFOG(1)=" << s.mumps_info1
                              << "\n";
                    return;
                }
                std::cout << "    analyze_s=" << s.analyze_seconds
                          << "  ordering_used(INFOG7)=" << s.ordering_used << "\n";
                std::cout << "    RINFOG(1) est flops        = " << s.est_flops
                          << "\n";
                std::cout << "    INFOG(3)  est real space   = " << s.est_real
                          << "\n";
                std::cout << "    INFOG(4)  est int space    = " << s.est_int
                          << "\n";
                std::cout << "    INFOG(5)  est max front    = " << s.max_front
                          << "\n";
                std::cout << "    INFOG(16) est RAM MB/rank  = " << s.est_ram_mb
                          << "\n";
                std::cout << "    INFOG(17) est RAM MB total = "
                          << s.est_ram_total_mb << "\n";
                if (s.factor_done) {
                    if (s.factor_ok) {
                        std::cout << "    FACTOR ok factor_s=" << s.factor_seconds
                                  << " icntl14=" << s.final_icntl14 << " (retries="
                                  << s.factor_retries << ")"
                                  << " INFOG(11)=" << s.infog11
                                  << " INFOG(13)delayed_pivots=" << s.delayed_pivots
                                  << " INFOG(21)mem_MB=" << s.factor_mem_mb << "\n";
                    } else {
                        std::cout << "    FACTOR FAILED INFOG(1)=" << s.mumps_info1
                                  << " (icntl14=" << s.final_icntl14 << ", retries="
                                  << s.factor_retries << ")\n";
                    }
                }
                if (s.solve_done) {
                    std::cout << "    SOLVE " << (s.solve_ok ? "ok" : "FAILED")
                              << " backward_error ||Ax-b||/||b||_inf = "
                              << s.backward_error << "  -> "
                              << (s.solve_ok && s.backward_error >= 0.0 &&
                                          s.backward_error < 1e-6
                                      ? "NONSINGULAR"
                                      : "CHECK (singular/ill-conditioned?)")
                              << "\n";
                }
                std::cout.flush();
            };

            // build A_FF by dropping a row set and a col set, reindex to compact.
            auto build_a_ff =
                [&](const std::vector<char>& drop_row,
                    const std::vector<char>& drop_col, std::vector<int>& irn_ff,
                    std::vector<int>& jcn_ff, std::vector<double>& a_ff,
                    int& n_rows_ff, int& n_cols_ff, long long& empty_rows,
                    long long& empty_cols) {
                    std::vector<int> new_row(static_cast<std::size_t>(n_mat), -1);
                    std::vector<int> new_col(static_cast<std::size_t>(n_mat), -1);
                    n_rows_ff = 0;
                    n_cols_ff = 0;
                    for (int r = 0; r < n_mat; ++r)
                        if (!drop_row[static_cast<std::size_t>(r)])
                            new_row[static_cast<std::size_t>(r)] = n_rows_ff++;
                    for (int c = 0; c < n_mat; ++c)
                        if (!drop_col[static_cast<std::size_t>(c)])
                            new_col[static_cast<std::size_t>(c)] = n_cols_ff++;
                    irn_ff.clear();
                    jcn_ff.clear();
                    a_ff.clear();
                    std::vector<char> row_has(static_cast<std::size_t>(n_rows_ff), 0);
                    std::vector<char> col_has(static_cast<std::size_t>(n_cols_ff), 0);
                    for (long long k = 0; k < nnz_full; ++k) {
                        const int r = coo.irn[static_cast<std::size_t>(k)] - 1;
                        const int c = coo.jcn[static_cast<std::size_t>(k)] - 1;
                        if (drop_row[static_cast<std::size_t>(r)] ||
                            drop_col[static_cast<std::size_t>(c)])
                            continue;
                        const int nr = new_row[static_cast<std::size_t>(r)];
                        const int nc = new_col[static_cast<std::size_t>(c)];
                        irn_ff.push_back(nr + 1);
                        jcn_ff.push_back(nc + 1);
                        a_ff.push_back(coo.a[static_cast<std::size_t>(k)]);
                        row_has[static_cast<std::size_t>(nr)] = 1;
                        col_has[static_cast<std::size_t>(nc)] = 1;
                    }
                    empty_rows = 0;
                    empty_cols = 0;
                    for (int r = 0; r < n_rows_ff; ++r)
                        if (!row_has[static_cast<std::size_t>(r)]) ++empty_rows;
                    for (int c = 0; c < n_cols_ff; ++c)
                        if (!col_has[static_cast<std::size_t>(c)]) ++empty_cols;
                };

            // ---- (4) MUMPS on full A --------------------------------------
            std::cout << "\n[ANALYSIS] full A vs A_FF (same ICNTL(7)="
                      << mumps_ordering << ", ICNTL(14)init=200)\n";
            BorderMumpsStats stats_A = run_mumps(
                "A (full)", n_mat, coo.irn, coo.jcn, coo.a, do_factor,
                /*solve=*/false);
            print_stats(stats_A);

            // Index-symmetric border runner: drop rows AND cols at the mask
            // indices (the MUMPS ICNTL(19)/LISTVAR_SCHUR convention), reindex,
            // report shape + orphans, factor+solve when the interior is a clean
            // square block. Returns the stats for the ratio table.
            auto run_border = [&](const std::string& name,
                                  const std::vector<char>& mask,
                                  long long border_size) -> BorderMumpsStats {
                std::cout << "\n[BORDER: " << name << "] |H|=" << border_size
                          << " (index-symmetric rows+cols)\n";
                std::vector<int> irn_b, jcn_b;
                std::vector<double> a_b;
                int nr = 0, nc = 0;
                long long er = 0, ec = 0;
                build_a_ff(mask, mask, irn_b, jcn_b, a_b, nr, nc, er, ec);
                std::cout << "  A_FF: rows=" << nr << " cols=" << nc
                          << " nnz=" << static_cast<long long>(a_b.size())
                          << " square=" << (nr == nc ? "yes" : "NO")
                          << " empty_rows=" << er << " empty_cols=" << ec << "\n";
                BorderMumpsStats st;
                if (nr == nc && er == 0 && ec == 0) {
                    const std::string tag = "A_FF " + name;
                    st = run_mumps(tag.c_str(), nr, irn_b, jcn_b, a_b,
                                   do_factor, /*solve=*/do_factor);
                    print_stats(st);
                } else {
                    std::cout << "  SKIP MUMPS: non-square or orphaned rows/cols "
                                 "(this index-symmetric border cannot isolate a "
                                 "factorable interior)\n";
                }
                return st;
            };

            // Border masks.
            //  * primary  = col_dom==-1 (204) -- the task's headline index set.
            //  * union    = col_dom==-1 OR GlobalInt-row -- isolates BOTH the
            //               dense global columns AND the dense integral rows,
            //               the only index-symmetric set that captures the whole
            //               asymmetric arrowhead (204 dense cols != 4036 rows).
            //  * scalar8  = ColumnClass::ScalarGlobal -- the true numeric scalars
            //               (Omega/Hc/com/...); tiny-border sanity control.
            std::vector<char> is_scalar_global(static_cast<std::size_t>(n_mat), 0);
            long long n_scalar_global_mask = 0;
            for (int c = 0; c < n_mat; ++c) {
                if (metadata.columns[static_cast<std::size_t>(c)].column_class ==
                    ColumnClass::ScalarGlobal) {
                    is_scalar_global[static_cast<std::size_t>(c)] = 1;
                    ++n_scalar_global_mask;
                }
            }
            std::vector<char> is_union(static_cast<std::size_t>(n_mat), 0);
            long long n_union = 0;
            for (int i = 0; i < n_mat; ++i) {
                if (is_global_col[static_cast<std::size_t>(i)] ||
                    is_globalint_row[static_cast<std::size_t>(i)]) {
                    is_union[static_cast<std::size_t>(i)] = 1;
                    ++n_union;
                }
            }

            // ---- (2,3,4,5) index-symmetric border variants ----------------
            BorderMumpsStats stats_FF_primary =
                run_border("index-sym 204 global-col (col_dom==-1)", is_global_col,
                           n_global_by_dom);
            BorderMumpsStats stats_FF_union = run_border(
                "index-sym UNION global-col + GlobalInt-row (whole dense border)",
                is_union, n_union);
            BorderMumpsStats stats_FF_scalar8 = run_border(
                "index-sym 8 ScalarGlobal (numeric scalars only)", is_scalar_global,
                n_scalar_global_mask);

            // ---- (2,3) SECONDARY: taxonomy-paired border ------------------
            // H_rows = GlobalInt rows, H_cols = global cols. Only a square drop
            // when the counts match; report the index-set relationship.
            const bool sets_identical =
                (tax_globalint == n_global_by_dom) &&
                std::equal(is_global_col.begin(), is_global_col.end(),
                           is_globalint_row.begin());
            std::cout << "\n[BORDER: SECONDARY taxonomy-paired] H_rows=GlobalInt("
                      << tax_globalint << ") H_cols=global(" << n_global_by_dom
                      << ")  index_sets_identical_to_primary="
                      << (sets_identical ? "yes" : "no") << "\n";
            if (sets_identical) {
                std::cout << "  taxonomy-paired border == index-symmetric border "
                             "(same indices); primary result applies, no re-run.\n";
            } else if (tax_globalint != n_global_by_dom) {
                std::cout << "  SKIP: |GlobalInt rows|=" << tax_globalint
                          << " != |global cols|=" << n_global_by_dom
                          << " -> no square index-symmetric drop from taxonomy; "
                             "the natural constraint-row set is not size-matched to "
                             "the global-scalar column block.\n";
            } else {
                std::vector<int> irn_s, jcn_s;
                std::vector<double> a_s;
                int n_rows_s = 0, n_cols_s = 0;
                long long empty_rows_s = 0, empty_cols_s = 0;
                build_a_ff(is_globalint_row, is_global_col, irn_s, jcn_s, a_s,
                           n_rows_s, n_cols_s, empty_rows_s, empty_cols_s);
                std::cout << "  A_FF: rows=" << n_rows_s << " cols=" << n_cols_s
                          << " nnz=" << static_cast<long long>(a_s.size())
                          << " square=" << (n_rows_s == n_cols_s ? "yes" : "NO")
                          << " empty_rows=" << empty_rows_s
                          << " empty_cols=" << empty_cols_s << "\n";
                if (n_rows_s == n_cols_s && empty_rows_s == 0 &&
                    empty_cols_s == 0) {
                    BorderMumpsStats stats_FF_tax = run_mumps(
                        "A_FF (taxonomy)", n_rows_s, irn_s, jcn_s, a_s,
                        do_factor, /*solve=*/do_factor);
                    print_stats(stats_FF_tax);
                } else {
                    std::cout << "  SKIP MUMPS on taxonomy A_FF: non-square or "
                                 "orphaned rows/cols\n";
                }
            }

            // ---- (6) RATIOS + VERDICT (A_FF variants / full A) ------------
            std::cout << "\n[RATIOS] (index-symmetric A_FF / full A)\n";
            auto print_ratio = [&](const std::string& name,
                                   const BorderMumpsStats& ff) {
                if (!(stats_A.analysis_ok && ff.analysis_ok)) {
                    std::cout << "  [" << name
                              << "] ratios unavailable (A_FF singular/skipped)\n";
                    return;
                }
                const double flop_ratio = stats_A.est_flops > 0.0
                                              ? ff.est_flops / stats_A.est_flops
                                              : -1.0;
                const double ram_ratio =
                    stats_A.est_ram_mb > 0
                        ? static_cast<double>(ff.est_ram_mb) /
                              static_cast<double>(stats_A.est_ram_mb)
                        : -1.0;
                const double front_ratio =
                    stats_A.max_front > 0
                        ? static_cast<double>(ff.max_front) /
                              static_cast<double>(stats_A.max_front)
                        : -1.0;
                std::cout << "  [" << name << "] flops=" << flop_ratio
                          << " RAM=" << ram_ratio << " front=" << front_ratio;
                if (stats_A.factor_ok && ff.factor_ok &&
                    stats_A.factor_seconds > 0.0)
                    std::cout << " factor_wall="
                              << ff.factor_seconds / stats_A.factor_seconds;
                std::cout << "\n";
            };
            print_ratio("204 global-col", stats_FF_primary);
            print_ratio("UNION whole-border", stats_FF_union);
            print_ratio("8 ScalarGlobal", stats_FF_scalar8);

            // Headline verdict uses the UNION border (the only index-symmetric
            // set that isolates the whole dense arrowhead: 204 dense cols + 4036
            // dense integral rows). Fall back to the 204-col border if the union
            // did not factor.
            const BorderMumpsStats& headline =
                stats_FF_union.analysis_ok ? stats_FF_union : stats_FF_primary;
            const char* headline_name =
                stats_FF_union.analysis_ok ? "UNION whole-border" : "204 global-col";
            std::cout << "  VERDICT (" << headline_name << "): ";
            if (stats_A.analysis_ok && headline.analysis_ok &&
                stats_A.est_flops > 0.0) {
                const double flop_ratio = headline.est_flops / stats_A.est_flops;
                if (flop_ratio <= 0.5) {
                    std::cout << "flops(A_FF)/flops(A)=" << flop_ratio
                              << " <= 0.5 -> dense-border removal is a REAL lever "
                                 "(fund production ICNTL(19) path)\n";
                } else {
                    std::cout << "flops(A_FF)/flops(A)=" << flop_ratio
                              << " > 0.5 -> fill is bulk-dominated; border-removal "
                                 "economics idea DEAD (conditioning findings "
                                 "separate)\n";
                }
            } else {
                std::cout << "no index-symmetric border yielded a factorable "
                             "square A_FF -> the XCTS dense border is an ASYMMETRIC "
                             "arrowhead (204 dense cols != 4036 dense integral rows) "
                             "that MUMPS's index-symmetric ICNTL(19) Schur cannot "
                             "isolate; payoff-1 as a MUMPS Schur lever is "
                             "structurally blocked\n";
            }

            // ---- (ICNTL19 wrapper gap list) -------------------------------
            std::cout << "\n[ICNTL(19) WRAPPER GAP LIST] (grep mumps_linear_solver"
                         ".cpp)\n";
            std::cout << "  PRESENT: MumpsLinearSolver::extract_schur() wires "
                         "ICNTL(19)=2 + size_schur + listvar_schur (diagnostic; "
                         "returns dense S only).\n";
            std::cout << "  MISSING for a production bordered solve:\n";
            std::cout << "   1. reduced-RHS back-substitution (redrhs/lredrhs, "
                         "ICNTL(26)) to finish a solve via the A_FF factor + dense "
                         "border -- extract_schur returns S but cannot solve.\n";
            std::cout << "   2. border auto-selection: no col_dom==-1 / "
                         "RowTaxonomy::GlobalInt -> 1-based LISTVAR_SCHUR helper.\n";
            std::cout << "   3. extract_schur forces ICNTL(28)=1 + ICNTL(6,8,11)=0 "
                         "(sequential analysis; no production ordering reconciliation).\n";
            std::cout << "   4. only centralized ICNTL(19)=2 (NPROW=NPCOL=1); no "
                         "distributed Schur (ICNTL(19)=3) for the res19/21 regime.\n";
            std::cout << "   5. no JFNK plumbing: do_newton_jfnk_mumps / "
                         "do_newton_sparse never call extract_schur -- always "
                         "factor the full A.\n";
            std::cout << "================ ARM 17 END ================\n";
            std::cout.flush();
            MPI_Finalize();
            std::exit(0);
        }

        const int ndom = espace.get_nbr_domains();
        std::vector<SchurAggregateBlock> layout_blocks;
        std::vector<int> domain_to_aggregate;
        std::string aggregate_source;
        std::string aggregate_error;

        const char* manual_aggregates = std::getenv("SCHUR_AGGREGATES");
        bool layout_ok = false;
        if (manual_aggregates != nullptr && manual_aggregates[0] != '\0') {
            layout_ok = parse_manual_aggregates(manual_aggregates, ndom, layout_blocks,
                                                domain_to_aggregate, aggregate_source,
                                                aggregate_error);
        } else if (const auto* bns_space = dynamic_cast<const Space_bin_ns*>(&espace)) {
            layout_ok = build_bns_aggregate_layout(*bns_space, layout_blocks,
                                                   domain_to_aggregate,
                                                   aggregate_source, aggregate_error);
        } else if (const auto* bns_nosym_space =
                       dynamic_cast<const Space_bin_ns_nosym*>(&espace)) {
            layout_ok = build_bns_aggregate_layout(*bns_nosym_space, layout_blocks,
                                                   domain_to_aggregate,
                                                   aggregate_source, aggregate_error);
        } else {
            add_singleton_aggregates(ndom, layout_blocks, domain_to_aggregate);
            aggregate_source = "raw-domain singletons fallback";
            layout_ok = true;
        }

        if (!layout_ok) {
            if (rank == 0) {
                std::cerr << "do_newton_jfnk_schur: aggregate layout FAILED: "
                          << aggregate_error << std::endl;
            }
            collective_exit();
        }

        SchurAggregatePartition partition;
        build_aggregate_partition(metadata, layout_blocks, domain_to_aggregate,
                                  aggregate_source, partition);
        const double partition_seconds = elapsed_time(partition_start);

        const SchurAggregateCensus& census = partition.census;
        const int n_aggregates = static_cast<int>(partition.blocks.size());
        const int n_interface_rows = static_cast<int>(partition.interface_rows.size());
        const int n_interface_cols = static_cast<int>(partition.interface_cols.size());

        // Hard gates.
        const bool gate_unclassified_rows = (census.unclassified_rows == 0);
        const bool gate_unclassified_cols = (census.unclassified_cols == 0);
        const bool gate_all_square = partition.all_square;
        const bool gate_interface_square = (n_interface_rows == n_interface_cols);
        const bool gate_interface_bounds =
            (n_interface_cols > 0 && n_interface_cols < column_count);
        const bool contract_ok = gate_unclassified_rows && gate_unclassified_cols &&
                                 gate_all_square && gate_interface_square &&
                                 gate_interface_bounds;

        if (rank == 0) {
            std::cout << "=== aggregate-Schur partition contract ===\n";
            std::cout << "  partition_build_s=" << partition_seconds
                      << "  n=" << column_count
                      << "  ndom=" << ndom
                      << "  aggregates=" << n_aggregates
                      << "  layout=" << partition.source << "\n";
            std::cout << "  aggregate domains:\n";
            for (const SchurAggregateBlock& block : partition.blocks) {
                std::cout << "    g" << block.id << " " << block.label
                          << " domains=" << domains_label(block.domains) << "\n";
            }
            std::cout << "  rows: Vol(bulk)=" << census.rows_vol
                      << " TauBc=" << census.rows_tau_bc
                      << " TauMatch=" << census.rows_tau_match
                      << " GlobalInt=" << census.rows_global_int
                      << " Unknown=" << census.rows_unknown << "\n";
            std::cout << "  cols: FieldInteriorVol=" << census.cols_field_interior_vol
                      << " FieldOuterShellTau=" << census.cols_field_outer_shell_tau
                      << " FieldMatching=" << census.cols_field_matching
                      << " FieldGauge=" << census.cols_field_gauge
                      << " ScalarGlobal=" << census.cols_scalar_global << "\n";
            std::cout << "  bulk_rows=" << census.bulk_rows
                      << " bulk_cols=" << census.bulk_cols
                      << " interface_rows=" << n_interface_rows
                      << " interface_cols=" << n_interface_cols << "\n";
            std::cout << "  bulk cols by class: FieldInteriorVol="
                      << census.bulk_cols_field_interior_vol
                      << " FieldBoundaryTau=" << census.bulk_cols_field_boundary_tau
                      << " FieldOuterShellTau=" << census.bulk_cols_field_outer_shell_tau
                      << " FieldMatching=" << census.bulk_cols_field_matching
                      << " VarDomain=" << census.bulk_cols_var_domain
                      << " square_closure=" << census.bulk_cols_square_closure << "\n";
            std::cout << "  aggregate bulk blocks (aggregate: rows x cols [square]):\n";
            for (const SchurAggregateShape& shape : partition.shapes) {
                const SchurAggregateBlock& block =
                    partition.blocks[static_cast<std::size_t>(shape.id)];
                std::cout << "    g" << shape.id << " " << block.label
                          << " " << domains_label(block.domains) << ": "
                          << shape.bulk_rows << " x "
                          << shape.bulk_cols << " ["
                          << (shape.square ? "square" : "NON-SQUARE delta=") ;
                if (!shape.square)
                    std::cout << (shape.bulk_rows - shape.bulk_cols);
                std::cout << "]\n";
            }
            std::cout << "  GATES: unclassified_rows=" << (gate_unclassified_rows ? "ok" : "FAIL")
                      << " unclassified_cols=" << (gate_unclassified_cols ? "ok" : "FAIL")
                      << " all_square=" << (gate_all_square ? "ok" : "FAIL")
                      << " interface_square=" << (gate_interface_square ? "ok" : "FAIL")
                      << " interface_bounds=" << (gate_interface_bounds ? "ok" : "FAIL") << "\n";
            std::cout << "  CONTRACT: " << (contract_ok ? "PASS" : "FAIL")
                      << "  (aggregate-local closure must still pass the BLAS probe)\n";
            std::cout.flush();
        }

        if (!contract_ok) {
            if (rank == 0)
                std::cerr << "do_newton_jfnk_schur: contract FAILED; refusing to run the "
                             "formation probe on an unsound aggregate partition." << std::endl;
            collective_exit();
        }
        // ====================================================================
        // Phase 2: HAND FORMATION PROBE (contract passed).
        // ====================================================================
        const double configured_drop_tol =
            (config.mumps.drop_tol > 0.0) ? config.mumps.drop_tol : 1e-8;
        constexpr double kDropTolMin = 1e-16;
        const double drop_tol =
            std::max(kDropTolMin,
                     configured_drop_tol * std::sqrt(std::sqrt(std::max(error, 0.0))));

        const auto coo_start = std::chrono::system_clock::now();
        JacobianAssembler assembler(*this, MPI_COMM_WORLD);
        AssembledJacobianCoo coo = assembler.assemble(drop_tol); // collective
        const double coo_build_seconds = elapsed_time(coo_start);

        // Block extraction + BLAS are rank-0 only (rank 0 owns the full COO).
        // Rank 0 does no further MPI, so non-root ranks finalize now.
        if (rank != 0) {
            MPI_Finalize();
            std::exit(0);
        }

        // ====================================================================
        // p-coarse two-level PC probe (PCOARSE_DUMP / _PC_PROBE).
        // DIAGNOSTIC ONLY, rank-0, self-exiting -- mirrors the SCHUR_PC_GMRES
        // block's structure. `residual` still holds the TRUE entry residual
        // (captured before the COO sweep) = the genuine post-regrid Newton RHS.
        // ====================================================================
        if (pcoarse_dump_on) {
            pcoarse_dump_probe_artifacts(pcoarse_dump_dir, coo, drop_tol);
            std::cout.flush();
            MPI_Finalize();
            std::exit(0);
        }
        if (pcoarse_probe_on) {
            pcoarse_run_pc_probe(pcoarse_probe_dir, coo, residual, config);
            std::cout.flush();
            MPI_Finalize();
            std::exit(0);
        }

        const int n_interface = n_interface_cols;

        // Global row/col -> (aggregate, local) or interface index.
        std::vector<int> row_aggregate(column_count, -1), row_local(column_count, -1);
        std::vector<int> col_aggregate(column_count, -1), col_local(column_count, -1);
        std::vector<int> row_interface(column_count, -1), col_interface(column_count, -1);
        for (int li = 0; li < n_interface_rows; ++li)
            row_interface[partition.interface_rows[static_cast<std::size_t>(li)]] = li;
        for (int li = 0; li < n_interface_cols; ++li)
            col_interface[partition.interface_cols[static_cast<std::size_t>(li)]] = li;
        std::vector<int> aggregate_dim(n_aggregates, 0);
        for (int aggregate = 0; aggregate < n_aggregates; ++aggregate) {
            const SchurAggregateBlock& blk =
                partition.blocks[static_cast<std::size_t>(aggregate)];
            aggregate_dim[aggregate] = static_cast<int>(blk.bulk_cols.size());
            for (int lr = 0; lr < static_cast<int>(blk.bulk_rows.size()); ++lr) {
                row_aggregate[blk.bulk_rows[static_cast<std::size_t>(lr)]] = aggregate;
                row_local[blk.bulk_rows[static_cast<std::size_t>(lr)]] = lr;
            }
            for (int lc = 0; lc < static_cast<int>(blk.bulk_cols.size()); ++lc) {
                col_aggregate[blk.bulk_cols[static_cast<std::size_t>(lc)]] = aggregate;
                col_local[blk.bulk_cols[static_cast<std::size_t>(lc)]] = lc;
            }
        }

        // Dense interface Schur S = A_II (col-major). COO is 1-based.
        const long long nnz = coo.nnz;
        std::vector<double> schur(static_cast<std::size_t>(n_interface) *
                                      static_cast<std::size_t>(n_interface),
                                  0.0);
        for (long long k = 0; k < nnz; ++k) {
            const int ri = row_interface[coo.irn[static_cast<std::size_t>(k)] - 1];
            const int ci = col_interface[coo.jcn[static_cast<std::size_t>(k)] - 1];
            if (ri >= 0 && ci >= 0)
                schur[static_cast<std::size_t>(ri) +
                      static_cast<std::size_t>(ci) * n_interface] +=
                    coo.a[static_cast<std::size_t>(k)];
        }

        const char trans_n = 'N';
        const double alpha_minus_one = -1.0;
        const double beta_one = 1.0;

        double per_aggregate_factor_seconds = 0.0;
        double solve_seconds = 0.0;
        double gemm_seconds = 0.0;
        long long cross_aggregate_bulk_entries = 0;
        int singular_blocks = 0;

        // The dense per-aggregate Schur formation below is the legacy aggregate-Schur
        // diagnostic (closed as non-viable: dense exterior LU dwarfs the sparse MUMPS
        // factor). The SCHUR_PC_GMRES harness (Schwarz + oracle + target probe)
        // never consumes its output -- it exits before the manual probe that uses
        // `schur`/`singular_blocks` -- so skip the multi-minute dense LU/solve/gemm tax
        // on that path.
        const bool skip_dense_schur =
            env_flag_enabled("SCHUR_PC_GMRES", false) || pcoarse_dump_on ||
            pcoarse_probe_on;
        for (int aggregate = 0; !skip_dense_schur && aggregate < n_aggregates;
             ++aggregate) {
            const int m = aggregate_dim[aggregate];
            if (m <= 0)
                continue;
            const std::size_t mm = static_cast<std::size_t>(m);
            std::vector<double> a_bb(mm * mm, 0.0);
            std::vector<double> a_bi(mm * static_cast<std::size_t>(n_interface), 0.0);
            std::vector<double> a_ib(static_cast<std::size_t>(n_interface) * mm, 0.0);

            for (long long k = 0; k < nnz; ++k) {
                const int r = coo.irn[static_cast<std::size_t>(k)] - 1;
                const int c = coo.jcn[static_cast<std::size_t>(k)] - 1;
                const double v = coo.a[static_cast<std::size_t>(k)];
                if (row_aggregate[r] == aggregate) {
                    if (col_aggregate[c] == aggregate) {
                        a_bb[static_cast<std::size_t>(row_local[r]) +
                             static_cast<std::size_t>(col_local[c]) * mm] += v;
                    } else if (col_interface[c] >= 0) {
                        a_bi[static_cast<std::size_t>(row_local[r]) +
                             static_cast<std::size_t>(col_interface[c]) * mm] += v;
                    } else if (col_aggregate[c] >= 0) {
                        ++cross_aggregate_bulk_entries;
                    }
                } else if (row_interface[r] >= 0 && col_aggregate[c] == aggregate) {
                    a_ib[static_cast<std::size_t>(row_interface[r]) +
                         static_cast<std::size_t>(col_local[c]) *
                             static_cast<std::size_t>(n_interface)] += v;
                }
            }

            int m_dim = m;
            std::vector<int> ipiv(mm);

            int info = 0;
            const auto t_factor = std::chrono::system_clock::now();
            dgetrf_(&m_dim, &m_dim, a_bb.data(), &m_dim, ipiv.data(), &info);
            per_aggregate_factor_seconds += elapsed_time(t_factor);
            if (info != 0) {
                ++singular_blocks;
                const SchurAggregateBlock& block =
                    partition.blocks[static_cast<std::size_t>(aggregate)];
                std::cerr << "do_newton_jfnk_schur: aggregate g" << aggregate
                          << " " << block.label
                          << " domains=" << domains_label(block.domains)
                          << " A_BB dgetrf info=" << info << " (singular bulk block)"
                          << std::endl;
            }

            if (info == 0) {
                int nrhs = n_interface;
                int ldb = m;
                int sinfo = 0;
                const auto t_solve = std::chrono::system_clock::now();
                dgetrs_(const_cast<char*>(&trans_n), &m_dim, &nrhs, a_bb.data(),
                        &m_dim, ipiv.data(), a_bi.data(), &ldb, &sinfo); // a_bi <- Y_d
                solve_seconds += elapsed_time(t_solve);

                int mm_out = n_interface, nn_out = n_interface, kk = m;
                int lda = n_interface, ldb2 = m, ldc = n_interface;
                const auto t_gemm = std::chrono::system_clock::now();
                dgemm_(const_cast<char*>(&trans_n), const_cast<char*>(&trans_n),
                       &mm_out, &nn_out, &kk, const_cast<double*>(&alpha_minus_one),
                       a_ib.data(), &lda, a_bi.data(), &ldb2,
                       const_cast<double*>(&beta_one), schur.data(), &ldc); // S -= A_IB Y_d
                gemm_seconds += elapsed_time(t_gemm);
            }
        }

        // ====================================================================
        // Frozen-neighbour (Dirichlet) local-operator rank probe
        // (SCHUR_PROBE_DIRICHLET=1). Audits the 2026-06-04 "singular bulk"
        // kill. A_BB is the FREE block: it drops each aggregate's boundary
        // matching rows (promoted to the interface), so the boundary unknowns are
        // unconstrained -> singular. The additive-Schwarz local operator instead
        // FREEZES the neighbour unknowns: every equation that touches the
        // aggregate's unknowns C_g (interior PDE rows AND the boundary matching
        // rows) becomes a function of C_g alone once the non-C_g columns are
        // dropped. That operator is A_dir = A[rows touching C_g, C_g] (tall; no
        // row/col DOF-pairing assumed, unlike the dead A[S,S] shortcut). Its
        // smallest singular value decides the kill: sigma_min > 0 (full column
        // rank) => C_g is uniquely determined under Dirichlet => the singularity
        // is a free-BC artefact, not a real DD obstruction. sigma_min ~ 0 => a
        // genuine kernel survives freezing => the kill is real. Reports sigma_min
        // / sigma_max / numerical rank for A_BB (free), A_dir (frozen-neighbour),
        // and A_dir_noGlob (frozen-neighbour minus global-constraint rows).
        if (env_flag_enabled("SCHUR_PROBE_DIRICHLET", false)) {
            int svd_dim_max = 12000;  // dense-SVD column cap; larger blocks skipped
            if (const char* cap = std::getenv("SCHUR_PROBE_DIRICHLET_DIM_MAX"))
                svd_dim_max = std::atoi(cap);
            const double rank_tol = 1e-10;  // sigma_i > rank_tol*sigma_max counts

            // Singular spectrum of a dense col-major M x N matrix (jobz='N').
            // Destroys 'a'.
            auto singular_values = [](std::vector<double>& a, int M, int N) {
                const int mn = std::min(M, N);
                std::vector<double> s(static_cast<std::size_t>(std::max(1, mn)), 0.0);
                char jobz = 'N';
                int lda = std::max(1, M), ldu = 1, ldvt = 1, info = 0;
                double u = 0.0, vt = 0.0, wq = 0.0;
                int lwork = -1;
                std::vector<int> iwork(8 * static_cast<std::size_t>(std::max(1, mn)));
                dgesdd_(&jobz, &M, &N, a.data(), &lda, s.data(), &u, &ldu, &vt, &ldvt,
                        &wq, &lwork, iwork.data(), &info);  // workspace query
                lwork = (info == 0) ? static_cast<int>(wq)
                                    : (3 * mn + std::max(M, N) + 64);
                std::vector<double> work(static_cast<std::size_t>(std::max(1, lwork)));
                dgesdd_(&jobz, &M, &N, a.data(), &lda, s.data(), &u, &ldu, &vt, &ldvt,
                        work.data(), &lwork, iwork.data(), &info);
                if (info != 0)
                    s.assign(s.size(), -1.0);  // signal failure
                return s;
            };
            auto report = [&](const char* tag, std::vector<double>& a, int M, int N) {
                const auto t0 = std::chrono::system_clock::now();
                std::vector<double> s = singular_values(a, M, N);
                const double svd_s = elapsed_time(t0);
                const double smax = s.empty() ? 0.0 : s.front();
                const double smin = s.empty() ? 0.0 : s.back();
                int rank = 0;
                for (double sv : s)
                    if (sv > rank_tol * smax)
                        ++rank;
                std::cout << " | " << tag << "[" << M << "x" << N << "]"
                          << " smax=" << smax << " smin=" << smin << " rank=" << rank
                          << "/" << N << " defic=" << (N - rank) << " svd_s=" << svd_s;
            };

            std::vector<int> loc(static_cast<std::size_t>(column_count), -1);
            std::cout << "SCHUR_PROBE_DIRICHLET (frozen-neighbour rank test, "
                         "rank_tol="
                      << rank_tol << "):\n";
            for (int aggregate = 0; aggregate < n_aggregates; ++aggregate) {
                const SchurAggregateBlock& block =
                    partition.blocks[static_cast<std::size_t>(aggregate)];
                const int m = aggregate_dim[aggregate];
                if (m <= 0)
                    continue;
                std::cout << "  g" << aggregate << " " << block.label
                          << " domains=" << domains_label(block.domains) << " m=" << m;
                if (m > svd_dim_max) {
                    std::cout << "  SKIPPED (m > " << svd_dim_max << ")\n";
                    continue;
                }
                const std::size_t mm = static_cast<std::size_t>(m);
                for (int k = 0; k < m; ++k)
                    loc[static_cast<std::size_t>(
                        block.bulk_cols[static_cast<std::size_t>(k)])] = k;

                // Pass 1: rows touching C_g -> compact local row index.
                std::vector<int> touch_local(static_cast<std::size_t>(column_count), -1);
                int n_touch = 0;
                for (long long k = 0; k < nnz; ++k) {
                    if (loc[static_cast<std::size_t>(
                            coo.jcn[static_cast<std::size_t>(k)] - 1)] < 0)
                        continue;
                    const int r = coo.irn[static_cast<std::size_t>(k)] - 1;
                    if (touch_local[static_cast<std::size_t>(r)] < 0)
                        touch_local[static_cast<std::size_t>(r)] = n_touch++;
                }
                const std::size_t nt = static_cast<std::size_t>(n_touch);

                // Build each operator, SVD, and free before the next (caps peak
                // RSS so the merged exterior fits). A_BB = free block
                // A[bulk_rows x C_g]; A_dir = frozen-neighbour A[touch x C_g];
                // A_dir_noGlob = A_dir minus the global-constraint rows -> decides
                // whether the boundary matching rows alone close the kernel, or
                // the GlobalInt rows' footprint is required for full rank.
                {
                    std::vector<double> a_bb(mm * mm, 0.0);
                    for (long long k = 0; k < nnz; ++k) {
                        const int c = coo.jcn[static_cast<std::size_t>(k)] - 1;
                        const int lc = loc[static_cast<std::size_t>(c)];
                        if (lc < 0)
                            continue;
                        const int r = coo.irn[static_cast<std::size_t>(k)] - 1;
                        if (row_aggregate[r] == aggregate)
                            a_bb[static_cast<std::size_t>(row_local[r]) +
                                 static_cast<std::size_t>(lc) * mm] +=
                                coo.a[static_cast<std::size_t>(k)];
                    }
                    report("A_BB ", a_bb, m, m);
                }
                {
                    std::vector<double> a_dir(nt * mm, 0.0);
                    for (long long k = 0; k < nnz; ++k) {
                        const int c = coo.jcn[static_cast<std::size_t>(k)] - 1;
                        const int lc = loc[static_cast<std::size_t>(c)];
                        if (lc < 0)
                            continue;
                        const int r = coo.irn[static_cast<std::size_t>(k)] - 1;
                        a_dir[static_cast<std::size_t>(
                                  touch_local[static_cast<std::size_t>(r)]) +
                              static_cast<std::size_t>(lc) * nt] +=
                            coo.a[static_cast<std::size_t>(k)];
                    }
                    report("A_dir", a_dir, n_touch, m);
                }
                {
                    std::vector<double> a_dir_ng(nt * mm, 0.0);
                    for (long long k = 0; k < nnz; ++k) {
                        const int c = coo.jcn[static_cast<std::size_t>(k)] - 1;
                        const int lc = loc[static_cast<std::size_t>(c)];
                        if (lc < 0)
                            continue;
                        const int r = coo.irn[static_cast<std::size_t>(k)] - 1;
                        if (metadata.rows[static_cast<std::size_t>(r)].taxonomy ==
                            RowTaxonomy::GlobalInt)
                            continue;  // drop global-constraint rows
                        a_dir_ng[static_cast<std::size_t>(
                                     touch_local[static_cast<std::size_t>(r)]) +
                                 static_cast<std::size_t>(lc) * nt] +=
                            coo.a[static_cast<std::size_t>(k)];
                    }
                    report("A_dir_noGlob", a_dir_ng, n_touch, m);
                }
                std::cout << "\n";

                for (int k = 0; k < m; ++k)
                    loc[static_cast<std::size_t>(
                        block.bulk_cols[static_cast<std::size_t>(k)])] = -1;
            }
            std::cout.flush();
        }

        // ====================================================================
        // Schwarz-Dirichlet preconditioner convergence harness
        // (SCHUR_PC_GMRES=1). DIAGNOSTIC ONLY: it measures how many GMRES
        // iterations a one-level additive Schwarz-Dirichlet preconditioner needs
        // on the FULL Jacobian, it does NOT converge the Newton system.
        //
        // Operator (matvec): the sparse Jacobian applied straight from the COO
        // triplets (coo_spmv), 1-based indices.
        // Preconditioner M^{-1} (additive, no damping): the columns split into
        // disjoint aggregate bulk sets C_g plus the interface set, which together
        // partition all columns. Each block is solved with neighbours frozen:
        //   - per aggregate g: the Dirichlet local operator A[touch, C_g]
        //     (every row that touches C_g, restricted to the C_g columns) solved
        //     in the least-squares sense via QR (dgeqrf/dormqr/dtrtrs). NO normal
        //     equations -- A^T A would square the condition number.
        //   - interface: the dense A_II block solved by LU (dgetrf/dgetrs).
        // Each block's local solution is scattered into its disjoint column set,
        // so M^{-1} populates the whole vector additively.
        if (env_flag_enabled("SCHUR_PC_GMRES", false)) {
            const long long nnz = coo.nnz;

            // --- 1. COO sparse matvec (the operator). 1-based COO. ---
            auto coo_spmv = [&](const std::vector<double>& v, std::vector<double>& out) {
                out.assign(static_cast<std::size_t>(column_count), 0.0);
                for (long long k = 0; k < nnz; ++k) {
                    out[static_cast<std::size_t>(
                            coo.irn[static_cast<std::size_t>(k)] - 1)] +=
                        coo.a[static_cast<std::size_t>(k)] *
                        v[static_cast<std::size_t>(
                            coo.jcn[static_cast<std::size_t>(k)] - 1)];
                }
            };

            // --- 2. Parity self-check: COO matvec vs do_JX (print only). ---
            // x_rand: rough deterministic pseudo-random in [-1,1] (broadband, a
            // fair preconditioner stress). Doubles as x_true for the GMRES
            // benchmark below.
            std::vector<double> v_test(static_cast<std::size_t>(column_count));
            for (int i = 0; i < column_count; ++i) {
                const double h =
                    std::sin(static_cast<double>(i) * 12.9898 + 78.233) * 43758.5453;
                v_test[static_cast<std::size_t>(i)] = (h - std::floor(h)) * 2.0 - 1.0;
            }
            std::vector<double> av_coo;
            coo_spmv(v_test, av_coo);
            Array<double> v_arr(column_count);
            for (int i = 0; i < column_count; ++i)
                v_arr.set(i) = v_test[static_cast<std::size_t>(i)];
            Array<double> av_jx = do_JX(v_arr);
            double diff_sq = 0.0, jx_sq = 0.0;
            for (int i = 0; i < column_count; ++i) {
                const double d =
                    av_coo[static_cast<std::size_t>(i)] - av_jx.get_data()[i];
                diff_sq += d * d;
                jx_sq += av_jx.get_data()[i] * av_jx.get_data()[i];
            }
            const double parity_diff =
                (jx_sq > 0.0) ? std::sqrt(diff_sq / jx_sq) : std::sqrt(diff_sq);

            // --- 3. Local Dirichlet solvers (QR least-squares), one per disjoint
            // column block: the aggregate bulk sets AND the interface set. The
            // interface unknowns are globally coupled, so the raw square block
            // A[interface_rows, interface_cols] is SINGULAR (rows != cols are not
            // DOF-paired) -- it must be solved the same frozen-neighbour least-
            // squares way as the aggregates, never by LU. One factored A_g per
            // block, moved into the container (no copy).
            struct SchwarzLocalSolve {
                int id = -1;
                std::string label;
                int m = 0;        // columns in this block
                int n_touch = 0;  // rows touching the block columns
                double setup_s = 0.0;
                std::vector<double> a_qr;        // factored A[touch, cols], col-major
                std::vector<double> tau;         // QR scalar factors, size m
                std::vector<int> touch_global;   // local touch row -> global row
                std::vector<int> cols;           // block global column indices
            };
            std::vector<SchwarzLocalSolve> local_solves;
            local_solves.reserve(static_cast<std::size_t>(n_aggregates) + 1);

            const auto setup_start = std::chrono::system_clock::now();
            std::vector<int> loc(static_cast<std::size_t>(column_count), -1);
            std::vector<int> touch_local(static_cast<std::size_t>(column_count), -1);

            // Build a frozen-neighbour QR least-squares solver for one disjoint
            // column set: A[rows touching cols, cols], QR-factored (dgeqrf). Used
            // uniformly for every aggregate AND the interface.
            auto add_block = [&](const std::vector<int>& cols, int id,
                                 const std::string& label) {
                const int m = static_cast<int>(cols.size());
                if (m <= 0)
                    return;
                const std::size_t mm = static_cast<std::size_t>(m);
                for (int k = 0; k < m; ++k)
                    loc[static_cast<std::size_t>(cols[static_cast<std::size_t>(k)])] = k;

                std::fill(touch_local.begin(), touch_local.end(), -1);
                int n_touch = 0;
                std::vector<int> touch_global;
                for (long long k = 0; k < nnz; ++k) {
                    if (loc[static_cast<std::size_t>(
                            coo.jcn[static_cast<std::size_t>(k)] - 1)] < 0)
                        continue;
                    const int r = coo.irn[static_cast<std::size_t>(k)] - 1;
                    if (touch_local[static_cast<std::size_t>(r)] < 0) {
                        touch_local[static_cast<std::size_t>(r)] = n_touch++;
                        touch_global.push_back(r);
                    }
                }
                const std::size_t nt = static_cast<std::size_t>(n_touch);

                // Dense A_g (n_touch x m), col-major: only the block columns kept
                // (neighbours frozen = Dirichlet).
                std::vector<double> a_g(nt * mm, 0.0);
                for (long long k = 0; k < nnz; ++k) {
                    const int c = coo.jcn[static_cast<std::size_t>(k)] - 1;
                    const int lc = loc[static_cast<std::size_t>(c)];
                    if (lc < 0)
                        continue;
                    const int r = coo.irn[static_cast<std::size_t>(k)] - 1;
                    a_g[static_cast<std::size_t>(
                            touch_local[static_cast<std::size_t>(r)]) +
                        static_cast<std::size_t>(lc) * nt] +=
                        coo.a[static_cast<std::size_t>(k)];
                }

                int qr_m = n_touch, qr_n = m, qr_lda = std::max(1, n_touch);
                std::vector<double> tau(mm);
                int info = 0;
                double wq = 0.0;
                int lwork = -1;
                const auto t_qr = std::chrono::system_clock::now();
                dgeqrf_(&qr_m, &qr_n, a_g.data(), &qr_lda, tau.data(), &wq, &lwork,
                        &info);  // workspace query
                lwork = (info == 0) ? static_cast<int>(wq) : std::max(1, m);
                std::vector<double> work(static_cast<std::size_t>(std::max(1, lwork)));
                dgeqrf_(&qr_m, &qr_n, a_g.data(), &qr_lda, tau.data(), work.data(),
                        &lwork, &info);
                const double qrt = elapsed_time(t_qr);
                if (info != 0)
                    std::cerr << "do_newton_jfnk_schur: SCHUR_PC_GMRES block "
                              << label << " dgeqrf info=" << info << std::endl;

                SchwarzLocalSolve solve;
                solve.id = id;
                solve.label = label;
                solve.m = m;
                solve.n_touch = n_touch;
                solve.setup_s = qrt;
                solve.a_qr = std::move(a_g);
                solve.tau = std::move(tau);
                solve.touch_global = std::move(touch_global);
                solve.cols = cols;
                local_solves.push_back(std::move(solve));

                for (int k = 0; k < m; ++k)
                    loc[static_cast<std::size_t>(cols[static_cast<std::size_t>(k)])] = -1;
            };

            for (int aggregate = 0; aggregate < n_aggregates; ++aggregate)
                add_block(
                    partition.blocks[static_cast<std::size_t>(aggregate)].bulk_cols,
                    aggregate,
                    partition.blocks[static_cast<std::size_t>(aggregate)].label);
            add_block(partition.interface_cols, n_aggregates, "interface");

            // --- 4b. Nicolaides coarse space (2-level). One constant per
            // (aggregate, term_idx) field block + one 1-hot column per global /
            // gauge DOF (carries the GlobalInt pins exactly). Galerkin coarse
            // operator A0 = R0 A R0^T, dense LU. term_idx (NOT var_idx) splits the
            // shift tensor components, which the rigid-rotation/CKV gauge near-null
            // mode requires. No basis_mode bucketing (basis_mode is the flat
            // coefficient index, not multipole l). The 1-level Schwarz alone
            // stalls (300 iters) because it cannot propagate these global modes;
            // the coarse correction deflates them. Toggle: SCHUR_PC_COARSE.
            const bool coarse_on =
                env_flag_enabled("SCHUR_PC_COARSE", true);
            // ANALYTIC GAUGE coarse space (SCHUR_PC_COARSE_GAUGE).
            // Replaces the Nicolaides per-aggregate constants with the eight
            // analytic XCTS gauge / conformal-Killing modes of the shift,
            // conformal factor and lapse: three rigid rotations beta^i =
            // eps^{iaj} X_j (LINEAR in position -- the slow modes constants miss),
            // three translations beta^i = delta^{ia}, and two global scalar
            // constants (conf=const, lapse=const). The GlobalInt-pin singletons
            // (ScalarGlobal / var_double) are still appended so A0 stays
            // nonsingular. Each analytic mode is rendered as a field, coef()-ed,
            // and projected onto the column-space tau slots via the same
            // affecte_tau_one_coef walk the column engine seeds with -- no
            // basis_mode decoding (the fragile path that broke the tail-decay
            // enrichment). Reuses the A0 = R0 A R0^T / LU / balanced-correction
            // machinery below unchanged.
            const bool coarse_gauge_on =
                env_flag_enabled("SCHUR_PC_COARSE_GAUGE", false);
            const bool coarse_oracle_on =
                coarse_on && env_flag_enabled("SCHUR_PC_COARSE_ORACLE", false);
            auto dot = [](const std::vector<double>& a,
                          const std::vector<double>& b) {
                double sum = 0.0;
                const std::size_t n = std::min(a.size(), b.size());
                for (std::size_t i = 0; i < n; ++i)
                    sum += a[i] * b[i];
                return sum;
            };
            auto norm2_sq = [&](const std::vector<double>& a) {
                return std::max(0.0, dot(a, a));
            };
            auto norm2 = [&](const std::vector<double>& a) {
                return std::sqrt(norm2_sq(a));
            };
            auto orthonormalize = [&](std::vector<std::vector<double>>& basis) {
                std::vector<std::vector<double>> kept;
                kept.reserve(basis.size());
                for (std::vector<double>& v : basis) {
                    for (int pass = 0; pass < 2; ++pass) {
                        for (const std::vector<double>& q : kept) {
                            const double alpha = dot(q, v);
                            for (int i = 0; i < column_count; ++i)
                                v[static_cast<std::size_t>(i)] -=
                                    alpha * q[static_cast<std::size_t>(i)];
                        }
                    }
                    const double nrm = norm2(v);
                    if (nrm <= 1e-12)
                        continue;
                    for (double& value : v)
                        value /= nrm;
                    kept.push_back(std::move(v));
                }
                basis = std::move(kept);
            };
            auto projection_residual_fraction =
                [&](const std::vector<std::vector<double>>& basis,
                    const std::vector<double>& re,
                    const std::vector<double>& im) {
                    const double total = norm2_sq(re) + norm2_sq(im);
                    if (total <= 0.0)
                        return 0.0;
                    double projected = 0.0;
                    for (const std::vector<double>& q : basis) {
                        const double ar = dot(q, re);
                        const double ai = dot(q, im);
                        projected += ar * ar + ai * ai;
                    }
                    const double residual = std::max(0.0, total - projected);
                    return std::sqrt(residual / total);
                };
            std::vector<std::vector<std::pair<int, double>>> coarse_rows;
            std::vector<std::vector<double>> oracle_right_basis;
            std::vector<std::vector<double>> oracle_left_basis;
            int oracle_biorth_info = 0;
            if (coarse_oracle_on) {
                int oracle_k = 12;
                if (const char* raw = std::getenv("SCHUR_PC_ORACLE_K"))
                    oracle_k = std::atoi(raw);
                int oracle_iters = 5;
                if (const char* raw = std::getenv("SCHUR_PC_ORACLE_ITERS"))
                    oracle_iters = std::atoi(raw);
                oracle_k = std::max(0, std::min(oracle_k, column_count));
                oracle_iters = std::max(1, oracle_iters);

                auto init_basis = [&](int k, double phase) {
                    std::vector<std::vector<double>> basis(
                        static_cast<std::size_t>(k),
                        std::vector<double>(static_cast<std::size_t>(column_count), 0.0));
                    for (int j = 0; j < k; ++j) {
                        for (int i = 0; i < column_count; ++i) {
                            const double x =
                                std::sin((static_cast<double>(i + 1) * 12.9898 +
                                          static_cast<double>(j + 1) * 78.233 + phase) *
                                         static_cast<double>(j + 3));
                            basis[static_cast<std::size_t>(j)]
                                [static_cast<std::size_t>(i)] =
                                    x - std::floor(x) - 0.5;
                        }
                    }
                    orthonormalize(basis);
                    return basis;
                };
                auto coo_spmtv = [&](const std::vector<double>& v,
                                     std::vector<double>& out) {
                    out.assign(static_cast<std::size_t>(column_count), 0.0);
                    for (long long k = 0; k < nnz; ++k) {
                        out[static_cast<std::size_t>(
                                coo.jcn[static_cast<std::size_t>(k)] - 1)] +=
                            coo.a[static_cast<std::size_t>(k)] *
                            v[static_cast<std::size_t>(
                                coo.irn[static_cast<std::size_t>(k)] - 1)];
                    }
                };

                const auto oracle_start = std::chrono::system_clock::now();
                MumpsLinearSolver oracle_solver(
                    column_count,
                    config.mumps.ordering,
                    config.mumps.out_of_core,
                    config.mumps.blr,
                    mumps_runtime_state.icntl14,
                    MPI_COMM_SELF,
                    0,
                    false,
                    config.mumps.out_of_core_touch,
                    config.mumps.out_of_core_safety,
                    config.mumps.out_of_core_budget_mb);
                oracle_solver.set_pattern(column_count, nnz, coo.irn.data(), coo.jcn.data());
                oracle_solver.analyze_pattern();
                oracle_solver.factor_analyzed(coo.a.data());
                mumps_runtime_state.icntl14 = oracle_solver.last_icntl14();

                oracle_right_basis = init_basis(oracle_k, 0.0);
                oracle_left_basis = init_basis(oracle_k, 0.3141592653589793);
                for (int iter = 0; iter < oracle_iters; ++iter) {
                    for (std::vector<double>& v : oracle_right_basis)
                        oracle_solver.solve(v.data());
                    orthonormalize(oracle_right_basis);
                    for (std::vector<double>& w : oracle_left_basis)
                        oracle_solver.solve_transpose(w.data());
                    orthonormalize(oracle_left_basis);
                }

                const int actual_k = std::min(static_cast<int>(oracle_right_basis.size()),
                                              static_cast<int>(oracle_left_basis.size()));
                oracle_right_basis.resize(static_cast<std::size_t>(actual_k));
                oracle_left_basis.resize(static_cast<std::size_t>(actual_k));

                // Biorthogonalize left vectors so W^T V ~= I. This keeps the
                // Petrov coarse operator numerically meaningful for nonnormal A.
                if (actual_k > 0) {
                    std::vector<double> b(static_cast<std::size_t>(actual_k) *
                                              static_cast<std::size_t>(actual_k),
                                          0.0);
                    for (int j = 0; j < actual_k; ++j)
                        for (int i = 0; i < actual_k; ++i)
                            b[static_cast<std::size_t>(i) +
                              static_cast<std::size_t>(j) *
                                  static_cast<std::size_t>(actual_k)] =
                                dot(oracle_left_basis[static_cast<std::size_t>(i)],
                                    oracle_right_basis[static_cast<std::size_t>(j)]);

                    std::vector<double> transform(static_cast<std::size_t>(actual_k) *
                                                      static_cast<std::size_t>(actual_k),
                                                  0.0);
                    for (int i = 0; i < actual_k; ++i)
                        transform[static_cast<std::size_t>(i) +
                                  static_cast<std::size_t>(i) *
                                      static_cast<std::size_t>(actual_k)] = 1.0;
                    std::vector<int> ipiv(static_cast<std::size_t>(actual_k));
                    int k_dim = actual_k;
                    dgetrf_(&k_dim, &k_dim, b.data(), &k_dim, ipiv.data(),
                            &oracle_biorth_info);
                    if (oracle_biorth_info == 0) {
                        char trans_t = 'T';
                        int nrhs = actual_k;
                        int solve_info = 0;
                        dgetrs_(&trans_t, &k_dim, &nrhs, b.data(), &k_dim,
                                ipiv.data(), transform.data(), &k_dim, &solve_info);
                        oracle_biorth_info = solve_info;
                    }
                    if (oracle_biorth_info == 0) {
                        std::vector<std::vector<double>> w_new(
                            static_cast<std::size_t>(actual_k),
                            std::vector<double>(static_cast<std::size_t>(column_count), 0.0));
                        for (int j = 0; j < actual_k; ++j) {
                            for (int l = 0; l < actual_k; ++l) {
                                const double alpha =
                                    transform[static_cast<std::size_t>(l) +
                                              static_cast<std::size_t>(j) *
                                                  static_cast<std::size_t>(actual_k)];
                                if (alpha == 0.0)
                                    continue;
                                for (int i = 0; i < column_count; ++i)
                                    w_new[static_cast<std::size_t>(j)]
                                         [static_cast<std::size_t>(i)] +=
                                        alpha *
                                        oracle_left_basis[static_cast<std::size_t>(l)]
                                                         [static_cast<std::size_t>(i)];
                            }
                        }
                        oracle_left_basis = std::move(w_new);
                    }
                }

                double max_wtv_err = 0.0;
                double max_right_resid = 0.0;
                double max_left_resid = 0.0;
                std::vector<double> av;
                for (int j = 0; j < actual_k; ++j) {
                    for (int i = 0; i < actual_k; ++i) {
                        const double target = (i == j) ? 1.0 : 0.0;
                        max_wtv_err = std::max(
                            max_wtv_err,
                            std::abs(dot(oracle_left_basis[static_cast<std::size_t>(i)],
                                         oracle_right_basis[static_cast<std::size_t>(j)]) -
                                     target));
                    }
                    coo_spmv(oracle_right_basis[static_cast<std::size_t>(j)], av);
                    const double right_nrm =
                        norm2(oracle_right_basis[static_cast<std::size_t>(j)]);
                    if (right_nrm > 0.0)
                        max_right_resid =
                            std::max(max_right_resid, norm2(av) / right_nrm);
                    coo_spmtv(oracle_left_basis[static_cast<std::size_t>(j)], av);
                    const double left_nrm =
                        norm2(oracle_left_basis[static_cast<std::size_t>(j)]);
                    if (left_nrm > 0.0)
                        max_left_resid =
                            std::max(max_left_resid, norm2(av) / left_nrm);
                }

                std::cout << "  oracle coarse setup:"
                          << " requested_k=" << oracle_k
                          << " actual_k=" << actual_k
                          << " inverse_iters=" << oracle_iters
                          << " biorth_info=" << oracle_biorth_info
                          << " max|WTV-I|=" << max_wtv_err
                          << " max||Av||/||v||=" << max_right_resid
                          << " max||ATw||/||w||=" << max_left_resid
                          << " factor_mem_MB/rank=" << oracle_solver.factor_memory_mb()
                          << " factor_mem_total_MB="
                          << oracle_solver.factor_memory_total_mb()
                          << " setup_s=" << elapsed_time(oracle_start) << "\n";
            } else if (coarse_on && coarse_gauge_on) {
                // Locate the shift / conformal / lapse variable indices by name.
                auto find_var = [&](std::initializer_list<const char*> names) {
                    for (const char* want : names)
                        for (int v = 0; v < nvar; ++v)
                            if (names_var[static_cast<std::size_t>(v)] == want)
                                return v;
                    return -1;
                };
                const int shift_idx = find_var({"bet", "beta", "shift"});
                const int conf_idx = find_var({"P", "conf", "psi"});
                const int lapse_idx = find_var({"N", "lapse"});

                // One analytic mode = a field instance (a copy of the owning var,
                // so its type/basis/component layout matches exactly), with a
                // per-(domain,component) Cartesian profile, std-based and coef()-ed.
                // gauge_modes[k] holds one coef()-ed field; mode_var[k] is its var
                // index (so the projection only touches that var's columns).
                std::vector<std::unique_ptr<Tensor>> gauge_modes;
                std::vector<int> mode_var;
                std::vector<std::string> mode_label;

                // Levi-Civita symbol for the rotation profile.
                auto levi = [](int i, int a, int j) {
                    if ((i == 1 && a == 2 && j == 3) ||
                        (i == 2 && a == 3 && j == 1) ||
                        (i == 3 && a == 1 && j == 2))
                        return 1.0;
                    if ((i == 1 && a == 3 && j == 2) ||
                        (i == 3 && a == 2 && j == 1) ||
                        (i == 2 && a == 1 && j == 3))
                        return -1.0;
                    return 0.0;
                };

                // Shift gauge modes: rotations (linear) and translations (const).
                if (shift_idx >= 0) {
                    for (int axis = 1; axis <= 3; ++axis) {
                        auto mode = std::make_unique<Tensor>(
                            *var[static_cast<std::size_t>(shift_idx)], true);
                        mode->annule_hard();
                        for (int comp = 1; comp <= 3; ++comp) {
                            bool any = false;
                            for (int dom = dom_min; dom <= dom_max; ++dom) {
                                for (int j = 1; j <= 3; ++j) {
                                    const double e = levi(comp, axis, j);
                                    if (e == 0.0)
                                        continue;
                                    mode->set(comp).set_domain(dom) +=
                                        e * espace.get_domain(dom)->get_cart(j);
                                    any = true;
                                }
                            }
                            (void)any;
                        }
                        mode->std_base();
                        mode->coef();
                        gauge_modes.push_back(std::move(mode));
                        mode_var.push_back(shift_idx);
                        mode_label.push_back(std::string("rot") +
                                             static_cast<char>('w' + axis));
                    }
                    for (int axis = 1; axis <= 3; ++axis) {
                        auto mode = std::make_unique<Tensor>(
                            *var[static_cast<std::size_t>(shift_idx)], true);
                        mode->annule_hard();
                        for (int dom = dom_min; dom <= dom_max; ++dom)
                            mode->set(axis).set_domain(dom) = 1.0;
                        mode->std_base();
                        mode->coef();
                        gauge_modes.push_back(std::move(mode));
                        mode_var.push_back(shift_idx);
                        mode_label.push_back(std::string("trans") +
                                             static_cast<char>('w' + axis));
                    }
                }

                // Global scalar constants for the conformal factor and lapse.
                for (int sv : {conf_idx, lapse_idx}) {
                    if (sv < 0)
                        continue;
                    auto mode = std::make_unique<Tensor>(
                        *var[static_cast<std::size_t>(sv)], true);
                    mode->annule_hard();
                    for (int dom = dom_min; dom <= dom_max; ++dom)
                        mode->set().set_domain(dom) = 1.0;
                    mode->std_base();
                    mode->coef();
                    gauge_modes.push_back(std::move(mode));
                    mode_var.push_back(sv);
                    mode_label.push_back(sv == conf_idx ? "conf_const"
                                                        : "lapse_const");
                }

                const int n_gauge = static_cast<int>(gauge_modes.size());
                coarse_rows.assign(static_cast<std::size_t>(n_gauge), {});

                // Project each analytic mode onto its var's tau columns. For a
                // column c the forward tau value xx_to_vars would consume is
                // recovered by the affecte_tau_one_coef walk: a zeroed probe of
                // the same tensor type gets exactly the basis function of slot
                // basis_mode (counter starts at 0 == that column's local index),
                // and the mode's tau value is the inner product of the probe's
                // nonzero coefficient cells with the mode field's coefficients.
                // For these low-order (constant / linear) modes the probe is
                // supported on a single primary cell whose Galerkin-image cells
                // multiply the mode's ~0 high-r coefficients, so the inner product
                // equals the mode's primary coefficient exactly.
                std::vector<Tensor> probe_for_var;  // one reusable probe per var
                std::vector<int> var_probe_slot(static_cast<std::size_t>(nvar), -1);
                for (int k = 0; k < n_gauge; ++k) {
                    const int vidx = mode_var[static_cast<std::size_t>(k)];
                    if (var_probe_slot[static_cast<std::size_t>(vidx)] < 0) {
                        var_probe_slot[static_cast<std::size_t>(vidx)] =
                            static_cast<int>(probe_for_var.size());
                        probe_for_var.emplace_back(
                            *var[static_cast<std::size_t>(vidx)], true);
                    }
                }

                for (int c = 0; c < column_count; ++c) {
                    const ColumnMetadata& cm =
                        metadata.columns[static_cast<std::size_t>(c)];
                    if (cm.var_idx < 0 || cm.domain < 0 || cm.basis_mode < 0)
                        continue;
                    // Does any gauge mode live on this column's variable?
                    bool any_mode = false;
                    for (int k = 0; k < n_gauge && !any_mode; ++k)
                        if (mode_var[static_cast<std::size_t>(k)] == cm.var_idx)
                            any_mode = true;
                    if (!any_mode)
                        continue;

                    Tensor& probe =
                        probe_for_var[static_cast<std::size_t>(
                            var_probe_slot[static_cast<std::size_t>(cm.var_idx)])];
                    probe.annule_hard();
                    int counter = 0;
                    espace.get_domain(cm.domain)->affecte_tau_one_coef(
                        probe, cm.domain, cm.basis_mode, counter);

                    // Inner product of the probe's set cells with each mode field,
                    // component by component, in this domain only.
                    for (int k = 0; k < n_gauge; ++k) {
                        if (mode_var[static_cast<std::size_t>(k)] != cm.var_idx)
                            continue;
                        const Tensor& mode =
                            *gauge_modes[static_cast<std::size_t>(k)];
                        double value = 0.0;
                        for (int comp = 0; comp < probe.get_n_comp(); ++comp) {
                            const Array<int> ind(probe.indices(comp));
                            const Val_domain& pv = probe(ind)(cm.domain);
                            if (pv.check_if_zero())
                                continue;
                            const Val_domain& mv = mode(ind)(cm.domain);
                            if (mv.check_if_zero())
                                continue;
                            const Array<double> pc(pv.get_coef());
                            const Array<double> mc(mv.get_coef());
                            const std::size_t n =
                                std::min(pc.get_nbr(), mc.get_nbr());
                            for (std::size_t t = 0; t < n; ++t)
                                value += pc.get_data()[t] * mc.get_data()[t];
                        }
                        if (value != 0.0)
                            coarse_rows[static_cast<std::size_t>(k)].push_back(
                                {c, value});
                    }
                }

                // Drop empty gauge rows (a missing var contributes nothing) so A0
                // never gets an all-zero coarse vector.
                {
                    std::vector<std::vector<std::pair<int, double>>> kept;
                    for (int k = 0; k < n_gauge; ++k)
                        if (!coarse_rows[static_cast<std::size_t>(k)].empty())
                            kept.push_back(
                                std::move(coarse_rows[static_cast<std::size_t>(k)]));
                    coarse_rows = std::move(kept);
                }

                // Keep the GlobalInt-pin singletons (ScalarGlobal / var_double)
                // so A0 stays nonsingular -- identical to the constants path.
                for (int c = 0; c < column_count; ++c) {
                    const ColumnMetadata& cm =
                        metadata.columns[static_cast<std::size_t>(c)];
                    if (cm.column_class == ColumnClass::ScalarGlobal ||
                        cm.var_double_idx >= 0)
                        coarse_rows.push_back({{c, 1.0}});
                }

                // Sanity gate: the rotation conformal-Killing modes must be
                // near-null under A. Print ||A v|| / ||v|| per gauge mode (the
                // prolonged column vector). A large ratio on a rotation row means
                // the component map / coordinates were built wrong.
                {
                    std::vector<double> v(static_cast<std::size_t>(column_count)),
                        av;
                    std::cout << "  gauge-mode ||Av||/||v|| sanity:";
                    for (std::size_t k = 0; k < coarse_rows.size(); ++k) {
                        std::fill(v.begin(), v.end(), 0.0);
                        for (const auto& e : coarse_rows[k])
                            v[static_cast<std::size_t>(e.first)] = e.second;
                        double vn = 0.0;
                        for (double vi : v)
                            vn += vi * vi;
                        vn = std::sqrt(vn);
                        if (vn == 0.0)
                            continue;
                        coo_spmv(v, av);
                        double an = 0.0;
                        for (double ai : av)
                            an += ai * ai;
                        an = std::sqrt(an);
                        const char* lbl =
                            (k < mode_label.size())
                                ? mode_label[k].c_str()
                                : "pin";
                        std::cout << " " << lbl << "=" << (an / vn);
                    }
                    std::cout << "\n";
                }

                for (auto& crow : coarse_rows) {  // unit-2-norm each coarse vector
                    double nrm = 0.0;
                    for (const auto& e : crow)
                        nrm += e.second * e.second;
                    nrm = std::sqrt(nrm);
                    if (nrm > 0.0)
                        for (auto& e : crow)
                            e.second /= nrm;
                }
            } else if (coarse_on) {
                std::map<std::pair<int, int>, int> key2row;
                for (int c = 0; c < column_count; ++c) {
                    const ColumnMetadata& cm =
                        metadata.columns[static_cast<std::size_t>(c)];
                    const int agg = col_aggregate[static_cast<std::size_t>(c)];
                    const ColumnClass cls = cm.column_class;
                    if (cls == ColumnClass::VarDomain)
                        continue;  // surface-mapping DOFs: leave to the 1-level part
                    // Pass B singletons are ONLY the few true global scalars (the
                    // GlobalInt pins). FieldGauge is a full field (~hundreds of
                    // columns) -> it goes through Pass A as constants, NOT
                    // singletons, else nC explodes and A0 is rank-deficient.
                    const bool global_scalar = (cls == ColumnClass::ScalarGlobal) ||
                                               (cm.var_double_idx >= 0);
                    if (global_scalar) {
                        coarse_rows.push_back({{c, 1.0}});  // Pass B: global pin
                    } else if (cm.term_idx >= 0 && cls != ColumnClass::Unknown) {
                        // Pass A: constant per (group, term_idx). Interface field
                        // columns (agg<0, incl FieldGauge) form one extra group so
                        // the gauge mode is a few constants, not many singletons.
                        const int grp = (agg >= 0) ? agg : n_aggregates;
                        const std::pair<int, int> key(grp, cm.term_idx);
                        auto it = key2row.find(key);
                        int row;
                        if (it == key2row.end()) {
                            row = static_cast<int>(coarse_rows.size());
                            key2row.emplace(key, row);
                            coarse_rows.emplace_back();
                        } else {
                            row = it->second;
                        }
                        coarse_rows[static_cast<std::size_t>(row)].push_back(
                            {c, 1.0});
                    }
                }
                for (auto& crow : coarse_rows) {  // unit-2-norm each coarse vector
                    double nrm = 0.0;
                    for (const auto& e : crow)
                        nrm += e.second * e.second;
                    nrm = std::sqrt(nrm);
                    if (nrm > 0.0)
                        for (auto& e : crow)
                            e.second /= nrm;
                }
            }
            const int nC = coarse_oracle_on
                                ? static_cast<int>(oracle_right_basis.size())
                                : static_cast<int>(coarse_rows.size());

            // Restriction R0 (n -> nC) and prolongation R0^T (nC -> n).
            auto restrict0 = [&](const std::vector<double>& in,
                                 std::vector<double>& c) {
                c.assign(static_cast<std::size_t>(nC), 0.0);
                if (coarse_oracle_on) {
                    for (int i = 0; i < nC; ++i) {
                        const std::vector<double>& w =
                            oracle_left_basis[static_cast<std::size_t>(i)];
                        double sum = 0.0;
                        for (int j = 0; j < column_count; ++j)
                            sum += w[static_cast<std::size_t>(j)] *
                                   in[static_cast<std::size_t>(j)];
                        c[static_cast<std::size_t>(i)] = sum;
                    }
                    return;
                }
                for (int i = 0; i < nC; ++i)
                    for (const auto& e : coarse_rows[static_cast<std::size_t>(i)])
                        c[static_cast<std::size_t>(i)] +=
                            e.second * in[static_cast<std::size_t>(e.first)];
            };
            auto prolong0 = [&](const std::vector<double>& c,
                                std::vector<double>& out) {
                out.assign(static_cast<std::size_t>(column_count), 0.0);
                if (coarse_oracle_on) {
                    for (int i = 0; i < nC; ++i) {
                        const double alpha = c[static_cast<std::size_t>(i)];
                        if (alpha == 0.0)
                            continue;
                        const std::vector<double>& v =
                            oracle_right_basis[static_cast<std::size_t>(i)];
                        for (int j = 0; j < column_count; ++j)
                            out[static_cast<std::size_t>(j)] +=
                                alpha * v[static_cast<std::size_t>(j)];
                    }
                    return;
                }
                for (int i = 0; i < nC; ++i)
                    for (const auto& e : coarse_rows[static_cast<std::size_t>(i)])
                        out[static_cast<std::size_t>(e.first)] +=
                            e.second * c[static_cast<std::size_t>(i)];
            };

            // Galerkin A0 = R0 A R0^T (dense nC x nC, col-major): nC sparse
            // mat-vecs, then dense LU.
            std::vector<double> a0(static_cast<std::size_t>(nC) *
                                       static_cast<std::size_t>(nC),
                                   0.0);
            int a0_info = 0;
            double a0_anorm = 0.0;  // 1-norm of the factored A0 (for rcond in the probe)
            std::vector<int> a0_ipiv(static_cast<std::size_t>(std::max(1, nC)));
            if (nC > 0) {
                std::vector<double> ej(static_cast<std::size_t>(nC), 0.0);
                std::vector<double> w, aw, col;
                for (int j = 0; j < nC; ++j) {
                    ej[static_cast<std::size_t>(j)] = 1.0;
                    prolong0(ej, w);     // w  = R0^T e_j
                    coo_spmv(w, aw);     // aw = A w
                    restrict0(aw, col);  // col = R0 A R0^T e_j
                    for (int i = 0; i < nC; ++i)
                        a0[static_cast<std::size_t>(i) +
                           static_cast<std::size_t>(j) *
                               static_cast<std::size_t>(nC)] =
                            col[static_cast<std::size_t>(i)];
                    ej[static_cast<std::size_t>(j)] = 0.0;
                }
                // Tikhonov-regularize: the coarse space targets near-null modes,
                // so A0 is near-singular by construction; a small diagonal shift
                // makes the deflation solve well-defined (SCHUR_PC_COARSE_REG).
                double reg_rel = coarse_oracle_on ? 0.0 : 1e-6;
                if (const char* rr = std::getenv("SCHUR_PC_COARSE_REG"))
                    reg_rel = std::atof(rr);
                double diag_abs = 0.0;
                for (int i = 0; i < nC; ++i)
                    diag_abs += std::abs(a0[static_cast<std::size_t>(i) +
                                            static_cast<std::size_t>(i) *
                                                static_cast<std::size_t>(nC)]);
                const double shift = (diag_abs / nC) * reg_rel;
                for (int i = 0; i < nC; ++i)
                    a0[static_cast<std::size_t>(i) +
                       static_cast<std::size_t>(i) *
                           static_cast<std::size_t>(nC)] += shift;
                for (int j = 0; j < nC; ++j) {  // 1-norm = max abs column sum
                    double colsum = 0.0;
                    for (int i = 0; i < nC; ++i)
                        colsum += std::abs(a0[static_cast<std::size_t>(i) +
                                              static_cast<std::size_t>(j) *
                                                  static_cast<std::size_t>(nC)]);
                    a0_anorm = std::max(a0_anorm, colsum);
                }
                int a0_dim = nC;
                dgetrf_(&a0_dim, &a0_dim, a0.data(), &a0_dim, a0_ipiv.data(),
                        &a0_info);
                if (a0_info != 0)
                    std::cerr << "do_newton_jfnk_schur: SCHUR_PC_GMRES "
                                 "coarse A0 dgetrf info="
                              << a0_info << " (nC=" << nC << ")" << std::endl;
            }
            const double setup_seconds = elapsed_time(setup_start);

            // --- 5. M^{-1} apply (the right preconditioner), additive. ---
            auto schwarz_inv = [&](const std::vector<double>& r, std::vector<double>& z) {
                z.assign(static_cast<std::size_t>(column_count), 0.0);
                // Per-aggregate Dirichlet least-squares: min ||A_g delta - r_touch||.
                for (const SchwarzLocalSolve& solve : local_solves) {
                    const int m = solve.m;
                    const int n_touch = solve.n_touch;
                    if (m <= 0 || n_touch <= 0)
                        continue;
                    const std::size_t nt = static_cast<std::size_t>(n_touch);
                    std::vector<double> rg(nt);
                    for (int i = 0; i < n_touch; ++i)
                        rg[static_cast<std::size_t>(i)] =
                            r[static_cast<std::size_t>(
                                solve.touch_global[static_cast<std::size_t>(i)])];

                    // Apply Q^T (dormqr, side='L', trans='T') in place on rg.
                    char side = 'L', trans = 'T';
                    int qm = n_touch, qn = 1, qk = m;
                    int qlda = std::max(1, n_touch), qldc = std::max(1, n_touch);
                    int qinfo = 0;
                    double wq = 0.0;
                    int lwork = -1;
                    dormqr_(&side, &trans, &qm, &qn, &qk,
                            const_cast<double*>(solve.a_qr.data()), &qlda,
                            const_cast<double*>(solve.tau.data()), rg.data(),
                            &qldc, &wq, &lwork, &qinfo);  // workspace query
                    lwork = (qinfo == 0) ? static_cast<int>(wq) : std::max(1, n_touch);
                    std::vector<double> work(
                        static_cast<std::size_t>(std::max(1, lwork)));
                    dormqr_(&side, &trans, &qm, &qn, &qk,
                            const_cast<double*>(solve.a_qr.data()), &qlda,
                            const_cast<double*>(solve.tau.data()), rg.data(),
                            &qldc, work.data(), &lwork, &qinfo);

                    // Solve R delta = (Q^T r)[0:m]. R is the top m x m upper
                    // triangle stored in a_qr with leading dim n_touch; copy the
                    // first m entries of rg into a length-m RHS (ldb=m) to avoid
                    // stride confusion.
                    std::vector<double> delta(rg.begin(), rg.begin() + m);
                    char uplo = 'U', tn = 'N', diag = 'N';
                    int rn = m, rnrhs = 1, rlda = std::max(1, n_touch), rldb = m;
                    int rinfo = 0;
                    dtrtrs_(&uplo, &tn, &diag, &rn, &rnrhs,
                            const_cast<double*>(solve.a_qr.data()), &rlda,
                            delta.data(), &rldb, &rinfo);

                    // Scatter into the disjoint column set C_g.
                    for (int k = 0; k < m; ++k)
                        z[static_cast<std::size_t>(
                            solve.cols[static_cast<std::size_t>(k)])] +=
                            delta[static_cast<std::size_t>(k)];
                }
            };
            auto m_inv = [&](const std::vector<double>& r, std::vector<double>& z) {
                schwarz_inv(r, z);
                // Balanced 2-level coarse correction on the DEFLATED residual:
                // z += R0^T A0^{-1} R0 (r - A z_schwarz). The raw-residual additive
                // form was unstable for this nonsymmetric operator (A0 targets
                // near-null modes, so A0^{-1} amplified them and swamped the
                // Schwarz part). One extra spmv per apply.
                if (coarse_on && nC > 0 && a0_info == 0) {
                    std::vector<double> az, defl(static_cast<std::size_t>(column_count)),
                        rc, zc;
                    coo_spmv(z, az);  // A * z_schwarz
                    for (int i = 0; i < column_count; ++i)
                        defl[static_cast<std::size_t>(i)] =
                            r[static_cast<std::size_t>(i)] -
                            az[static_cast<std::size_t>(i)];
                    restrict0(defl, rc);  // rc = R0 (r - A z_schwarz)
                    char c_tn = 'N';
                    int c_dim = nC, c_nrhs = 1, c_ldb = nC, c_info = 0;
                    dgetrs_(&c_tn, &c_dim, &c_nrhs, a0.data(), &c_dim,
                            a0_ipiv.data(), rc.data(), &c_ldb,
                            &c_info);     // rc <- A0^{-1} R0 (r - A z_schwarz)
                    prolong0(rc, zc);     // zc = R0^T A0^{-1} R0 (r - A z_schwarz)
                    for (int i = 0; i < column_count; ++i)
                        z[static_cast<std::size_t>(i)] +=
                            zc[static_cast<std::size_t>(i)];
                }
            };

            auto run_target_probe = [&]() {
                if (!env_flag_enabled("SCHUR_PC_TARGET_PROBE", false))
                    return;

                const auto probe_start = std::chrono::system_clock::now();
                int probe_dim = 24;
                if (const char* raw = std::getenv("SCHUR_PC_TARGET_PROBE_DIM"))
                    probe_dim = std::atoi(raw);
                int probe_report = 4;
                if (const char* raw = std::getenv("SCHUR_PC_TARGET_PROBE_REPORT"))
                    probe_report = std::atoi(raw);
                probe_dim = std::max(2, std::min(probe_dim, column_count - 1));
                probe_report = std::max(1, probe_report);

                std::cout << "=== Schwarz coarse target probe "
                             "(SCHUR_PC_TARGET_PROBE) ===\n";
                if (column_count <= 2) {
                    std::cout << "  target-probe skipped: column_count="
                              << column_count << "\n";
                    return;
                }
                if (!coarse_oracle_on || oracle_right_basis.empty() ||
                    oracle_left_basis.empty()) {
                    std::cout << "  target-probe skipped: raw-A oracle basis is absent"
                              << " coarse_on=" << std::boolalpha << coarse_on
                              << " oracle=" << coarse_oracle_on
                              << " nC=" << nC << "\n";
                    return;
                }

                std::vector<std::vector<double>> oracle_v_q = oracle_right_basis;
                orthonormalize(oracle_v_q);
                std::vector<std::vector<double>> oracle_w_q = oracle_left_basis;
                orthonormalize(oracle_w_q);
                std::vector<std::vector<double>> oracle_av_q;
                oracle_av_q.reserve(oracle_v_q.size());
                for (const std::vector<double>& v : oracle_v_q) {
                    std::vector<double> av;
                    coo_spmv(v, av);
                    oracle_av_q.push_back(std::move(av));
                }
                orthonormalize(oracle_av_q);

                const int ldh = probe_dim + 1;
                std::vector<double> hbar(static_cast<std::size_t>(ldh) *
                                             static_cast<std::size_t>(probe_dim),
                                         0.0);
                std::vector<std::vector<double>> q_basis;
                std::vector<std::vector<double>> z_basis;
                std::vector<std::vector<double>> b_images;
                q_basis.reserve(static_cast<std::size_t>(probe_dim) + 1);
                z_basis.reserve(static_cast<std::size_t>(probe_dim));
                b_images.reserve(static_cast<std::size_t>(probe_dim));

                std::vector<double> q0 = av_coo;
                const double q0_norm = norm2(q0);
                if (q0_norm <= 0.0) {
                    std::cout << "  target-probe skipped: zero RHS/operator seed\n";
                    return;
                }
                for (double& value : q0)
                    value /= q0_norm;
                q_basis.push_back(std::move(q0));

                int actual_dim = 0;
                double terminal_beta = 0.0;
                for (int j = 0; j < probe_dim; ++j) {
                    std::vector<double> z, bq;
                    schwarz_inv(q_basis[static_cast<std::size_t>(j)], z);
                    coo_spmv(z, bq);
                    z_basis.push_back(z);
                    b_images.push_back(bq);

                    std::vector<double> residual = bq;
                    for (int pass = 0; pass < 2; ++pass) {
                        for (int i = 0; i <= j; ++i) {
                            const double alpha =
                                dot(q_basis[static_cast<std::size_t>(i)], residual);
                            hbar[static_cast<std::size_t>(i) +
                                 static_cast<std::size_t>(j) *
                                     static_cast<std::size_t>(ldh)] += alpha;
                            for (int row = 0; row < column_count; ++row)
                                residual[static_cast<std::size_t>(row)] -=
                                    alpha *
                                    q_basis[static_cast<std::size_t>(i)]
                                           [static_cast<std::size_t>(row)];
                        }
                    }
                    terminal_beta = norm2(residual);
                    hbar[static_cast<std::size_t>(j + 1) +
                         static_cast<std::size_t>(j) *
                             static_cast<std::size_t>(ldh)] = terminal_beta;
                    actual_dim = j + 1;
                    if (terminal_beta <= 1e-12)
                        break;
                    for (double& value : residual)
                        value /= terminal_beta;
                    q_basis.push_back(std::move(residual));
                }

                const int q_count = static_cast<int>(q_basis.size());
                double orth_fro_sq = 0.0;
                double orth_max = 0.0;
                for (int i = 0; i < q_count; ++i) {
                    for (int j = 0; j < q_count; ++j) {
                        const double target = (i == j) ? 1.0 : 0.0;
                        const double err =
                            dot(q_basis[static_cast<std::size_t>(i)],
                                q_basis[static_cast<std::size_t>(j)]) -
                            target;
                        orth_fro_sq += err * err;
                        orth_max = std::max(orth_max, std::abs(err));
                    }
                }
                double arnoldi_max_col_rel = 0.0;
                for (int j = 0; j < actual_dim; ++j) {
                    std::vector<double> recon(
                        static_cast<std::size_t>(column_count), 0.0);
                    const int rows = std::min(q_count, actual_dim + 1);
                    for (int i = 0; i < rows; ++i) {
                        const double alpha =
                            hbar[static_cast<std::size_t>(i) +
                                 static_cast<std::size_t>(j) *
                                     static_cast<std::size_t>(ldh)];
                        if (alpha == 0.0)
                            continue;
                        for (int row = 0; row < column_count; ++row)
                            recon[static_cast<std::size_t>(row)] +=
                                alpha *
                                q_basis[static_cast<std::size_t>(i)]
                                       [static_cast<std::size_t>(row)];
                    }
                    for (int row = 0; row < column_count; ++row)
                        recon[static_cast<std::size_t>(row)] -=
                            b_images[static_cast<std::size_t>(j)]
                                    [static_cast<std::size_t>(row)];
                    const double denom =
                        norm2(b_images[static_cast<std::size_t>(j)]);
                    if (denom > 0.0)
                        arnoldi_max_col_rel =
                            std::max(arnoldi_max_col_rel, norm2(recon) / denom);
                }

                std::cout << "  target-probe arnoldi: requested_dim=" << probe_dim
                          << " actual_dim=" << actual_dim
                          << " q_count=" << q_count
                          << " terminal_beta=" << terminal_beta
                          << " orth_fro=" << std::sqrt(orth_fro_sq)
                          << " orth_max=" << orth_max
                          << " max_col_rel=" << arnoldi_max_col_rel << "\n";
                std::cout << "  target-probe oracle ranks:"
                          << " V_rank=" << oracle_v_q.size()
                          << " W_rank=" << oracle_w_q.size()
                          << " AV_rank=" << oracle_av_q.size() << "\n";

                if (actual_dim <= 0) {
                    std::cout << "  target-probe skipped: empty Arnoldi basis\n";
                    return;
                }

                auto combine = [&](const std::vector<std::vector<double>>& basis,
                                   const std::vector<double>& coeff) {
                    std::vector<double> out(
                        static_cast<std::size_t>(column_count), 0.0);
                    const int n =
                        std::min(static_cast<int>(basis.size()),
                                 static_cast<int>(coeff.size()));
                    for (int j = 0; j < n; ++j) {
                        const double alpha = coeff[static_cast<std::size_t>(j)];
                        if (alpha == 0.0)
                            continue;
                        for (int row = 0; row < column_count; ++row)
                            out[static_cast<std::size_t>(row)] +=
                                alpha *
                                basis[static_cast<std::size_t>(j)]
                                     [static_cast<std::size_t>(row)];
                    }
                    return out;
                };
                auto subtract_scaled = [&](std::vector<double>& y,
                                           const std::vector<double>& x,
                                           double alpha) {
                    for (int row = 0; row < column_count; ++row)
                        y[static_cast<std::size_t>(row)] -=
                            alpha * x[static_cast<std::size_t>(row)];
                };

                struct CoarseProbeStats {
                    double before2 = 0.0;
                    double after2 = 0.0;
                    double zs2 = 0.0;
                    double zc2 = 0.0;
                    double azs2 = 0.0;
                    double azc2 = 0.0;
                    int samples = 0;
                    int solve_failures = 0;
                };
                auto accumulate_coarse_stats =
                    [&](const std::vector<double>& r, CoarseProbeStats& stats) {
                        if (nC <= 0 || a0_info != 0 || norm2(r) <= 0.0)
                            return;
                        std::vector<double> zs, azs, defl, rc, zc, azc, rc_after;
                        schwarz_inv(r, zs);
                        coo_spmv(zs, azs);
                        defl.assign(static_cast<std::size_t>(column_count), 0.0);
                        for (int i = 0; i < column_count; ++i)
                            defl[static_cast<std::size_t>(i)] =
                                r[static_cast<std::size_t>(i)] -
                                azs[static_cast<std::size_t>(i)];
                        restrict0(defl, rc);
                        const double before = norm2_sq(rc);
                        if (before <= 0.0)
                            return;
                        char c_tn = 'N';
                        int c_dim = nC, c_nrhs = 1, c_ldb = nC, c_info = 0;
                        dgetrs_(&c_tn, &c_dim, &c_nrhs, a0.data(), &c_dim,
                                a0_ipiv.data(), rc.data(), &c_ldb, &c_info);
                        if (c_info != 0) {
                            ++stats.solve_failures;
                            return;
                        }
                        prolong0(rc, zc);
                        coo_spmv(zc, azc);
                        for (int i = 0; i < column_count; ++i)
                            defl[static_cast<std::size_t>(i)] -=
                                azc[static_cast<std::size_t>(i)];
                        restrict0(defl, rc_after);
                        stats.before2 += before;
                        stats.after2 += norm2_sq(rc_after);
                        stats.zs2 += norm2_sq(zs);
                        stats.zc2 += norm2_sq(zc);
                        stats.azs2 += norm2_sq(azs);
                        stats.azc2 += norm2_sq(azc);
                        ++stats.samples;
                    };

                auto print_candidate =
                    [&](const std::string& label,
                        const std::vector<double>& coeff_re,
                        const std::vector<double>& coeff_im,
                        bool has_lambda,
                        double lambda_re,
                        double lambda_im,
                        double sigma) {
                        std::vector<double> q_re = combine(q_basis, coeff_re);
                        std::vector<double> q_im = combine(q_basis, coeff_im);
                        std::vector<double> z_re = combine(z_basis, coeff_re);
                        std::vector<double> z_im = combine(z_basis, coeff_im);
                        std::vector<double> b_re = combine(b_images, coeff_re);
                        std::vector<double> b_im = combine(b_images, coeff_im);

                        std::vector<double> eig_re = b_re;
                        std::vector<double> eig_im = b_im;
                        if (has_lambda) {
                            subtract_scaled(eig_re, q_re, lambda_re);
                            subtract_scaled(eig_re, q_im, -lambda_im);
                            subtract_scaled(eig_im, q_re, lambda_im);
                            subtract_scaled(eig_im, q_im, lambda_re);
                        }
                        const double eig_denom =
                            std::sqrt(norm2_sq(b_re) + norm2_sq(b_im));
                        const double eig_resid =
                            (has_lambda && eig_denom > 0.0)
                                ? std::sqrt(norm2_sq(eig_re) + norm2_sq(eig_im)) /
                                      eig_denom
                                : -1.0;

                        std::vector<double> defl_re = q_re;
                        std::vector<double> defl_im = q_im;
                        subtract_scaled(defl_re, b_re, 1.0);
                        subtract_scaled(defl_im, b_im, 1.0);
                        const double defl_norm =
                            std::sqrt(norm2_sq(defl_re) + norm2_sq(defl_im));
                        double w_obs = 0.0;
                        if (defl_norm > 0.0) {
                            double seen2 = 0.0;
                            for (const std::vector<double>& w : oracle_w_q) {
                                const double ar = dot(w, defl_re);
                                const double ai = dot(w, defl_im);
                                seen2 += ar * ar + ai * ai;
                            }
                            w_obs = std::sqrt(seen2) / defl_norm;
                        }

                        CoarseProbeStats stats;
                        accumulate_coarse_stats(q_re, stats);
                        accumulate_coarse_stats(q_im, stats);
                        const double restricted_after_ratio =
                            (stats.before2 > 0.0)
                                ? std::sqrt(stats.after2 / stats.before2)
                                : -1.0;
                        const double zc_over_zs =
                            (stats.zs2 > 0.0) ? std::sqrt(stats.zc2 / stats.zs2)
                                               : -1.0;
                        const double azc_over_azs =
                            (stats.azs2 > 0.0) ? std::sqrt(stats.azc2 / stats.azs2)
                                                : -1.0;

                        std::cout << "  target-probe " << label;
                        if (has_lambda)
                            std::cout << " lambda=(" << lambda_re << ","
                                      << lambda_im << ")"
                                      << " eig_resid=" << eig_resid;
                        if (sigma >= 0.0)
                            std::cout << " sigma=" << sigma;
                        std::cout << " sol_V_proj_resid="
                                  << projection_residual_fraction(oracle_v_q, z_re,
                                                                  z_im)
                                  << " input_AV_proj_resid="
                                  << projection_residual_fraction(oracle_av_q, q_re,
                                                                  q_im)
                                  << " W_defl_observability=" << w_obs
                                  << " restricted_after_ratio="
                                  << restricted_after_ratio
                                  << " zc_over_zs=" << zc_over_zs
                                  << " Azc_over_Azs=" << azc_over_azs
                                  << " coarse_samples=" << stats.samples
                                  << " coarse_solve_failures="
                                  << stats.solve_failures << "\n";
                    };

                const int sv_m = actual_dim + 1;
                const int sv_n = actual_dim;
                const int sv_min = std::min(sv_m, sv_n);
                if (sv_min > 0) {
                    std::vector<double> h_svd(
                        static_cast<std::size_t>(sv_m) *
                        static_cast<std::size_t>(sv_n));
                    for (int j = 0; j < sv_n; ++j)
                        for (int i = 0; i < sv_m; ++i)
                            h_svd[static_cast<std::size_t>(i) +
                                  static_cast<std::size_t>(j) *
                                      static_cast<std::size_t>(sv_m)] =
                                hbar[static_cast<std::size_t>(i) +
                                     static_cast<std::size_t>(j) *
                                         static_cast<std::size_t>(ldh)];
                    std::vector<double> singular_values(
                        static_cast<std::size_t>(sv_min));
                    std::vector<double> u(static_cast<std::size_t>(sv_m) *
                                          static_cast<std::size_t>(sv_min));
                    std::vector<double> vt(static_cast<std::size_t>(sv_min) *
                                           static_cast<std::size_t>(sv_n));
                    std::vector<int> iwork(static_cast<std::size_t>(8 * sv_min));
                    char jobz = 'S';
                    int m = sv_m, n = sv_n, lda = sv_m, ldu = sv_m, ldvt = sv_min;
                    int info = 0, lwork = -1;
                    double wk = 0.0;
                    dgesdd_(&jobz, &m, &n, h_svd.data(), &lda,
                            singular_values.data(), u.data(), &ldu, vt.data(), &ldvt,
                            &wk, &lwork, iwork.data(), &info);
                    lwork = (info == 0) ? static_cast<int>(wk) : std::max(1, 8 * sv_min);
                    std::vector<double> work(
                        static_cast<std::size_t>(std::max(1, lwork)));
                    dgesdd_(&jobz, &m, &n, h_svd.data(), &lda,
                            singular_values.data(), u.data(), &ldu, vt.data(), &ldvt,
                            work.data(), &lwork, iwork.data(), &info);
                    std::cout << "  target-probe svd: info=" << info;
                    if (info == 0) {
                        std::cout << " sigma_min="
                                  << singular_values[static_cast<std::size_t>(sv_min - 1)]
                                  << " sigma_max="
                                  << singular_values[static_cast<std::size_t>(0)];
                    }
                    std::cout << "\n";
                    if (info == 0) {
                        // DECIDER: does A*S have a singular-value GAP? A cluster of
                        // small singular values cleanly separated from the bulk is the
                        // only thing a finite coarse space can deflate. A smooth ramp
                        // (the compactified r^-(l+1) exterior decay continuum) means no
                        // coarse space exists -- for raw-A, A*S, or any targeting.
                        // singular_values are descending (dgesdd 'S'): index 0 = max.
                        int gap_at = -1;
                        double gap_ratio = 1.0;
                        for (int i = 0; i + 1 < sv_min; ++i) {
                            const double hi =
                                singular_values[static_cast<std::size_t>(i)];
                            const double lo =
                                singular_values[static_cast<std::size_t>(i + 1)];
                            const double ratio = (lo > 0.0) ? hi / lo : 0.0;
                            if (ratio > gap_ratio) {
                                gap_ratio = ratio;
                                gap_at = i;
                            }
                        }
                        std::cout << "  target-probe AS_spectrum:";
                        for (int i = 0; i < sv_min; ++i)
                            std::cout << " "
                                      << singular_values[static_cast<std::size_t>(i)];
                        std::cout << "\n";
                        std::cout << "  target-probe AS_gap: largest_ratio=" << gap_ratio
                                  << " after_index=" << gap_at
                                  << " implied_coarse_dim="
                                  << (gap_at >= 0 ? sv_min - 1 - gap_at : 0)
                                  << "  (ratio>>1 => deflatable cluster; "
                                     "smooth ramp => no coarse space)\n";
                        // rcond of the raw-A Galerkin coarse operator A0 = W^T A V:
                        // tests the near-singular-A0 amplification hypothesis directly
                        // on the real matrix (oracle path runs A0 unregularized).
                        if (nC > 0 && a0_info == 0 && a0_anorm > 0.0) {
                            std::vector<double> cwork(
                                static_cast<std::size_t>(4 * nC));
                            std::vector<int> ciwork(static_cast<std::size_t>(nC));
                            char cnorm = '1';
                            int c_n = nC, c_lda = nC, c_info = 0;
                            double a0_rcond = 0.0;
                            dgecon_(&cnorm, &c_n, a0.data(), &c_lda, &a0_anorm,
                                    &a0_rcond, cwork.data(), ciwork.data(), &c_info);
                            std::cout << "  target-probe A0_cond: rcond=" << a0_rcond
                                      << " cond="
                                      << (a0_rcond > 0.0 ? 1.0 / a0_rcond : -1.0)
                                      << " anorm=" << a0_anorm
                                      << " dgecon_info=" << c_info << "\n";
                        }
                    }
                    if (info == 0) {
                        const int count = std::min(probe_report, sv_min);
                        for (int k = 0; k < count; ++k) {
                            const int row = sv_min - 1 - k;
                            std::vector<double> coeff_re(
                                static_cast<std::size_t>(actual_dim), 0.0);
                            std::vector<double> coeff_im(
                                static_cast<std::size_t>(actual_dim), 0.0);
                            for (int col = 0; col < actual_dim; ++col)
                                coeff_re[static_cast<std::size_t>(col)] =
                                    vt[static_cast<std::size_t>(row) +
                                       static_cast<std::size_t>(col) *
                                           static_cast<std::size_t>(ldvt)];
                            std::ostringstream label;
                            label << "svslow[" << k << "]";
                            print_candidate(
                                label.str(), coeff_re, coeff_im, false, 0.0, 0.0,
                                singular_values[static_cast<std::size_t>(row)]);
                        }
                    }
                }

                std::vector<double> h_square(
                    static_cast<std::size_t>(actual_dim) *
                    static_cast<std::size_t>(actual_dim));
                for (int j = 0; j < actual_dim; ++j)
                    for (int i = 0; i < actual_dim; ++i)
                        h_square[static_cast<std::size_t>(i) +
                                 static_cast<std::size_t>(j) *
                                     static_cast<std::size_t>(actual_dim)] =
                            hbar[static_cast<std::size_t>(i) +
                                 static_cast<std::size_t>(j) *
                                     static_cast<std::size_t>(ldh)];
                std::vector<double> wr(static_cast<std::size_t>(actual_dim));
                std::vector<double> wi(static_cast<std::size_t>(actual_dim));
                std::vector<double> vl(1, 0.0);
                std::vector<double> vr(static_cast<std::size_t>(actual_dim) *
                                       static_cast<std::size_t>(actual_dim));
                char jobvl = 'N', jobvr = 'V';
                int n = actual_dim, lda = actual_dim, ldvl = 1, ldvr = actual_dim;
                int eig_info = 0, eig_lwork = -1;
                double eig_wk = 0.0;
                dgeev_(&jobvl, &jobvr, &n, h_square.data(), &lda, wr.data(),
                       wi.data(), vl.data(), &ldvl, vr.data(), &ldvr, &eig_wk,
                       &eig_lwork, &eig_info);
                eig_lwork = (eig_info == 0) ? static_cast<int>(eig_wk)
                                            : std::max(1, 4 * actual_dim);
                std::vector<double> eig_work(
                    static_cast<std::size_t>(std::max(1, eig_lwork)));
                dgeev_(&jobvl, &jobvr, &n, h_square.data(), &lda, wr.data(),
                       wi.data(), vl.data(), &ldvl, vr.data(), &ldvr,
                       eig_work.data(), &eig_lwork, &eig_info);
                std::cout << "  target-probe ritz: info=" << eig_info << "\n";
                if (eig_info == 0) {
                    std::vector<int> order;
                    order.reserve(static_cast<std::size_t>(actual_dim));
                    for (int j = 0; j < actual_dim; ++j)
                        if (!(wi[static_cast<std::size_t>(j)] < 0.0))
                            order.push_back(j);
                    std::sort(order.begin(), order.end(), [&](int a, int b) {
                        const double ma =
                            std::hypot(wr[static_cast<std::size_t>(a)],
                                       wi[static_cast<std::size_t>(a)]);
                        const double mb =
                            std::hypot(wr[static_cast<std::size_t>(b)],
                                       wi[static_cast<std::size_t>(b)]);
                        return ma < mb;
                    });
                    const int count =
                        std::min(probe_report, static_cast<int>(order.size()));
                    for (int k = 0; k < count; ++k) {
                        const int eig = order[static_cast<std::size_t>(k)];
                        std::vector<double> coeff_re(
                            static_cast<std::size_t>(actual_dim), 0.0);
                        std::vector<double> coeff_im(
                            static_cast<std::size_t>(actual_dim), 0.0);
                        for (int row = 0; row < actual_dim; ++row)
                            coeff_re[static_cast<std::size_t>(row)] =
                                vr[static_cast<std::size_t>(row) +
                                   static_cast<std::size_t>(eig) *
                                       static_cast<std::size_t>(actual_dim)];
                        if (wi[static_cast<std::size_t>(eig)] > 0.0 &&
                            eig + 1 < actual_dim) {
                            for (int row = 0; row < actual_dim; ++row)
                                coeff_im[static_cast<std::size_t>(row)] =
                                    vr[static_cast<std::size_t>(row) +
                                       static_cast<std::size_t>(eig + 1) *
                                           static_cast<std::size_t>(actual_dim)];
                        }
                        std::ostringstream label;
                        label << "ritz[" << k << "]";
                        print_candidate(label.str(), coeff_re, coeff_im, true,
                                        wr[static_cast<std::size_t>(eig)],
                                        wi[static_cast<std::size_t>(eig)], -1.0);
                    }
                }

                std::cout << "  target-probe setup_s="
                          << elapsed_time(probe_start) << "\n";
            };
            run_target_probe();

            // --- 6. Drive GMRES on the full system. ---
            // Synthetic in-range RHS b = A * x_true (x_true = v_test): the fixture
            // is a converged binary so the nonlinear residual is ~0 (degenerate
            // GMRES RHS). b in range(A) gives a meaningful solve and a verifiable
            // solution x ~ x_true. av_coo already holds A * v_test (parity step).
            std::vector<double>& rhs = av_coo;
            std::vector<double> x_sol(static_cast<std::size_t>(column_count), 0.0);
            double rhs_norm = 0.0;
            for (double bi : rhs)
                rhs_norm += bi * bi;
            rhs_norm = std::sqrt(rhs_norm);

            GmresConfig gmres_config;
            gmres_config.max_iters = 200;
            if (const char* maxit = std::getenv("SCHUR_PC_GMRES_MAXIT"))
                gmres_config.max_iters = std::atoi(maxit);
            gmres_config.tolerance = 1e-8;

            const auto gmres_start = std::chrono::system_clock::now();
            GmresStatus gmres_status = right_preconditioned_gmres(
                rhs, x_sol, coo_spmv, m_inv, gmres_config);
            const double gmres_seconds = elapsed_time(gmres_start);

            std::cout << "=== Schwarz-Dirichlet PC GMRES harness "
                         "(SCHUR_PC_GMRES) ===\n";
            std::cout << "  matvec parity (COO vs do_JX) rel_L2=" << parity_diff
                      << "\n";
            for (const SchwarzLocalSolve& solve : local_solves)
                std::cout << "  block g" << solve.id << " " << solve.label
                          << " m=" << solve.m << " n_touch=" << solve.n_touch
                          << " qr_s=" << solve.setup_s << "\n";
            std::cout << "  coarse nC=" << nC << " a0_info=" << a0_info
                      << " coarse_on=" << std::boolalpha << coarse_on
                      << " oracle=" << coarse_oracle_on
                      << " biorth_info=" << oracle_biorth_info << "\n";
            const double rel_resid =
                (rhs_norm > 0.0) ? gmres_status.residual_norm / rhs_norm
                                 : gmres_status.residual_norm;
            // Solution accuracy vs the known x_true = v_test.
            double sol_err_sq = 0.0, x_true_sq = 0.0;
            for (int i = 0; i < column_count; ++i) {
                const double d = x_sol[static_cast<std::size_t>(i)] -
                                 v_test[static_cast<std::size_t>(i)];
                sol_err_sq += d * d;
                x_true_sq += v_test[static_cast<std::size_t>(i)] *
                             v_test[static_cast<std::size_t>(i)];
            }
            const double sol_err =
                (x_true_sq > 0.0) ? std::sqrt(sol_err_sq / x_true_sq) : 0.0;
            std::cout << "  Schwarz-GMRES: converged=" << std::boolalpha
                      << gmres_status.converged
                      << " iters=" << gmres_status.iterations
                      << " rel_resid=" << rel_resid
                      << " sol_err=" << sol_err
                      << " setup_s=" << setup_seconds
                      << " gmres_s=" << gmres_seconds << std::endl;
            std::cout.flush();

            MPI_Finalize();
            std::exit(0);
        }

        int s_dim = n_interface;
        std::vector<int> s_ipiv(static_cast<std::size_t>(n_interface));
        int s_info = 0;
        const auto t_s_factor = std::chrono::system_clock::now();
        dgetrf_(&s_dim, &s_dim, schur.data(), &s_dim, s_ipiv.data(), &s_info);
        const double s_factor_seconds = elapsed_time(t_s_factor);

        const double formation_seconds =
            per_aggregate_factor_seconds + solve_seconds + gemm_seconds;
        const double total_seconds = formation_seconds + s_factor_seconds;
        const bool probe_ok =
            (cross_aggregate_bulk_entries == 0) && (singular_blocks == 0) &&
            (s_info == 0);

        std::cout << "SCHUR_PROBE manual (step 0):\n"
                  << "  n=" << column_count << " nnz=" << nnz
                  << " size_schur=" << n_interface
                  << " bulk_cols=" << census.bulk_cols
                  << " aggregates=" << n_aggregates
                  << " layout=" << partition.source << "\n"
                  << "  coo_build_s=" << coo_build_seconds
                  << " peraggregate_ABB_factor_s=" << per_aggregate_factor_seconds
                  << " solve_ABBinv_ABI_s=" << solve_seconds << "\n"
                  << "  gemm_accumulate_S_s=" << gemm_seconds
                  << " dense_S_factor_s=" << s_factor_seconds
                  << " formation_total_s=" << formation_seconds
                  << " grand_total_s=" << total_seconds << "\n"
                  << "  cross_aggregate_bulk_entries=" << cross_aggregate_bulk_entries
                  << " singular_blocks=" << singular_blocks
                  << " S_dgetrf_info=" << s_info
                  << " peak_rss_GB=" << peak_rss_gb();
        std::cout << "\n"
                  << "  PROBE: " << (probe_ok ? "PASS" : "FAIL")
                  << "  (requires cross_aggregate_bulk_entries=0, singular_blocks=0, "
                     "S_dgetrf_info=0)"
                  << std::endl;
        std::cout.flush();

        MPI_Finalize();
        std::exit(probe_ok ? 0 : 2);
#else
        (void)precision;
        (void)config;
        Array<double> residual(sec_member());
        error = infinity_norm(residual);
        int rank = 0;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        if (rank == 0)
            std::cerr << "do_newton_jfnk_schur: built without MUMPS.\n";
        return false;
#endif // CELEPHAIS_USE_MUMPS
    }

} // namespace Kadath
