/*
    Added 2026 -- diagnostic only, default off.

    y = 0 reflection-parity mass probe for the assembled Jacobian.

    The nosym domains store the azimuthal dependence in a unified COSSIN series:
    index k even holds cos((k/2).phi) and index k odd holds sin(((k-1)/2).phi)
    (summation_1d.cpp:176).  The domain chart determines whether y -> -y maps
    phi to -phi or pi-phi and grades each coefficient accordingly.  Combined
    with the field grading of the boosted-star configuration (spin axis in the
    x-z plane) this splits every unknown into a symmetric sector S and an
    antisymmetric sector A.  If the configuration really is parity graded, the
    Jacobian block-decouples: no row couples an S column to an A column.

    This probe measures the violation three ways:
      * predicted sector masses, using the field grading below;
      * a grading-free structural 2-colouring (union-find over the stored
        nonzeros), which decides decoupling without trusting the grading;
      * per-domain phi-column occupancy, resolving whether phi is an isolated
        identity block on the vacuum domains.
*/

#include "Linear_algebra/jacobian_parity_mass.hpp"

#include "Linear_algebra/jacobian_parity_mask.hpp"

#include "For_Kadath/Array/exceptions.hpp"
#include "For_Kadath/Domain/bispheric.hpp"
#include "For_Kadath/Domain/bispheric_nosym.hpp"
#include "For_Kadath/Space/space.hpp"
#include "For_Kadath/System_of_eqs/Jacobian/column_types.hpp"
#include "For_Kadath/Tensor/tensor.hpp"
#include "For_Kadath/Term_eq/term_eq.hpp"
#include "For_Kadath/Val_domain/val_domain.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

namespace Kadath
{
    namespace
    {
        // Variable names reach the column map padded (add_var normalizes to a
        // fixed-width token), so every comparison has to trim first.
        std::string trim_ascii_space(const std::string& text)
        {
            const std::size_t first = text.find_first_not_of(" \t\n\r");
            if (first == std::string::npos)
                return {};
            const std::size_t last = text.find_last_not_of(" \t\n\r");
            return text.substr(first, last - first + 1);
        }

        class UnionFind
        {
          public:
            explicit UnionFind(std::size_t count) : parent_(count)
            {
                std::iota(parent_.begin(), parent_.end(), std::size_t{0});
            }

            std::size_t find(std::size_t node)
            {
                while (parent_[node] != node) {
                    parent_[node] = parent_[parent_[node]];
                    node = parent_[node];
                }
                return node;
            }

            void unite(std::size_t left, std::size_t right)
            {
                const std::size_t left_root = find(left);
                const std::size_t right_root = find(right);
                if (left_root != right_root)
                    parent_[left_root] = right_root;
            }

          private:
            std::vector<std::size_t> parent_;
        };

        struct SectorAccumulator
        {
            double max_abs = 0.0;
            double sum_abs = 0.0;
            long long count = 0;

            void add(double value)
            {
                const double magnitude = std::fabs(value);
                if (magnitude > max_abs)
                    max_abs = magnitude;
                sum_abs += magnitude;
                ++count;
            }
        };

        std::size_t checked_domain_bucket(int domain, int domain_count)
        {
            if (domain == -1)
                return static_cast<std::size_t>(domain_count);
            if (domain < 0 || domain >= domain_count)
                KADATH_THROW("Jacobian domain census metadata domain out of range");
            return static_cast<std::size_t>(domain);
        }

        const char* domain_kind(const Domain* domain)
        {
            if (dynamic_cast<const Domain_bispheric_chi_first_nosym*>(domain) != nullptr)
                return "bispheric_chi_first_nosym";
            if (dynamic_cast<const Domain_bispheric_rect_nosym*>(domain) != nullptr)
                return "bispheric_rect_nosym";
            if (dynamic_cast<const Domain_bispheric_eta_first_nosym*>(domain) != nullptr)
                return "bispheric_eta_first_nosym";
            if (dynamic_cast<const Domain_bispheric_chi_first*>(domain) != nullptr)
                return "bispheric_chi_first";
            if (dynamic_cast<const Domain_bispheric_rect*>(domain) != nullptr)
                return "bispheric_rect";
            if (dynamic_cast<const Domain_bispheric_eta_first*>(domain) != nullptr)
                return "bispheric_eta_first";
            return "other";
        }

        void emit_domain_census(
            std::ostream& out,
            const Space& space,
            const jacobian_parity_mass_detail::DomainCensus& census)
        {
            const int domain_count = census.domain_count;
            const std::size_t bucket_count = static_cast<std::size_t>(domain_count + 1);
            out << "domain_count," << domain_count << '\n';
            out << "domain_census_bucket_count," << bucket_count << '\n';
            out << "domain_kind,-1,global\n";
            for (int domain = 0; domain < domain_count; ++domain)
                out << "domain_kind," << domain << ','
                    << domain_kind(space.get_domain(domain)) << '\n';

            for (std::size_t bucket = 0; bucket < bucket_count; ++bucket) {
                const int domain = bucket == static_cast<std::size_t>(domain_count)
                                       ? -1
                                       : static_cast<int>(bucket);
                out << "domain_rows," << domain << ','
                    << census.row_dimensions[bucket] << '\n';
                out << "domain_columns," << domain << ','
                    << census.column_dimensions[bucket] << '\n';
            }

            long long total_entries = 0;
            long long total_matching_entries = 0;
            for (std::size_t row_bucket = 0; row_bucket < bucket_count; ++row_bucket) {
                const int row_domain =
                    row_bucket == static_cast<std::size_t>(domain_count)
                        ? -1
                        : static_cast<int>(row_bucket);
                for (std::size_t column_bucket = 0; column_bucket < bucket_count;
                     ++column_bucket) {
                    const int column_domain =
                        column_bucket == static_cast<std::size_t>(domain_count)
                            ? -1
                            : static_cast<int>(column_bucket);
                    const std::size_t index = row_bucket * bucket_count + column_bucket;
                    const long long entries = census.stored_coo_entries[index];
                    const long long matching_entries =
                        census.matching_stored_coo_entries[index];
                    total_entries += entries;
                    total_matching_entries += matching_entries;
                    out << "domain_block_entries," << row_domain << ','
                        << column_domain << ',' << entries << '\n';
                    out << "matching_block_entries," << row_domain << ','
                        << column_domain << ',' << matching_entries << '\n';
                }
            }
            out << "domain_block_entries_total," << total_entries << '\n';
            out << "matching_block_entries_total," << total_matching_entries << '\n';
        }
    } // namespace

    namespace jacobian_parity_mass_detail
    {
        DomainCensus build_domain_census(
            int domain_count,
            const std::vector<RowMetadata>& rows,
            const std::vector<ColumnMetadata>& columns,
            const AssembledJacobianCoo& coo)
        {
            if (domain_count < 0 || coo.n < 0 || coo.nnz < 0)
                KADATH_THROW("Jacobian domain census has negative dimensions");
            const std::size_t n = static_cast<std::size_t>(coo.n);
            const std::size_t nnz = static_cast<std::size_t>(coo.nnz);
            if (static_cast<long long>(nnz) != coo.nnz || rows.size() != n ||
                columns.size() != n || coo.irn.size() != nnz ||
                coo.jcn.size() != nnz) {
                KADATH_THROW("Jacobian domain census metadata/COO size mismatch");
            }

            DomainCensus census;
            census.domain_count = domain_count;
            const std::size_t bucket_count = static_cast<std::size_t>(domain_count + 1);
            census.row_dimensions.assign(bucket_count, 0);
            census.column_dimensions.assign(bucket_count, 0);
            census.stored_coo_entries.assign(bucket_count * bucket_count, 0);
            census.matching_stored_coo_entries.assign(
                bucket_count * bucket_count, 0);

            for (std::size_t row = 0; row < n; ++row) {
                if (rows[row].row != static_cast<int>(row))
                    KADATH_THROW("Jacobian domain census row metadata is not ordered");
                ++census.row_dimensions[
                    checked_domain_bucket(rows[row].dom, domain_count)];
            }
            for (std::size_t column = 0; column < n; ++column) {
                if (columns[column].column != static_cast<int>(column))
                    KADATH_THROW("Jacobian domain census column metadata is not ordered");
                ++census.column_dimensions[
                    checked_domain_bucket(columns[column].domain, domain_count)];
            }

            for (std::size_t entry = 0; entry < nnz; ++entry) {
                const int row = coo.irn[entry] - 1;
                const int column = coo.jcn[entry] - 1;
                if (row < 0 || row >= coo.n || column < 0 || column >= coo.n)
                    KADATH_THROW("Jacobian domain census COO coordinate out of range");
                const std::size_t row_bucket = checked_domain_bucket(
                    rows[static_cast<std::size_t>(row)].dom, domain_count);
                const std::size_t column_bucket = checked_domain_bucket(
                    columns[static_cast<std::size_t>(column)].domain, domain_count);
                const std::size_t index = row_bucket * bucket_count + column_bucket;
                ++census.stored_coo_entries[index];
                if (rows[static_cast<std::size_t>(row)].taxonomy ==
                    RowTaxonomy::TauMatch) {
                    ++census.matching_stored_coo_entries[index];
                }
            }

            if (std::accumulate(census.row_dimensions.begin(),
                                census.row_dimensions.end(), 0LL) != coo.n ||
                std::accumulate(census.column_dimensions.begin(),
                                census.column_dimensions.end(), 0LL) != coo.n ||
                std::accumulate(census.stored_coo_entries.begin(),
                                census.stored_coo_entries.end(), 0LL) != coo.nnz) {
                KADATH_THROW("Jacobian domain census count invariant failed");
            }
            return census;
        }
    } // namespace jacobian_parity_mass_detail

    void jacobian_parity_mass_report(System_of_eqs& system,
                                     const AssembledJacobianCoo& coo,
                                     const std::string& report_path)
    {
        const int n = coo.n;
        std::vector<ColumnInfo> column_map;
        system.build_column_map(column_map, false);
        if (static_cast<int>(column_map.size()) != n)
            KADATH_THROW("jacobian_parity_mass_report: column map size != n");
        // ---- column signatures, shared with the production mask ----------
        const JacobianParityColumnGrading grading =
            grade_jacobian_parity_columns(system);
        const std::string disable_reason =
            jacobian_parity_column_grading_disable_reason(grading);
        if (!disable_reason.empty()) {
            std::ofstream out(report_path);
            if (!out)
                KADATH_THROW("Could not open JACOBIAN_PARITY_MASS output");
            out << "# y=0 reflection-parity mass probe\n";
            out << "status,unavailable\n";
            out << "reason," << disable_reason << '\n';
            out.close();
            if (!out)
                KADATH_THROW("Could not finish JACOBIAN_PARITY_MASS output");
            std::cout << "PARITY MASS PROBE: unavailable, " << disable_reason
                      << " report=" << report_path << std::endl;
            return;
        }
        TaggedJacobianMetadata tagged_metadata;
        system.build_tagged_jacobian_metadata(
            tagged_metadata, /*include_row_incidence=*/false);
        const auto domain_census =
            jacobian_parity_mass_detail::build_domain_census(
                system.get_space().get_nbr_domains(), tagged_metadata.rows,
                tagged_metadata.columns, coo);
        const std::vector<signed char>& signature = grading.sector;
        const std::vector<int>& phi_index = grading.phi_index;
        const std::vector<int>& component_index = grading.component_index;

        // ---- pass 1: per-row mass on each side, derive the row grading -----
        std::vector<double> row_mass_symmetric(static_cast<std::size_t>(n), 0.0);
        std::vector<double> row_mass_antisymmetric(static_cast<std::size_t>(n), 0.0);
        double max_abs_overall = 0.0;
        for (long long entry = 0; entry < coo.nnz; ++entry) {
            const std::size_t index = static_cast<std::size_t>(entry);
            const int row = coo.irn[index] - 1;
            const int column = coo.jcn[index] - 1;
            const double magnitude = std::fabs(coo.a[index]);
            if (magnitude > max_abs_overall)
                max_abs_overall = magnitude;
            if (signature[static_cast<std::size_t>(column)] > 0)
                row_mass_symmetric[static_cast<std::size_t>(row)] += magnitude;
            else
                row_mass_antisymmetric[static_cast<std::size_t>(row)] += magnitude;
        }
        // A row belongs to the sector that carries most of its mass; the
        // remainder is the irreducible parity violation of that row.  This is
        // the sharpest available row grading: it is derived from the matrix
        // rather than assumed, so it cannot manufacture a false violation.
        std::vector<signed char> row_signature(static_cast<std::size_t>(n), 0);
        for (int row = 0; row < n; ++row) {
            row_signature[static_cast<std::size_t>(row)] =
                (row_mass_symmetric[static_cast<std::size_t>(row)] >=
                 row_mass_antisymmetric[static_cast<std::size_t>(row)])
                    ? static_cast<signed char>(1)
                    : static_cast<signed char>(-1);
        }
        const JacobianParityRowPrediction descriptor_prediction =
            predict_jacobian_parity_rows(system);
        const JacobianParityRowOracleComparison descriptor_comparison =
            compare_jacobian_parity_row_prediction(descriptor_prediction,
                                                    row_signature);
        const bool descriptor_full_exact =
            descriptor_comparison.whole_fixture_covered &&
            descriptor_comparison.exact_on_covered_rows &&
            descriptor_comparison.compared_rows == n &&
            descriptor_comparison.mismatched_rows == 0 &&
            descriptor_comparison.failure_reason.empty();
        const char* descriptor_status =
            descriptor_full_exact
                ? "exact"
                : (!descriptor_comparison.failure_reason.empty()
                       ? "invalid"
                       : (descriptor_comparison.mismatched_rows > 0
                              ? "mismatch"
                              : (descriptor_prediction.all_rows_available
                                     ? "incomplete-comparison"
                                     : "partial")));

        // ---- pass 2: sector masses and the largest cross entry -------------
        SectorAccumulator sector_ss;
        SectorAccumulator sector_sa;
        SectorAccumulator sector_as;
        SectorAccumulator sector_aa;
        double worst_cross = 0.0;
        int worst_cross_row = -1;
        int worst_cross_column = -1;
        for (long long entry = 0; entry < coo.nnz; ++entry) {
            const std::size_t index = static_cast<std::size_t>(entry);
            const int row = coo.irn[index] - 1;
            const int column = coo.jcn[index] - 1;
            const double value = coo.a[index];
            const bool row_symmetric = row_signature[static_cast<std::size_t>(row)] > 0;
            const bool column_symmetric = signature[static_cast<std::size_t>(column)] > 0;
            if (row_symmetric && column_symmetric)
                sector_ss.add(value);
            else if (row_symmetric && !column_symmetric)
                sector_sa.add(value);
            else if (!row_symmetric && column_symmetric)
                sector_as.add(value);
            else
                sector_aa.add(value);
            if (row_symmetric != column_symmetric && std::fabs(value) > worst_cross) {
                worst_cross = std::fabs(value);
                worst_cross_row = row;
                worst_cross_column = column;
            }
        }

        // ---- grading-free structural 2-colouring ---------------------------
        // Rows are nodes [0,n), columns are nodes [n,2n).  Every retained
        // nonzero forces its row and column into the same sector.  If the
        // Jacobian really block-decouples there must be at least two big
        // components, and no choice of grading can beat that partition.  The
        // sweep is needed because the assembler stores entries all the way down
        // to a 1e-15 drop tolerance: parity that holds to 1e-10 still shows a
        // single component when noise-level entries are counted as couplings.
        struct StructuralResult {
            double relative_threshold = 0.0;
            long long components = 0;
            long long largest = 0;
            long long second = 0;
            long long signature_conflicts = 0;
        };
        const double relative_thresholds[] = {0.0,  1e-14, 1e-12,
                                              1e-10, 1e-8,  1e-6};
        std::vector<StructuralResult> structural_results;
        for (double relative_threshold : relative_thresholds) {
            const double cutoff = relative_threshold * max_abs_overall;
            UnionFind components(static_cast<std::size_t>(2 * n));
            for (long long entry = 0; entry < coo.nnz; ++entry) {
                const std::size_t index = static_cast<std::size_t>(entry);
                if (std::fabs(coo.a[index]) <= cutoff)
                    continue;
                components.unite(static_cast<std::size_t>(coo.irn[index] - 1),
                                 static_cast<std::size_t>(n + coo.jcn[index] - 1));
            }
            std::vector<long long> component_size(
                static_cast<std::size_t>(2 * n), 0);
            for (int node = 0; node < 2 * n; ++node)
                ++component_size[components.find(static_cast<std::size_t>(node))];
            StructuralResult result;
            result.relative_threshold = relative_threshold;
            for (int node = 0; node < 2 * n; ++node) {
                const long long size = component_size[static_cast<std::size_t>(node)];
                if (size <= 1)
                    continue;
                ++result.components;
                if (size > result.largest) {
                    result.second = result.largest;
                    result.largest = size;
                } else if (size > result.second) {
                    result.second = size;
                }
            }
            // Does the predicted colouring ever disagree inside one component?
            std::vector<signed char> component_signature(
                static_cast<std::size_t>(2 * n), 0);
            for (int column = 0; column < n; ++column) {
                const std::size_t root =
                    components.find(static_cast<std::size_t>(n + column));
                const signed char value = signature[static_cast<std::size_t>(column)];
                if (component_signature[root] == 0)
                    component_signature[root] = value;
                else if (component_signature[root] != value)
                    ++result.signature_conflicts;
            }
            structural_results.push_back(result);
        }

        // ---- phi-column occupancy per domain -------------------------------
        const int domain_count = system.get_space().get_nbr_domains();
        std::vector<long long> phi_column_count(
            static_cast<std::size_t>(domain_count), 0);
        std::vector<long long> phi_column_nnz(
            static_cast<std::size_t>(domain_count), 0);
        std::vector<int> phi_domain_of_column(static_cast<std::size_t>(n), -1);
        for (int column = 0; column < n; ++column) {
            const ColumnInfo& info = column_map[static_cast<std::size_t>(column)];
            if (trim_ascii_space(info.var_name) != "phi" || info.domain < 0 ||
                info.domain >= domain_count)
                continue;
            phi_domain_of_column[static_cast<std::size_t>(column)] = info.domain;
            ++phi_column_count[static_cast<std::size_t>(info.domain)];
        }
        for (long long entry = 0; entry < coo.nnz; ++entry) {
            const std::size_t index = static_cast<std::size_t>(entry);
            const int domain =
                phi_domain_of_column[static_cast<std::size_t>(coo.jcn[index] - 1)];
            if (domain >= 0)
                ++phi_column_nnz[static_cast<std::size_t>(domain)];
        }

        // ---- report ---------------------------------------------------------
        std::ofstream out(report_path);
        if (!out)
            KADATH_THROW("Could not open JACOBIAN_PARITY_MASS output");
        out << std::setprecision(17);
        out << "# y=0 reflection-parity mass probe\n";
        out << "n," << n << '\n';
        out << "nnz," << coo.nnz << '\n';
        out << "drop_tol," << coo.drop_tol_used << '\n';
        out << "max_abs_overall," << max_abs_overall << '\n';

        long long symmetric_columns = 0;
        for (int column = 0; column < n; ++column)
            if (signature[static_cast<std::size_t>(column)] > 0)
                ++symmetric_columns;
        long long symmetric_rows = 0;
        for (int row = 0; row < n; ++row)
            if (row_signature[static_cast<std::size_t>(row)] > 0)
                ++symmetric_rows;
        out << "columns_S," << symmetric_columns << '\n';
        out << "columns_A," << n - symmetric_columns << '\n';
        out << "rows_S," << symmetric_rows << '\n';
        out << "rows_A," << n - symmetric_rows << '\n';
        out << "ungraded_columns," << grading.ungraded_columns << '\n';
        out << "mixed_phi_columns," << grading.mixed_phi_columns << '\n';
        out << "unsupported_phi_basis_columns,"
            << grading.unsupported_phi_basis_columns << '\n';
        out << "descriptor_oracle_status," << descriptor_status << '\n';
        out << "descriptor_rows_total," << n << '\n';
        out << "descriptor_rows_compared,"
            << descriptor_comparison.compared_rows << '\n';
        out << "descriptor_rows_unavailable,"
            << descriptor_comparison.unavailable_rows << '\n';
        out << "descriptor_rows_mismatched,"
            << descriptor_comparison.mismatched_rows << '\n';
        out << "descriptor_first_mismatch,"
            << descriptor_comparison.first_mismatch << '\n';
        out << "descriptor_all_rows_available,"
            << (descriptor_prediction.all_rows_available ? 1 : 0) << '\n';
        out << "descriptor_whole_fixture_covered,"
            << (descriptor_comparison.whole_fixture_covered ? 1 : 0) << '\n';
        out << "descriptor_exact_on_covered_rows,"
            << (descriptor_comparison.exact_on_covered_rows ? 1 : 0) << '\n';
        out << "descriptor_unavailable_rows,"
            << descriptor_prediction.unavailable_rows << '\n';
        out << "descriptor_ungraded_rows,"
            << descriptor_prediction.ungraded_rows << '\n';
        out << "descriptor_unsupported_phi_basis_rows,"
            << descriptor_prediction.unsupported_phi_basis_rows << '\n';
        out << "descriptor_failure_reason,"
            << descriptor_comparison.failure_reason << '\n';
        if (descriptor_comparison.unavailable_rows > 0) {
            std::vector<System_of_eqs::RowMetadata> row_metadata;
            system.classify_equation_row_metadata(row_metadata);
            if (row_metadata.size() != static_cast<std::size_t>(n))
                KADATH_THROW("Jacobian parity descriptor oracle row metadata size mismatch");
            out << "# descriptor ungraded: row,matrix,eq_index,eq_local_row,"
                   "domain,equation_type,owner\n";
            for (int row = 0; row < n; ++row) {
                if (descriptor_prediction.sector[static_cast<std::size_t>(row)] != 0)
                    continue;
                const System_of_eqs::RowMetadata& metadata =
                    row_metadata[static_cast<std::size_t>(row)];
                out << "descriptor_ungraded," << row << ','
                    << static_cast<int>(row_signature[static_cast<std::size_t>(row)])
                    << ',' << metadata.eq_index << ',' << metadata.eq_local_row
                    << ',' << metadata.dom << ',' << metadata.equation_type << ','
                    << metadata.owner_var_name << '\n';
            }
        }
        if (descriptor_comparison.mismatched_rows > 0) {
            std::vector<System_of_eqs::RowMetadata> row_metadata;
            system.classify_equation_row_metadata(row_metadata);
            if (row_metadata.size() != static_cast<std::size_t>(n))
                KADATH_THROW("Jacobian parity descriptor oracle row metadata size mismatch");
            out << "# descriptor mismatches: row,predicted,matrix,eq_index,"
                   "eq_local_row,domain,equation_type,owner\n";
            for (int row = 0; row < n; ++row) {
                const signed char predicted =
                    descriptor_prediction.sector[static_cast<std::size_t>(row)];
                if (predicted == 0 ||
                    predicted == row_signature[static_cast<std::size_t>(row)])
                    continue;
                const System_of_eqs::RowMetadata& metadata =
                    row_metadata[static_cast<std::size_t>(row)];
                out << "descriptor_mismatch," << row << ','
                    << static_cast<int>(predicted) << ','
                    << static_cast<int>(row_signature[static_cast<std::size_t>(row)])
                    << ',' << metadata.eq_index << ',' << metadata.eq_local_row
                    << ',' << metadata.dom << ',' << metadata.equation_type << ','
                    << metadata.owner_var_name << '\n';
            }
        }
        out << "ungraded_names,";
        for (const std::string& name : grading.ungraded_names)
            out << name << ' ';
        out << '\n';

        const auto emit_sector = [&out](const char* label,
                                        const SectorAccumulator& sector) {
            out << "max_" << label << ',' << sector.max_abs << '\n';
            out << "sum_" << label << ',' << sector.sum_abs << '\n';
            out << "nnz_" << label << ',' << sector.count << '\n';
        };
        emit_sector("J_SS", sector_ss);
        emit_sector("J_SA", sector_sa);
        emit_sector("J_AS", sector_as);
        emit_sector("J_AA", sector_aa);
        out << "max_cross," << worst_cross << '\n';
        out << "max_cross_over_max_overall,"
            << (max_abs_overall > 0.0 ? worst_cross / max_abs_overall : 0.0) << '\n';
        out << "max_cross_row," << worst_cross_row << '\n';
        out << "max_cross_column," << worst_cross_column << '\n';
        if (worst_cross_column >= 0) {
            const ColumnInfo& info =
                column_map[static_cast<std::size_t>(worst_cross_column)];
            out << "max_cross_column_var," << info.var_name << '\n';
            out << "max_cross_column_domain," << info.domain << '\n';
            out << "max_cross_column_component,"
                << component_index[static_cast<std::size_t>(worst_cross_column)] << '\n';
            out << "max_cross_column_phi_index,"
                << phi_index[static_cast<std::size_t>(worst_cross_column)] << '\n';
        }
        // Rows ranked by irreducible violation, so the carrier is named even
        // when the predicted grading is imperfect.
        std::vector<int> rows_by_violation(static_cast<std::size_t>(n));
        std::iota(rows_by_violation.begin(), rows_by_violation.end(), 0);
        const auto violation_of = [&](int row) {
            return std::min(row_mass_symmetric[static_cast<std::size_t>(row)],
                            row_mass_antisymmetric[static_cast<std::size_t>(row)]);
        };
        std::sort(rows_by_violation.begin(), rows_by_violation.end(),
                  [&](int left, int right) {
                      return violation_of(left) > violation_of(right);
                  });
        long long rows_with_violation = 0;
        for (int row = 0; row < n; ++row)
            if (violation_of(row) > 0.0)
                ++rows_with_violation;
        out << "rows_with_cross_mass," << rows_with_violation << '\n';
        out << "# top violating rows: row,cross_mass,dominant_mass\n";
        for (int rank = 0; rank < std::min(20, n); ++rank) {
            const int row = rows_by_violation[static_cast<std::size_t>(rank)];
            out << "row_violation," << row << ',' << violation_of(row) << ','
                << std::max(row_mass_symmetric[static_cast<std::size_t>(row)],
                            row_mass_antisymmetric[static_cast<std::size_t>(row)])
                << '\n';
        }

        out << "# structural 2-colouring: rel_threshold,components,largest,"
               "second,predicted_colouring_conflicts (2n = "
            << 2 * n << " nodes)\n";
        for (const StructuralResult& result : structural_results) {
            out << "structural," << result.relative_threshold << ','
                << result.components << ',' << result.largest << ','
                << result.second << ',' << result.signature_conflicts << '\n';
        }

        for (int domain = 0; domain < domain_count; ++domain) {
            out << "phi_columns_domain_" << domain << ','
                << phi_column_count[static_cast<std::size_t>(domain)] << '\n';
            out << "phi_column_nnz_domain_" << domain << ','
                << phi_column_nnz[static_cast<std::size_t>(domain)] << '\n';
        }
        emit_domain_census(out, system.get_space(), domain_census);
        out.close();
        if (!out)
            KADATH_THROW("Could not finish JACOBIAN_PARITY_MASS output");

        std::cout << "PARITY MASS PROBE: n=" << n << " nnz=" << coo.nnz
                  << " max|J|=" << max_abs_overall
                  << " max_cross=" << worst_cross
                  << " ratio="
                  << (max_abs_overall > 0.0 ? worst_cross / max_abs_overall : 0.0)
                  << " ungraded_columns=" << grading.ungraded_columns
                  << " descriptor_status=" << descriptor_status
                  << " descriptor_rows="
                  << descriptor_comparison.compared_rows << '/' << n
                  << " descriptor_unavailable="
                  << descriptor_comparison.unavailable_rows
                  << " descriptor_mismatches="
                  << descriptor_comparison.mismatched_rows
                  << " report=" << report_path << std::endl;
        if (descriptor_prediction.all_rows_available &&
            !descriptor_full_exact) {
            KADATH_THROW(
                "Jacobian parity descriptor oracle failed whole-fixture agreement");
        }
    }
} // namespace Kadath
