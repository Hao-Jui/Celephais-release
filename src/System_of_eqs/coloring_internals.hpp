#pragma once

#include "For_Kadath/System_of_eqs/system_of_eqs.hpp"
#include "For_Kadath/Array/exceptions.hpp"
#include "For_Kadath/Domain/bispheric.hpp"
#include "For_Kadath/Domain/bispheric_nosym.hpp"
#include "For_Kadath/Domain/spheric.hpp"
#include "For_Kadath/Tensor/tensor.hpp"
#include "For_Kadath/Term_eq/term_eq.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <map>
#include <numeric>
#include <set>
#include <sstream>
#include <string>
#include <vector>

namespace Kadath::coloring_internals
{
    inline constexpr const char* kColoringNeedsSecMember =
        "Column coloring analysis unavailable: call sec_member() first.";

    struct SupportSummary {
        int min_rows = 0;
        int max_rows = 0;
        double avg_rows = 0.0;
    };

    SupportSummary summarize_support(const std::vector<std::set<int>>& rows_per_column);
    void print_support_summary(std::ostream& os, const char* label, const SupportSummary& summary);

    struct EligibilitySummary {
        int eligible_groups = 0;
        int eligible_columns = 0;
        int single_groups = 0;
        int single_columns = 0;
        int multi_domain_groups = 0;
        int multi_domain_columns = 0;
        int var_domain_groups = 0;
        int var_domain_columns = 0;
        int var_double_groups = 0;
        int var_double_columns = 0;
    };

    struct MetadataSupportSummary {
        int groups = 0;
        int columns = 0;
        int single_groups = 0;
        int single_columns = 0;
        int safe_multi_groups = 0;
        int safe_multi_columns = 0;
        int unsafe_multi_groups = 0;
        int unsafe_multi_columns = 0;
        int truncated_groups = 0;
        int total_row_collisions = 0;
        int max_row_collisions_in_group = 0;
    };

    using SparseColumnMap = std::map<int, double>;

    struct SparseColumnComparison {
        int reference_nnz = 0;
        int decoded_nnz = 0;
        int missing_rows = 0;
        int extra_rows = 0;
        double max_abs_error = 0.0;
        double max_relative_error = 0.0;
        int first_mismatch_row = -1;
        double first_reference_value = 0.0;
        double first_decoded_value = 0.0;
        bool passed = true;
    };

    using SparseCooKey = std::pair<int, int>;
    using SparseCooMap = std::map<SparseCooKey, double>;

    struct SparseCooComparison {
        int reference_nnz = 0;
        int seeded_nnz = 0;
        int missing_entries = 0;
        int extra_entries = 0;
        double max_abs_error = 0.0;
        double max_relative_error = 0.0;
        int first_mismatch_row = -1;
        int first_mismatch_col = -1;
        double first_reference_value = 0.0;
        double first_seeded_value = 0.0;
        bool passed = true;
    };

    EligibilitySummary summarize_seeded_eligibility(const std::vector<std::vector<int>>& groups,
                                                    const std::vector<ColumnInfo>& column_map);
    void print_eligibility_summary(std::ostream& os, const EligibilitySummary& summary);

    MetadataSupportSummary summarize_metadata_group_actual_support(
        const std::vector<std::vector<int>>& groups,
        const std::vector<std::set<int>>& rows_per_column,
        const std::vector<char>& selected_columns);
    void print_metadata_support_summary(std::ostream& os, const MetadataSupportSummary& summary);

    SparseColumnMap sparse_map_from_array(const Array<double>& column, double drop_tol);
    SparseColumnComparison compare_sparse_columns(const SparseColumnMap& reference,
                                                  const SparseColumnMap& decoded,
                                                  double rtol,
                                                  double atol);
    SparseCooComparison compare_sparse_coo(const SparseCooMap& reference,
                                           const SparseCooMap& seeded,
                                           double rtol,
                                           double atol);
} // namespace Kadath::coloring_internals
