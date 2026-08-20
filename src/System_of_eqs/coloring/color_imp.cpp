/*
    Added 2026 Hao-Jui Kuan
*/

#include "For_Kadath/System_of_eqs/system_of_eqs.hpp"
#include "For_Kadath/Utilities/name_tools.hpp"
#include "For_Kadath/Scalar/scalar.hpp"
#include <algorithm>
#include <numeric>
#include <map>
#include <set>

namespace Kadath
{

    void System_of_eqs::collect_vars_for_eq(int dom, const char* eq, int bb, std::set<std::string>& vars) const
    {
        if (eq == nullptr)
            return;
        std::string expr(eq);
        if (expr.find('=') == std::string::npos) {
            expr += " = 0";
        }
        Ope_eq* ope = parse_eq(dom, expr.c_str(), bb);
        if (ope == nullptr)
            return;
        ope->collect_vars(vars);
        delete ope;
    }

    void System_of_eqs::build_eq_dependencies(std::vector<std::set<std::string>>& eq_vars,
                                              std::vector<std::set<std::string>>& eq_int_vars) const
    {
        eq_vars.clear();
        eq_int_vars.clear();

        eq_vars.resize(eq_list.size());
        for (std::size_t i = 0; i < eq_list.size(); ++i) {
            const auto& [name, dom, bc] = eq_list[i];
            collect_vars_for_eq(dom, name.c_str(), bc, eq_vars[i]);
        }

        eq_int_vars.resize(eq_int_list.size());
        for (std::size_t i = 0; i < eq_int_list.size(); ++i) {
            const auto& [name, dom, bc] = eq_int_list[i];
            collect_vars_for_eq(dom, name.c_str(), bc, eq_int_vars[i]);
        }
    }

    void System_of_eqs::dump_eq_dependency_coloring(std::ostream& os) const
    {
        std::vector<std::set<std::string>> eq_vars;
        std::vector<std::set<std::string>> eq_int_vars;
        build_eq_dependencies(eq_vars, eq_int_vars);

        const std::size_t n_eq = eq_vars.size();
        const std::size_t n_eq_int = eq_int_vars.size();
        const std::size_t n_total = n_eq + n_eq_int;
        if (n_total == 0) {
            os << "Greedy coloring: no equations to color." << std::endl;
            return;
        }

        std::vector<const std::set<std::string>*> all_vars;
        all_vars.reserve(n_total);
        for (const auto& vars : eq_vars)
            all_vars.push_back(&vars);
        for (const auto& vars : eq_int_vars)
            all_vars.push_back(&vars);

        std::vector<std::vector<int>> adj(n_total);
        for (std::size_t i = 0; i < n_total; ++i) {
            for (std::size_t j = i + 1; j < n_total; ++j) {
                const auto& a = *all_vars[i];
                const auto& b = *all_vars[j];
                const auto& small = (a.size() <= b.size()) ? a : b;
                const auto& large = (a.size() <= b.size()) ? b : a;
                bool connected = false;
                for (const auto& v : small) {
                    if (large.find(v) != large.end()) {
                        connected = true;
                        break;
                    }
                }
                if (connected) {
                    adj[i].push_back(static_cast<int>(j));
                    adj[j].push_back(static_cast<int>(i));
                }
            }
        }

        std::vector<int> order(n_total);
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(), [&](int a, int b) { return adj[a].size() > adj[b].size(); });

        std::vector<int> color(n_total, -1);
        int max_color = -1;
        for (int v : order) {
            std::vector<char> used(static_cast<std::size_t>(max_color + 1), 0);
            for (int u : adj[v]) {
                if (color[u] >= 0 && color[u] <= max_color)
                    used[static_cast<std::size_t>(color[u])] = 1;
            }
            int c = 0;
            while (c <= max_color && used[static_cast<std::size_t>(c)])
                ++c;
            color[v] = c;
            if (c > max_color)
                max_color = c;
        }

        std::vector<int> counts(static_cast<std::size_t>(max_color + 1), 0);
        for (int c : color)
            ++counts[static_cast<std::size_t>(c)];

        os << "Greedy coloring summary" << std::endl;
        os << "  Rows/Equations: total=" << n_total << " (eq=" << n_eq << ", int=" << n_eq_int
           << "), colors=" << (max_color + 1) << std::endl;
        os << "  Row color counts:";
        for (std::size_t c = 0; c < counts.size(); ++c)
            os << " c" << c << "=" << counts[c];
        os << std::endl;
        os << "  Row colors (per equation):" << std::endl;
        for (std::size_t i = 0; i < n_total; ++i) {
            os << "    [" << i << "] color=" << color[i];
            if (i < n_eq) {
                const auto& [name, dom, bc] = eq_list[i];
                os << " eq " << name << " dom=" << dom << " bc=" << bc;
            } else {
                const auto& [name, dom, bc] = eq_int_list[i - n_eq];
                os << " int " << name << " dom=" << dom << " bc=" << bc;
            }
            os << std::endl;
        }
        os << std::endl;

        // -------------------------------------------------------
        // Variable (Column) Coloring for Jacobian Sparsity
        // -------------------------------------------------------
        std::map<std::string, std::vector<int>> var_to_eqs;
        std::set<std::string> all_vars_set;

        auto register_vars = [&](const std::vector<std::set<std::string>>& eq_list_vars, int offset) {
            for (size_t i = 0; i < eq_list_vars.size(); ++i) {
                int eq_idx = offset + static_cast<int>(i);
                for (const auto& v : eq_list_vars[i]) {
                    var_to_eqs[v].push_back(eq_idx);
                    all_vars_set.insert(v);
                }
            }
        };
        register_vars(eq_vars, 0);
        register_vars(eq_int_vars, static_cast<int>(n_eq));

        std::vector<std::string> vars_list(all_vars_set.begin(), all_vars_set.end());
        size_t n_vars = vars_list.size();

        if (n_vars == 0) {
            os << "Greedy coloring (Cols/Variables): no variables found." << std::endl;
            return;
        }

        std::map<std::string, int> var_to_idx;
        for (size_t i = 0; i < n_vars; ++i)
            var_to_idx[vars_list[i]] = static_cast<int>(i);

        // Build adjacency: Two variables are connected if they appear in the same equation
        std::vector<std::set<int>> adj_vars(n_vars);
        auto build_clique = [&](const std::vector<std::set<std::string>>& eq_list_vars) {
            for (const auto& evars : eq_list_vars) {
                std::vector<int> v_indices;
                for (const auto& v : evars)
                    v_indices.push_back(var_to_idx[v]);

                for (size_t i = 0; i < v_indices.size(); ++i) {
                    for (size_t j = i + 1; j < v_indices.size(); ++j) {
                        int u = v_indices[i];
                        int w = v_indices[j];
                        adj_vars[u].insert(w);
                        adj_vars[w].insert(u);
                    }
                }
            }
        };
        build_clique(eq_vars);
        build_clique(eq_int_vars);

        // Greedy Coloring for Variables
        std::vector<int> order_vars(n_vars);
        std::iota(order_vars.begin(), order_vars.end(), 0);
        std::sort(order_vars.begin(), order_vars.end(),
                  [&](int a, int b) { return adj_vars[a].size() > adj_vars[b].size(); });

        std::vector<int> color_vars(n_vars, -1);
        int max_color_vars = -1;

        for (int v : order_vars) {
            std::vector<char> used(static_cast<std::size_t>(max_color_vars + 1), 0);
            for (int u : adj_vars[v]) {
                if (color_vars[u] >= 0 && color_vars[u] <= max_color_vars)
                    used[static_cast<std::size_t>(color_vars[u])] = 1;
            }
            int c = 0;
            while (c <= max_color_vars && used[static_cast<std::size_t>(c)])
                ++c;
            color_vars[v] = c;
            if (c > max_color_vars)
                max_color_vars = c;
        }

        os << "Cols/Variables: total=" << n_vars << ", colors=" << (max_color_vars + 1) << std::endl;
        // Group variables by color for display
        std::vector<std::vector<std::string>> color_groups(max_color_vars + 1);
        for (size_t i = 0; i < n_vars; ++i) {
            color_groups[color_vars[i]].push_back(vars_list[i]);
        }
        for (int c = 0; c <= max_color_vars; ++c) {
            os << "  Color " << c << " (" << color_groups[c].size() << " vars): ";
            for (const auto& vname : color_groups[c])
                os << vname << " ";
            os << std::endl;
        }
    }

} // namespace Kadath
