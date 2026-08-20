/*
    Copyright 2017 Philippe Grandclement

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

/*
 * Modifications (Celephais):
 *   2026-08-09  Added typed multiplication profiling.
 */

#include "For_Kadath/Ope_eq/ope_eq.hpp"
#include <chrono>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <string>

namespace Kadath
{
namespace
{
    // Env-gated Ope_mult profile. Counts and times `operator*(Term_eq, Term_eq)`
    // separately from the recursive `parts[i]->action()` evaluation, split by
    // operand type (TERM_T tensor or TERM_D double).
    //
    // Enable: export OPE_MULT_PROFILE=1
    // Dump:   call dump_ope_mult_profile() (linked from solver diagnostics)
    struct OpeMultProfile {
        long long calls = 0;
        long long calls_TT = 0;  // tensor * tensor
        long long calls_TD = 0;  // tensor * double
        long long calls_DT = 0;  // double * tensor
        long long calls_DD = 0;  // double * double
        long long scalar_layout_TT = 0;
        long long scalar_layout_TD = 0;
        long long scalar_layout_DT = 0;
        double t_action_total = 0.0;     // total time inside Ope_mult::action()
        double t_recurse_total = 0.0;    // time spent in the two parts[i]->action() calls
        double t_multiply_total = 0.0;   // time spent in the Term_eq * Term_eq operator only
        bool enabled = false;
        bool initialized = false;

        void init_if_needed()
        {
            if (initialized) return;
            const char* env = std::getenv("OPE_MULT_PROFILE");
            enabled = (env != nullptr && env[0] != '\0' && std::strcmp(env, "0") != 0);
            initialized = true;
        }
    };

    OpeMultProfile& profile()
    {
        static thread_local OpeMultProfile p;
        return p;
    }

    bool scalar_tensor_layout(const Tensor* tensor)
    {
        return tensor != nullptr && tensor->get_valence() == 0 && tensor->get_n_comp() == 1;
    }

    bool scalar_tensor_term_layout(const Term_eq& term)
    {
        if (term.get_type_data() != TERM_T || !scalar_tensor_layout(term.get_p_val_t()) ||
            (term.get_p_der_t(0) != nullptr && !scalar_tensor_layout(term.get_p_der_t(0)))) {
            return false;
        }
        for (int lane = 1; lane < term.get_derivative_lane_count(); ++lane) {
            if (term.get_p_der_t(lane) != nullptr && !scalar_tensor_layout(term.get_p_der_t(lane)))
                return false;
        }
        return true;
    }

}

extern "C" void dump_ope_mult_profile()
{
    auto& p = profile();
    if (!p.enabled) return;
    if (p.calls == 0) {
        std::fprintf(stderr, "[OPE_MULT_PROFILE] no calls recorded\n");
        return;
    }
    std::fprintf(stderr,
        "[OPE_MULT_PROFILE] calls=%lld (TT=%lld TD=%lld DT=%lld DD=%lld) "
        "scalar_layout=(TT=%lld TD=%lld DT=%lld) "
        "t_action=%.3fs t_recurse=%.3fs t_multiply=%.3fs avg_action_us=%.3f\n",
        p.calls, p.calls_TT, p.calls_TD, p.calls_DT, p.calls_DD,
        p.scalar_layout_TT, p.scalar_layout_TD, p.scalar_layout_DT,
        p.t_action_total, p.t_recurse_total, p.t_multiply_total,
        1e6 * p.t_action_total / static_cast<double>(p.calls));
}

extern "C" void reset_ope_mult_profile()
{
    auto& p = profile();
    p.calls = 0;
    p.calls_TT = 0;
    p.calls_TD = 0;
    p.calls_DT = 0;
    p.calls_DD = 0;
    p.scalar_layout_TT = 0;
    p.scalar_layout_TD = 0;
    p.scalar_layout_DT = 0;
    p.t_action_total = 0.0;
    p.t_recurse_total = 0.0;
    p.t_multiply_total = 0.0;
}

    Ope_mult::Ope_mult(const System_of_eqs* zesys, Ope_eq* aa, Ope_eq* bb) : Ope_eq(zesys, aa->get_dom(), 2)
    {
        assert(aa->get_dom() == bb->get_dom());
        parts[0].reset(aa);
        parts[1].reset(bb);
    }

    Ope_mult::~Ope_mult() {}

    Term_eq Ope_mult::action() const
    {
        ScopedOpeActionProfile ope_action_profile_scope(*this);
        auto& p = profile();
        p.init_if_needed();
        if (!p.enabled) {
            ope_action_detail::OperandScratchLease lhs_scratch(syst);
            const Term_eq& lhs = parts[0]->action_operand(lhs_scratch.storage());
            ope_action_detail::OperandScratchLease rhs_scratch(syst);
            const Term_eq& rhs = parts[1]->action_operand(rhs_scratch.storage());
            return lhs * rhs;
        }

        using clock = std::chrono::steady_clock;
        const auto t_action_start = clock::now();

        const auto t_recurse_start = clock::now();
        ope_action_detail::OperandScratchLease lhs_scratch(syst);
        const Term_eq& lhs = parts[0]->action_operand(lhs_scratch.storage());
        ope_action_detail::OperandScratchLease rhs_scratch(syst);
        const Term_eq& rhs = parts[1]->action_operand(rhs_scratch.storage());
        const auto t_recurse_end = clock::now();

        const auto t_mul_start = clock::now();
        Term_eq result(lhs * rhs);
        const auto t_mul_end = clock::now();

        const auto t_action_end = clock::now();

        p.calls += 1;
        const int la = lhs.get_type_data();
        const int rb = rhs.get_type_data();
        if (la == TERM_T && rb == TERM_T) {
            ++p.calls_TT;
            if (scalar_tensor_term_layout(lhs) && scalar_tensor_term_layout(rhs))
                ++p.scalar_layout_TT;
        } else if (la == TERM_T && rb == TERM_D) {
            ++p.calls_TD;
            if (scalar_tensor_term_layout(lhs))
                ++p.scalar_layout_TD;
        } else if (la == TERM_D && rb == TERM_T) {
            ++p.calls_DT;
            if (scalar_tensor_term_layout(rhs))
                ++p.scalar_layout_DT;
        } else {
            ++p.calls_DD;
        }

        p.t_action_total += std::chrono::duration<double>(t_action_end - t_action_start).count();
        p.t_recurse_total += std::chrono::duration<double>(t_recurse_end - t_recurse_start).count();
        p.t_multiply_total += std::chrono::duration<double>(t_mul_end - t_mul_start).count();
        return result;
    }

} // namespace Kadath
