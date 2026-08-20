/*
    Added 2026 Hao-Jui Kuan
    Guarded backtracking line search for the JFNK-MUMPS Newton step.

    A trial Newton step on an adapted space cannot be undone by re-applying a
    scaled correction (Domain::update_variable remaps fields onto the moved
    surface nonlinearly), so backtracking is driven by an exact in-memory
    snapshot/restore of the full unknown state. That primitive
    (System_of_eqs::snapshot_state / restore_state) is proven bit-exact by the
    JFNK_LS_SELFTEST gate before this line search relies on it.
*/

#include "For_Kadath/System_of_eqs/system_of_eqs.hpp"
#include "For_Kadath/Array/array.hpp"
#include "For_Kadath/Val_domain/val_domain.hpp"
#include "For_Kadath/Utilities/runtime_env.hpp"

#include "newton_norms.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

namespace Kadath
{
    double System_of_eqs::apply_jfnk_line_search(Array<double>& delta, double error_before, int rank)
    {
        // Tunables (read once per step). Defaults give a cheap guard: accept any
        // non-worse full step, otherwise halve the damping a few times.
        const double min_lambda = env_double_value("JFNK_LS_MIN_LAMBDA", 0.1);
        const int max_backtrack = std::max(0, env_int_value("JFNK_LS_MAX_BACKTRACK", 3));
        const bool armijo = env_flag_enabled("JFNK_LS_ARMIJO", false);
        const double alpha = env_double_value("JFNK_LS_ALPHA", 1e-4);
        const std::string guard_field_name = env_string_value("JFNK_LS_GUARD_FIELD");
        const double guard_min_initial_ratio =
            env_double_value("JFNK_LS_GUARD_FIELD_MIN_INITIAL_RATIO", 0.0);
        const double guard_min_step_ratio =
            env_double_value("JFNK_LS_GUARD_FIELD_MIN_STEP_RATIO", 0.0);
        const double guard_min_abs =
            env_double_value("JFNK_LS_GUARD_FIELD_MIN_ABS", 0.0);
        const std::vector<int> guard_sample_domains =
            env_int_list("JFNK_LS_GUARD_FIELD_DOMAINS");

        const int n = delta.get_size(0);
        int guard_field_index = -1;
        if (!guard_field_name.empty()) {
            std::string guard_lookup_name = guard_field_name;
            if (!guard_lookup_name.empty() && guard_lookup_name.back() != ' ')
                guard_lookup_name.push_back(' ');
            for (int i = 0; i < nvar; ++i) {
                if (names_var[static_cast<std::size_t>(i)] == guard_lookup_name) {
                    guard_field_index = i;
                    break;
                }
            }
            if (rank == 0 && guard_field_index < 0) {
                std::cout << "JFNK-MUMPS line search: guard field '" << guard_field_name
                          << "' not found; branch guard disabled" << std::endl;
            }
        }
        const bool guard_enabled =
            guard_field_index >= 0 &&
            (guard_min_initial_ratio > 0.0 || guard_min_step_ratio > 0.0 || guard_min_abs > 0.0);

        // Apply a correction (move adapted surfaces + update field coefficients).
        const auto apply_correction = [this](Array<double>& correction) {
            int offset = 0;
            espace.xx_to_vars_variable_domains(this, correction, offset);
            xx_to_vars_delta(correction, offset);
        };
        // All-rank collective residual: reuses the do_JX MPI row partition
        // once built (replicated until then / on fallback). The backtrack
        // decisions feed off the identically reassembled full vector, so
        // every rank takes the same number of trials.
        const auto residual_norm = [this]() {
            Array<double> residual(sec_member_partitioned());
            return infinity_norm(residual);
        };
        struct FieldGuardMeasurement {
            double amplitude = 0.0;
            std::vector<double> samples;
        };

        const auto guarded_field_measurement = [&]() {
            FieldGuardMeasurement measurement;
            if (!guard_enabled)
                return measurement;

            const Tensor& guarded_field = *var[static_cast<std::size_t>(guard_field_index)];
            if (guard_sample_domains.empty() || guarded_field.get_valence() != 0) {
                measurement.amplitude = maxval(guarded_field);
                return measurement;
            }

            const Scalar& guarded_scalar = guarded_field();
            measurement.amplitude = std::numeric_limits<double>::infinity();
            for (int domain : guard_sample_domains) {
                if (domain < guarded_scalar.get_nbr_domains()) {
                    Index pos(guarded_scalar.get_domain(domain)->get_nbr_points());
                    pos.set_start();
                    const double value = guarded_scalar(domain)(pos);
                    measurement.samples.push_back(value);
                    measurement.amplitude = std::min(measurement.amplitude, std::abs(value));
                }
            }
            if (measurement.samples.empty())
                measurement.amplitude = maxval(guarded_field);
            return measurement;
        };
        const FieldGuardMeasurement base_guard_measurement = guarded_field_measurement();
        const double base_guard_norm = base_guard_measurement.amplitude;
        if (guard_enabled && base_guard_norm > 0.0 && mumps_runtime_state.jfnk_guard_initial_field_norm < 0.0)
            mumps_runtime_state.jfnk_guard_initial_field_norm = base_guard_norm;
        const auto guard_minimum = [&]() {
            if (!guard_enabled)
                return 0.0;
            double minimum = guard_min_abs;
            if (mumps_runtime_state.jfnk_guard_initial_field_norm > 0.0)
                minimum = std::max(minimum, guard_min_initial_ratio * mumps_runtime_state.jfnk_guard_initial_field_norm);
            if (base_guard_norm > 0.0)
                minimum = std::max(minimum, guard_min_step_ratio * base_guard_norm);
            return minimum;
        };
        const auto passes_field_guard = [&](const FieldGuardMeasurement& measurement) {
            if (!guard_enabled)
                return true;
            if (measurement.samples.size() == base_guard_measurement.samples.size()) {
                for (std::size_t i = 0; i < measurement.samples.size(); ++i) {
                    const double base_value = base_guard_measurement.samples[i];
                    if (std::abs(base_value) >= guard_min_abs && base_value * measurement.samples[i] < 0.0)
                        return false;
                }
            }
            return measurement.amplitude >= guard_minimum();
        };
        // Guard mode: accept any finite step that does not increase ||F||.
        // Strict (Armijo) mode: require sufficient decrease scaled by lambda.
        const auto residual_accepts = [&](double err, double lambda) {
            if (!std::isfinite(err))
                return false;
            if (armijo)
                return err <= (1.0 - alpha * lambda) * error_before;
            return err <= error_before;
        };

        // lambda = 1: the full (already step_scale-scaled) Newton step. On the
        // happy path it is accepted here with no extra work beyond the residual
        // evaluation the caller would have done anyway.
        State_snapshot& base = jfnk_line_search_snapshot_;
        snapshot_state_into(base);
        apply_correction(delta);
        const double full_err = residual_norm();
        const FieldGuardMeasurement full_field_measurement = guarded_field_measurement();
        if (residual_accepts(full_err, 1.0) && passes_field_guard(full_field_measurement))
            return full_err;

        // The full step worsened ||F|| (or went non-finite): back off, halving
        // the damping. Track the best finite trial seen (including the full step)
        // so a failed search never commits something worse than the best option.
        if (rank == 0) {
            if (residual_accepts(full_err, 1.0) && !passes_field_guard(full_field_measurement)) {
                std::cout << "JFNK-MUMPS line search: full step rejected by " << guard_field_name
                          << " guard (amplitude=" << full_field_measurement.amplitude
                          << ", min=" << guard_minimum()
                          << "); backtracking" << std::endl;
            } else {
                std::cout << "JFNK-MUMPS line search: full step worsened residual ("
                          << full_err << " vs before=" << error_before << "); backtracking" << std::endl;
            }
        }

        double best_lambda = 1.0;
        double best_err = std::numeric_limits<double>::quiet_NaN();
        if (std::isfinite(full_err) && passes_field_guard(full_field_measurement))
            best_err = full_err;
        double lambda = 1.0;
        Array<double> trial(delta);

        for (int bt = 0; bt < max_backtrack; ++bt) {
            const double next_lambda = lambda * 0.5;
            if (next_lambda < min_lambda)
                break;
            lambda = next_lambda;

            restore_state(base);
            for (int i = 0; i < n; ++i)
                trial.set(i) = delta(i) * lambda;
            apply_correction(trial);
            const double err = residual_norm();
            const FieldGuardMeasurement field_measurement = guarded_field_measurement();
            const bool guard_ok = passes_field_guard(field_measurement);

            if (rank == 0)
                std::cout << "JFNK-MUMPS line search: lambda=" << lambda << " error=" << err
                          << " (before=" << error_before << ")" << std::endl;
            if (rank == 0 && residual_accepts(err, lambda) && !guard_ok) {
                std::cout << "JFNK-MUMPS line search: lambda=" << lambda << " rejected by "
                          << guard_field_name << " guard (amplitude=" << field_measurement.amplitude
                          << ", min=" << guard_minimum() << ")" << std::endl;
            }

            if (std::isfinite(err) && guard_ok && (!std::isfinite(best_err) || err < best_err)) {
                best_err = err;
                best_lambda = lambda;
            }
            if (residual_accepts(err, lambda) && guard_ok)
                return err; // accepted; this lambda is already committed

        }

        // No damping satisfied acceptance. Commit the best finite trial found
        // (which may be the full step) so behaviour degrades to "take the best
        // available direction" and the caller's growth-triggered PC refresh can
        // react on the next step.
        restore_state(base);
        if (!std::isfinite(best_err)) {
            if (rank == 0) {
                std::cout << "JFNK-MUMPS line search: no acceptable damping preserved "
                          << guard_field_name << " guard; restoring previous iterate"
                          << " (error=" << error_before << ")" << std::endl;
            }
            return error_before;
        }
        for (int i = 0; i < n; ++i)
            trial.set(i) = delta(i) * best_lambda;
        apply_correction(trial);
        if (rank == 0)
            std::cout << "JFNK-MUMPS line search: no acceptable damping in " << max_backtrack
                      << " backtracks; committing best lambda=" << best_lambda << " (error=" << best_err
                      << " vs before=" << error_before << ")" << std::endl;
        return best_err;
    }
} // namespace Kadath
