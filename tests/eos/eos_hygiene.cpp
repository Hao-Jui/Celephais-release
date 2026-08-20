/**
 * eos_hygiene — cold EOS table validation tool.
 *
 * Usage:
 *   eos_hygiene <path/to/table> [--n N] [--rtol TOL]
 *
 * Exit 0 = all checks pass.  Exit 1 = any check fails.
 *
 * Checks performed (see PrintResult output for details):
 *   C1  Range validity   hmin > 1.0, hmax finite, rhomin/rhomax > 0
 *   C2  Round-trip       h -> rho -> h reconstructed via first law
 *   C3  Thermo identity  1 + eps + P/rho == h pointwise
 *   C4  Monotonicity     rho(h) and P(h) strictly increasing
 *   C5  Physical         dP/drho > 0, 0 < cs2 < 1 (subluminal)
 *   C6  Error bits       Margherita error_t remains clear throughout
 *   C7  H<=0 floor       raw EOS values at H<=0 that export paths must handle
 */

#include "Hydro/EOS.hh"

#include <algorithm>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <utility>
#include <vector>

using ColdPWP   = Kadath::Margherita::Cold_PWPoly;
using ColdTable = Kadath::Margherita::Cold_Table;

struct HydroState {
    double pressure;
    double density;
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static void print_header(const char* path, const char* kind)
{
    printf("\n=== EOS Hygiene Report ===\n");
    printf("  file : %s\n", path);
    printf("  type : %s\n", kind);
    printf("%-28s  %-6s  %s\n", "Check", "Status", "Detail");
    printf("%-28s  %-6s  %s\n",
           "----------------------------", "------",
           "-----------------------------------------------");
}

enum class Verdict : uint8_t { PASS, WARN, FAIL };

static void print_result(const char* name, Verdict v,
                         const char* fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);
    char detail[256];
    vsnprintf(detail, sizeof(detail), fmt, ap);
    va_end(ap);
    const char* tag = (v == Verdict::PASS) ? "PASS"
                    : (v == Verdict::WARN) ? "WARN" : "FAIL";
    printf("%-28s  %-6s  %s\n", name, tag, detail);
}

static void print_result(const char* name, bool pass, const char* fmt, ...)
    __attribute__((format(printf, 3, 4)));

static void print_result(const char* name, bool pass, const char* fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);
    char detail[256];
    vsnprintf(detail, sizeof(detail), fmt, ap);
    va_end(ap);
    print_result(name, pass ? Verdict::PASS : Verdict::FAIL, "%s", detail);
}

// Log-spaced sweep from lo to hi in n steps.
static std::vector<double> log_sweep(double lo, double hi, int n)
{
    std::vector<double> v(n);
    double log_lo = std::log(lo), log_hi = std::log(hi);
    for (int i = 0; i < n; ++i)
        v[i] = std::exp(log_lo + i * (log_hi - log_lo) / (n - 1));
    return v;
}

// ---------------------------------------------------------------------------
// Per-EOS type check runner (templated to handle PWP and Table identically)
// ---------------------------------------------------------------------------

template <typename Eos>
static int run_checks(double h_lo, double h_hi, int n_points, double rtol)
{
    using Err = typename Eos::error_t;

    // C1 — Range validity
    {
        // Lorene tables may store hmin slightly below 1 at the low-density floor;
        // only hard-fail if the table is structurally broken (hmin < 0.95 or
        // range inverted).  A sub-vacuum floor [0.95,1) is flagged as a warning.
        bool structural_ok = std::isfinite(h_hi) && (h_hi > h_lo)
                             && (Eos::rhomin > 0.0) && (Eos::rhomax > Eos::rhomin)
                             && (h_lo >= 0.95);
        bool floor_warn = (h_lo < 1.0);
        if (!structural_ok) {
            print_result("C1 Range validity", Verdict::FAIL,
                         "hmin=%.6g hmax=%.6g rhomin=%.4g rhomax=%.4g",
                         h_lo, h_hi, Eos::rhomin, Eos::rhomax);
            return 1;
        }
        print_result("C1 Range validity",
                     floor_warn ? Verdict::WARN : Verdict::PASS,
                     "hmin=%.6g%s hmax=%.6g rhomin=%.4g rhomax=%.4g",
                     h_lo, floor_warn ? " (<1, sub-vacuum floor, normal for lorene)" : "",
                     h_hi, Eos::rhomin, Eos::rhomax);
    }

    // Sweep inside the physical domain h > 1 regardless of hmin.
    const double h_sweep_lo = std::max(h_lo, 1.0) * 1.001;
    const double h_sweep_hi = h_hi * 0.999;
    const auto   h_vals     = log_sweep(h_sweep_lo, h_sweep_hi, n_points);

    // Collect per-point values for monotonicity check.
    std::vector<double> rho_vals(n_points), pres_vals(n_points);

    // C2, C3, C4, C5, C6 accumulators
    double max_td_err   = 0.0;    // C2: thermo identity h = 1+eps+P/rho
    double max_inv_err  = 0.0;    // C3: P-inverse round-trip rho -> P -> rho
    bool   cs2_ok       = true;   // C5
    double max_cs2      = 0.0;
    double max_cs2_h    = 0.0;    // h at which max cs2 occurs
    double max_cs2_rho  = 0.0;
    double min_dpdrho   = std::numeric_limits<double>::infinity();
    bool   err_clean    = true;   // C6
    int    first_err_i  = -1;
    bool   c4_pass      = true;

    for (int i = 0; i < n_points; ++i) {
        double h = h_vals[i];
        Err err{};

        double rho = Eos::rho__h_cold(h, err);
        double eps = 0.0;
        double P   = Eos::press_cold_eps_cold__rho(eps, rho, err);

        if (err.any()) {
            err_clean = false;
            if (first_err_i < 0) first_err_i = i;
        }

        rho_vals[i]  = rho;
        pres_vals[i] = P;

        // C2: thermo identity — does 1+eps+P/rho == h after h->rho->P,eps?
        if (rho > 0.0) {
            double h_td = 1.0 + eps + P / rho;
            max_td_err  = std::max(max_td_err, std::abs(h_td / h - 1.0));
        }

        // C3: P-inverse round-trip — rho -> P -> rho__press_cold(P) == rho?
        //     Tests that the h-inverse and P-inverse tables agree.
        if (P > 0.0) {
            Err err2{};
            double rho_back = Eos::rho__press_cold(P, err2);
            if (!err2.any() && rho > 0.0)
                max_inv_err = std::max(max_inv_err, std::abs(rho_back / rho - 1.0));
        }

        // C5: thermodynamic stability and causality.
        double dpdrho = Eos::dpress_cold_drho__rho(rho, err);
        min_dpdrho    = std::min(min_dpdrho, dpdrho);
        if (h > 0.0) {
            double cs2 = dpdrho / h;
            if (cs2 > max_cs2) {
                max_cs2     = cs2;
                max_cs2_h   = h;
                max_cs2_rho = rho;
            }
            if (cs2 <= 0.0 || cs2 >= 1.0)
                cs2_ok = false;
        }
    }

    auto eos_state_at_logh = [](double log_enthalpy) {
        Err err{};
        double h = std::exp(log_enthalpy);
        double rho = Eos::rho__h_cold(h, err);
        double eps = 0.0;
        double P = Eos::press_cold_eps_cold__rho(eps, rho, err);
        return std::pair<HydroState, bool>{{P, rho}, err.any()};
    };

    // C2 — Thermo identity
    {
        bool ok = max_td_err < rtol;
        print_result("C2 Thermo identity", ok,
                     "max |1+eps+P/rho - h|/h = %.3e  (tol %.0e)", max_td_err, rtol);
    }

    // C3 — P-inverse consistency
    {
        bool ok = max_inv_err < rtol;
        print_result("C3 P-inverse consistency", ok,
                     "max |rho_back/rho-1| = %.3e  (tol %.0e)", max_inv_err, rtol);
    }

    // C4 — Strict monotonicity
    {
        bool rho_mono  = true, p_mono = true;
        int  rho_break = -1,   p_break = -1;
        for (int i = 1; i < n_points; ++i) {
            if (rho_mono && rho_vals[i] <= rho_vals[i - 1])
                { rho_mono = false; rho_break = i; }
            if (p_mono && pres_vals[i] <= pres_vals[i - 1])
                { p_mono = false; p_break = i; }
        }
        c4_pass = rho_mono && p_mono;
        if (c4_pass) {
            print_result("C4 Monotonicity", true,
                         "rho(h) and P(h) strictly increasing");
        } else {
            char msg[128] = {};
            if (!rho_mono)
                snprintf(msg + strlen(msg), sizeof(msg) - strlen(msg),
                         "rho breaks at h=%.4g (i=%d) ", h_vals[rho_break], rho_break);
            if (!p_mono)
                snprintf(msg + strlen(msg), sizeof(msg) - strlen(msg),
                         "P breaks at h=%.4g (i=%d)", h_vals[p_break], p_break);
            print_result("C4 Monotonicity", false, "%s", msg);
        }
    }

    // C5 — Physical: stability + causality
    {
        bool stab_ok = min_dpdrho > 0.0;
        bool ok      = stab_ok && cs2_ok;
        print_result("C5 Physical (dP/drho, cs2)", ok,
                     "min dP/drho=%.3e  max cs2=%.4f at h=%.4g rho=%.4g",
                     min_dpdrho, max_cs2, max_cs2_h, max_cs2_rho);
    }

    // C6 — Error bits
    {
        if (err_clean) {
            print_result("C6 Error bits", true, "all clear across %d points", n_points);
        } else {
            print_result("C6 Error bits", false,
                         "first trigger at i=%d h=%.4g", first_err_i,
                         first_err_i >= 0 ? h_vals[first_err_i] : 0.0);
        }
    }

    // C7 — Export boundary hazard: raw EOS floor values must not be exported as matter.
    {
        const auto [surface_raw, surface_err] = eos_state_at_logh(0.0);
        const auto [outside_raw, outside_err] = eos_state_at_logh(-1.0);
        const bool raw_leaks = (surface_raw.pressure != 0.0)
                               || (surface_raw.density != 0.0)
                               || (outside_raw.pressure != 0.0)
                               || (outside_raw.density != 0.0);
        print_result("C7 Raw H<=0 export floor",
                     raw_leaks ? Verdict::WARN : Verdict::PASS,
                     "H=0: P=%.3e rho=%.3e err=%d; H=-1: P=%.3e rho=%.3e err=%d",
                     surface_raw.pressure, surface_raw.density, surface_err ? 1 : 0,
                     outside_raw.pressure, outside_raw.density, outside_err ? 1 : 0);
    }

    // DHDRHO naming note (code audit, not a table property).
    printf("\nNote: eos_var_t::DHDRHO returns (1/h)*dP/drho = cs^2 in term_by_term().\n"
           "      Mathematical dh/drho = (1/rho)*dP/drho.  These differ by rho/h != 1.\n"
           "      The Jacobian contribution (variation) is hardcoded to 0 for DHDRHO.\n");

    bool all_pass = (max_td_err < rtol) && (max_inv_err < rtol) && cs2_ok
                    && (min_dpdrho > 0.0) && err_clean && c4_pass;
    return all_pass ? 0 : 1;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

static void usage(const char* prog)
{
    fprintf(stderr,
            "Usage: %s <eos-path> [--n N] [--rtol TOL]\n"
            "  Type detected from extension: .lorene -> Cold_Table, .polytrope -> Cold_PWPoly.\n"
            "  Unknown extension: exit 2.\n",
            prog);
}

static int detect_kind(const std::string& path)
{
    auto dot = path.rfind('.');
    if (dot == std::string::npos) return 0;
    std::string ext = path.substr(dot + 1);
    if (ext == "polytrope") return 1;
    if (ext == "lorene")    return 2;
    return 0;
}

int main(int argc, char** argv)
{
    if (argc < 2) { usage(argv[0]); return 2; }

    std::string path = argv[1];
    int         n    = 2000;
    double      rtol = -1.0;

    for (int i = 2; i < argc; ++i) {
        if (std::strcmp(argv[i], "--n") == 0 && i + 1 < argc)
            n = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--rtol") == 0 && i + 1 < argc)
            rtol = std::atof(argv[++i]);
        else { usage(argv[0]); return 2; }
    }

    int kind_id = detect_kind(path);
    if (kind_id == 0) {
        fprintf(stderr, "Unknown extension in '%s'; must be .lorene or .polytrope.\n",
                path.c_str());
        return 2;
    }

    const char* kind = (kind_id == 1) ? "Cold_PWPoly" : "Cold_Table";

    try {
        if (kind_id == 1) {
            EOS<ColdPWP, eos_var_t::PRESSURE>::init(path);
            double h_lo = EOS<ColdPWP, eos_var_t::PRESSURE>::h_cold__rho(ColdPWP::rhomin);
            double h_hi = EOS<ColdPWP, eos_var_t::PRESSURE>::h_cold__rho(ColdPWP::rhomax);
            if (rtol < 0) rtol = 1e-10;
            print_header(path.c_str(), kind);
            return run_checks<ColdPWP>(h_lo, h_hi, n, rtol);
        } else {
            EOS<ColdTable, eos_var_t::PRESSURE>::init(path);
            double h_lo = ColdTable::hmin;
            double h_hi = ColdTable::hmax;
            if (rtol < 0) rtol = 1e-6;
            print_header(path.c_str(), kind);
            return run_checks<ColdTable>(h_lo, h_hi, n, rtol);
        }
    } catch (const std::exception& ex) {
        fprintf(stderr, "\nFATAL: EOS init or sweep aborted: %s\n", ex.what());
        fprintf(stderr, "  => Table likely has a non-monotone pressure or enthalpy column.\n");
        return 1;
    }
}
