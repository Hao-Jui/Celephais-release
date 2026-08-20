/*
 * p-coarse two-level preconditioner probe — tuple decoders + GMRES A/B harness.
 *
 * DIAGNOSTIC ONLY. Nothing here is on a production solve path: the two public
 * entry points (pcoarse_dump_probe_artifacts / pcoarse_run_pc_probe) are called
 * exclusively from the rank-0, self-exiting PCOARSE_* blocks in
 * do_newton_jfnk_schur.cpp. See .omx/plans/PCOARSE_PC_PROBE_SPEC_20260701.md.
 *
 * The decoders answer one question per column/row: which spectral tuple
 * (variable/component/domain/i/j/k) does this Jacobian index correspond to? The
 * coarse (previous-resolution) rung and the fine (current) rung are matched by
 * tuple equality; p-nesting guarantees every coarse tuple exists on the fine
 * rung. The maps R (fine rows -> coarse rows) and P (coarse cols -> fine cols)
 * then wrap the coarse MUMPS factor into a fine-rung preconditioner.
 *
 * COLUMN decode mirrors System_of_eqs::describe_column_seed: seed one tau
 * coefficient with affecte_tau_one_coef, read back the primary cell.
 *
 * ROW decode is a code-seeded export replay: deep-copy each equation's residual
 * Term_eq, overwrite its coefficient arrays with an integer tuple encoding, call
 * the equation's own export_val, and decode the emitted doubles. Four seed
 * variants (bulk/boundary x plain/galerkin-base-zeroed) auto-classify every row
 * without re-implementing per-domain export_tau. Two-sided matching equations
 * zero their far side; the first-integral equation (whose export_val performs
 * origin subtraction that would corrupt the encoding) is replayed through
 * export_tau directly with the identical per-domain enumeration.
 */

#include "For_Kadath/System_of_eqs/system_of_eqs.hpp"
#include "For_Kadath/Space/bin_ns.hpp"
#include "For_Kadath/Space/bin_ns_nosym.hpp"
#include "For_Kadath/Scalar/scalar.hpp"
#include "For_Kadath/Term_eq/term_eq.hpp"
#include "For_Kadath/Tensor/tensor.hpp"
#include "For_Kadath/Val_domain/val_domain.hpp"
#include "For_Kadath/Array/array.hpp"
#include "For_Kadath/Array/index.hpp"
#include "For_Kadath/Utilities/runtime_env.hpp"

#include "Linear_algebra/jacobian_assembler.hpp"
#include "Linear_algebra/krylov_solver.hpp"

#ifdef CELEPHAIS_USE_MUMPS
#include "Linear_algebra/mumps_linear_solver.hpp"
#include <mpi.h>
#endif

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#ifdef CELEPHAIS_USE_MUMPS
// LAPACK QR (col-major) for the fine-level Schwarz-Dirichlet smoother -- the
// same primitives SCHUR_PC_GMRES uses (do_newton_jfnk_schur.cpp).
extern "C" {
void dgeqrf_(int* m, int* n, double* a, int* lda, double* tau, double* work,
             int* lwork, int* info);
void dormqr_(char* side, char* trans, int* m, int* n, int* k, double* a, int* lda,
             double* tau, double* c, int* ldc, double* work, int* lwork, int* info);
void dtrtrs_(char* uplo, char* trans, char* diag, int* n, int* nrhs, double* a,
             int* lda, double* b, int* ldb, int* info);
// Dense LU (col-major) for the arm-14 bordered global-block Schur complement.
void dgetrf_(int* m, int* n, double* a, int* lda, int* ipiv, int* info);
void dgetrs_(char* trans, int* n, int* nrhs, double* a, int* lda, int* ipiv,
             double* b, int* ldb, int* info);
// Form Q from a dgeqrf factorization (arm-15 subspace orthonormalization).
void dorgqr_(int* m, int* n, int* k, double* a, int* lda, double* tau,
             double* work, int* lwork, int* info);
// Nonsymmetric dense eigenvalues (arm-15 Ritz spectrum of V^T A V; arm-15d
// also requests right eigenvectors for the per-mode invariance residual).
void dgeev_(char* jobvl, char* jobvr, int* n, double* a, int* lda, double* wr,
            double* wi, double* vl, int* ldvl, double* vr, int* ldvr,
            double* work, int* lwork, int* info);
// Dense 1-norm + LU condition estimate for the arm-15d Galerkin block guard.
double dlange_(char* norm, int* m, int* n, double* a, int* lda, double* work);
void dgecon_(char* norm, int* n, double* a, int* lda, double* anorm,
             double* rcond, double* work, int* iwork, int* info);
// Triangular solve (arm-15d: propagate the sorted-Ritz orthonormalization R to
// A*W, i.e. AW := AW * R^{-1}).
void dtrsm_(char* side, char* uplo, char* transa, char* diag, int* m, int* n,
            double* alpha, double* a, int* lda, double* b, int* ldb);
}
#endif

namespace Kadath
{
    namespace
    {
        // Integer tuple encoding with a large additive BASE. A row that reads a
        // SINGLE coefficient cell carries value BASE + tuple; any linear
        // combination of n cells (Galerkin lift, boundary radial sum) carries
        // n*BASE + (small combination), so round(value/BASE) is the number of
        // cells summed. Requiring it == 1 rejects every combination and kills
        // the coincidental-integer false positives a bare encoding produced.
        constexpr double kPcoarseBase = 1073741824.0; // 2^30 >> any tuple part
        constexpr int kPcoarseCompStride = 524288;    // 2^19 = 64*64*128

        double pcoarse_encode(int comp, int i, int j, int k)
        {
            return kPcoarseBase +
                   static_cast<double>(1 + i + 64 * j + 4096 * k +
                                       kPcoarseCompStride * comp);
        }

        // Decode iff v is a clean SINGLE-cell read (round(v/BASE)==1) whose tuple
        // lies inside the domain's real coefficient grid (nc0,nc1,nc2).
        bool pcoarse_decode(double v, int nc0, int nc1, int nc2, int& comp, int& i,
                            int& j, int& k)
        {
            if (std::llround(v / kPcoarseBase) != 1)
                return false; // not a single-cell read
            const double small = v - kPcoarseBase;
            const long long e = std::llround(small);
            if (std::abs(small - static_cast<double>(e)) > 1e-6 || e < 1)
                return false;
            const long long e0 = e - 1;
            comp = static_cast<int>(e0 / kPcoarseCompStride);
            const long long r = e0 % kPcoarseCompStride;
            i = static_cast<int>(r % 64);
            j = static_cast<int>((r / 64) % 64);
            k = static_cast<int>(r / 4096);
            if (comp < 0 || i < 0 || j < 0 || k < 0)
                return false;
            if (i >= nc0 || j >= nc1 || k >= nc2)
                return false; // outside the real spectral grid -> false positive
            return pcoarse_encode(comp, i, j, k) == v;
        }

        // One export-replay pass: which base planes to null out, plus whether to
        // seed only the i=0 radial plane (boundary evaluations sum the radial
        // series, so only i=0 survives with T_0 = P_0 = 1 on either face).
        struct PcoarsePass {
            bool zero_i0;
            bool zero_j0;
            bool zero_j1;
            bool boundary;
            const char* tag;
        };

        // Ordered least-to-most zeroing. Each row is claimed by the first pass
        // that turns it into a single-cell read: bulk interior, then theta/chi
        // Galerkin (base j0 or j1), then nucleus/adapted radial+theta Galerkin
        // (base i0), then Dirichlet boundary and its theta Galerkin. Priority
        // prevents a heavier pass from corrupting a row an earlier one owns.
        const PcoarsePass kPcoarsePasses[] = {
            {false, false, false, false, "bulk"},     // interior single cell
            {false, true, false, false, "galerkin"},  // theta/chi base j=0
            {false, false, true, false, "galerkin"},  // theta base j=1 (SIN_EVEN)
            {true, true, false, false, "galerkin"},   // nucleus radial+theta base
            {true, false, false, false, "galerkin"},  // nucleus k=0 radial base
            {false, false, false, true, "boundary"},  // Dirichlet boundary
            {false, true, false, true, "boundary"},   // boundary theta Galerkin
        };

        // Overwrite a Val_domain's coefficient array with the tuple encoding for
        // one pass, preserving its spectral basis (needed so export_tau reads the
        // correct theta parity). A logically-zero component is given a standard
        // basis so it still enumerates its ncond rows.
        void pcoarse_seed_valdomain(Val_domain& vd, int comp, const PcoarsePass& p)
        {
            if (vd.check_if_zero() || !vd.get_base().is_def()) {
                vd.std_base();
                vd.allocate_coef();
            } else {
                vd.coef();
            }
            Index ci(vd.get_domain()->get_nbr_coefs());
            do {
                const int i = ci(0), j = ci(1), k = ci(2);
                const bool suppress = (p.boundary && i != 0) ||
                                      (p.zero_i0 && i == 0) ||
                                      (p.zero_j0 && j == 0) || (p.zero_j1 && j == 1);
                vd.set_coef(ci) = suppress ? 0.0 : pcoarse_encode(comp, i, j, k);
            } while (ci.inc());
        }

        // Mirror of assign_column_probe_base (column_map.cpp): copy the live
        // variable's per-component basis onto a freshly seeded probe.
        void pcoarse_assign_probe_bases(const Tensor& reference, Tensor& probe,
                                        int dom)
        {
            for (int comp = 0; comp < probe.get_n_comp(); ++comp) {
                const Array<int> idx(probe.indices(comp));
                if (!probe(idx)(dom).check_if_zero() &&
                    !probe(idx)(dom).get_base().is_def()) {
                    probe.set(idx).set_domain(dom).set_base() =
                        reference(idx)(dom).get_base();
                }
            }
        }

        std::string pcoarse_taxonomy_name(RowTaxonomy tax)
        {
            switch (tax) {
                case RowTaxonomy::Vol: return "Vol";
                case RowTaxonomy::TauBc: return "TauBc";
                case RowTaxonomy::TauMatch: return "TauMatch";
                case RowTaxonomy::GlobalInt: return "GlobalInt";
                default: return "Unknown";
            }
        }

        std::string pcoarse_col_canon(const PcoarseColumnKey& c)
        {
            std::ostringstream os;
            os << "C|" << c.block << '|' << c.name << '|' << c.comp << '|'
               << c.dom << '|' << c.i << '|' << c.j << '|' << c.k;
            return os.str();
        }

        std::string pcoarse_row_canon(const PcoarseRowKey& r)
        {
            std::ostringstream os;
            if (r.tag == "eq_int") {
                os << "RI|" << r.expr;
            } else {
                os << "R|" << r.eq_index << '|' << r.comp << '|' << r.dom << '|'
                   << r.i << '|' << r.j << '|' << r.k << '|' << r.tag;
            }
            return os.str();
        }
    } // namespace

    long long System_of_eqs::pcoarse_read_coarse_n(const std::string& directory) const
    {
        std::ifstream bin(directory + "/coo.bin", std::ios::binary);
        if (!bin)
            return -1;
        std::int64_t n64 = 0;
        bin.read(reinterpret_cast<char*>(&n64), sizeof(n64));
        return bin ? static_cast<long long>(n64) : -1;
    }

    // =======================================================================
    // Column decode
    // =======================================================================
    void System_of_eqs::pcoarse_decode_columns(
        std::vector<PcoarseColumnKey>& out) const
    {
        out.assign(static_cast<std::size_t>(nbr_unknowns), PcoarseColumnKey{});

        std::vector<ColumnMetadata> cols;
        classify_columns(cols);

        std::vector<int> term_start(static_cast<std::size_t>(nterm), -1);
        for (const ColumnMetadata& m : cols)
            if (m.term_idx >= 0 && m.term_idx < nterm &&
                term_start[static_cast<std::size_t>(m.term_idx)] < 0)
                term_start[static_cast<std::size_t>(m.term_idx)] = m.column;

        int adapted1 = -1, adapted2 = -1;
        if (const auto* s = dynamic_cast<const Space_bin_ns*>(&espace)) {
            adapted1 = s->ADAPTED1;
            adapted2 = s->ADAPTED2;
        } else if (const auto* s =
                       dynamic_cast<const Space_bin_ns_nosym*>(&espace)) {
            adapted1 = s->ADAPTED1;
            adapted2 = s->ADAPTED2;
        }
        const int n_adapted1 =
            (adapted1 >= 0)
                ? espace.get_domain(adapted1)->nbr_unknowns_from_adapted()
                : 0;

        for (int cc = 0; cc < nbr_unknowns; ++cc) {
            const ColumnMetadata& m = cols[static_cast<std::size_t>(cc)];
            PcoarseColumnKey key;

            if (m.column_class == ColumnClass::VarDomain) {
                // Adapted-surface coefficient. Replicate the (k,j) walk of
                // Domain_shell_outer_adapted::nbr_unknowns_from_adapted, keyed
                // by (star, k, j). Star 1 block precedes star 2.
                key.block = 0;
                key.name = "__vardom__";
                int star, local;
                const Domain* adapted;
                if (m.vardom_param < n_adapted1) {
                    star = 1;
                    local = m.vardom_param;
                    adapted = espace.get_domain(adapted1);
                } else {
                    star = 2;
                    local = m.vardom_param - n_adapted1;
                    adapted = (adapted2 >= 0) ? espace.get_domain(adapted2)
                                              : espace.get_domain(adapted1);
                }
                const Dim_array nc = adapted->get_nbr_coefs();
                int idx = 0, found_k = -1, found_j = -1;
                for (int k = 0; k < nc(2) && found_k < 0; ++k) {
                    if (k == 1 || k == nc(2) - 1)
                        continue;
                    const int mm = (k % 2 == 0) ? k / 2 : (k - 1) / 2;
                    const int jmax = (mm % 2 == 0) ? nc(1) : nc(1) - 1;
                    const int jmin = (k >= 4) ? 1 : 0;
                    for (int j = jmin; j < jmax; ++j) {
                        if (idx == local) {
                            found_k = k;
                            found_j = j;
                            break;
                        }
                        ++idx;
                    }
                }
                key.comp = star;
                key.j = found_j;
                key.k = found_k;
            } else if (m.var_double_idx >= 0) {
                key.block = 1;
                key.name = (m.var_double_idx <
                                static_cast<int>(names_var_double.size()) &&
                            !names_var_double[static_cast<std::size_t>(
                                                  m.var_double_idx)]
                                 .empty())
                               ? names_var_double[static_cast<std::size_t>(
                                     m.var_double_idx)]
                               : "__double__";
                key.comp = m.var_double_idx;
            } else if (m.term_idx >= 0 && m.term_idx < nterm && m.domain >= 0) {
                key.block = 2;
                key.name = m.var_name;
                key.dom = m.domain;
                Tensor probe(term[static_cast<std::size_t>(m.term_idx)]->get_val_t(),
                             false);
                probe.annule_hard();
                int counter = term_start[static_cast<std::size_t>(m.term_idx)];
                espace.get_domain(m.domain)->affecte_tau_one_coef(probe, m.domain,
                                                                  cc, counter);
                pcoarse_assign_probe_bases(
                    term[static_cast<std::size_t>(m.term_idx)]->get_val_t(), probe,
                    m.domain);
                int best_comp = -1, best_i = -1, best_j = -1, best_k = -1;
                for (int comp = 0; comp < probe.get_n_comp(); ++comp) {
                    const Array<int> cind = probe.indices(comp);
                    const Val_domain& vd = probe(cind)(m.domain);
                    if (vd.check_if_zero())
                        continue;
                    const Array<double> cf = vd.get_coef();
                    Index pos(cf.get_dimensions());
                    do {
                        if (std::fabs(cf(pos)) > 0.0) {
                            const int i = pos(0), j = pos(1), k = pos(2);
                            // Primary cell = lexicographically greatest (j, i);
                            // Galerkin lifts write extra cells at i=0 and/or j=0.
                            if (best_comp < 0 || j > best_j ||
                                (j == best_j && i > best_i)) {
                                best_comp = comp;
                                best_i = i;
                                best_j = j;
                                best_k = k;
                            }
                        }
                    } while (pos.inc());
                }
                key.comp = best_comp;
                key.i = best_i;
                key.j = best_j;
                key.k = best_k;
            } else {
                // ScalarGlobal / unmapped: key by name + registration slot.
                key.block = 3;
                key.name = !m.var_name.empty() ? m.var_name : "__scalar_global__";
                key.comp = m.basis_mode;
            }
            out[static_cast<std::size_t>(cc)] = key;
        }
    }

    // =======================================================================
    // Row decode
    // =======================================================================
    void System_of_eqs::pcoarse_decode_rows(std::vector<PcoarseRowKey>& out)
    {
        out.assign(static_cast<std::size_t>(nbr_conditions), PcoarseRowKey{});

        std::vector<RowMetadata> meta;
        classify_equation_row_metadata(meta);

        // Integral head rows: matched by expression string.
        for (int i = 0; i < neq_int; ++i) {
            PcoarseRowKey key;
            key.eq_index = i;
            key.tag = "eq_int";
            key.taxonomy = "GlobalInt";
            key.decoded = true;
            key.expr = (i < static_cast<int>(eq_int_list.size()))
                           ? std::get<0>(eq_int_list[static_cast<std::size_t>(i)])
                           : ("__eqint_" + std::to_string(i));
            out[static_cast<std::size_t>(i)] = key;
        }

        // Repopulate results with the residual Term_eqs (the assembler's do_JX
        // sweeps overwrote them); bare sec_member() is a local rank-safe pass.
        (void)sec_member();
        Term_eq** results_raw = results_shadow_view();

        // Per-equation operator offsets (part_off) and row offsets (row_begin).
        std::vector<int> part_off(static_cast<std::size_t>(neq) + 1, 0);
        {
            int conte = 0;
            Array<double> scratch(nbr_conditions);
            scratch = 0.0;
            int pos = neq_int;
            for (int i = 0; i < neq; ++i) {
                part_off[static_cast<std::size_t>(i)] = conte;
                eq[i]->export_val(conte, results_raw, scratch, pos);
            }
            part_off[static_cast<std::size_t>(neq)] = conte;
        }
        std::vector<int> row_begin(static_cast<std::size_t>(neq) + 1, neq_int);
        for (int i = 0; i < neq; ++i)
            row_begin[static_cast<std::size_t>(i) + 1] =
                row_begin[static_cast<std::size_t>(i)] + eq[i]->get_n_cond_tot();

        const int ndom_all = espace.get_nbr_domains();

        for (int i = 0; i < neq; ++i) {
            const int nrows = eq[i]->get_n_cond_tot();
            const int rb = row_begin[static_cast<std::size_t>(i)];
            const int nope = part_off[static_cast<std::size_t>(i) + 1] -
                             part_off[static_cast<std::size_t>(i)];
            const int ndom_eq = eq[i]->get_ndom();

            const int n_passes =
                static_cast<int>(sizeof(kPcoarsePasses) / sizeof(kPcoarsePasses[0]));

            // First-integral: export_val subtracts the origin value, which
            // corrupts the encoding. Replay its per-domain export_tau directly:
            // same (i,j,k) enumeration, uncorrupted values. It is a pure bulk
            // export, so the boundary passes are skipped.
            if (auto* first_integral =
                    dynamic_cast<Eq_first_integral*>(eq[i].get())) {
                const int dmin = first_integral->dom_min;
                const int dmax = first_integral->dom_max;
                int base_pos = rb;
                for (int d = dmin; d <= dmax; ++d) {
                    const Dim_array dnc = espace.get_domain(d)->get_nbr_coefs();
                    std::vector<std::vector<double>> pass_vals(
                        static_cast<std::size_t>(n_passes));
                    int dom_rows = 0;
                    for (int pi = 0; pi < n_passes; ++pi) {
                        if (kPcoarsePasses[pi].boundary)
                            continue;
                        Scalar probe(espace);
                        Val_domain& vd = probe.set_domain(d);
                        pcoarse_seed_valdomain(vd, 0, kPcoarsePasses[pi]);
                        Array<int> dom_ncond(1);
                        dom_ncond.set(0) = nrows;
                        Array<double> seci(nbr_conditions);
                        seci = 0.0;
                        int seci_pos = 0;
                        espace.get_domain(d)->export_tau(probe, d, 0, seci, seci_pos,
                                                         dom_ncond);
                        dom_rows = seci_pos;
                        pass_vals[static_cast<std::size_t>(pi)].assign(
                            static_cast<std::size_t>(seci_pos), 0.0);
                        for (int r = 0; r < seci_pos; ++r)
                            pass_vals[static_cast<std::size_t>(pi)]
                                     [static_cast<std::size_t>(r)] = seci(r);
                    }
                    for (int r = 0; r < dom_rows; ++r) {
                        const int gr = base_pos + r;
                        if (gr >= nbr_conditions)
                            break;
                        PcoarseRowKey key;
                        key.eq_index = i;
                        key.dom = d;
                        if (gr < static_cast<int>(meta.size()))
                            key.taxonomy = pcoarse_taxonomy_name(
                                meta[static_cast<std::size_t>(gr)].taxonomy);
                        int comp, ii, jj, kk;
                        for (int pi = 0; pi < n_passes; ++pi) {
                            if (kPcoarsePasses[pi].boundary ||
                                r >= static_cast<int>(
                                         pass_vals[static_cast<std::size_t>(pi)].size()))
                                continue;
                            if (pcoarse_decode(pass_vals[static_cast<std::size_t>(pi)]
                                                        [static_cast<std::size_t>(r)],
                                               dnc(0), dnc(1), dnc(2), comp, ii, jj,
                                               kk)) {
                                key.comp = comp;
                                key.i = ii;
                                key.j = jj;
                                key.k = kk;
                                key.tag = kPcoarsePasses[pi].tag;
                                key.decoded = true;
                                break;
                            }
                        }
                        if (!key.decoded)
                            key.tag = "undecoded";
                        out[static_cast<std::size_t>(gr)] = key;
                    }
                    base_pos += dom_rows;
                }
                continue;
            }

            const bool is_matching =
                dynamic_cast<Eq_matching*>(eq[i].get()) != nullptr ||
                dynamic_cast<Eq_matching_non_std*>(eq[i].get()) != nullptr ||
                dynamic_cast<Eq_matching_import*>(eq[i].get()) != nullptr;
            const Dim_array eq_nc = espace.get_domain(ndom_eq)->get_nbr_coefs();

            // Deep-copy the equation's parts for doctoring.
            std::vector<std::unique_ptr<Term_eq>> store;
            std::vector<Term_eq*> local(static_cast<std::size_t>(nope));
            for (int p = 0; p < nope; ++p) {
                store.emplace_back(std::make_unique<Term_eq>(
                    *results_raw[part_off[static_cast<std::size_t>(i)] + p]));
                local[static_cast<std::size_t>(p)] = store.back().get();
            }

            auto run_pass = [&](const PcoarsePass& pass, std::vector<double>& vals) {
                for (int p = 0; p < nope; ++p) {
                    Tensor* T = local[static_cast<std::size_t>(p)]->set_val_t();
                    const bool zero_far = is_matching && p >= 1;
                    for (int comp = 0; comp < T->get_n_comp(); ++comp) {
                        const Array<int> ind = T->indices(comp);
                        if (zero_far) {
                            for (int d = 0; d < ndom_all; ++d)
                                if (!(*T)(ind)(d).check_if_zero())
                                    T->set(ind).set_domain(d) = 0.0;
                        } else {
                            Val_domain& vd = T->set(ind).set_domain(ndom_eq);
                            pcoarse_seed_valdomain(vd, comp, pass);
                        }
                    }
                }
                Array<double> seci(nbr_conditions);
                seci = 0.0;
                int conte = 0, pos = 0;
                eq[i]->export_val(conte, local.data(), seci, pos);
                vals.assign(static_cast<std::size_t>(nrows), 0.0);
                for (int r = 0; r < nrows && r < pos; ++r)
                    vals[static_cast<std::size_t>(r)] = seci(r);
            };

            std::vector<std::vector<double>> pass_vals(
                static_cast<std::size_t>(n_passes));
            for (int pi = 0; pi < n_passes; ++pi)
                run_pass(kPcoarsePasses[pi], pass_vals[static_cast<std::size_t>(pi)]);

            for (int r = 0; r < nrows; ++r) {
                const int gr = rb + r;
                PcoarseRowKey key;
                key.eq_index = i;
                key.dom = ndom_eq;
                if (gr < static_cast<int>(meta.size()))
                    key.taxonomy = pcoarse_taxonomy_name(
                        meta[static_cast<std::size_t>(gr)].taxonomy);
                int comp, ii, jj, kk;
                for (int pi = 0; pi < n_passes; ++pi) {
                    if (pcoarse_decode(
                            pass_vals[static_cast<std::size_t>(pi)]
                                     [static_cast<std::size_t>(r)],
                            eq_nc(0), eq_nc(1), eq_nc(2), comp, ii, jj, kk)) {
                        key.comp = comp;
                        key.i = kPcoarsePasses[pi].boundary ? -1 : ii;
                        key.j = jj;
                        key.k = kk;
                        key.tag = kPcoarsePasses[pi].tag;
                        key.decoded = true;
                        break;
                    }
                }
                if (!key.decoded)
                    key.tag = "undecoded";
                out[static_cast<std::size_t>(gr)] = key;
            }
        }
    }

    // =======================================================================
    // Dump (PCOARSE_DUMP=<dir>)
    // =======================================================================
    void System_of_eqs::pcoarse_dump_probe_artifacts(const std::string& directory,
                                                     const AssembledJacobianCoo& coo,
                                                     double drop_tol_used)
    {
        std::vector<PcoarseColumnKey> col_keys;
        std::vector<PcoarseRowKey> row_keys;
        pcoarse_decode_columns(col_keys);
        pcoarse_decode_rows(row_keys);

        std::vector<RowMetadata> meta;
        classify_equation_row_metadata(meta);

        auto path = [&](const char* name) { return directory + "/" + name; };

        // coo.bin : n(int64) nnz(int64) drop_tol(double) irn(int32*) jcn(int32*) a(double*)
        {
            std::ofstream bin(path("coo.bin"), std::ios::binary);
            const std::int64_t n64 = nbr_unknowns;
            const std::int64_t nnz64 = coo.nnz;
            bin.write(reinterpret_cast<const char*>(&n64), sizeof(n64));
            bin.write(reinterpret_cast<const char*>(&nnz64), sizeof(nnz64));
            bin.write(reinterpret_cast<const char*>(&drop_tol_used),
                      sizeof(drop_tol_used));
            bin.write(reinterpret_cast<const char*>(coo.irn.data()),
                      static_cast<std::streamsize>(nnz64) *
                          static_cast<std::streamsize>(sizeof(int)));
            bin.write(reinterpret_cast<const char*>(coo.jcn.data()),
                      static_cast<std::streamsize>(nnz64) *
                          static_cast<std::streamsize>(sizeof(int)));
            bin.write(reinterpret_cast<const char*>(coo.a.data()),
                      static_cast<std::streamsize>(nnz64) *
                          static_cast<std::streamsize>(sizeof(double)));
        }

        // columns.csv
        {
            std::ofstream csv(path("columns.csv"));
            csv << "col,block,var_name,comp,dom,i,j,k\n";
            for (int cc = 0; cc < nbr_unknowns; ++cc) {
                const PcoarseColumnKey& c = col_keys[static_cast<std::size_t>(cc)];
                csv << cc << ',' << c.block << ',' << c.name << ',' << c.comp << ','
                    << c.dom << ',' << c.i << ',' << c.j << ',' << c.k << '\n';
            }
        }
        // col_keys.txt : canonical strings, probe-consumed
        {
            std::ofstream f(path("col_keys.txt"));
            for (const PcoarseColumnKey& c : col_keys)
                f << pcoarse_col_canon(c) << '\n';
        }

        // rows.csv
        {
            std::ofstream csv(path("rows.csv"));
            csv << "row,eq_index,taxonomy,dom,comp,i,j,k,tag,decoded\n";
            for (int rr = 0; rr < nbr_conditions; ++rr) {
                const PcoarseRowKey& r = row_keys[static_cast<std::size_t>(rr)];
                csv << rr << ',' << r.eq_index << ',' << r.taxonomy << ',' << r.dom
                    << ',' << r.comp << ',' << r.i << ',' << r.j << ',' << r.k << ','
                    << r.tag << ',' << (r.decoded ? 1 : 0) << '\n';
            }
        }
        // row_keys.txt : canonical strings, probe-consumed
        {
            std::ofstream f(path("row_keys.txt"));
            for (const PcoarseRowKey& r : row_keys)
                f << (r.decoded ? "" : "UNDECODED ") << pcoarse_row_canon(r) << '\n';
        }

        // eq_ints.txt : index<TAB>expression<TAB>dom<TAB>bound
        {
            std::ofstream f(path("eq_ints.txt"));
            for (int i = 0; i < neq_int; ++i) {
                std::string expr =
                    (i < static_cast<int>(eq_int_list.size()))
                        ? std::get<0>(eq_int_list[static_cast<std::size_t>(i)])
                        : "";
                const int dom = (i < static_cast<int>(eq_int_list.size()))
                                    ? std::get<1>(eq_int_list[static_cast<std::size_t>(i)])
                                    : -1;
                const int bound = (i < static_cast<int>(eq_int_list.size()))
                                      ? std::get<2>(eq_int_list[static_cast<std::size_t>(i)])
                                      : -1;
                f << i << '\t' << expr << '\t' << dom << '\t' << bound << '\n';
            }
        }

        // census.txt
        int decoded_rows = 0, decoded_cols = 0;
        for (const PcoarseRowKey& r : row_keys)
            if (r.decoded)
                ++decoded_rows;
        for (const PcoarseColumnKey& c : col_keys)
            if (c.block >= 0)
                ++decoded_cols;
        {
            std::ofstream f(path("census.txt"));
            f << "nvar " << static_cast<int>(names_var.size()) << '\n';
            f << "nvar_double " << nvar_double << '\n';
            f << "neq " << neq << '\n';
            f << "neq_int " << neq_int << '\n';
            f << "n " << nbr_unknowns << '\n';
            f << "nbr_conditions " << nbr_conditions << '\n';
            f << "decoded_cols " << decoded_cols << '\n';
            f << "decoded_rows " << decoded_rows << '\n';
            for (std::size_t v = 0; v < names_var.size(); ++v)
                f << "var " << v << ' ' << names_var[v] << '\n';
            for (int v = 0; v < nvar_double; ++v)
                f << "var_double " << v << ' '
                  << (v < static_cast<int>(names_var_double.size())
                          ? names_var_double[static_cast<std::size_t>(v)]
                          : std::string("__double__"))
                  << '\n';
        }

        // Column cross-check: describe_column_seed vs decoded key, spanning
        // domains AND spectral modes. Sample four columns per domain (at 0/25/
        // 50/75% into the domain's field block) so nonzero (i,j,k) are exercised,
        // not just the (0,0,0) leading coefficient.
        int cross_checked = 0, cross_ok = 0;
        std::map<int, std::vector<int>> field_cols_of_dom;
        for (int cc = 0; cc < nbr_unknowns; ++cc)
            if (col_keys[static_cast<std::size_t>(cc)].block == 2)
                field_cols_of_dom[col_keys[static_cast<std::size_t>(cc)].dom]
                    .push_back(cc);
        std::vector<int> sample_cols;
        for (const auto& kv : field_cols_of_dom) {
            const std::vector<int>& v = kv.second;
            const std::size_t n = v.size();
            for (double frac : {0.0, 0.25, 0.5, 0.75}) {
                const std::size_t idx = std::min(
                    n - 1, static_cast<std::size_t>(frac * static_cast<double>(n)));
                sample_cols.push_back(v[idx]);
            }
        }
        std::cout << "=== pcoarse column cross-check (describe_column_seed) ===\n";
        for (const int cc : sample_cols) {
            const PcoarseColumnKey& c = col_keys[static_cast<std::size_t>(cc)];
            std::ostringstream oss;
            describe_column_seed(cc, oss, 32);
            // Recover the primary (max-j, max-i) cell from the describe dump.
            int dj = -1, di = -1, dk = -1;
            std::istringstream in(oss.str());
            std::string line;
            while (std::getline(in, line)) {
                const std::size_t p = line.find("coeff=(");
                if (p == std::string::npos)
                    continue;
                int a = 0, b = 0, cch = 0;
                if (std::sscanf(line.c_str() + p, "coeff=(%d,%d,%d)", &a, &b, &cch) ==
                    3) {
                    if (dj < 0 || b > dj || (b == dj && a > di)) {
                        di = a;
                        dj = b;
                        dk = cch;
                    }
                }
            }
            const bool ok = (di == c.i && dj == c.j && dk == c.k);
            ++cross_checked;
            if (ok)
                ++cross_ok;
            std::cout << "  col=" << cc << " dom=" << c.dom << " var=" << c.name
                      << " decoded(i,j,k)=(" << c.i << ',' << c.j << ',' << c.k
                      << ") describe(i,j,k)=(" << di << ',' << dj << ',' << dk
                      << ") " << (ok ? "OK" : "MISMATCH") << '\n';
        }
        std::cout << "  cross-check: " << cross_ok << '/' << cross_checked
                  << (cross_ok == cross_checked ? "  PASS" : "  FAIL") << '\n';

        std::cout << "=== pcoarse dump complete -> " << directory << " ===\n"
                  << "  n=" << nbr_unknowns << " nnz=" << coo.nnz
                  << " drop_tol=" << drop_tol_used << '\n'
                  << "  decoded cols=" << decoded_cols << '/' << nbr_unknowns
                  << " rows=" << decoded_rows << '/' << nbr_conditions << '\n';
        std::cout.flush();
    }

    // =======================================================================
    // Probe (PCOARSE_PC_PROBE=<dir>)
    // =======================================================================
    void System_of_eqs::pcoarse_run_pc_probe(const std::string& directory,
                                             const AssembledJacobianCoo& coo,
                                             const Array<double>& fine_residual,
                                             const SolverRuntimeConfig& config)
    {
#ifdef CELEPHAIS_USE_MUMPS
        auto path = [&](const char* name) { return directory + "/" + name; };
        auto fail = [&](const std::string& why) {
            std::cerr << "pcoarse probe FAIL: " << why << '\n';
            std::cerr.flush();
            std::cout.flush();
        };

        // --- Load coarse COO ---
        std::int64_t n_coarse = 0, nnz_coarse = 0;
        double coarse_drop_tol = 0.0;
        std::vector<int> irn_c, jcn_c;
        std::vector<double> a_c;
        {
            std::ifstream bin(path("coo.bin"), std::ios::binary);
            if (!bin) {
                fail("cannot open coo.bin");
                return;
            }
            bin.read(reinterpret_cast<char*>(&n_coarse), sizeof(n_coarse));
            bin.read(reinterpret_cast<char*>(&nnz_coarse), sizeof(nnz_coarse));
            bin.read(reinterpret_cast<char*>(&coarse_drop_tol),
                     sizeof(coarse_drop_tol));
            irn_c.resize(static_cast<std::size_t>(nnz_coarse));
            jcn_c.resize(static_cast<std::size_t>(nnz_coarse));
            a_c.resize(static_cast<std::size_t>(nnz_coarse));
            bin.read(reinterpret_cast<char*>(irn_c.data()),
                     static_cast<std::streamsize>(nnz_coarse) *
                         static_cast<std::streamsize>(sizeof(int)));
            bin.read(reinterpret_cast<char*>(jcn_c.data()),
                     static_cast<std::streamsize>(nnz_coarse) *
                         static_cast<std::streamsize>(sizeof(int)));
            bin.read(reinterpret_cast<char*>(a_c.data()),
                     static_cast<std::streamsize>(nnz_coarse) *
                         static_cast<std::streamsize>(sizeof(double)));
            if (!bin) {
                fail("short read on coo.bin");
                return;
            }
        }

        // --- Load coarse canonical keys ---
        std::vector<std::string> coarse_col_canon, coarse_row_canon;
        {
            std::ifstream f(path("col_keys.txt"));
            std::string line;
            while (std::getline(f, line))
                if (!line.empty())
                    coarse_col_canon.push_back(line);
        }
        {
            std::ifstream f(path("row_keys.txt"));
            std::string line;
            while (std::getline(f, line))
                if (!line.empty())
                    coarse_row_canon.push_back(line);
        }
        if (static_cast<int>(coarse_col_canon.size()) != n_coarse) {
            fail("col_keys.txt count != coarse n");
            return;
        }

        // --- Census compatibility ---
        {
            std::ifstream f(path("census.txt"));
            std::string tok;
            int c_neq = -1, c_neq_int = -1;
            std::vector<std::string> c_var;
            std::string line;
            while (std::getline(f, line)) {
                std::istringstream in(line);
                in >> tok;
                if (tok == "neq")
                    in >> c_neq;
                else if (tok == "neq_int")
                    in >> c_neq_int;
                else if (tok == "var") {
                    int idx;
                    std::string name;
                    in >> idx >> name;
                    c_var.push_back(name);
                }
            }
            // names_var carries the parser's trailing whitespace ("H "); the
            // census reader strips it. Trim both sides before comparing (the map
            // keys keep the raw name consistently on both rungs, so only this
            // compatibility check is affected).
            auto trim = [](const std::string& s) {
                const std::size_t e = s.find_last_not_of(" \t\r\n");
                return (e == std::string::npos) ? std::string() : s.substr(0, e + 1);
            };
            bool census_ok = (c_neq == neq) && (c_neq_int == neq_int) &&
                             (c_var.size() == names_var.size());
            if (census_ok)
                for (std::size_t v = 0; v < c_var.size(); ++v)
                    if (trim(c_var[v]) != trim(names_var[v])) {
                        census_ok = false;
                        std::cout << "  var mismatch at " << v << ": coarse='"
                                  << trim(c_var[v]) << "' fine='"
                                  << trim(names_var[v]) << "'\n";
                        break;
                    }
            std::cout << "=== pcoarse census compat ===\n"
                      << "  coarse neq=" << c_neq << " fine neq=" << neq
                      << " coarse neq_int=" << c_neq_int << " fine neq_int="
                      << neq_int << " var-name-order "
                      << (census_ok ? "MATCH" : "MISMATCH") << '\n';
            if (!census_ok) {
                fail("census incompatible (stage/variable misalignment)");
                return;
            }
        }

        // --- Decode fine keys, build fine canonical -> index maps ---
        std::vector<PcoarseColumnKey> fine_cols;
        std::vector<PcoarseRowKey> fine_rows;
        pcoarse_decode_columns(fine_cols);
        pcoarse_decode_rows(fine_rows);

        std::unordered_map<std::string, int> fine_col_of_canon, fine_row_of_canon;
        fine_col_of_canon.reserve(fine_cols.size() * 2);
        fine_row_of_canon.reserve(fine_rows.size() * 2);
        int fine_col_dupes = 0, fine_row_dupes = 0;
        for (int cc = 0; cc < static_cast<int>(fine_cols.size()); ++cc)
            if (fine_cols[static_cast<std::size_t>(cc)].block >= 0) {
                const std::string s =
                    pcoarse_col_canon(fine_cols[static_cast<std::size_t>(cc)]);
                if (!fine_col_of_canon.emplace(s, cc).second)
                    ++fine_col_dupes;
            }
        for (int rr = 0; rr < static_cast<int>(fine_rows.size()); ++rr)
            if (fine_rows[static_cast<std::size_t>(rr)].decoded) {
                const std::string s =
                    pcoarse_row_canon(fine_rows[static_cast<std::size_t>(rr)]);
                if (!fine_row_of_canon.emplace(s, rr).second)
                    ++fine_row_dupes;
            }

        // --- Build transfer maps (coarse index -> fine index) ---
        std::vector<int> col_map(static_cast<std::size_t>(n_coarse), -1);
        std::vector<int> row_map(static_cast<std::size_t>(n_coarse), -1);
        int col_matched = 0, col_undecoded = 0, col_missed = 0;
        int row_matched = 0, row_undecoded = 0, row_missed = 0;
        std::vector<std::string> missed_examples;
        for (int cc = 0; cc < static_cast<int>(n_coarse); ++cc) {
            const std::string& s = coarse_col_canon[static_cast<std::size_t>(cc)];
            const auto it = fine_col_of_canon.find(s);
            if (it != fine_col_of_canon.end()) {
                col_map[static_cast<std::size_t>(cc)] = it->second;
                ++col_matched;
            } else {
                ++col_missed;
                if (missed_examples.size() < 12)
                    missed_examples.push_back("col " + s);
            }
        }
        for (int cr = 0; cr < static_cast<int>(coarse_row_canon.size()) &&
                         cr < static_cast<int>(n_coarse);
             ++cr) {
            const std::string& s = coarse_row_canon[static_cast<std::size_t>(cr)];
            if (s.rfind("UNDECODED ", 0) == 0) {
                ++row_undecoded;
                continue;
            }
            const auto it = fine_row_of_canon.find(s);
            if (it != fine_row_of_canon.end()) {
                row_map[static_cast<std::size_t>(cr)] = it->second;
                ++row_matched;
            } else {
                ++row_missed;
                if (missed_examples.size() < 24)
                    missed_examples.push_back("row " + s);
            }
        }
        for (const std::string& s : coarse_col_canon)
            if (s.find("|-1|-1|-1|-1") != std::string::npos && s.rfind("C|2|", 0) == 0)
                ++col_undecoded;

        std::cout << "=== pcoarse map census ===\n"
                  << "  coarse n=" << n_coarse << " fine n=" << nbr_unknowns << '\n'
                  << "  columns: matched=" << col_matched << " missed=" << col_missed
                  << " undecoded(field)=" << col_undecoded
                  << " fine_dupes=" << fine_col_dupes << '\n'
                  << "  rows:    matched=" << row_matched << " missed=" << row_missed
                  << " undecoded=" << row_undecoded << " fine_dupes=" << fine_row_dupes
                  << '\n';
        for (const std::string& s : missed_examples)
            std::cout << "    MISS " << s << '\n';
        const bool census_pass = (col_missed == 0) && (row_missed == 0) &&
                                 (col_undecoded == 0) && (row_undecoded == 0) &&
                                 (fine_col_dupes == 0) && (fine_row_dupes == 0);
        std::cout << "  MAP CENSUS: " << (census_pass ? "PASS (full nesting)"
                                                      : "INCOMPLETE (see misses)")
                  << '\n';
        std::cout.flush();

        // Matched-fine column / row sets (for the complement + decomposition).
        std::vector<char> fine_col_matched(static_cast<std::size_t>(nbr_unknowns), 0);
        std::vector<char> fine_row_matched(static_cast<std::size_t>(nbr_conditions),
                                           0);
        for (int cc = 0; cc < static_cast<int>(n_coarse); ++cc)
            if (col_map[static_cast<std::size_t>(cc)] >= 0)
                fine_col_matched[static_cast<std::size_t>(
                    col_map[static_cast<std::size_t>(cc)])] = 1;
        for (int cr = 0; cr < static_cast<int>(n_coarse); ++cr)
            if (row_map[static_cast<std::size_t>(cr)] >= 0)
                fine_row_matched[static_cast<std::size_t>(
                    row_map[static_cast<std::size_t>(cr)])] = 1;

        // --- Factor coarse COO (exact MUMPS oracle pattern) ---
        const long long nnz_c = nnz_coarse;
        MumpsLinearSolver coarse_solver(static_cast<int>(n_coarse),
                                        config.mumps.ordering, false, 0,
                                        mumps_runtime_state.icntl14, MPI_COMM_SELF);
        coarse_solver.set_pattern(static_cast<int>(n_coarse), nnz_c, irn_c.data(),
                                  jcn_c.data());
        coarse_solver.analyze_pattern();
        coarse_solver.factor_analyzed(a_c.data());
        std::cout << "  coarse factor done (n=" << n_coarse << " nnz=" << nnz_c
                  << ")\n";
        std::cout.flush();

        // --- Fine COO matvec + diagonal ---
        const long long nnz_f = coo.nnz;
        const int n_f = nbr_unknowns;
        auto coo_spmv = [&](const std::vector<double>& v, std::vector<double>& out) {
            out.assign(static_cast<std::size_t>(n_f), 0.0);
            for (long long e = 0; e < nnz_f; ++e)
                out[static_cast<std::size_t>(coo.irn[static_cast<std::size_t>(e)] -
                                             1)] +=
                    coo.a[static_cast<std::size_t>(e)] *
                    v[static_cast<std::size_t>(coo.jcn[static_cast<std::size_t>(e)] -
                                               1)];
        };
        std::vector<double> diag(static_cast<std::size_t>(n_f), 0.0);
        for (long long e = 0; e < nnz_f; ++e) {
            const int rr = coo.irn[static_cast<std::size_t>(e)] - 1;
            const int cc = coo.jcn[static_cast<std::size_t>(e)] - 1;
            if (rr == cc)
                diag[static_cast<std::size_t>(rr)] +=
                    coo.a[static_cast<std::size_t>(e)];
        }

        const double alpha = env_double_value("PCOARSE_ALPHA", 1.0);

        // Coarse two-level correction: z = P A_c^{-1} R r.
        auto coarse_correction = [&](const std::vector<double>& r,
                                     std::vector<double>& z) {
            std::vector<double> rc(static_cast<std::size_t>(n_coarse), 0.0);
            for (int cr = 0; cr < static_cast<int>(n_coarse); ++cr)
                if (row_map[static_cast<std::size_t>(cr)] >= 0)
                    rc[static_cast<std::size_t>(cr)] =
                        r[static_cast<std::size_t>(row_map[static_cast<std::size_t>(cr)])];
            coarse_solver.solve(rc.data()); // in place: rc <- A_c^{-1} rc
            z.assign(static_cast<std::size_t>(n_f), 0.0);
            for (int cc = 0; cc < static_cast<int>(n_coarse); ++cc)
                if (col_map[static_cast<std::size_t>(cc)] >= 0)
                    z[static_cast<std::size_t>(col_map[static_cast<std::size_t>(cc)])] =
                        rc[static_cast<std::size_t>(cc)];
        };

        // PC arms.
        KrylovOperator pc_identity =
            [](const std::vector<double>& v, std::vector<double>& out) { out = v; };
        KrylovOperator pc_coarse_id = [&](const std::vector<double>& r,
                                          std::vector<double>& z) {
            coarse_correction(r, z);
            for (int i = 0; i < n_f; ++i)
                if (!fine_col_matched[static_cast<std::size_t>(i)])
                    z[static_cast<std::size_t>(i)] +=
                        alpha * r[static_cast<std::size_t>(i)];
        };
        KrylovOperator pc_coarse_jacobi = [&](const std::vector<double>& r,
                                              std::vector<double>& z) {
            coarse_correction(r, z);
            for (int i = 0; i < n_f; ++i)
                if (!fine_col_matched[static_cast<std::size_t>(i)]) {
                    const double d = diag[static_cast<std::size_t>(i)];
                    z[static_cast<std::size_t>(i)] +=
                        (std::abs(d) < 1e-30)
                            ? alpha * r[static_cast<std::size_t>(i)]
                            : r[static_cast<std::size_t>(i)] / d;
                }
        };

        // Additive arms (arms 3-5): M = I + s * P A_c^{-1} R -- identity applied
        // EVERYWHERE with the coarse correction added on top (closes the design
        // gap where matched columns had no identity fallback).
        const double coarse_scale =
            env_double_value("PCOARSE_COARSE_SCALE", 1.0);
        auto additive_apply = [&](const std::vector<double>& r,
                                  std::vector<double>& z, double s) {
            coarse_correction(r, z);
            for (int i = 0; i < n_f; ++i)
                z[static_cast<std::size_t>(i)] =
                    s * z[static_cast<std::size_t>(i)] + r[static_cast<std::size_t>(i)];
        };
        KrylovOperator pc_add_1 = [&](const std::vector<double>& r,
                                      std::vector<double>& z) {
            additive_apply(r, z, 1.0);
        };
        KrylovOperator pc_add_env = [&](const std::vector<double>& r,
                                        std::vector<double>& z) {
            additive_apply(r, z, coarse_scale);
        };
        KrylovOperator pc_add_01 = [&](const std::vector<double>& r,
                                       std::vector<double>& z) {
            additive_apply(r, z, 0.1);
        };

        // --- b = TRUE post-regrid residual ---
        std::vector<double> b(static_cast<std::size_t>(n_f), 0.0);
        for (int i = 0; i < n_f; ++i)
            b[static_cast<std::size_t>(i)] = fine_residual.get_data()[i];
        double b_norm = 0.0;
        for (double v : b)
            b_norm += v * v;
        b_norm = std::sqrt(b_norm);

        // ================================================================
        // Discriminators: prove the mechanism behind the arm-vs-control result.
        // All reuse the existing maps (col_map/row_map) and coarse factor.
        // ================================================================
        // A_c spmv (coarse COO, 1-based).
        auto coarse_spmv = [&](const std::vector<double>& e,
                               std::vector<double>& y) {
            y.assign(static_cast<std::size_t>(n_coarse), 0.0);
            for (long long z = 0; z < nnz_c; ++z)
                y[static_cast<std::size_t>(irn_c[static_cast<std::size_t>(z)] - 1)] +=
                    a_c[static_cast<std::size_t>(z)] *
                    e[static_cast<std::size_t>(jcn_c[static_cast<std::size_t>(z)] - 1)];
        };
        // P: coarse column vector -> fine column vector.
        auto prolong = [&](const std::vector<double>& e, std::vector<double>& pf) {
            pf.assign(static_cast<std::size_t>(n_f), 0.0);
            for (int cc = 0; cc < static_cast<int>(n_coarse); ++cc)
                if (col_map[static_cast<std::size_t>(cc)] >= 0)
                    pf[static_cast<std::size_t>(
                        col_map[static_cast<std::size_t>(cc)])] =
                        e[static_cast<std::size_t>(cc)];
        };
        // R: fine row vector -> coarse row vector (matched rows only).
        auto restrict_rows = [&](const std::vector<double>& f,
                                 std::vector<double>& rc) {
            rc.assign(static_cast<std::size_t>(n_coarse), 0.0);
            for (int cr = 0; cr < static_cast<int>(n_coarse); ++cr)
                if (row_map[static_cast<std::size_t>(cr)] >= 0)
                    rc[static_cast<std::size_t>(cr)] = f[static_cast<std::size_t>(
                        row_map[static_cast<std::size_t>(cr)])];
        };
        // Low-mode coarse-column mask: field columns with all of (i,j,k) <= 4,
        // well inside both the res9 and res11 truncations.
        std::vector<char> coarse_col_lowmode(static_cast<std::size_t>(n_coarse), 0);
        for (int cc = 0; cc < static_cast<int>(n_coarse); ++cc) {
            const std::string& s = coarse_col_canon[static_cast<std::size_t>(cc)];
            std::vector<std::string> fld;
            std::size_t p0 = 0, bar;
            while ((bar = s.find('|', p0)) != std::string::npos) {
                fld.push_back(s.substr(p0, bar - p0));
                p0 = bar + 1;
            }
            fld.push_back(s.substr(p0));
            if (fld.size() >= 8 && fld[1] == "2") {
                const int i = std::atoi(fld[5].c_str());
                const int j = std::atoi(fld[6].c_str());
                const int k = std::atoi(fld[7].c_str());
                if (i >= 0 && i <= 4 && j >= 0 && j <= 4 && k >= 0 && k <= 4)
                    coarse_col_lowmode[static_cast<std::size_t>(cc)] = 1;
            }
        }
        int n_lowmode = 0;
        for (char c : coarse_col_lowmode)
            n_lowmode += c;

        // Deterministic unit coarse vector (optionally low-mode masked).
        auto make_ec = [&](double seed, bool low_only, std::vector<double>& e) {
            e.assign(static_cast<std::size_t>(n_coarse), 0.0);
            for (int cc = 0; cc < static_cast<int>(n_coarse); ++cc) {
                if (low_only && !coarse_col_lowmode[static_cast<std::size_t>(cc)])
                    continue;
                const double h =
                    std::sin((static_cast<double>(cc) + 1.0) * 12.9898 +
                             seed * 78.233) *
                    43758.5453;
                e[static_cast<std::size_t>(cc)] = (h - std::floor(h)) * 2.0 - 1.0;
            }
            double nrm = 0.0;
            for (double v : e)
                nrm += v * v;
            nrm = std::sqrt(nrm);
            if (nrm > 0.0)
                for (double& v : e)
                    v /= nrm;
        };
        auto normalize = [](std::vector<double>& v) {
            double nrm = 0.0;
            for (double x : v)
                nrm += x * x;
            nrm = std::sqrt(nrm);
            if (nrm > 0.0)
                for (double& x : v)
                    x /= nrm;
        };
        // rel_L2(A_c e - R A_f P e) / ||A_c e||, matched rows only.
        auto consistency = [&](const std::vector<double>& e) {
            std::vector<double> y1, pf, af, y2;
            coarse_spmv(e, y1);
            prolong(e, pf);
            coo_spmv(pf, af);
            restrict_rows(af, y2);
            double num = 0.0, den = 0.0;
            for (int cr = 0; cr < static_cast<int>(n_coarse); ++cr)
                if (row_map[static_cast<std::size_t>(cr)] >= 0) {
                    const double d = y1[static_cast<std::size_t>(cr)] -
                                     y2[static_cast<std::size_t>(cr)];
                    num += d * d;
                    den += y1[static_cast<std::size_t>(cr)] *
                           y1[static_cast<std::size_t>(cr)];
                }
            return (den > 0.0) ? std::sqrt(num) / std::sqrt(den) : std::sqrt(num);
        };

        std::cout << "=== pcoarse discriminator 1: Galerkin consistency "
                     "(A_c e vs R A_f P e, matched rows) ===\n"
                  << "  low-mode coarse columns (block2, i,j,k<=4): " << n_lowmode
                  << '\n';
        double lowmode_worst_rel = 0.0;
        for (int s = 0; s < 5; ++s) {
            std::vector<double> e;
            make_ec(1.0 + s, false, e);
            std::cout << "  (a) random   e_c[" << s << "] rel_diff=" << consistency(e)
                      << '\n';
        }
        for (int s = 0; s < 5; ++s) {
            std::vector<double> e;
            make_ec(101.0 + s, true, e);
            const double rr = consistency(e);
            lowmode_worst_rel = std::max(lowmode_worst_rel, rr);
            std::cout << "  (b) low-mode e_c[" << s << "] rel_diff=" << rr << '\n';
        }
        std::vector<double> ec_rb;
        restrict_rows(b, ec_rb);
        normalize(ec_rb);
        std::cout << "  (c) e_c = R*b (normalized) rel_diff=" << consistency(ec_rb)
                  << '\n';
        const bool maps_broken = (lowmode_worst_rel > 0.5);
        std::cout << "  low-mode worst rel_diff=" << lowmode_worst_rel << " -> "
                  << (maps_broken ? "MAPS/PREMISE BROKEN (verdict void)"
                                  : "coarse operator consistent on low modes")
                  << '\n';

        if (maps_broken) {
            std::vector<double> y1, pf, af, y2;
            coarse_spmv(ec_rb, y1);
            prolong(ec_rb, pf);
            coo_spmv(pf, af);
            restrict_rows(af, y2);
            std::vector<std::pair<double, int>> diffs;
            for (int cr = 0; cr < static_cast<int>(n_coarse); ++cr)
                if (row_map[static_cast<std::size_t>(cr)] >= 0)
                    diffs.emplace_back(std::abs(y1[static_cast<std::size_t>(cr)] -
                                                y2[static_cast<std::size_t>(cr)]),
                                       cr);
            std::sort(diffs.begin(), diffs.end(),
                      [](const std::pair<double, int>& x,
                         const std::pair<double, int>& y) {
                          return x.first > y.first;
                      });
            std::cout << "=== TOP-10 consistency disagreements (e_c = R*b) ===\n";
            for (int t = 0; t < 10 && t < static_cast<int>(diffs.size()); ++t) {
                const int cr = diffs[static_cast<std::size_t>(t)].second;
                std::cout << "  |d|=" << diffs[static_cast<std::size_t>(t)].first
                          << " y1(A_c)=" << y1[static_cast<std::size_t>(cr)]
                          << " y2(R A_f P)=" << y2[static_cast<std::size_t>(cr)]
                          << " coarse_row=" << coarse_row_canon[static_cast<std::size_t>(cr)]
                          << '\n';
            }
            std::cout << "  STOP: maps/premise inconsistent on low modes; GMRES "
                         "arms skipped (verdict void, not a real FAIL).\n";
            std::cout.flush();
            return;
        }

        // Discriminator 2: one apply of the arm-1 preconditioner to b; residual
        // reduction restricted to matched rows (a consistent coarse correction
        // should knock the matched residual down in a SINGLE apply).
        {
            std::vector<double> z, az;
            pc_coarse_id(b, z);
            coo_spmv(z, az);
            double num = 0.0, den = 0.0;
            for (int i = 0; i < n_f; ++i)
                if (fine_row_matched[static_cast<std::size_t>(i)]) {
                    const double rr = b[static_cast<std::size_t>(i)] -
                                      az[static_cast<std::size_t>(i)];
                    num += rr * rr;
                    den += b[static_cast<std::size_t>(i)] *
                           b[static_cast<std::size_t>(i)];
                }
            std::cout << "=== pcoarse discriminator 2: one-apply of arm-1 M ===\n"
                      << "  ||b - A_f (M b)||_matched / ||b||_matched = "
                      << ((den > 0.0) ? std::sqrt(num) / std::sqrt(den) : 0.0)
                      << "   (consistent coarse correction -> large one-apply drop)\n";
            std::cout.flush();
        }

        int maxit = env_int_value("PCOARSE_GMRES_MAXIT", 300);
        GmresConfig gmres_config;
        gmres_config.max_iters = maxit;
        gmres_config.tolerance = 1e-6 * b_norm; // GmresConfig.tolerance is absolute

        struct ArmResult {
            std::string name;
            bool converged;
            int iters;
            double rel_resid;
            double wall;
            std::vector<double> x;
            std::vector<double> curve; // per-iteration absolute residual norm
        };
        std::vector<ArmResult> arms;
        auto run_arm = [&](const std::string& name, const KrylovOperator& pc) {
            std::vector<double> x(static_cast<std::size_t>(n_f), 0.0);
            std::vector<double> curve;
            gmres_config.residual_history = &curve;
            std::cout << "--- pcoarse GMRES arm: " << name << " (curve below) ---\n";
            std::cout.flush();
            const auto t0 = std::chrono::steady_clock::now();
            GmresStatus st =
                right_preconditioned_gmres(b, x, coo_spmv, pc, gmres_config);
            gmres_config.residual_history = nullptr;
            const double wall =
                std::chrono::duration<double>(std::chrono::steady_clock::now() - t0)
                    .count();
            const double rel = (b_norm > 0.0) ? st.residual_norm / b_norm
                                              : st.residual_norm;
            arms.push_back({name, st.converged, st.iterations, rel, wall,
                            std::move(x), std::move(curve)});
            std::cout << "  arm " << name << ": converged=" << std::boolalpha
                      << st.converged << " iters=" << st.iterations
                      << " rel_resid=" << rel << " wall=" << wall << "s\n";
            std::cout.flush();
        };

        // Energy decomposition of b ITSELF (matched / unmatched-high-p / TauMatch)
        // -- sharpens the "post-regrid error is high-p by construction" claim.
        {
            std::vector<RowMetadata> meta0;
            classify_equation_row_metadata(meta0);
            double em = 0.0, eu = 0.0, et = 0.0, etot = 0.0;
            for (int i = 0; i < n_f; ++i) {
                const double e2 =
                    b[static_cast<std::size_t>(i)] * b[static_cast<std::size_t>(i)];
                etot += e2;
                const bool is_tm =
                    (i < static_cast<int>(meta0.size())) &&
                    meta0[static_cast<std::size_t>(i)].taxonomy ==
                        RowTaxonomy::TauMatch;
                if (is_tm)
                    et += e2;
                else if (fine_row_matched[static_cast<std::size_t>(i)])
                    em += e2;
                else
                    eu += e2;
            }
            auto fr = [&](double e) {
                return (etot > 0.0) ? std::sqrt(e / etot) : 0.0;
            };
            std::cout << "=== pcoarse decomposition of b itself ===\n"
                      << "  ||b||_matched/||b||   = " << fr(em) << '\n'
                      << "  ||b||_unmatched/||b|| = " << fr(eu)
                      << "   (high-p error present in the post-regrid RHS)\n"
                      << "  ||b||_TauMatch/||b||  = " << fr(et) << '\n';
            std::cout.flush();
        }

        const bool smoother_on =
            env_flag_enabled("PCOARSE_SMOOTHER", false);
        // Smoother variant selector (case-insensitive). "ras" / "ras-interface"
        // route to the RAS overlap branch (arms 8r/9/9m or 8r/10); any other
        // truthy value keeps the per-domain Dirichlet smoother (arms 6-8).
        std::string smoother_variant;
        {
            const std::string raw =
                env_string_value("PCOARSE_SMOOTHER", "");
            smoother_variant.reserve(raw.size());
            for (char ch : raw)
                smoother_variant += static_cast<char>(
                    std::tolower(static_cast<unsigned char>(ch)));
        }
        const bool want_ras = (smoother_variant == "ras");
        const bool want_ras_interface = (smoother_variant == "ras-interface" ||
                                         smoother_variant == "ras_interface");
        run_arm("0_control_noPC", pc_identity);
        if (!smoother_on) {
            run_arm("1_coarse_identity_complement", pc_coarse_id);
            run_arm("2_coarse_jacobi_complement", pc_coarse_jacobi);
            run_arm("3_additive_I_plus_coarse", pc_add_1);
            run_arm("4_additive_scale" + std::to_string(coarse_scale), pc_add_env);
            run_arm("5_additive_scale0.1", pc_add_01);
        } else if (want_ras || want_ras_interface) {
            // ============================================================
            // ARMS 8r / 9 / 9m / 10: restricted additive Schwarz (RAS) smoother
            // composed with the p-coarse two-level correction. Base partition =
            // the per-DOMAIN column blocks (column ownership = ColumnInfo.domain,
            // same as arms 6-8); vardom/var_double/global columns keep the extra
            // "global" block. The RAS blocks widen each per-domain block by a
            // graph-distance halo (delta=1 algebraic, or TauMatch-adjacent for
            // the interface variant) but RESTRICT the correction scatter to the
            // owned base columns -- partition of unity by domain ownership, so
            // overlaps are not double-counted.
            //
            // Memory: the Dirichlet smoother (arm 8r baseline) is built, run,
            // and FREED before the RAS smoother is built, so peak QR storage is
            // max(dirichlet, ras) rather than the sum, and the 16 GB cap below
            // is a RAS-only budget (memory-over-speed policy).
            //
            // This branch deliberately duplicates the arms 6-8 block machinery
            // rather than refactoring the committed path: arm 8r must reproduce
            // the committed arm-8 number bit-for-bit to stay a valid baseline.
            // ============================================================
            std::vector<ColumnInfo> cmap;
            build_column_map(cmap, false);
            std::map<int, std::vector<int>> dom_cols;
            std::vector<int> global_cols;
            for (int cc = 0; cc < n_f; ++cc) {
                const ColumnInfo& ci = cmap[static_cast<std::size_t>(cc)];
                if (ci.is_var_domain || ci.var_double_idx >= 0 || ci.domain < 0)
                    global_cols.push_back(cc);
                else
                    dom_cols[ci.domain].push_back(cc);
            }

            std::vector<RowMetadata> smoother_meta;
            classify_equation_row_metadata(smoother_meta);

            struct SchwarzBlock {
                int m = 0;      // local columns (base + halo)
                int n_base = 0; // leading base columns that receive the scatter
                int n_touch = 0;
                std::vector<double> a_qr; // factored A[touch, cols], col-major
                std::vector<double> tau;
                std::vector<int> touch_global;
                std::vector<int> cols; // base columns first, then halo
            };
            std::vector<int> loc(static_cast<std::size_t>(n_f), -1);
            std::vector<int> touch_local(static_cast<std::size_t>(n_f), -1);

            // Build+factor one QR block from an explicit column list (base
            // first, then halo). The correction is scattered only to the leading
            // n_base columns (Dirichlet: n_base==m -> scatter all cols).
            auto build_block = [&](std::vector<SchwarzBlock>& out_blocks,
                                   long long& a_qr_doubles,
                                   const std::vector<int>& cols, int n_base,
                                   const std::string& label) {
                const int m = static_cast<int>(cols.size());
                if (m <= 0)
                    return;
                const int nb = std::min(n_base, m);
                for (int k = 0; k < m; ++k)
                    loc[static_cast<std::size_t>(cols[static_cast<std::size_t>(k)])] = k;
                std::fill(touch_local.begin(), touch_local.end(), -1);
                int n_touch = 0;
                std::vector<int> touch_global;
                for (long long e = 0; e < nnz_f; ++e) {
                    if (loc[static_cast<std::size_t>(
                            coo.jcn[static_cast<std::size_t>(e)] - 1)] < 0)
                        continue;
                    const int r = coo.irn[static_cast<std::size_t>(e)] - 1;
                    if (touch_local[static_cast<std::size_t>(r)] < 0) {
                        touch_local[static_cast<std::size_t>(r)] = n_touch++;
                        touch_global.push_back(r);
                    }
                }
                const std::size_t nt = static_cast<std::size_t>(n_touch);
                std::vector<double> a_g(nt * static_cast<std::size_t>(m), 0.0);
                for (long long e = 0; e < nnz_f; ++e) {
                    const int lc = loc[static_cast<std::size_t>(
                        coo.jcn[static_cast<std::size_t>(e)] - 1)];
                    if (lc < 0)
                        continue;
                    const int r = coo.irn[static_cast<std::size_t>(e)] - 1;
                    a_g[static_cast<std::size_t>(
                            touch_local[static_cast<std::size_t>(r)]) +
                        static_cast<std::size_t>(lc) * nt] +=
                        coo.a[static_cast<std::size_t>(e)];
                }
                int qr_m = n_touch, qr_n = m, qr_lda = std::max(1, n_touch), info = 0;
                std::vector<double> tau(static_cast<std::size_t>(m));
                double wq = 0.0;
                int lwork = -1;
                dgeqrf_(&qr_m, &qr_n, a_g.data(), &qr_lda, tau.data(), &wq, &lwork,
                        &info);
                lwork = (info == 0) ? static_cast<int>(wq) : std::max(1, m);
                std::vector<double> work(static_cast<std::size_t>(std::max(1, lwork)));
                dgeqrf_(&qr_m, &qr_n, a_g.data(), &qr_lda, tau.data(), work.data(),
                        &lwork, &info);
                if (info != 0)
                    std::cerr << "  schwarz block " << label << " dgeqrf info=" << info
                              << '\n';
                a_qr_doubles += static_cast<long long>(nt) * m;
                SchwarzBlock blk;
                blk.m = m;
                blk.n_base = nb;
                blk.n_touch = n_touch;
                blk.a_qr = std::move(a_g);
                blk.tau = std::move(tau);
                blk.touch_global = std::move(touch_global);
                blk.cols = cols;
                out_blocks.push_back(std::move(blk));
                for (int k = 0; k < m; ++k)
                    loc[static_cast<std::size_t>(cols[static_cast<std::size_t>(k)])] =
                        -1;
                std::cout << "  schwarz block " << label << " m=" << m
                          << " n_base=" << nb << " n_touch=" << n_touch
                          << " (a_qr GB so far ~"
                          << (static_cast<double>(a_qr_doubles) * 8.0 / 1e9) << ")\n";
                std::cout.flush();
            };

            // S(r): additive per-block QR least-squares over a block set; the
            // correction is restricted to each block's owned base columns.
            auto schwarz_apply = [&](std::vector<SchwarzBlock>& blks,
                                     const std::vector<double>& r,
                                     std::vector<double>& z) {
                z.assign(static_cast<std::size_t>(n_f), 0.0);
                for (SchwarzBlock& blk : blks) {
                    const int m = blk.m, n_touch = blk.n_touch;
                    if (m <= 0 || n_touch <= 0)
                        continue;
                    std::vector<double> rg(static_cast<std::size_t>(n_touch));
                    for (int i = 0; i < n_touch; ++i)
                        rg[static_cast<std::size_t>(i)] = r[static_cast<std::size_t>(
                            blk.touch_global[static_cast<std::size_t>(i)])];
                    char side = 'L', trans = 'T';
                    int qm = n_touch, qn = 1, qk = m, qlda = std::max(1, n_touch),
                        qldc = std::max(1, n_touch), qinfo = 0, lwork = -1;
                    double wq = 0.0;
                    dormqr_(&side, &trans, &qm, &qn, &qk, blk.a_qr.data(), &qlda,
                            blk.tau.data(), rg.data(), &qldc, &wq, &lwork, &qinfo);
                    lwork = (qinfo == 0) ? static_cast<int>(wq) : std::max(1, n_touch);
                    std::vector<double> work(
                        static_cast<std::size_t>(std::max(1, lwork)));
                    dormqr_(&side, &trans, &qm, &qn, &qk, blk.a_qr.data(), &qlda,
                            blk.tau.data(), rg.data(), &qldc, work.data(), &lwork,
                            &qinfo);
                    std::vector<double> delta(rg.begin(), rg.begin() + m);
                    char uplo = 'U', tn = 'N', diag = 'N';
                    int rn = m, rnrhs = 1, rlda = std::max(1, n_touch), rldb = m,
                        rinfo = 0;
                    dtrtrs_(&uplo, &tn, &diag, &rn, &rnrhs, blk.a_qr.data(), &rlda,
                            delta.data(), &rldb, &rinfo);
                    for (int k = 0; k < blk.n_base; ++k)
                        z[static_cast<std::size_t>(
                            blk.cols[static_cast<std::size_t>(k)])] +=
                            delta[static_cast<std::size_t>(k)];
                }
            };

            // Graph-distance halo: delta hops from base through shared rows.
            // interface_only restricts the contributing rows to TauMatch (the
            // measured interface channel). all_cols = base ++ halo. Scratch is
            // reused across blocks; each call leaves loc all -1 for build_block.
            std::vector<char> in_all(static_cast<std::size_t>(n_f), 0);
            std::vector<char> row_hit(static_cast<std::size_t>(nbr_conditions), 0);
            auto compute_halo = [&](const std::vector<int>& base, bool interface_only,
                                    int delta, std::vector<int>& all_cols) {
                std::fill(in_all.begin(), in_all.end(), 0);
                all_cols = base;
                for (int c : base)
                    in_all[static_cast<std::size_t>(c)] = 1;
                std::vector<int> frontier = base;
                for (int hop = 0; hop < delta && !frontier.empty(); ++hop) {
                    std::fill(row_hit.begin(), row_hit.end(), 0);
                    for (int c : frontier)
                        loc[static_cast<std::size_t>(c)] = 1;
                    for (long long e = 0; e < nnz_f; ++e) {
                        const int c = coo.jcn[static_cast<std::size_t>(e)] - 1;
                        if (loc[static_cast<std::size_t>(c)] != 1)
                            continue;
                        row_hit[static_cast<std::size_t>(
                            coo.irn[static_cast<std::size_t>(e)] - 1)] = 1;
                    }
                    for (int c : frontier)
                        loc[static_cast<std::size_t>(c)] = -1;
                    std::vector<int> next;
                    for (long long e = 0; e < nnz_f; ++e) {
                        const int r = coo.irn[static_cast<std::size_t>(e)] - 1;
                        if (!row_hit[static_cast<std::size_t>(r)])
                            continue;
                        if (interface_only &&
                            (r >= static_cast<int>(smoother_meta.size()) ||
                             smoother_meta[static_cast<std::size_t>(r)].taxonomy !=
                                 RowTaxonomy::TauMatch))
                            continue;
                        const int c = coo.jcn[static_cast<std::size_t>(e)] - 1;
                        if (in_all[static_cast<std::size_t>(c)])
                            continue;
                        in_all[static_cast<std::size_t>(c)] = 1;
                        all_cols.push_back(c);
                        next.push_back(c);
                    }
                    frontier.swap(next);
                }
            };
            // Count rows touched by a column list (sizing pass, no allocation).
            auto count_touch = [&](const std::vector<int>& cols) -> long long {
                std::fill(row_hit.begin(), row_hit.end(), 0);
                for (int c : cols)
                    loc[static_cast<std::size_t>(c)] = 1;
                long long nt = 0;
                for (long long e = 0; e < nnz_f; ++e) {
                    const int c = coo.jcn[static_cast<std::size_t>(e)] - 1;
                    if (loc[static_cast<std::size_t>(c)] != 1)
                        continue;
                    const int r = coo.irn[static_cast<std::size_t>(e)] - 1;
                    if (!row_hit[static_cast<std::size_t>(r)]) {
                        row_hit[static_cast<std::size_t>(r)] = 1;
                        ++nt;
                    }
                }
                for (int c : cols)
                    loc[static_cast<std::size_t>(c)] = -1;
                return nt;
            };

            // --- 1) Dirichlet smoother -> arm 8r baseline, then freed. ---
            const auto dir_t0 = std::chrono::steady_clock::now();
            std::vector<SchwarzBlock> dir_blocks;
            long long dir_doubles = 0;
            for (const auto& kv : dom_cols)
                build_block(dir_blocks, dir_doubles, kv.second,
                            static_cast<int>(kv.second.size()),
                            "dir-dom" + std::to_string(kv.first));
            build_block(dir_blocks, dir_doubles, global_cols,
                        static_cast<int>(global_cols.size()), "dir-global");
            std::cout << "  dirichlet smoother: " << dir_blocks.size()
                      << " blocks, a_qr ~"
                      << (static_cast<double>(dir_doubles) * 8.0 / 1e9)
                      << " GB, wall="
                      << std::chrono::duration<double>(
                             std::chrono::steady_clock::now() - dir_t0)
                             .count()
                      << "s\n";
            std::cout.flush();
            KrylovOperator pc_arm8r = [&](const std::vector<double>& r,
                                          std::vector<double>& z) {
                std::vector<double> sr, cc;
                schwarz_apply(dir_blocks, r, sr);
                coarse_correction(r, cc);
                z.assign(static_cast<std::size_t>(n_f), 0.0);
                for (int i = 0; i < n_f; ++i)
                    z[static_cast<std::size_t>(i)] = sr[static_cast<std::size_t>(i)] +
                                                     cc[static_cast<std::size_t>(i)];
            };
            run_arm("8r_dirichlet_additive", pc_arm8r);
            // Free the Dirichlet QR before building RAS: the two smoothers feed
            // disjoint arms, so peak QR = max(dir, ras), not the sum.
            std::vector<SchwarzBlock>().swap(dir_blocks);

            // --- 2) Size the requested RAS smoother (halo only, no QR yet). ---
            struct RasPlan {
                std::string label;
                std::vector<int> all_cols;
                int n_base = 0;
                long long n_touch = 0;
                long long doubles = 0;
            };
            const int ras_delta =
                std::max(1, env_int_value("PCOARSE_RAS_DELTA", 1));
            const double kCapGb = 16.0;
            auto size_ras = [&](bool interface_only,
                                std::vector<RasPlan>& plans) -> double {
                plans.clear();
                long long total = 0;
                for (const auto& kv : dom_cols) {
                    RasPlan p;
                    p.label = (interface_only ? "ras-i-dom" : "ras-dom") +
                              std::to_string(kv.first);
                    p.n_base = static_cast<int>(kv.second.size());
                    compute_halo(kv.second, interface_only,
                                 interface_only ? 1 : ras_delta, p.all_cols);
                    p.n_touch = count_touch(p.all_cols);
                    p.doubles =
                        p.n_touch * static_cast<long long>(p.all_cols.size());
                    total += p.doubles;
                    plans.push_back(std::move(p));
                }
                return static_cast<double>(total) * 8.0 / 1e9;
            };
            auto print_sizes = [&](const std::string& title,
                                   const std::vector<RasPlan>& plans, double gb) {
                std::cout << "=== " << title << " ===\n";
                for (const RasPlan& p : plans)
                    std::cout << "  " << p.label << " n_rows=" << p.n_touch
                              << " n_cols=" << p.all_cols.size() << " (base="
                              << p.n_base << " halo="
                              << (static_cast<int>(p.all_cols.size()) - p.n_base)
                              << ") QR="
                              << (static_cast<double>(p.doubles) * 8.0 / 1e9)
                              << " GB\n";
                std::cout << "  RAS QR total=" << gb << " GB (cap " << kCapGb
                          << " GB, dirichlet already freed)\n";
                std::cout.flush();
            };

            bool interface_mode = want_ras_interface;
            std::vector<RasPlan> ras_plans;
            double ras_gb = 0.0;
            if (!interface_mode) {
                ras_gb = size_ras(false, ras_plans);
                print_sizes("RAS delta=" + std::to_string(ras_delta) +
                                " halo sizing (per-domain, base+halo)",
                            ras_plans, ras_gb);
                if (ras_gb > kCapGb) {
                    std::vector<const RasPlan*> byd;
                    for (const RasPlan& p : ras_plans)
                        byd.push_back(&p);
                    std::sort(byd.begin(), byd.end(),
                              [](const RasPlan* a, const RasPlan* b) {
                                  return a->doubles > b->doubles;
                              });
                    std::cout << "  *** RAS delta=" << ras_delta << " busts the "
                              << kCapGb
                              << " GB cap; offending blocks (largest first): ***\n";
                    double acc = 0.0;
                    for (const RasPlan* p : byd) {
                        acc += static_cast<double>(p->doubles) * 8.0 / 1e9;
                        std::cout << "    " << p->label << " "
                                  << (static_cast<double>(p->doubles) * 8.0 / 1e9)
                                  << " GB (cumulative " << acc << " GB)\n";
                    }
                    std::cout << "  -> falling back to ras-interface only.\n";
                    std::cout.flush();
                    interface_mode = true;
                }
            }
            if (interface_mode) {
                ras_gb = size_ras(true, ras_plans);
                print_sizes("RAS-interface halo sizing (TauMatch-adjacent)",
                            ras_plans, ras_gb);
            }
            const bool ras_feasible = (ras_gb <= kCapGb);
            if (!ras_feasible)
                std::cout << "  *** ras-interface ALSO exceeds the " << kCapGb
                          << " GB cap; skipping RAS arms (only dirichlet 8r ran) "
                             "***\n";

            // --- 3) Build the chosen RAS smoother, run 9/9m or 10. ---
            if (ras_feasible) {
                const auto ras_t0 = std::chrono::steady_clock::now();
                std::vector<SchwarzBlock> ras_blocks;
                long long ras_doubles = 0;
                for (const RasPlan& p : ras_plans)
                    build_block(ras_blocks, ras_doubles, p.all_cols, p.n_base,
                                p.label);
                // global block (plain, scatter-all) mirrors the dirichlet global.
                build_block(ras_blocks, ras_doubles, global_cols,
                            static_cast<int>(global_cols.size()),
                            interface_mode ? "ras-i-global" : "ras-global");
                std::cout << "  RAS smoother: " << ras_blocks.size()
                          << " blocks, a_qr ~"
                          << (static_cast<double>(ras_doubles) * 8.0 / 1e9)
                          << " GB, wall="
                          << std::chrono::duration<double>(
                                 std::chrono::steady_clock::now() - ras_t0)
                                 .count()
                          << "s\n";
                std::cout.flush();

                // Arm 9 / 10: additive. z = S_ras(r) + P A_c^-1 R r.
                KrylovOperator pc_ras_add = [&](const std::vector<double>& r,
                                                std::vector<double>& z) {
                    std::vector<double> sr, cc;
                    schwarz_apply(ras_blocks, r, sr);
                    coarse_correction(r, cc);
                    z.assign(static_cast<std::size_t>(n_f), 0.0);
                    for (int i = 0; i < n_f; ++i)
                        z[static_cast<std::size_t>(i)] =
                            sr[static_cast<std::size_t>(i)] +
                            cc[static_cast<std::size_t>(i)];
                };
                if (interface_mode) {
                    run_arm("10_ras_interface_additive", pc_ras_add);
                } else {
                    run_arm("9_ras_additive", pc_ras_add);
                    // Arm 9m: multiplicative. z1=S_ras(r);
                    // z = z1 + P A_c^-1 R (r - A_f z1).
                    KrylovOperator pc_ras_mult = [&](const std::vector<double>& r,
                                                     std::vector<double>& z) {
                        std::vector<double> z1, az1, defl, cc;
                        schwarz_apply(ras_blocks, r, z1);
                        coo_spmv(z1, az1);
                        defl.assign(static_cast<std::size_t>(n_f), 0.0);
                        for (int i = 0; i < n_f; ++i)
                            defl[static_cast<std::size_t>(i)] =
                                r[static_cast<std::size_t>(i)] -
                                az1[static_cast<std::size_t>(i)];
                        coarse_correction(defl, cc);
                        z.assign(static_cast<std::size_t>(n_f), 0.0);
                        for (int i = 0; i < n_f; ++i)
                            z[static_cast<std::size_t>(i)] =
                                z1[static_cast<std::size_t>(i)] +
                                cc[static_cast<std::size_t>(i)];
                    };
                    run_arm("9m_ras_multiplicative", pc_ras_mult);
                }
            }
        } else {
            // ============================================================
            // ARM 6-8: fine-level Schwarz-Dirichlet smoother S composed with
            // the p-coarse two-level correction (the textbook two-level DD
            // method). Same construction as SCHUR_PC_GMRES, but the
            // per-aggregate blocks are infeasible at res11 (the bispheric
            // aggregate ~37.7k cols -> ~12 GB dense QR), so per-DOMAIN Dirichlet
            // blocks are used (sanctioned memory fallback).
            // ============================================================
            const auto s_setup_t0 = std::chrono::steady_clock::now();
            std::vector<ColumnInfo> cmap;
            build_column_map(cmap, false);
            std::map<int, std::vector<int>> dom_cols;
            std::vector<int> global_cols;
            for (int cc = 0; cc < n_f; ++cc) {
                const ColumnInfo& ci = cmap[static_cast<std::size_t>(cc)];
                if (ci.is_var_domain || ci.var_double_idx >= 0 || ci.domain < 0)
                    global_cols.push_back(cc);
                else
                    dom_cols[ci.domain].push_back(cc);
            }

            struct SchwarzBlock {
                int m = 0;
                int n_touch = 0;
                std::vector<double> a_qr; // factored A[touch, cols], col-major
                std::vector<double> tau;
                std::vector<int> touch_global;
                std::vector<int> cols;
            };
            std::vector<SchwarzBlock> blocks;
            std::vector<int> loc(static_cast<std::size_t>(n_f), -1);
            std::vector<int> touch_local(static_cast<std::size_t>(n_f), -1);
            long long a_qr_doubles = 0;
            auto add_block = [&](const std::vector<int>& cols,
                                 const std::string& label) {
                const int m = static_cast<int>(cols.size());
                if (m <= 0)
                    return;
                for (int k = 0; k < m; ++k)
                    loc[static_cast<std::size_t>(cols[static_cast<std::size_t>(k)])] = k;
                std::fill(touch_local.begin(), touch_local.end(), -1);
                int n_touch = 0;
                std::vector<int> touch_global;
                for (long long e = 0; e < nnz_f; ++e) {
                    if (loc[static_cast<std::size_t>(
                            coo.jcn[static_cast<std::size_t>(e)] - 1)] < 0)
                        continue;
                    const int r = coo.irn[static_cast<std::size_t>(e)] - 1;
                    if (touch_local[static_cast<std::size_t>(r)] < 0) {
                        touch_local[static_cast<std::size_t>(r)] = n_touch++;
                        touch_global.push_back(r);
                    }
                }
                const std::size_t nt = static_cast<std::size_t>(n_touch);
                std::vector<double> a_g(nt * static_cast<std::size_t>(m), 0.0);
                for (long long e = 0; e < nnz_f; ++e) {
                    const int lc = loc[static_cast<std::size_t>(
                        coo.jcn[static_cast<std::size_t>(e)] - 1)];
                    if (lc < 0)
                        continue;
                    const int r = coo.irn[static_cast<std::size_t>(e)] - 1;
                    a_g[static_cast<std::size_t>(
                            touch_local[static_cast<std::size_t>(r)]) +
                        static_cast<std::size_t>(lc) * nt] +=
                        coo.a[static_cast<std::size_t>(e)];
                }
                int qr_m = n_touch, qr_n = m, qr_lda = std::max(1, n_touch), info = 0;
                std::vector<double> tau(static_cast<std::size_t>(m));
                double wq = 0.0;
                int lwork = -1;
                dgeqrf_(&qr_m, &qr_n, a_g.data(), &qr_lda, tau.data(), &wq, &lwork,
                        &info);
                lwork = (info == 0) ? static_cast<int>(wq) : std::max(1, m);
                std::vector<double> work(static_cast<std::size_t>(std::max(1, lwork)));
                dgeqrf_(&qr_m, &qr_n, a_g.data(), &qr_lda, tau.data(), work.data(),
                        &lwork, &info);
                if (info != 0)
                    std::cerr << "  schwarz block " << label << " dgeqrf info=" << info
                              << '\n';
                a_qr_doubles += static_cast<long long>(nt) * m;
                SchwarzBlock blk;
                blk.m = m;
                blk.n_touch = n_touch;
                blk.a_qr = std::move(a_g);
                blk.tau = std::move(tau);
                blk.touch_global = std::move(touch_global);
                blk.cols = cols;
                blocks.push_back(std::move(blk));
                for (int k = 0; k < m; ++k)
                    loc[static_cast<std::size_t>(cols[static_cast<std::size_t>(k)])] =
                        -1;
                std::cout << "  schwarz block " << label << " m=" << m
                          << " n_touch=" << n_touch << " (a_qr GB so far ~"
                          << (static_cast<double>(a_qr_doubles) * 8.0 / 1e9) << ")\n";
                std::cout.flush();
            };
            for (const auto& kv : dom_cols)
                add_block(kv.second, "dom" + std::to_string(kv.first));
            add_block(global_cols, "global");
            const double s_setup_wall =
                std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                              s_setup_t0)
                    .count();
            std::cout << "  schwarz smoother setup: " << blocks.size()
                      << " per-domain blocks, a_qr ~"
                      << (static_cast<double>(a_qr_doubles) * 8.0 / 1e9)
                      << " GB, wall=" << s_setup_wall << "s\n";
            std::cout.flush();

            // S(r): additive per-block Dirichlet QR least-squares.
            auto schwarz_apply = [&](const std::vector<double>& r,
                                     std::vector<double>& z) {
                z.assign(static_cast<std::size_t>(n_f), 0.0);
                for (SchwarzBlock& blk : blocks) {
                    const int m = blk.m, n_touch = blk.n_touch;
                    if (m <= 0 || n_touch <= 0)
                        continue;
                    std::vector<double> rg(static_cast<std::size_t>(n_touch));
                    for (int i = 0; i < n_touch; ++i)
                        rg[static_cast<std::size_t>(i)] = r[static_cast<std::size_t>(
                            blk.touch_global[static_cast<std::size_t>(i)])];
                    char side = 'L', trans = 'T';
                    int qm = n_touch, qn = 1, qk = m, qlda = std::max(1, n_touch),
                        qldc = std::max(1, n_touch), qinfo = 0, lwork = -1;
                    double wq = 0.0;
                    dormqr_(&side, &trans, &qm, &qn, &qk, blk.a_qr.data(), &qlda,
                            blk.tau.data(), rg.data(), &qldc, &wq, &lwork, &qinfo);
                    lwork = (qinfo == 0) ? static_cast<int>(wq) : std::max(1, n_touch);
                    std::vector<double> work(
                        static_cast<std::size_t>(std::max(1, lwork)));
                    dormqr_(&side, &trans, &qm, &qn, &qk, blk.a_qr.data(), &qlda,
                            blk.tau.data(), rg.data(), &qldc, work.data(), &lwork,
                            &qinfo);
                    std::vector<double> delta(rg.begin(), rg.begin() + m);
                    char uplo = 'U', tn = 'N', diag = 'N';
                    int rn = m, rnrhs = 1, rlda = std::max(1, n_touch), rldb = m,
                        rinfo = 0;
                    dtrtrs_(&uplo, &tn, &diag, &rn, &rnrhs, blk.a_qr.data(), &rlda,
                            delta.data(), &rldb, &rinfo);
                    for (int k = 0; k < m; ++k)
                        z[static_cast<std::size_t>(
                            blk.cols[static_cast<std::size_t>(k)])] +=
                            delta[static_cast<std::size_t>(k)];
                }
            };

            // Arm 6: multiplicative pre-smooth. z1=S(r); z=z1+P A_c^-1 R (r - A_f z1).
            KrylovOperator pc_arm6 = [&](const std::vector<double>& r,
                                         std::vector<double>& z) {
                std::vector<double> z1, az1, defl, cc;
                schwarz_apply(r, z1);
                coo_spmv(z1, az1);
                defl.assign(static_cast<std::size_t>(n_f), 0.0);
                for (int i = 0; i < n_f; ++i)
                    defl[static_cast<std::size_t>(i)] = r[static_cast<std::size_t>(i)] -
                                                        az1[static_cast<std::size_t>(i)];
                coarse_correction(defl, cc);
                z.assign(static_cast<std::size_t>(n_f), 0.0);
                for (int i = 0; i < n_f; ++i)
                    z[static_cast<std::size_t>(i)] = z1[static_cast<std::size_t>(i)] +
                                                     cc[static_cast<std::size_t>(i)];
            };
            // Arm 7: pre + post smooth. z = arm6; z += S(r - A_f z).
            KrylovOperator pc_arm7 = [&](const std::vector<double>& r,
                                         std::vector<double>& z) {
                pc_arm6(r, z);
                std::vector<double> az, defl, s2;
                coo_spmv(z, az);
                defl.assign(static_cast<std::size_t>(n_f), 0.0);
                for (int i = 0; i < n_f; ++i)
                    defl[static_cast<std::size_t>(i)] = r[static_cast<std::size_t>(i)] -
                                                        az[static_cast<std::size_t>(i)];
                schwarz_apply(defl, s2);
                for (int i = 0; i < n_f; ++i)
                    z[static_cast<std::size_t>(i)] += s2[static_cast<std::size_t>(i)];
            };
            // Arm 8: additive control. z = S(r) + P A_c^-1 R r.
            KrylovOperator pc_arm8 = [&](const std::vector<double>& r,
                                         std::vector<double>& z) {
                std::vector<double> sr, cc;
                schwarz_apply(r, sr);
                coarse_correction(r, cc);
                z.assign(static_cast<std::size_t>(n_f), 0.0);
                for (int i = 0; i < n_f; ++i)
                    z[static_cast<std::size_t>(i)] = sr[static_cast<std::size_t>(i)] +
                                                     cc[static_cast<std::size_t>(i)];
            };

            const bool trace_on = env_flag_enabled("PCOARSE_TRACE", false);
            if (!trace_on) {
                run_arm("6_schwarz_precoarse_mult", pc_arm6);
                run_arm("7_schwarz_pre_post_mult", pc_arm7);
                run_arm("8_schwarz_plus_coarse_additive", pc_arm8);
            } else {
                // ========================================================
                // DtN/trace composite arm (Stage A diagnostic -> gate ->
                // Stage B arms 11/12). See PCOARSE_DTN_COMPOSITE_ARM_SPEC.
                // S(r) here is the per-domain Dirichlet smoother above.
                // ========================================================
                std::vector<RowMetadata> tmeta;
                classify_equation_row_metadata(tmeta);

                // ====================================================================
                // ARM 16: smoother-side levers (PCOARSE_ARM16). Arm-13 proved
                // the 8r2 residual, not the error, lives in TauMatch (interface) and
                // high-p rows. Every TauMatch row sits in the touch-row set of BOTH
                // adjacent domain blocks, so additive Dirichlet Schwarz corrects each
                // seam row twice with the neighbour's columns frozen at 0 -- a classic
                // frozen-boundary double-correction that caps the slope. This arm
                // tests three cheap fixes on the smoother, all composed with the SAME
                // additive p-coarse correction as arm 8:
                //   diag  -- per-block seam double-correction instrument (arm-8 apply)
                //   16a   -- exclusive TauMatch row ownership (owner = block of
                //            tmeta.dom; the other adjacent block drops the row from its
                //            least-squares set). Blocks re-QR'd.
                //   16b   -- block Gauss-Seidel: multiplicative sweep dom0..global on
                //            the running residual, then additive coarse as arm 8.
                //   16c   -- row equilibration: run the whole arm on the row-scaled
                //            system A~=D A, b~=D b (D=diag(1/||row||2)); report scaled
                //            AND unscaled true rel.
                //   16d   -- 16a exclusive rows + 16b Gauss-Seidel combined.
                // Self-contained; factors the fine solver once (reused for attrib),
                // builds up to three block sets sequentially (baseline reused, then
                // 16a-exclusive, then 16c-scaled), freeing each before the next to keep
                // peak QR storage at one set. Returns before the Stage-A flow.
                // ====================================================================
                if (env_flag_enabled("PCOARSE_ARM16", false)) {
                    std::cout << "\n############ ARM 16: smoother-side levers "
                                 "(exclusive seam rows / block GS / row equilibration) "
                                 "############\n";
                    std::cout.flush();

                    const std::size_t size_n = static_cast<std::size_t>(n_f);

                    // --- Column / row masks (same predicates arms 14/15 use) ---
                    std::vector<char> is_global(size_n, 0);
                    for (int cc = 0; cc < n_f; ++cc) {
                        const ColumnInfo& ci = cmap[static_cast<std::size_t>(cc)];
                        if (ci.is_var_domain || ci.var_double_idx >= 0 ||
                            ci.domain < 0)
                            is_global[static_cast<std::size_t>(cc)] = 1;
                    }
                    std::vector<char> is_tm_row(size_n, 0);
                    for (int rr = 0; rr < n_f; ++rr)
                        if (rr < static_cast<int>(tmeta.size()) &&
                            tmeta[static_cast<std::size_t>(rr)].taxonomy ==
                                RowTaxonomy::TauMatch)
                            is_tm_row[static_cast<std::size_t>(rr)] = 1;
                    std::vector<char> col_seam(size_n, 0);
                    for (long long e = 0; e < nnz_f; ++e) {
                        const std::size_t ee = static_cast<std::size_t>(e);
                        if (is_tm_row[static_cast<std::size_t>(coo.irn[ee] - 1)])
                            col_seam[static_cast<std::size_t>(coo.jcn[ee] - 1)] = 1;
                    }

                    // --- CSC of the fine matrix (column -> (row,val)) for cheap
                    // per-block residual updates in the Gauss-Seidel sweeps and the
                    // solo-block spmv in the diagnostic. Columns partition across the
                    // 13 blocks, so a full GS sweep costs one pass over nnz. ---
                    std::vector<long long> csc_ptr(size_n + 1, 0);
                    for (long long e = 0; e < nnz_f; ++e)
                        ++csc_ptr[static_cast<std::size_t>(coo.jcn[static_cast<
                                      std::size_t>(e)])];
                    for (std::size_t c = 0; c < size_n; ++c)
                        csc_ptr[c + 1] += csc_ptr[c];
                    std::vector<int> csc_row(static_cast<std::size_t>(nnz_f));
                    std::vector<double> csc_val(static_cast<std::size_t>(nnz_f));
                    {
                        std::vector<long long> cursor(csc_ptr.begin(),
                                                      csc_ptr.end() - 1);
                        for (long long e = 0; e < nnz_f; ++e) {
                            const std::size_t ee = static_cast<std::size_t>(e);
                            const int c = coo.jcn[ee] - 1;
                            const long long p = cursor[static_cast<std::size_t>(c)]++;
                            csc_row[static_cast<std::size_t>(p)] = coo.irn[ee] - 1;
                            csc_val[static_cast<std::size_t>(p)] = coo.a[ee];
                        }
                    }

                    // --- Fine factor FIRST (reused for attribution of every arm) ---
                    const auto tfac0 = std::chrono::steady_clock::now();
                    MumpsLinearSolver fine_solver(
                        n_f, config.mumps.ordering, false, 0,
                        mumps_runtime_state.icntl14, MPI_COMM_SELF);
                    fine_solver.set_pattern(n_f, nnz_f, coo.irn.data(),
                                            coo.jcn.data());
                    fine_solver.analyze_pattern();
                    fine_solver.factor_analyzed(coo.a.data());
                    mumps_runtime_state.icntl14 = fine_solver.last_icntl14();
                    std::cout << "  (setup) fine MUMPS factor wall="
                              << std::chrono::duration<double>(
                                     std::chrono::steady_clock::now() - tfac0)
                                     .count()
                              << "s\n";
                    std::cout.flush();

                    auto refined_solve =
                        [&](const std::vector<double>& r) -> std::vector<double> {
                        std::vector<double> e(r);
                        fine_solver.solve(e.data());
                        std::vector<double> ae;
                        coo_spmv(e, ae);
                        std::vector<double> corr(size_n);
                        for (int i = 0; i < n_f; ++i)
                            corr[static_cast<std::size_t>(i)] =
                                r[static_cast<std::size_t>(i)] -
                                ae[static_cast<std::size_t>(i)];
                        fine_solver.solve(corr.data());
                        for (int i = 0; i < n_f; ++i)
                            e[static_cast<std::size_t>(i)] +=
                                corr[static_cast<std::size_t>(i)];
                        return e;
                    };
                    struct ArmShare {
                        double global;
                        double seam;
                        double hi;
                        double solve_rel;
                    };
                    // e = A^-1 (b - A x); shares of ||e||^2 in global / seam / high-p
                    // columns (identical definition to arm 15d/e).
                    auto attrib_share =
                        [&](const std::vector<double>& xarm) -> ArmShare {
                        std::vector<double> ax;
                        coo_spmv(xarm, ax);
                        std::vector<double> r(size_n);
                        for (int i = 0; i < n_f; ++i)
                            r[static_cast<std::size_t>(i)] =
                                b[static_cast<std::size_t>(i)] -
                                ax[static_cast<std::size_t>(i)];
                        std::vector<double> e = refined_solve(r);
                        std::vector<double> ae;
                        coo_spmv(e, ae);
                        double num = 0.0, den = 0.0, tot = 0.0, gl = 0.0, sm = 0.0,
                               hp = 0.0;
                        for (int i = 0; i < n_f; ++i) {
                            const double d = ae[static_cast<std::size_t>(i)] -
                                             r[static_cast<std::size_t>(i)];
                            num += d * d;
                            den += r[static_cast<std::size_t>(i)] *
                                   r[static_cast<std::size_t>(i)];
                            const double e2 = e[static_cast<std::size_t>(i)] *
                                              e[static_cast<std::size_t>(i)];
                            tot += e2;
                            if (is_global[static_cast<std::size_t>(i)])
                                gl += e2;
                            if (col_seam[static_cast<std::size_t>(i)])
                                sm += e2;
                            if (fine_col_matched[static_cast<std::size_t>(i)] == 0)
                                hp += e2;
                        }
                        const double inv = (tot > 0.0) ? 1.0 / tot : 0.0;
                        return {gl * inv, sm * inv, hp * inv,
                                (den > 0.0) ? std::sqrt(num / den) : 0.0};
                    };

                    // --- Per-block QR least-squares solve (gathers r at touch rows,
                    // returns delta on the block's columns). Mirrors schwarz_apply's
                    // inner loop; non-const block ref because LAPACK wants double*. ---
                    auto solve_block =
                        [&](SchwarzBlock& blk,
                            const std::vector<double>& rfull) -> std::vector<double> {
                        const int m = blk.m, nt = blk.n_touch;
                        // Defensive: an underdetermined block (fewer touch rows than
                        // columns) has no overdetermined-QR least-squares solve; the
                        // dormqr/dtrtrs calls below would be illegal (side='L' needs
                        // M>=K, LDA>=N). Callers must never run arms on such a set;
                        // return a zero correction rather than invoke UB.
                        if (nt < m)
                            return std::vector<double>(
                                static_cast<std::size_t>(std::max(0, m)), 0.0);
                        std::vector<double> rg(static_cast<std::size_t>(nt));
                        for (int i = 0; i < nt; ++i)
                            rg[static_cast<std::size_t>(i)] =
                                rfull[static_cast<std::size_t>(
                                    blk.touch_global[static_cast<std::size_t>(i)])];
                        char side = 'L', trans = 'T';
                        int qm = nt, qn = 1, qk = m, qlda = std::max(1, nt),
                            qldc = std::max(1, nt), qinfo = 0, lwork = -1;
                        double wq = 0.0;
                        dormqr_(&side, &trans, &qm, &qn, &qk, blk.a_qr.data(), &qlda,
                                blk.tau.data(), rg.data(), &qldc, &wq, &lwork, &qinfo);
                        lwork = (qinfo == 0) ? static_cast<int>(wq) : std::max(1, nt);
                        std::vector<double> work(
                            static_cast<std::size_t>(std::max(1, lwork)));
                        dormqr_(&side, &trans, &qm, &qn, &qk, blk.a_qr.data(), &qlda,
                                blk.tau.data(), rg.data(), &qldc, work.data(), &lwork,
                                &qinfo);
                        std::vector<double> delta(rg.begin(), rg.begin() + m);
                        char uplo = 'U', tn = 'N', diag = 'N';
                        int rn = m, rnrhs = 1, rlda = std::max(1, nt), rldb = m,
                            rinfo = 0;
                        dtrtrs_(&uplo, &tn, &diag, &rn, &rnrhs, blk.a_qr.data(), &rlda,
                                delta.data(), &rldb, &rinfo);
                        return delta;
                    };
                    // work -= A * (delta scattered on blk.cols), via CSC (only the
                    // block's columns contribute).
                    auto csc_block_update = [&](const SchwarzBlock& blk,
                                                const std::vector<double>& delta,
                                                std::vector<double>& work) {
                        for (int j = 0; j < blk.m; ++j) {
                            const int col = blk.cols[static_cast<std::size_t>(j)];
                            const double dv = delta[static_cast<std::size_t>(j)];
                            if (dv == 0.0)
                                continue;
                            for (long long p = csc_ptr[static_cast<std::size_t>(col)];
                                 p < csc_ptr[static_cast<std::size_t>(col) + 1]; ++p)
                                work[static_cast<std::size_t>(
                                    csc_row[static_cast<std::size_t>(p)])] -=
                                    csc_val[static_cast<std::size_t>(p)] * dv;
                        }
                    };
                    auto all_finite = [](const std::vector<double>& v) -> bool {
                        for (double x : v)
                            if (!std::isfinite(x))
                                return false;
                        return true;
                    };

                    // --- Additive smoother over an arbitrary block set + coarse. ---
                    auto pc_additive_set = [&](std::vector<SchwarzBlock>& blks,
                                               const std::vector<double>& r,
                                               std::vector<double>& z) {
                        z.assign(size_n, 0.0);
                        for (SchwarzBlock& blk : blks) {
                            if (blk.m <= 0 || blk.n_touch <= 0)
                                continue;
                            std::vector<double> delta = solve_block(blk, r);
                            for (int k = 0; k < blk.m; ++k)
                                z[static_cast<std::size_t>(
                                    blk.cols[static_cast<std::size_t>(k)])] +=
                                    delta[static_cast<std::size_t>(k)];
                        }
                        std::vector<double> cc;
                        coarse_correction(r, cc);
                        for (int i = 0; i < n_f; ++i)
                            z[static_cast<std::size_t>(i)] +=
                                cc[static_cast<std::size_t>(i)];
                    };
                    // --- Block Gauss-Seidel smoother over a block set + additive
                    // coarse (as arm 8). print_first: dump the running residual norm
                    // (total + seam) after each block on the FIRST apply. ---
                    auto seam_norm = [&](const std::vector<double>& v) -> double {
                        double s = 0.0;
                        for (int i = 0; i < n_f; ++i)
                            if (is_tm_row[static_cast<std::size_t>(i)])
                                s += v[static_cast<std::size_t>(i)] *
                                     v[static_cast<std::size_t>(i)];
                        return std::sqrt(s);
                    };
                    auto full_norm = [&](const std::vector<double>& v) -> double {
                        double s = 0.0;
                        for (int i = 0; i < n_f; ++i)
                            s += v[static_cast<std::size_t>(i)] *
                                 v[static_cast<std::size_t>(i)];
                        return std::sqrt(s);
                    };
                    auto pc_gs_set = [&](std::vector<SchwarzBlock>& blks,
                                         const std::vector<double>& r,
                                         std::vector<double>& z, bool* print_first,
                                         const std::string& label) {
                        z.assign(size_n, 0.0);
                        std::vector<double> work(r);
                        const bool doprint = (print_first && *print_first);
                        if (doprint)
                            std::cout << "  [" << label
                                      << " GS first-apply per-block running residual] "
                                         "start ||work||="
                                      << full_norm(work)
                                      << " ||work||_seam=" << seam_norm(work) << '\n';
                        int bidx = 0;
                        for (SchwarzBlock& blk : blks) {
                            if (blk.m > 0 && blk.n_touch > 0) {
                                std::vector<double> delta = solve_block(blk, work);
                                for (int k = 0; k < blk.m; ++k)
                                    z[static_cast<std::size_t>(
                                        blk.cols[static_cast<std::size_t>(k)])] +=
                                        delta[static_cast<std::size_t>(k)];
                                csc_block_update(blk, delta, work);
                            }
                            if (doprint)
                                std::cout << "    after block " << bidx
                                          << " ||work||=" << full_norm(work)
                                          << " ||work||_seam=" << seam_norm(work)
                                          << '\n';
                            ++bidx;
                        }
                        std::vector<double> cc;
                        coarse_correction(r, cc);
                        for (int i = 0; i < n_f; ++i)
                            z[static_cast<std::size_t>(i)] +=
                                cc[static_cast<std::size_t>(i)];
                        if (print_first && *print_first) {
                            std::cout.flush();
                            *print_first = false;
                        }
                    };

                    // --- Build a block set from column groups, optional exclusive
                    // TauMatch ownership, optional row-scaled A values. Returns false
                    // if any block's dgeqrf failed. ---
                    // avals is a raw pointer (nnz_f doubles) so it binds to both
                    // coo.a (custom allocator) and the plain a_scaled vector.
                    auto build_blockset = [&](std::vector<SchwarzBlock>& out,
                                              const double* avals,
                                              bool exclusive_tau, int* owned_tau,
                                              int* orphan_tau,
                                              int* underdet) -> bool {
                        out.clear();
                        bool ok = true;
                        int underdet_local = 0;
                        long long dbl = 0;
                        std::vector<int> loc2(size_n, -1), tl(size_n, -1);
                        std::vector<char> tau_seen;
                        if (exclusive_tau)
                            tau_seen.assign(size_n, 0);
                        auto add = [&](const std::vector<int>& cols, int owner_dom,
                                       const std::string& label) {
                            const int m = static_cast<int>(cols.size());
                            if (m <= 0)
                                return;
                            for (int k = 0; k < m; ++k)
                                loc2[static_cast<std::size_t>(
                                    cols[static_cast<std::size_t>(k)])] = k;
                            std::fill(tl.begin(), tl.end(), -1);
                            int nt = 0;
                            std::vector<int> tg;
                            for (long long e = 0; e < nnz_f; ++e) {
                                const std::size_t ee = static_cast<std::size_t>(e);
                                if (loc2[static_cast<std::size_t>(coo.jcn[ee] - 1)] <
                                    0)
                                    continue;
                                const int r = coo.irn[ee] - 1;
                                if (exclusive_tau &&
                                    is_tm_row[static_cast<std::size_t>(r)] &&
                                    tmeta[static_cast<std::size_t>(r)].dom !=
                                        owner_dom)
                                    continue;
                                if (tl[static_cast<std::size_t>(r)] < 0) {
                                    tl[static_cast<std::size_t>(r)] = nt++;
                                    tg.push_back(r);
                                    if (exclusive_tau &&
                                        is_tm_row[static_cast<std::size_t>(r)]) {
                                        tau_seen[static_cast<std::size_t>(r)] = 1;
                                        if (owned_tau)
                                            ++(*owned_tau);
                                    }
                                }
                            }
                            const std::size_t nts = static_cast<std::size_t>(nt);
                            if (nt < m)
                                ++underdet_local;
                            std::vector<double> a_g(
                                nts * static_cast<std::size_t>(m), 0.0);
                            for (long long e = 0; e < nnz_f; ++e) {
                                const std::size_t ee = static_cast<std::size_t>(e);
                                const int lc =
                                    loc2[static_cast<std::size_t>(coo.jcn[ee] - 1)];
                                if (lc < 0)
                                    continue;
                                const int r = coo.irn[ee] - 1;
                                if (exclusive_tau &&
                                    is_tm_row[static_cast<std::size_t>(r)] &&
                                    tmeta[static_cast<std::size_t>(r)].dom !=
                                        owner_dom)
                                    continue;
                                a_g[static_cast<std::size_t>(
                                        tl[static_cast<std::size_t>(r)]) +
                                    static_cast<std::size_t>(lc) * nts] +=
                                    avals[ee];
                            }
                            int qr_m = nt, qr_n = m, qr_lda = std::max(1, nt),
                                info = 0;
                            std::vector<double> tau(static_cast<std::size_t>(m));
                            double wq = 0.0;
                            int lwork = -1;
                            dgeqrf_(&qr_m, &qr_n, a_g.data(), &qr_lda, tau.data(),
                                    &wq, &lwork, &info);
                            lwork = (info == 0) ? static_cast<int>(wq)
                                                : std::max(1, m);
                            std::vector<double> work(
                                static_cast<std::size_t>(std::max(1, lwork)));
                            dgeqrf_(&qr_m, &qr_n, a_g.data(), &qr_lda, tau.data(),
                                    work.data(), &lwork, &info);
                            if (info != 0) {
                                std::cerr << "  arm16 block " << label
                                          << " dgeqrf info=" << info << '\n';
                                ok = false;
                            }
                            dbl += static_cast<long long>(nt) * m;
                            SchwarzBlock blk;
                            blk.m = m;
                            blk.n_touch = nt;
                            blk.a_qr = std::move(a_g);
                            blk.tau = std::move(tau);
                            blk.touch_global = std::move(tg);
                            blk.cols = cols;
                            out.push_back(std::move(blk));
                            for (int k = 0; k < m; ++k)
                                loc2[static_cast<std::size_t>(
                                    cols[static_cast<std::size_t>(k)])] = -1;
                        };
                        const auto t0 = std::chrono::steady_clock::now();
                        for (const auto& kv : dom_cols)
                            add(kv.second, kv.first,
                                "dom" + std::to_string(kv.first));
                        add(global_cols, -1, "global");
                        if (exclusive_tau && orphan_tau) {
                            int orph = 0;
                            for (int r = 0; r < n_f; ++r)
                                if (is_tm_row[static_cast<std::size_t>(r)] &&
                                    tau_seen[static_cast<std::size_t>(r)] == 0)
                                    ++orph;
                            *orphan_tau = orph;
                        }
                        if (underdet)
                            *underdet = underdet_local;
                        std::cout << "  arm16 blockset built: " << out.size()
                                  << " blocks, a_qr ~"
                                  << (static_cast<double>(dbl) * 8.0 / 1e9)
                                  << " GB, wall="
                                  << std::chrono::duration<double>(
                                         std::chrono::steady_clock::now() - t0)
                                         .count()
                                  << "s\n";
                        std::cout.flush();
                        return ok;
                    };

                    // ============================================================
                    // DIAGNOSTIC: per-block seam double-correction on an additive
                    // arm-8 apply of r=b, over the baseline blocks (outer `blocks`).
                    // For each block: ||r|| on its touch rows and on its TauMatch
                    // touch rows, comparing (before) b, (solo) the block acting alone
                    // b - A z_k, and (additive) after all blocks b - A z_full.
                    // additive_seam > before_seam flags a block whose seam rows are
                    // WORSENED once the neighbours' overlapping corrections land.
                    // ============================================================
                    std::cout << "\n  === DIAGNOSTIC: per-block seam double-correction "
                                 "(arm-8 additive apply on r=b) ===\n";
                    std::vector<double> z_full, az_full, r_full(size_n);
                    schwarz_apply(b, z_full); // baseline additive smoother S(b)
                    coo_spmv(z_full, az_full);
                    for (int i = 0; i < n_f; ++i)
                        r_full[static_cast<std::size_t>(i)] =
                            b[static_cast<std::size_t>(i)] -
                            az_full[static_cast<std::size_t>(i)];
                    std::cout << "  block | label | m | n_touch | n_seam_touch | "
                                 "||r||_touch(before/solo/add) | "
                                 "||r||_seam(before/solo/add) | worsens_seam\n";
                    int diag_worsen = 0;
                    for (std::size_t bk = 0; bk < blocks.size(); ++bk) {
                        SchwarzBlock& blk = blocks[bk];
                        if (blk.m <= 0 || blk.n_touch <= 0)
                            continue;
                        std::string label;
                        {
                            const ColumnInfo& ci = cmap[static_cast<std::size_t>(
                                blk.cols[0])];
                            label = (ci.is_var_domain || ci.var_double_idx >= 0 ||
                                     ci.domain < 0)
                                        ? std::string("global")
                                        : "dom" + std::to_string(ci.domain);
                        }
                        // solo apply of this block only
                        std::vector<double> delta = solve_block(blk, b);
                        std::vector<double> work_solo(b);
                        csc_block_update(blk, delta, work_solo); // b - A z_k
                        int n_seam = 0;
                        double bef2 = 0, solo2 = 0, add2 = 0, befs = 0, solos = 0,
                               adds = 0;
                        for (int t = 0; t < blk.n_touch; ++t) {
                            const int r =
                                blk.touch_global[static_cast<std::size_t>(t)];
                            const std::size_t rs = static_cast<std::size_t>(r);
                            const double bv = b[rs], sv = work_solo[rs],
                                         av = r_full[rs];
                            bef2 += bv * bv;
                            solo2 += sv * sv;
                            add2 += av * av;
                            if (is_tm_row[rs]) {
                                ++n_seam;
                                befs += bv * bv;
                                solos += sv * sv;
                                adds += av * av;
                            }
                        }
                        const bool worsen = (std::sqrt(adds) >
                                             std::sqrt(befs) * (1.0 + 1e-12));
                        if (worsen)
                            ++diag_worsen;
                        std::cout << "  " << bk << " | " << label << " | " << blk.m
                                  << " | " << blk.n_touch << " | " << n_seam << " | "
                                  << std::sqrt(bef2) << "/" << std::sqrt(solo2) << "/"
                                  << std::sqrt(add2) << " | " << std::sqrt(befs)
                                  << "/" << std::sqrt(solos) << "/" << std::sqrt(adds)
                                  << " | " << (worsen ? "YES" : "no") << '\n';
                    }
                    std::cout << "  DIAGNOSTIC: " << diag_worsen
                              << " blocks worsen their own seam rows under additive "
                                 "composition (double-correction signature)\n";
                    std::cout.flush();

                    // ============================================================
                    // RUN ARMS. arms[] indices from a_first onward belong to arm 16.
                    // ============================================================
                    const std::size_t a_first = arms.size();
                    // (1) baseline arm-8 additive (uses outer blocks) -- in-run guard
                    run_arm("8r2_baseline", pc_arm8);

                    // (2) 16b: block Gauss-Seidel on the BASELINE blocks + coarse
                    bool gs_print_16b = true;
                    KrylovOperator pc_16b = [&](const std::vector<double>& r,
                                                std::vector<double>& z) {
                        pc_gs_set(blocks, r, z, &gs_print_16b, "16b");
                    };
                    {
                        // The guard apply on r=b both checks finiteness and prints
                        // the per-block GS trajectory once (sets gs_print_16b=false),
                        // so the GMRES run below does not re-print.
                        std::vector<double> zt;
                        pc_16b(b, zt);
                        if (all_finite(zt))
                            run_arm("16b_block_gauss_seidel", pc_16b);
                        else
                            std::cout << "  16b SKIPPED (non-finite PC apply)\n";
                    }

                    // Free the baseline blocks before building 16a (peak = one set).
                    std::vector<SchwarzBlock>().swap(blocks);

                    // (3) 16a: exclusive TauMatch row ownership, additive + coarse
                    std::vector<SchwarzBlock> blocks16a;
                    int owned16a = 0, orphan16a = 0, underdet16a = 0;
                    const bool ok16a =
                        build_blockset(blocks16a, coo.a.data(), true, &owned16a,
                                       &orphan16a, &underdet16a);
                    std::cout << "  16a exclusive-row ownership: owned_tau_rows="
                              << owned16a << " orphan_tau_rows=" << orphan16a
                              << " underdet_blocks(n_touch<m)=" << underdet16a
                              << " (blocks_ok=" << (ok16a ? 1 : 0) << ")\n";
                    std::cout.flush();
                    // Exclusive assignment drops each seam row from one adjacent
                    // block's least-squares row set; if that pushes any block below
                    // its column count (n_touch < m) the per-block LS is
                    // underdetermined -- no overdetermined-QR solution exists and the
                    // additive smoother is ill-posed. Refuse to run garbage GMRES on
                    // it; report the structural refutation.
                    const bool wellposed16a = ok16a && (underdet16a == 0);
                    KrylovOperator pc_16a = [&](const std::vector<double>& r,
                                                std::vector<double>& z) {
                        pc_additive_set(blocks16a, r, z);
                    };
                    bool gs_print_16d = true;
                    KrylovOperator pc_16d = [&](const std::vector<double>& r,
                                                std::vector<double>& z) {
                        pc_gs_set(blocks16a, r, z, &gs_print_16d, "16d");
                    };
                    if (wellposed16a) {
                        std::vector<double> zt;
                        pc_16a(b, zt);
                        if (all_finite(zt))
                            run_arm("16a_exclusive_seam_rows", pc_16a);
                        else
                            std::cout << "  16a SKIPPED (non-finite PC apply)\n";
                        // (5) 16d: 16a exclusive rows under Gauss-Seidel. Guard
                        // apply prints the trajectory once and checks finiteness.
                        std::vector<double> zt2;
                        pc_16d(b, zt2);
                        if (all_finite(zt2))
                            run_arm("16d_exclusive_plus_gs", pc_16d);
                        else
                            std::cout << "  16d SKIPPED (non-finite PC apply)\n";
                    } else if (!ok16a) {
                        std::cout << "  16a/16d SKIPPED (blockset QR failed)\n";
                    } else {
                        std::cout << "  16a/16d SKIPPED -- exclusive assignment is "
                                     "STRUCTURALLY ILL-POSED: " << underdet16a
                                  << "/" << blocks16a.size()
                                  << " blocks underdetermined (n_touch<m after "
                                     "dropping neighbour-owned seam rows); no "
                                     "well-posed per-block least-squares exists. "
                                     "Exclusive-seam-row lever REFUTED.\n";
                    }
                    std::vector<SchwarzBlock>().swap(blocks16a);

                    // (4) 16c: row equilibration -- run the WHOLE arm on A~=D A,
                    // b~=D b with D=diag(1/||row||2). x converges to the same true
                    // solution; report scaled AND unscaled true rel.
                    std::cout << "\n  === 16c: row-equilibrated system (A~=D A, "
                                 "b~=D b) ===\n";
                    std::vector<double> rownorm(size_n, 0.0);
                    for (long long e = 0; e < nnz_f; ++e) {
                        const std::size_t ee = static_cast<std::size_t>(e);
                        const int r = coo.irn[ee] - 1;
                        rownorm[static_cast<std::size_t>(r)] +=
                            coo.a[ee] * coo.a[ee];
                    }
                    std::vector<double> dscale(size_n, 1.0);
                    int zero_rows = 0;
                    for (int i = 0; i < n_f; ++i) {
                        const double rn = std::sqrt(rownorm[static_cast<std::size_t>(
                            i)]);
                        if (rn > 0.0)
                            dscale[static_cast<std::size_t>(i)] = 1.0 / rn;
                        else
                            ++zero_rows;
                    }
                    std::vector<double> a_scaled(static_cast<std::size_t>(nnz_f));
                    for (long long e = 0; e < nnz_f; ++e) {
                        const std::size_t ee = static_cast<std::size_t>(e);
                        a_scaled[ee] = coo.a[ee] *
                                       dscale[static_cast<std::size_t>(
                                           coo.irn[ee] - 1)];
                    }
                    std::vector<double> b_scaled(size_n);
                    double b_scaled_norm = 0.0;
                    for (int i = 0; i < n_f; ++i) {
                        b_scaled[static_cast<std::size_t>(i)] =
                            b[static_cast<std::size_t>(i)] *
                            dscale[static_cast<std::size_t>(i)];
                        b_scaled_norm += b_scaled[static_cast<std::size_t>(i)] *
                                         b_scaled[static_cast<std::size_t>(i)];
                    }
                    b_scaled_norm = std::sqrt(b_scaled_norm);
                    std::cout << "  zero_rows(guarded to D=1)=" << zero_rows
                              << " ||b~||=" << b_scaled_norm << '\n';
                    auto spmv_scaled = [&](const std::vector<double>& v,
                                           std::vector<double>& out) {
                        out.assign(size_n, 0.0);
                        for (long long e = 0; e < nnz_f; ++e) {
                            const std::size_t ee = static_cast<std::size_t>(e);
                            out[static_cast<std::size_t>(coo.irn[ee] - 1)] +=
                                a_scaled[ee] *
                                v[static_cast<std::size_t>(coo.jcn[ee] - 1)];
                        }
                    };
                    std::vector<SchwarzBlock> blocks16c;
                    const bool ok16c =
                        build_blockset(blocks16c, a_scaled.data(), false, nullptr,
                                       nullptr, nullptr);
                    // scaled additive PC: scaled smoother + coarse (coarse operator is
                    // a fixed coarse solve, applied to the scaled residual it sees).
                    KrylovOperator pc_16c = [&](const std::vector<double>& r,
                                                std::vector<double>& z) {
                        pc_additive_set(blocks16c, r, z);
                    };
                    double c_r300s = 0, c_r600s = 0;         // scaled (secondary)
                    double c_relu300 = 0, c_relu600 = 0;     // UNSCALED true (table)
                    double c_wpi = 0;
                    bool c_ran = false;
                    ArmShare c_sh{0, 0, 0, 0};
                    // Unscaled TRUE rel of a scaled-system iterate: ||b - A x||/||b||.
                    // x solves A x = b iff A~ x = b~ (same solution), so the scaled
                    // arm's iterate is directly comparable in the UNSCALED norm.
                    auto unscaled_rel = [&](const std::vector<double>& x) -> double {
                        std::vector<double> ax;
                        coo_spmv(x, ax);
                        double s = 0.0;
                        for (int i = 0; i < n_f; ++i) {
                            const double d = b[static_cast<std::size_t>(i)] -
                                             ax[static_cast<std::size_t>(i)];
                            s += d * d;
                        }
                        return (b_norm > 0.0) ? std::sqrt(s) / b_norm : std::sqrt(s);
                    };
                    if (ok16c) {
                        std::vector<double> zt;
                        pc_16c(b_scaled, zt);
                        if (all_finite(zt)) {
                            const double saved_tol = gmres_config.tolerance;
                            const int saved_maxit = gmres_config.max_iters;
                            // Absolute kernel tolerance; the scaled residual scale is
                            // ||b~||, not ||b|| (else 1e-6*||b|| trips false early
                            // convergence / curve truncation).
                            gmres_config.tolerance = 1e-6 * b_scaled_norm;
                            // GMRES(x0=0) is deterministic, so the maxit=300 solve is
                            // the exact 300-iter prefix of the maxit=600 solve; two
                            // runs give the TRUE unscaled residual at 300 and 600 (the
                            // kernel exposes no intermediate iterate, and the scaled
                            // residual curve is ||D r||, not the comparable ||r||).
                            auto run_scaled =
                                [&](int maxit, std::vector<double>& curve,
                                    GmresStatus& st) -> std::vector<double> {
                                std::vector<double> x(size_n, 0.0);
                                gmres_config.max_iters = maxit;
                                gmres_config.residual_history = &curve;
                                st = right_preconditioned_gmres(
                                    b_scaled, x, spmv_scaled, pc_16c, gmres_config);
                                gmres_config.residual_history = nullptr;
                                return x;
                            };
                            std::cout << "--- pcoarse GMRES arm: 16c_row_equilibration "
                                         "(scaled system; 2 runs @300/@600 for exact "
                                         "unscaled rel) ---\n";
                            std::cout.flush();
                            std::vector<double> curve300, curve600;
                            GmresStatus st300, st600;
                            const std::vector<double> x300 =
                                run_scaled(300, curve300, st300);
                            c_relu300 = unscaled_rel(x300);
                            const auto tc0 = std::chrono::steady_clock::now();
                            const std::vector<double> x600 =
                                run_scaled(600, curve600, st600);
                            const double wall =
                                std::chrono::duration<double>(
                                    std::chrono::steady_clock::now() - tc0)
                                    .count();
                            gmres_config.tolerance = saved_tol;
                            gmres_config.max_iters = saved_maxit;
                            c_wpi = (st600.iterations > 0)
                                        ? wall / st600.iterations
                                        : 0.0;
                            c_relu600 = unscaled_rel(x600);
                            auto rel_scaled = [&](const std::vector<double>& cv,
                                                  int k) -> double {
                                if (cv.empty() || b_scaled_norm <= 0.0)
                                    return 0.0;
                                const int idx =
                                    std::min(k, static_cast<int>(cv.size())) - 1;
                                return cv[static_cast<std::size_t>(idx)] /
                                       b_scaled_norm;
                            };
                            c_r300s = rel_scaled(curve300, 300);
                            c_r600s = rel_scaled(curve600, 600);
                            c_sh = attrib_share(x600);
                            c_ran = true;
                            std::cout << "  arm 16c: iters=" << st600.iterations
                                      << " UNSCALED_true_rel@300=" << c_relu300
                                      << " UNSCALED_true_rel@600=" << c_relu600
                                      << " (scaled_rel@300=" << c_r300s
                                      << " scaled_rel@600=" << c_r600s << ") wall="
                                      << wall << "s\n";
                            std::cout.flush();
                        } else {
                            std::cout << "  16c SKIPPED (non-finite PC apply)\n";
                        }
                    } else {
                        std::cout << "  16c SKIPPED (scaled blockset QR failed)\n";
                    }
                    std::vector<SchwarzBlock>().swap(blocks16c);

                    // ============================================================
                    // RESULTS TABLE (unscaled arms) + rubric
                    // ============================================================
                    std::cout << "\n=== ARM 16 RESULTS (canonical current-ladder "
                                 "fixture) ===\n";
                    std::cout << "  arm | rel@300 | rel@600 | ord[0,300) | "
                                 "ord[300,600) | non_decel | e:global | e:seam | "
                                 "e:high-p | solve_rel | wall/iter(s)\n";
                    auto rel_at = [&](const ArmResult& a, int k) -> double {
                        if (a.curve.empty())
                            return 0.0;
                        const int idx =
                            std::min(k, static_cast<int>(a.curve.size())) - 1;
                        return (b_norm > 0.0)
                                   ? a.curve[static_cast<std::size_t>(idx)] / b_norm
                                   : a.curve[static_cast<std::size_t>(idx)];
                    };
                    struct ArmEval {
                        std::string name;
                        double r300;
                        double r600;
                        bool non_decel;
                        ArmShare sh;
                    };
                    std::vector<ArmEval> evals;
                    double base_r600 = 0.0;
                    for (std::size_t ai = a_first; ai < arms.size(); ++ai) {
                        const ArmResult& a = arms[ai];
                        const double r300 = rel_at(a, 300);
                        const double r600 = rel_at(a, 600);
                        const double ord_a =
                            (r300 > 0.0) ? std::log10(1.0 / r300) : 0.0;
                        const double ord_b =
                            (r300 > 0.0 && r600 > 0.0) ? std::log10(r300 / r600)
                                                       : 0.0;
                        const ArmShare sh = attrib_share(a.x);
                        const bool non_decel = (ord_b >= ord_a - 1e-12);
                        const double wpi = (a.iters > 0) ? a.wall / a.iters : 0.0;
                        if (a.name == "8r2_baseline")
                            base_r600 = r600;
                        std::cout << "  " << a.name << " | " << r300 << " | " << r600
                                  << " | " << ord_a << " | " << ord_b << " | "
                                  << (non_decel ? "YES" : "no") << " | " << sh.global
                                  << " | " << sh.seam << " | " << sh.hi << " | "
                                  << sh.solve_rel << " | " << wpi << '\n';
                        evals.push_back({a.name, r300, r600, non_decel, sh});
                    }
                    if (c_ran) {
                        // 16c table row uses the UNSCALED true rel (comparable);
                        // scaled rels shown in brackets for reference only.
                        const double ord_a =
                            (c_relu300 > 0.0) ? std::log10(1.0 / c_relu300) : 0.0;
                        const double ord_b =
                            (c_relu300 > 0.0 && c_relu600 > 0.0)
                                ? std::log10(c_relu300 / c_relu600)
                                : 0.0;
                        std::cout << "  16c_row_equilibration | " << c_relu300
                                  << " | " << c_relu600 << " | " << ord_a << " | "
                                  << ord_b << " | "
                                  << (ord_b >= ord_a - 1e-12 ? "YES" : "no") << " | "
                                  << c_sh.global << " | " << c_sh.seam << " | "
                                  << c_sh.hi << " | " << c_sh.solve_rel << " | "
                                  << c_wpi << "   [UNSCALED true rel; scaled_rel@300/"
                                     "600=" << c_r300s << "/" << c_r600s << "]\n";
                    }
                    std::cout.flush();

                    // --- Rubric ---
                    std::cout << "\n  === ARM 16 RUBRIC ===\n";
                    bool any_pass = false, any_partial = false;
                    auto judge = [&](const std::string& name, double r300,
                                     double r600, bool non_decel,
                                     const ArmShare& sh) {
                        std::string v;
                        // non_decel alone is not enough: a DIVERGENT arm (rel>>1)
                        // can plateau and read "non-decelerating" while being orders
                        // worse than baseline (16c row-equilibration: rel@600=94 in
                        // the true norm, non_decel=1). PARTIAL requires the arm to be
                        // at least not-worse than baseline.
                        if (r600 <= 1e-4 && non_decel) {
                            v = "PASS (rel@600<=1e-4, non-decel)";
                            any_pass = true;
                        } else if (base_r600 > 0.0 &&
                                   (r600 <= base_r600 / 3.0 ||
                                    (non_decel && r600 < base_r600))) {
                            v = "PARTIAL (>=3x better than baseline, or non-decel and "
                                "not worse)";
                            any_partial = true;
                        } else {
                            v = "INERT/WORSE";
                        }
                        std::cout << "  VERDICT " << name << ": " << v
                                  << "  (rel@300=" << r300 << " rel@600=" << r600
                                  << " non_decel=" << (non_decel ? 1 : 0)
                                  << " e:seam=" << sh.seam << ")\n";
                    };
                    for (const ArmEval& ev : evals)
                        if (ev.name != "8r2_baseline")
                            judge(ev.name, ev.r300, ev.r600, ev.non_decel, ev.sh);
                    if (c_ran) {
                        // 16c judged on the UNSCALED comparable rel at 300/600.
                        const bool nd = (c_relu600 > 0.0 && c_relu300 > 0.0)
                                            ? (std::log10(c_relu300 / c_relu600) >=
                                               std::log10(1.0 / c_relu300) - 1e-12)
                                            : false;
                        judge("16c_row_equilibration(unscaled)", c_relu300,
                              c_relu600, nd, c_sh);
                    }
                    std::cout << "  RUBRIC BRANCH: ";
                    if (any_pass)
                        std::cout << "PASS -- a smoother-side lever fixes the slope; "
                                     "recommend an end-to-end Newton gate next.";
                    else if (any_partial)
                        std::cout << "PARTIAL -- a lever is >=3x better or "
                                     "non-decelerating; interface-row coarse "
                                     "enrichment is the follow-up.";
                    else
                        std::cout << "INERT/WORSE -- every cheap smoother lever tested; "
                                     "PARK the p-coarse PC lane at the E-W grade "
                                     "(final).";
                    std::cout << "\n  (baseline 8r2 rel@600=" << base_r600
                              << "; diag blocks worsening seam=" << diag_worsen
                              << ")\n";
                    std::cout << "############ ARM 16 END ############\n";
                    std::cout.flush();
                    return;
                }

                // ====================================================================
                // ARM 18 PRE-GATES (PCOARSE_ARM18PRE). Two cheap falsifiers
                // BEFORE any physics-PC build, because arms 13-16 all aimed at
                // INFERRED targets and were refuted:
                //   pre-A -- Ritz spectrum + attribution of the PRECONDITIONED
                //            operator A*M (arm-8). Never measured. From the raw
                //            (pre-Givens) Hessenberg of the arm-8 @600 GMRES run,
                //            dgeev -> Ritz values of A*M; the ~40 smallest-|theta|
                //            Ritz vectors x=V*y are attributed by ROW taxonomy and
                //            u=M(x) by COLUMN taxonomy. Names the channel that
                //            actually decelerates arm 8.
                //   pre-B -- COO-surgery upper bound of the C1 frozen principal-part
                //            PC. M_surgery is built from the EXISTING in-run fine COO
                //            (frozen production entries >= analytic skeleton) by
                //            pinning {204 global + H + phi} columns, keeping
                //            TauMatch/TauBc rows entirely and Vol(P/N/bet) rows same-
                //            field only, and completing to a nonsingular square system
                //            with single-entry pin rows. MUMPS-factored and run as a
                //            PC on the TRUE system @600. Falsifies C1 in hours.
                // Self-contained; returns before the arm-14/Stage-A flow.
                // ====================================================================
                if (env_flag_enabled("PCOARSE_ARM18PRE", false)) {
                    std::cout << "\n############ ARM 18 PRE-GATES: A*M Ritz "
                                 "attribution + C1 COO-surgery upper bound "
                                 "############\n";
                    std::cout.flush();
                    const std::size_t size_n = static_cast<std::size_t>(n_f);

                    // --- shared masks (same predicates arms 14/15/16 use) ---
                    std::vector<char> is_global(size_n, 0);
                    for (int cc = 0; cc < n_f; ++cc) {
                        const ColumnInfo& ci = cmap[static_cast<std::size_t>(cc)];
                        if (ci.is_var_domain || ci.var_double_idx >= 0 ||
                            ci.domain < 0)
                            is_global[static_cast<std::size_t>(cc)] = 1;
                    }
                    std::vector<char> is_tm_row(size_n, 0);
                    for (int rr = 0; rr < n_f; ++rr)
                        if (rr < static_cast<int>(tmeta.size()) &&
                            tmeta[static_cast<std::size_t>(rr)].taxonomy ==
                                RowTaxonomy::TauMatch)
                            is_tm_row[static_cast<std::size_t>(rr)] = 1;
                    std::vector<char> col_seam(size_n, 0);
                    for (long long e = 0; e < nnz_f; ++e) {
                        const std::size_t ee = static_cast<std::size_t>(e);
                        if (is_tm_row[static_cast<std::size_t>(coo.irn[ee] - 1)])
                            col_seam[static_cast<std::size_t>(coo.jcn[ee] - 1)] = 1;
                    }
                    // per-column field index from ColumnInfo.var_idx (census:
                    // 0=H 1=P 2=N 3=bet 4=phi; -1 = global/scalar/other).
                    std::vector<int> col_field(size_n, -1);
                    for (int cc = 0; cc < n_f; ++cc)
                        col_field[static_cast<std::size_t>(cc)] =
                            cmap[static_cast<std::size_t>(cc)].var_idx;
                    auto coo_apply = [&](const std::vector<double>& v,
                                         std::vector<double>& out) {
                        out.assign(size_n, 0.0);
                        for (long long e = 0; e < nnz_f; ++e) {
                            const std::size_t ee = static_cast<std::size_t>(e);
                            out[static_cast<std::size_t>(coo.irn[ee] - 1)] +=
                                coo.a[ee] *
                                v[static_cast<std::size_t>(coo.jcn[ee] - 1)];
                        }
                    };

                    // ============================================================
                    // GATE 18-pre-A: Ritz spectrum + attribution of A*M (arm 8)
                    // ============================================================
                    std::cout << "\n==== 18-pre-A: A*M Ritz spectrum (arm-8 @"
                              << gmres_config.max_iters << ") ====\n";
                    std::cout.flush();
                    std::vector<std::vector<double>> Vbasis; // n x m Arnoldi basis
                    std::vector<std::vector<double>> Hraw;   // (m+1) x m raw Hessenberg
                    const std::size_t a8_idx = arms.size();
                    gmres_config.arnoldi_basis_out = &Vbasis;
                    gmres_config.hessenberg_raw_out = &Hraw;
                    run_arm("8r2_baseline", pc_arm8);
                    gmres_config.arnoldi_basis_out = nullptr;
                    gmres_config.hessenberg_raw_out = nullptr;
                    const int m = static_cast<int>(Vbasis.size());

                    // pre-A dominant-channel summary (filled below; consumed by the
                    // final rubric to decide the C1-DEAD branch).
                    std::string preA_channel = "n/a";
                    int n_below_1e2 = 0, n_below_1e3 = 0;
                    bool preA_ok = (m >= 2 && static_cast<int>(Hraw.size()) >= m + 1);
                    if (!preA_ok) {
                        std::cout << "  18-pre-A SKIP: captured m=" << m
                                  << " Hraw.rows=" << Hraw.size()
                                  << " (need m>=2); spectral analysis unavailable\n";
                    } else {
                        // dense col-major leading m x m raw Hessenberg for dgeev.
                        std::vector<double> A_col(static_cast<std::size_t>(m) *
                                                      static_cast<std::size_t>(m),
                                                  0.0);
                        for (int j = 0; j < m; ++j)
                            for (int i = 0; i < m; ++i)
                                A_col[static_cast<std::size_t>(i) +
                                      static_cast<std::size_t>(j) *
                                          static_cast<std::size_t>(m)] =
                                    Hraw[static_cast<std::size_t>(i)]
                                        [static_cast<std::size_t>(j)];
                        std::vector<double> wr(static_cast<std::size_t>(m), 0.0),
                            wi(static_cast<std::size_t>(m), 0.0),
                            vr(static_cast<std::size_t>(m) *
                                   static_cast<std::size_t>(m),
                               0.0),
                            vl_dummy(1, 0.0);
                        char jn = 'N', jv = 'V';
                        int mm = m, ldvl = 1, info = 0, lwork = -1;
                        double wq = 0.0;
                        dgeev_(&jn, &jv, &mm, A_col.data(), &mm, wr.data(),
                               wi.data(), vl_dummy.data(), &ldvl, vr.data(), &mm,
                               &wq, &lwork, &info);
                        lwork = (info == 0) ? static_cast<int>(wq)
                                            : std::max(1, 4 * m);
                        std::vector<double> work(
                            static_cast<std::size_t>(std::max(1, lwork)));
                        dgeev_(&jn, &jv, &mm, A_col.data(), &mm, wr.data(),
                               wi.data(), vl_dummy.data(), &ldvl, vr.data(), &mm,
                               work.data(), &lwork, &info);
                        if (info != 0) {
                            std::cout << "  18-pre-A dgeev FAIL info=" << info
                                      << " (spectral analysis skipped)\n";
                            preA_ok = false;
                        } else {
                            struct EigBlk {
                                double mag;
                                int j;
                                bool cplx;
                            };
                            std::vector<EigBlk> blks;
                            for (int j = 0; j < m;) {
                                if (wi[static_cast<std::size_t>(j)] == 0.0) {
                                    blks.push_back(
                                        {std::fabs(wr[static_cast<std::size_t>(j)]),
                                         j, false});
                                    j += 1;
                                } else {
                                    blks.push_back(
                                        {std::hypot(
                                             wr[static_cast<std::size_t>(j)],
                                             wi[static_cast<std::size_t>(j)]),
                                         j, true});
                                    j += 2;
                                }
                            }
                            std::sort(blks.begin(), blks.end(),
                                      [](const EigBlk& a, const EigBlk& c) {
                                          return a.mag < c.mag;
                                      });
                            for (const EigBlk& bl : blks) {
                                if (bl.mag < 1e-2)
                                    ++n_below_1e2;
                                if (bl.mag < 1e-3)
                                    ++n_below_1e3;
                            }
                            // smallest-60 table + histogram of the rest.
                            std::cout << "  Ritz values of A*M (eigs of leading "
                                      << m << "x" << m
                                      << " raw Hessenberg), smallest-60 |theta|:\n";
                            const int n_show =
                                std::min<int>(60, static_cast<int>(blks.size()));
                            for (int t = 0; t < n_show; ++t)
                                std::cout << "    [" << (t + 1) << "] |theta|="
                                          << blks[static_cast<std::size_t>(t)].mag
                                          << (blks[static_cast<std::size_t>(t)].cplx
                                                  ? " (cplx)"
                                                  : "")
                                          << '\n';
                            // decade histogram over ALL Ritz magnitudes.
                            int hist[7] = {0, 0, 0, 0, 0, 0, 0};
                            for (const EigBlk& bl : blks) {
                                if (bl.mag < 1e-3)
                                    ++hist[0];
                                else if (bl.mag < 1e-2)
                                    ++hist[1];
                                else if (bl.mag < 1e-1)
                                    ++hist[2];
                                else if (bl.mag < 1e0)
                                    ++hist[3];
                                else if (bl.mag < 1e1)
                                    ++hist[4];
                                else if (bl.mag < 1e2)
                                    ++hist[5];
                                else
                                    ++hist[6];
                            }
                            std::cout << "  |theta| histogram: <1e-3:" << hist[0]
                                      << " [1e-3,1e-2):" << hist[1]
                                      << " [1e-2,1e-1):" << hist[2]
                                      << " [1e-1,1):" << hist[3]
                                      << " [1,1e1):" << hist[4]
                                      << " [1e1,1e2):" << hist[5]
                                      << " >=1e2:" << hist[6] << '\n';
                            std::cout << "  cluster: |theta|<1e-2 = " << n_below_1e2
                                      << ", |theta|<1e-3 = " << n_below_1e3
                                      << "  (compare to the ~75-iters/order slope)\n";
                            std::cout.flush();

                            // --- attribute the NA smallest-|theta| Ritz vectors ---
                            const int NA = std::min<int>(
                                env_int_value("PCOARSE_ARM18PRE_NMODES", 40),
                                static_cast<int>(blks.size()));
                            std::cout << "\n  slow-mode attribution (smallest-" << NA
                                      << " |theta|): x=V*y row-space, u=M(x) col-"
                                         "space\n";
                            // running means for the aggregate classification.
                            double sVol = 0, sTauBc = 0, sTauMatch = 0, sGlob = 0,
                                   sUnm = 0;       // ROW shares
                            double cGl = 0, cSm = 0, cHp = 0, cH = 0, cPhi = 0,
                                   cP = 0, cN = 0, cBet = 0; // COL shares
                            double sVol10 = 0, sGlob10 = 0, sTM10 = 0; // smallest-10
                            for (int bi = 0; bi < NA; ++bi) {
                                const int jcol = blks[static_cast<std::size_t>(bi)].j;
                                // real Ritz vector x = V * Re(y): y = vr column jcol.
                                std::vector<double> x(size_n, 0.0);
                                for (int k = 0; k < m; ++k) {
                                    const double yk =
                                        vr[static_cast<std::size_t>(k) +
                                           static_cast<std::size_t>(jcol) *
                                               static_cast<std::size_t>(m)];
                                    if (yk == 0.0)
                                        continue;
                                    const std::vector<double>& vk =
                                        Vbasis[static_cast<std::size_t>(k)];
                                    for (int r = 0; r < n_f; ++r)
                                        x[static_cast<std::size_t>(r)] +=
                                            yk * vk[static_cast<std::size_t>(r)];
                                }
                                // ROW taxonomy energy shares of x.
                                double tot = 0, eVol = 0, eTauBc = 0, eTM = 0,
                                       eGI = 0, eUnm = 0;
                                for (int r = 0; r < n_f; ++r) {
                                    const double e2 =
                                        x[static_cast<std::size_t>(r)] *
                                        x[static_cast<std::size_t>(r)];
                                    tot += e2;
                                    RowTaxonomy tx =
                                        (r < static_cast<int>(tmeta.size()))
                                            ? tmeta[static_cast<std::size_t>(r)]
                                                  .taxonomy
                                            : RowTaxonomy::Unknown;
                                    if (tx == RowTaxonomy::Vol)
                                        eVol += e2;
                                    else if (tx == RowTaxonomy::TauBc)
                                        eTauBc += e2;
                                    else if (tx == RowTaxonomy::TauMatch)
                                        eTM += e2;
                                    else if (tx == RowTaxonomy::GlobalInt)
                                        eGI += e2;
                                    if (!fine_row_matched[static_cast<std::size_t>(
                                            r)])
                                        eUnm += e2;
                                }
                                const double inv = (tot > 0.0) ? 1.0 / tot : 0.0;
                                // COLUMN taxonomy energy shares of u = M(x).
                                std::vector<double> u;
                                pc_arm8(x, u);
                                double utot = 0, uGl = 0, uSm = 0, uHp = 0, uH = 0,
                                       uPhi = 0, uP = 0, uN = 0, uBet = 0;
                                for (int c = 0; c < n_f; ++c) {
                                    const double e2 =
                                        u[static_cast<std::size_t>(c)] *
                                        u[static_cast<std::size_t>(c)];
                                    utot += e2;
                                    if (is_global[static_cast<std::size_t>(c)])
                                        uGl += e2;
                                    if (col_seam[static_cast<std::size_t>(c)])
                                        uSm += e2;
                                    if (!fine_col_matched[static_cast<std::size_t>(
                                            c)])
                                        uHp += e2;
                                    switch (col_field[static_cast<std::size_t>(c)]) {
                                        case 0: uH += e2; break;
                                        case 1: uP += e2; break;
                                        case 2: uN += e2; break;
                                        case 3: uBet += e2; break;
                                        case 4: uPhi += e2; break;
                                        default: break;
                                    }
                                }
                                const double uinv =
                                    (utot > 0.0) ? 1.0 / utot : 0.0;
                                sVol += eVol * inv;
                                sTauBc += eTauBc * inv;
                                sTauMatch += eTM * inv;
                                sGlob += eGI * inv;
                                sUnm += eUnm * inv;
                                cGl += uGl * uinv;
                                cSm += uSm * uinv;
                                cHp += uHp * uinv;
                                cH += uH * uinv;
                                cPhi += uPhi * uinv;
                                cP += uP * uinv;
                                cN += uN * uinv;
                                cBet += uBet * uinv;
                                if (bi < 10) {
                                    sVol10 += eVol * inv;
                                    sGlob10 += eGI * inv;
                                    sTM10 += eTM * inv;
                                }
                                if (bi < std::min(NA, 20))
                                    std::cout
                                        << "    |theta|="
                                        << blks[static_cast<std::size_t>(bi)].mag
                                        << " | row: Vol=" << eVol * inv
                                        << " TauMatch=" << eTM * inv
                                        << " GlobalInt=" << eGI * inv
                                        << " unmatched=" << eUnm * inv
                                        << " | col: global=" << uGl * uinv
                                        << " seam=" << uSm * uinv
                                        << " H=" << uH * uinv
                                        << " phi=" << uPhi * uinv
                                        << " bet=" << uBet * uinv << '\n';
                            }
                            const double an = (NA > 0) ? 1.0 / NA : 0.0;
                            const double an10 =
                                (std::min(NA, 10) > 0) ? 1.0 / std::min(NA, 10)
                                                       : 0.0;
                            std::cout << "\n  AGGREGATE over smallest-" << NA
                                      << " modes:\n"
                                      << "    ROW: Vol=" << sVol * an
                                      << " TauBc=" << sTauBc * an
                                      << " TauMatch=" << sTauMatch * an
                                      << " GlobalInt=" << sGlob * an
                                      << " unmatched-high-p=" << sUnm * an << '\n'
                                      << "    COL(u=Mx): global=" << cGl * an
                                      << " seam=" << cSm * an << " high-p=" << cHp * an
                                      << " | field H=" << cH * an
                                      << " P=" << cP * an << " N=" << cN * an
                                      << " bet=" << cBet * an << " phi=" << cPhi * an
                                      << '\n'
                                      << "    ROW smallest-10: Vol=" << sVol10 * an10
                                      << " TauMatch=" << sTM10 * an10
                                      << " GlobalInt=" << sGlob10 * an10 << '\n';
                            // dominant-channel verdict for the rubric.
                            const double mVol = sVol * an, mTM = sTauMatch * an,
                                         mGI = sGlob * an, mUnm = sUnm * an;
                            const double mcGl = cGl * an, mcH = cH * an,
                                         mcPhi = cPhi * an, mcSm = cSm * an;
                            if (mcH + mcPhi > 0.5 || (mGI > 0.4 && mcH > 0.25))
                                preA_channel = "H/first-integral+hydro sector "
                                               "(H/phi cols, GlobalInt rows)";
                            else if (mcGl > 0.5 || mGI > 0.5)
                                preA_channel = "global-scalar channel";
                            else if (mTM > 0.4 || mcSm > 0.5)
                                preA_channel = "seam/tau (TauMatch rows)";
                            else if (mVol > 0.5 && mUnm > 0.4)
                                preA_channel =
                                    "star-surface/bulk high-p (Vol unmatched)";
                            else
                                preA_channel = "mixed/no single dominant channel";
                            std::cout << "  pre-A SLOW-MODE CHANNEL: " << preA_channel
                                      << '\n';
                            std::cout.flush();
                        }
                    }
                    // free the (large) Krylov capture before the pre-B factors.
                    std::vector<std::vector<double>>().swap(Vbasis);
                    std::vector<std::vector<double>>().swap(Hraw);

                    // ============================================================
                    // GATE 18-pre-B: C1 COO-surgery upper bound
                    // ============================================================
                    std::cout << "\n==== 18-pre-B: C1 frozen principal-part surgery "
                                 "upper bound ====\n";
                    std::cout.flush();

                    // (1) pin column set = 204 global + H cols + phi cols.
                    std::vector<char> is_pin_col(size_n, 0);
                    int nc_global = 0, nc_H = 0, nc_phi = 0;
                    for (int c = 0; c < n_f; ++c) {
                        const bool g = is_global[static_cast<std::size_t>(c)] != 0;
                        const bool h = (col_field[static_cast<std::size_t>(c)] == 0);
                        const bool ph = (col_field[static_cast<std::size_t>(c)] == 4);
                        if (g || h || ph)
                            is_pin_col[static_cast<std::size_t>(c)] = 1;
                        if (g)
                            ++nc_global; // disjoint census (global takes precedence)
                        else if (h)
                            ++nc_H;
                        else if (ph)
                            ++nc_phi;
                    }
                    int n_pin = 0;
                    for (int c = 0; c < n_f; ++c)
                        n_pin += is_pin_col[static_cast<std::size_t>(c)];
                    std::cout << "  pin cols: global(col_dom==-1)=" << nc_global
                              << " H=" << nc_H << " phi=" << nc_phi
                              << " -> |pin|=" << n_pin << '\n';

                    // (2) per-row field (owner_var_name, trimmed) + taxonomy.
                    auto trim = [](const std::string& s) {
                        const std::size_t a = s.find_first_not_of(" \t\r\n");
                        if (a == std::string::npos)
                            return std::string();
                        const std::size_t b = s.find_last_not_of(" \t\r\n");
                        return s.substr(a, b - a + 1);
                    };
                    auto field_of = [](const std::string& s) -> int {
                        if (s == "H")
                            return 0;
                        if (s == "P")
                            return 1;
                        if (s == "N")
                            return 2;
                        if (s == "bet")
                            return 3;
                        if (s == "phi")
                            return 4;
                        return -1;
                    };
                    std::vector<int> row_field(size_n, -1);
                    for (int r = 0; r < n_f; ++r)
                        if (r < static_cast<int>(tmeta.size()))
                            row_field[static_cast<std::size_t>(r)] = field_of(
                                trim(tmeta[static_cast<std::size_t>(r)]
                                         .owner_var_name));
                    // keep_type: 1=interface(TauMatch/TauBc), 2=field(Vol P/N/bet),
                    // 0=forced drop (GlobalInt / Vol H,phi,unknown / Unknown tax).
                    std::vector<char> keep_type(size_n, 0);
                    // census: rows per (taxonomy x field).
                    long long cens[5][6];
                    for (auto& a : cens)
                        for (auto& v : a)
                            v = 0;
                    auto tax_idx = [](RowTaxonomy t) -> int {
                        switch (t) {
                            case RowTaxonomy::Vol: return 0;
                            case RowTaxonomy::TauBc: return 1;
                            case RowTaxonomy::TauMatch: return 2;
                            case RowTaxonomy::GlobalInt: return 3;
                            default: return 4; // Unknown
                        }
                    };
                    for (int r = 0; r < n_f; ++r) {
                        RowTaxonomy tx =
                            (r < static_cast<int>(tmeta.size()))
                                ? tmeta[static_cast<std::size_t>(r)].taxonomy
                                : RowTaxonomy::Unknown;
                        const int fi = row_field[static_cast<std::size_t>(r)];
                        cens[tax_idx(tx)][static_cast<std::size_t>(fi + 1)]++;
                        if (tx == RowTaxonomy::TauMatch || tx == RowTaxonomy::TauBc)
                            keep_type[static_cast<std::size_t>(r)] = 1;
                        else if (tx == RowTaxonomy::Vol &&
                                 (fi == 1 || fi == 2 || fi == 3))
                            keep_type[static_cast<std::size_t>(r)] = 2;
                    }
                    std::cout << "  row census [tax x field] (field col order: "
                                 "none,H,P,N,bet,phi):\n";
                    const char* txn[5] = {"Vol     ", "TauBc   ", "TauMatch",
                                          "GlobalIn", "Unknown "};
                    for (int t = 0; t < 5; ++t) {
                        std::cout << "    " << txn[t] << ":";
                        for (int f = 0; f < 6; ++f)
                            std::cout << ' ' << cens[t][f];
                        std::cout << '\n';
                    }

                    // (3) surviving-entry count per row under the keep rule.
                    std::vector<long long> surviving(size_n, 0);
                    for (long long e = 0; e < nnz_f; ++e) {
                        const std::size_t ee = static_cast<std::size_t>(e);
                        const int r = coo.irn[ee] - 1;
                        const int c = coo.jcn[ee] - 1;
                        if (is_pin_col[static_cast<std::size_t>(c)])
                            continue;
                        const char kt = keep_type[static_cast<std::size_t>(r)];
                        if (kt == 1)
                            surviving[static_cast<std::size_t>(r)]++;
                        else if (kt == 2 &&
                                 col_field[static_cast<std::size_t>(c)] ==
                                     row_field[static_cast<std::size_t>(r)])
                            surviving[static_cast<std::size_t>(r)]++;
                    }
                    // A row is KEPT iff keep-type and has >=1 surviving entry.
                    std::vector<char> is_drop(size_n, 0);
                    long long n_forced_drop = 0, n_empty_keep = 0;
                    for (int r = 0; r < n_f; ++r) {
                        const char kt = keep_type[static_cast<std::size_t>(r)];
                        if (kt == 0) {
                            is_drop[static_cast<std::size_t>(r)] = 1;
                            ++n_forced_drop;
                        } else if (surviving[static_cast<std::size_t>(r)] == 0) {
                            is_drop[static_cast<std::size_t>(r)] = 1;
                            ++n_forced_drop;
                            ++n_empty_keep;
                        }
                    }
                    long long n_drop = n_forced_drop;
                    std::cout << "  forced drops (non-keep + empty-after-filter)="
                              << n_forced_drop << " (of which keep-type-but-empty="
                              << n_empty_keep << ")  |pin|=" << n_pin << '\n';

                    // (4) balance |drop| == |pin| by moving extra kept rows to drop.
                    bool singular_structure = false;
                    if (n_drop < n_pin) {
                        const long long extra = n_pin - n_drop;
                        // move `extra` kept rows with the fewest surviving entries
                        // (deterministic tie-break by row index).
                        std::vector<std::pair<long long, int>> cand;
                        for (int r = 0; r < n_f; ++r)
                            if (!is_drop[static_cast<std::size_t>(r)])
                                cand.emplace_back(
                                    surviving[static_cast<std::size_t>(r)], r);
                        std::sort(cand.begin(), cand.end(),
                                  [](const std::pair<long long, int>& a,
                                     const std::pair<long long, int>& b) {
                                      if (a.first != b.first)
                                          return a.first < b.first;
                                      return a.second < b.second;
                                  });
                        for (long long t = 0;
                             t < extra && t < static_cast<long long>(cand.size());
                             ++t) {
                            is_drop[static_cast<std::size_t>(
                                cand[static_cast<std::size_t>(t)].second)] = 1;
                            ++n_drop;
                        }
                        std::cout << "  balanced: moved " << extra
                                  << " lowest-value kept rows to drop -> |drop|="
                                  << n_drop << '\n';
                    } else if (n_drop > n_pin) {
                        singular_structure = true;
                        std::cout << "  STRUCTURAL VERDICT: forced drops (" << n_drop
                                  << ") > pin cols (" << n_pin
                                  << "); the frozen {P,N,bet} same-field skeleton is "
                                     "UNDERDETERMINED by "
                                  << (n_drop - n_pin)
                                  << " rows -- no square nonsingular pin completion "
                                     "exists.\n";
                    } else {
                        std::cout << "  balanced exactly: |drop|=|pin|=" << n_pin
                                  << '\n';
                    }

                    // final residual-attribution rel@ helper (curve is absolute).
                    auto rel_at = [&](const ArmResult& a, int k) -> double {
                        if (a.curve.empty())
                            return 0.0;
                        const int idx =
                            std::min(k, static_cast<int>(a.curve.size())) - 1;
                        return (b_norm > 0.0)
                                   ? a.curve[static_cast<std::size_t>(idx)] / b_norm
                                   : a.curve[static_cast<std::size_t>(idx)];
                    };

                    double preB_r300 = 0.0, preB_r600 = 0.0;
                    bool preB_non_decel = false, preB_ran = false;
                    bool near_singular = false;
                    double eH_share = 0.0, ePhi_share = 0.0, eGl_share = 0.0,
                           eSm_share = 0.0, eHp_share = 0.0;

                    if (!singular_structure) {
                        // (5) build M_surgery COO. Bijection: sorted drop rows ->
                        // sorted pin cols; kept rows -> surviving entries.
                        std::vector<int> pin_cols;
                        pin_cols.reserve(static_cast<std::size_t>(n_pin));
                        for (int c = 0; c < n_f; ++c)
                            if (is_pin_col[static_cast<std::size_t>(c)])
                                pin_cols.push_back(c);
                        std::vector<int> drop_rows;
                        drop_rows.reserve(static_cast<std::size_t>(n_drop));
                        for (int r = 0; r < n_f; ++r)
                            if (is_drop[static_cast<std::size_t>(r)])
                                drop_rows.push_back(r);
                        // (both ascending already; sizes equal by construction)
                        std::vector<int> pin_of_row(size_n, -1);
                        for (std::size_t t = 0; t < drop_rows.size() &&
                                                t < pin_cols.size();
                             ++t)
                            pin_of_row[static_cast<std::size_t>(
                                drop_rows[t])] = pin_cols[t];

                        std::vector<int> irn2, jcn2;
                        std::vector<double> a2;
                        irn2.reserve(static_cast<std::size_t>(nnz_f / 2));
                        jcn2.reserve(static_cast<std::size_t>(nnz_f / 2));
                        a2.reserve(static_cast<std::size_t>(nnz_f / 2));
                        // per-field-block nnz census on kept Vol part.
                        long long blk_nnz[4] = {0, 0, 0, 0}; // P,N,bet,interface
                        long long xfield_vol = 0;            // cross-field in Vol keep
                        for (long long e = 0; e < nnz_f; ++e) {
                            const std::size_t ee = static_cast<std::size_t>(e);
                            const int r = coo.irn[ee] - 1;
                            const int c = coo.jcn[ee] - 1;
                            if (is_drop[static_cast<std::size_t>(r)])
                                continue; // drop rows become pin rows below
                            if (is_pin_col[static_cast<std::size_t>(c)])
                                continue;
                            const char kt = keep_type[static_cast<std::size_t>(r)];
                            bool keep = false;
                            if (kt == 1) {
                                keep = true; // interface: all non-pinned
                                blk_nnz[3]++;
                            } else if (kt == 2) {
                                const int rf =
                                    row_field[static_cast<std::size_t>(r)];
                                const int cf =
                                    col_field[static_cast<std::size_t>(c)];
                                if (cf == rf) {
                                    keep = true;
                                    if (rf == 1)
                                        blk_nnz[0]++;
                                    else if (rf == 2)
                                        blk_nnz[1]++;
                                    else if (rf == 3)
                                        blk_nnz[2]++;
                                } else {
                                    ++xfield_vol; // never kept; must be 0
                                }
                            }
                            if (keep) {
                                irn2.push_back(r + 1);
                                jcn2.push_back(c + 1);
                                a2.push_back(coo.a[ee]);
                            }
                        }
                        // pin rows: single 1.0 at the row's assigned pin col.
                        for (int r : drop_rows) {
                            const int pc = pin_of_row[static_cast<std::size_t>(r)];
                            if (pc < 0)
                                continue;
                            irn2.push_back(r + 1);
                            jcn2.push_back(pc + 1);
                            a2.push_back(1.0);
                        }
                        const long long nnz2 =
                            static_cast<long long>(irn2.size());

                        // (5b) verify square, no empty rows/cols.
                        std::vector<char> row_hit(size_n, 0), col_hit(size_n, 0);
                        for (long long e = 0; e < nnz2; ++e) {
                            row_hit[static_cast<std::size_t>(
                                irn2[static_cast<std::size_t>(e)] - 1)] = 1;
                            col_hit[static_cast<std::size_t>(
                                jcn2[static_cast<std::size_t>(e)] - 1)] = 1;
                        }
                        long long empty_rows = 0, empty_cols = 0;
                        for (int i = 0; i < n_f; ++i) {
                            empty_rows += (row_hit[static_cast<std::size_t>(i)] == 0);
                            empty_cols += (col_hit[static_cast<std::size_t>(i)] == 0);
                        }
                        std::cout << "  M_surgery: n=" << n_f << " nnz=" << nnz2
                                  << " empty_rows=" << empty_rows
                                  << " empty_cols=" << empty_cols << '\n';
                        std::cout << "  kept-Vol per-field nnz: P=" << blk_nnz[0]
                                  << " N=" << blk_nnz[1] << " bet=" << blk_nnz[2]
                                  << " | interface(TauMatch/TauBc) nnz=" << blk_nnz[3]
                                  << " | kept-Vol cross-field nnz=0 by construction; "
                                     "cross-field Vol entries DROPPED=" << xfield_vol
                                  << " (C6 go/no-go: coupling discarded by the "
                                     "same-field freeze)\n";
                        std::cout.flush();

                        if (empty_rows > 0 || empty_cols > 0)
                            std::cout << "  WARNING: empty rows/cols present -> "
                                         "MUMPS factor expected singular\n";

                        // (6) MUMPS factor with null-pivot detection (structural
                        // singularity -> verdict rather than SIGKILL/garbage).
                        const auto tf0 = std::chrono::steady_clock::now();
                        bool factor_ok = true;
                        int npiv = 0;
                        std::unique_ptr<MumpsLinearSolver> surg;
                        try {
                            surg.reset(new MumpsLinearSolver(
                                n_f, config.mumps.ordering, false, 0,
                                mumps_runtime_state.icntl14, MPI_COMM_SELF));
                            surg->enable_null_pivot_detection(true, 0.0);
                            surg->set_pattern(n_f, nnz2, irn2.data(), jcn2.data());
                            surg->analyze_pattern();
                            surg->factor_analyzed(a2.data());
                            mumps_runtime_state.icntl14 = surg->last_icntl14();
                            npiv = surg->last_null_pivot_count();
                        } catch (const std::exception& ex) {
                            factor_ok = false;
                            std::cout << "  MUMPS factor THREW: " << ex.what()
                                      << " -> structural verdict (surgery singular)\n";
                        }
                        std::cout << "  surgery factor wall="
                                  << std::chrono::duration<double>(
                                         std::chrono::steady_clock::now() - tf0)
                                         .count()
                                  << "s null_pivots=" << npiv << '\n';
                        std::cout.flush();

                        // A GROSS null count (hundreds+) means the frozen
                        // {P,N,bet} skeleton is genuinely underdetermined -> hard
                        // structural verdict, arm not run. A near-miss (a handful of
                        // gauge/constant nulls in bet/phi) is projected out by MUMPS
                        // null-pivot detection, leaving a usable PC: run the arm to
                        // get the actual convergence upper bound, and record the
                        // near-singularity as a structural NOTE.
                        const long long kGrossNull = std::max<long long>(64, n_f / 100);
                        if (factor_ok && npiv > 0) {
                            long long npc_field[6] = {0, 0, 0, 0, 0, 0};
                            for (int v1 :
                                 surg->last_null_pivot_list_1based()) {
                                const int c = v1 - 1;
                                if (c >= 0 && c < n_f)
                                    npc_field[static_cast<std::size_t>(
                                        col_field[static_cast<std::size_t>(c)] +
                                        1)]++;
                            }
                            const bool gross = (npiv >= kGrossNull);
                            std::cout << (gross ? "  STRUCTURAL VERDICT: M_surgery "
                                                  "grossly singular ("
                                                : "  STRUCTURAL NOTE: M_surgery "
                                                  "near-singular (")
                                      << npiv
                                      << " null pivots; MUMPS-projected). null-pivot "
                                         "col field counts (none,H,P,N,bet,phi):";
                            for (int f = 0; f < 6; ++f)
                                std::cout << ' ' << npc_field[f];
                            std::cout << '\n';
                            std::cout.flush();
                            if (gross)
                                singular_structure = true;
                            else
                                near_singular = true;
                        }
                        if (factor_ok && !singular_structure) {
                            // (6b) run the surgery PC on the TRUE system @maxit. The
                            // finite guard neutralizes any non-finite output from the
                            // null-projected solve (GMRES has no non-finite guard).
                            KrylovOperator pc_surg =
                                [&](const std::vector<double>& r,
                                    std::vector<double>& z) {
                                    z = r;
                                    surg->solve(z.data());
                                    for (double& v : z)
                                        if (!std::isfinite(v))
                                            v = 0.0;
                                };
                            run_arm("18preB_surgery", pc_surg);
                            preB_ran = true;
                            const ArmResult& ab = arms.back();
                            preB_r300 = rel_at(ab, 300);
                            preB_r600 = rel_at(ab, 600);
                            const double ord_a =
                                (preB_r300 > 0.0) ? std::log10(1.0 / preB_r300)
                                                  : 0.0;
                            const double ord_b =
                                (preB_r300 > 0.0 && preB_r600 > 0.0)
                                    ? std::log10(preB_r300 / preB_r600)
                                    : 0.0;
                            preB_non_decel = (ord_b >= ord_a - 1e-12);
                            std::cout << "  18preB rel@300=" << preB_r300
                                      << " rel@600=" << preB_r600
                                      << " ord[0,300)=" << ord_a
                                      << " ord[300,600)=" << ord_b
                                      << " non_decel=" << (preB_non_decel ? 1 : 0)
                                      << '\n';
                            std::cout.flush();
                            // free the surgery factor before the TRUE-A attrib factor.
                            surg.reset();

                            // (7) attribution of the 18preB (and 8r2) residual via a
                            // refined solve of the TRUE fine system A.
                            const auto ta0 = std::chrono::steady_clock::now();
                            MumpsLinearSolver fine_solver(
                                n_f, config.mumps.ordering, false, 0,
                                mumps_runtime_state.icntl14, MPI_COMM_SELF);
                            fine_solver.set_pattern(n_f, nnz_f, coo.irn.data(),
                                                    coo.jcn.data());
                            fine_solver.analyze_pattern();
                            fine_solver.factor_analyzed(coo.a.data());
                            mumps_runtime_state.icntl14 =
                                fine_solver.last_icntl14();
                            std::cout << "  (attrib) TRUE-A factor wall="
                                      << std::chrono::duration<double>(
                                             std::chrono::steady_clock::now() - ta0)
                                             .count()
                                      << "s\n";
                            std::cout.flush();
                            auto refined_solve =
                                [&](const std::vector<double>& r)
                                -> std::vector<double> {
                                std::vector<double> e(r);
                                fine_solver.solve(e.data());
                                std::vector<double> ae;
                                coo_apply(e, ae);
                                std::vector<double> corr(size_n);
                                for (int i = 0; i < n_f; ++i)
                                    corr[static_cast<std::size_t>(i)] =
                                        r[static_cast<std::size_t>(i)] -
                                        ae[static_cast<std::size_t>(i)];
                                fine_solver.solve(corr.data());
                                for (int i = 0; i < n_f; ++i)
                                    e[static_cast<std::size_t>(i)] +=
                                        corr[static_cast<std::size_t>(i)];
                                return e;
                            };
                            auto attrib = [&](const std::vector<double>& xarm,
                                              const char* tag, double* out5) {
                                std::vector<double> ax;
                                coo_apply(xarm, ax);
                                std::vector<double> r(size_n);
                                for (int i = 0; i < n_f; ++i)
                                    r[static_cast<std::size_t>(i)] =
                                        b[static_cast<std::size_t>(i)] -
                                        ax[static_cast<std::size_t>(i)];
                                std::vector<double> e = refined_solve(r);
                                double tot = 0, gl = 0, sm = 0, hp = 0, eh = 0,
                                       eph = 0;
                                for (int i = 0; i < n_f; ++i) {
                                    const double e2 =
                                        e[static_cast<std::size_t>(i)] *
                                        e[static_cast<std::size_t>(i)];
                                    tot += e2;
                                    if (is_global[static_cast<std::size_t>(i)])
                                        gl += e2;
                                    if (col_seam[static_cast<std::size_t>(i)])
                                        sm += e2;
                                    if (!fine_col_matched[static_cast<std::size_t>(
                                            i)])
                                        hp += e2;
                                    if (col_field[static_cast<std::size_t>(i)] == 0)
                                        eh += e2;
                                    if (col_field[static_cast<std::size_t>(i)] == 4)
                                        eph += e2;
                                }
                                const double iv = (tot > 0.0) ? 1.0 / tot : 0.0;
                                std::cout << "  attrib[" << tag
                                          << "] e:global=" << gl * iv
                                          << " e:seam=" << sm * iv
                                          << " e:high-p=" << hp * iv
                                          << " e:H-col=" << eh * iv
                                          << " e:phi-col=" << eph * iv << '\n';
                                if (out5 != nullptr) {
                                    out5[0] = gl * iv;
                                    out5[1] = sm * iv;
                                    out5[2] = hp * iv;
                                    out5[3] = eh * iv;
                                    out5[4] = eph * iv;
                                }
                            };
                            if (a8_idx < arms.size())
                                attrib(arms[a8_idx].x, "8r2_baseline", nullptr);
                            double sh5[5] = {0, 0, 0, 0, 0};
                            attrib(arms.back().x, "18preB_surgery", sh5);
                            eGl_share = sh5[0];
                            eSm_share = sh5[1];
                            eHp_share = sh5[2];
                            eH_share = sh5[3];
                            ePhi_share = sh5[4];
                            std::cout.flush();
                        }
                    }

                    // ============================================================
                    // RUBRIC
                    // ============================================================
                    std::cout << "\n=== ARM 18 PRE-GATE RUBRIC ===\n";
                    const double base_r600 =
                        (a8_idx < arms.size()) ? rel_at(arms[a8_idx], 600) : 0.0;
                    std::cout << "  baseline 8r2 rel@600=" << base_r600
                              << "  (target: pre-B well below 1.21e-2, non-decel)\n";
                    std::string branch;
                    if (singular_structure) {
                        branch =
                            "STRUCTURAL VERDICT -- M_surgery singular; the frozen "
                            "same-field {P,N,bet} skeleton with H/phi/global pinned "
                            "does not admit a square nonsingular completion. C1 as "
                            "specified cannot be built by this pin/drop bookkeeping.";
                    } else if (preB_ran && preB_r600 > 0.0 && preB_r600 < 1e-3 &&
                               preB_non_decel) {
                        branch =
                            "C1 FUNDED -- surgery upper bound converges geometrically "
                            "well below baseline and does not decelerate. Building "
                            "the analytic frozen principal-part skeleton is justified; "
                            "note that productionizing the COO-surgery itself (rebuild "
                            "per rung + 5-block factor) may beat the analytic build.";
                    } else if (preB_ran &&
                               (preB_r600 >= base_r600 || !preB_non_decel)) {
                        branch =
                            std::string(
                                "C1-AS-DESIGNED DEAD -- surgery plateaus/decelerates; "
                                "pre-A slow-mode channel = ") +
                            preA_channel +
                            ". A viable design must treat that channel (not the "
                            "frozen same-field blocks). PARK C1 as specified.";
                    } else if (preB_ran) {
                        branch =
                            std::string(
                                "PARTIAL -- surgery beats baseline but not to 1e-3 / "
                                "shows some deceleration; pre-A channel = ") +
                            preA_channel +
                            ". Weigh interface/global enrichment before an analytic "
                            "C1 build.";
                    } else {
                        branch = "INCONCLUSIVE -- surgery arm did not run (factor "
                                 "failed before verdict).";
                    }
                    if (near_singular)
                        branch +=
                            " [NOTE: M_surgery was near-singular (a few bet/phi gauge "
                            "nulls, MUMPS-projected); the frozen same-field skeleton "
                            "is factorable only after projecting those modes -- a real "
                            "C1 build must pin/regularize them explicitly.]";
                    std::cout << "  RUBRIC BRANCH: " << branch << '\n';
                    std::cout << "  (pre-B: ran=" << (preB_ran ? 1 : 0)
                              << " near_singular=" << (near_singular ? 1 : 0)
                              << " rel@300=" << preB_r300 << " rel@600=" << preB_r600
                              << " non_decel=" << (preB_non_decel ? 1 : 0)
                              << " e:H-col=" << eH_share << " e:phi-col=" << ePhi_share
                              << " e:global=" << eGl_share << " e:seam=" << eSm_share
                              << " e:high-p=" << eHp_share << ")\n";
                    std::cout << "  (pre-A: channel=" << preA_channel
                              << " |theta|<1e-2=" << n_below_1e2
                              << " |theta|<1e-3=" << n_below_1e3 << ")\n";
                    std::cout << "############ ARM 18 PRE-GATES END ############\n";
                    std::cout.flush();
                    return;
                }

                // ====================================================================
                // ARM 19: RANGE-SPACE SPLITTING of the global-column sector
                // (PCOARSE_ARM19). Arm-18pre measured A*M_arm8 for the first
                // time: NO near-null cluster (smallest |theta|=0.041), the ~40 slowest
                // modes decelerate in the {global-scalar column x tau/interface row}
                // coupled sector. Naive in-loop global-block LS (arm-14a) AMPLIFIED
                // (1.31e-2, worse than baseline). Arm 19 excises range(A_G) (G = the
                // 204 col_dom==-1 global columns) from the Krylov problem EXACTLY:
                // dense QR of A[:,G]; project the operator and RHS onto range(A_G)^perp
                // (P = I - Q1 Q1^T); run GMRES in the complement so the Krylov space
                // never sees range(A_G) (no in-loop amplification by construction);
                // recover the global block in closed form post-loop
                // (y = R^-1 Q1^T (b - A x_rest), x = x_rest + E_G y). This is the last
                // untested lever; INERT -> the p-coarse PC lane closes at E-W, FINAL.
                // Self-contained; returns before the arm-14/Stage-A flow.
                // ====================================================================
                if (env_flag_enabled("PCOARSE_ARM19", false)) {
                    std::cout << "\n############ ARM 19: RANGE-SPACE SPLITTING of the "
                                 "global-column sector ############\n";
                    std::cout.flush();
                    const std::size_t size_n = static_cast<std::size_t>(n_f);

                    // --- global-column set G (same predicate arms 14/18pre use) ---
                    std::vector<char> is_global(size_n, 0);
                    for (int cc = 0; cc < n_f; ++cc) {
                        const ColumnInfo& ci = cmap[static_cast<std::size_t>(cc)];
                        if (ci.is_var_domain || ci.var_double_idx >= 0 || ci.domain < 0)
                            is_global[static_cast<std::size_t>(cc)] = 1;
                    }
                    std::vector<int> G_cols;
                    G_cols.reserve(256);
                    for (int cc = 0; cc < n_f; ++cc)
                        if (is_global[static_cast<std::size_t>(cc)])
                            G_cols.push_back(cc);
                    const int g = static_cast<int>(G_cols.size());
                    std::vector<int> g_index(size_n, -1);
                    for (int j = 0; j < g; ++j)
                        g_index[static_cast<std::size_t>(
                            G_cols[static_cast<std::size_t>(j)])] = j;
                    std::cout << "  |G| = |col_dom==-1| global columns = " << g << '\n';
                    {
                        // cross-check vs the arm-8 global QR block (must be identical).
                        const SchwarzBlock& gblk = blocks.back();
                        bool match = (static_cast<int>(gblk.cols.size()) == g);
                        for (std::size_t j = 0; match && j < gblk.cols.size(); ++j)
                            if (!is_global[static_cast<std::size_t>(gblk.cols[j])])
                                match = false;
                        std::cout << "  (cross-check) arm-8 global block cols="
                                  << gblk.cols.size() << " identical="
                                  << (match ? "YES" : "NO") << '\n';
                    }
                    std::cout.flush();
                    if (g <= 0) {
                        std::cout << "  ARM 19 ABORT: no global columns.\n";
                        std::cout.flush();
                        return;
                    }

                    // column taxonomy for the final attribution.
                    std::vector<char> is_tm_row(size_n, 0);
                    for (int rr = 0; rr < n_f; ++rr)
                        if (rr < static_cast<int>(tmeta.size()) &&
                            tmeta[static_cast<std::size_t>(rr)].taxonomy ==
                                RowTaxonomy::TauMatch)
                            is_tm_row[static_cast<std::size_t>(rr)] = 1;
                    std::vector<char> col_seam(size_n, 0);
                    for (long long e = 0; e < nnz_f; ++e) {
                        const std::size_t ee = static_cast<std::size_t>(e);
                        if (is_tm_row[static_cast<std::size_t>(coo.irn[ee] - 1)])
                            col_seam[static_cast<std::size_t>(coo.jcn[ee] - 1)] = 1;
                    }

                    // ============================================================
                    // (1) Densify A_G (col-major n x g) + LAPACK QR (Householder form).
                    // ============================================================
                    const auto tqr0 = std::chrono::steady_clock::now();
                    std::vector<double> AG(size_n * static_cast<std::size_t>(g), 0.0);
                    for (long long e = 0; e < nnz_f; ++e) {
                        const std::size_t ee = static_cast<std::size_t>(e);
                        const int gc =
                            g_index[static_cast<std::size_t>(coo.jcn[ee] - 1)];
                        if (gc >= 0)
                            AG[static_cast<std::size_t>(coo.irn[ee] - 1) +
                               static_cast<std::size_t>(gc) * size_n] += coo.a[ee];
                    }
                    std::cout << "  A_G densified: n=" << n_f << " g=" << g << " ("
                              << (static_cast<double>(size_n) *
                                  static_cast<double>(g) * 8.0 / 1e6)
                              << " MB)\n";
                    std::cout.flush();
                    std::vector<double> qr_tau(static_cast<std::size_t>(g), 0.0);
                    {
                        int qm = n_f, qn = g, qlda = std::max(1, n_f), info = 0,
                            lwork = -1;
                        double wq = 0.0;
                        dgeqrf_(&qm, &qn, AG.data(), &qlda, qr_tau.data(), &wq, &lwork,
                                &info);
                        lwork = (info == 0) ? static_cast<int>(wq) : std::max(1, 2 * g);
                        std::vector<double> work(
                            static_cast<std::size_t>(std::max(1, lwork)));
                        dgeqrf_(&qm, &qn, AG.data(), &qlda, qr_tau.data(), work.data(),
                                &lwork, &info);
                        if (info != 0) {
                            std::cout << "  ARM 19 ABORT: dgeqrf(A_G) info=" << info
                                      << '\n';
                            std::cout.flush();
                            return;
                        }
                    }
                    const double t_qr = std::chrono::duration<double>(
                                            std::chrono::steady_clock::now() - tqr0)
                                            .count();

                    // ---- R-diagonal census + dgecon condition estimate ----
                    double rii_min = std::numeric_limits<double>::infinity(),
                           rii_max = 0.0;
                    int argmin = -1;
                    for (int i = 0; i < g; ++i) {
                        const double v =
                            std::fabs(AG[static_cast<std::size_t>(i) +
                                         static_cast<std::size_t>(i) * size_n]);
                        if (v < rii_min) {
                            rii_min = v;
                            argmin = i;
                        }
                        if (v > rii_max)
                            rii_max = v;
                    }
                    // compact g x g copy of R for dgecon (dgetrf overwrites its input).
                    double rcond = 0.0;
                    {
                        std::vector<double> Rmat(
                            static_cast<std::size_t>(g) * static_cast<std::size_t>(g),
                            0.0);
                        for (int j = 0; j < g; ++j)
                            for (int i = 0; i <= j; ++i)
                                Rmat[static_cast<std::size_t>(i) +
                                     static_cast<std::size_t>(j) *
                                         static_cast<std::size_t>(g)] =
                                    AG[static_cast<std::size_t>(i) +
                                       static_cast<std::size_t>(j) * size_n];
                        char norm1 = '1';
                        int gn = g, glda = std::max(1, g), info = 0;
                        std::vector<double> wk(
                            static_cast<std::size_t>(std::max(1, 4 * g)));
                        double anorm =
                            dlange_(&norm1, &gn, &gn, Rmat.data(), &glda, wk.data());
                        std::vector<int> ipiv(
                            static_cast<std::size_t>(std::max(1, g)));
                        dgetrf_(&gn, &gn, Rmat.data(), &glda, ipiv.data(), &info);
                        if (info == 0) {
                            std::vector<int> iwk(
                                static_cast<std::size_t>(std::max(1, g)));
                            int cinfo = 0;
                            dgecon_(&norm1, &gn, Rmat.data(), &glda, &anorm, &rcond,
                                    wk.data(), iwk.data(), &cinfo);
                            if (cinfo != 0)
                                rcond = 0.0;
                        } else {
                            std::cout << "  (dgecon) dgetrf(R) info=" << info
                                      << " -> R singular in LU\n";
                        }
                    }
                    std::cout << "  R-diag census: |R_ii| min=" << rii_min << " (G["
                              << argmin
                              << "]=" << (argmin >= 0 ? G_cols[static_cast<std::size_t>(
                                                            argmin)]
                                                      : -1)
                              << ") max=" << rii_max << " ratio="
                              << (rii_max > 0.0 ? rii_min / rii_max : 0.0)
                              << " dgecon_rcond(R)=" << rcond << " QR_wall=" << t_qr
                              << "s\n";
                    std::cout.flush();
                    const double rank_tol = 1e-12 * rii_max;
                    if (rii_min < rank_tol) {
                        std::cout << "  STRUCTURAL VERDICT: R rank-deficient (min|R_ii|="
                                  << rii_min << " < 1e-12*max=" << rank_tol
                                  << "); range(A_G) not full-rank. Depressed cols:";
                        for (int i = 0; i < g; ++i) {
                            const double v = std::fabs(
                                AG[static_cast<std::size_t>(i) +
                                   static_cast<std::size_t>(i) * size_n]);
                            if (v < rank_tol)
                                std::cout << ' '
                                          << G_cols[static_cast<std::size_t>(i)];
                        }
                        std::cout << "\n  ARM 19 STOP: exact range-split ill-posed on "
                                     "the global block.\n"
                                     "############ ARM 19 END ############\n";
                        std::cout.flush();
                        return;
                    }

                    // ============================================================
                    // (2) Projector P = I - Q1 Q1^T ; M_rest ; A_op = P*A.
                    // ============================================================
                    // in place: v <- Q^T v ('T') or Q v ('N') via Householder form.
                    auto apply_Q = [&](std::vector<double>& v, char trans) {
                        char side = 'L';
                        int qm = n_f, qn = 1, qk = g, qlda = std::max(1, n_f),
                            qldc = std::max(1, n_f), info = 0, lwork = -1;
                        double wq = 0.0;
                        dormqr_(&side, &trans, &qm, &qn, &qk, AG.data(), &qlda,
                                qr_tau.data(), v.data(), &qldc, &wq, &lwork, &info);
                        lwork = (info == 0) ? static_cast<int>(wq) : std::max(1, n_f);
                        std::vector<double> work(
                            static_cast<std::size_t>(std::max(1, lwork)));
                        dormqr_(&side, &trans, &qm, &qn, &qk, AG.data(), &qlda,
                                qr_tau.data(), v.data(), &qldc, work.data(), &lwork,
                                &info);
                    };
                    auto project_perp = [&](std::vector<double>& v) {
                        std::vector<double> w(v);
                        apply_Q(w, 'T'); // w = Q^T v
                        for (int i = g; i < n_f; ++i)
                            w[static_cast<std::size_t>(i)] = 0.0; // keep first g comps
                        apply_Q(w, 'N'); // w = Q1 (Q1^T v)
                        for (int i = 0; i < n_f; ++i)
                            v[static_cast<std::size_t>(i)] -=
                                w[static_cast<std::size_t>(i)];
                    };

                    // one-shot finiteness guards (GMRES has no non-finite guard).
                    bool mrest_checked = false, aop_checked = false;
                    auto finite_guard = [&](const std::vector<double>& v,
                                            const char* tag, bool& flag) {
                        if (flag)
                            return;
                        flag = true;
                        long long bad = 0;
                        for (double x : v)
                            if (!std::isfinite(x))
                                ++bad;
                        std::cout << "  [finite-guard] first " << tag << " apply: "
                                  << bad << " non-finite of " << v.size() << '\n';
                        std::cout.flush();
                    };

                    // M_rest(r) = pc_arm8(r) with the G-components ZEROED (corrections
                    // stay in the complement so A*correction ignores A_G).
                    KrylovOperator M_rest = [&](const std::vector<double>& r,
                                                std::vector<double>& z) {
                        pc_arm8(r, z);
                        for (int j = 0; j < g; ++j)
                            z[static_cast<std::size_t>(
                                G_cols[static_cast<std::size_t>(j)])] = 0.0;
                        finite_guard(z, "M_rest", mrest_checked);
                    };
                    // A_op(v) = P (A v).
                    KrylovOperator A_op = [&](const std::vector<double>& v,
                                             std::vector<double>& out) {
                        coo_spmv(v, out);
                        project_perp(out);
                        finite_guard(out, "A_op", aop_checked);
                    };

                    std::vector<double> Pb(b);
                    project_perp(Pb);
                    double Pb_norm = 0.0;
                    for (double v : Pb)
                        Pb_norm += v * v;
                    Pb_norm = std::sqrt(Pb_norm);
                    std::cout << "  ||b||=" << b_norm << " ||P b||=" << Pb_norm
                              << " (fraction of b in range(A_G)^perp="
                              << (b_norm > 0.0 ? Pb_norm / b_norm : 0.0) << ")\n";
                    std::cout.flush();

                    // closed-form global-block recovery. Returns r_true = b - A x_full,
                    // sets rel_true, ||Q1^T r_true|| (LS-exactness check), and x_full.
                    auto recover =
                        [&](const std::vector<double>& x_rest, double& rel_true,
                            double& qt_resid,
                            std::vector<double>& x_full) -> std::vector<double> {
                        std::vector<double> ax;
                        coo_spmv(x_rest, ax);
                        std::vector<double> rr(size_n);
                        for (int i = 0; i < n_f; ++i)
                            rr[static_cast<std::size_t>(i)] =
                                b[static_cast<std::size_t>(i)] -
                                ax[static_cast<std::size_t>(i)];
                        // y = R^-1 (Q1^T rr)
                        std::vector<double> w(rr);
                        apply_Q(w, 'T');
                        std::vector<double> y(w.begin(), w.begin() + g);
                        {
                            char uplo = 'U', tn = 'N', diag = 'N';
                            int rn = g, rnrhs = 1, rlda = std::max(1, n_f),
                                rldb = std::max(1, g), info = 0;
                            dtrtrs_(&uplo, &tn, &diag, &rn, &rnrhs, AG.data(), &rlda,
                                    y.data(), &rldb, &info);
                            if (info != 0)
                                std::cout << "  [recover] dtrtrs info=" << info << '\n';
                        }
                        x_full = x_rest;
                        for (int j = 0; j < g; ++j)
                            x_full[static_cast<std::size_t>(
                                G_cols[static_cast<std::size_t>(j)])] +=
                                y[static_cast<std::size_t>(j)];
                        std::vector<double> ax2;
                        coo_spmv(x_full, ax2);
                        std::vector<double> rt(size_n);
                        double nrm = 0.0;
                        for (int i = 0; i < n_f; ++i) {
                            rt[static_cast<std::size_t>(i)] =
                                b[static_cast<std::size_t>(i)] -
                                ax2[static_cast<std::size_t>(i)];
                            nrm += rt[static_cast<std::size_t>(i)] *
                                   rt[static_cast<std::size_t>(i)];
                        }
                        rel_true = (b_norm > 0.0) ? std::sqrt(nrm) / b_norm
                                                  : std::sqrt(nrm);
                        std::vector<double> wq(rt);
                        apply_Q(wq, 'T');
                        double qn2 = 0.0;
                        for (int i = 0; i < g; ++i)
                            qn2 += wq[static_cast<std::size_t>(i)] *
                                   wq[static_cast<std::size_t>(i)];
                        qt_resid = std::sqrt(qn2);
                        return rt;
                    };

                    // projected GMRES driver (optional Ritz capture); returns x_rest.
                    auto run_projected =
                        [&](int maxit, std::vector<std::vector<double>>* Vout,
                            std::vector<std::vector<double>>* Hout,
                            std::vector<double>& curve_out) -> std::vector<double> {
                        std::vector<double> x_rest(size_n, 0.0);
                        curve_out.clear();
                        const int saved = gmres_config.max_iters;
                        gmres_config.max_iters = maxit;
                        gmres_config.residual_history = &curve_out;
                        gmres_config.arnoldi_basis_out = Vout;
                        gmres_config.hessenberg_raw_out = Hout;
                        const auto t0 = std::chrono::steady_clock::now();
                        GmresStatus st = right_preconditioned_gmres(Pb, x_rest, A_op,
                                                                    M_rest, gmres_config);
                        const double wall = std::chrono::duration<double>(
                                                std::chrono::steady_clock::now() - t0)
                                                .count();
                        gmres_config.residual_history = nullptr;
                        gmres_config.arnoldi_basis_out = nullptr;
                        gmres_config.hessenberg_raw_out = nullptr;
                        gmres_config.max_iters = saved;
                        std::cout << "  [proj GMRES @" << maxit
                                  << "] converged=" << std::boolalpha << st.converged
                                  << " iters=" << st.iterations << " proj_rel="
                                  << (Pb_norm > 0.0 ? st.residual_norm / Pb_norm
                                                    : st.residual_norm)
                                  << " wall=" << wall << "s\n";
                        std::cout.flush();
                        return x_rest;
                    };

                    // Ritz spectrum of a captured (basis, raw-Hessenberg): dgeev of the
                    // leading m x m raw Hessenberg -> smallest-60 |theta| + histogram.
                    auto ritz_report =
                        [&](std::vector<std::vector<double>>& Vb,
                            std::vector<std::vector<double>>& Hr, const char* label,
                            int& n_lo, int& n_dec1, double& smallest) {
                        n_lo = 0;
                        n_dec1 = 0;
                        smallest = 0.0;
                        const int m = static_cast<int>(Vb.size());
                        if (m < 2 || static_cast<int>(Hr.size()) < m + 1) {
                            std::cout << "  [" << label << "] Ritz SKIP: m=" << m
                                      << '\n';
                            return;
                        }
                        std::vector<double> A_col(
                            static_cast<std::size_t>(m) * static_cast<std::size_t>(m),
                            0.0);
                        for (int j = 0; j < m; ++j)
                            for (int i = 0; i < m; ++i)
                                A_col[static_cast<std::size_t>(i) +
                                      static_cast<std::size_t>(j) *
                                          static_cast<std::size_t>(m)] =
                                    Hr[static_cast<std::size_t>(i)]
                                      [static_cast<std::size_t>(j)];
                        std::vector<double> wr(static_cast<std::size_t>(m), 0.0),
                            wi(static_cast<std::size_t>(m), 0.0), vdum(1, 0.0);
                        char jn = 'N';
                        int mm = m, ld1 = 1, info = 0, lwork = -1;
                        double wq = 0.0;
                        dgeev_(&jn, &jn, &mm, A_col.data(), &mm, wr.data(), wi.data(),
                               vdum.data(), &ld1, vdum.data(), &ld1, &wq, &lwork,
                               &info);
                        lwork = (info == 0) ? static_cast<int>(wq) : std::max(1, 4 * m);
                        std::vector<double> work(
                            static_cast<std::size_t>(std::max(1, lwork)));
                        dgeev_(&jn, &jn, &mm, A_col.data(), &mm, wr.data(), wi.data(),
                               vdum.data(), &ld1, vdum.data(), &ld1, work.data(),
                               &lwork, &info);
                        if (info != 0) {
                            std::cout << "  [" << label << "] dgeev info=" << info
                                      << '\n';
                            return;
                        }
                        std::vector<double> mag;
                        for (int j = 0; j < m;) {
                            if (wi[static_cast<std::size_t>(j)] == 0.0) {
                                mag.push_back(
                                    std::fabs(wr[static_cast<std::size_t>(j)]));
                                j += 1;
                            } else {
                                const double mg =
                                    std::hypot(wr[static_cast<std::size_t>(j)],
                                               wi[static_cast<std::size_t>(j)]);
                                mag.push_back(mg);
                                mag.push_back(mg);
                                j += 2;
                            }
                        }
                        std::sort(mag.begin(), mag.end());
                        smallest = mag.empty() ? 0.0 : mag[0];
                        int hist[7] = {0, 0, 0, 0, 0, 0, 0};
                        for (double v : mag) {
                            if (v < 1e-3)
                                ++hist[0];
                            else if (v < 1e-2)
                                ++hist[1];
                            else if (v < 1e-1)
                                ++hist[2];
                            else if (v < 1e0)
                                ++hist[3];
                            else if (v < 1e1)
                                ++hist[4];
                            else if (v < 1e2)
                                ++hist[5];
                            else
                                ++hist[6];
                            if (v < 1e-2)
                                ++n_lo;
                        }
                        n_dec1 = hist[2]; // [1e-2,1e-1)
                        std::cout << "  [" << label << "] Ritz (eigs leading " << m
                                  << "x" << m
                                  << " raw Hessenberg) smallest-60 |theta|:\n";
                        const int nshow =
                            std::min<int>(60, static_cast<int>(mag.size()));
                        for (int t = 0; t < nshow; ++t)
                            std::cout << "    [" << (t + 1) << "] "
                                      << mag[static_cast<std::size_t>(t)] << '\n';
                        std::cout << "  [" << label << "] |theta| hist: <1e-3:"
                                  << hist[0] << " [1e-3,1e-2):" << hist[1]
                                  << " [1e-2,1e-1):" << hist[2] << " [1e-1,1):"
                                  << hist[3] << " [1,1e1):" << hist[4] << " [1e1,1e2):"
                                  << hist[5] << " >=1e2:" << hist[6]
                                  << " smallest=" << smallest << '\n';
                        std::cout.flush();
                    };

                    // ============================================================
                    // (3) 8r2 baseline @600 (unprojected A*M_arm8) + Ritz capture.
                    // ============================================================
                    const std::size_t a8_idx = arms.size();
                    std::vector<std::vector<double>> V8, H8;
                    gmres_config.arnoldi_basis_out = &V8;
                    gmres_config.hessenberg_raw_out = &H8;
                    run_arm("8r2_baseline", pc_arm8);
                    gmres_config.arnoldi_basis_out = nullptr;
                    gmres_config.hessenberg_raw_out = nullptr;
                    int unp_lo = 0, unp_dec1 = 0;
                    double unp_small = 0.0;
                    std::cout << "\n==== UNPROJECTED A*M_arm8 Ritz (8r2 baseline @"
                              << gmres_config.max_iters << ") ====\n";
                    ritz_report(V8, H8, "unproj", unp_lo, unp_dec1, unp_small);
                    std::vector<std::vector<double>>().swap(V8);
                    std::vector<std::vector<double>>().swap(H8);

                    // ============================================================
                    // (4) Arm-19 projected runs @300 and @600 (two-prefix pattern).
                    // ============================================================
                    std::vector<double> curve300, curve600;
                    std::vector<double> x_rest300 =
                        run_projected(300, nullptr, nullptr, curve300);
                    std::vector<double> x300;
                    double rel300 = 0.0, qt300 = 0.0;
                    std::vector<double> rt300 = recover(x_rest300, rel300, qt300, x300);
                    std::cout << "  [19 @300] rel_true=" << rel300
                              << " ||Q1^T r_true||=" << qt300 << '\n';
                    std::cout.flush();

                    std::vector<std::vector<double>> V19, H19;
                    std::vector<double> x_rest600 =
                        run_projected(600, &V19, &H19, curve600);
                    std::vector<double> x600;
                    double rel600 = 0.0, qt600 = 0.0;
                    std::vector<double> rt600 = recover(x_rest600, rel600, qt600, x600);
                    std::cout << "  [19 @600] rel_true=" << rel600
                              << " ||Q1^T r_true||=" << qt600 << '\n';
                    std::cout.flush();

                    int prj_lo = 0, prj_dec1 = 0;
                    double prj_small = 0.0;
                    std::cout << "\n==== PROJECTED P*A*M_rest Ritz (arm-19 @600) ====\n";
                    ritz_report(V19, H19, "proj", prj_lo, prj_dec1, prj_small);
                    std::vector<std::vector<double>>().swap(V19);
                    std::vector<std::vector<double>>().swap(H19);

                    // ============================================================
                    // (5) Attribution: e = A^-1 r_true (1 refinement); energy shares.
                    // ============================================================
                    MumpsLinearSolver fine_solver(n_f, config.mumps.ordering, false, 0,
                                                  mumps_runtime_state.icntl14,
                                                  MPI_COMM_SELF);
                    fine_solver.set_pattern(n_f, nnz_f, coo.irn.data(),
                                            coo.jcn.data());
                    fine_solver.analyze_pattern();
                    fine_solver.factor_analyzed(coo.a.data());
                    mumps_runtime_state.icntl14 = fine_solver.last_icntl14();
                    auto attrib = [&](const std::vector<double>& rin, const char* tag,
                                      double& e_gl, double& e_sm, double& e_hp) {
                        std::vector<double> e(rin);
                        fine_solver.solve(e.data());
                        std::vector<double> ae;
                        coo_spmv(e, ae);
                        std::vector<double> corr(size_n);
                        for (int i = 0; i < n_f; ++i)
                            corr[static_cast<std::size_t>(i)] =
                                rin[static_cast<std::size_t>(i)] -
                                ae[static_cast<std::size_t>(i)];
                        fine_solver.solve(corr.data());
                        for (int i = 0; i < n_f; ++i)
                            e[static_cast<std::size_t>(i)] +=
                                corr[static_cast<std::size_t>(i)];
                        double tot = 0, gl = 0, sm = 0, hp = 0;
                        for (int i = 0; i < n_f; ++i) {
                            const double e2 = e[static_cast<std::size_t>(i)] *
                                              e[static_cast<std::size_t>(i)];
                            tot += e2;
                            if (is_global[static_cast<std::size_t>(i)])
                                gl += e2;
                            if (col_seam[static_cast<std::size_t>(i)])
                                sm += e2;
                            if (!fine_col_matched[static_cast<std::size_t>(i)])
                                hp += e2;
                        }
                        const double iv = (tot > 0.0) ? 1.0 / tot : 0.0;
                        e_gl = gl * iv;
                        e_sm = sm * iv;
                        e_hp = hp * iv;
                        std::cout << "  attrib[" << tag << "] e:global=" << e_gl
                                  << " e:seam=" << e_sm << " e:high-p=" << e_hp << '\n';
                        std::cout.flush();
                    };
                    double e8_gl = 0, e8_sm = 0, e8_hp = 0, e19_gl = 0, e19_sm = 0,
                           e19_hp = 0;
                    if (a8_idx < arms.size()) {
                        std::vector<double> ax8;
                        coo_spmv(arms[a8_idx].x, ax8);
                        std::vector<double> r8(size_n);
                        for (int i = 0; i < n_f; ++i)
                            r8[static_cast<std::size_t>(i)] =
                                b[static_cast<std::size_t>(i)] -
                                ax8[static_cast<std::size_t>(i)];
                        attrib(r8, "8r2_baseline", e8_gl, e8_sm, e8_hp);
                    }
                    attrib(rt600, "19_rangesplit", e19_gl, e19_sm, e19_hp);

                    // ============================================================
                    // (6) Results table + rubric.
                    // ============================================================
                    auto rel_at = [&](const ArmResult& a, int k) -> double {
                        if (a.curve.empty())
                            return 0.0;
                        const int idx =
                            std::min(k, static_cast<int>(a.curve.size())) - 1;
                        return (b_norm > 0.0)
                                   ? a.curve[static_cast<std::size_t>(idx)] / b_norm
                                   : a.curve[static_cast<std::size_t>(idx)];
                    };
                    const double b8_300 =
                        (a8_idx < arms.size()) ? rel_at(arms[a8_idx], 300) : 0.0;
                    const double b8_600 =
                        (a8_idx < arms.size()) ? rel_at(arms[a8_idx], 600) : 0.0;
                    auto ord = [](double from, double to) -> double {
                        return (from > 0.0 && to > 0.0) ? std::log10(from / to) : 0.0;
                    };
                    const double o8_a = ord(1.0, b8_300), o8_b = ord(b8_300, b8_600);
                    const double o19_a = ord(1.0, rel300),
                                 o19_b = ord(rel300, rel600);
                    std::cout << "\n=== ARM 19 RESULTS TABLE (canonical fixture) ===\n";
                    std::cout << "  arm           | rel@300      | rel@600      | "
                                 "ord[0,300) | ord[300,600) | decel\n";
                    std::cout << "  8r2_baseline  | " << b8_300 << " | " << b8_600
                              << " | " << o8_a << " | " << o8_b << " | "
                              << (o8_b < o8_a ? "YES" : "no") << "  (true resid)\n";
                    std::cout << "  19_rangesplit | " << rel300 << " | " << rel600
                              << " | " << o19_a << " | " << o19_b << " | "
                              << (o19_b < o19_a ? "YES" : "no") << "  (rel_true)\n";
                    std::cout << "  attribution: 8r2  e:global=" << e8_gl
                              << " e:seam=" << e8_sm << " e:high-p=" << e8_hp << '\n';
                    std::cout << "               19   e:global=" << e19_gl
                              << " e:seam=" << e19_sm << " e:high-p=" << e19_hp << '\n';
                    std::cout << "  ||Q1^T r_true||: @300=" << qt300 << " @600="
                              << qt600 << " (LS exactness; ~machine-eps*scale expected)"
                              << '\n';
                    std::cout << "  Ritz: unproj smallest=" << unp_small
                              << " [1e-2,1e-1)=" << unp_dec1 << " <1e-2=" << unp_lo
                              << "  ||  proj smallest=" << prj_small
                              << " [1e-2,1e-1)=" << prj_dec1 << " <1e-2=" << prj_lo
                              << '\n';
                    std::cout.flush();

                    std::cout << "\n=== ARM 19 RUBRIC ===\n";
                    const double base = 0.0121022; // guard baseline (8r2 rel@600)
                    const bool non_decel = (o19_b >= o19_a - 1e-12);
                    const bool egl_collapsed = (e19_gl < 0.1);
                    const bool tail_thinned =
                        (prj_lo + prj_dec1) < (unp_lo + unp_dec1);
                    std::string branch;
                    if (rel600 <= 1e-4 && non_decel && egl_collapsed && tail_thinned) {
                        branch =
                            "PASS -- rel_true@600<=1e-4, non-decelerating, e:global "
                            "collapsed, projected Ritz tail thinned. p-coarse PC lane "
                            "RESCUED. Productionize: QR of A_G per Newton step is "
                            "trivial (n x 204); M_arm8 setup cost remains the res19 "
                            "question.";
                    } else if (rel600 <= base / 5.0 && o19_b > o8_b && tail_thinned) {
                        branch =
                            "PARTIAL -- >=5x better than baseline 0.0121 with slope "
                            "improvement and thinned Ritz tail. One follow-up "
                            "discussion warranted.";
                    } else {
                        branch =
                            "INERT/WORSE -- range-space splitting does not rescue the "
                            "lane. Every idea in the p-coarse ledger is now measured "
                            "dead. THE LANE CLOSES AT E-W GRADE, FINAL.";
                    }
                    std::cout << "  RUBRIC BRANCH: " << branch << '\n';
                    std::cout << "  (rel_true@600=" << rel600 << " base=" << base
                              << " non_decel=" << (non_decel ? 1 : 0)
                              << " e:global=" << e19_gl
                              << " tail_thinned=" << (tail_thinned ? 1 : 0) << ")\n";
                    std::cout << "############ ARM 19 END ############\n";
                    std::cout.flush();
                    return;
                }

                // ====================================================================
                // ARM 14: multiplicative / bordered treatment of the global-scalar
                // channel (PCOARSE_ARM14). Arm-13 proved e=A^-1 r8 lives
                // ~0.99 in the global/scalar columns (the g-column arm-8 "global" QR
                // block); additive composition cannot reach them. 14a = multiplic-
                // ative (Gauss-Seidel) global pass on the arm-8-updated residual;
                // 14a2 = symmetrized (global pre + arm-8 + global post); 14b =
                // bordered block-LU (approximate Schur on the global block). Runs
                // 8r2 baseline + 14a + 14a2 + 14b @maxit, then reruns the arm-13(C)
                // fine-solve attribution on each final residual (global share must
                // collapse). Self-contained: returns before the arm-13/Stage-A flow.
                // ====================================================================
                if (env_flag_enabled("PCOARSE_ARM14", false)) {
                    std::cout << "\n############ ARM 14: global-scalar channel "
                                 "treatment ############\n";

                    // --- PRE-CHECK: global set == arm-8 global QR block ---
                    std::vector<char> is_global(static_cast<std::size_t>(n_f), 0);
                    for (int cc = 0; cc < n_f; ++cc) {
                        const ColumnInfo& ci = cmap[static_cast<std::size_t>(cc)];
                        if (ci.is_var_domain || ci.var_double_idx >= 0 ||
                            ci.domain < 0)
                            is_global[static_cast<std::size_t>(cc)] = 1;
                    }
                    long long g_pred = 0;
                    for (int cc = 0; cc < n_f; ++cc)
                        g_pred += is_global[static_cast<std::size_t>(cc)];
                    SchwarzBlock& gblk = blocks.back();
                    const std::vector<int> G_cols = gblk.cols;
                    const int g = static_cast<int>(G_cols.size());
                    bool set_match = (static_cast<long long>(g) == g_pred);
                    for (int j = 0; set_match && j < g; ++j)
                        if (!is_global[static_cast<std::size_t>(
                                G_cols[static_cast<std::size_t>(j)])])
                            set_match = false;
                    std::vector<int> g_index(static_cast<std::size_t>(n_f), -1);
                    for (int j = 0; j < g; ++j)
                        g_index[static_cast<std::size_t>(
                            G_cols[static_cast<std::size_t>(j)])] = j;
                    std::cout << "  (pre-check) |col_dom==-1|=" << g_pred
                              << "  |arm-8 global block|=" << g << "  identical="
                              << (set_match ? "YES" : "NO") << "  g=" << g << '\n';
                    std::cout.flush();
                    if (!set_match) {
                        std::cout << "  ARM 14 ABORT: global set != arm-8 global "
                                     "block; cannot treat exactly the 0.99-energy "
                                     "columns.\n";
                        std::cout.flush();
                        return;
                    }

                    // Q_G: least-squares solve of the global QR block ONLY.
                    auto global_apply = [&](const std::vector<double>& r,
                                            std::vector<double>& z) {
                        z.assign(static_cast<std::size_t>(n_f), 0.0);
                        const int m = gblk.m, n_touch = gblk.n_touch;
                        if (m <= 0 || n_touch <= 0)
                            return;
                        std::vector<double> rg(static_cast<std::size_t>(n_touch));
                        for (int i = 0; i < n_touch; ++i)
                            rg[static_cast<std::size_t>(i)] =
                                r[static_cast<std::size_t>(
                                    gblk.touch_global[static_cast<std::size_t>(i)])];
                        char side = 'L', trans = 'T';
                        int qm = n_touch, qn = 1, qk = m,
                            qlda = std::max(1, n_touch),
                            qldc = std::max(1, n_touch), qinfo = 0, lwork = -1;
                        double wq = 0.0;
                        dormqr_(&side, &trans, &qm, &qn, &qk, gblk.a_qr.data(),
                                &qlda, gblk.tau.data(), rg.data(), &qldc, &wq,
                                &lwork, &qinfo);
                        lwork = (qinfo == 0) ? static_cast<int>(wq)
                                             : std::max(1, n_touch);
                        std::vector<double> work(
                            static_cast<std::size_t>(std::max(1, lwork)));
                        dormqr_(&side, &trans, &qm, &qn, &qk, gblk.a_qr.data(),
                                &qlda, gblk.tau.data(), rg.data(), &qldc,
                                work.data(), &lwork, &qinfo);
                        std::vector<double> delta(rg.begin(), rg.begin() + m);
                        char uplo = 'U', tn = 'N', diag = 'N';
                        int rn = m, rnrhs = 1, rlda = std::max(1, n_touch), rldb = m,
                            rinfo = 0;
                        dtrtrs_(&uplo, &tn, &diag, &rn, &rnrhs, gblk.a_qr.data(),
                                &rlda, delta.data(), &rldb, &rinfo);
                        for (int k = 0; k < m; ++k)
                            z[static_cast<std::size_t>(
                                gblk.cols[static_cast<std::size_t>(k)])] +=
                                delta[static_cast<std::size_t>(k)];
                    };

                    // 14a: multiplicative global pass on the arm-8-updated residual.
                    KrylovOperator pc_14a = [&](const std::vector<double>& r,
                                                std::vector<double>& z) {
                        std::vector<double> z1, az1, rp, zg;
                        pc_arm8(r, z1);
                        coo_spmv(z1, az1);
                        rp.assign(static_cast<std::size_t>(n_f), 0.0);
                        for (int i = 0; i < n_f; ++i)
                            rp[static_cast<std::size_t>(i)] =
                                r[static_cast<std::size_t>(i)] -
                                az1[static_cast<std::size_t>(i)];
                        global_apply(rp, zg);
                        z.assign(static_cast<std::size_t>(n_f), 0.0);
                        for (int i = 0; i < n_f; ++i)
                            z[static_cast<std::size_t>(i)] =
                                z1[static_cast<std::size_t>(i)] +
                                zg[static_cast<std::size_t>(i)];
                    };
                    // 14a2: symmetrized -- global pre, arm-8 middle, global post.
                    KrylovOperator pc_14a2 = [&](const std::vector<double>& r,
                                                 std::vector<double>& z) {
                        std::vector<double> zg0, a0, r1, z1, a1, r2, zg2;
                        global_apply(r, zg0);
                        coo_spmv(zg0, a0);
                        r1.assign(static_cast<std::size_t>(n_f), 0.0);
                        for (int i = 0; i < n_f; ++i)
                            r1[static_cast<std::size_t>(i)] =
                                r[static_cast<std::size_t>(i)] -
                                a0[static_cast<std::size_t>(i)];
                        pc_arm8(r1, z1);
                        coo_spmv(z1, a1);
                        r2.assign(static_cast<std::size_t>(n_f), 0.0);
                        for (int i = 0; i < n_f; ++i)
                            r2[static_cast<std::size_t>(i)] =
                                r1[static_cast<std::size_t>(i)] -
                                a1[static_cast<std::size_t>(i)];
                        global_apply(r2, zg2);
                        z.assign(static_cast<std::size_t>(n_f), 0.0);
                        for (int i = 0; i < n_f; ++i)
                            z[static_cast<std::size_t>(i)] =
                                zg0[static_cast<std::size_t>(i)] +
                                z1[static_cast<std::size_t>(i)] +
                                zg2[static_cast<std::size_t>(i)];
                    };

                    // --- 14b setup: bordered block-LU on the global block ---
                    // W[j] = M_arm8(A e_{G_j})  (g arm-8 applies);
                    // S~ = A[G,G] - A[G,F] W  (dense g x g, col-major), LU.
                    const auto t14b0 = std::chrono::steady_clock::now();
                    std::vector<std::vector<std::pair<int, double>>> gcol_entries(
                        static_cast<std::size_t>(g));
                    struct GRowEntry {
                        int gi;
                        int c;
                        double a;
                    };
                    std::vector<GRowEntry> grow;
                    for (long long e = 0; e < nnz_f; ++e) {
                        const std::size_t ee = static_cast<std::size_t>(e);
                        const int rr = coo.irn[ee] - 1;
                        const int cc = coo.jcn[ee] - 1;
                        const double a = coo.a[ee];
                        const int gc = g_index[static_cast<std::size_t>(cc)];
                        if (gc >= 0)
                            gcol_entries[static_cast<std::size_t>(gc)].push_back(
                                {rr, a});
                        const int gr = g_index[static_cast<std::size_t>(rr)];
                        if (gr >= 0)
                            grow.push_back({gr, cc, a});
                    }
                    std::vector<std::vector<double>> W(
                        static_cast<std::size_t>(g));
                    {
                        std::vector<double> col(static_cast<std::size_t>(n_f));
                        for (int j = 0; j < g; ++j) {
                            std::fill(col.begin(), col.end(), 0.0);
                            for (const auto& pr :
                                 gcol_entries[static_cast<std::size_t>(j)])
                                col[static_cast<std::size_t>(pr.first)] += pr.second;
                            pc_arm8(col, W[static_cast<std::size_t>(j)]);
                        }
                    }
                    // S~ col-major: Stil[i + j*g].
                    std::vector<double> Stil(
                        static_cast<std::size_t>(g) * static_cast<std::size_t>(g),
                        0.0);
                    for (const GRowEntry& en : grow) {
                        const int gj = g_index[static_cast<std::size_t>(en.c)];
                        if (gj >= 0) // A[G,G]
                            Stil[static_cast<std::size_t>(en.gi) +
                                 static_cast<std::size_t>(gj) *
                                     static_cast<std::size_t>(g)] += en.a;
                    }
                    for (const GRowEntry& en : grow) {
                        if (g_index[static_cast<std::size_t>(en.c)] >= 0)
                            continue; // F columns only
                        const std::size_t ci = static_cast<std::size_t>(en.c);
                        for (int j = 0; j < g; ++j)
                            Stil[static_cast<std::size_t>(en.gi) +
                                 static_cast<std::size_t>(j) *
                                     static_cast<std::size_t>(g)] -=
                                en.a * W[static_cast<std::size_t>(j)][ci];
                    }
                    std::vector<int> ipivS(static_cast<std::size_t>(std::max(1, g)));
                    int gg = g, luinfo = 0;
                    dgetrf_(&gg, &gg, Stil.data(), &gg, ipivS.data(), &luinfo);
                    const double t14b_setup =
                        std::chrono::duration<double>(
                            std::chrono::steady_clock::now() - t14b0)
                            .count();
                    std::cout << "  (14b setup) g=" << g
                              << " global-row nnz=" << grow.size()
                              << " Stil LU info=" << luinfo
                              << " wall=" << t14b_setup << "s\n";
                    std::cout.flush();

                    KrylovOperator pc_14b = [&](const std::vector<double>& r,
                                                std::vector<double>& z) {
                        // y_F = M_arm8(r restricted to F).
                        std::vector<double> rF(r);
                        for (int j = 0; j < g; ++j)
                            rF[static_cast<std::size_t>(
                                G_cols[static_cast<std::size_t>(j)])] = 0.0;
                        std::vector<double> yF;
                        pc_arm8(rF, yF);
                        // rhs_G = r_G - A[G,F] y_F.
                        std::vector<double> rhsG(static_cast<std::size_t>(g), 0.0);
                        for (int j = 0; j < g; ++j)
                            rhsG[static_cast<std::size_t>(j)] =
                                r[static_cast<std::size_t>(
                                    G_cols[static_cast<std::size_t>(j)])];
                        for (const GRowEntry& en : grow) {
                            if (g_index[static_cast<std::size_t>(en.c)] >= 0)
                                continue;
                            rhsG[static_cast<std::size_t>(en.gi)] -=
                                en.a * yF[static_cast<std::size_t>(en.c)];
                        }
                        // dG = S~^{-1} rhs_G.
                        std::vector<double> dG(rhsG);
                        char tr = 'N';
                        int gn = g, nrhs = 1, info = 0;
                        dgetrs_(&tr, &gn, &nrhs, Stil.data(), &gn, ipivS.data(),
                                dG.data(), &gn, &info);
                        // z_F = y_F - M_arm8(A[F,G] dG); z_G = dG.
                        std::vector<double> d(static_cast<std::size_t>(n_f), 0.0);
                        for (int j = 0; j < g; ++j)
                            d[static_cast<std::size_t>(
                                G_cols[static_cast<std::size_t>(j)])] =
                                dG[static_cast<std::size_t>(j)];
                        std::vector<double> Ad;
                        coo_spmv(d, Ad);
                        for (int j = 0; j < g; ++j)
                            Ad[static_cast<std::size_t>(
                                G_cols[static_cast<std::size_t>(j)])] = 0.0;
                        std::vector<double> corr;
                        pc_arm8(Ad, corr);
                        z.assign(static_cast<std::size_t>(n_f), 0.0);
                        for (int i = 0; i < n_f; ++i)
                            z[static_cast<std::size_t>(i)] =
                                yF[static_cast<std::size_t>(i)] -
                                corr[static_cast<std::size_t>(i)];
                        for (int j = 0; j < g; ++j)
                            z[static_cast<std::size_t>(
                                G_cols[static_cast<std::size_t>(j)])] =
                                dG[static_cast<std::size_t>(j)];
                    };

                    // --- Run: 8r2 baseline + 14a + 14a2 + 14b @maxit ---
                    const std::size_t a14_first = arms.size();
                    run_arm("8r2_baseline", pc_arm8);
                    run_arm("14a_mult_global", pc_14a);
                    run_arm("14a2_sym_global", pc_14a2);
                    // 14b guard: a singular border S~ (dgetrf info != 0) means
                    // dgetrs divides by a zero pivot -> non-finite z -> the GMRES
                    // Arnoldi has no non-finite guard (reads an unbuilt basis
                    // vector -> UB/segfault). The index-symmetric global block is
                    // rank-deficient here (global columns lack DOF-paired global
                    // rows), so skip the arm rather than crash the run.
                    if (luinfo == 0) {
                        run_arm("14b_bordered_blocklu", pc_14b);
                    } else {
                        std::cout << "  14b_bordered_blocklu: S~ singular (dgetrf "
                                     "info=" << luinfo
                                  << "), arm SKIPPED -- index-symmetric global "
                                     "block rank-deficient (formulation kill, not "
                                     "a channel kill).\n";
                        std::cout.flush();
                    }

                    // --- Fine factor (once) + per-arm attribution rerun ---
                    std::vector<char> col_seam(static_cast<std::size_t>(n_f), 0);
                    for (long long e = 0; e < nnz_f; ++e) {
                        const std::size_t ee = static_cast<std::size_t>(e);
                        const int rr = coo.irn[ee] - 1;
                        if (rr < static_cast<int>(tmeta.size()) &&
                            tmeta[static_cast<std::size_t>(rr)].taxonomy ==
                                RowTaxonomy::TauMatch)
                            col_seam[static_cast<std::size_t>(coo.jcn[ee] - 1)] = 1;
                    }
                    MumpsLinearSolver fine_solver(
                        n_f, config.mumps.ordering, false, 0,
                        mumps_runtime_state.icntl14, MPI_COMM_SELF);
                    fine_solver.set_pattern(n_f, nnz_f, coo.irn.data(),
                                            coo.jcn.data());
                    fine_solver.analyze_pattern();
                    fine_solver.factor_analyzed(coo.a.data());
                    mumps_runtime_state.icntl14 = fine_solver.last_icntl14();

                    struct ArmShare {
                        double global;
                        double seam;
                        double hi;
                        double solve_rel;
                    };
                    auto attrib_share =
                        [&](const std::vector<double>& xarm) -> ArmShare {
                        std::vector<double> ax;
                        coo_spmv(xarm, ax);
                        std::vector<double> r(static_cast<std::size_t>(n_f));
                        for (int i = 0; i < n_f; ++i)
                            r[static_cast<std::size_t>(i)] =
                                b[static_cast<std::size_t>(i)] -
                                ax[static_cast<std::size_t>(i)];
                        std::vector<double> e(r);
                        fine_solver.solve(e.data());
                        std::vector<double> ae;
                        coo_spmv(e, ae);
                        std::vector<double> corr(static_cast<std::size_t>(n_f));
                        for (int i = 0; i < n_f; ++i)
                            corr[static_cast<std::size_t>(i)] =
                                r[static_cast<std::size_t>(i)] -
                                ae[static_cast<std::size_t>(i)];
                        fine_solver.solve(corr.data()); // 1 refinement step
                        for (int i = 0; i < n_f; ++i)
                            e[static_cast<std::size_t>(i)] +=
                                corr[static_cast<std::size_t>(i)];
                        coo_spmv(e, ae);
                        double num = 0.0, den = 0.0, tot = 0.0, gl = 0.0,
                               sm = 0.0, hp = 0.0;
                        for (int i = 0; i < n_f; ++i) {
                            const double d = ae[static_cast<std::size_t>(i)] -
                                             r[static_cast<std::size_t>(i)];
                            num += d * d;
                            den += r[static_cast<std::size_t>(i)] *
                                   r[static_cast<std::size_t>(i)];
                            const double e2 = e[static_cast<std::size_t>(i)] *
                                              e[static_cast<std::size_t>(i)];
                            tot += e2;
                            if (is_global[static_cast<std::size_t>(i)])
                                gl += e2;
                            if (col_seam[static_cast<std::size_t>(i)])
                                sm += e2;
                            if (fine_col_matched[static_cast<std::size_t>(i)] == 0)
                                hp += e2;
                        }
                        const double inv = (tot > 0.0) ? 1.0 / tot : 0.0;
                        return {gl * inv, sm * inv, hp * inv,
                                (den > 0.0) ? std::sqrt(num / den) : 0.0};
                    };

                    // --- Results table + rubric ---
                    std::cout << "\n=== ARM 14 RESULTS (canonical current-ladder "
                                 "fixture) ===\n";
                    std::cout << "  arm | rel@300 | rel@600 | ord[0,300) | "
                                 "ord[300,600) | decel | e:global | e:seam | "
                                 "e:high-p | solve_rel | wall/iter(s)\n";
                    auto rel_at = [&](const ArmResult& a, int k) -> double {
                        if (a.curve.empty())
                            return 0.0;
                        const int idx =
                            std::min(k, static_cast<int>(a.curve.size())) - 1;
                        return (b_norm > 0.0)
                                   ? a.curve[static_cast<std::size_t>(idx)] / b_norm
                                   : a.curve[static_cast<std::size_t>(idx)];
                    };
                    for (std::size_t ai = a14_first; ai < arms.size(); ++ai) {
                        const ArmResult& a = arms[ai];
                        const double r300 = rel_at(a, 300);
                        const double r600 = rel_at(a, 600);
                        const double ord_a =
                            (r300 > 0.0) ? std::log10(1.0 / r300) : 0.0;
                        const double ord_b =
                            (r300 > 0.0 && r600 > 0.0) ? std::log10(r300 / r600)
                                                       : 0.0;
                        const ArmShare sh = attrib_share(a.x);
                        const double wall_per_iter =
                            (a.iters > 0) ? a.wall / a.iters : 0.0;
                        std::cout << "  " << a.name << " | " << r300 << " | "
                                  << r600 << " | " << ord_a << " | " << ord_b
                                  << " | " << (ord_b < ord_a ? "YES" : "no")
                                  << " | " << sh.global << " | " << sh.seam
                                  << " | " << sh.hi << " | " << sh.solve_rel
                                  << " | " << wall_per_iter << '\n';
                    }
                    std::cout.flush();

                    // --- Verdict rubric (14a and 14b are the candidates) ---
                    auto verdict = [&](const std::string& name) {
                        for (std::size_t ai = a14_first; ai < arms.size(); ++ai) {
                            if (arms[ai].name != name)
                                continue;
                            const ArmResult& a = arms[ai];
                            const double r300 = rel_at(a, 300);
                            const double r600 = rel_at(a, 600);
                            const double ord_a =
                                (r300 > 0.0) ? std::log10(1.0 / r300) : 0.0;
                            const double ord_b =
                                (r300 > 0.0 && r600 > 0.0)
                                    ? std::log10(r300 / r600)
                                    : 0.0;
                            const ArmShare sh = attrib_share(a.x);
                            const bool non_decel = (ord_b >= ord_a - 1e-12);
                            const bool coll = (sh.global < 0.1);
                            std::string v;
                            if (r600 <= 1e-6 && coll)
                                v = "STRETCH (full PASS: rel@600<=1e-6 + collapsed)";
                            else if (non_decel && r600 <= 1e-4 && coll)
                                v = "PASS";
                            else if (sh.global > 0.5)
                                v = "FAIL: channel confirmed, composition wrong "
                                    "(global share still >0.5)";
                            else
                                v = "PARTIAL/NO-PASS";
                            std::cout << "  VERDICT " << name << ": " << v
                                      << "  (rel@600=" << r600
                                      << " non_decel=" << (non_decel ? 1 : 0)
                                      << " e:global=" << sh.global << ")\n";
                        }
                    };
                    verdict("14a_mult_global");
                    verdict("14a2_sym_global");
                    verdict("14b_bordered_blocklu");
                    std::cout << "############ ARM 14 END ############\n";
                    std::cout.flush();
                    return;
                }

                // ====================================================================
                // ARM 15: oracle deflation via inverse subspace iteration
                // (PCOARSE_ARM15). Arm-14 proved the arm-8 blocking error
                // lives in near-null directions coupling the 204 global columns to
                // field columns that a frozen-F least-squares cannot reach. Arm-15
                // tests the UPPER BOUND: deflate the EXACT smallest-magnitude modes
                // of A using the in-run fine MUMPS factor. Inverse subspace
                // iteration (V <- A^-1 V, orthonormalize, x10) builds an orthonormal
                // V spanning the k_max=64 smallest modes; a least-squares deflation
                // correction z = M_arm8 r + W_k y (y solves the residual LS against
                // A W_k via a stored QR -- cannot go singular) augments arm-8.
                // Prints the Ritz spectrum (gap vs gapless continuum), the
                // projection f_k of the 8r2 fine-solve error onto W_k, then runs
                // 8r2 baseline + 15a(k=16) + 15b(k=32) + 15c(k=64) with the
                // arm-13(C) attribution rerun. Factors the fine solver FIRST (reused
                // for subspace iteration, projection and attribution). Self-
                // contained: returns before the Stage-A flow.
                // ====================================================================
                if (env_flag_enabled("PCOARSE_ARM15", false)) {
                    std::cout << "\n############ ARM 15: oracle deflation "
                                 "(inverse subspace iteration) ############\n";
                    std::cout.flush();

                    const int k_max = 64;
                    auto vidx = [&](int i, int j) -> std::size_t {
                        return static_cast<std::size_t>(i) +
                               static_cast<std::size_t>(j) *
                                   static_cast<std::size_t>(n_f);
                    };

                    // is_global (same predicate arm-14 uses) + col_seam (TauMatch
                    // columns) for the arm-13(C) attribution table.
                    std::vector<char> is_global(static_cast<std::size_t>(n_f), 0);
                    for (int cc = 0; cc < n_f; ++cc) {
                        const ColumnInfo& ci = cmap[static_cast<std::size_t>(cc)];
                        if (ci.is_var_domain || ci.var_double_idx >= 0 ||
                            ci.domain < 0)
                            is_global[static_cast<std::size_t>(cc)] = 1;
                    }
                    std::vector<char> col_seam(static_cast<std::size_t>(n_f), 0);
                    for (long long e = 0; e < nnz_f; ++e) {
                        const std::size_t ee = static_cast<std::size_t>(e);
                        const int rr = coo.irn[ee] - 1;
                        if (rr < static_cast<int>(tmeta.size()) &&
                            tmeta[static_cast<std::size_t>(rr)].taxonomy ==
                                RowTaxonomy::TauMatch)
                            col_seam[static_cast<std::size_t>(coo.jcn[ee] - 1)] = 1;
                    }

                    // --- Fine factor FIRST (reused for the whole probe) ---
                    const auto tfac0 = std::chrono::steady_clock::now();
                    MumpsLinearSolver fine_solver(
                        n_f, config.mumps.ordering, false, 0,
                        mumps_runtime_state.icntl14, MPI_COMM_SELF);
                    fine_solver.set_pattern(n_f, nnz_f, coo.irn.data(),
                                            coo.jcn.data());
                    fine_solver.analyze_pattern();
                    fine_solver.factor_analyzed(coo.a.data());
                    mumps_runtime_state.icntl14 = fine_solver.last_icntl14();
                    const double t_factor =
                        std::chrono::duration<double>(
                            std::chrono::steady_clock::now() - tfac0)
                            .count();
                    std::cout << "  (setup) fine MUMPS factor wall=" << t_factor
                              << "s\n";
                    std::cout.flush();

                    auto all_finite = [](const std::vector<double>& M) -> bool {
                        for (double x : M)
                            if (!std::isfinite(x))
                                return false;
                        return true;
                    };
                    // Thin QR orthonormalization of an n_f x cols block (col-major,
                    // lda=n_f); overwrites M with Q. Returns false on LAPACK error.
                    auto orthonormalize = [&](std::vector<double>& M,
                                              int cols) -> bool {
                        int m = n_f, ncol = cols, lda = std::max(1, n_f), info = 0,
                            lwork = -1;
                        std::vector<double> tau(static_cast<std::size_t>(cols));
                        double wq = 0.0;
                        dgeqrf_(&m, &ncol, M.data(), &lda, tau.data(), &wq, &lwork,
                                &info);
                        lwork = (info == 0) ? static_cast<int>(wq)
                                            : std::max(1, cols);
                        std::vector<double> work(
                            static_cast<std::size_t>(std::max(1, lwork)));
                        dgeqrf_(&m, &ncol, M.data(), &lda, tau.data(), work.data(),
                                &lwork, &info);
                        if (info != 0)
                            return false;
                        int kref = cols;
                        lwork = -1;
                        dorgqr_(&m, &ncol, &kref, M.data(), &lda, tau.data(), &wq,
                                &lwork, &info);
                        lwork = (info == 0) ? static_cast<int>(wq)
                                            : std::max(1, cols);
                        work.assign(static_cast<std::size_t>(std::max(1, lwork)),
                                    0.0);
                        dorgqr_(&m, &ncol, &kref, M.data(), &lda, tau.data(),
                                work.data(), &lwork, &info);
                        return info == 0;
                    };

                    // --- Inverse subspace iteration: V <- A^-1 V, orthonormalize,
                    // x10. Fixed seed 42 for a reproducible random start. ---
                    std::vector<double> V(static_cast<std::size_t>(n_f) *
                                              static_cast<std::size_t>(k_max),
                                          0.0);
                    {
                        std::mt19937_64 gen(42ULL);
                        std::uniform_real_distribution<double> dist(-1.0, 1.0);
                        for (double& x : V)
                            x = dist(gen);
                    }
                    const auto tsi0 = std::chrono::steady_clock::now();
                    bool si_ok = true;
                    {
                        std::vector<double> col(static_cast<std::size_t>(n_f));
                        for (int it = 0; si_ok && it < 10; ++it) {
                            for (int j = 0; j < k_max; ++j) {
                                for (int i = 0; i < n_f; ++i)
                                    col[static_cast<std::size_t>(i)] = V[vidx(i, j)];
                                fine_solver.solve(col.data());
                                for (int i = 0; i < n_f; ++i)
                                    V[vidx(i, j)] = col[static_cast<std::size_t>(i)];
                            }
                            if (!all_finite(V)) {
                                si_ok = false;
                                break;
                            }
                            if (!orthonormalize(V, k_max) || !all_finite(V)) {
                                si_ok = false;
                                break;
                            }
                        }
                    }
                    const double t_subspace =
                        std::chrono::duration<double>(
                            std::chrono::steady_clock::now() - tsi0)
                            .count();
                    if (!si_ok) {
                        std::cout << "  ARM 15 ABORT: inverse subspace iteration "
                                     "produced non-finite / QR failure; cannot "
                                     "build the oracle subspace.\n";
                        std::cout.flush();
                        return;
                    }
                    std::cout << "  (setup) inverse subspace iteration: 10 iters x "
                              << k_max << " solves, wall=" << t_subspace << "s\n";
                    std::cout.flush();

                    // --- AV = A V (n x k_max), reused for Ritz + deflation QRs ---
                    std::vector<double> AV(static_cast<std::size_t>(n_f) *
                                               static_cast<std::size_t>(k_max),
                                           0.0);
                    {
                        std::vector<double> vc(static_cast<std::size_t>(n_f)), avc;
                        for (int j = 0; j < k_max; ++j) {
                            for (int i = 0; i < n_f; ++i)
                                vc[static_cast<std::size_t>(i)] = V[vidx(i, j)];
                            coo_spmv(vc, avc);
                            for (int i = 0; i < n_f; ++i)
                                AV[vidx(i, j)] = avc[static_cast<std::size_t>(i)];
                        }
                    }

                    // --- Ritz diagnostics: H = V^T (A V) (k_max x k_max), dgeev ---
                    std::vector<double> H(static_cast<std::size_t>(k_max) *
                                              static_cast<std::size_t>(k_max),
                                          0.0);
                    for (int jj = 0; jj < k_max; ++jj)
                        for (int ii = 0; ii < k_max; ++ii) {
                            double s = 0.0;
                            for (int r = 0; r < n_f; ++r)
                                s += V[vidx(r, ii)] * AV[vidx(r, jj)];
                            H[static_cast<std::size_t>(ii) +
                              static_cast<std::size_t>(jj) *
                                  static_cast<std::size_t>(k_max)] = s;
                        }
                    std::vector<double> wr(static_cast<std::size_t>(k_max), 0.0),
                        wi(static_cast<std::size_t>(k_max), 0.0);
                    {
                        char jobvl = 'N', jobvr = 'N';
                        int nH = k_max, ldH = k_max, ldvl = 1, ldvr = 1, info = 0,
                            lwork = -1;
                        double wq = 0.0, vldummy = 0.0, vrdummy = 0.0;
                        dgeev_(&jobvl, &jobvr, &nH, H.data(), &ldH, wr.data(),
                               wi.data(), &vldummy, &ldvl, &vrdummy, &ldvr, &wq,
                               &lwork, &info);
                        lwork = (info == 0) ? static_cast<int>(wq)
                                            : std::max(1, 4 * k_max);
                        std::vector<double> work(
                            static_cast<std::size_t>(std::max(1, lwork)));
                        dgeev_(&jobvl, &jobvr, &nH, H.data(), &ldH, wr.data(),
                               wi.data(), &vldummy, &ldvl, &vrdummy, &ldvr,
                               work.data(), &lwork, &info);
                        if (info != 0) {
                            std::cout << "  ARM 15 WARN: dgeev info=" << info
                                      << " -- Ritz spectrum may be unreliable.\n";
                            std::cout.flush();
                        }
                    }
                    std::vector<double> ritz(static_cast<std::size_t>(k_max), 0.0);
                    for (int i = 0; i < k_max; ++i)
                        ritz[static_cast<std::size_t>(i)] = std::sqrt(
                            wr[static_cast<std::size_t>(i)] *
                                wr[static_cast<std::size_t>(i)] +
                            wi[static_cast<std::size_t>(i)] *
                                wi[static_cast<std::size_t>(i)]);
                    std::sort(ritz.begin(), ritz.end());
                    std::cout << "  (Ritz) " << k_max
                              << " |lambda| ascending (smallest modes of A):\n";
                    for (int i = 0; i < k_max; ++i) {
                        if (i % 8 == 0)
                            std::cout << "   ";
                        std::cout << ritz[static_cast<std::size_t>(i)]
                                  << ((i % 8 == 7) ? "\n" : "  ");
                    }
                    double max_gap_ratio = 1.0;
                    for (int i = 0; i < k_max - 1; ++i)
                        if (ritz[static_cast<std::size_t>(i)] > 0.0)
                            max_gap_ratio = std::max(
                                max_gap_ratio,
                                ritz[static_cast<std::size_t>(i) + 1] /
                                    ritz[static_cast<std::size_t>(i)]);
                    const bool gapless = (max_gap_ratio < 10.0);
                    std::cout << "  (Ritz) min=" << ritz[0] << " max="
                              << ritz[static_cast<std::size_t>(k_max - 1)]
                              << " max_consecutive_ratio=" << max_gap_ratio
                              << " -> " << (gapless ? "GAPLESS/continuum" : "GAP")
                              << '\n';
                    std::cout.flush();

                    // --- Deflation QRs: thin QR of A W_k for k in {16,32,64}. LS
                    // form (min_y ||r' - (A W_k) y||) cannot go singular unless
                    // A W_k drops rank; guard on min|R_ii| > 1e-12 and skip that
                    // arm if it does. ---
                    const int ks[3] = {16, 32, 64};
                    std::vector<std::vector<double>> AWqr(3), AWtau(3);
                    std::vector<char> k_ok(3, 1);
                    for (int t = 0; t < 3; ++t) {
                        const int kk = ks[t];
                        AWqr[static_cast<std::size_t>(t)].assign(
                            static_cast<std::size_t>(n_f) *
                                static_cast<std::size_t>(kk),
                            0.0);
                        for (int j = 0; j < kk; ++j)
                            for (int i = 0; i < n_f; ++i)
                                AWqr[static_cast<std::size_t>(t)][vidx(i, j)] =
                                    AV[vidx(i, j)];
                        AWtau[static_cast<std::size_t>(t)].assign(
                            static_cast<std::size_t>(kk), 0.0);
                        int m = n_f, ncol = kk, lda = std::max(1, n_f), info = 0,
                            lwork = -1;
                        double wq = 0.0;
                        dgeqrf_(&m, &ncol, AWqr[static_cast<std::size_t>(t)].data(),
                                &lda, AWtau[static_cast<std::size_t>(t)].data(), &wq,
                                &lwork, &info);
                        lwork = (info == 0) ? static_cast<int>(wq)
                                            : std::max(1, kk);
                        std::vector<double> work(
                            static_cast<std::size_t>(std::max(1, lwork)));
                        dgeqrf_(&m, &ncol, AWqr[static_cast<std::size_t>(t)].data(),
                                &lda, AWtau[static_cast<std::size_t>(t)].data(),
                                work.data(), &lwork, &info);
                        double rmin = std::numeric_limits<double>::max();
                        for (int d = 0; d < kk; ++d)
                            rmin = std::min(
                                rmin,
                                std::fabs(AWqr[static_cast<std::size_t>(t)]
                                              [vidx(d, d)]));
                        if (info != 0 || rmin <= 1e-12) {
                            k_ok[static_cast<std::size_t>(t)] = 0;
                            std::cout << "  (deflation QR k=" << kk
                                      << ") RANK-DEFICIENT (dgeqrf info=" << info
                                      << " min|R_ii|=" << rmin
                                      << ") -> arm SKIPPED\n";
                        } else {
                            std::cout << "  (deflation QR k=" << kk
                                      << ") min|R_ii|=" << rmin << " OK\n";
                        }
                    }
                    std::cout.flush();

                    // --- refined A^-1 r (2 solves + 1 refinement), shared ---
                    auto refined_solve =
                        [&](const std::vector<double>& r) -> std::vector<double> {
                        std::vector<double> e(r);
                        fine_solver.solve(e.data());
                        std::vector<double> ae;
                        coo_spmv(e, ae);
                        std::vector<double> corr(static_cast<std::size_t>(n_f));
                        for (int i = 0; i < n_f; ++i)
                            corr[static_cast<std::size_t>(i)] =
                                r[static_cast<std::size_t>(i)] -
                                ae[static_cast<std::size_t>(i)];
                        fine_solver.solve(corr.data());
                        for (int i = 0; i < n_f; ++i)
                            e[static_cast<std::size_t>(i)] +=
                                corr[static_cast<std::size_t>(i)];
                        return e;
                    };

                    struct ArmShare {
                        double global;
                        double seam;
                        double hi;
                        double solve_rel;
                    };
                    auto attrib_share =
                        [&](const std::vector<double>& xarm) -> ArmShare {
                        std::vector<double> ax;
                        coo_spmv(xarm, ax);
                        std::vector<double> r(static_cast<std::size_t>(n_f));
                        for (int i = 0; i < n_f; ++i)
                            r[static_cast<std::size_t>(i)] =
                                b[static_cast<std::size_t>(i)] -
                                ax[static_cast<std::size_t>(i)];
                        std::vector<double> e = refined_solve(r);
                        std::vector<double> ae;
                        coo_spmv(e, ae);
                        double num = 0.0, den = 0.0, tot = 0.0, gl = 0.0, sm = 0.0,
                               hp = 0.0;
                        for (int i = 0; i < n_f; ++i) {
                            const double d = ae[static_cast<std::size_t>(i)] -
                                             r[static_cast<std::size_t>(i)];
                            num += d * d;
                            den += r[static_cast<std::size_t>(i)] *
                                   r[static_cast<std::size_t>(i)];
                            const double e2 = e[static_cast<std::size_t>(i)] *
                                              e[static_cast<std::size_t>(i)];
                            tot += e2;
                            if (is_global[static_cast<std::size_t>(i)])
                                gl += e2;
                            if (col_seam[static_cast<std::size_t>(i)])
                                sm += e2;
                            if (fine_col_matched[static_cast<std::size_t>(i)] == 0)
                                hp += e2;
                        }
                        const double inv = (tot > 0.0) ? 1.0 / tot : 0.0;
                        return {gl * inv, sm * inv, hp * inv,
                                (den > 0.0) ? std::sqrt(num / den) : 0.0};
                    };

                    // --- LS deflation apply: z = M_arm8(r) + W_k y, where
                    // y = argmin ||r' - (A W_k) y||, r' = r - A M_arm8(r), solved
                    // through the stored thin QR of A W_k. Linear, deterministic,
                    // local buffers only. ---
                    auto apply_deflation = [&](int t, const std::vector<double>& r,
                                               std::vector<double>& z) {
                        const int kk = ks[t];
                        std::vector<double> z1, az1;
                        pc_arm8(r, z1);
                        coo_spmv(z1, az1);
                        std::vector<double> c(static_cast<std::size_t>(n_f));
                        for (int i = 0; i < n_f; ++i)
                            c[static_cast<std::size_t>(i)] =
                                r[static_cast<std::size_t>(i)] -
                                az1[static_cast<std::size_t>(i)];
                        char side = 'L', trans = 'T';
                        int m = n_f, ncol = 1, kref = kk, lda = std::max(1, n_f),
                            ldc = std::max(1, n_f), info = 0, lwork = -1;
                        double wq = 0.0;
                        dormqr_(&side, &trans, &m, &ncol, &kref,
                                AWqr[static_cast<std::size_t>(t)].data(), &lda,
                                AWtau[static_cast<std::size_t>(t)].data(), c.data(),
                                &ldc, &wq, &lwork, &info);
                        lwork = (info == 0) ? static_cast<int>(wq)
                                            : std::max(1, n_f);
                        std::vector<double> work(
                            static_cast<std::size_t>(std::max(1, lwork)));
                        dormqr_(&side, &trans, &m, &ncol, &kref,
                                AWqr[static_cast<std::size_t>(t)].data(), &lda,
                                AWtau[static_cast<std::size_t>(t)].data(), c.data(),
                                &ldc, work.data(), &lwork, &info);
                        std::vector<double> y(c.begin(), c.begin() + kk);
                        char uplo = 'U', tn = 'N', diag = 'N';
                        int rn = kk, rnrhs = 1, rlda = std::max(1, n_f), rldb = kk,
                            rinfo = 0;
                        dtrtrs_(&uplo, &tn, &diag, &rn, &rnrhs,
                                AWqr[static_cast<std::size_t>(t)].data(), &rlda,
                                y.data(), &rldb, &rinfo);
                        z.assign(static_cast<std::size_t>(n_f), 0.0);
                        for (int i = 0; i < n_f; ++i)
                            z[static_cast<std::size_t>(i)] =
                                z1[static_cast<std::size_t>(i)];
                        for (int j = 0; j < kk; ++j) {
                            const double yj = y[static_cast<std::size_t>(j)];
                            for (int i = 0; i < n_f; ++i)
                                z[static_cast<std::size_t>(i)] += yj * V[vidx(i, j)];
                        }
                    };
                    KrylovOperator pc_15a = [&](const std::vector<double>& r,
                                                std::vector<double>& z) {
                        apply_deflation(0, r, z);
                    };
                    KrylovOperator pc_15b = [&](const std::vector<double>& r,
                                                std::vector<double>& z) {
                        apply_deflation(1, r, z);
                    };
                    KrylovOperator pc_15c = [&](const std::vector<double>& r,
                                                std::vector<double>& z) {
                        apply_deflation(2, r, z);
                    };

                    // --- Run: 8r2 baseline first (guard) ---
                    const std::size_t a15_first = arms.size();
                    run_arm("8r2_baseline", pc_arm8);

                    // --- Projection diagnostic: f_k = ||W_k^T e8||^2 / ||e8||^2,
                    // e8 = refined A^-1 r8, r8 = b - A x8 (8r2 solution). V is
                    // orthonormal so P_{W_k} = W_k W_k^T. ---
                    double f8 = 0.0, f16 = 0.0, f32 = 0.0, f64 = 0.0;
                    {
                        const std::vector<double>& x8 = arms[a15_first].x;
                        std::vector<double> ax8;
                        coo_spmv(x8, ax8);
                        std::vector<double> r8(static_cast<std::size_t>(n_f));
                        for (int i = 0; i < n_f; ++i)
                            r8[static_cast<std::size_t>(i)] =
                                b[static_cast<std::size_t>(i)] -
                                ax8[static_cast<std::size_t>(i)];
                        std::vector<double> e8 = refined_solve(r8);
                        double e8n2 = 0.0;
                        for (int i = 0; i < n_f; ++i)
                            e8n2 += e8[static_cast<std::size_t>(i)] *
                                    e8[static_cast<std::size_t>(i)];
                        std::vector<double> cdot(static_cast<std::size_t>(k_max),
                                                 0.0);
                        for (int j = 0; j < k_max; ++j) {
                            double s = 0.0;
                            for (int i = 0; i < n_f; ++i)
                                s += V[vidx(i, j)] * e8[static_cast<std::size_t>(i)];
                            cdot[static_cast<std::size_t>(j)] = s;
                        }
                        auto fk = [&](int k) -> double {
                            double s = 0.0;
                            for (int j = 0; j < k; ++j)
                                s += cdot[static_cast<std::size_t>(j)] *
                                     cdot[static_cast<std::size_t>(j)];
                            return (e8n2 > 0.0) ? s / e8n2 : 0.0;
                        };
                        f8 = fk(8);
                        f16 = fk(16);
                        f32 = fk(32);
                        f64 = fk(64);
                        std::cout << "  (projection) ||e8||^2=" << e8n2
                                  << "  f_8=" << f8 << "  f_16=" << f16
                                  << "  f_32=" << f32 << "  f_64=" << f64 << '\n';
                        std::cout.flush();
                    }

                    // --- Run: 15a/15b/15c (skip rank-deficient k) ---
                    if (k_ok[0])
                        run_arm("15a_deflate_k16", pc_15a);
                    else
                        std::cout << "  15a_deflate_k16 SKIPPED (rank-deficient "
                                     "A W_16)\n";
                    if (k_ok[1])
                        run_arm("15b_deflate_k32", pc_15b);
                    else
                        std::cout << "  15b_deflate_k32 SKIPPED (rank-deficient "
                                     "A W_32)\n";
                    if (k_ok[2])
                        run_arm("15c_deflate_k64", pc_15c);
                    else
                        std::cout << "  15c_deflate_k64 SKIPPED (rank-deficient "
                                     "A W_64)\n";

                    // --- Results table + rubric ---
                    std::cout << "\n=== ARM 15 RESULTS (canonical current-ladder "
                                 "fixture) ===\n";
                    std::cout << "  arm | rel@300 | rel@600 | ord[0,300) | "
                                 "ord[300,600) | non_decel | e:global | e:seam | "
                                 "e:high-p | solve_rel | wall/iter(s)\n";
                    auto rel_at = [&](const ArmResult& a, int k) -> double {
                        if (a.curve.empty())
                            return 0.0;
                        const int idx =
                            std::min(k, static_cast<int>(a.curve.size())) - 1;
                        return (b_norm > 0.0)
                                   ? a.curve[static_cast<std::size_t>(idx)] / b_norm
                                   : a.curve[static_cast<std::size_t>(idx)];
                    };
                    struct ArmEval {
                        std::string name;
                        double r600;
                        bool non_decel;
                        ArmShare sh;
                    };
                    std::vector<ArmEval> evals;
                    for (std::size_t ai = a15_first; ai < arms.size(); ++ai) {
                        const ArmResult& a = arms[ai];
                        const double r300 = rel_at(a, 300);
                        const double r600 = rel_at(a, 600);
                        const double ord_a =
                            (r300 > 0.0) ? std::log10(1.0 / r300) : 0.0;
                        const double ord_b =
                            (r300 > 0.0 && r600 > 0.0) ? std::log10(r300 / r600)
                                                       : 0.0;
                        const ArmShare sh = attrib_share(a.x);
                        const bool non_decel = (ord_b >= ord_a - 1e-12);
                        const double wall_per_iter =
                            (a.iters > 0) ? a.wall / a.iters : 0.0;
                        std::cout << "  " << a.name << " | " << r300 << " | " << r600
                                  << " | " << ord_a << " | " << ord_b << " | "
                                  << (non_decel ? "YES" : "no") << " | " << sh.global
                                  << " | " << sh.seam << " | " << sh.hi << " | "
                                  << sh.solve_rel << " | " << wall_per_iter << '\n';
                        evals.push_back({a.name, r600, non_decel, sh});
                    }
                    std::cout.flush();

                    // --- Per-arm verdict + overall rubric branch ---
                    bool any_pass = false;
                    for (const ArmEval& ev : evals) {
                        if (ev.name.rfind("15", 0) != 0)
                            continue; // only deflation arms are candidates
                        const bool coll = (ev.sh.global < 0.1);
                        std::string v;
                        if (ev.non_decel && ev.r600 <= 1e-4 && coll) {
                            v = "PASS";
                            any_pass = true;
                        } else if (ev.sh.global > 0.5)
                            v = "FAIL: channel not reached (global share >0.5)";
                        else
                            v = "PARTIAL/NO-PASS";
                        std::cout << "  VERDICT " << ev.name << ": " << v
                                  << "  (rel@600=" << ev.r600
                                  << " non_decel=" << (ev.non_decel ? 1 : 0)
                                  << " e:global=" << ev.sh.global << ")\n";
                    }
                    std::cout << "  RUBRIC BRANCH: ";
                    if (any_pass)
                        std::cout << "PASS -- oracle ceiling proven; a production "
                                     "deflation (row-partitioned border / cheap "
                                     "mode approximation) becomes fundable.";
                    else if (f64 >= 0.9)
                        std::cout << "FAIL with f_64>=0.9 -- subspace captures the "
                                     "error but arms do not converge; implementation "
                                     "or composition suspect (report, do not "
                                     "improvise).";
                    else if (f64 < 0.5 && gapless)
                        std::cout << "FAIL with f_64<0.5 + gapless Ritz -- blocking "
                                     "error is high-dimensional; deflation class "
                                     "DEAD; recommend park at E-W grade.";
                    else
                        std::cout << "INCONCLUSIVE (f_64=" << f64
                                  << " gapless=" << (gapless ? 1 : 0)
                                  << " any_pass=" << (any_pass ? 1 : 0) << ").";
                    std::cout << "\n  (rubric inputs: f_8=" << f8 << " f_16=" << f16
                              << " f_32=" << f32 << " f_64=" << f64
                              << " gapless=" << (gapless ? 1 : 0)
                              << " max_gap_ratio=" << max_gap_ratio << ")\n";
                    std::cout << "############ ARM 15 END ############\n";
                    std::cout.flush();
                    return;
                }

                // ====================================================================
                // ARM 15d/e: small-k Galerkin + one-shot deflation
                // (PCOARSE_ARM15D). Arm-15 showed the 8r2 blocking error is
                // LOW-dimensional (f_8=0.99) but LS deflation of k>=16 raw modes
                // DEGRADED GMRES monotonically. Diagnosis: (a) the near-null cluster
                // is the near-degenerate smallest PAIR; k>=16 arms dragged in
                // continuum modes that 10 subspace iterations cannot make invariant,
                // turning the correction into noise; (b) the 1/lambda-amplified LS
                // correction re-applied inside the PC may poison the Krylov operator
                // regardless. This arm isolates both: it builds the SORTED smallest
                // modes (dgeev eigenvectors of H, ordered by |theta|, orthonormalized
                // -- NOT raw V columns, which are an arbitrary slice), runs k=2 and
                // k=8 with LS and Galerkin (z1 + W (W^T A W)^-1 W^T r') in-PC, and a
                // ONE-SHOT post-loop Galerkin cleanup (15e) that never touches the
                // Krylov operator. New diagnostics: per-mode invariance residual
                // (pair-invariant vs continuum-junk) and ||y|| of the first PC
                // applies (amplification signature). Self-contained; factors the
                // fine solver first (reused). Returns before the Stage-A flow.
                // ====================================================================
                if (env_flag_enabled("PCOARSE_ARM15D", false)) {
                    std::cout << "\n############ ARM 15d/e: small-k Galerkin + "
                                 "one-shot deflation ############\n";
                    std::cout.flush();

                    const int k_max = 64;
                    const std::size_t size_n = static_cast<std::size_t>(n_f);
                    auto vidx = [&](int i, int j) -> std::size_t {
                        return static_cast<std::size_t>(i) +
                               static_cast<std::size_t>(j) * size_n;
                    };

                    std::vector<char> is_global(size_n, 0);
                    for (int cc = 0; cc < n_f; ++cc) {
                        const ColumnInfo& ci = cmap[static_cast<std::size_t>(cc)];
                        if (ci.is_var_domain || ci.var_double_idx >= 0 ||
                            ci.domain < 0)
                            is_global[static_cast<std::size_t>(cc)] = 1;
                    }
                    std::vector<char> col_seam(size_n, 0);
                    for (long long e = 0; e < nnz_f; ++e) {
                        const std::size_t ee = static_cast<std::size_t>(e);
                        const int rr = coo.irn[ee] - 1;
                        if (rr < static_cast<int>(tmeta.size()) &&
                            tmeta[static_cast<std::size_t>(rr)].taxonomy ==
                                RowTaxonomy::TauMatch)
                            col_seam[static_cast<std::size_t>(coo.jcn[ee] - 1)] = 1;
                    }

                    // --- Fine factor FIRST (reused for subspace iter + attrib) ---
                    const auto tfac0 = std::chrono::steady_clock::now();
                    MumpsLinearSolver fine_solver(
                        n_f, config.mumps.ordering, false, 0,
                        mumps_runtime_state.icntl14, MPI_COMM_SELF);
                    fine_solver.set_pattern(n_f, nnz_f, coo.irn.data(),
                                            coo.jcn.data());
                    fine_solver.analyze_pattern();
                    fine_solver.factor_analyzed(coo.a.data());
                    mumps_runtime_state.icntl14 = fine_solver.last_icntl14();
                    std::cout << "  (setup) fine MUMPS factor wall="
                              << std::chrono::duration<double>(
                                     std::chrono::steady_clock::now() - tfac0)
                                     .count()
                              << "s\n";
                    std::cout.flush();

                    auto all_finite = [](const std::vector<double>& M) -> bool {
                        for (double x : M)
                            if (!std::isfinite(x))
                                return false;
                        return true;
                    };
                    auto orthonormalize = [&](std::vector<double>& M,
                                              int cols) -> bool {
                        int m = n_f, ncol = cols, lda = std::max(1, n_f), info = 0,
                            lwork = -1;
                        std::vector<double> tau(static_cast<std::size_t>(cols));
                        double wq = 0.0;
                        dgeqrf_(&m, &ncol, M.data(), &lda, tau.data(), &wq, &lwork,
                                &info);
                        lwork = (info == 0) ? static_cast<int>(wq)
                                            : std::max(1, cols);
                        std::vector<double> work(
                            static_cast<std::size_t>(std::max(1, lwork)));
                        dgeqrf_(&m, &ncol, M.data(), &lda, tau.data(), work.data(),
                                &lwork, &info);
                        if (info != 0)
                            return false;
                        int kref = cols;
                        lwork = -1;
                        dorgqr_(&m, &ncol, &kref, M.data(), &lda, tau.data(), &wq,
                                &lwork, &info);
                        lwork = (info == 0) ? static_cast<int>(wq)
                                            : std::max(1, cols);
                        work.assign(static_cast<std::size_t>(std::max(1, lwork)),
                                    0.0);
                        dorgqr_(&m, &ncol, &kref, M.data(), &lda, tau.data(),
                                work.data(), &lwork, &info);
                        return info == 0;
                    };

                    // --- Inverse subspace iteration -> V (identical to arm-15) ---
                    std::vector<double> V(size_n * static_cast<std::size_t>(k_max),
                                          0.0);
                    {
                        std::mt19937_64 gen(42ULL);
                        std::uniform_real_distribution<double> dist(-1.0, 1.0);
                        for (double& x : V)
                            x = dist(gen);
                    }
                    const auto tsi0 = std::chrono::steady_clock::now();
                    bool si_ok = true;
                    {
                        std::vector<double> col(size_n);
                        for (int it = 0; si_ok && it < 10; ++it) {
                            for (int j = 0; j < k_max; ++j) {
                                for (int i = 0; i < n_f; ++i)
                                    col[static_cast<std::size_t>(i)] = V[vidx(i, j)];
                                fine_solver.solve(col.data());
                                for (int i = 0; i < n_f; ++i)
                                    V[vidx(i, j)] = col[static_cast<std::size_t>(i)];
                            }
                            if (!all_finite(V) || !orthonormalize(V, k_max) ||
                                !all_finite(V)) {
                                si_ok = false;
                                break;
                            }
                        }
                    }
                    if (!si_ok) {
                        std::cout << "  ARM 15D ABORT: subspace iteration non-finite "
                                     "/ QR failure.\n";
                        std::cout.flush();
                        return;
                    }
                    std::cout << "  (setup) inverse subspace iteration wall="
                              << std::chrono::duration<double>(
                                     std::chrono::steady_clock::now() - tsi0)
                                     .count()
                              << "s\n";
                    std::cout.flush();

                    // --- AV = A V ; H = V^T A V (kept; dgeev works on a copy) ---
                    std::vector<double> AV(size_n * static_cast<std::size_t>(k_max),
                                           0.0);
                    {
                        std::vector<double> vc(size_n), avc;
                        for (int j = 0; j < k_max; ++j) {
                            for (int i = 0; i < n_f; ++i)
                                vc[static_cast<std::size_t>(i)] = V[vidx(i, j)];
                            coo_spmv(vc, avc);
                            for (int i = 0; i < n_f; ++i)
                                AV[vidx(i, j)] = avc[static_cast<std::size_t>(i)];
                        }
                    }
                    std::vector<double> Hc(static_cast<std::size_t>(k_max) *
                                               static_cast<std::size_t>(k_max),
                                           0.0);
                    for (int jj = 0; jj < k_max; ++jj)
                        for (int ii = 0; ii < k_max; ++ii) {
                            double s = 0.0;
                            for (int r = 0; r < n_f; ++r)
                                s += V[vidx(r, ii)] * AV[vidx(r, jj)];
                            Hc[static_cast<std::size_t>(ii) +
                               static_cast<std::size_t>(jj) *
                                   static_cast<std::size_t>(k_max)] = s;
                        }

                    // --- dgeev with right eigenvectors -> Ritz values + vectors ---
                    std::vector<double> wr(static_cast<std::size_t>(k_max), 0.0),
                        wi(static_cast<std::size_t>(k_max), 0.0),
                        VR(static_cast<std::size_t>(k_max) *
                               static_cast<std::size_t>(k_max),
                           0.0);
                    {
                        char jobvl = 'N', jobvr = 'V';
                        int nH = k_max, ldH = k_max, ldvl = 1, ldvr = k_max,
                            info = 0, lwork = -1;
                        double wq = 0.0, vldummy = 0.0;
                        dgeev_(&jobvl, &jobvr, &nH, Hc.data(), &ldH, wr.data(),
                               wi.data(), &vldummy, &ldvl, VR.data(), &ldvr, &wq,
                               &lwork, &info);
                        lwork = (info == 0) ? static_cast<int>(wq)
                                            : std::max(1, 4 * k_max);
                        std::vector<double> work(
                            static_cast<std::size_t>(std::max(1, lwork)));
                        dgeev_(&jobvl, &jobvr, &nH, Hc.data(), &ldH, wr.data(),
                               wi.data(), &vldummy, &ldvl, VR.data(), &ldvr,
                               work.data(), &lwork, &info);
                        if (info != 0) {
                            std::cout << "  ARM 15D ABORT: dgeev info=" << info
                                      << ".\n";
                            std::cout.flush();
                            return;
                        }
                    }
                    std::vector<double> ritz(static_cast<std::size_t>(k_max), 0.0);
                    for (int i = 0; i < k_max; ++i)
                        ritz[static_cast<std::size_t>(i)] = std::sqrt(
                            wr[static_cast<std::size_t>(i)] *
                                wr[static_cast<std::size_t>(i)] +
                            wi[static_cast<std::size_t>(i)] *
                                wi[static_cast<std::size_t>(i)]);
                    {
                        std::vector<double> rs(ritz);
                        std::sort(rs.begin(), rs.end());
                        double mgr = 1.0;
                        for (int i = 0; i < k_max - 1; ++i)
                            if (rs[static_cast<std::size_t>(i)] > 0.0)
                                mgr = std::max(mgr,
                                               rs[static_cast<std::size_t>(i) + 1] /
                                                   rs[static_cast<std::size_t>(i)]);
                        std::cout << "  (Ritz) min=" << rs[0] << " max="
                                  << rs[static_cast<std::size_t>(k_max - 1)]
                                  << " max_consecutive_ratio=" << mgr << " -> "
                                  << (mgr < 10.0 ? "GAPLESS/continuum" : "GAP")
                                  << '\n';
                        std::cout.flush();
                    }

                    // --- Sort eigen-blocks by |theta|; build sorted real basis
                    // Gsort (64x64), then Wbig = V Gsort and AWbig = AV Gsort
                    // (A W for free). Complex conjugate pairs contribute their
                    // real+imag parts as two consecutive columns. ---
                    struct EigBlk {
                        double mag;
                        int j;
                        bool cplx;
                    };
                    std::vector<EigBlk> blks;
                    for (int j = 0; j < k_max;) {
                        if (wi[static_cast<std::size_t>(j)] == 0.0) {
                            blks.push_back(
                                {std::fabs(wr[static_cast<std::size_t>(j)]), j,
                                 false});
                            j += 1;
                        } else {
                            blks.push_back({ritz[static_cast<std::size_t>(j)], j,
                                            true});
                            j += 2;
                        }
                    }
                    std::sort(blks.begin(), blks.end(),
                              [](const EigBlk& a, const EigBlk& c) {
                                  return a.mag < c.mag;
                              });
                    std::vector<double> Gsort(static_cast<std::size_t>(k_max) *
                                                  static_cast<std::size_t>(k_max),
                                              0.0);
                    std::vector<int> blk_col(blks.size(), 0);
                    int off = 0;
                    for (std::size_t bi = 0; bi < blks.size() && off < k_max;
                         ++bi) {
                        blk_col[bi] = off;
                        const int j = blks[bi].j;
                        for (int i = 0; i < k_max; ++i)
                            Gsort[static_cast<std::size_t>(i) +
                                  static_cast<std::size_t>(off) *
                                      static_cast<std::size_t>(k_max)] =
                                VR[static_cast<std::size_t>(i) +
                                   static_cast<std::size_t>(j) *
                                       static_cast<std::size_t>(k_max)];
                        off += 1;
                        if (blks[bi].cplx && off < k_max) {
                            for (int i = 0; i < k_max; ++i)
                                Gsort[static_cast<std::size_t>(i) +
                                      static_cast<std::size_t>(off) *
                                          static_cast<std::size_t>(k_max)] =
                                    VR[static_cast<std::size_t>(i) +
                                       static_cast<std::size_t>(j + 1) *
                                           static_cast<std::size_t>(k_max)];
                            off += 1;
                        }
                    }
                    std::vector<double> Wbig(size_n *
                                                 static_cast<std::size_t>(k_max),
                                             0.0),
                        AWbig(size_n * static_cast<std::size_t>(k_max), 0.0);
                    for (int c = 0; c < k_max; ++c)
                        for (int row = 0; row < n_f; ++row) {
                            double sw = 0.0, sa = 0.0;
                            for (int i = 0; i < k_max; ++i) {
                                const double g =
                                    Gsort[static_cast<std::size_t>(i) +
                                          static_cast<std::size_t>(c) *
                                              static_cast<std::size_t>(k_max)];
                                sw += V[vidx(row, i)] * g;
                                sa += AV[vidx(row, i)] * g;
                            }
                            Wbig[vidx(row, c)] = sw;
                            AWbig[vidx(row, c)] = sa;
                        }

                    // --- Diagnostic (i): per-mode invariance residual for the 16
                    // smallest sorted modes: ||A y - theta y|| / (|theta| ||y||).
                    // Small => that mode is (nearly) invariant under 10 subspace
                    // iterations; large => continuum junk. ---
                    std::cout << "  (invariance) i | |theta| | "
                                 "||A y - theta y||/(|theta| ||y||):\n";
                    const int n_inv =
                        std::min(16, static_cast<int>(blks.size()));
                    for (int bi = 0; bi < n_inv; ++bi) {
                        const int c = blk_col[static_cast<std::size_t>(bi)];
                        const double mag = blks[static_cast<std::size_t>(bi)].mag;
                        double resn = 0.0, yn = 0.0;
                        if (!blks[static_cast<std::size_t>(bi)].cplx) {
                            const double th = wr[static_cast<std::size_t>(
                                blks[static_cast<std::size_t>(bi)].j)];
                            for (int row = 0; row < n_f; ++row) {
                                const double y = Wbig[vidx(row, c)];
                                const double res = AWbig[vidx(row, c)] - th * y;
                                resn += res * res;
                                yn += y * y;
                            }
                        } else {
                            const std::size_t jj = static_cast<std::size_t>(
                                blks[static_cast<std::size_t>(bi)].j);
                            const double a = wr[jj], bb = wi[jj];
                            for (int row = 0; row < n_f; ++row) {
                                const double yr = Wbig[vidx(row, c)];
                                const double yi = Wbig[vidx(row, c + 1)];
                                const double rr =
                                    AWbig[vidx(row, c)] - (a * yr - bb * yi);
                                const double ri =
                                    AWbig[vidx(row, c + 1)] - (a * yi + bb * yr);
                                resn += rr * rr + ri * ri;
                                yn += yr * yr + yi * yi;
                            }
                        }
                        resn = std::sqrt(resn);
                        yn = std::sqrt(yn);
                        const double rel_inv =
                            (mag * yn > 0.0) ? resn / (mag * yn) : 0.0;
                        std::cout << "    i=" << (bi + 1) << " |theta|=" << mag
                                  << " resid=" << rel_inv
                                  << (blks[static_cast<std::size_t>(bi)].cplx
                                          ? " (cplx pair)"
                                          : "")
                                  << '\n';
                    }
                    std::cout.flush();

                    // --- Orthonormalize the smallest-8 sorted-Ritz block; propagate
                    // to A W via AW := AW R^-1 so Worth is orthonormal and
                    // AWorth = A Worth exactly. ---
                    const int KORTH = 8;
                    std::vector<double> Worth(size_n *
                                              static_cast<std::size_t>(KORTH)),
                        AWorth(size_n * static_cast<std::size_t>(KORTH));
                    for (int c = 0; c < KORTH; ++c)
                        for (int row = 0; row < n_f; ++row) {
                            Worth[vidx(row, c)] = Wbig[vidx(row, c)];
                            AWorth[vidx(row, c)] = AWbig[vidx(row, c)];
                        }
                    auto ortho_propagate = [&](std::vector<double>& Wk,
                                               std::vector<double>& AWk,
                                               int kk) -> bool {
                        int m = n_f, ncol = kk, lda = std::max(1, n_f), info = 0,
                            lwork = -1;
                        std::vector<double> tau(static_cast<std::size_t>(kk));
                        double wq = 0.0;
                        dgeqrf_(&m, &ncol, Wk.data(), &lda, tau.data(), &wq, &lwork,
                                &info);
                        lwork = (info == 0) ? static_cast<int>(wq)
                                            : std::max(1, kk);
                        std::vector<double> work(
                            static_cast<std::size_t>(std::max(1, lwork)));
                        dgeqrf_(&m, &ncol, Wk.data(), &lda, tau.data(), work.data(),
                                &lwork, &info);
                        if (info != 0)
                            return false;
                        std::vector<double> R(static_cast<std::size_t>(kk) *
                                                  static_cast<std::size_t>(kk),
                                              0.0);
                        for (int j = 0; j < kk; ++j)
                            for (int i = 0; i <= j; ++i)
                                R[static_cast<std::size_t>(i) +
                                  static_cast<std::size_t>(j) *
                                      static_cast<std::size_t>(kk)] =
                                    Wk[vidx(i, j)];
                        for (int d = 0; d < kk; ++d)
                            if (std::fabs(R[static_cast<std::size_t>(d) +
                                            static_cast<std::size_t>(d) *
                                                static_cast<std::size_t>(kk)]) <=
                                1e-12)
                                return false;
                        int kref = kk;
                        lwork = -1;
                        dorgqr_(&m, &ncol, &kref, Wk.data(), &lda, tau.data(), &wq,
                                &lwork, &info);
                        lwork = (info == 0) ? static_cast<int>(wq)
                                            : std::max(1, kk);
                        work.assign(static_cast<std::size_t>(std::max(1, lwork)),
                                    0.0);
                        dorgqr_(&m, &ncol, &kref, Wk.data(), &lda, tau.data(),
                                work.data(), &lwork, &info);
                        if (info != 0)
                            return false;
                        char side = 'R', uplo = 'U', transa = 'N', diag = 'N';
                        double alpha = 1.0;
                        int dm = n_f, dn = kk, dlda = std::max(1, kk),
                            dldb = std::max(1, n_f);
                        dtrsm_(&side, &uplo, &transa, &diag, &dm, &dn, &alpha,
                               R.data(), &dlda, AWk.data(), &dldb);
                        return true;
                    };
                    if (!ortho_propagate(Worth, AWorth, KORTH)) {
                        std::cout << "  ARM 15D ABORT: sorted-Ritz orthonormalization "
                                     "failed (smallest-8 block rank-deficient).\n";
                        std::cout.flush();
                        return;
                    }

                    // --- Per-k LS QR (of A Worth_k) and Galerkin factor
                    // (Worth_k^T A Worth_k) for k in {2,8}. ---
                    const int kd[2] = {2, 8};
                    std::vector<std::vector<double>> AWls(2), LStau(2);
                    std::vector<char> LSok(2, 1);
                    std::vector<std::vector<double>> Glu(2);
                    std::vector<std::vector<int>> Gipiv(2);
                    std::vector<char> Galok(2, 1);
                    for (int t = 0; t < 2; ++t) {
                        const int kk = kd[t];
                        // LS QR of A Worth_k (n x kk).
                        AWls[static_cast<std::size_t>(t)].assign(
                            size_n * static_cast<std::size_t>(kk), 0.0);
                        for (int c = 0; c < kk; ++c)
                            for (int row = 0; row < n_f; ++row)
                                AWls[static_cast<std::size_t>(t)][vidx(row, c)] =
                                    AWorth[vidx(row, c)];
                        LStau[static_cast<std::size_t>(t)].assign(
                            static_cast<std::size_t>(kk), 0.0);
                        {
                            int m = n_f, ncol = kk, lda = std::max(1, n_f),
                                info = 0, lwork = -1;
                            double wq = 0.0;
                            dgeqrf_(&m, &ncol,
                                    AWls[static_cast<std::size_t>(t)].data(), &lda,
                                    LStau[static_cast<std::size_t>(t)].data(), &wq,
                                    &lwork, &info);
                            lwork = (info == 0) ? static_cast<int>(wq)
                                                : std::max(1, kk);
                            std::vector<double> work(
                                static_cast<std::size_t>(std::max(1, lwork)));
                            dgeqrf_(&m, &ncol,
                                    AWls[static_cast<std::size_t>(t)].data(), &lda,
                                    LStau[static_cast<std::size_t>(t)].data(),
                                    work.data(), &lwork, &info);
                            double rmin = std::numeric_limits<double>::max();
                            for (int d = 0; d < kk; ++d)
                                rmin = std::min(
                                    rmin,
                                    std::fabs(AWls[static_cast<std::size_t>(t)]
                                                  [vidx(d, d)]));
                            if (info != 0 || rmin <= 1e-12) {
                                LSok[static_cast<std::size_t>(t)] = 0;
                                std::cout << "  (15d LS QR k=" << kk
                                          << ") RANK-DEFICIENT min|R_ii|=" << rmin
                                          << " -> arm SKIPPED\n";
                            } else
                                std::cout << "  (15d LS QR k=" << kk
                                          << ") min|R_ii|=" << rmin << " OK\n";
                        }
                        // Galerkin block G = Worth_k^T A Worth_k (kk x kk).
                        Glu[static_cast<std::size_t>(t)].assign(
                            static_cast<std::size_t>(kk) *
                                static_cast<std::size_t>(kk),
                            0.0);
                        for (int jc = 0; jc < kk; ++jc)
                            for (int ic = 0; ic < kk; ++ic) {
                                double s = 0.0;
                                for (int row = 0; row < n_f; ++row)
                                    s += Worth[vidx(row, ic)] * AWorth[vidx(row, jc)];
                                Glu[static_cast<std::size_t>(t)]
                                   [static_cast<std::size_t>(ic) +
                                    static_cast<std::size_t>(jc) *
                                        static_cast<std::size_t>(kk)] = s;
                            }
                        char nrm = '1';
                        int gm = kk, gn = kk, glda = std::max(1, kk), info = 0;
                        std::vector<double> lwk(static_cast<std::size_t>(kk));
                        double anorm = dlange_(&nrm, &gm, &gn,
                                               Glu[static_cast<std::size_t>(t)]
                                                   .data(),
                                               &glda, lwk.data());
                        Gipiv[static_cast<std::size_t>(t)].assign(
                            static_cast<std::size_t>(kk), 0);
                        dgetrf_(&gm, &gn,
                                Glu[static_cast<std::size_t>(t)].data(), &glda,
                                Gipiv[static_cast<std::size_t>(t)].data(), &info);
                        double rcond = 0.0;
                        int cinfo = 0;
                        std::vector<double> cwk(static_cast<std::size_t>(4 * kk));
                        std::vector<int> ciwk(static_cast<std::size_t>(kk));
                        if (info == 0)
                            dgecon_(&nrm, &gn,
                                    Glu[static_cast<std::size_t>(t)].data(), &glda,
                                    &anorm, &rcond, cwk.data(), ciwk.data(),
                                    &cinfo);
                        if (info != 0 || rcond < 1e-8) {
                            Galok[static_cast<std::size_t>(t)] = 0;
                            std::cout << "  (15d Galerkin k=" << kk
                                      << ") info=" << info << " rcond=" << rcond
                                      << " -> arm SKIPPED\n";
                        } else
                            std::cout << "  (15d Galerkin k=" << kk
                                      << ") rcond=" << rcond << " OK\n";
                    }
                    std::cout.flush();

                    auto refined_solve =
                        [&](const std::vector<double>& r) -> std::vector<double> {
                        std::vector<double> e(r);
                        fine_solver.solve(e.data());
                        std::vector<double> ae;
                        coo_spmv(e, ae);
                        std::vector<double> corr(size_n);
                        for (int i = 0; i < n_f; ++i)
                            corr[static_cast<std::size_t>(i)] =
                                r[static_cast<std::size_t>(i)] -
                                ae[static_cast<std::size_t>(i)];
                        fine_solver.solve(corr.data());
                        for (int i = 0; i < n_f; ++i)
                            e[static_cast<std::size_t>(i)] +=
                                corr[static_cast<std::size_t>(i)];
                        return e;
                    };
                    struct ArmShare {
                        double global;
                        double seam;
                        double hi;
                        double solve_rel;
                    };
                    auto attrib_share =
                        [&](const std::vector<double>& xarm) -> ArmShare {
                        std::vector<double> ax;
                        coo_spmv(xarm, ax);
                        std::vector<double> r(size_n);
                        for (int i = 0; i < n_f; ++i)
                            r[static_cast<std::size_t>(i)] =
                                b[static_cast<std::size_t>(i)] -
                                ax[static_cast<std::size_t>(i)];
                        std::vector<double> e = refined_solve(r);
                        std::vector<double> ae;
                        coo_spmv(e, ae);
                        double num = 0.0, den = 0.0, tot = 0.0, gl = 0.0, sm = 0.0,
                               hp = 0.0;
                        for (int i = 0; i < n_f; ++i) {
                            const double d = ae[static_cast<std::size_t>(i)] -
                                             r[static_cast<std::size_t>(i)];
                            num += d * d;
                            den += r[static_cast<std::size_t>(i)] *
                                   r[static_cast<std::size_t>(i)];
                            const double e2 = e[static_cast<std::size_t>(i)] *
                                              e[static_cast<std::size_t>(i)];
                            tot += e2;
                            if (is_global[static_cast<std::size_t>(i)])
                                gl += e2;
                            if (col_seam[static_cast<std::size_t>(i)])
                                sm += e2;
                            if (fine_col_matched[static_cast<std::size_t>(i)] == 0)
                                hp += e2;
                        }
                        const double inv = (tot > 0.0) ? 1.0 / tot : 0.0;
                        return {gl * inv, sm * inv, hp * inv,
                                (den > 0.0) ? std::sqrt(num / den) : 0.0};
                    };

                    // --- In-PC applies (LS + Galerkin), with ||y|| print on the
                    // first 3 calls per arm (amplification signature). ---
                    auto apply_ls = [&](int t, const std::string& label, int& ncall,
                                        const std::vector<double>& r,
                                        std::vector<double>& z) {
                        const int kk = kd[t];
                        std::vector<double> z1, az1;
                        pc_arm8(r, z1);
                        coo_spmv(z1, az1);
                        std::vector<double> c(size_n);
                        for (int i = 0; i < n_f; ++i)
                            c[static_cast<std::size_t>(i)] =
                                r[static_cast<std::size_t>(i)] -
                                az1[static_cast<std::size_t>(i)];
                        char side = 'L', trans = 'T';
                        int m = n_f, ncol = 1, kref = kk, lda = std::max(1, n_f),
                            ldc = std::max(1, n_f), info = 0, lwork = -1;
                        double wq = 0.0;
                        dormqr_(&side, &trans, &m, &ncol, &kref,
                                AWls[static_cast<std::size_t>(t)].data(), &lda,
                                LStau[static_cast<std::size_t>(t)].data(),
                                c.data(), &ldc, &wq, &lwork, &info);
                        lwork = (info == 0) ? static_cast<int>(wq)
                                            : std::max(1, n_f);
                        std::vector<double> work(
                            static_cast<std::size_t>(std::max(1, lwork)));
                        dormqr_(&side, &trans, &m, &ncol, &kref,
                                AWls[static_cast<std::size_t>(t)].data(), &lda,
                                LStau[static_cast<std::size_t>(t)].data(),
                                c.data(), &ldc, work.data(), &lwork, &info);
                        std::vector<double> y(c.begin(), c.begin() + kk);
                        char uplo = 'U', tn = 'N', diag = 'N';
                        int rn = kk, rnrhs = 1, rlda = std::max(1, n_f), rldb = kk,
                            rinfo = 0;
                        dtrtrs_(&uplo, &tn, &diag, &rn, &rnrhs,
                                AWls[static_cast<std::size_t>(t)].data(), &rlda,
                                y.data(), &rldb, &rinfo);
                        if (ncall < 3) {
                            double yn = 0.0;
                            for (int j = 0; j < kk; ++j)
                                yn += y[static_cast<std::size_t>(j)] *
                                      y[static_cast<std::size_t>(j)];
                            std::cout << "  (||y|| " << label << " apply=" << ncall
                                      << ") = " << std::sqrt(yn) << '\n';
                            std::cout.flush();
                        }
                        ++ncall;
                        z.assign(size_n, 0.0);
                        for (int i = 0; i < n_f; ++i)
                            z[static_cast<std::size_t>(i)] =
                                z1[static_cast<std::size_t>(i)];
                        for (int j = 0; j < kk; ++j) {
                            const double yj = y[static_cast<std::size_t>(j)];
                            for (int i = 0; i < n_f; ++i)
                                z[static_cast<std::size_t>(i)] +=
                                    yj * Worth[vidx(i, j)];
                        }
                    };
                    auto apply_gal = [&](int t, const std::string& label,
                                         int& ncall,
                                         const std::vector<double>& r,
                                         std::vector<double>& z) {
                        const int kk = kd[t];
                        std::vector<double> z1, az1;
                        pc_arm8(r, z1);
                        coo_spmv(z1, az1);
                        std::vector<double> rp(size_n);
                        for (int i = 0; i < n_f; ++i)
                            rp[static_cast<std::size_t>(i)] =
                                r[static_cast<std::size_t>(i)] -
                                az1[static_cast<std::size_t>(i)];
                        std::vector<double> rhs(static_cast<std::size_t>(kk), 0.0);
                        for (int j = 0; j < kk; ++j) {
                            double s = 0.0;
                            for (int i = 0; i < n_f; ++i)
                                s += Worth[vidx(i, j)] *
                                     rp[static_cast<std::size_t>(i)];
                            rhs[static_cast<std::size_t>(j)] = s;
                        }
                        char tr = 'N';
                        int gn = kk, nrhs = 1, glda = std::max(1, kk), info = 0;
                        dgetrs_(&tr, &gn, &nrhs,
                                Glu[static_cast<std::size_t>(t)].data(), &glda,
                                Gipiv[static_cast<std::size_t>(t)].data(),
                                rhs.data(), &gn, &info);
                        if (ncall < 3) {
                            double yn = 0.0;
                            for (int j = 0; j < kk; ++j)
                                yn += rhs[static_cast<std::size_t>(j)] *
                                      rhs[static_cast<std::size_t>(j)];
                            std::cout << "  (||y|| " << label << " apply=" << ncall
                                      << ") = " << std::sqrt(yn) << '\n';
                            std::cout.flush();
                        }
                        ++ncall;
                        z.assign(size_n, 0.0);
                        for (int i = 0; i < n_f; ++i)
                            z[static_cast<std::size_t>(i)] =
                                z1[static_cast<std::size_t>(i)];
                        for (int j = 0; j < kk; ++j) {
                            const double cj = rhs[static_cast<std::size_t>(j)];
                            for (int i = 0; i < n_f; ++i)
                                z[static_cast<std::size_t>(i)] +=
                                    cj * Worth[vidx(i, j)];
                        }
                    };
                    int nc1 = 0, nc2 = 0, nc3 = 0, nc4 = 0;
                    KrylovOperator pc_15d1 = [&](const std::vector<double>& r,
                                                 std::vector<double>& z) {
                        apply_ls(0, "15d1_k2_LS", nc1, r, z);
                    };
                    KrylovOperator pc_15d2 = [&](const std::vector<double>& r,
                                                 std::vector<double>& z) {
                        apply_ls(1, "15d2_k8_LS", nc2, r, z);
                    };
                    KrylovOperator pc_15d3 = [&](const std::vector<double>& r,
                                                 std::vector<double>& z) {
                        apply_gal(0, "15d3_k2_Gal", nc3, r, z);
                    };
                    KrylovOperator pc_15d4 = [&](const std::vector<double>& r,
                                                 std::vector<double>& z) {
                        apply_gal(1, "15d4_k8_Gal", nc4, r, z);
                    };

                    // --- 8r2 baseline guard first ---
                    const std::size_t a_first = arms.size();
                    run_arm("8r2_baseline", pc_arm8);

                    // --- Projection of the 8r2 error onto the sorted subspace
                    // (orthonormal Worth: f_k = sum_{j<k} (worth_j . e8)^2 / ||e8||^2)
                    // + one-shot post-loop Galerkin cleanup (ARM 15e). ---
                    const std::vector<double> x8 = arms[a_first].x;
                    std::vector<double> ax8;
                    coo_spmv(x8, ax8);
                    std::vector<double> r8(size_n);
                    for (int i = 0; i < n_f; ++i)
                        r8[static_cast<std::size_t>(i)] =
                            b[static_cast<std::size_t>(i)] -
                            ax8[static_cast<std::size_t>(i)];
                    std::vector<double> e8 = refined_solve(r8);
                    double e8n2 = 0.0;
                    for (int i = 0; i < n_f; ++i)
                        e8n2 += e8[static_cast<std::size_t>(i)] *
                                e8[static_cast<std::size_t>(i)];
                    auto fk_orth = [&](int k) -> double {
                        double s = 0.0;
                        for (int j = 0; j < k; ++j) {
                            double d = 0.0;
                            for (int i = 0; i < n_f; ++i)
                                d += Worth[vidx(i, j)] *
                                     e8[static_cast<std::size_t>(i)];
                            s += d * d;
                        }
                        return (e8n2 > 0.0) ? s / e8n2 : 0.0;
                    };
                    const double f2 = fk_orth(2), f8 = fk_orth(8);
                    const ArmShare sh8 = attrib_share(x8);
                    std::cout << "  (projection sorted-Ritz) ||e8||^2=" << e8n2
                              << " f_2=" << f2 << " f_8=" << f8
                              << " e:global(8r2)=" << sh8.global << '\n';
                    std::cout.flush();

                    bool oneshot_ran = false;
                    double oneshot_rb = 0.0, oneshot_ra = 0.0, oneshot_eg = 1.0;
                    if (Galok[1]) {
                        std::vector<double> rhs8(static_cast<std::size_t>(KORTH),
                                                 0.0);
                        for (int j = 0; j < KORTH; ++j) {
                            double s = 0.0;
                            for (int i = 0; i < n_f; ++i)
                                s += Worth[vidx(i, j)] *
                                     r8[static_cast<std::size_t>(i)];
                            rhs8[static_cast<std::size_t>(j)] = s;
                        }
                        char tr = 'N';
                        int gn = KORTH, nrhs = 1, glda = KORTH, info = 0;
                        dgetrs_(&tr, &gn, &nrhs, Glu[1].data(), &glda,
                                Gipiv[1].data(), rhs8.data(), &gn, &info);
                        std::vector<double> xplus(x8);
                        for (int j = 0; j < KORTH; ++j) {
                            const double cj = rhs8[static_cast<std::size_t>(j)];
                            for (int i = 0; i < n_f; ++i)
                                xplus[static_cast<std::size_t>(i)] +=
                                    cj * Worth[vidx(i, j)];
                        }
                        double rb = 0.0;
                        for (int i = 0; i < n_f; ++i)
                            rb += r8[static_cast<std::size_t>(i)] *
                                  r8[static_cast<std::size_t>(i)];
                        rb = std::sqrt(rb) / ((b_norm > 0.0) ? b_norm : 1.0);
                        std::vector<double> axp;
                        coo_spmv(xplus, axp);
                        double ra = 0.0;
                        for (int i = 0; i < n_f; ++i) {
                            const double d = b[static_cast<std::size_t>(i)] -
                                             axp[static_cast<std::size_t>(i)];
                            ra += d * d;
                        }
                        ra = std::sqrt(ra) / ((b_norm > 0.0) ? b_norm : 1.0);
                        const ArmShare shp = attrib_share(xplus);
                        oneshot_ran = true;
                        oneshot_rb = rb;
                        oneshot_ra = ra;
                        oneshot_eg = shp.global;
                        std::cout << "  ARM 15e one-shot (k=8 Galerkin post-loop): "
                                     "rel_before="
                                  << rb << " rel_after=" << ra
                                  << " e:global_before=" << sh8.global
                                  << " e:global_after=" << shp.global
                                  << " e:seam_after=" << shp.seam
                                  << " e:high-p_after=" << shp.hi << '\n';
                    } else {
                        std::cout << "  ARM 15e one-shot SKIPPED (k=8 Galerkin block "
                                     "ill-conditioned)\n";
                    }
                    std::cout.flush();

                    // --- Run the in-PC deflation arms (skip guarded) ---
                    if (LSok[0])
                        run_arm("15d1_k2_LS", pc_15d1);
                    else
                        std::cout << "  15d1_k2_LS SKIPPED\n";
                    if (LSok[1])
                        run_arm("15d2_k8_LS", pc_15d2);
                    else
                        std::cout << "  15d2_k8_LS SKIPPED\n";
                    if (Galok[0])
                        run_arm("15d3_k2_Gal", pc_15d3);
                    else
                        std::cout << "  15d3_k2_Gal SKIPPED\n";
                    if (Galok[1])
                        run_arm("15d4_k8_Gal", pc_15d4);
                    else
                        std::cout << "  15d4_k8_Gal SKIPPED\n";

                    // --- Results table + rubric ---
                    std::cout << "\n=== ARM 15d/e RESULTS (canonical current-ladder "
                                 "fixture) ===\n";
                    std::cout << "  arm | rel@300 | rel@600 | ord[0,300) | "
                                 "ord[300,600) | non_decel | e:global | e:seam | "
                                 "e:high-p | solve_rel | wall/iter(s)\n";
                    auto rel_at = [&](const ArmResult& a, int k) -> double {
                        if (a.curve.empty())
                            return 0.0;
                        const int idx =
                            std::min(k, static_cast<int>(a.curve.size())) - 1;
                        return (b_norm > 0.0)
                                   ? a.curve[static_cast<std::size_t>(idx)] / b_norm
                                   : a.curve[static_cast<std::size_t>(idx)];
                    };
                    struct ArmEval {
                        std::string name;
                        double r600;
                        bool non_decel;
                        ArmShare sh;
                    };
                    std::vector<ArmEval> evals;
                    for (std::size_t ai = a_first; ai < arms.size(); ++ai) {
                        const ArmResult& a = arms[ai];
                        const double r300 = rel_at(a, 300);
                        const double r600 = rel_at(a, 600);
                        const double ord_a =
                            (r300 > 0.0) ? std::log10(1.0 / r300) : 0.0;
                        const double ord_b =
                            (r300 > 0.0 && r600 > 0.0) ? std::log10(r300 / r600)
                                                       : 0.0;
                        const ArmShare sh = attrib_share(a.x);
                        const bool non_decel = (ord_b >= ord_a - 1e-12);
                        const double wpi = (a.iters > 0) ? a.wall / a.iters : 0.0;
                        std::cout << "  " << a.name << " | " << r300 << " | " << r600
                                  << " | " << ord_a << " | " << ord_b << " | "
                                  << (non_decel ? "YES" : "no") << " | " << sh.global
                                  << " | " << sh.seam << " | " << sh.hi << " | "
                                  << sh.solve_rel << " | " << wpi << '\n';
                        evals.push_back({a.name, r600, non_decel, sh});
                    }
                    std::cout.flush();

                    bool pass_comp = false;
                    for (const ArmEval& ev : evals) {
                        if (ev.name.rfind("15d", 0) != 0)
                            continue;
                        const bool coll = (ev.sh.global < 0.1);
                        std::string v;
                        if (ev.non_decel && ev.r600 <= 1e-4 && coll) {
                            v = "PASS";
                            pass_comp = true;
                        } else if (ev.sh.global > 0.5)
                            v = "FAIL: channel not reached (global >0.5)";
                        else
                            v = "PARTIAL/NO-PASS";
                        std::cout << "  VERDICT " << ev.name << ": " << v
                                  << "  (rel@600=" << ev.r600
                                  << " non_decel=" << (ev.non_decel ? 1 : 0)
                                  << " e:global=" << ev.sh.global << ")\n";
                    }
                    const bool pass_oneshot =
                        oneshot_ran && oneshot_eg < 0.1 &&
                        oneshot_ra <= oneshot_rb;
                    std::cout << "  RUBRIC BRANCH: ";
                    if (pass_comp)
                        std::cout << "PASS-composition -- a small-k in-PC deflation "
                                     "converges; production border/coarse-mode PC is "
                                     "fundable.";
                    else if (pass_oneshot)
                        std::cout << "PASS-oneshot -- one exact near-null cleanup at "
                                     "the end collapses e:global with rel no worse; "
                                     "cheap-PC-to-E-W + post-loop cleanup is a "
                                     "production-viable pattern.";
                    else
                        std::cout << "PARK (final) -- the near-null channel is not "
                                     "treatable by deflation in any form here "
                                     "(small-k in-PC + one-shot both fail).";
                    std::cout << "\n  (rubric inputs: f_2=" << f2 << " f_8=" << f8
                              << " oneshot_rel_before=" << oneshot_rb
                              << " oneshot_rel_after=" << oneshot_ra
                              << " oneshot_e:global=" << oneshot_eg
                              << " pass_comp=" << (pass_comp ? 1 : 0)
                              << " pass_oneshot=" << (pass_oneshot ? 1 : 0)
                              << ")\n";
                    std::cout << "############ ARM 15d/e END ############\n";
                    std::cout.flush();
                    return;
                }

                // Field-unknown names carry the parser's trailing whitespace
                // ("H "); column keys and owner_var_name can disagree on it, so
                // trim both before matching (else the trace blocks find no
                // columns and silently no-op).
                auto trim = [](const std::string& s) {
                    const std::size_t b = s.find_first_not_of(" \t\r\n");
                    if (b == std::string::npos)
                        return std::string();
                    const std::size_t e = s.find_last_not_of(" \t\r\n");
                    return s.substr(b, e - b + 1);
                };

                // r = b - A_f x (true GMRES residual of an arm's iterate x).
                auto residual_of = [&](const std::vector<double>& x,
                                       std::vector<double>& r) {
                    coo_spmv(x, r);
                    for (int i = 0; i < n_f; ++i)
                        r[static_cast<std::size_t>(i)] =
                            b[static_cast<std::size_t>(i)] -
                            r[static_cast<std::size_t>(i)];
                };

                // matched / unmatched-high-p / TauMatch energy fractions.
                auto decomp = [&](const std::vector<double>& r) {
                    double em = 0, eu = 0, et = 0, etot = 0;
                    for (int i = 0; i < n_f; ++i) {
                        const double e2 = r[static_cast<std::size_t>(i)] *
                                          r[static_cast<std::size_t>(i)];
                        etot += e2;
                        const bool tm =
                            (i < static_cast<int>(tmeta.size())) &&
                            tmeta[static_cast<std::size_t>(i)].taxonomy ==
                                RowTaxonomy::TauMatch;
                        if (tm)
                            et += e2;
                        else if (fine_row_matched[static_cast<std::size_t>(i)])
                            em += e2;
                        else
                            eu += e2;
                    }
                    auto fr = [&](double e) {
                        return (etot > 0.0) ? std::sqrt(e / etot) : 0.0;
                    };
                    std::cout << "  decomp ||r||_matched=" << fr(em)
                              << " ||r||_unmatched(high-p)=" << fr(eu)
                              << " ||r||_TauMatch=" << fr(et) << '\n';
                };

                // slope: orders-of-magnitude reduction per 100-iteration window.
                auto print_slope = [&](const ArmResult& a) {
                    const int m =
                        std::min(static_cast<int>(a.curve.size()), a.iters);
                    std::cout << "  slope (orders reduced per 100 iters):\n";
                    double cum = 0.0;
                    for (int w = 0; w * 100 < m; ++w) {
                        const int aa = w * 100;
                        const int bb = std::min((w + 1) * 100, m);
                        const double rel_a =
                            (aa == 0)
                                ? 1.0
                                : a.curve[static_cast<std::size_t>(aa - 1)] /
                                      b_norm;
                        const double rel_b =
                            a.curve[static_cast<std::size_t>(bb - 1)] / b_norm;
                        const double orders =
                            (rel_a > 0.0 && rel_b > 0.0)
                                ? std::log10(rel_a / rel_b)
                                : 0.0;
                        cum += orders;
                        std::cout << "    [" << aa << "," << bb << "): " << orders
                                  << " orders (cum " << cum << ")\n";
                    }
                };

                // ---- Stage A: interface identity + aligned/misaligned test ----
                std::map<std::pair<int, int>, int> pair_index;
                std::vector<std::pair<int, int>> pair_key;
                std::vector<std::vector<int>> pair_rows;
                for (int i = 0; i < n_f && i < static_cast<int>(tmeta.size());
                     ++i) {
                    if (tmeta[static_cast<std::size_t>(i)].taxonomy !=
                        RowTaxonomy::TauMatch)
                        continue;
                    int d = tmeta[static_cast<std::size_t>(i)].dom;
                    int dp = tmeta[static_cast<std::size_t>(i)].dom_pair;
                    if (dp < 0)
                        dp = d;
                    const std::pair<int, int> pk(std::min(d, dp),
                                                 std::max(d, dp));
                    auto it = pair_index.find(pk);
                    int id;
                    if (it == pair_index.end()) {
                        id = static_cast<int>(pair_key.size());
                        pair_index[pk] = id;
                        pair_key.push_back(pk);
                        pair_rows.emplace_back();
                    } else {
                        id = it->second;
                    }
                    pair_rows[static_cast<std::size_t>(id)].push_back(i);
                }
                const int npairs = static_cast<int>(pair_key.size());

                // Sample up to 20 decoded rows/pair; one COO pass measures the
                // fraction of each sampled row's nnz on same-(j,k) columns.
                std::vector<int> pair_ndecoded(static_cast<std::size_t>(npairs), 0);
                std::vector<int> sample_pair(static_cast<std::size_t>(n_f), -1);
                for (int id = 0; id < npairs; ++id) {
                    int taken = 0;
                    for (int row : pair_rows[static_cast<std::size_t>(id)]) {
                        const bool dec =
                            fine_rows[static_cast<std::size_t>(row)].decoded &&
                            fine_rows[static_cast<std::size_t>(row)].j >= 0 &&
                            fine_rows[static_cast<std::size_t>(row)].k >= 0;
                        if (dec)
                            ++pair_ndecoded[static_cast<std::size_t>(id)];
                        if (dec && taken < 20) {
                            sample_pair[static_cast<std::size_t>(row)] = id;
                            ++taken;
                        }
                    }
                }
                std::vector<long long> nnz_tot(static_cast<std::size_t>(npairs),
                                               0);
                std::vector<long long> nnz_al(static_cast<std::size_t>(npairs), 0);
                for (long long e = 0; e < nnz_f; ++e) {
                    const int rr =
                        coo.irn[static_cast<std::size_t>(e)] - 1;
                    const int id = sample_pair[static_cast<std::size_t>(rr)];
                    if (id < 0)
                        continue;
                    const int cc =
                        coo.jcn[static_cast<std::size_t>(e)] - 1;
                    ++nnz_tot[static_cast<std::size_t>(id)];
                    if (fine_cols[static_cast<std::size_t>(cc)].j ==
                            fine_rows[static_cast<std::size_t>(rr)].j &&
                        fine_cols[static_cast<std::size_t>(cc)].k ==
                            fine_rows[static_cast<std::size_t>(rr)].k)
                        ++nnz_al[static_cast<std::size_t>(id)];
                }
                std::vector<double> pair_alignfrac(
                    static_cast<std::size_t>(npairs), 0.0);
                std::vector<char> pair_aligned(static_cast<std::size_t>(npairs),
                                               0);
                for (int id = 0; id < npairs; ++id) {
                    pair_alignfrac[static_cast<std::size_t>(id)] =
                        (nnz_tot[static_cast<std::size_t>(id)] > 0)
                            ? static_cast<double>(
                                  nnz_al[static_cast<std::size_t>(id)]) /
                                  static_cast<double>(
                                      nnz_tot[static_cast<std::size_t>(id)])
                            : 0.0;
                    pair_aligned[static_cast<std::size_t>(id)] =
                        (pair_ndecoded[static_cast<std::size_t>(id)] > 0) &&
                        (pair_alignfrac[static_cast<std::size_t>(id)] >= 0.5);
                }

                // Ranked per-interface table on a given residual; returns the
                // misaligned TauMatch energy fraction (the gate quantity).
                auto print_interface_table = [&](const std::string& title,
                                                 const std::vector<double>& r)
                    -> double {
                    std::vector<double> energy(
                        static_cast<std::size_t>(npairs), 0.0);
                    double total_tm = 0.0;
                    for (int id = 0; id < npairs; ++id)
                        for (int row : pair_rows[static_cast<std::size_t>(id)]) {
                            const double e2 = r[static_cast<std::size_t>(row)] *
                                              r[static_cast<std::size_t>(row)];
                            energy[static_cast<std::size_t>(id)] += e2;
                            total_tm += e2;
                        }
                    std::vector<int> order(static_cast<std::size_t>(npairs));
                    for (int id = 0; id < npairs; ++id)
                        order[static_cast<std::size_t>(id)] = id;
                    std::sort(order.begin(), order.end(),
                              [&](int a, int c) {
                                  return energy[static_cast<std::size_t>(a)] >
                                         energy[static_cast<std::size_t>(c)];
                              });
                    std::cout << "=== " << title << " ===\n"
                              << "  interface(d1,d2) | share ||r_pair||/||r_TM|| "
                                 "| aligned | n_rows | n_decoded | align_frac\n";
                    double e_aligned = 0, e_mis = 0;
                    for (int id : order) {
                        const double share =
                            (total_tm > 0.0)
                                ? std::sqrt(energy[static_cast<std::size_t>(id)] /
                                            total_tm)
                                : 0.0;
                        const bool al =
                            pair_aligned[static_cast<std::size_t>(id)] != 0;
                        if (al)
                            e_aligned += energy[static_cast<std::size_t>(id)];
                        else
                            e_mis += energy[static_cast<std::size_t>(id)];
                        std::cout << "  ("
                                  << pair_key[static_cast<std::size_t>(id)].first
                                  << ","
                                  << pair_key[static_cast<std::size_t>(id)].second
                                  << ")\t" << share << "\t" << (al ? "YES" : "no")
                                  << "\t"
                                  << pair_rows[static_cast<std::size_t>(id)].size()
                                  << "\t"
                                  << pair_ndecoded[static_cast<std::size_t>(id)]
                                  << "\t"
                                  << pair_alignfrac[static_cast<std::size_t>(id)]
                                  << '\n';
                    }
                    const double f_mis =
                        (total_tm > 0.0) ? e_mis / total_tm : 0.0;
                    std::cout << "  aligned    TauMatch energy fraction = "
                              << ((total_tm > 0.0) ? e_aligned / total_tm : 0.0)
                              << '\n'
                              << "  misaligned TauMatch energy fraction = "
                              << f_mis << '\n';
                    std::cout.flush();
                    return f_mis;
                };

                // Run the arm-8 baseline first; its final residual feeds Stage A.
                const std::size_t stageb_first = arms.size();
                run_arm("8r2_dirichlet_additive", pc_arm8);
                std::vector<double> r8;
                residual_of(arms.back().x, r8);

                // ====================================================================
                // ARM 13: error-attribution falsifier (PCOARSE_ARM13).
                // Reuses the arm-8r2 @600 iterate residual r8 (the same vector
                // that feeds the Stage-A gate below) and the fine RHS b. Three
                // rank-0 measurements that do NOT perturb the Stage-A verdict:
                //   (A) |a|^2-weighted interface alignment (vs the nnz-count one),
                //   (B) row-norm-scaled Stage-A reprint (raw | scaled),
                //   (C) fine-COO direct solve e=A^{-1}r8, e0=A^{-1}b + energy split.
                // ====================================================================
                if (env_flag_enabled("PCOARSE_ARM13", false)) {
                    std::cout << "\n############ ARM 13: error-attribution "
                                 "falsifier ############\n";

                    // ---- (A) |a|^2-weighted interface alignment fraction ----
                    // Same sampled decoded rows as the nnz-count align_frac
                    // above; accumulate Sum|a|^2 on same-(j,k) columns vs all.
                    std::vector<double> a2_tot(static_cast<std::size_t>(npairs),
                                               0.0);
                    std::vector<double> a2_al(static_cast<std::size_t>(npairs),
                                              0.0);
                    for (long long e = 0; e < nnz_f; ++e) {
                        const std::size_t ee = static_cast<std::size_t>(e);
                        const int rr = coo.irn[ee] - 1;
                        const int id = sample_pair[static_cast<std::size_t>(rr)];
                        if (id < 0)
                            continue;
                        const int cc = coo.jcn[ee] - 1;
                        const double a2 = coo.a[ee] * coo.a[ee];
                        a2_tot[static_cast<std::size_t>(id)] += a2;
                        if (fine_cols[static_cast<std::size_t>(cc)].j ==
                                fine_rows[static_cast<std::size_t>(rr)].j &&
                            fine_cols[static_cast<std::size_t>(cc)].k ==
                                fine_rows[static_cast<std::size_t>(rr)].k)
                            a2_al[static_cast<std::size_t>(id)] += a2;
                    }
                    // Rank pairs by arm-8r2 TauMatch residual energy so the wall
                    // pairs sort to the top (matches the Stage-A table ordering).
                    std::vector<double> pair_e8(static_cast<std::size_t>(npairs),
                                                0.0);
                    for (int id = 0; id < npairs; ++id)
                        for (int row : pair_rows[static_cast<std::size_t>(id)])
                            pair_e8[static_cast<std::size_t>(id)] +=
                                r8[static_cast<std::size_t>(row)] *
                                r8[static_cast<std::size_t>(row)];
                    std::vector<int> a_order(static_cast<std::size_t>(npairs));
                    for (int id = 0; id < npairs; ++id)
                        a_order[static_cast<std::size_t>(id)] = id;
                    std::sort(a_order.begin(), a_order.end(), [&](int a, int c) {
                        return pair_e8[static_cast<std::size_t>(a)] >
                               pair_e8[static_cast<std::size_t>(c)];
                    });
                    std::cout << "=== (A) |a|^2-weighted vs nnz-count interface "
                                 "alignment ===\n"
                              << "  interface(d1,d2) | nnz_align_frac | "
                                 "a2_align_frac | n_decoded | n_rows\n";
                    for (int id : a_order) {
                        const double a2f =
                            (a2_tot[static_cast<std::size_t>(id)] > 0.0)
                                ? a2_al[static_cast<std::size_t>(id)] /
                                      a2_tot[static_cast<std::size_t>(id)]
                                : 0.0;
                        std::cout
                            << "  ("
                            << pair_key[static_cast<std::size_t>(id)].first << ","
                            << pair_key[static_cast<std::size_t>(id)].second
                            << ")\t"
                            << pair_alignfrac[static_cast<std::size_t>(id)] << "\t"
                            << a2f << "\t"
                            << pair_ndecoded[static_cast<std::size_t>(id)] << "\t"
                            << pair_rows[static_cast<std::size_t>(id)].size()
                            << (pair_ndecoded[static_cast<std::size_t>(id)] == 0
                                    ? "  (misaligned-by-default)"
                                    : "")
                            << '\n';
                    }
                    std::cout.flush();

                    // ---- (B) row-norm-scaled Stage-A reprint (raw | scaled) ----
                    // rownorm2[row] = Sum_j A(row,j)^2; scaled weight is
                    // r_i^2 / rownorm2[i] (rows with rownorm2==0 skipped).
                    std::vector<double> rownorm2(static_cast<std::size_t>(n_f),
                                                 0.0);
                    for (long long e = 0; e < nnz_f; ++e) {
                        const std::size_t ee = static_cast<std::size_t>(e);
                        const int rr = coo.irn[ee] - 1;
                        rownorm2[static_cast<std::size_t>(rr)] +=
                            coo.a[ee] * coo.a[ee];
                    }
                    auto rowscaled = [&](int row) -> double {
                        const double rn = rownorm2[static_cast<std::size_t>(row)];
                        if (rn <= 0.0)
                            return 0.0;
                        return r8[static_cast<std::size_t>(row)] *
                               r8[static_cast<std::size_t>(row)] / rn;
                    };
                    std::vector<double> raw_e(static_cast<std::size_t>(npairs),
                                              0.0);
                    std::vector<double> scl_e(static_cast<std::size_t>(npairs),
                                              0.0);
                    double raw_tm = 0.0, scl_tm = 0.0;
                    for (int id = 0; id < npairs; ++id)
                        for (int row : pair_rows[static_cast<std::size_t>(id)]) {
                            const double e2raw = r8[static_cast<std::size_t>(row)] *
                                                 r8[static_cast<std::size_t>(row)];
                            const double e2scl = rowscaled(row);
                            raw_e[static_cast<std::size_t>(id)] += e2raw;
                            scl_e[static_cast<std::size_t>(id)] += e2scl;
                            raw_tm += e2raw;
                            scl_tm += e2scl;
                        }
                    std::vector<int> b_order(static_cast<std::size_t>(npairs));
                    for (int id = 0; id < npairs; ++id)
                        b_order[static_cast<std::size_t>(id)] = id;
                    std::sort(b_order.begin(), b_order.end(), [&](int a, int c) {
                        return raw_e[static_cast<std::size_t>(a)] >
                               raw_e[static_cast<std::size_t>(c)];
                    });
                    std::cout << "=== (B) row-norm-scaled Stage-A interface table "
                                 "(raw | scaled shares) ===\n"
                              << "  interface(d1,d2) | raw_share | scaled_share | "
                                 "aligned\n";
                    double raw_mis = 0.0, scl_mis = 0.0;
                    for (int id : b_order) {
                        const bool al =
                            pair_aligned[static_cast<std::size_t>(id)] != 0;
                        if (!al) {
                            raw_mis += raw_e[static_cast<std::size_t>(id)];
                            scl_mis += scl_e[static_cast<std::size_t>(id)];
                        }
                        const double rs =
                            (raw_tm > 0.0)
                                ? std::sqrt(raw_e[static_cast<std::size_t>(id)] /
                                            raw_tm)
                                : 0.0;
                        const double ss =
                            (scl_tm > 0.0)
                                ? std::sqrt(scl_e[static_cast<std::size_t>(id)] /
                                            scl_tm)
                                : 0.0;
                        std::cout
                            << "  ("
                            << pair_key[static_cast<std::size_t>(id)].first << ","
                            << pair_key[static_cast<std::size_t>(id)].second
                            << ")\t" << rs << "\t" << ss << "\t"
                            << (al ? "YES" : "no") << '\n';
                    }
                    std::cout << "  misaligned TauMatch energy fraction: raw="
                              << ((raw_tm > 0.0) ? raw_mis / raw_tm : 0.0)
                              << "  scaled="
                              << ((scl_tm > 0.0) ? scl_mis / scl_tm : 0.0)
                              << '\n';
                    double rm = 0, ru = 0, rt = 0, rtot = 0;
                    double sm = 0, su = 0, st = 0, stot = 0;
                    for (int i = 0; i < n_f; ++i) {
                        const double e2raw = r8[static_cast<std::size_t>(i)] *
                                             r8[static_cast<std::size_t>(i)];
                        const double e2scl = rowscaled(i);
                        rtot += e2raw;
                        stot += e2scl;
                        const bool tm = (i < static_cast<int>(tmeta.size())) &&
                                        tmeta[static_cast<std::size_t>(i)]
                                                .taxonomy == RowTaxonomy::TauMatch;
                        if (tm) {
                            rt += e2raw;
                            st += e2scl;
                        } else if (fine_row_matched[static_cast<std::size_t>(i)]) {
                            rm += e2raw;
                            sm += e2scl;
                        } else {
                            ru += e2raw;
                            su += e2scl;
                        }
                    }
                    auto rf = [](double e, double tot) {
                        return (tot > 0.0) ? std::sqrt(e / tot) : 0.0;
                    };
                    std::cout << "=== (B) 3-channel decomposition of r8 (raw | "
                                 "scaled) ===\n"
                              << "  ||r||_matched      raw=" << rf(rm, rtot)
                              << " scaled=" << rf(sm, stot) << '\n'
                              << "  ||r||_unmatched-hp raw=" << rf(ru, rtot)
                              << " scaled=" << rf(su, stot) << '\n'
                              << "  ||r||_TauMatch     raw=" << rf(rt, rtot)
                              << " scaled=" << rf(st, stot) << '\n';
                    std::cout.flush();

                    // ---- (C) fine-COO direct solve: e=A^{-1}r8, e0=A^{-1}b ----
                    // Column ownership (domain), seam-adjacency (nonzero in any
                    // TauMatch row), and low/high-p masks -> energy attribution
                    // of the true error vectors.
                    std::vector<int> col_dom(static_cast<std::size_t>(n_f), -1);
                    int ndom_cols = 0;
                    for (int cc = 0; cc < n_f; ++cc) {
                        const ColumnInfo& ci = cmap[static_cast<std::size_t>(cc)];
                        if (!(ci.is_var_domain || ci.var_double_idx >= 0 ||
                              ci.domain < 0)) {
                            col_dom[static_cast<std::size_t>(cc)] = ci.domain;
                            ndom_cols = std::max(ndom_cols, ci.domain + 1);
                        }
                    }
                    std::vector<char> col_seam(static_cast<std::size_t>(n_f), 0);
                    for (long long e = 0; e < nnz_f; ++e) {
                        const std::size_t ee = static_cast<std::size_t>(e);
                        const int rr = coo.irn[ee] - 1;
                        if (rr < static_cast<int>(tmeta.size()) &&
                            tmeta[static_cast<std::size_t>(rr)].taxonomy ==
                                RowTaxonomy::TauMatch)
                            col_seam[static_cast<std::size_t>(coo.jcn[ee] - 1)] =
                                1;
                    }

                    const auto fine_factor_t0 = std::chrono::steady_clock::now();
                    MumpsLinearSolver fine_solver(
                        n_f, config.mumps.ordering, false, 0,
                        mumps_runtime_state.icntl14, MPI_COMM_SELF);
                    fine_solver.set_pattern(n_f, nnz_f, coo.irn.data(),
                                            coo.jcn.data());
                    fine_solver.analyze_pattern();
                    fine_solver.factor_analyzed(coo.a.data());
                    mumps_runtime_state.icntl14 = fine_solver.last_icntl14();
                    const double fine_factor_wall =
                        std::chrono::duration<double>(
                            std::chrono::steady_clock::now() - fine_factor_t0)
                            .count();
                    std::cout << "  (C) fine MUMPS factor: n=" << n_f
                              << " nnz=" << nnz_f << " wall=" << fine_factor_wall
                              << "s\n";
                    std::cout.flush();

                    // Accuracy: iterative refinement of the fine solve on the
                    // same factor. The force-balance Jacobian is near-singular
                    // (Mb<->Hc coupling), so a single MUMPS solve on the drop-tol
                    // COO can leave ~1e-5. Refine, printing the seam / high-p
                    // energy shares at each iterate; the amended GATE 3 accepts
                    // <=1e-6 ONLY when those shares stop moving (<0.01 absolute).
                    auto quick_split = [&](const std::vector<double>& ev)
                        -> std::pair<double, double> {
                        double tot = 0.0, seam = 0.0, hi = 0.0;
                        for (int i = 0; i < n_f; ++i) {
                            const double e2 = ev[static_cast<std::size_t>(i)] *
                                              ev[static_cast<std::size_t>(i)];
                            tot += e2;
                            if (col_seam[static_cast<std::size_t>(i)] != 0)
                                seam += e2;
                            if (fine_col_matched[static_cast<std::size_t>(i)] == 0)
                                hi += e2;
                        }
                        const double inv = (tot > 0.0) ? 1.0 / tot : 0.0;
                        return {seam * inv, hi * inv};
                    };
                    auto solve_refined =
                        [&](const std::string& tag, const std::vector<double>& rhs,
                            std::vector<double>& e_out) -> std::pair<double, bool> {
                        e_out = rhs;
                        fine_solver.solve(e_out.data()); // e = A^{-1} rhs
                        const int kMaxRefine = 5;
                        double rel = 0.0, prev_seam = 0.0, prev_hi = 0.0,
                               max_move = 0.0;
                        for (int it = 0; it <= kMaxRefine; ++it) {
                            std::vector<double> ax;
                            coo_spmv(e_out, ax);
                            double num = 0.0, den = 0.0;
                            for (int i = 0; i < n_f; ++i) {
                                const double d = ax[static_cast<std::size_t>(i)] -
                                                 rhs[static_cast<std::size_t>(i)];
                                num += d * d;
                                den += rhs[static_cast<std::size_t>(i)] *
                                       rhs[static_cast<std::size_t>(i)];
                            }
                            rel = (den > 0.0) ? std::sqrt(num / den) : 0.0;
                            const std::pair<double, double> sh = quick_split(e_out);
                            const double move =
                                (it == 0)
                                    ? 0.0
                                    : std::max(std::fabs(sh.first - prev_seam),
                                               std::fabs(sh.second - prev_hi));
                            if (it > 0)
                                max_move = std::max(max_move, move);
                            std::cout << "  (C) " << tag << " refine it=" << it
                                      << " ||Ae-rhs||/||rhs||=" << rel
                                      << " seam=" << sh.first
                                      << " high-p=" << sh.second;
                            if (it > 0)
                                std::cout << " d_share=" << move;
                            std::cout << '\n';
                            prev_seam = sh.first;
                            prev_hi = sh.second;
                            if (rel < 1e-8 || it == kMaxRefine)
                                break;
                            std::vector<double> corr(static_cast<std::size_t>(n_f));
                            for (int i = 0; i < n_f; ++i)
                                corr[static_cast<std::size_t>(i)] =
                                    rhs[static_cast<std::size_t>(i)] -
                                    ax[static_cast<std::size_t>(i)];
                            fine_solver.solve(corr.data()); // A^{-1} residual
                            for (int i = 0; i < n_f; ++i)
                                e_out[static_cast<std::size_t>(i)] +=
                                    corr[static_cast<std::size_t>(i)];
                        }
                        const bool ok =
                            (rel < 1e-8) || (rel <= 1e-6 && max_move < 0.01);
                        std::cout << "  (C) " << tag
                                  << " final ||Ae-rhs||/||rhs||=" << rel
                                  << " max_share_move=" << max_move << "  "
                                  << (rel < 1e-8
                                          ? "[<1e-8 OK]"
                                          : (ok ? "[<=1e-6 + stable OK]"
                                                : "*** FAIL (>1e-6 or unstable)"))
                                  << '\n';
                        std::cout.flush();
                        return {rel, ok};
                    };
                    std::vector<double> e_err, e0_err;
                    const std::pair<double, bool> res_e =
                        solve_refined("e(r8)", r8, e_err);
                    const std::pair<double, bool> res_e0 =
                        solve_refined("e0(b)", b, e0_err);
                    const bool solve_ok = res_e.second && res_e0.second;
                    std::cout << "  (C) GATE 3 (fine-solve accuracy): "
                              << (solve_ok ? "PASS" : "FAIL") << " (e rel="
                              << res_e.first << " e0 rel=" << res_e0.first
                              << ")\n";

                    auto attribute = [&](const std::string& tag,
                                         const std::vector<double>& ev) {
                        double tot = 0.0;
                        for (int i = 0; i < n_f; ++i)
                            tot += ev[static_cast<std::size_t>(i)] *
                                   ev[static_cast<std::size_t>(i)];
                        std::vector<double> dom_e(
                            static_cast<std::size_t>(std::max(1, ndom_cols)),
                            0.0);
                        double glob_e = 0.0, seam_e = 0.0, bulk_e = 0.0;
                        double hi_e = 0.0, lo_e = 0.0;
                        double seam_hi = 0.0, seam_lo = 0.0, bulk_hi = 0.0,
                               bulk_lo = 0.0;
                        for (int i = 0; i < n_f; ++i) {
                            const double e2 = ev[static_cast<std::size_t>(i)] *
                                              ev[static_cast<std::size_t>(i)];
                            const int dm = col_dom[static_cast<std::size_t>(i)];
                            if (dm >= 0)
                                dom_e[static_cast<std::size_t>(dm)] += e2;
                            else
                                glob_e += e2;
                            const bool seam =
                                col_seam[static_cast<std::size_t>(i)] != 0;
                            const bool hi =
                                fine_col_matched[static_cast<std::size_t>(i)] == 0;
                            if (seam)
                                seam_e += e2;
                            else
                                bulk_e += e2;
                            if (hi)
                                hi_e += e2;
                            else
                                lo_e += e2;
                            if (seam && hi)
                                seam_hi += e2;
                            else if (seam)
                                seam_lo += e2;
                            else if (hi)
                                bulk_hi += e2;
                            else
                                bulk_lo += e2;
                        }
                        const double inv = (tot > 0.0) ? 1.0 / tot : 0.0;
                        std::cout << "=== (C) energy attribution of " << tag
                                  << " (||.||^2=" << tot << ") ===\n";
                        std::vector<int> dorder(dom_e.size());
                        for (std::size_t d = 0; d < dom_e.size(); ++d)
                            dorder[d] = static_cast<int>(d);
                        std::sort(dorder.begin(), dorder.end(), [&](int a, int c) {
                            return dom_e[static_cast<std::size_t>(a)] >
                                   dom_e[static_cast<std::size_t>(c)];
                        });
                        std::cout << "  per-domain top-5 (frac of ||.||^2):\n";
                        for (int t = 0;
                             t < 5 && t < static_cast<int>(dorder.size()); ++t) {
                            const int d = dorder[static_cast<std::size_t>(t)];
                            std::cout << "    dom" << d << " = "
                                      << dom_e[static_cast<std::size_t>(d)] * inv
                                      << '\n';
                        }
                        std::cout << "    global/scalar = " << glob_e * inv
                                  << '\n';
                        double dom_sum = glob_e;
                        for (double de : dom_e)
                            dom_sum += de;
                        std::cout << "  seam-adjacent = " << seam_e * inv
                                  << "  bulk = " << bulk_e * inv << '\n'
                                  << "  unmatched-high-p = " << hi_e * inv
                                  << "  matched-low-p = " << lo_e * inv << '\n'
                                  << "  2x2 [seam x high-p]:\n"
                                  << "    seam&high-p=" << seam_hi * inv
                                  << "  seam&low-p=" << seam_lo * inv << '\n'
                                  << "    bulk&high-p=" << bulk_hi * inv
                                  << "  bulk&low-p=" << bulk_lo * inv << '\n';
                        auto chk = [&](const char* nm, double s) {
                            const double d = std::fabs(s - 1.0);
                            std::cout << "  [assert " << nm << " sum=" << s << "] "
                                      << (d < 1e-9 ? "OK" : "*** FAIL") << '\n';
                        };
                        chk("domain", dom_sum * inv);
                        chk("seam/bulk", (seam_e + bulk_e) * inv);
                        chk("hi/lo", (hi_e + lo_e) * inv);
                        chk("2x2",
                            (seam_hi + seam_lo + bulk_hi + bulk_lo) * inv);
                        std::cout.flush();
                    };
                    attribute("e = A^{-1} r8 (arm-8r2 error)", e_err);
                    attribute("e0 = A^{-1} b (initial error)", e0_err);

                    std::cout << "############ ARM 13 END ############\n";
                    std::cout.flush();
                }

                std::cout << "\n############ STAGE A: interface decomposition "
                             "############\n";
                const double f_mis = print_interface_table(
                    "Stage A per-interface TauMatch decomposition (8r2 final "
                    "residual)",
                    r8);

                std::cout << "=== STAGE GATE ===\n";
                if (f_mis > 0.5) {
                    std::cout
                        << "  VERDICT: MISALIGNED interfaces carry the majority "
                           "of TauMatch residual (misaligned energy frac="
                        << f_mis << " > 0.5).\n"
                        << "  STOP after Stage A per spec: the cheap per-mode "
                           "trace arm cannot fix the dominant channel.\n"
                        << "  Misaligned seams need inter-basis angular "
                           "projection (full DtN build, separate work order).\n"
                        << "  Stage B (arms 11/12) NOT built.\n";
                    std::cout.flush();
                    return; // STOP after Stage A
                }
                std::cout
                    << "  VERDICT: ALIGNED interfaces carry the majority "
                       "(aligned energy frac="
                    << (1.0 - f_mis)
                    << "); proceeding to Stage B (arms 11/12).\n";
                std::cout.flush();

                // ---- Stage B: per-(pair,var,comp,j,k) min-norm trace blocks ----
                // Field-column index: (var,comp,dom,j,k) -> radial i-series.
                auto skey = [](const std::string& v, int comp, int dm, int jj,
                               int kk) {
                    std::ostringstream o;
                    o << v << '|' << comp << '|' << dm << '|' << jj << '|' << kk;
                    return o.str();
                };
                std::unordered_map<std::string, std::vector<int>> field_col_index;
                for (int c = 0; c < n_f; ++c) {
                    const PcoarseColumnKey& fc =
                        fine_cols[static_cast<std::size_t>(c)];
                    if (fc.block == 2)
                        field_col_index[skey(trim(fc.name), fc.comp, fc.dom,
                                             fc.j, fc.k)]
                            .push_back(c);
                }

                // Group aligned+decoded TauMatch rows by (pair,var,comp,j,k).
                struct TraceGroup {
                    int d1 = -1, d2 = -1, comp = -1, j = -1, k = -1;
                    std::string var;
                    std::vector<int> rows;
                };
                std::unordered_map<std::string, TraceGroup> groups;
                for (int id = 0; id < npairs; ++id) {
                    if (pair_aligned[static_cast<std::size_t>(id)] == 0)
                        continue;
                    const int d1 = pair_key[static_cast<std::size_t>(id)].first;
                    const int d2 = pair_key[static_cast<std::size_t>(id)].second;
                    for (int row : pair_rows[static_cast<std::size_t>(id)]) {
                        const PcoarseRowKey& rk =
                            fine_rows[static_cast<std::size_t>(row)];
                        const std::string var = trim(
                            tmeta[static_cast<std::size_t>(row)].owner_var_name);
                        if (!rk.decoded || var.empty() || rk.comp < 0 ||
                            rk.j < 0 || rk.k < 0)
                            continue;
                        std::ostringstream gk;
                        gk << d1 << ':' << d2 << '#' << var << '#' << rk.comp
                           << '#' << rk.j << '#' << rk.k;
                        TraceGroup& g = groups[gk.str()];
                        g.d1 = d1;
                        g.d2 = d2;
                        g.var = var;
                        g.comp = rk.comp;
                        g.j = rk.j;
                        g.k = rk.k;
                        g.rows.push_back(row);
                    }
                }

                struct TraceBlock {
                    int p = 0, q = 0;
                    std::vector<double> at_qr; // q x p col-major, QR of A^T
                    std::vector<double> tau;   // length p
                    std::vector<int> rows_g;
                    std::vector<int> cols_g;
                };

                // CSR over the union of trace rows (one COO pass).
                std::vector<int> trace_row_local(
                    static_cast<std::size_t>(n_f), -1);
                std::vector<int> trace_rows;
                for (const auto& kv : groups)
                    for (int row : kv.second.rows)
                        if (trace_row_local[static_cast<std::size_t>(row)] < 0) {
                            trace_row_local[static_cast<std::size_t>(row)] =
                                static_cast<int>(trace_rows.size());
                            trace_rows.push_back(row);
                        }
                std::vector<std::vector<std::pair<int, double>>> csr(
                    trace_rows.size());
                for (long long e = 0; e < nnz_f; ++e) {
                    const int rr = coo.irn[static_cast<std::size_t>(e)] - 1;
                    const int lr = trace_row_local[static_cast<std::size_t>(rr)];
                    if (lr < 0)
                        continue;
                    csr[static_cast<std::size_t>(lr)].emplace_back(
                        coo.jcn[static_cast<std::size_t>(e)] - 1,
                        coo.a[static_cast<std::size_t>(e)]);
                }

                // Build+factor one min-norm block per group.
                std::vector<TraceBlock> tblocks;
                std::vector<int> col_owner(static_cast<std::size_t>(n_f), -1);
                std::vector<int> col_local(static_cast<std::size_t>(n_f), -1);
                long long overlap_cols = 0, trace_bytes = 0;
                int skipped_blocks = 0;
                const auto tb_t0 = std::chrono::steady_clock::now();
                for (const auto& kv : groups) {
                    const TraceGroup& g = kv.second;
                    const int p = static_cast<int>(g.rows.size());
                    std::vector<int> cols;
                    const auto add_dom_cols = [&](int dm) {
                        auto it = field_col_index.find(
                            skey(g.var, g.comp, dm, g.j, g.k));
                        if (it != field_col_index.end())
                            for (int c : it->second)
                                cols.push_back(c);
                    };
                    add_dom_cols(g.d1);
                    if (g.d2 != g.d1)
                        add_dom_cols(g.d2);
                    const int q = static_cast<int>(cols.size());
                    if (p <= 0 || q <= 0 || p > q) {
                        ++skipped_blocks;
                        continue;
                    }
                    const int bidx = static_cast<int>(tblocks.size());
                    for (int c : cols) {
                        if (col_owner[static_cast<std::size_t>(c)] >= 0)
                            ++overlap_cols;
                        else
                            col_owner[static_cast<std::size_t>(c)] = bidx;
                    }
                    for (int cl = 0; cl < q; ++cl)
                        col_local[static_cast<std::size_t>(
                            cols[static_cast<std::size_t>(cl)])] = cl;
                    std::vector<double> at(
                        static_cast<std::size_t>(q) *
                            static_cast<std::size_t>(p),
                        0.0);
                    for (int rl = 0; rl < p; ++rl) {
                        const int lr = trace_row_local[static_cast<std::size_t>(
                            g.rows[static_cast<std::size_t>(rl)])];
                        for (const auto& cv :
                             csr[static_cast<std::size_t>(lr)]) {
                            const int cl =
                                col_local[static_cast<std::size_t>(cv.first)];
                            if (cl >= 0)
                                at[static_cast<std::size_t>(cl) +
                                   static_cast<std::size_t>(rl) *
                                       static_cast<std::size_t>(q)] += cv.second;
                        }
                    }
                    for (int cl = 0; cl < q; ++cl)
                        col_local[static_cast<std::size_t>(
                            cols[static_cast<std::size_t>(cl)])] = -1;
                    int qm = q, qn = p, qlda = std::max(1, q), info = 0;
                    std::vector<double> tau(static_cast<std::size_t>(p));
                    double wq = 0.0;
                    int lwork = -1;
                    dgeqrf_(&qm, &qn, at.data(), &qlda, tau.data(), &wq, &lwork,
                            &info);
                    lwork = (info == 0) ? static_cast<int>(wq) : std::max(1, p);
                    std::vector<double> work(
                        static_cast<std::size_t>(std::max(1, lwork)));
                    dgeqrf_(&qm, &qn, at.data(), &qlda, tau.data(), work.data(),
                            &lwork, &info);
                    if (info != 0)
                        std::cerr << "  trace block dgeqrf info=" << info << '\n';
                    trace_bytes += static_cast<long long>(q) * p * 8;
                    TraceBlock blk;
                    blk.p = p;
                    blk.q = q;
                    blk.at_qr = std::move(at);
                    blk.tau = std::move(tau);
                    blk.rows_g = g.rows;
                    blk.cols_g = std::move(cols);
                    tblocks.push_back(std::move(blk));
                }
                std::cout << "=== Stage B trace smoother T ===\n"
                          << "  aligned groups=" << groups.size()
                          << " blocks built=" << tblocks.size()
                          << " skipped=" << skipped_blocks
                          << " total A^T bytes=" << trace_bytes << " ("
                          << (static_cast<double>(trace_bytes) / 1e6)
                          << " MB)\n"
                          << "  column-disjointness: overlapping column-writes="
                          << overlap_cols
                          << (overlap_cols == 0
                                  ? " (DISJOINT: pure additive scatter)\n"
                                  : " (OVERLAP: shared cols summed additively; "
                                    "still a valid PC)\n")
                          << "  build wall="
                          << std::chrono::duration<double>(
                                 std::chrono::steady_clock::now() - tb_t0)
                                 .count()
                          << "s\n";
                std::cout.flush();

                // T(r): min-norm radial adjustment per block, additive scatter.
                auto trace_apply = [&](const std::vector<double>& r,
                                       std::vector<double>& t) {
                    t.assign(static_cast<std::size_t>(n_f), 0.0);
                    for (TraceBlock& blk : tblocks) {
                        const int p = blk.p, q = blk.q;
                        if (p <= 0 || q <= 0)
                            continue;
                        std::vector<double> y(static_cast<std::size_t>(p));
                        for (int rl = 0; rl < p; ++rl)
                            y[static_cast<std::size_t>(rl)] =
                                r[static_cast<std::size_t>(
                                    blk.rows_g[static_cast<std::size_t>(rl)])];
                        // Solve R^T y = b_rows (R = upper factor, lda=q).
                        char uplo = 'U', tr = 'T', dg = 'N';
                        int nn = p, nrhs = 1, lda = std::max(1, q), ldb = p,
                            info = 0;
                        dtrtrs_(&uplo, &tr, &dg, &nn, &nrhs, blk.at_qr.data(),
                                &lda, y.data(), &ldb, &info);
                        if (info != 0)
                            continue; // singular block -> no correction
                        // x = Q * [y; 0]  (min-norm solution in row space).
                        std::vector<double> c(static_cast<std::size_t>(q), 0.0);
                        for (int rl = 0; rl < p; ++rl)
                            c[static_cast<std::size_t>(rl)] =
                                y[static_cast<std::size_t>(rl)];
                        char side = 'L', trn = 'N';
                        int qm = q, qn = 1, qk = p, qlda = std::max(1, q),
                            qldc = std::max(1, q), qinfo = 0, lwork = -1;
                        double wq = 0.0;
                        dormqr_(&side, &trn, &qm, &qn, &qk, blk.at_qr.data(),
                                &qlda, blk.tau.data(), c.data(), &qldc, &wq,
                                &lwork, &qinfo);
                        lwork = (qinfo == 0) ? static_cast<int>(wq)
                                             : std::max(1, q);
                        std::vector<double> work(
                            static_cast<std::size_t>(std::max(1, lwork)));
                        dormqr_(&side, &trn, &qm, &qn, &qk, blk.at_qr.data(),
                                &qlda, blk.tau.data(), c.data(), &qldc,
                                work.data(), &lwork, &qinfo);
                        for (int cl = 0; cl < q; ++cl)
                            t[static_cast<std::size_t>(
                                blk.cols_g[static_cast<std::size_t>(cl)])] +=
                                c[static_cast<std::size_t>(cl)];
                    }
                };

                // Arm 11: multiplicative trace last.
                // z = S(r) + P A_c^-1 R r ; z += T(r - A_f z).
                KrylovOperator pc_arm11 = [&](const std::vector<double>& r,
                                              std::vector<double>& z) {
                    std::vector<double> sr, cc, az, defl, t;
                    schwarz_apply(r, sr);
                    coarse_correction(r, cc);
                    z.assign(static_cast<std::size_t>(n_f), 0.0);
                    for (int i = 0; i < n_f; ++i)
                        z[static_cast<std::size_t>(i)] =
                            sr[static_cast<std::size_t>(i)] +
                            cc[static_cast<std::size_t>(i)];
                    coo_spmv(z, az);
                    defl.assign(static_cast<std::size_t>(n_f), 0.0);
                    for (int i = 0; i < n_f; ++i)
                        defl[static_cast<std::size_t>(i)] =
                            r[static_cast<std::size_t>(i)] -
                            az[static_cast<std::size_t>(i)];
                    trace_apply(defl, t);
                    for (int i = 0; i < n_f; ++i)
                        z[static_cast<std::size_t>(i)] +=
                            t[static_cast<std::size_t>(i)];
                };
                // Arm 12: fully additive (double-count control).
                // z = S(r) + P A_c^-1 R r + T(r).
                KrylovOperator pc_arm12 = [&](const std::vector<double>& r,
                                              std::vector<double>& z) {
                    std::vector<double> sr, cc, t;
                    schwarz_apply(r, sr);
                    coarse_correction(r, cc);
                    trace_apply(r, t);
                    z.assign(static_cast<std::size_t>(n_f), 0.0);
                    for (int i = 0; i < n_f; ++i)
                        z[static_cast<std::size_t>(i)] =
                            sr[static_cast<std::size_t>(i)] +
                            cc[static_cast<std::size_t>(i)] +
                            t[static_cast<std::size_t>(i)];
                };
                run_arm("11_mult_trace_last", pc_arm11);
                run_arm("12_additive_trace", pc_arm12);

                // ---- Stage B report: slope + decomp + interface table/arm ----
                std::cout << "\n############ STAGE B REPORT ############\n";
                for (std::size_t ai = stageb_first; ai < arms.size(); ++ai) {
                    const ArmResult& a = arms[ai];
                    std::vector<double> ra;
                    residual_of(a.x, ra);
                    std::cout << "\n=== arm " << a.name << ": iters=" << a.iters
                              << " rel_resid=" << a.rel_resid
                              << " wall=" << a.wall << "s ===\n";
                    print_slope(a);
                    decomp(ra);
                    print_interface_table(
                        "per-interface TauMatch on " + a.name + " final residual",
                        ra);
                }
            }
        }

        // --- Final-residual decomposition for the best (lowest rel_resid) arm ---
        int best = 0;
        for (int a = 1; a < static_cast<int>(arms.size()); ++a)
            if (arms[static_cast<std::size_t>(a)].rel_resid <
                arms[static_cast<std::size_t>(best)].rel_resid)
                best = a;
        std::vector<RowMetadata> meta;
        classify_equation_row_metadata(meta);
        std::vector<double> ax;
        coo_spmv(arms[static_cast<std::size_t>(best)].x, ax);
        double e_matched = 0.0, e_unmatched = 0.0, e_taumatch = 0.0, e_total = 0.0;
        for (int i = 0; i < n_f; ++i) {
            const double res = b[static_cast<std::size_t>(i)] -
                               ax[static_cast<std::size_t>(i)];
            const double e2 = res * res;
            e_total += e2;
            const bool is_taumatch =
                (i < static_cast<int>(meta.size())) &&
                meta[static_cast<std::size_t>(i)].taxonomy == RowTaxonomy::TauMatch;
            if (is_taumatch)
                e_taumatch += e2;
            else if (fine_row_matched[static_cast<std::size_t>(i)])
                e_matched += e2;
            else
                e_unmatched += e2;
        }
        auto frac = [&](double e) {
            return (e_total > 0.0) ? std::sqrt(e / e_total) : 0.0;
        };
        std::cout << "=== pcoarse final-residual decomposition (best arm: "
                  << arms[static_cast<std::size_t>(best)].name << ") ===\n"
                  << "  ||r||_matched/||r||   = " << frac(e_matched) << '\n'
                  << "  ||r||_unmatched/||r|| = " << frac(e_unmatched)
                  << "   (high-p modes -> smoother wall)\n"
                  << "  ||r||_TauMatch/||r||  = " << frac(e_taumatch)
                  << "   (Gamma-trace/interface -> DtN next)\n";

        std::cout << "=== pcoarse A/B summary ===\n";
        for (const ArmResult& a : arms)
            std::cout << "  " << a.name << ": converged=" << std::boolalpha
                      << a.converged << " iters=" << a.iters
                      << " rel_resid=" << a.rel_resid << " wall=" << a.wall << "s\n";
        std::cout << "  gate: PASS if an arm reaches rel_resid<=1e-6 in <=60 iters "
                     "with geometric slope; CONDITIONAL 60-150; FAIL on plateau\n";
        std::cout.flush();
#else
        (void)directory;
        (void)coo;
        (void)fine_residual;
        (void)config;
        std::cerr << "pcoarse probe: built without MUMPS.\n";
#endif
    }
} // namespace Kadath
