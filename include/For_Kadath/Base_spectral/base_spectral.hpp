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
 *   2026-06-16  Modified for the Celephais tree; see
 *               PATCHES-KADATH-UPSTREAM.md and LICENSE_SOURCE_AUDIT.tsv.
 *   2026-08-11  Shared the outer basis-slot snapshot and deferred its
 *               three-dimensional allocation until first mutation.
 *   2026-08-12  Added exact-order four-point spectral summation.
 */

#pragma once

#include "For_Kadath/Array/dim_array.hpp"
#include "For_Kadath/Array/array.hpp"
#include "For_Kadath/Array/memory.hpp"

#include <array>
#include <cstddef>
#include <memory>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

#define NBR_MAX_BASE 30
#define CHEB 1
#define CHEB_EVEN 2
#define CHEB_ODD 3
#define COSSIN 4
#define COS 5
#define COS_EVEN 6
#define COS_ODD 7
#define SIN 8
#define SIN_EVEN 9
#define SIN_ODD 10
#define LEG 11
#define LEG_EVEN 12
#define LEG_ODD 13
#define COSSIN_EVEN 14
#define COSSIN_ODD 15

namespace Kadath
{
    class Point;

    struct Transform1dTrafficSnapshot
    {
        unsigned long long forward = 0;
        unsigned long long backward = 0;
    };

    void begin_transform_1d_traffic_profile(bool enabled);
    void record_forward_transform_1d(unsigned long long count);
    void record_backward_transform_1d(unsigned long long count);
    Transform1dTrafficSnapshot end_transform_1d_traffic_profile();

    class Transform1dTrafficScope
    {
      public:
        explicit Transform1dTrafficScope(bool const enabled) : active_{enabled}
        {
            begin_transform_1d_traffic_profile(enabled);
        }

        ~Transform1dTrafficScope()
        {
            if (active_)
                static_cast<void>(end_transform_1d_traffic_profile());
        }

        Transform1dTrafficScope(const Transform1dTrafficScope&) = delete;
        Transform1dTrafficScope& operator=(const Transform1dTrafficScope&) = delete;

        [[nodiscard]] Transform1dTrafficSnapshot finish()
        {
            if (!active_)
                return {};
            active_ = false;
            return end_transform_1d_traffic_profile();
        }

      private:
        bool active_ = false;
    };

    /**
     * Assembly-lifetime traversal-plan statistics for the derivative-only
     * one-dimensional operator. The counters are thread-local, matching the
     * solver's pure-MPI execution model.
     */
    struct OpeDer1dWorkspaceStats
    {
        unsigned long long calls = 0;
        unsigned long long lanes = 0;
        unsigned long long plan_hits = 0;
        unsigned long long plan_misses = 0;
        unsigned long long plan_fallbacks = 0;
        unsigned long long scratch_hits = 0;
        unsigned long long scratch_misses = 0;
        unsigned long long aosoa_calls = 0;
        unsigned long long aosoa_lanes = 0;
        unsigned long long aosoa_fallbacks = 0;
        std::size_t retained_bytes = 0;
        std::size_t peak_retained_bytes = 0;
        std::size_t retained_byte_cap = 0;
    };

    /**
     * Enables bounded logical ownership of derivative traversal plans and one
     * line-scratch buffer. The outermost scope releases traversal-vector
     * capacity and workspace ownership. MemoryMapper may retain the one
     * legacy-sized Array backing block for reuse, so this does not guarantee
     * a reduction in process RSS.
     */
    class OpeDer1dAssemblyWorkspaceScope
    {
      public:
        OpeDer1dAssemblyWorkspaceScope();
        ~OpeDer1dAssemblyWorkspaceScope();
        OpeDer1dAssemblyWorkspaceScope(const OpeDer1dAssemblyWorkspaceScope&) = delete;
        OpeDer1dAssemblyWorkspaceScope& operator=(const OpeDer1dAssemblyWorkspaceScope&) = delete;
    };

    void reset_ope_der_1d_workspace();
    void reset_ope_der_1d_workspace_stats();
    OpeDer1dWorkspaceStats ope_der_1d_workspace_stats();

    /**
     * Result buffer for the one-dimensional spectral kernels, retained per line
     * length so a kernel no longer pays a \c MemoryMapper round trip for every
     * 1-D line it transforms. Kernels write the whole buffer before copying it
     * back over their argument, exactly as they must already do today: the
     * mapper hands out recycled, non-zeroed slabs, so an unwritten entry has
     * always been a read of stale data.
     *
     * Matching lengths exactly keeps \c Array::operator= usable unchanged. The
     * retained set is bounded by the largest collocation count on any axis, a
     * few kilobytes in total.
     *
     * Plain (not thread-local) storage mirrors \c coef_mem: production solvers
     * run one compute thread per MPI rank, and thread_local resolution was
     * measurable in this same first-J hot path.
     */
    inline Array<double>& ope_1d_line_scratch(int length)
    {
        assert(length > 0);
        static std::vector<std::unique_ptr<Array<double>>> buffers_by_length;
        const auto index = static_cast<std::size_t>(length);
        if (buffers_by_length.size() <= index)
            buffers_by_length.resize(index + 1);
        auto& buffer = buffers_by_length[index];
        if (buffer == nullptr)
            buffer = std::make_unique<Array<double>>(length);
        return *buffer;
    }

    /**
     * Class for storing the basis of decompositions of a field.
     *
     * It mainly consists in a list of integers, encoding the various basis.
     * \li \c CHEB : Chebyshev polynomials.
     * \li \c CHEB_EVEN : even Chebyshev polynomials.
     * \li \c CHEB_ODD : odd Chebyshev polynomials.
     * \li \c COSSIN : series of sines and cosines.
     * \li \c COS : series of cosines.
     * \li \c COS_EVEN : series of even cosines.
     * \li \c COS_ODD : series of odd cosines.
     * \li \c SIN : series of sines.
     * \li \c SIN_EVEN : series of even sines.
     * \li \c SIN_ODD : series of odd sines.
     * \li \c LEG : Legendre polynomials.
     * \li \c LEG_EVEN : even Legendre polynomials.
     * \li \c LEG_ODD : odd Legendre polynomials.
     * \li \c COSSIN_EVEN : series of sines and cosines with even harmonics only.
     * \li \c COSSIN_ODD : series of sines and cosines with odd harmonics only.
     * \ingroup spectral
     **/

    class Base_spectral : public MemoryMappable
    {
      protected:
        /**
         * Pointer-compatible copy-on-write ownership for one basis array.
         * Const access shares the immutable snapshot; every mutable access
         * detaches first so copied Base_spectral objects remain independent.
         */
        class SharedBasisArray
        {
          private:
            std::shared_ptr<Array<int>> value;

            void detach()
            {
                if (value && value.use_count() != 1)
                    value = std::make_shared<Array<int>>(*value);
            }

          public:
            SharedBasisArray() noexcept = default;
            SharedBasisArray(std::nullptr_t) noexcept {}
            SharedBasisArray(std::unique_ptr<Array<int>> source) : value(std::move(source)) {}

            SharedBasisArray& operator=(std::unique_ptr<Array<int>> source)
            {
                value = std::move(source);
                return *this;
            }

            SharedBasisArray& operator=(std::nullptr_t) noexcept
            {
                value.reset();
                return *this;
            }

            Array<int>* get()
            {
                detach();
                return value.get();
            }
            const Array<int>* get() const noexcept { return value.get(); }

            Array<int>& operator*()
            {
                detach();
                return *value;
            }
            const Array<int>& operator*() const noexcept { return *value; }

            Array<int>* operator->()
            {
                detach();
                return value.get();
            }
            const Array<int>* operator->() const noexcept { return value.get(); }

            explicit operator bool() const noexcept { return static_cast<bool>(value); }
            bool operator==(std::nullptr_t) const noexcept { return value == nullptr; }
            bool operator!=(std::nullptr_t) const noexcept { return value != nullptr; }

            void reset() noexcept { value.reset(); }
        };

        /**
         * Copy-on-write ownership for the outer list of basis arrays.
         *
         * Dimensions up to three initially point at an ownerless static null
         * snapshot.  This keeps construction of the overwhelmingly common
         * undefined three-dimensional base allocation-free.  The first write
         * obtains an owned snapshot, while copies continue to share both the
         * outer slots and their existing per-axis SharedBasisArray handles.
         * Larger dimensions always use the general heap-backed snapshot.
         */
        class SharedBasisList
        {
          private:
            static constexpr std::size_t bounded_capacity = 3;
            using Snapshot = std::shared_ptr<SharedBasisArray[]>;
            using Snapshot3 = std::array<SharedBasisArray, bounded_capacity>;

            std::size_t size_ = 0;
            Snapshot snapshot_;

            [[nodiscard]] static SharedBasisArray* null_slots() noexcept
            {
                static SharedBasisArray slots[bounded_capacity];
                return slots;
            }

            [[nodiscard]] static Snapshot null_snapshot() noexcept
            {
                return Snapshot{Snapshot{}, null_slots()};
            }

            [[nodiscard]] static std::size_t checked_size(int dimensions)
            {
                if (dimensions < 0)
                    throw std::length_error("negative Base_spectral dimension");
                return static_cast<std::size_t>(dimensions);
            }

            [[nodiscard]] static Snapshot make_snapshot(std::size_t size)
            {
                if (size == bounded_capacity) {
                    auto owner = std::make_shared<Snapshot3>();
                    SharedBasisArray* const data = owner->data();
                    return Snapshot{std::move(owner), data};
                }
                return std::make_shared<SharedBasisArray[]>(size);
            }

            [[nodiscard]] const SharedBasisArray& read(std::size_t index) const noexcept
            {
                assert(index < size_);
                return snapshot_.get()[index];
            }

            void detach()
            {
                const long owners = snapshot_.use_count();
                if (owners == 1)
                    return;

                Snapshot replacement = make_snapshot(size_);
                if (owners != 0) {
                    for (std::size_t i = 0; i < size_; ++i)
                        replacement[i] = snapshot_[i];
                }
                snapshot_ = std::move(replacement);
            }

            [[nodiscard]] SharedBasisArray& write(std::size_t index)
            {
                assert(index < size_);
                detach();
                return snapshot_[index];
            }

          public:
            class SlotProxy
            {
              private:
                SharedBasisList* list_;
                std::size_t index_;

                [[nodiscard]] const SharedBasisArray& read() const noexcept
                {
                    return list_->read(index_);
                }

                [[nodiscard]] SharedBasisArray& write()
                {
                    return list_->write(index_);
                }

              public:
                SlotProxy(SharedBasisList& list, std::size_t index) noexcept
                    : list_{&list}, index_{index}
                {}

                SlotProxy& operator=(const SlotProxy& source)
                {
                    SharedBasisArray value = source.read();
                    write() = std::move(value);
                    return *this;
                }

                SlotProxy& operator=(const SharedBasisArray& source)
                {
                    SharedBasisArray value = source;
                    write() = std::move(value);
                    return *this;
                }

                SlotProxy& operator=(std::unique_ptr<Array<int>> source)
                {
                    write() = std::move(source);
                    return *this;
                }

                SlotProxy& operator=(std::nullptr_t)
                {
                    if (read() != nullptr)
                        write() = nullptr;
                    return *this;
                }

                Array<int>* get()
                {
                    return write().get();
                }
                const Array<int>* get() const noexcept { return read().get(); }

                Array<int>& operator*()
                {
                    return *write();
                }
                const Array<int>& operator*() const noexcept { return *read(); }

                Array<int>* operator->()
                {
                    return write().operator->();
                }
                const Array<int>* operator->() const noexcept { return read().operator->(); }

                explicit operator bool() const noexcept { return static_cast<bool>(read()); }
                bool operator==(std::nullptr_t) const noexcept { return read() == nullptr; }
                bool operator!=(std::nullptr_t) const noexcept { return read() != nullptr; }

                void reset()
                {
                    if (read() != nullptr)
                        write().reset();
                }
            };

            SharedBasisList() noexcept : snapshot_{null_snapshot()} {}

            explicit SharedBasisList(int dimensions)
                : size_{checked_size(dimensions)},
                  snapshot_{size_ <= bounded_capacity ? null_snapshot() : make_snapshot(size_)}
            {}

            SharedBasisList(const SharedBasisList&) noexcept = default;
            SharedBasisList& operator=(const SharedBasisList&) noexcept = default;

            SharedBasisList(SharedBasisList&& source) noexcept
                : size_{source.size_},
                  snapshot_{source.size_ <= bounded_capacity
                                ? std::move(source.snapshot_)
                                : source.snapshot_}
            {
                if (source.size_ <= bounded_capacity)
                    source.snapshot_ = null_snapshot();
            }

            SharedBasisList& operator=(SharedBasisList&& source) noexcept
            {
                swap(source);
                return *this;
            }

            [[nodiscard]] SlotProxy operator[](std::size_t index) noexcept
            {
                return SlotProxy{*this, index};
            }

            [[nodiscard]] const SharedBasisArray& operator[](std::size_t index) const noexcept
            {
                return read(index);
            }

            void reset(int dimensions)
            {
                const std::size_t replacement_size = checked_size(dimensions);
                Snapshot replacement = replacement_size <= bounded_capacity
                                           ? null_snapshot()
                                           : make_snapshot(replacement_size);
                size_ = replacement_size;
                snapshot_ = std::move(replacement);
            }

            void clear()
            {
                if (size_ <= bounded_capacity) {
                    snapshot_ = null_snapshot();
                    return;
                }

                bool already_clear = true;
                for (std::size_t i = 0; i < size_; ++i) {
                    if (snapshot_[i] != nullptr) {
                        already_clear = false;
                        break;
                    }
                }
                if (already_clear)
                    return;

                if (snapshot_.use_count() != 1) {
                    snapshot_ = make_snapshot(size_);
                    return;
                }

                for (std::size_t i = 0; i < size_; ++i)
                    snapshot_[i].reset();
            }

            void swap(SharedBasisList& source) noexcept
            {
                std::swap(size_, source.size_);
                snapshot_.swap(source.snapshot_);
            }
        };

        static_assert(sizeof(SharedBasisList) == 3 * sizeof(void*));

        bool def; ///< \c true if the \c Base_spectral is defined and \c false otherwise.
        int ndim; ///< Number of dimensions.
        /**
         * Arrays containing the various basis of decomposition.
         * The size of each array depends on the order of the various numerical coordinates.
         */
        SharedBasisList bases_1d;

      public:
        /**
         * Standard constructor, the \c Base_spectral is not defined.
         * @param i [input] : number of dimensions.
         */
        explicit Base_spectral(int i);
        Base_spectral(const Base_spectral&);     ///< Copy constructor
        Base_spectral(Base_spectral&&) noexcept; ///< Move constructor.
        Base_spectral(BinarySource&);            ///< Constructor from a BinarySource (modern API).

      public:
        ~Base_spectral();       ///< Destructor
        void save(BinarySink&) const; ///< Save via BinarySink (modern API).
      public:
        /**
         * @returns : \c true if the basis is defined, \c false otherwise.
         */
        bool is_def() const { return def; };

        /**
         * Sets all the basis to the undefined state.
         */
        void set_non_def();

        void operator=(const Base_spectral&);               ///< Assignement operator.
        Base_spectral& operator=(Base_spectral&&) noexcept; ///< Move assignment operator.
        void swap(Base_spectral&) noexcept;                 ///< Swaps the content with the source.

        const Array<int>* get_base_1d(int i) const { return bases_1d[i].get(); }; ///< Returns one of the 1d base array.

        /**
         * Allocates the various arrays, for a given number of coefficients.
         * @param nbr_coefs [input] : a \c Dim_array storing the number of coefficients in each dimenions.
         */
        void allocate(const Dim_array& nbr_coefs);
        /**
         * Allocates the various arrays, for a given number of coefficients and sets basis to some values (same for all
         * harmonics). It assumes one is working in 3D.
         * @param nbr_coefs [input] : a \c Dim_array storing the number of coefficients in each dimenions.
         * @param basephi : basis for \f$varphi\f$
         * @param basetheta : basis for \f$varphi\f$
         * @param baser : basis for \f$varphi\f$
         */
        void set(Dim_array const& nbr_coefs, int basephi, int basetheta, int baser);

        /**
         * performs the coefficient transformation for one particular variable.
         * @param var [input] : the variable for which the coefficients are to be computed.
         * @param nbr [input] : number of coefficients for \c var.
         * @param tab [input] : values of the field (read only; left unchanged).
         * @returns a fresh array holding the coefficients along \c var.
         */
        Array<double> coef_dim(int var, int nbr, const Array<double>& tab) const;
        /**
         * Computes the coefficients for all the variables.
         * @param nbr_coefs [input] : number of coefficients.
         * @param so [input] : values of the field in the configuration space.
         * @returns the coefficients of the field.
         */
        Array<double> coef(const Dim_array& nbr_coefs, const Array<double>& so) const;
        /**
         * Performs the inverse coefficient transformation for one particular variable.
         * @param var [input] : the variable for which the coefficients are to be computed.
         * @param nbr [input] : number of points in the configuration space for \c var.
         * @param tab [input] : values of the field (read only; left unchanged).
         * @returns a fresh array holding the configuration-space values along \c var.
         */
        Array<double> coef_i_dim(int var, int nbr, const Array<double>& tab) const;
        /**
         * Computes the values in the configuration space.
         * @param nbr_points [input] : number of points.
         * @param so [input] : values of the field in the coefficients space.
         * @returns the coefficients of the field.
         */
        Array<double> coef_i(const Dim_array& nbr_points, const Array<double>& so) const;
        /**
         * Computes the spectral summation.
         * @param num [input] : numerical coordinates used in the summation.
         * @param tab [input] : spectral coefficients of the field.
         * @returns the summation.
         */
        double summation(const Point& num, const Array<double>& tab) const;
        /**
         * Computes four independent spectral sums over one coefficient array.
         * The four point lanes retain the scalar coefficient and accumulation
         * order while allowing independent recurrences to execute together.
         * @param num [input] : four non-null numerical-coordinate pointers.
         * @param tab [input] : shared spectral coefficients of the field.
         * @param result [output] : one summation for each point lane.
         */
        void summation_points4(std::span<const Point* const, 4> num,
                               const Array<double>& tab,
                               std::span<double, 4> result) const;
        /**
         * Collapses every spectral dimension except the last, evaluating the
         * series at the numerical coordinates \c num(1..ndim-1). The result is
         * the 1D coefficient array in the last dimension; pair it with
         * \c summation_last_dim to sweep the final coordinate without redoing
         * this collapse (e.g. a quadrature node loop over the last variable).
         * @param num [input] : numerical coordinates; only components 1..ndim-1 are read.
         * @param tab [input] : spectral coefficients of the field.
         * @returns the partially summed 1D coefficient array.
         */
        Array<double> summation_but_last(const Point& num, const Array<double>& tab) const;
        /**
         * Evaluates the last spectral dimension of a partial result produced by
         * \c summation_but_last at the numerical coordinate \c x.
         * @param x [input] : numerical coordinate along the last dimension.
         * @param partial [input] : 1D coefficient array from \c summation_but_last.
         * @returns the summation.
         */
        double summation_last_dim(double x, const Array<double>& partial) const;
        /**
         * One-dimensional operator acting in the coefficient space.
         * @param function [input] : function pointing to the particular operation to be performed (i.e. like
         * \c mult_sin_x_1d )
         * @param var [input] : variable on which the operatio is to be performed.
         * @param so [input] : coefficients of the field before the operation.
         * @param base [input/output] : basis of the field. After the operation, the basis associated with \c var
         * may is changed but not the others basis which should be changed by hand, if needed.
         * @returns the coefficients of the result.
         */
        Array<double> ope_1d(int (*function)(int, Array<double>&), int var, const Array<double>& so,
                             Base_spectral& base) const;

        /**
         * Same operator, for kernels that read and write their line in place at
         * the traversal stride instead of through a gathered contiguous copy.
         *
         * The gathering form above makes four passes over every line: gather
         * from \c so, kernel, the kernel's own copy back over its argument, and
         * scatter into the result. A strided kernel makes one. Only kernels
         * whose every output is an independent expression over the input line
         * may take this form: relocating loads and stores must not reassociate
         * anything, so accumulating recurrences stay on the gathering form.
         *
         * @param function [input] : kernel, invoked once per line as
         * \c (base, line of \c so, line of the result, line length, stride).
         * It must write every one of the line's entries: unlike the gathering
         * form there is no contiguous copy behind it to supply a missed slot.
         * @param var [input] : variable on which the operation is performed.
         * @param so [input] : coefficients of the field before the operation.
         * @param base [input/output] : basis of the field.
         * @returns the coefficients of the result.
         */
        Array<double> ope_1d(int (*function)(int, const double*, double*, int, int), int var,
                             const Array<double>& so, Base_spectral& base) const;

        /**
         * Derivative-only form of ope_1d that reuses bounded traversal and
         * contiguous line-scratch workspace around the legacy kernel.
         */
        Array<double> ope_der_1d(int var, const Array<double>& so,
                                 Base_spectral& base) const;

        /**
         * Apply the derivative-only operator to compatible lanes in fixed
         * tiles of four, sequentially reusing one line-scratch buffer. Every
         * output Array must already have the same shape as its corresponding
         * input; input/output storage must not alias.
         */
        static void ope_der_1d_batch(int var,
                                     const Base_spectral* const* bases_in,
                                     const Array<double>* const* inputs,
                                     Base_spectral* const* bases_out,
                                     Array<double>* const* outputs,
                                     int lane_count);

      private:
        static void validate_ope_der_1d_batch(int var,
                                              const Base_spectral* const* bases_in,
                                              const Array<double>* const* inputs,
                                              Base_spectral* const* bases_out,
                                              Array<double>* const* outputs,
                                              int lane_count);
        static void ope_der_1d_batch_workspace(int var,
                                               const Base_spectral* const* bases_in,
                                               const Array<double>* const* inputs,
                                               Base_spectral* const* bases_out,
                                               Array<double>* const* outputs,
                                               int lane_count);
        static void ope_der_1d_batch_aosoa(int var,
                                           const Base_spectral* const* bases_in,
                                           const Array<double>* const* inputs,
                                           Base_spectral* const* bases_out,
                                           Array<double>* const* outputs,
                                           int lane_count);

      public:

        friend bool operator==(const Base_spectral&, const Base_spectral&); ///< Comparison operator
        friend ostream& operator<<(ostream& o, const Base_spectral&);       ///< Display
        friend class Space;
        friend class Space_spheric;
        friend class Scalar;
        friend class Domain_nucleus;
        friend class Domain_shell;
        friend class Domain_shell_log;
        friend class Domain_compact;
        friend class Domain_bispheric_rect;
        friend class Domain_bispheric_chi_first;
        friend class Domain_bispheric_eta_first;
        friend class Domain_bispheric_rect_nosym;
        friend class Domain_bispheric_chi_first_nosym;
        friend class Domain_bispheric_eta_first_nosym;
        friend class Domain_critic_inner;
        friend class Domain_critic_outer;
        friend class Domain_polar_nucleus;
        friend class Domain_polar_shell;
        friend class Domain_polar_shell_inner_adapted;
        friend class Domain_polar_shell_outer_adapted;
        friend class Domain_polar_shell_bilateral_adapted;
        friend class Domain_polar_compact;
        friend class Domain_oned_ori;
        friend class Domain_oned_qcq;
        friend class Domain_oned_inf;
        friend class Domain_spheric_periodic_nucleus;
        friend class Domain_spheric_periodic_shell;
        friend class Domain_spheric_periodic_compact;
        friend class Domain_spheric_time_nucleus;
        friend class Domain_spheric_time_shell;
        friend class Domain_spheric_time_compact;
        friend class Domain_shell_inner_adapted;
        friend class Domain_shell_outer_adapted;
        friend class Domain_shell_inner_homothetic;
        friend class Domain_shell_outer_homothetic;
        friend class Domain_polar_shell_inner_homothetic;
        friend class Domain_polar_shell_outer_homothetic;
        friend class Domain_nucleus_symphi;
        friend class Domain_shell_symphi;
        friend class Domain_compact_symphi;
        friend class Domain_nucleus_nosym;
        friend class Domain_shell_nosym;
        friend class Domain_compact_nosym;
        friend class Domain_shell_inner_adapted_nosym;
        friend class Domain_shell_outer_adapted_nosym;
        friend class Domain_shell_inner_homothetic_nosym;
        friend class Domain_shell_outer_homothetic_nosym;
        friend class Domain_polar_periodic_nucleus;
        friend class Domain_polar_periodic_shell;
        friend class Domain_fourD_periodic_nucleus;
        friend class Domain_fourD_periodic_shell;
    };
} // namespace Kadath
