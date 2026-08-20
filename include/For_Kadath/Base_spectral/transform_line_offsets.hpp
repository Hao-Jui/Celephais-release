#pragma once

#include "For_Kadath/Array/dim_array.hpp"

#include <cassert>

namespace Kadath
{
    /**
     * Flat buffer offsets of the line a 1-D transform driver is about to read
     * and write, carried across the traversal instead of rebuilt per line.
     *
     * The drivers (\c Base_spectral::coef_dim, \c coef_i_dim, \c ope_1d) walk
     * one \c Index over the axes above the transformed one and another over
     * the axes below it, then rebuild a flat offset with a Horner pass for
     * every single line. Both walks are plain mixed-radix odometers with the
     * lowest axis as the fastest digit, so the flat offset they address can be
     * carried: one add in the common case, one wrap-subtract per carried
     * digit. This is addressing only. The sequence of digit tuples is the one
     * \c Index::inc() produces, so every line is visited in the same order,
     * reads the same values and runs the same kernel arithmetic.
     *
     * Preserving the order (not merely the per-line arithmetic) is required:
     * the drivers size their line scratch to \c max(nbr_coef,nbr_conf) but
     * gather only the input count, so when a transform grows an axis the tail
     * of the scratch carries the previous line's output into the kernel.
     *
     * Two offsets are carried because a transform's input and output differ in
     * the size of the transformed axis, which scales the flat weight of every
     * axis below it. Callers pass those two scales; for the axes above the
     * transformed one the weights coincide and one offset is enough.
     */
    class Transform_line_offsets
    {
      public:
        /**
         * Odometer over axes [first, last) of \c sizes.
         * @param sizes [input] shape whose axes are being walked.
         * @param first [input] fastest digit (first axis incremented).
         * @param last [input] one past the slowest digit.
         * @param in_scale [input] weight multiplier of the input buffer.
         * @param out_scale [input] weight multiplier of the output buffer.
         */
        Transform_line_offsets(const Dim_array& sizes, int first, int last, int in_scale, int out_scale)
            : count_(last - first)
        {
            assert(first >= 0 && last <= sizes.get_ndim() && count_ <= max_axes);
            int weight = 1;
            for (int axis = last - 1; axis >= first; axis--) {
                const int slot = axis - first;
                extent_[slot] = sizes(axis);
                digit_[slot] = 0;
                in_step_[slot] = weight * in_scale;
                out_step_[slot] = weight * out_scale;
                weight *= extent_[slot];
            }
        }

        int in_offset() const { return in_offset_; }
        int out_offset() const { return out_offset_; }

        /**
         * Moves to the next digit tuple, exactly as \c Index::inc() would over
         * the same axis range. A full cycle returns to the all-zero tuple, so
         * one odometer can be reused for every pass of an enclosing loop.
         */
        void advance()
        {
            for (int slot = 0; slot < count_; slot++) {
                if (++digit_[slot] < extent_[slot]) {
                    in_offset_ += in_step_[slot];
                    out_offset_ += out_step_[slot];
                    return;
                }
                digit_[slot] = 0;
                in_offset_ -= (extent_[slot] - 1) * in_step_[slot];
                out_offset_ -= (extent_[slot] - 1) * out_step_[slot];
            }
        }

      private:
        /// Fields are one- to three-dimensional; the cap only bounds the frame.
        static constexpr int max_axes = 8;

        int extent_[max_axes];
        int digit_[max_axes];
        int in_step_[max_axes];
        int out_step_[max_axes];
        int count_;
        int in_offset_ = 0;
        int out_offset_ = 0;
    };
} // namespace Kadath
