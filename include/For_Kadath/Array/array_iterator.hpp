//
// Created by sauliac on 20/03/2020.
//

#pragma once

#include "dim_array.hpp"
#include "memory.hpp"

namespace Kadath
{

    /**
     * Version of the Index class optimized for incremental access to Array components.
     * Stores a precomputed flat position and per-dimension step sizes, so inc() and
     * inc1() are a single addition rather than a multi-dimensional index recomputation.
     */
    class Array_iterator
    {
      protected:
        Dim_array sizes; ///< Sizes of the associated Array in each dimension.
        int* steps;      ///< Precomputed strides: steps[0]=1, steps[i+1]=steps[i]*sizes(i).
        int position;    ///< Current flat 1D index.

      public:
        /// Constructor from a Dim_array. All coordinates initialised to zero.
        explicit Array_iterator(const Dim_array& dim)
            : sizes{dim}, steps{MemoryMapper::get_memory<int>(dim.get_ndim() + 1)}, position{0}
        {
            steps[0] = 1;
            for (int i = 0; i < dim.get_ndim(); i++)
                steps[i + 1] = steps[i] * sizes(i);
        }

        /// Constructor for uniform dimensions (all axes have size tndim).
        explicit Array_iterator(int ndim, int tndim)
            : sizes{ndim}, steps{MemoryMapper::get_memory<int>(ndim + 1)}, position{0}
        {
            steps[0] = 1;
            for (int i = 0; i < ndim; i++) {
                sizes.set(i) = tndim;
                steps[i + 1] = steps[i] * tndim;
            }
        }

        /// Copy constructor.
        Array_iterator(const Array_iterator& so)
            : sizes{so.sizes}, steps{MemoryMapper::get_memory<int>(so.sizes.get_ndim() + 1)}, position{so.position}
        {
            for (int i = 0; i <= so.sizes.get_ndim(); i++)
                steps[i] = so.steps[i];
        }

        /// Destructor.
        ~Array_iterator() { MemoryMapper::release_memory<int>(steps, sizes.get_ndim() + 1); }

        int get_ndim() const { return sizes.get_ndim(); }
        int get_position() const { return position; }
        const Dim_array& get_sizes() const { return sizes; }

        bool operator==(const Array_iterator& xx) const
        {
            return (position == xx.position && get_ndim() == xx.get_ndim());
        }

        /// Directly set the flat position.
        Array_iterator& set_value(int p)
        {
            position = p;
            return *this;
        }
        /// Copy position from another iterator.
        Array_iterator& set(const Array_iterator& so)
        {
            position = so.position;
            return *this;
        }
        /// Reset to the first element.
        void set_start() { position = 0; }

        /// Returns false when position has reached or passed the total size.
        bool check_value()
        {
            if (position >= steps[sizes.get_ndim()]) {
                position = steps[sizes.get_ndim()] - 1;
                return false;
            }
            return true;
        }

        /// Increment dimension var by increm steps.
        bool inc(int increm, int var = 0)
        {
            position += increm * steps[var];
            return check_value();
        }
        /// Unit increment along dimension var.
        bool inc1(int var)
        {
            position += steps[var];
            return check_value();
        }
        /// Unit increment (dimension 0, i.e. flat increment by 1).
        bool inc()
        {
            position++;
            return check_value();
        }

        template <class> friend class Array;
    };

} // namespace Kadath
