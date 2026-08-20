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

#include "For_Kadath/Term_eq/term_eq.hpp"
#include "For_Kadath/Array/exceptions.hpp"
#include "For_Kadath/Scalar/scalar.hpp"
#include "For_Kadath/Space/space.hpp"
#include "term_eq_derivative_lane_layout.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <memory>
#include <utility>
namespace Kadath
{
    void affecte_one_dom(int, Tensor*, const Tensor*);

    class TermEqExtraDerivativeLanes : public MemoryMappable
    {
      public:
        std::array<std::unique_ptr<double>, Term_eq::max_derivative_lanes - 1> der_d_lanes{};
        std::array<std::unique_ptr<Tensor>, Term_eq::max_derivative_lanes - 1> der_t_lanes{};
        std::uint32_t active_lane_mask = 0;
        std::size_t allocated_lane_count = 0;
    };

    namespace
    {
        void copy_tensor_domain(int domain, Tensor& target, const Tensor& source)
        {
            affecte_one_dom(domain, &target, &source);
        }

        void validate_derivative_lane(int lane)
        {
            if (lane < 0 || lane >= Term_eq::max_derivative_lanes) {
                KADATH_THROW("Term_eq derivative lane index is outside the supported range");
            }
        }

        std::uint32_t derivative_lane_bit(int lane)
        {
            assert(lane > 0 && lane < Term_eq::max_derivative_lanes);
            return std::uint32_t{1} << static_cast<unsigned int>(lane - 1);
        }

        std::uint32_t derivative_lanes_below_count(int lane_count)
        {
            if (lane_count <= 1)
                return 0;
            return (std::uint32_t{1} << static_cast<unsigned int>(lane_count - 1)) - 1;
        }

        bool derivative_lane_is_active(const TermEqExtraDerivativeLanes* lanes, int lane)
        {
            return lanes != nullptr &&
                (lanes->active_lane_mask & derivative_lane_bit(lane)) != 0;
        }

        void activate_derivative_lane(TermEqExtraDerivativeLanes& lanes, int lane)
        {
            lanes.active_lane_mask |= derivative_lane_bit(lane);
        }

        void deactivate_derivative_lane(TermEqExtraDerivativeLanes* lanes, int lane)
        {
            if (lanes != nullptr)
                lanes->active_lane_mask &= ~derivative_lane_bit(lane);
        }

        bool reusable_tensor_domain_layout(const Tensor& target, const Tensor& result)
        {
            if (!derivative_lane_detail::same_tensor_component_layout(target, result))
                return false;
            if (!target.is_name_affected() || !result.is_name_affected())
                return true;
            for (int index = 0; index < target.get_valence(); ++index) {
                if (target.get_name_ind()[index] != result.get_name_ind()[index])
                    return false;
            }
            return true;
        }

        /**
         * True when @p result can hand its pertinent-domain storage to
         * @p target without changing anything the copying path would install:
         * identical component layout, identical index-name state and values,
         * and no tensor parameters on either side (\c affecte_one_dom does not
         * carry parameters, so a reused shell must not need them).
         */
        bool transferable_tensor_domain_layout(const Tensor& target, const Tensor& result)
        {
            if (!derivative_lane_detail::same_tensor_component_layout(target, result))
                return false;
            if (target.get_parameters() != nullptr || result.get_parameters() != nullptr)
                return false;
            if (target.is_name_affected() != result.is_name_affected())
                return false;
            if (!target.is_name_affected())
                return true;
            for (int index = 0; index < target.get_valence(); ++index) {
                if (target.get_name_ind()[index] != result.get_name_ind()[index])
                    return false;
            }
            return true;
        }

        /** Every component of both tensors owns storage in @p domain. */
        bool tensor_domains_are_materialized(int domain, const Tensor& target, const Tensor& result)
        {
            for (int component = 0; component < target.get_n_comp(); ++component) {
                const Array<int> index(target.indices(component));
                if (!target(index).has_domain_storage(domain) ||
                    !result(index).has_domain_storage(domain)) {
                    return false;
                }
            }
            return true;
        }

        void move_tensor_domain(int domain, Tensor& target, Tensor& result)
        {
            if (result.is_name_affected() && !target.is_name_affected()) {
                target.set_name_affected();
                for (int index = 0; index < target.get_valence(); ++index)
                    target.set_name_ind(index, result.get_name_ind()[index]);
            }
            for (int component = 0; component < target.get_n_comp(); ++component) {
                const Array<int> indices(target.indices(component));
                target.set(indices).set_domain(domain) =
                    std::move(result.set(indices).set_domain(domain));
            }
        }
    } // namespace

    Term_eq::Term_eq(int dd, int tipe)
        : dom(dd), val_d(nullptr), der_d(nullptr), val_t(nullptr), der_t(nullptr), n_derivative_lanes(1),
          extra_derivative_lanes(nullptr), type_data(tipe)
    {
        assert((tipe == TERM_D) || (tipe == TERM_T));
    }

    Term_eq::Term_eq(int dd, double vx)
        : dom(dd), der_d(nullptr), val_t(nullptr), der_t(nullptr), n_derivative_lanes(1),
          extra_derivative_lanes(nullptr), type_data(TERM_D)
    {

        val_d = new double(vx);
    }

    Term_eq::Term_eq(int dd, double vx, double dx)
        : dom(dd), val_t(nullptr), der_t(nullptr), n_derivative_lanes(1),
          extra_derivative_lanes(nullptr), type_data(TERM_D)
    {
        val_d = new double(vx);
        der_d = new double(dx);
    }

    Term_eq::Term_eq(int dd, const Tensor& vx)
        : dom(dd), val_d(nullptr), der_d(nullptr), der_t(nullptr), n_derivative_lanes(1),
          extra_derivative_lanes(nullptr), type_data(TERM_T)
    {
        val_t = new Tensor(one_domain_storage, dom, vx);
    }

    Term_eq::Term_eq(int dd, const Tensor& vx, const Tensor& dx)
        : dom(dd), val_d(nullptr), der_d(nullptr), n_derivative_lanes(1),
          extra_derivative_lanes(nullptr), type_data(TERM_T)
    {
        val_t = new Tensor(one_domain_storage, dom, vx);
        der_t = new Tensor(one_domain_storage, dom, dx);
    }

    Term_eq::Term_eq(const Term_eq& so)
        : dom(so.dom), val_d(nullptr), der_d(nullptr), val_t(nullptr), der_t(nullptr),
          n_derivative_lanes(so.n_derivative_lanes),
          extra_derivative_lanes(nullptr), type_data(so.type_data)
    {

        if (so.val_d != nullptr)
            val_d = new double(*so.val_d);
        if (so.der_d != nullptr)
            der_d = new double(*so.der_d);
        if (so.val_t != nullptr)
            val_t = new Tensor(one_domain_storage, dom, *so.val_t);

        if (so.der_t != nullptr)
            der_t = new Tensor(one_domain_storage, dom, *so.der_t);

        if (so.extra_derivative_lanes != nullptr) {
            for (std::size_t slot = 0; slot < so.extra_derivative_lanes->der_d_lanes.size(); ++slot) {
                const auto& source_lane = so.extra_derivative_lanes->der_d_lanes[slot];
                const int lane = static_cast<int>(slot) + 1;
                if (!so.has_der_d(lane))
                    continue;
                if (extra_derivative_lanes == nullptr)
                    extra_derivative_lanes = std::make_unique<TermEqExtraDerivativeLanes>();
                extra_derivative_lanes->der_d_lanes[slot] = std::make_unique<double>(*source_lane);
                ++extra_derivative_lanes->allocated_lane_count;
                activate_derivative_lane(*extra_derivative_lanes, lane);
            }

            for (std::size_t slot = 0; slot < so.extra_derivative_lanes->der_t_lanes.size(); ++slot) {
                const auto& source_lane = so.extra_derivative_lanes->der_t_lanes[slot];
                const int lane_index = static_cast<int>(slot) + 1;
                if (!so.has_der_t(lane_index))
                    continue;
                auto lane = std::make_unique<Tensor>(
                    one_domain_storage, dom, *source_lane);
                if (extra_derivative_lanes == nullptr)
                    extra_derivative_lanes = std::make_unique<TermEqExtraDerivativeLanes>();
                extra_derivative_lanes->der_t_lanes[slot] = std::move(lane);
                ++extra_derivative_lanes->allocated_lane_count;
                activate_derivative_lane(*extra_derivative_lanes, lane_index);
            }
        }
    }

    Term_eq::Term_eq(Term_eq&& so) noexcept
        : dom(so.dom), val_d(so.val_d), der_d(so.der_d), val_t(so.val_t), der_t(so.der_t),
          n_derivative_lanes(so.n_derivative_lanes),
          extra_derivative_lanes(std::move(so.extra_derivative_lanes)), type_data(so.type_data)
    {
        so.val_d = nullptr;
        so.der_d = nullptr;
        so.val_t = nullptr;
        so.der_t = nullptr;
    }

    Term_eq::~Term_eq()
    {

        if (val_d != nullptr)
            delete val_d;
        if (der_d != nullptr)
            delete der_d;
        if (val_t != nullptr)
            delete val_t;
        if (der_t != nullptr)
            delete der_t;
    }

    double Term_eq::get_val_d() const
    {

        if (type_data != TERM_D) {
            KADATH_THROW("Wrong type of data in Term_eq");
        }
        if (val_d == nullptr) {
            KADATH_THROW("val_d uninitialised in Term_eq");
        }
        return *val_d;
    }

    double Term_eq::get_der_d() const
    {

        if (type_data != TERM_D) {
            KADATH_THROW("Wrong type of data in Term_eq");
        }
        if (der_d == nullptr) {
            KADATH_THROW("der_d uninitialised in Term_eq");
        }
        return *der_d;
    }

    double Term_eq::get_der_d(int lane) const
    {
        validate_derivative_lane(lane);
        if (lane == 0)
            return get_der_d();
        if (type_data != TERM_D) {
            KADATH_THROW("Wrong type of data in Term_eq");
        }
        if (!has_der_d(lane)) {
            KADATH_THROW("requested der_d lane uninitialised in Term_eq");
        }
        return *extra_derivative_lanes->der_d_lanes[static_cast<std::size_t>(lane - 1)];
    }

    bool Term_eq::has_der_d(int lane) const
    {
        validate_derivative_lane(lane);
        if (type_data != TERM_D)
            return false;
        if (lane == 0)
            return der_d != nullptr;
        const bool active = derivative_lane_is_active(extra_derivative_lanes.get(), lane);
        assert(!active ||
               extra_derivative_lanes->der_d_lanes[static_cast<std::size_t>(lane - 1)] != nullptr);
        return active;
    }

    const Tensor& Term_eq::get_val_t() const
    {

        if (type_data != TERM_T) {
            KADATH_THROW("Wrong type of data in Term_eq");
        }
        if (val_t == nullptr) {
            KADATH_THROW("val_t uninitialised in Term_eq");
        }
        return *val_t;
    }

    const Tensor& Term_eq::get_der_t() const
    {

        if (type_data != TERM_T) {
            KADATH_THROW("Wrong type of data in Term_eq");
        }
        if (der_t == nullptr) {
            KADATH_THROW("der_t uninitialised in Term_eq");
        }
        return *der_t;
    }

    const Tensor& Term_eq::get_der_t(int lane) const
    {
        validate_derivative_lane(lane);
        if (lane == 0)
            return get_der_t();
        if (type_data != TERM_T) {
            KADATH_THROW("Wrong type of data in Term_eq");
        }
        if (!has_der_t(lane)) {
            KADATH_THROW("requested der_t lane uninitialised in Term_eq");
        }
        return *extra_derivative_lanes->der_t_lanes[static_cast<std::size_t>(lane - 1)];
    }

    bool Term_eq::has_der_t(int lane) const
    {
        validate_derivative_lane(lane);
        if (type_data != TERM_T)
            return false;
        if (lane == 0)
            return der_t != nullptr;
        const bool active = derivative_lane_is_active(extra_derivative_lanes.get(), lane);
        assert(!active ||
               extra_derivative_lanes->der_t_lanes[static_cast<std::size_t>(lane - 1)] != nullptr);
        return active;
    }

    const Tensor* Term_eq::get_p_der_t(int lane) const
    {
        validate_derivative_lane(lane);
        if (lane == 0)
            return der_t;
        if (!has_der_t(lane))
            return nullptr;
        return extra_derivative_lanes->der_t_lanes[static_cast<std::size_t>(lane - 1)].get();
    }

    void Term_eq::set_derivative_lane_count(int lane_count)
    {
        if (lane_count < 1 || lane_count > max_derivative_lanes) {
            KADATH_THROW("unsupported Term_eq derivative lane count");
        }
        if (extra_derivative_lanes != nullptr)
            extra_derivative_lanes->active_lane_mask &=
                derivative_lanes_below_count(lane_count);
        n_derivative_lanes = lane_count;
    }

    void Term_eq::reset_derivative_tile(int lane_count)
    {
        if (lane_count < 1 || lane_count > max_derivative_lanes) {
            KADATH_THROW("unsupported Term_eq derivative lane count");
        }
        n_derivative_lanes = lane_count;
        set_der_zero(0);
        extra_derivative_lanes.reset();
    }

    void Term_eq::reset_derivative_tile_retain_storage(int lane_count)
    {
        if (lane_count < 1 || lane_count > max_derivative_lanes) {
            KADATH_THROW("unsupported Term_eq derivative lane count");
        }
        n_derivative_lanes = lane_count;
        set_der_zero(0);
        if (extra_derivative_lanes != nullptr)
            extra_derivative_lanes->active_lane_mask = 0;
    }

    void Term_eq::operator=(const Term_eq& so)
    {

        if (this == &so)
            return;

        assert(dom == so.dom);

        if (type_data != so.type_data) {
            KADATH_THROW("Wrong type of data in Term_eq");
        }

        if (type_data == TERM_D) {
            if (so.val_d != nullptr) {
                if (val_d == nullptr)
                    val_d = new double(*so.val_d);
                else
                    *val_d = *so.val_d;
            }
            if (so.der_d != nullptr) {
                if (der_d == nullptr)
                    der_d = new double(*so.der_d);
                else
                    *der_d = *so.der_d;
            }
            set_derivative_lane_count(so.n_derivative_lanes);
            for (int lane = 1; lane < max_derivative_lanes; ++lane) {
                if (so.has_der_d(lane))
                    set_der_d(lane, so.get_der_d(lane));
                else
                    clear_der(lane);
            }
        }

        if (type_data == TERM_T) {

            if (so.val_t != nullptr) {
                if (val_t == nullptr)
                    val_t = new Tensor(one_domain_storage, dom, *so.val_t);
                else
                    affecte_one_dom(dom, val_t, so.val_t);
            }
            if (so.der_t != nullptr) {
                if (der_t == nullptr)
                    der_t = new Tensor(one_domain_storage, dom, *so.der_t);
                else
                    affecte_one_dom(dom, der_t, so.der_t);
            }
            set_derivative_lane_count(so.n_derivative_lanes);
            for (int lane = 1; lane < max_derivative_lanes; ++lane) {
                if (so.has_der_t(lane))
                    set_der_t(lane, so.get_der_t(lane));
                else
                    clear_der(lane);
            }
        }
    }

    bool Term_eq::try_write_action_result(Term_eq& result)
    {
        if (this == &result)
            return true;
        if (dom != result.dom || type_data != result.type_data)
            return false;

        if (type_data == TERM_T) {
            if (val_t != nullptr && result.val_t != nullptr &&
                !reusable_tensor_domain_layout(*val_t, *result.val_t)) {
                return false;
            }
            if (der_t != nullptr && result.der_t != nullptr &&
                !reusable_tensor_domain_layout(*der_t, *result.der_t)) {
                return false;
            }
            for (int lane = 1; lane < max_derivative_lanes; ++lane) {
                const std::size_t slot = static_cast<std::size_t>(lane - 1);
                const Tensor* target_lane = extra_derivative_lanes == nullptr
                    ? nullptr
                    : extra_derivative_lanes->der_t_lanes[slot].get();
                const Tensor* result_lane = result.get_p_der_t(lane);
                if (target_lane != nullptr && result_lane != nullptr &&
                    (target_lane->is_name_affected() ||
                     !derivative_lane_detail::same_tensor_component_layout(
                         *target_lane, *result_lane))) {
                    return false;
                }
            }
        }

        if (type_data == TERM_D) {
            if (result.val_d != nullptr) {
                if (val_d != nullptr)
                    *val_d = *result.val_d;
                else
                    std::swap(val_d, result.val_d);
            }
            if (result.der_d != nullptr) {
                if (der_d != nullptr)
                    *der_d = *result.der_d;
                else
                    std::swap(der_d, result.der_d);
            }
        } else {
            if (result.val_t != nullptr) {
                if (val_t != nullptr)
                    move_tensor_domain(dom, *val_t, *result.val_t);
                else
                    std::swap(val_t, result.val_t);
            }
            if (result.der_t != nullptr) {
                if (der_t != nullptr)
                    move_tensor_domain(dom, *der_t, *result.der_t);
                else
                    std::swap(der_t, result.der_t);
            }
        }

        set_derivative_lane_count(result.n_derivative_lanes);
        for (int lane = 1; lane < max_derivative_lanes; ++lane) {
            const std::size_t slot = static_cast<std::size_t>(lane - 1);
            if (type_data == TERM_D) {
                const bool result_has_lane = result.has_der_d(lane);
                if (!result_has_lane) {
                    deactivate_derivative_lane(extra_derivative_lanes.get(), lane);
                    continue;
                }
                if (extra_derivative_lanes != nullptr &&
                    extra_derivative_lanes->der_d_lanes[slot] != nullptr) {
                    *extra_derivative_lanes->der_d_lanes[slot] = result.get_der_d(lane);
                    activate_derivative_lane(*extra_derivative_lanes, lane);
                    continue;
                }
                if (extra_derivative_lanes == nullptr)
                    extra_derivative_lanes = std::make_unique<TermEqExtraDerivativeLanes>();
                extra_derivative_lanes->der_d_lanes[slot] =
                    std::move(result.extra_derivative_lanes->der_d_lanes[slot]);
                ++extra_derivative_lanes->allocated_lane_count;
                activate_derivative_lane(*extra_derivative_lanes, lane);
                deactivate_derivative_lane(result.extra_derivative_lanes.get(), lane);
                --result.extra_derivative_lanes->allocated_lane_count;
                if (result.extra_derivative_lanes->allocated_lane_count == 0)
                    result.extra_derivative_lanes.reset();
                continue;
            }

            const bool result_has_lane = result.has_der_t(lane);
            if (!result_has_lane) {
                deactivate_derivative_lane(extra_derivative_lanes.get(), lane);
                continue;
            }
            if (extra_derivative_lanes != nullptr &&
                extra_derivative_lanes->der_t_lanes[slot] != nullptr) {
                move_tensor_domain(dom, *extra_derivative_lanes->der_t_lanes[slot],
                                   *result.extra_derivative_lanes->der_t_lanes[slot]);
                activate_derivative_lane(*extra_derivative_lanes, lane);
                continue;
            }
            if (extra_derivative_lanes == nullptr)
                extra_derivative_lanes = std::make_unique<TermEqExtraDerivativeLanes>();
            extra_derivative_lanes->der_t_lanes[slot] =
                std::move(result.extra_derivative_lanes->der_t_lanes[slot]);
            ++extra_derivative_lanes->allocated_lane_count;
            activate_derivative_lane(*extra_derivative_lanes, lane);
            deactivate_derivative_lane(result.extra_derivative_lanes.get(), lane);
            --result.extra_derivative_lanes->allocated_lane_count;
            if (result.extra_derivative_lanes->allocated_lane_count == 0)
                result.extra_derivative_lanes.reset();
        }
        return true;
    }

    void Term_eq::set_val_d(double so)
    {
        if (type_data != TERM_D) {
            KADATH_THROW("Wrong type of data in Term_eq");
        }
        if (val_d == nullptr)
            val_d = new double(so);
        else
            *val_d = so;
    }

    void Term_eq::set_der_d(double so)
    {
        if (type_data != TERM_D) {
            KADATH_THROW("Wrong type of data in Term_eq");
        }
        if (der_d == nullptr)
            der_d = new double(so);
        else
            *der_d = so;
    }

    void Term_eq::set_der_d(int lane, double so)
    {
        validate_derivative_lane(lane);
        if (lane == 0) {
            set_der_d(so);
            return;
        }
        if (type_data != TERM_D) {
            KADATH_THROW("Wrong type of data in Term_eq");
        }
        n_derivative_lanes = std::max(n_derivative_lanes, lane + 1);
        const std::size_t slot = static_cast<std::size_t>(lane - 1);
        if (extra_derivative_lanes == nullptr)
            extra_derivative_lanes = std::make_unique<TermEqExtraDerivativeLanes>();
        auto& lane_ptr = extra_derivative_lanes->der_d_lanes[slot];
        if (lane_ptr == nullptr) {
            lane_ptr = std::make_unique<double>(so);
            ++extra_derivative_lanes->allocated_lane_count;
        }
        else
            *lane_ptr = so;
        activate_derivative_lane(*extra_derivative_lanes, lane);
    }

    void Term_eq::set_val_t(const Tensor& so)
    {
        if (type_data != TERM_T) {
            KADATH_THROW("Wrong type of data in Term_eq");
        }

        if (val_t == &so)
            return;

        // System_of_eqs::vars_to_terms repeatedly writes unnamed variables
        // with a stable tensor layout into the same per-domain Term_eq. Reuse
        // that shell; affecte_one_dom preserves the established component and
        // domain ordering while Val_domain assignment refreshes bases and
        // derivative caches. Named or parameterized tensors retain the legacy
        // reconstruction path because that metadata is externally mutable.
        const bool reusable = val_t != nullptr && !val_t->is_name_affected() &&
                              !so.is_name_affected() &&
                              val_t->get_parameters() == nullptr &&
                              so.get_parameters() == nullptr &&
                              derivative_lane_detail::same_tensor_component_layout(*val_t, so);
        if (reusable) {
            copy_tensor_domain(dom, *val_t, so);
        } else {
            delete val_t;
            val_t = new Tensor(one_domain_storage, dom, so);
        }
    }

    void Term_eq::set_der_t(const Tensor& so)
    {
        if (type_data != TERM_T) {
            KADATH_THROW("Wrong type of data in Term_eq");
        }

        if (der_t != nullptr &&
            (der_t->is_name_affected() ||
             !derivative_lane_detail::same_tensor_component_layout(*der_t, so))) {
            delete der_t;
            der_t = nullptr;
        }
        if (der_t == nullptr)
            der_t = new Tensor(one_domain_storage, dom, so, false);

        copy_tensor_domain(dom, *der_t, so);
    }

    void Term_eq::set_der_t(int lane, const Tensor& so)
    {
        validate_derivative_lane(lane);
        if (lane == 0) {
            set_der_t(so);
            return;
        }
        if (type_data != TERM_T) {
            KADATH_THROW("Wrong type of data in Term_eq");
        }

        n_derivative_lanes = std::max(n_derivative_lanes, lane + 1);
        const std::size_t slot = static_cast<std::size_t>(lane - 1);
        if (extra_derivative_lanes == nullptr)
            extra_derivative_lanes = std::make_unique<TermEqExtraDerivativeLanes>();
        auto& lane_owner = extra_derivative_lanes->der_t_lanes[slot];
        Tensor* lane_ptr = lane_owner.get();
        if (lane_ptr != nullptr &&
            (lane_ptr->is_name_affected() ||
             !derivative_lane_detail::same_tensor_component_layout(*lane_ptr, so))) {
            deactivate_derivative_lane(extra_derivative_lanes.get(), lane);
            lane_owner.reset();
            lane_ptr = nullptr;
            --extra_derivative_lanes->allocated_lane_count;
        }
        if (lane_ptr == nullptr) {
            lane_owner = std::make_unique<Tensor>(
                one_domain_storage, dom, so, false);
            lane_ptr = lane_owner.get();
            ++extra_derivative_lanes->allocated_lane_count;
        }

        copy_tensor_domain(dom, *lane_ptr, so);
        activate_derivative_lane(*extra_derivative_lanes, lane);
    }

    void Term_eq::set_der_t(int lane, Tensor&& so)
    {
        validate_derivative_lane(lane);
        // ponytail: lane 0 keeps the copying path -- it is not on any measured
        // hot route (0 of 17,076 J1 samples), so it does not need an overload.
        if (lane == 0) {
            set_der_t(so);
            return;
        }
        if (type_data != TERM_T) {
            KADATH_THROW("Wrong type of data in Term_eq");
        }

        n_derivative_lanes = std::max(n_derivative_lanes, lane + 1);
        const std::size_t slot = static_cast<std::size_t>(lane - 1);
        if (extra_derivative_lanes == nullptr)
            extra_derivative_lanes = std::make_unique<TermEqExtraDerivativeLanes>();
        auto& lane_owner = extra_derivative_lanes->der_t_lanes[slot];
        Tensor* lane_ptr = lane_owner.get();

        // A retained lane shell is reusable here whenever the transfer is
        // observationally identical to the copy; the copying overload is
        // deliberately stricter (it rebuilds on any named lane) and stays so.
        const bool reuse_lane = lane_ptr != nullptr &&
                                transferable_tensor_domain_layout(*lane_ptr, so);
        if (lane_ptr != nullptr && !reuse_lane &&
            (lane_ptr->is_name_affected() ||
             !derivative_lane_detail::same_tensor_component_layout(*lane_ptr, so))) {
            deactivate_derivative_lane(extra_derivative_lanes.get(), lane);
            lane_owner.reset();
            lane_ptr = nullptr;
            --extra_derivative_lanes->allocated_lane_count;
        }
        if (lane_ptr == nullptr) {
            lane_owner = std::make_unique<Tensor>(one_domain_storage, dom, so, false);
            lane_ptr = lane_owner.get();
            ++extra_derivative_lanes->allocated_lane_count;
        }

        // Never move-construct a Tensor from `so`: a Scalar argument arrives
        // here as a Tensor&& to its own base subobject, whose cmp[0] is a
        // self-pointer that must not be transferred. Moving one Val_domain per
        // component is safe for either shape.
        if (transferable_tensor_domain_layout(*lane_ptr, so) &&
            tensor_domains_are_materialized(dom, *lane_ptr, so)) {
            move_tensor_domain(dom, *lane_ptr, so);
        } else {
            copy_tensor_domain(dom, *lane_ptr, so);
        }
        activate_derivative_lane(*extra_derivative_lanes, lane);
    }

    bool Term_eq::try_accumulate_der_t(int lane, Tensor&& contribution)
    {
        validate_derivative_lane(lane);
        if (type_data != TERM_T)
            return false;

        Tensor* target = set_der_t(lane);
        if (target == nullptr || target == &contribution ||
            !reusable_tensor_domain_layout(*target, contribution)) {
            return false;
        }

        // Validate sparse-domain availability for every component before the
        // first write so refusal is mutation-free.
        for (int component = 0; component < target->get_n_comp(); ++component) {
            const Array<int> index(target->indices(component));
            if (!(*target)(index).has_domain_storage(dom) ||
                !contribution(index).has_domain_storage(dom)) {
                return false;
            }
        }

        for (int component = 0; component < target->get_n_comp(); ++component) {
            const Array<int> index(target->indices(component));
            Val_domain& existing = target->set(index).set_domain(dom);
            Val_domain& incoming = contribution.set(index).set_domain(dom);

            // Preserve mapping + field exactly. Avoid copying either side for
            // logical zeros, then move the already allocated incoming storage
            // into the persistent derivative lane.
            if (incoming.check_if_zero())
                continue;
            if (existing.check_if_zero()) {
                existing = std::move(incoming);
                continue;
            }
            incoming += existing;
            existing = std::move(incoming);
        }
        return true;
    }

    void Term_eq::set_der_zero()
    {
        for (int lane = 0; lane < n_derivative_lanes; ++lane)
            set_der_zero(lane);
    }

    void Term_eq::set_der_zero(int lane)
    {
        validate_derivative_lane(lane);

        switch (type_data) {
            case (TERM_D):
                set_der_d(lane, 0.);
                break;
            case (TERM_T):
                assert(val_t != nullptr);
                if (lane == 0) {
                    if (der_t == nullptr)
                        der_t = new Tensor(
                            one_domain_storage, dom, *val_t, false);
                    for (int i = 0; i < der_t->get_n_comp(); i++)
                        der_t->set(der_t->indices(i)).set_domain(dom).set_zero();
                } else {
                    n_derivative_lanes = std::max(n_derivative_lanes, lane + 1);
                    const std::size_t slot = static_cast<std::size_t>(lane - 1);
                    if (extra_derivative_lanes == nullptr)
                        extra_derivative_lanes = std::make_unique<TermEqExtraDerivativeLanes>();
                    auto& lane_owner = extra_derivative_lanes->der_t_lanes[slot];
                    if (lane_owner == nullptr) {
                        lane_owner = std::make_unique<Tensor>(
                            one_domain_storage, dom, *val_t, false);
                        ++extra_derivative_lanes->allocated_lane_count;
                    }
                    Tensor* lane_ptr = lane_owner.get();
                    for (int i = 0; i < lane_ptr->get_n_comp(); i++)
                        lane_ptr->set(lane_ptr->indices(i)).set_domain(dom).set_zero();
                    activate_derivative_lane(*extra_derivative_lanes, lane);
                }
                break;
            default:
                KADATH_THROW("Wrong type of data in Term_eq");
        }
    }

    void Term_eq::clear_der()
    {
        clear_der(0);
        extra_derivative_lanes.reset();
        n_derivative_lanes = 1;
    }

    void Term_eq::clear_der(int lane)
    {
        validate_derivative_lane(lane);

        switch (type_data) {
            case (TERM_D):
                if (lane == 0) {
                    if (der_d != nullptr) {
                        delete der_d;
                        der_d = nullptr;
                    }
                } else {
                    if (extra_derivative_lanes != nullptr) {
                        deactivate_derivative_lane(extra_derivative_lanes.get(), lane);
                        auto& slot_ptr =
                            extra_derivative_lanes->der_d_lanes[static_cast<std::size_t>(lane - 1)];
                        if (slot_ptr != nullptr) {
                            slot_ptr.reset();
                            --extra_derivative_lanes->allocated_lane_count;
                            if (extra_derivative_lanes->allocated_lane_count == 0)
                                extra_derivative_lanes.reset();
                        }
                    }
                }
                break;
            case (TERM_T):
                if (lane == 0) {
                    if (der_t != nullptr) {
                        delete der_t;
                        der_t = nullptr;
                    }
                } else {
                    if (extra_derivative_lanes != nullptr) {
                        deactivate_derivative_lane(extra_derivative_lanes.get(), lane);
                        auto& slot_ptr =
                            extra_derivative_lanes->der_t_lanes[static_cast<std::size_t>(lane - 1)];
                        if (slot_ptr != nullptr) {
                            slot_ptr.reset();
                            --extra_derivative_lanes->allocated_lane_count;
                            if (extra_derivative_lanes->allocated_lane_count == 0)
                                extra_derivative_lanes.reset();
                        }
                    }
                }
                break;
            default:
                KADATH_THROW("Wrong type of data in Term_eq");
        }
    }

    Tensor* Term_eq::set_der_t(int lane)
    {
        validate_derivative_lane(lane);
        if (lane == 0)
            return der_t;
        if (!has_der_t(lane))
            return nullptr;
        return extra_derivative_lanes->der_t_lanes[static_cast<std::size_t>(lane - 1)].get();
    }

    ostream& operator<<(ostream& flux, const Term_eq& so)
    {
        flux << "Data defined in domain = " << so.dom << endl;
        switch (so.type_data) {
            case (TERM_D):
                flux << "double data" << endl;
                if (so.val_d != nullptr)
                    flux << "val = " << *so.val_d << endl;
                else
                    flux << "val undefined" << endl;
                if (so.der_d != nullptr)
                    flux << "der = " << *so.der_d << endl;
                else
                    flux << "der undefined" << endl;
                break;
            case (TERM_T):
                flux << "tensorial data" << endl;
                if (so.val_t != nullptr)
                    flux << "val = " << *so.val_t << endl;
                else
                    flux << "val undefined" << endl;
                if (so.der_t != nullptr)
                    flux << "der = " << *so.der_t << endl;
                else
                    flux << "der undefined" << endl;
                break;
            default:
                KADATH_THROW("Unknown data type in Term_eq");
        }
        return flux;
    }
} // namespace Kadath
