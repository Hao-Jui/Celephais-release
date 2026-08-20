#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "For_Kadath/Domain/oned.hpp"
#include "For_Kadath/Domain/polar.hpp"
#include "For_Kadath/Domain/spheric.hpp"
#include "For_Kadath/IO/memory_sink.hpp"
#include "For_Kadath/Scalar/scalar.hpp"
#include "For_Kadath/Array/memory.hpp"
#include "For_Kadath/Array/point.hpp"
#include "For_Kadath/System_of_eqs/system_of_eqs.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <type_traits>
#include <utility>

using namespace Kadath;
using Catch::Matchers::WithinAbs;

static_assert(std::is_move_constructible_v<Tensor>);
static_assert(!std::is_nothrow_move_constructible_v<Tensor>);
static_assert(std::is_constructible_v<Tensor, Scalar&&>);
static_assert(!std::is_nothrow_constructible_v<Tensor, Scalar&&>);
static_assert(sizeof(Tensor) == 208U);
static_assert(sizeof(Scalar) == 216U);

namespace {
struct PrescribedComponents {
    explicit constexpr PrescribedComponents() = default;
};

constexpr PrescribedComponents prescribed_components{};

class TensorComponentStorageProbe : public Tensor {
  public:
    using Tensor::operator=;

    explicit TensorComponentStorageProbe(const Space& space) : Tensor(space) {}

    TensorComponentStorageProbe(const Space& space, int valence, int type,
                                const Base_tensor& basis)
        : Tensor(space, valence, type, basis) {}

    TensorComponentStorageProbe(PrescribedComponents, const Space& space,
                                int component_count, const Base_tensor& basis)
        : Tensor(space, 0, Array<int>(0), component_count, basis) {}

    TensorComponentStorageProbe(OneDomainStorageTag tag, int active_domain,
                                const Space& space)
        : Tensor(tag, active_domain, space) {}

    TensorComponentStorageProbe(const Space& space, BinarySource& source)
        : Tensor(space, source) {}

    TensorComponentStorageProbe(const TensorComponentStorageProbe& source)
        : Tensor(source) {}

    TensorComponentStorageProbe(TensorComponentStorageProbe&& source)
        : Tensor(std::move(source)) {}

    [[nodiscard]] bool component_table_is_inline() {
        return cmp == &inline_cmp;
    }

    [[nodiscard]] Scalar** component_table() const { return cmp; }

    [[nodiscard]] std::ptrdiff_t component_table_offset() const {
        return reinterpret_cast<const char*>(&cmp) -
               reinterpret_cast<const char*>(this);
    }

    [[nodiscard]] std::ptrdiff_t component_count_offset() const {
        return reinterpret_cast<const char*>(&n_comp) -
               reinterpret_cast<const char*>(this);
    }

    [[nodiscard]] std::ptrdiff_t scalar_domain_storage_tag_offset() const {
        return reinterpret_cast<const char*>(&scalar_domain_storage_tag) -
               reinterpret_cast<const char*>(this);
    }

    [[nodiscard]] std::ptrdiff_t index_name_offset() const {
        return reinterpret_cast<const char*>(&name_indice) -
               reinterpret_cast<const char*>(this);
    }

    [[nodiscard]] std::ptrdiff_t inline_component_offset() const {
        return reinterpret_cast<const char*>(&inline_cmp) -
               reinterpret_cast<const char*>(this);
    }
};

class ScalarDomainStorageProbe : public Scalar {
  public:
    using Scalar::Scalar;

    ScalarDomainStorageProbe(const ScalarDomainStorageProbe& source,
                             bool copy = true)
        : Scalar(source, copy) {}

    ScalarDomainStorageProbe(ScalarDomainStorageProbe&& source) noexcept
        : Scalar(std::move(source)) {}

    [[nodiscard]] std::uint8_t storage_tag() const {
        return scalar_domain_storage_tag;
    }

    [[nodiscard]] const void* storage_word() const {
        return domain_storage_word;
    }

    [[nodiscard]] std::ptrdiff_t storage_word_offset() const {
        return reinterpret_cast<const char*>(&domain_storage_word) -
               reinterpret_cast<const char*>(this);
    }

    void copy_assign_from(const Scalar& source) {
        Scalar::operator=(source);
    }

    void tensor_assign_from(const Tensor& source) {
        Scalar::operator=(source);
    }

    void move_assign_from(Scalar&& source) {
        Scalar::operator=(std::move(source));
    }
};

Space_spheric make_space() {
    Point center(3);
    center.set(1) = 0; center.set(2) = 0; center.set(3) = 0;
    Dim_array res(3);
    res.set(0) = 13; res.set(1) = 9; res.set(2) = 8;
    Dim_array bd(1); bd.set(0) = 2;
    Array<double> bounds(bd);
    bounds.set(0) = 1.0; bounds.set(1) = 10.0;
    return Space_spheric(CHEB_TYPE, center, res, bounds);
}

Space_oned make_two_domain_oned_space() {
    Dim_array resolution(1);
    resolution.set(0) = 7;
    Array<double> bounds(1);
    bounds.set(0) = 1.0;
    return Space_oned(CHEB_TYPE, resolution, bounds);
}

Space_oned make_many_domain_oned_space(int domain_count) {
    Dim_array resolution(1);
    resolution.set(0) = 3;
    Dim_array bounds_shape(1);
    bounds_shape.set(0) = domain_count - 1;
    Array<double> bounds(bounds_shape);
    for (int bound = 0; bound < domain_count - 1; ++bound)
        bounds.set(bound) = static_cast<double>(bound + 1);
    return Space_oned(CHEB_TYPE, resolution, bounds);
}

Space_polar make_polar_space(int radial_points = 9, int angular_points = 5) {
    Point center(2);
    center.set(1) = 0.0;
    center.set(2) = 0.0;

    Dim_array resolution(2);
    resolution.set(0) = radial_points;
    resolution.set(1) = angular_points;

    Dim_array bounds_shape(1);
    bounds_shape.set(0) = 2;
    Array<double> bounds(bounds_shape);
    bounds.set(0) = 1.0;
    bounds.set(1) = 2.0;
    return Space_polar(CHEB_TYPE, center, resolution, bounds);
}

void require_arrays_exactly_equal(const Array<double>& actual, const Array<double>& expected) {
    REQUIRE(actual.get_dimensions() == expected.get_dimensions());
    REQUIRE(actual.get_nbr() == expected.get_nbr());
    for (std::size_t i = 0; i < actual.get_nbr(); ++i) {
        CAPTURE(i);
        REQUIRE(actual.get_data()[i] == expected.get_data()[i]);
    }
}
}

TEST_CASE("Constant field val_point round-trip", "[spectral]") {
    auto space = make_space();
    Scalar field(space);
    field = 42.0;
    field.std_base();
    field.coef();

    Point test_pt(3);
    test_pt.set(1) = 0.3; test_pt.set(2) = 0.4; test_pt.set(3) = 0.5;
    REQUIRE_THAT(field.val_point(test_pt), WithinAbs(42.0, 1e-10));
}

TEST_CASE("Zero-valued val_point releases its MemoryMapper slab", "[spectral][memory]") {
    auto space = make_space();
    Scalar field(space);
    field = 0.0;

    Point point(3);
    point.set(1) = 0.3;
    point.set(2) = 0.4;
    point.set(3) = 0.5;

    bool* expected_reuse = MemoryMapper::get_memory<bool>(space.get_nbr_domains());
    MemoryMapper::release_memory<bool>(expected_reuse, space.get_nbr_domains());

    for (int sample = 0; sample < 4096; ++sample)
        REQUIRE(field.val_point(point) == 0.0);

    bool* actual_reuse = MemoryMapper::get_memory<bool>(space.get_nbr_domains());
    REQUIRE(actual_reuse == expected_reuse);
    MemoryMapper::release_memory<bool>(actual_reuse, space.get_nbr_domains());
}

TEST_CASE("Scalar arithmetic preserves values", "[spectral]") {
    auto space = make_space();
    Scalar a(space), b(space);
    a = 3.0; b = 4.0;
    Scalar c = a + b;
    c.std_base();
    c.coef();

    Point test_pt(3);
    test_pt.set(1) = 0.5; test_pt.set(2) = 0.0; test_pt.set(3) = 0.0;
    REQUIRE_THAT(c.val_point(test_pt), WithinAbs(7.0, 1e-10));
}

TEST_CASE("Scalar evaluates in multiple domains", "[spectral]") {
    auto space = make_space();
    Scalar field(space);
    field = 1.0;
    field.std_base();
    field.coef();

    // Point inside nucleus (r < 1.0)
    Point p1(3);
    p1.set(1) = 0.3; p1.set(2) = 0.0; p1.set(3) = 0.0;
    REQUIRE_THAT(field.val_point(p1), WithinAbs(1.0, 1e-10));

    // Point inside shell (1.0 < r < 10.0)
    Point p2(3);
    p2.set(1) = 5.0; p2.set(2) = 0.0; p2.set(3) = 0.0;
    REQUIRE_THAT(field.val_point(p2), WithinAbs(1.0, 1e-10));
}

TEST_CASE("Scalar val_point off-axis", "[spectral]") {
    auto space = make_space();
    Scalar field(space);
    field = 2.5;
    field.std_base();
    field.coef();

    // Off-axis point (exercises angular transforms)
    Point p(3);
    p.set(1) = 0.3; p.set(2) = 0.4; p.set(3) = 0.5;
    REQUIRE_THAT(field.val_point(p), WithinAbs(2.5, 1e-10));
}

TEST_CASE("Tensor copy duplicates owned storage", "[spectral][tensor][move]") {
    auto space = make_space();
    Base_tensor basis(space, CARTESIAN_BASIS);
    Tensor source(space, 1, COV, basis);
    source.set_name_ind(0, 'i');
    source.affect_parameters();
    source.set_parameters()->set_m_quant() = 2;

    const Scalar* source_component = &source.set(1);
    const char* source_name = source.get_name_ind();
    const Param_tensor* source_parameters = source.get_parameters();
    Tensor copy(source);

    REQUIRE(&copy.set(1) != source_component);
    REQUIRE(copy.get_name_ind() != source_name);
    REQUIRE(copy.get_parameters() != source_parameters);
    REQUIRE(copy.get_name_ind()[0] == 'i');
    REQUIRE(copy.get_parameters()->get_m_quant() == 2);
}

TEST_CASE("Tensor move transfers owned storage and outlives its source", "[spectral][tensor][move]") {
    auto space = make_space();
    Base_tensor basis(space, CARTESIAN_BASIS);
    std::unique_ptr<Tensor> moved;
    const Scalar* source_component = nullptr;
    const char* source_name = nullptr;
    const Param_tensor* source_parameters = nullptr;

    {
        Tensor source(space, 1, COV, basis);
        source.set_name_ind(0, 'i');
        source.affect_parameters();
        source.set_parameters()->set_m_quant() = 2;
        source_component = &source.set(1);
        source_name = source.get_name_ind();
        source_parameters = source.get_parameters();

        moved = std::make_unique<Tensor>(std::move(source));

        REQUIRE(&moved->set(1) == source_component);
        REQUIRE(moved->get_name_ind() == source_name);
        REQUIRE(moved->get_parameters() == source_parameters);
        REQUIRE(moved->get_valence() == 1);
        REQUIRE(moved->get_n_comp() == 3);
        REQUIRE(moved->get_index_type(0) == COV);
        REQUIRE(moved->get_basis().get_basis(0) == CARTESIAN_BASIS);
        REQUIRE(source.get_n_comp() == 0);
        REQUIRE(source.get_name_ind() == nullptr);
        REQUIRE(source.get_parameters() == nullptr);
    }

    *moved = 4.0;
    moved->std_base();
    moved->coef();
    const Scalar& component = (*moved)(1);
    Point p(3);
    p.set(1) = 0.3; p.set(2) = 0.4; p.set(3) = 0.5;
    REQUIRE_THAT(component.val_point(p), WithinAbs(4.0, 1e-10));
}

TEST_CASE("Scalar-shaped Tensor moves without stealing Scalar self-ownership", "[spectral][tensor][move]") {
    auto space = make_space();
    Base_tensor basis(space, CARTESIAN_BASIS);

    Tensor scalar_shaped(space, 0, COV, basis);
    const Scalar* heap_component = &scalar_shaped.set();
    Tensor moved_tensor(std::move(scalar_shaped));
    REQUIRE(&moved_tensor.set() == heap_component);
    REQUIRE(scalar_shaped.get_n_comp() == 0);

    Scalar scalar(space);
    Tensor sliced_scalar(std::move(scalar));
    REQUIRE(&scalar.set() == &scalar);
    REQUIRE(&sliced_scalar.set() != &scalar);

    Scalar moved_scalar(std::move(scalar));
    REQUIRE(&scalar.set() == &scalar);
    REQUIRE(&moved_scalar.set() == &moved_scalar);
}

TEST_CASE("Tensor embeds a single component pointer and preserves ownership transfers",
          "[spectral][tensor][inline-storage]") {
    auto space = make_space();
    Base_tensor basis(space, CARTESIAN_BASIS);

    REQUIRE(sizeof(Tensor) == 208U);
    REQUIRE(sizeof(Scalar) == 216U);

    TensorComponentStorageProbe single(space, 0, COV, basis);
    REQUIRE(single.get_n_comp() == 1);
    REQUIRE(single.component_table_is_inline());
    REQUIRE(single.scalar_domain_storage_tag_offset() == 145);
    REQUIRE(single.component_count_offset() == 148);
    REQUIRE(single.index_name_offset() == 152);
    REQUIRE(single.inline_component_offset() == 160);
    REQUIRE(single.component_table_offset() == 168);
    Scalar* const owned_component = &single.set();

    TensorComponentStorageProbe copied(single);
    REQUIRE(copied.component_table_is_inline());
    REQUIRE(&copied.set() != owned_component);

    TensorComponentStorageProbe moved(std::move(single));
    REQUIRE(moved.component_table_is_inline());
    REQUIRE(&moved.set() == owned_component);
    REQUIRE(single.get_n_comp() == 0);
    REQUIRE(single.component_table() == nullptr);

    TensorComponentStorageProbe fallback(space, 1, COV, basis);
    REQUIRE(fallback.get_n_comp() == 3);
    REQUIRE_FALSE(fallback.component_table_is_inline());
    Scalar** const fallback_table = fallback.component_table();
    Scalar* const fallback_component = &fallback.set(1);

    TensorComponentStorageProbe fallback_copy(fallback);
    REQUIRE_FALSE(fallback_copy.component_table_is_inline());
    REQUIRE(fallback_copy.component_table() != fallback_table);
    REQUIRE(&fallback_copy.set(1) != fallback_component);

    TensorComponentStorageProbe fallback_move(std::move(fallback));
    REQUIRE_FALSE(fallback_move.component_table_is_inline());
    REQUIRE(fallback_move.component_table() == fallback_table);
    REQUIRE(&fallback_move.set(1) == fallback_component);
    REQUIRE(fallback.get_n_comp() == 0);
    REQUIRE(fallback.component_table() == nullptr);

    std::unique_ptr<TensorComponentStorageProbe> surviving_move;
    Scalar* surviving_component = nullptr;
    {
        TensorComponentStorageProbe short_lived_source(space, 0, COV, basis);
        surviving_component = &short_lived_source.set();
        surviving_move = std::make_unique<TensorComponentStorageProbe>(
            std::move(short_lived_source));
    }
    REQUIRE(surviving_move->component_table_is_inline());
    REQUIRE(&surviving_move->set() == surviving_component);
}

TEST_CASE("Scalar self-components survive copy and move with inline Tensor storage",
          "[spectral][scalar][tensor][inline-storage]") {
    auto space = make_space();
    Scalar original(space);
    REQUIRE(&original.set() == &original);

    Scalar copied(original);
    REQUIRE(&copied.set() == &copied);

    Scalar moved(std::move(copied));
    REQUIRE(&moved.set() == &moved);
    REQUIRE(&copied.set() == &copied);

    Scalar assigned(space);
    assigned = std::move(moved);
    REQUIRE(&assigned.set() == &assigned);
    REQUIRE(&moved.set() == &moved);
}

TEST_CASE("Scalar direct-domain storage preserves sparse values and representation",
          "[spectral][scalar][direct-domain-storage]") {
    auto space = make_two_domain_oned_space();
    constexpr int active_domain = 1;

    REQUIRE(sizeof(Tensor) == 208U);
    REQUIRE(sizeof(Scalar) == 216U);

    ScalarDomainStorageProbe source(one_domain_storage, active_domain, space);
    REQUIRE(source.storage_word_offset() == 208);
    REQUIRE(source.storage_tag() == active_domain + 1);
    REQUIRE_FALSE(source.has_domain_storage(0));
    REQUIRE(source.has_domain_storage(active_domain));

    Val_domain* const source_direct = &source.set_domain(active_domain);
    REQUIRE(source.storage_word() == source_direct);
    *source_direct = 3.25;
    REQUIRE(&source.set_domain(active_domain) == source_direct);
    REQUIRE(source.storage_tag() == active_domain + 1);

    ScalarDomainStorageProbe copied(source);
    REQUIRE(copied.storage_tag() == source.storage_tag());
    REQUIRE(copied.storage_word() != source.storage_word());
    MemorySink source_domain_bytes;
    source(active_domain).save(source_domain_bytes);
    MemorySink copied_domain_bytes;
    copied(active_domain).save(copied_domain_bytes);
    REQUIRE(copied_domain_bytes.buffer() == source_domain_bytes.buffer());

    ScalarDomainStorageProbe layout_copy(source, false);
    REQUIRE(layout_copy.storage_tag() == source.storage_tag());
    MemorySink layout_domain_bytes;
    layout_copy(active_domain).save(layout_domain_bytes);
    REQUIRE(layout_domain_bytes.buffer() != source_domain_bytes.buffer());

    Val_domain* const copied_direct = &copied.set_domain(active_domain);
    ScalarDomainStorageProbe moved(std::move(copied));
    REQUIRE(moved.storage_tag() == active_domain + 1);
    REQUIRE(moved.storage_word() == copied_direct);
    REQUIRE(copied.storage_tag() == 0U);
    REQUIRE(copied.storage_word() == nullptr);
    REQUIRE_FALSE(copied.has_domain_storage(0));
    REQUIRE_FALSE(copied.has_domain_storage(1));

    Val_domain& rematerialized = copied.set_domain(0);
    REQUIRE(rematerialized.get_domain() == space.get_domain(0));
    REQUIRE(copied.storage_tag() == 0U);
    REQUIRE(copied.storage_word() != nullptr);
    REQUIRE(copied.has_domain_storage(0));
    REQUIRE_FALSE(copied.has_domain_storage(1));

    ScalarDomainStorageProbe promoted(one_domain_storage, active_domain, space);
    Val_domain* const stable_direct = &promoted.set_domain(active_domain);
    stable_direct->set_zero();
    Val_domain& added = promoted.set_domain(0);
    REQUIRE(added.get_domain() == space.get_domain(0));
    REQUIRE(promoted.storage_tag() == 0U);
    REQUIRE(promoted.storage_word() != stable_direct);
    REQUIRE(&promoted.set_domain(active_domain) == stable_direct);
    REQUIRE(promoted(active_domain).check_if_zero());
    REQUIRE(promoted.has_domain_storage(0));
    REQUIRE(promoted.has_domain_storage(active_domain));

    REQUIRE_THROWS(ScalarDomainStorageProbe(one_domain_storage, -1, space));
    REQUIRE_THROWS(ScalarDomainStorageProbe(
        one_domain_storage, space.get_nbr_domains(), space));
}

TEST_CASE("Scalar direct-domain assignment covers sparse dense and empty states",
          "[spectral][scalar][direct-domain-storage][move]") {
    auto space = make_two_domain_oned_space();
    constexpr int active_domain = 1;

    ScalarDomainStorageProbe dense_source(space);
    dense_source.set_domain(0).set_zero();
    ScalarDomainStorageProbe direct_source(one_domain_storage, active_domain, space);
    direct_source.set_domain(active_domain).set_zero();

    ScalarDomainStorageProbe dense_target(space);
    const void* const dense_table = dense_target.storage_word();
    dense_target.copy_assign_from(dense_source);
    REQUIRE(dense_target.storage_tag() == 0U);
    REQUIRE(dense_target.storage_word() == dense_table);
    REQUIRE(dense_target.has_domain_storage(0));
    REQUIRE(dense_target.has_domain_storage(1));

    const void* const self_table = dense_target.storage_word();
    Val_domain* const self_domain = &dense_target.set_domain(0);
    dense_target.copy_assign_from(dense_target);
    REQUIRE(dense_target.storage_word() == self_table);
    REQUIRE(&dense_target.set_domain(0) == self_domain);
    dense_target.tensor_assign_from(static_cast<const Tensor&>(dense_target));
    REQUIRE(dense_target.storage_word() == self_table);
    REQUIRE(&dense_target.set_domain(0) == self_domain);

    dense_target.copy_assign_from(direct_source);
    REQUIRE(dense_target.storage_tag() == active_domain + 1);
    REQUIRE(dense_target.storage_word() != direct_source.storage_word());
    REQUIRE_FALSE(dense_target.has_domain_storage(0));
    REQUIRE(dense_target.has_domain_storage(active_domain));

    ScalarDomainStorageProbe direct_target(one_domain_storage, active_domain, space);
    direct_target.copy_assign_from(dense_source);
    REQUIRE(direct_target.storage_tag() == 0U);
    REQUIRE(direct_target.has_domain_storage(0));
    REQUIRE(direct_target.has_domain_storage(1));

    ScalarDomainStorageProbe second_direct(one_domain_storage, active_domain, space);
    second_direct.copy_assign_from(direct_source);
    REQUIRE(second_direct.storage_tag() == active_domain + 1);
    REQUIRE(second_direct.storage_word() != direct_source.storage_word());

    ScalarDomainStorageProbe empty_source(one_domain_storage, active_domain, space);
    ScalarDomainStorageProbe empty_source_owner(std::move(empty_source));
    REQUIRE(empty_source.storage_tag() == 0U);
    REQUIRE(empty_source.storage_word() == nullptr);

    second_direct.copy_assign_from(empty_source);
    REQUIRE(second_direct.storage_tag() == 0U);
    REQUIRE(second_direct.storage_word() == nullptr);
    second_direct.copy_assign_from(dense_source);
    REQUIRE(second_direct.storage_tag() == 0U);
    REQUIRE(second_direct.has_domain_storage(0));
    REQUIRE(second_direct.has_domain_storage(1));

    ScalarDomainStorageProbe empty_target(one_domain_storage, active_domain, space);
    ScalarDomainStorageProbe empty_target_owner(std::move(empty_target));
    empty_target.copy_assign_from(direct_source);
    REQUIRE(empty_target.storage_tag() == active_domain + 1);
    REQUIRE(empty_target.storage_word() != direct_source.storage_word());

    ScalarDomainStorageProbe move_source(one_domain_storage, active_domain, space);
    Val_domain* const moved_direct = &move_source.set_domain(active_domain);
    ScalarDomainStorageProbe move_target(space);
    const void* const moved_dense_table = move_target.storage_word();
    move_target.move_assign_from(std::move(move_source));
    REQUIRE(move_target.storage_tag() == active_domain + 1);
    REQUIRE(move_target.storage_word() == moved_direct);
    REQUIRE(move_source.storage_tag() == 0U);
    REQUIRE(move_source.storage_word() == moved_dense_table);
    REQUIRE(move_source.has_domain_storage(0));
    REQUIRE(move_source.has_domain_storage(1));
}

TEST_CASE("Scalar direct-domain storage keeps boundary and binary fallbacks",
          "[spectral][scalar][direct-domain-storage][binary]") {
    auto space = make_two_domain_oned_space();
    constexpr int active_domain = 1;

    Scalar dense(space);
    dense.set_domain(active_domain).set_zero();
    ScalarDomainStorageProbe sparse(one_domain_storage, active_domain, space);
    sparse.set_domain(active_domain).set_zero();

    MemorySink dense_bytes;
    dense.save(dense_bytes);
    MemorySink sparse_bytes;
    sparse.save(sparse_bytes);
    REQUIRE(sparse_bytes.buffer() == dense_bytes.buffer());
    REQUIRE(sparse.storage_tag() == 0U);
    REQUIRE(sparse.has_domain_storage(0));
    REQUIRE(sparse.has_domain_storage(1));

    MemorySource source(sparse_bytes.buffer());
    ScalarDomainStorageProbe restored(space, source);
    REQUIRE(restored.storage_tag() == 0U);
    MemorySink restored_bytes;
    restored.save(restored_bytes);
    REQUIRE(restored_bytes.buffer() == sparse_bytes.buffer());

    auto boundary_space = make_many_domain_oned_space(256);
    ScalarDomainStorageProbe largest_direct(
        one_domain_storage, 254, boundary_space);
    REQUIRE(largest_direct.storage_tag() == 255U);
    REQUIRE(largest_direct.has_domain_storage(254));
    REQUIRE_FALSE(largest_direct.has_domain_storage(255));

    ScalarDomainStorageProbe dense_fallback(
        one_domain_storage, 255, boundary_space);
    REQUIRE(dense_fallback.storage_tag() == 0U);
    REQUIRE_FALSE(dense_fallback.has_domain_storage(254));
    REQUIRE(dense_fallback.has_domain_storage(255));
}

TEST_CASE("Tensor component storage preserves one and fallback binary formats",
          "[spectral][tensor][inline-storage][binary]") {
    auto space = make_space();
    Base_tensor basis(space, CARTESIAN_BASIS);

    TensorComponentStorageProbe single(space, 0, COV, basis);
    single = 3.25;
    single.std_base();
    MemorySink single_sink;
    single.save(single_sink);
    MemorySource single_source(single_sink.buffer());
    TensorComponentStorageProbe single_restored(space, single_source);
    REQUIRE(single_restored.component_table_is_inline());
    MemorySink single_resaved;
    single_restored.save(single_resaved);
    REQUIRE(single_resaved.buffer() == single_sink.buffer());

    TensorComponentStorageProbe fallback(space, 1, COV, basis);
    fallback = -2.5;
    fallback.std_base();
    MemorySink fallback_sink;
    fallback.save(fallback_sink);
    MemorySource fallback_source(fallback_sink.buffer());
    TensorComponentStorageProbe fallback_restored(space, fallback_source);
    REQUIRE_FALSE(fallback_restored.component_table_is_inline());
    MemorySink fallback_resaved;
    fallback_restored.save(fallback_resaved);
    REQUIRE(fallback_resaved.buffer() == fallback_sink.buffer());

    REQUIRE_THROWS(TensorComponentStorageProbe(one_domain_storage, -1, space));
    REQUIRE_THROWS(TensorComponentStorageProbe(
        one_domain_storage, space.get_nbr_domains(), space));
}

TEST_CASE("Tensor single-component pointer storage bypasses MemoryMapper",
          "[memory][memory-phase-profile][tensor][inline-storage]") {
    if (!MemoryMapper::phase_profile_enabled())
        SKIP("requires MEMORY_MAPPER_PHASE_PROFILE=1 at process start");

    auto space = make_two_domain_oned_space();
    Base_tensor basis(space, CARTESIAN_BASIS);

    MemoryMapperPhaseSnapshot before = MemoryMapper::phase_snapshot();
    {
        TensorComponentStorageProbe single_table(space);
        REQUIRE(single_table.component_table_is_inline());
    }
    MemoryMapperPhaseSnapshot after = MemoryMapper::phase_snapshot();
    REQUIRE(after.get_calls - before.get_calls == 0U);
    REQUIRE(after.release_calls - before.release_calls == 0U);

    before = MemoryMapper::phase_snapshot();
    {
        TensorComponentStorageProbe ordinary_single(space, 0, COV, basis);
    }
    after = MemoryMapper::phase_snapshot();
    const std::uint64_t ordinary_single_gets = after.get_calls - before.get_calls;
    const std::uint64_t ordinary_single_releases =
        after.release_calls - before.release_calls;

    before = MemoryMapper::phase_snapshot();
    {
        auto scalar = std::make_unique<Scalar>(space);
    }
    after = MemoryMapper::phase_snapshot();
    REQUIRE(after.get_calls - before.get_calls == ordinary_single_gets);
    REQUIRE(after.release_calls - before.release_calls ==
            ordinary_single_releases);

    before = MemoryMapper::phase_snapshot();
    {
        TensorComponentStorageProbe fallback(
            prescribed_components, space, 3, basis);
    }
    after = MemoryMapper::phase_snapshot();
    const std::uint64_t fallback_gets = after.get_calls - before.get_calls;
    const std::uint64_t fallback_releases = after.release_calls - before.release_calls;

    before = MemoryMapper::phase_snapshot();
    {
        TensorComponentStorageProbe first(space, 0, COV, basis);
        TensorComponentStorageProbe second(space, 0, COV, basis);
        TensorComponentStorageProbe third(space, 0, COV, basis);
    }
    after = MemoryMapper::phase_snapshot();
    REQUIRE(fallback_gets == after.get_calls - before.get_calls + 1U);
    REQUIRE(fallback_releases == after.release_calls - before.release_calls + 1U);
}

TEST_CASE("Scalar direct-domain storage removes the sparse pointer table",
          "[memory][memory-phase-profile][scalar][direct-domain-storage]") {
    if (!MemoryMapper::phase_profile_enabled())
        SKIP("requires MEMORY_MAPPER_PHASE_PROFILE=1 at process start");

    auto direct_space = make_two_domain_oned_space();
    auto fallback_space = make_many_domain_oned_space(256);

    MemoryMapperPhaseSnapshot before = MemoryMapper::phase_snapshot();
    {
        ScalarDomainStorageProbe direct(one_domain_storage, 1, direct_space);
        REQUIRE(direct.storage_tag() == 2U);
        REQUIRE(direct.has_domain_storage(1));
        REQUIRE(direct.storage_word() == &direct.set_domain(1));
    }
    MemoryMapperPhaseSnapshot after = MemoryMapper::phase_snapshot();
    const std::uint64_t direct_gets = after.get_calls - before.get_calls;
    const std::uint64_t direct_releases =
        after.release_calls - before.release_calls;

    before = MemoryMapper::phase_snapshot();
    {
        ScalarDomainStorageProbe boundary_direct(
            one_domain_storage, 254, fallback_space);
        REQUIRE(boundary_direct.storage_tag() == 255U);
        REQUIRE(boundary_direct.has_domain_storage(254));
    }
    after = MemoryMapper::phase_snapshot();
    const std::uint64_t boundary_direct_gets =
        after.get_calls - before.get_calls;
    const std::uint64_t boundary_direct_releases =
        after.release_calls - before.release_calls;

    before = MemoryMapper::phase_snapshot();
    {
        ScalarDomainStorageProbe fallback(
            one_domain_storage, 255, fallback_space);
        REQUIRE(fallback.storage_tag() == 0U);
        REQUIRE(fallback.has_domain_storage(255));
    }
    after = MemoryMapper::phase_snapshot();
    const std::uint64_t fallback_gets = after.get_calls - before.get_calls;
    const std::uint64_t fallback_releases =
        after.release_calls - before.release_calls;

    REQUIRE(direct_gets == 1U);
    REQUIRE(direct_releases == 1U);
    REQUIRE(boundary_direct_gets == 2U);
    REQUIRE(boundary_direct_releases == 2U);
    REQUIRE(fallback_gets == boundary_direct_gets + 1U);
    REQUIRE(fallback_releases == boundary_direct_releases + 1U);
}

TEST_CASE("Polar transforms preserve dimension order at the smallest supported shape",
          "[spectral][polar][parity]") {
    // Polar collocation divides by (n - 1), and the angular parity transform
    // is inactive below five points. Zero/one-sized dimensions therefore do
    // not have a safe Polar-domain construction contract; 3x5 is the smallest
    // useful shape supported by the current transform implementations.
    auto space = make_polar_space(3, 5);
    const Domain* domain = space.get_domain(1);
    const Dim_array points = domain->get_nbr_points();
    const Dim_array coefficients = domain->get_nbr_coefs();

    Val_domain sample(domain);
    sample.allocate_conf();
    Index position(points);
    do {
        sample.set(position) = 1.0 + 0.25 * position(0) - 0.125 * position(1)
                               + 0.03125 * position(0) * position(1);
    } while (position.inc());
    sample.std_base();

    const Array<double> configuration = sample.get_conf();
    const Array<double> configuration_snapshot(configuration);
    const Base_spectral& base = sample.get_base();

    Array<double> expected_coefficients =
        base.coef_dim(1, coefficients(1), configuration);
    expected_coefficients =
        base.coef_dim(0, coefficients(0), expected_coefficients);
    const Array<double> actual_coefficients =
        base.coef(coefficients, configuration);

    require_arrays_exactly_equal(actual_coefficients, expected_coefficients);
    require_arrays_exactly_equal(configuration, configuration_snapshot);

    const Array<double> coefficient_snapshot(actual_coefficients);
    Array<double> expected_configuration =
        base.coef_i_dim(0, points(0), actual_coefficients);
    expected_configuration =
        base.coef_i_dim(1, points(1), expected_configuration);
    const Array<double> actual_configuration =
        base.coef_i(points, actual_coefficients);

    require_arrays_exactly_equal(actual_configuration, expected_configuration);
    require_arrays_exactly_equal(actual_coefficients, coefficient_snapshot);
}

TEST_CASE("Equal-shape 3D transforms match explicit dimension sequencing",
          "[spectral][transform][parity]") {
    Dim_array shape(3);
    shape.set(0) = 5;
    shape.set(1) = 5;
    shape.set(2) = 5;

    Base_spectral base(3);
    base.set(shape, CHEB, CHEB, CHEB);

    Array<double> configuration(shape);
    Index position(shape);
    do {
        configuration.set(position) =
            1.0 + 0.25 * position(0) - 0.125 * position(1)
            + 0.0625 * position(2) + 0.03125 * position(0) * position(2);
    } while (position.inc());
    const Array<double> configuration_snapshot(configuration);

    Array<double> expected_coefficients = base.coef_dim(2, shape(2), configuration);
    expected_coefficients = base.coef_dim(1, shape(1), expected_coefficients);
    expected_coefficients = base.coef_dim(0, shape(0), expected_coefficients);
    const Array<double> actual_coefficients = base.coef(shape, configuration);

    require_arrays_exactly_equal(actual_coefficients, expected_coefficients);
    require_arrays_exactly_equal(configuration, configuration_snapshot);

    const Array<double> coefficient_snapshot(actual_coefficients);
    Array<double> expected_configuration = base.coef_i_dim(0, shape(0), actual_coefficients);
    expected_configuration = base.coef_i_dim(1, shape(1), expected_configuration);
    expected_configuration = base.coef_i_dim(2, shape(2), expected_configuration);
    const Array<double> actual_configuration = base.coef_i(shape, actual_coefficients);

    require_arrays_exactly_equal(actual_configuration, expected_configuration);
    require_arrays_exactly_equal(actual_coefficients, coefficient_snapshot);
}

TEST_CASE("Polar volume integration preserves source coefficients",
          "[spectral][polar][integration][parity]") {
    auto space = make_polar_space();
    Scalar one(space);
    one = 1.0;
    one.std_base();
    one.coef();

    SECTION("nucleus") {
        const Array<double> before(one(0).get_coef());
        REQUIRE_THAT(one(0).integ_volume(), WithinAbs(2.0 / 3.0, 1e-12));
        require_arrays_exactly_equal(one(0).get_coef_ref(), before);
    }

    SECTION("shell") {
        const Array<double> before(one(1).get_coef());
        REQUIRE_THAT(one(1).integ_volume(), WithinAbs(1.5 * M_PI, 1e-12));
        require_arrays_exactly_equal(one(1).get_coef_ref(), before);
    }

    SECTION("compact") {
        Val_domain inverse_r4(one(2));
        for (int power = 0; power < 4; ++power)
            inverse_r4 = inverse_r4.div_r();
        inverse_r4.coef();
        const Array<double> before(inverse_r4.get_coef());

        REQUIRE_THAT(inverse_r4.integ_volume(), WithinAbs(M_PI / 8.0, 1e-12));
        require_arrays_exactly_equal(inverse_r4.get_coef_ref(), before);
    }
}

TEST_CASE("Polar point equation matches scalar point evaluation without mutating coefficients",
          "[spectral][polar][point][parity]") {
    auto space = make_polar_space();
    Scalar field(space);
    field = 2.5;
    field.std_base();
    field.coef();

    Point point(2);
    point.set(1) = 0.9;
    point.set(2) = 1.2;
    const Array<double> before(field(1).get_coef());
    const double expected = field.val_point(point);

    System_of_eqs system(space, 0, space.get_nbr_domains() - 1);
    system.add_var("u", field);
    space.add_eq_point(system, point, "u");
    const Array<double> residual = system.sec_member();

    REQUIRE(residual.get_nbr() == 1);
    REQUIRE_THAT(residual(0), WithinAbs(expected, 1e-13));
    require_arrays_exactly_equal(field(1).get_coef_ref(), before);
}
