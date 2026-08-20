#include <catch2/catch_test_macros.hpp>

#include "For_Kadath/Base_spectral/base_spectral.hpp"

#include <array>
#include <bit>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <initializer_list>
#include <memory>
#include <string>
#include <vector>

using namespace Kadath;

namespace Kadath
{
    int der_1d(int base, Array<double>& tab);
}

namespace
{
    class ScopedEnvironmentValue
    {
      public:
        ScopedEnvironmentValue(const char* name, const char* value)
            : name_(name)
        {
            const char* old = std::getenv(name);
            if (old != nullptr) {
                had_old_ = true;
                old_value_ = old;
            }
            setenv(name, value, 1);
        }

        ~ScopedEnvironmentValue()
        {
            if (had_old_)
                setenv(name_.c_str(), old_value_.c_str(), 1);
            else
                unsetenv(name_.c_str());
        }

      private:
        std::string name_;
        std::string old_value_;
        bool had_old_ = false;
    };

    class TestSpectralBase : public Base_spectral
    {
      public:
        explicit TestSpectralBase(const Dim_array& dimensions)
            : Base_spectral(dimensions.get_ndim())
        {
            allocate(dimensions);
            def = true;
        }

        void set_axis_pattern(int axis, int first_basis)
        {
            static const std::array<int, 15> bases = {
                CHEB, CHEB_EVEN, CHEB_ODD, COSSIN, COS,
                COS_EVEN, COS_ODD, SIN, SIN_EVEN, SIN_ODD,
                LEG, LEG_EVEN, LEG_ODD, COSSIN_EVEN, COSSIN_ODD};
            Array<int>& axis_bases = *bases_1d[static_cast<std::size_t>(axis)];
            std::size_t start = 0;
            while (bases[start] != first_basis)
                ++start;
            for (std::size_t i = 0; i < axis_bases.get_nbr(); ++i)
                axis_bases.get_data()[i] = bases[(start + i) % bases.size()];
        }

        void set_axis_uniform(int axis, int basis)
        {
            Array<int>& axis_bases = *bases_1d[static_cast<std::size_t>(axis)];
            for (std::size_t i = 0; i < axis_bases.get_nbr(); ++i)
                axis_bases.get_data()[i] = basis;
        }
    };

    Dim_array make_dimensions(std::initializer_list<int> extents)
    {
        Dim_array dimensions(static_cast<int>(extents.size()));
        int axis = 0;
        for (int extent : extents)
            dimensions.set(axis++) = extent;
        return dimensions;
    }

    Dim_array derivative_test_dimensions()
    {
        return make_dimensions({4, 6, 8});
    }

    Base_spectral uniform_base(const Dim_array& dimensions, int basis)
    {
        TestSpectralBase prepared(dimensions);
        for (int axis = 0; axis < dimensions.get_ndim(); ++axis)
            prepared.set_axis_uniform(axis, basis);
        return Base_spectral(prepared);
    }

    Base_spectral varying_target_base(const Dim_array& dimensions, int var,
                                      int first_basis)
    {
        TestSpectralBase prepared(dimensions);
        for (int axis = 0; axis < dimensions.get_ndim(); ++axis) {
            if (axis == var)
                prepared.set_axis_pattern(axis, first_basis);
            else
                prepared.set_axis_uniform(axis, CHEB);
        }
        return Base_spectral(prepared);
    }

    Array<double> coefficient_fixture(const Dim_array& dimensions, int lane = 0)
    {
        Array<double> result(dimensions);
        for (std::size_t i = 0; i < result.get_nbr(); ++i) {
            const int centered = static_cast<int>((i * 17 + 3 * lane) % 29) - 14;
            result.get_data()[i] = static_cast<double>(centered) / 7.;
        }
        result.get_data()[static_cast<std::size_t>(lane) % result.get_nbr()] = -0.;
        return result;
    }

    void require_byte_equal(const Array<double>& actual,
                            const Array<double>& expected)
    {
        REQUIRE(actual.get_nbr() == expected.get_nbr());
        for (std::size_t i = 0; i < actual.get_nbr(); ++i) {
            if (std::memcmp(actual.get_data() + i, expected.get_data() + i,
                            sizeof(double)) != 0) {
                std::uint64_t actual_bits = 0;
                std::uint64_t expected_bits = 0;
                std::memcpy(&actual_bits, actual.get_data() + i, sizeof(double));
                std::memcpy(&expected_bits, expected.get_data() + i, sizeof(double));
                UNSCOPED_INFO("first mismatch flat=" << i << " actual_bits=0x"
                                                      << std::hex << actual_bits
                                                      << " expected_bits=0x"
                                                      << expected_bits);
                break;
            }
        }
        REQUIRE(std::memcmp(actual.get_data(), expected.get_data(),
                            actual.get_nbr() * sizeof(double)) == 0);
    }

    bool aosoa_test_environment_enabled()
    {
        const char* workspace = std::getenv("OPE1D_WORKSPACE");
        const char* batch = std::getenv("OPE_DER_BATCH");
        const char* aosoa = std::getenv("OPE_DER_AOSOA");
        return workspace != nullptr && std::string(workspace) == "1" &&
               batch != nullptr && std::string(batch) == "1" &&
               aosoa != nullptr && std::string(aosoa) == "1";
    }

    class DerivativeBatchFixture
    {
      public:
        DerivativeBatchFixture(int lane_count, const Dim_array& dimensions,
                               int var, int basis)
            : lane_count_(lane_count), var_(var)
        {
            input_bases_.reserve(static_cast<std::size_t>(lane_count));
            output_bases_.reserve(static_cast<std::size_t>(lane_count));
            reference_bases_.reserve(static_cast<std::size_t>(lane_count));
            inputs_.reserve(static_cast<std::size_t>(lane_count));
            outputs_.reserve(static_cast<std::size_t>(lane_count));
            references_.reserve(static_cast<std::size_t>(lane_count));
            input_base_ptrs_.reserve(static_cast<std::size_t>(lane_count));
            output_base_ptrs_.reserve(static_cast<std::size_t>(lane_count));
            input_ptrs_.reserve(static_cast<std::size_t>(lane_count));
            output_ptrs_.reserve(static_cast<std::size_t>(lane_count));

            for (int lane = 0; lane < lane_count; ++lane) {
                input_bases_.push_back(std::make_unique<Base_spectral>(
                    uniform_base(dimensions, basis)));
                output_bases_.push_back(
                    std::make_unique<Base_spectral>(*input_bases_.back()));
                reference_bases_.push_back(
                    std::make_unique<Base_spectral>(*input_bases_.back()));
                inputs_.push_back(std::make_unique<Array<double>>(
                    coefficient_fixture(dimensions, lane)));
                outputs_.push_back(
                    std::make_unique<Array<double>>(dimensions));
                references_.push_back(std::make_unique<Array<double>>(
                    input_bases_.back()->ope_1d(
                        der_1d, var_, *inputs_.back(),
                        *reference_bases_.back())));

                input_base_ptrs_.push_back(input_bases_.back().get());
                output_base_ptrs_.push_back(output_bases_.back().get());
                input_ptrs_.push_back(inputs_.back().get());
                output_ptrs_.push_back(outputs_.back().get());
            }
        }

        void apply()
        {
            Base_spectral::ope_der_1d_batch(
                var_, input_base_ptrs_.data(), input_ptrs_.data(),
                output_base_ptrs_.data(), output_ptrs_.data(), lane_count_);
        }

        void require_oracle_match() const
        {
            for (int lane = 0; lane < lane_count_; ++lane) {
                INFO("lane=" << lane);
                require_byte_equal(*outputs_[static_cast<std::size_t>(lane)],
                                   *references_[static_cast<std::size_t>(lane)]);
                REQUIRE(*output_bases_[static_cast<std::size_t>(lane)] ==
                        *reference_bases_[static_cast<std::size_t>(lane)]);
            }
        }

        std::vector<const Base_spectral*>& input_base_ptrs()
        {
            return input_base_ptrs_;
        }

        std::vector<Base_spectral*>& output_base_ptrs()
        {
            return output_base_ptrs_;
        }

        std::vector<const Array<double>*>& input_ptrs() { return input_ptrs_; }
        std::vector<Array<double>*>& output_ptrs() { return output_ptrs_; }

        void set_input_value(int lane, std::size_t flat, double value)
        {
            inputs_[static_cast<std::size_t>(lane)]->get_data()[flat] = value;
        }

        std::size_t input_size() const { return inputs_.front()->get_nbr(); }

        void refresh_oracles()
        {
            for (int lane = 0; lane < lane_count_; ++lane) {
                const std::size_t slot = static_cast<std::size_t>(lane);
                reference_bases_[slot] =
                    std::make_unique<Base_spectral>(*input_bases_[slot]);
                references_[slot] = std::make_unique<Array<double>>(
                    input_bases_[slot]->ope_1d(
                        der_1d, var_, *inputs_[slot],
                        *reference_bases_[slot]));
            }
        }

      private:
        int lane_count_ = 0;
        int var_ = 0;
        std::vector<std::unique_ptr<Base_spectral>> input_bases_;
        std::vector<std::unique_ptr<Base_spectral>> output_bases_;
        std::vector<std::unique_ptr<Base_spectral>> reference_bases_;
        std::vector<std::unique_ptr<Array<double>>> inputs_;
        std::vector<std::unique_ptr<Array<double>>> outputs_;
        std::vector<std::unique_ptr<Array<double>>> references_;
        std::vector<const Base_spectral*> input_base_ptrs_;
        std::vector<Base_spectral*> output_base_ptrs_;
        std::vector<const Array<double>*> input_ptrs_;
        std::vector<Array<double>*> output_ptrs_;
    };
}

TEST_CASE("AoSoA ope_der_1d matches scalar W1 W4 W8 W16 W32",
          "[ope-der-1d][aosoa]")
{
    if (!aosoa_test_environment_enabled())
        SKIP("AoSoA parity requires its three default-off runtime gates");

    const std::array<int, 5> widths = {1, 4, 8, 16, 32};
    const std::array<int, 10> hot_bases = {
        CHEB, CHEB_EVEN, CHEB_ODD, COSSIN, COS,
        COS_EVEN, COS_ODD, SIN, SIN_EVEN, SIN_ODD};
    const Dim_array dimensions = make_dimensions({7, 8, 6});

    for (int width : widths) {
        for (int basis : hot_bases) {
            for (int var = 0; var < dimensions.get_ndim(); ++var) {
                INFO("width=" << width << " basis=" << basis
                              << " var=" << var);
                reset_ope_der_1d_workspace();
                reset_ope_der_1d_workspace_stats();
                DerivativeBatchFixture fixture(width, dimensions, var, basis);
                {
                    OpeDer1dAssemblyWorkspaceScope scope;
                    fixture.apply();
                }
                fixture.require_oracle_match();
                const OpeDer1dWorkspaceStats stats =
                    ope_der_1d_workspace_stats();
                REQUIRE(stats.aosoa_calls == 1);
                REQUIRE(stats.aosoa_lanes ==
                        static_cast<unsigned long long>(width));
                REQUIRE(stats.aosoa_fallbacks == 0);
            }
        }
    }
}

TEST_CASE("AoSoA Chebyshev arithmetic contract is optimization independent",
          "[ope-der-1d][aosoa][arithmetic-contract]")
{
    if (!aosoa_test_environment_enabled())
        SKIP("AoSoA arithmetic parity requires its three default-off runtime gates");

    const std::array<int, 3> bases = {CHEB, CHEB_EVEN, CHEB_ODD};
    const std::array<int, 4> line_lengths = {8, 9, 16, 17};
    const std::array<int, 3> widths = {1, 4, 32};
    const std::array<std::uint64_t, 8> special_values = {
        0x0000000000000000ULL, 0x8000000000000000ULL,
        0x7ff0000000000000ULL, 0xfff0000000000000ULL,
        0x7ff8000000000001ULL, 0xfff8000000000042ULL,
        0x0000000000000001ULL, 0x8000000000000001ULL};

    for (int basis : bases) {
        for (int line_length : line_lengths) {
            for (int width : widths) {
                INFO("basis=" << basis << " line_length=" << line_length
                              << " width=" << width);
                const Dim_array dimensions = make_dimensions({line_length});
                DerivativeBatchFixture fixture(
                    width, dimensions, 0, basis);
                OpeDer1dAssemblyWorkspaceScope scope;

                for (std::uint64_t variant = 0; variant < 32; ++variant) {
                    std::uint64_t state =
                        0x9e3779b97f4a7c15ULL ^
                        (variant * 0xbf58476d1ce4e5b9ULL) ^
                        (static_cast<std::uint64_t>(basis) << 48) ^
                        (static_cast<std::uint64_t>(line_length) << 32) ^
                        static_cast<std::uint64_t>(width);
                    const auto next_bits = [&state]() {
                        state ^= state << 7;
                        state ^= state >> 9;
                        state ^= state << 8;
                        return state;
                    };

                    for (int lane = 0; lane < width; ++lane) {
                        for (std::size_t flat = 0;
                             flat < fixture.input_size(); ++flat) {
                            const std::uint64_t first = next_bits();
                            const std::uint64_t second = next_bits();
                            const std::uint64_t value_bits = variant == 0
                                ? special_values[
                                      (flat + static_cast<std::size_t>(lane)) %
                                      special_values.size()]
                                : (first & 0x8000000000000000ULL) |
                                      ((900ULL + first % 200ULL) << 52) |
                                      (second & 0x000fffffffffffffULL);
                            fixture.set_input_value(
                                lane, flat, std::bit_cast<double>(value_bits));
                        }
                    }

                    INFO("variant=" << variant);
                    fixture.refresh_oracles();
                    fixture.apply();
                    fixture.require_oracle_match();
                }
            }
        }
    }
}

TEST_CASE("AoSoA ope_der_1d falls back for unsupported valid bases",
          "[ope-der-1d][aosoa][fallback]")
{
    if (!aosoa_test_environment_enabled())
        SKIP("AoSoA fallback requires its three default-off runtime gates");

    constexpr int width = 8;
    const Dim_array dimensions = make_dimensions({7, 8, 6});
    reset_ope_der_1d_workspace();
    reset_ope_der_1d_workspace_stats();
    DerivativeBatchFixture fixture(width, dimensions, 0, LEG);
    {
        OpeDer1dAssemblyWorkspaceScope scope;
        fixture.apply();
    }
    fixture.require_oracle_match();
    const OpeDer1dWorkspaceStats stats = ope_der_1d_workspace_stats();
    REQUIRE(stats.aosoa_calls == 0);
    REQUIRE(stats.aosoa_lanes == 0);
    REQUIRE(stats.aosoa_fallbacks == 1);
    REQUIRE(stats.calls == 2);
    REQUIRE(stats.lanes == width);
}

TEST_CASE("AoSoA ope_der_1d keeps W32 alias refusal",
          "[ope-der-1d][aosoa][failure]")
{
    constexpr int width = 32;
    const Dim_array dimensions = make_dimensions({7, 8, 6});
    DerivativeBatchFixture fixture(width, dimensions, 1, CHEB);
    fixture.output_ptrs().back() =
        const_cast<Array<double>*>(fixture.input_ptrs().front());

    REQUIRE_THROWS(Base_spectral::ope_der_1d_batch(
        1, fixture.input_base_ptrs().data(), fixture.input_ptrs().data(),
        fixture.output_base_ptrs().data(), fixture.output_ptrs().data(), width));
}

TEST_CASE("workspace ope_der_1d is a bitwise oracle match for all bases",
          "[ope-der-1d]")
{
    reset_ope_der_1d_workspace();
    OpeDer1dAssemblyWorkspaceScope scope;

    const std::array<int, 15> bases = {
        CHEB, CHEB_EVEN, CHEB_ODD, COSSIN, COS,
        COS_EVEN, COS_ODD, SIN, SIN_EVEN, SIN_ODD,
        LEG, LEG_EVEN, LEG_ODD, COSSIN_EVEN, COSSIN_ODD};

    const std::array<Dim_array, 4> shapes = {
        make_dimensions({4}),
        make_dimensions({4, 6}),
        make_dimensions({4, 6, 8}),
        make_dimensions({4, 6, 8, 10})};

    for (const Dim_array& dimensions : shapes) {
        for (int basis : bases) {
            for (int var = 0; var < dimensions.get_ndim(); ++var) {
                INFO("ndim=" << dimensions.get_ndim()
                               << " basis=" << basis << " var=" << var);
                Base_spectral input_base =
                    varying_target_base(dimensions, var, basis);
                Base_spectral reference_base(input_base);
                Base_spectral workspace_base(input_base);
                Array<double> input = coefficient_fixture(dimensions, basis + var);

                Array<double> reference =
                    input_base.ope_1d(der_1d, var, input, reference_base);
                Array<double> workspace =
                    input_base.ope_der_1d(var, input, workspace_base);

                require_byte_equal(workspace, reference);
                REQUIRE(workspace_base == reference_base);
            }
        }
    }
}

TEST_CASE("ope_der_1d accepts a one-coefficient CHEB line",
          "[ope-der-1d][edge]")
{
    reset_ope_der_1d_workspace();
    OpeDer1dAssemblyWorkspaceScope scope;

    const Dim_array dimensions = make_dimensions({1});
    Base_spectral input_base = uniform_base(dimensions, CHEB);
    Base_spectral reference_base(input_base);
    Base_spectral workspace_base(input_base);
    Array<double> input = coefficient_fixture(dimensions);

    Array<double> reference =
        input_base.ope_1d(der_1d, 0, input, reference_base);
    Array<double> workspace =
        input_base.ope_der_1d(0, input, workspace_base);

    require_byte_equal(workspace, reference);
    REQUIRE(workspace_base == reference_base);
}

TEST_CASE("ope_der_1d APIs stay on the legacy path outside an assembly scope",
          "[ope-der-1d][workspace][scope]")
{
    reset_ope_der_1d_workspace();
    reset_ope_der_1d_workspace_stats();

    const Dim_array dimensions = derivative_test_dimensions();
    Base_spectral input_base = uniform_base(dimensions, CHEB);
    Array<double> input = coefficient_fixture(dimensions);

    Base_spectral reference_base(input_base);
    Array<double> reference =
        input_base.ope_1d(der_1d, 1, input, reference_base);

    Base_spectral scalar_base(input_base);
    Array<double> scalar = input_base.ope_der_1d(1, input, scalar_base);
    require_byte_equal(scalar, reference);
    REQUIRE(scalar_base == reference_base);

    Base_spectral batch_base(input_base);
    Array<double> batch(dimensions);
    const Base_spectral* batch_bases_in[1] = {&input_base};
    const Array<double>* batch_inputs[1] = {&input};
    Base_spectral* batch_bases_out[1] = {&batch_base};
    Array<double>* batch_outputs[1] = {&batch};
    Base_spectral::ope_der_1d_batch(
        1, batch_bases_in, batch_inputs, batch_bases_out, batch_outputs, 1);
    require_byte_equal(batch, reference);
    REQUIRE(batch_base == reference_base);

    const OpeDer1dWorkspaceStats stats = ope_der_1d_workspace_stats();
    REQUIRE(stats.calls == 0);
    REQUIRE(stats.lanes == 0);
    REQUIRE(stats.plan_hits == 0);
    REQUIRE(stats.plan_misses == 0);
    REQUIRE(stats.plan_fallbacks == 0);
    REQUIRE(stats.scratch_hits == 0);
    REQUIRE(stats.scratch_misses == 0);
    REQUIRE(stats.retained_bytes == 0);
    REQUIRE(stats.peak_retained_bytes == 0);
}

TEST_CASE("ope_der_1d batch matches five independent oracle lanes",
          "[ope-der-1d][batch]")
{
    reset_ope_der_1d_workspace();
    reset_ope_der_1d_workspace_stats();

    constexpr int lane_count = 5; // one full tile plus a remainder lane
    const Dim_array dimensions = derivative_test_dimensions();
    const std::array<int, lane_count> basis_codes = {
        CHEB, LEG_EVEN, COSSIN, COS_ODD, SIN_EVEN};

    std::array<std::unique_ptr<Base_spectral>, lane_count> input_bases;
    std::array<std::unique_ptr<Base_spectral>, lane_count> output_bases;
    std::array<std::unique_ptr<Base_spectral>, lane_count> reference_bases;
    std::array<std::unique_ptr<Array<double>>, lane_count> inputs;
    std::array<std::unique_ptr<Array<double>>, lane_count> outputs;
    std::array<std::unique_ptr<Array<double>>, lane_count> references;
    std::array<const Base_spectral*, lane_count> input_base_ptrs{};
    std::array<Base_spectral*, lane_count> output_base_ptrs{};
    std::array<const Array<double>*, lane_count> input_ptrs{};
    std::array<Array<double>*, lane_count> output_ptrs{};

    for (int lane = 0; lane < lane_count; ++lane) {
        input_bases[static_cast<std::size_t>(lane)] =
            std::make_unique<Base_spectral>(
                uniform_base(dimensions, basis_codes[static_cast<std::size_t>(lane)]));
        output_bases[static_cast<std::size_t>(lane)] =
            std::make_unique<Base_spectral>(*input_bases[static_cast<std::size_t>(lane)]);
        reference_bases[static_cast<std::size_t>(lane)] =
            std::make_unique<Base_spectral>(*input_bases[static_cast<std::size_t>(lane)]);
        inputs[static_cast<std::size_t>(lane)] =
            std::make_unique<Array<double>>(coefficient_fixture(dimensions, lane));
        outputs[static_cast<std::size_t>(lane)] =
            std::make_unique<Array<double>>(dimensions);
        references[static_cast<std::size_t>(lane)] = std::make_unique<Array<double>>(
            input_bases[static_cast<std::size_t>(lane)]->ope_1d(
                der_1d, 1, *inputs[static_cast<std::size_t>(lane)],
                *reference_bases[static_cast<std::size_t>(lane)]));

        input_base_ptrs[static_cast<std::size_t>(lane)] =
            input_bases[static_cast<std::size_t>(lane)].get();
        output_base_ptrs[static_cast<std::size_t>(lane)] =
            output_bases[static_cast<std::size_t>(lane)].get();
        input_ptrs[static_cast<std::size_t>(lane)] =
            inputs[static_cast<std::size_t>(lane)].get();
        output_ptrs[static_cast<std::size_t>(lane)] =
            outputs[static_cast<std::size_t>(lane)].get();
    }

    {
        OpeDer1dAssemblyWorkspaceScope scope;
        Base_spectral::ope_der_1d_batch(
            1, input_base_ptrs.data(), input_ptrs.data(), output_base_ptrs.data(),
            output_ptrs.data(), lane_count);

        const OpeDer1dWorkspaceStats stats = ope_der_1d_workspace_stats();
        REQUIRE(stats.calls == 2);
        REQUIRE(stats.lanes == lane_count);
        REQUIRE(stats.plan_misses == 1);
        REQUIRE(stats.plan_hits == 1);
        REQUIRE(stats.scratch_misses == 1);
        REQUIRE(stats.scratch_hits == 1);
    }

    for (int lane = 0; lane < lane_count; ++lane) {
        INFO("lane=" << lane);
        require_byte_equal(*outputs[static_cast<std::size_t>(lane)],
                           *references[static_cast<std::size_t>(lane)]);
        REQUIRE(*output_bases[static_cast<std::size_t>(lane)] ==
                *reference_bases[static_cast<std::size_t>(lane)]);
    }
}

TEST_CASE("ope_der_1d workspace is nest-safe and releases retained capacity",
          "[ope-der-1d][workspace]")
{
    reset_ope_der_1d_workspace();
    reset_ope_der_1d_workspace_stats();
    REQUIRE(ope_der_1d_workspace_stats().retained_bytes == 0);

    const Dim_array dimensions = derivative_test_dimensions();
    Base_spectral input_base = uniform_base(dimensions, CHEB);
    Array<double> input = coefficient_fixture(dimensions);

    {
        OpeDer1dAssemblyWorkspaceScope outer_scope;
        Base_spectral first_base(input_base);
        (void)input_base.ope_der_1d(0, input, first_base);
        REQUIRE(ope_der_1d_workspace_stats().retained_bytes > 0);

        {
            OpeDer1dAssemblyWorkspaceScope inner_scope;
            Base_spectral second_base(input_base);
            (void)input_base.ope_der_1d(0, input, second_base);
            REQUIRE(ope_der_1d_workspace_stats().plan_hits >= 1);
            REQUIRE(ope_der_1d_workspace_stats().scratch_hits >= 1);
        }
        REQUIRE(ope_der_1d_workspace_stats().retained_bytes > 0);
    }

    const OpeDer1dWorkspaceStats final_stats = ope_der_1d_workspace_stats();
    REQUIRE(final_stats.retained_bytes == 0);
    REQUIRE(final_stats.peak_retained_bytes > 0);
}

TEST_CASE("ope_der_1d workspace invalidates its one-shape plan",
          "[ope-der-1d][workspace]")
{
    reset_ope_der_1d_workspace();
    reset_ope_der_1d_workspace_stats();

    const Dim_array smaller = make_dimensions({4, 6});
    const Dim_array larger = make_dimensions({4, 6, 8});
    Base_spectral smaller_base = uniform_base(smaller, CHEB);
    Base_spectral larger_base = uniform_base(larger, CHEB);
    Array<double> smaller_input = coefficient_fixture(smaller);
    Array<double> larger_input = coefficient_fixture(larger);

    {
        OpeDer1dAssemblyWorkspaceScope scope;
        Base_spectral first_smaller_base(smaller_base);
        (void)smaller_base.ope_der_1d(0, smaller_input, first_smaller_base);

        Base_spectral larger_output_base(larger_base);
        (void)larger_base.ope_der_1d(0, larger_input, larger_output_base);

        Base_spectral second_smaller_base(smaller_base);
        (void)smaller_base.ope_der_1d(0, smaller_input, second_smaller_base);

        const OpeDer1dWorkspaceStats stats = ope_der_1d_workspace_stats();
        REQUIRE(stats.plan_misses == 3);
        REQUIRE(stats.plan_hits == 0);
        REQUIRE(stats.retained_bytes > 0);
    }

    REQUIRE(ope_der_1d_workspace_stats().retained_bytes == 0);
}

TEST_CASE("ope_der_1d falls back locally under a tiny retained-byte cap",
          "[ope-der-1d][workspace][cap]")
{
    ScopedEnvironmentValue cap("OPE1D_WORKSPACE_MAX_BYTES", "128");
    reset_ope_der_1d_workspace();
    reset_ope_der_1d_workspace_stats();

    const Dim_array dimensions = derivative_test_dimensions();
    Base_spectral input_base = uniform_base(dimensions, COS);
    Base_spectral reference_base(input_base);
    Base_spectral workspace_base(input_base);
    Array<double> input = coefficient_fixture(dimensions);
    Array<double> reference =
        input_base.ope_1d(der_1d, 1, input, reference_base);

    {
        OpeDer1dAssemblyWorkspaceScope scope;
        Array<double> workspace =
            input_base.ope_der_1d(1, input, workspace_base);
        require_byte_equal(workspace, reference);
        REQUIRE(workspace_base == reference_base);

        const OpeDer1dWorkspaceStats stats = ope_der_1d_workspace_stats();
        REQUIRE(stats.retained_byte_cap == 128);
        REQUIRE(stats.plan_fallbacks >= 1);
        REQUIRE(stats.retained_bytes <= stats.retained_byte_cap);
    }

    REQUIRE(ope_der_1d_workspace_stats().retained_bytes == 0);
}

TEST_CASE("ope_der_1d rejects malformed or overflowing workspace caps",
          "[ope-der-1d][workspace][cap]")
{
    constexpr std::size_t default_cap = 1024 * 1024;

    SECTION("integer overflow")
    {
        ScopedEnvironmentValue cap(
            "OPE1D_WORKSPACE_MAX_BYTES", "2147483648");
        reset_ope_der_1d_workspace_stats();
        REQUIRE(ope_der_1d_workspace_stats().retained_byte_cap == default_cap);
    }

    SECTION("trailing junk")
    {
        ScopedEnvironmentValue cap(
            "OPE1D_WORKSPACE_MAX_BYTES", "1048576junk");
        reset_ope_der_1d_workspace_stats();
        REQUIRE(ope_der_1d_workspace_stats().retained_byte_cap == default_cap);
    }

    SECTION("negative value")
    {
        ScopedEnvironmentValue cap(
            "OPE1D_WORKSPACE_MAX_BYTES", "-1");
        reset_ope_der_1d_workspace_stats();
        REQUIRE(ope_der_1d_workspace_stats().retained_byte_cap == default_cap);
    }
}

TEST_CASE("ope_der_1d preserves legacy-only cases and rejects invalid requests",
          "[ope-der-1d][failure]")
{
    Dim_array odd_dimensions(3);
    odd_dimensions.set(0) = 3;
    odd_dimensions.set(1) = 4;
    odd_dimensions.set(2) = 4;
    Array<double> odd_input = coefficient_fixture(odd_dimensions);
    Base_spectral odd_base = uniform_base(odd_dimensions, COSSIN_EVEN);
    Base_spectral odd_output_base(odd_base);
    // The legacy kernel leaves the final coefficient unspecified for this
    // shape. The workspace path therefore delegates the whole operation to
    // the compatibility implementation instead of defining new arithmetic or
    // turning an existing accepted call into a refusal.
    REQUIRE_NOTHROW(odd_base.ope_der_1d(0, odd_input, odd_output_base));

    const Dim_array dimensions = derivative_test_dimensions();
    Array<double> input = coefficient_fixture(dimensions);
    Base_spectral invalid_base = uniform_base(dimensions, NBR_MAX_BASE - 1);
    Base_spectral invalid_output_base(invalid_base);
    REQUIRE_THROWS(invalid_base.ope_der_1d(1, input, invalid_output_base));

    Base_spectral valid_base = uniform_base(dimensions, CHEB);
    Base_spectral valid_output_base(valid_base);
    const Base_spectral* batch_bases_in[1] = {&valid_base};
    Base_spectral* batch_bases_out[1] = {&valid_output_base};
    const Array<double>* batch_inputs[1] = {&input};
    Array<double>* aliased_outputs[1] = {&input};
    REQUIRE_THROWS(Base_spectral::ope_der_1d_batch(
        1, batch_bases_in, batch_inputs, batch_bases_out, aliased_outputs, 1));

    Array<double> separate_output(dimensions);
    Array<double>* separate_outputs[1] = {&separate_output};
    const Array<double>* null_inputs[1] = {nullptr};
    REQUIRE_THROWS(Base_spectral::ope_der_1d_batch(
        1, batch_bases_in, null_inputs, batch_bases_out, separate_outputs, 1));

    Dim_array empty_dimensions(1);
    empty_dimensions.set(0) = 0;
    Array<double> empty(empty_dimensions);
    Base_spectral empty_base(1);
    Base_spectral empty_output_base(empty_base);
    REQUIRE_THROWS(empty_base.ope_der_1d(0, empty, empty_output_base));
}
