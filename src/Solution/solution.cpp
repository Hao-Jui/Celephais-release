/*
 * This file is part of Celephais.
 *
 * Celephais is free software: you can redistribute it and/or modify it under
 * the terms of the GNU General Public License as published by the Free
 * Software Foundation, either version 3 of the License, or (at your option)
 * any later version.
 *
 * Celephais is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for
 * more details.
 */

#include "Celephais/solution.hpp"

#include "For_Kadath/Array/exceptions.hpp"
#include "For_Kadath/Array/point.hpp"
#include "For_Kadath/Config/config_bco.hpp"
#include "For_Kadath/Config/config_binary.hpp"
#include "For_Kadath/Domain/adapted.hpp"
#include "For_Kadath/Domain/spheric_adapted_nosym.hpp"
#include "For_Kadath/IO/be_file_source.hpp"
#include "For_Kadath/Scalar/scalar.hpp"
#include "For_Kadath/Space/bin_ns.hpp"
#include "For_Kadath/Space/bin_ns_nosym.hpp"
#include "For_Kadath/Tensor/vector.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace Celephais
{
namespace
{

using NamedField = std::pair<std::string, const Kadath::Scalar*>;

std::filesystem::path resolve_config_path(const std::filesystem::path& input)
{
    if (input.empty()) {
        throw std::invalid_argument("Celephais::Solution: solution path is empty");
    }

    std::filesystem::path config = std::filesystem::absolute(input).lexically_normal();
    if (config.extension() == ".dat") {
        config.replace_extension(".toml");
    } else if (config.extension() != ".toml") {
        throw std::invalid_argument(
            "Celephais::Solution: expected a .toml or .dat solution path: " +
            input.string());
    }

    if (!std::filesystem::is_regular_file(config)) {
        throw std::runtime_error("Celephais::Solution: companion config not found: " +
                                 config.string());
    }
    return config;
}

void validate_data_path(const std::filesystem::path& path)
{
    if (!std::filesystem::is_regular_file(path)) {
        throw std::runtime_error("Celephais::Solution: data file not found: " + path.string());
    }
}

template <typename Config> void validate_gr_config(const Config& config)
{
    if (config.has_gravity_setting(GRAV_THEORY)) {
        const std::string theory = config.template gravity<std::string>(GRAV_THEORY);
        if (theory != "GR") {
            throw std::invalid_argument(
                "Celephais::Solution: this API currently supports GR checkpoints; got theory '" +
                theory + "'");
        }
    }
}

bool is_binary_kind(const SolutionKind kind)
{
    return kind == SolutionKind::binary_ns || kind == SolutionKind::binary_ns_nosym;
}

void validate_config_family(const std::filesystem::path& config_path, const SolutionKind kind)
{
    const ConfigTree tree = read_toml_config_tree(config_path.string());
    const bool has_binary = tree.find("binary") != tree.not_found();
    const bool has_isolated_ns = tree.find("ns") != tree.not_found();

    if (has_binary == has_isolated_ns) {
        throw std::invalid_argument(
            "Celephais::Solution: config must contain exactly one top-level 'binary' or 'ns' branch: " +
            config_path.string());
    }
    if (is_binary_kind(kind) != has_binary) {
        throw std::invalid_argument(
            "Celephais::Solution: SolutionKind does not match the checkpoint family in " +
            config_path.string());
    }
}

void validate_point(const Point point)
{
    if (!std::isfinite(point.x) || !std::isfinite(point.y) || !std::isfinite(point.z)) {
        throw std::invalid_argument("Celephais::Solution: point coordinates must be finite");
    }
}

class Dataset
{
  public:
    virtual ~Dataset() = default;

    virtual const std::filesystem::path& data_path() const noexcept = 0;
    virtual const std::vector<NamedField>& fields() const noexcept = 0;
};

void prepare_fields(const std::vector<NamedField>& fields)
{
    // Scalar::val_point() otherwise creates coefficient storage lazily on the
    // first call. Materialise it while loading so evaluation is deterministic.
    for (const auto& [name, field] : fields) {
        static_cast<void>(name);
        field->coef();
    }
}

template <typename Space> class IsolatedNsDataset final : public Dataset
{
  public:
    explicit IsolatedNsDataset(const std::filesystem::path& config_path)
        : config_(config_path.string()), data_path_(config_.space_filename()),
          source_(checked_data_path()), space_(source_), conf_(space_, source_),
          lapse_(space_, source_), shift_(space_, source_), logh_(space_, source_),
          has_phi_(config_.field(PHI) == true)
    {
        if (has_phi_) {
            phi_ = std::make_unique<Kadath::Scalar>(space_, source_);
        }
        if (shift_.get_valence() != 1 || shift_.get_index_type() != CON) {
            throw std::runtime_error(
                "Celephais::Solution: saved shift is not a contravariant vector");
        }

        fields_ = {{"conf", &conf_},         {"lapse", &lapse_},
                   {"shift_x", &shift_(1)}, {"shift_y", &shift_(2)},
                   {"shift_z", &shift_(3)}, {"logh", &logh_}};
        if (has_phi_) {
            fields_.emplace_back("phi", phi_.get());
        }
        prepare_fields(fields_);
    }

    const std::filesystem::path& data_path() const noexcept override { return data_path_; }
    const std::vector<NamedField>& fields() const noexcept override { return fields_; }

  private:
    const std::filesystem::path& checked_data_path()
    {
        validate_gr_config(config_);
        validate_data_path(data_path_);
        return data_path_;
    }

    kadath_config<BCO_NS_INFO> config_;
    std::filesystem::path data_path_;
    Kadath::BeFileSource source_;
    Space space_;
    Kadath::Scalar conf_;
    Kadath::Scalar lapse_;
    Kadath::Vector shift_;
    Kadath::Scalar logh_;
    bool has_phi_ = false;
    std::unique_ptr<Kadath::Scalar> phi_;
    std::vector<NamedField> fields_;
};

template <typename Space> class BinaryNsDataset final : public Dataset
{
  public:
    explicit BinaryNsDataset(const std::filesystem::path& config_path)
        : config_(config_path.string()), data_path_(config_.space_filename()),
          source_(checked_data_path()), space_(source_), conf_(space_, source_),
          lapse_(space_, source_), shift_(space_, source_), logh_(space_, source_),
          phi_(space_, source_)
    {
        if (shift_.get_valence() != 1 || shift_.get_index_type() != CON) {
            throw std::runtime_error(
                "Celephais::Solution: saved shift is not a contravariant vector");
        }

        fields_ = {{"conf", &conf_},         {"lapse", &lapse_},
                   {"shift_x", &shift_(1)}, {"shift_y", &shift_(2)},
                   {"shift_z", &shift_(3)}, {"logh", &logh_},
                   {"phi", &phi_}};
        prepare_fields(fields_);
    }

    const std::filesystem::path& data_path() const noexcept override { return data_path_; }
    const std::vector<NamedField>& fields() const noexcept override { return fields_; }

  private:
    const std::filesystem::path& checked_data_path()
    {
        validate_gr_config(config_);
        validate_data_path(data_path_);
        return data_path_;
    }

    kadath_config<BIN_INFO> config_;
    std::filesystem::path data_path_;
    Kadath::BeFileSource source_;
    Space space_;
    Kadath::Scalar conf_;
    Kadath::Scalar lapse_;
    Kadath::Vector shift_;
    Kadath::Scalar logh_;
    Kadath::Scalar phi_;
    std::vector<NamedField> fields_;
};

std::unique_ptr<Dataset> load_dataset(const std::filesystem::path& config_path,
                                      const SolutionKind kind)
{
    // Native .dat files have no safe family discriminator. Validate the TOML
    // branch before a space constructor interprets any untrusted binary sizes.
    validate_config_family(config_path, kind);
    switch (kind) {
        case SolutionKind::isolated_ns:
            return std::make_unique<IsolatedNsDataset<Kadath::Space_spheric_adapted>>(
                config_path);
        case SolutionKind::isolated_ns_nosym:
            return std::make_unique<IsolatedNsDataset<Kadath::Space_spheric_adapted_nosym>>(
                config_path);
        case SolutionKind::binary_ns:
            return std::make_unique<BinaryNsDataset<Kadath::Space_bin_ns>>(config_path);
        case SolutionKind::binary_ns_nosym:
            return std::make_unique<BinaryNsDataset<Kadath::Space_bin_ns_nosym>>(config_path);
    }
    throw std::invalid_argument("Celephais::Solution: unsupported solution kind");
}

const Kadath::Scalar& find_field(const Dataset& dataset, const std::string_view requested)
{
    const auto& fields = dataset.fields();
    const auto found = std::find_if(fields.begin(), fields.end(), [requested](const auto& entry) {
        return entry.first == requested;
    });
    if (found == fields.end()) {
        throw std::invalid_argument("Celephais::Solution: field not found: " +
                                    std::string(requested));
    }
    return *found->second;
}

double evaluate_scalar(const Kadath::Scalar& field, const Point point)
{
    validate_point(point);
    Kadath::Point position(3);
    position.set(1) = point.x;
    position.set(2) = point.y;
    position.set(3) = point.z;
    return field.val_point(position);
}

} // namespace

struct Solution::Impl
{
    Impl(std::filesystem::path config, const SolutionKind solution_kind)
        : kind(solution_kind), config_path(std::move(config)),
          dataset(load_dataset(config_path, kind))
    {
    }

    SolutionKind kind;
    std::filesystem::path config_path;
    std::unique_ptr<Dataset> dataset;
};

Solution Solution::load(const std::filesystem::path& path, const SolutionKind kind)
{
    return Solution(std::make_unique<Impl>(resolve_config_path(path), kind));
}

Solution::Solution(std::unique_ptr<Impl> impl) noexcept : impl_(std::move(impl)) {}
Solution::Solution(Solution&&) noexcept = default;
Solution& Solution::operator=(Solution&&) noexcept = default;
Solution::~Solution() = default;

SolutionKind Solution::kind() const noexcept { return impl_->kind; }

const std::filesystem::path& Solution::config_path() const noexcept
{
    return impl_->config_path;
}

const std::filesystem::path& Solution::data_path() const noexcept
{
    return impl_->dataset->data_path();
}

std::vector<std::string> Solution::field_names() const
{
    std::vector<std::string> names;
    names.reserve(impl_->dataset->fields().size());
    for (const auto& [name, field] : impl_->dataset->fields()) {
        static_cast<void>(field);
        names.push_back(name);
    }
    return names;
}

bool Solution::has_field(const std::string_view requested) const noexcept
{
    const auto& fields = impl_->dataset->fields();
    return std::any_of(fields.begin(), fields.end(), [requested](const auto& entry) {
        return entry.first == requested;
    });
}

double Solution::evaluate(const std::string_view field, const Point point) const
{
    return evaluate_scalar(find_field(*impl_->dataset, field), point);
}

std::vector<double> Solution::evaluate(const std::string_view field,
                                       const std::span<const Point> points) const
{
    const Kadath::Scalar& scalar = find_field(*impl_->dataset, field);
    std::vector<double> values;
    values.reserve(points.size());
    for (const Point point : points) {
        values.push_back(evaluate_scalar(scalar, point));
    }
    return values;
}

} // namespace Celephais
