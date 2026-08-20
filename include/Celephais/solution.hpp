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

#pragma once

#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace Celephais
{

/** Native GR checkpoint layouts supported by Solution::load(). */
enum class SolutionKind
{
    isolated_ns,
    isolated_ns_nosym,
    binary_ns,
    binary_ns_nosym,
};

/** Absolute Cartesian coordinates in the checkpoint's geometric units. */
struct Point
{
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
};

/**
 * An owning, read-only view of a native Celephais solution.
 *
 * The on-disk .dat stream does not contain a type discriminator, so callers
 * must provide the matching SolutionKind. The loader validates the isolated
 * versus binary TOML family before deserialisation; the symmetry variant still
 * must match exactly. The .toml and same-stem .dat files must be kept together.
 * Evaluation is currently serial: do not call methods on one or more Solution
 * objects concurrently from different threads.
 */
class Solution
{
  public:
    /** Load a solution from either its .toml path or its same-stem .dat path. */
    static Solution load(const std::filesystem::path& path, SolutionKind kind);

    Solution(Solution&&) noexcept;
    Solution& operator=(Solution&&) noexcept;
    ~Solution();

    Solution(const Solution&) = delete;
    Solution& operator=(const Solution&) = delete;

    SolutionKind kind() const noexcept;
    const std::filesystem::path& config_path() const noexcept;
    const std::filesystem::path& data_path() const noexcept;

    std::vector<std::string> field_names() const;
    bool has_field(std::string_view field) const noexcept;

    /** Evaluate one saved field at one absolute Cartesian point. */
    double evaluate(std::string_view field, Point point) const;

    /** Evaluate one saved field at a sequence of absolute Cartesian points. */
    std::vector<double> evaluate(std::string_view field,
                                 std::span<const Point> points) const;

  private:
    struct Impl;

    explicit Solution(std::unique_ptr<Impl> impl) noexcept;
    std::unique_ptr<Impl> impl_;
};

} // namespace Celephais
