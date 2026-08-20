/*
 * This file is part of Celephais and is distributed under the GNU General
 * Public License, version 3 or (at your option) any later version.
 */

#include <Celephais/solution.hpp>

#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string_view>

namespace
{

Celephais::SolutionKind parse_kind(const std::string_view value)
{
    using Celephais::SolutionKind;
    if (value == "ns")
        return SolutionKind::isolated_ns;
    if (value == "ns_nosym")
        return SolutionKind::isolated_ns_nosym;
    if (value == "bns")
        return SolutionKind::binary_ns;
    if (value == "bns_nosym")
        return SolutionKind::binary_ns_nosym;
    throw std::invalid_argument("unknown kind: " + std::string(value));
}

} // namespace

int main(int argc, char** argv)
{
    if (argc != 6) {
        std::cerr << "usage: solution_reader <ns|ns_nosym|bns|bns_nosym> "
                     "<solution.toml|solution.dat> <x> <y> <z>\n";
        return 2;
    }

    try {
        const auto solution = Celephais::Solution::load(argv[2], parse_kind(argv[1]));
        const Celephais::Point point{std::stod(argv[3]), std::stod(argv[4]),
                                       std::stod(argv[5])};

        std::cout << std::setprecision(17);
        for (const std::string& field : solution.field_names()) {
            std::cout << field << '=' << solution.evaluate(field, point) << '\n';
        }
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
