/*
    Copyright 2019 Samuel Tootle

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

#include "For_Kadath/Config/config_bco.hpp"
#include "For_Kadath/Config/config_binary.hpp"
#include "For_Kadath/Config/config_three_body.hpp"

namespace
{
std::string config_title(const std::string& label)
{
    int n = ((42 - label.size()) > 0) ? 42 - label.size() : label.size() - 42;
    n /= 2;
    return std::string(n, '*') + label + std::string(n, '*');
}

} // namespace

void print_config_bco_info_with_label(std::ostream& out, const BCO_INFO& BCO, const std::string& label)
{
    out << config_title(label) << std::endl;
    print_params(BCO.bco_map, BCO.bco_params, out);
    if (auto ns_ptr = dynamic_cast<const BCO_NS_INFO*>(&BCO)) {
        print_params(ns_ptr->return_eos_map(), ns_ptr->return_eos_params(), out);
        print_params(ns_ptr->return_diffrot_map(), ns_ptr->return_diffrot_params(), out);
    }
}

std::ostream &operator<<(std::ostream &out, const BCO_INFO &BCO)
{
    print_config_bco_info_with_label(out, BCO, BCO.get_type() + " info");
    return out;
}

std::ostream &operator<<(std::ostream &out, const BIN_INFO &BIN)
{
    out << config_title(BIN.get_type() + " info") << std::endl;
    print_params(BIN.bin_map, BIN.bin_params, out);
    print_config_bco_info_with_label(out, *BIN.BCOS[0], BIN.BCOS[0]->get_type() + "1 info");
    print_config_bco_info_with_label(out, *BIN.BCOS[1], BIN.BCOS[1]->get_type() + "2 info");
    return out;
}

std::ostream &operator<<(std::ostream &out, const TRI_INFO &TRI)
{
    out << config_title(TRI.get_type() + " info") << std::endl;
    print_params(TRI.tri_map, TRI.tri_params, out);
    for (int i = 0; i < 3; ++i)
        print_config_bco_info_with_label(out, *TRI.BCOS[i],
                                         TRI.BCOS[i]->get_type() + std::to_string(i + 1) + " info");
    return out;
}
