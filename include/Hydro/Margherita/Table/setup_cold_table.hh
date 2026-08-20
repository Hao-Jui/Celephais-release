#pragma once

#include <string>

namespace Kadath {
namespace Margherita {

void setup_Cold_Table(std::string cold_table_name, int cold_lintp_points,
                      double h_cut = 0.0, double mnuc_cgs = 0.0);

}
}
