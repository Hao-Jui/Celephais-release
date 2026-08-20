#pragma once

#include <iomanip>
#include <sstream>
#include <string>
#include <iostream>

namespace sacra_fmt {

inline std::string eWpDe3(double val, int width, int decim)
{
    // SACRA consumes exactly e17.8e3 or e21.12e3 records.
    if (!((width == 17 && decim == 8) || (width == 21 && decim == 12))) {
        std::cerr << "sacra_fmt::eWpDe3: unsupported format (width=" << width
                  << ", decim=" << decim << "); expected (17,8) or (21,12)\n";
    }

    std::stringstream ss;
    ss << std::scientific << std::uppercase << std::setprecision(decim) << val;
    std::string s = ss.str();
    int len = static_cast<int>(s.length());

    int exp_pos = width - 4 - (val < 0 ? 0 : 1);
    std::string pad = (val < 0) ? " " : "  ";

    if (len == width - 1 - (val < 0 ? 0 : 1))
        return pad + s;
    if (len == width - 2 - (val < 0 ? 0 : 1))
        return pad + s.substr(0, exp_pos) + "0" + s.substr(exp_pos);
    if (len == width - 3 - (val < 0 ? 0 : 1))
        return pad + s.substr(0, exp_pos) + "00" + s.substr(exp_pos);

    std::cerr << "sacra_fmt::eWpDe3: unexpected string length " << len
              << " for value " << val << " (width=" << width << ")\n";
    return pad + s;
}

inline std::string iW(int val, int width)
{
    std::stringstream ss;
    ss << std::right << std::setfill(' ') << std::setw(width) << val;
    return ss.str();
}

} // namespace sacra_fmt
