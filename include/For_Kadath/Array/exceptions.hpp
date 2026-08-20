//
// Created by sauliac on 29/06/2020.
//

#pragma once

#include "headcpp.hpp"
#include <stdexcept>
#include <string>

namespace Kadath
{

    class KadathError : public std::runtime_error
    {
      public:
        KadathError(const char* file, int line, const std::string& msg)
            : std::runtime_error{compose(file, line, msg)}
        {
        }

      private:
        static std::string compose(const char* file, int line, const std::string& msg)
        {
            std::string out;
            out.reserve(msg.size() + 64);
            out.append(msg);
            out.append(" [");
            out.append(file ? file : "<unknown>");
            out.push_back(':');
            out.append(std::to_string(line));
            out.push_back(']');
            return out;
        }
    };

    class Unknown_base_error : public std::runtime_error
    {
      public:
        int base;
        std::string const where;
        mutable std::string explanation;
        Unknown_base_error(int b, std::string const& w) : std::runtime_error{"Unknown base"}, base{b}, where{w} {}

        const char* what() const noexcept override
        {
            explanation = "Unknown base in";
            explanation += where + " - base code = ";
            explanation += std::to_string(base);
            return explanation.c_str();
        }
    };

} // namespace Kadath

#define KADATH_THROW(msg) throw ::Kadath::KadathError(__FILE__, __LINE__, (msg))
