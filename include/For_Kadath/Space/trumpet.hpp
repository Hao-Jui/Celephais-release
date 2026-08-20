#pragma once

#include "For_Kadath/Domain/adapted.hpp"

namespace Kadath
{
    class Space_trumpet : public Space_spheric_adapted
    {
      public:
        Space_trumpet(int ttype, const Point& center, const Dim_array& res, const Array<double>& bounds);
        Space_trumpet(int ttype, const Point& center, const Dim_array& res, const std::vector<double>& bounds);
        Space_trumpet(BinarySource& source);
        ~Space_trumpet() override = default;

        using Space_spheric_adapted::add_eq;
        using Space_spheric_adapted::add_eq_int;
        using Space_spheric_adapted::add_eq_int_inf;
        using Space_spheric_adapted::add_eq_int_volume;
        using Space_spheric_adapted::add_eq_ori;

        void xx_to_vars_variable_domains(System_of_eqs* sys, const Array<double>& xx, int& pos) const override;

        void add_bc_horizon(System_of_eqs& syst, const char* eq, int nused = -1, Array<int>** pused = nullptr);
        void add_bc_inf(System_of_eqs& syst, const char* eq, int nused = -1, Array<int>** pused = nullptr);
        void add_eq_int_horizon(System_of_eqs& syst, const char* eq);
    };
} // namespace Kadath
