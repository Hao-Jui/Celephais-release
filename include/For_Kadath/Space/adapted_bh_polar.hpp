#pragma once

#include <vector>

#include "space.hpp"
#include "For_Kadath/Term_eq/term_eq.hpp"
#include "For_Kadath/Domain/adapted_polar.hpp"
#include "For_Kadath/Domain/homothetic_polar.hpp"
#include "For_Kadath/IO/binary_sink.hpp"
#include "For_Kadath/IO/binary_source.hpp"

namespace Kadath
{

    class Space_adapted_bh_polar : public Space
    {
      private:
        int n_shells;

      public:
        int HOMOTHETIC_OUTER;
        int HOMOTHETIC_INNER;

        Space_adapted_bh_polar(int ttype, const Point& cr, const Dim_array& nbr, const Array<double>& bounds);
        Space_adapted_bh_polar(int ttype, const Point& cr, const Dim_array& nbr, const std::vector<double>& bounds);
        Space_adapted_bh_polar(const Space_adapted_bh_polar& sp);
        Space_adapted_bh_polar(BinarySource& source); ///< Modern API.
        ~Space_adapted_bh_polar() override;

        void save(BinarySink& sink) const override; ///< Modern API.

        int nbr_unknowns_from_variable_domains() const override;
        void affecte_coef_to_variable_domains(int&, int, Array<int>&) const override;
        void xx_to_ders_variable_domains(const Array<double>&, int&) const override;
        void xx_to_vars_variable_domains(System_of_eqs*, const Array<double>&, int&) const override;

        void add_bc_bh(System_of_eqs& sys, const char* name, int nused = -1, Array<int>** pused = nullptr);
        void add_bc_inf(System_of_eqs& sys, const char* name, int nused = -1, Array<int>** pused = nullptr);
        void add_eq_ori(System_of_eqs& syst, const char* eq);
        void add_eq(System_of_eqs& syst, const char* eq, const char* rac, const char* rac_der, int nused = -1,
                    Array<int>** pused = nullptr) const;
        void add_eq_free_horizon(System_of_eqs& syst, const char* eq, const char* rac, const char* rac_der,
                                 int nused = -1, Array<int>** pused = nullptr) const;
        void add_eq_inside(System_of_eqs& syst, const char* eq, int nused = -1, Array<int>** pused = nullptr) const;
        void add_eq_one_side(System_of_eqs& syst, const char* eq, int nused = -1, Array<int>** pused = nullptr) const;
        void add_eq_one_side(System_of_eqs& syst, const char* eq, const char* rac, int nused = -1,
                             Array<int>** pused = nullptr) const;
        void add_eq_int_inf(System_of_eqs& sys, const char* nom);
        void add_eq_int_volume(System_of_eqs& syst, int nz, const char* eq);
        void add_eq_int_horizon(System_of_eqs& sys, const char* nom);
        void add_eq_zero_mode_inf(System_of_eqs& sys, const char* name, int j, int k);
    };

} // namespace Kadath
