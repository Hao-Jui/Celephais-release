#pragma once

#include <vector>

#include "space.hpp"
#include "For_Kadath/Term_eq/term_eq.hpp"
#include "For_Kadath/Domain/spheric_nosym.hpp"
#include "For_Kadath/Metric/metric.hpp"
#include "For_Kadath/Domain/spheric_adapted_nosym.hpp"
#include "For_Kadath/Domain/homothetic_nosym.hpp"
#include "For_Kadath/IO/binary_sink.hpp"
#include "For_Kadath/IO/binary_source.hpp"

namespace Kadath
{

    class Space_adapted_bh_nosym : public Space
    {

      public:
        Space_adapted_bh_nosym(int ttype, const Point& cr, const Dim_array& nbr, const std::vector<double>& BH_bounds);

        Space_adapted_bh_nosym(BinarySource& source); ///< Modern API.

        ~Space_adapted_bh_nosym() override;

        void add_eq(System_of_eqs& syst, const char* eq, const char* rac, const char* rac_der, int nused = -1,
                    Array<int>** pused = nullptr);

        void add_bc_bh(System_of_eqs& syst, const char* eq, int nused = -1, Array<int>** pused = nullptr);

        void add_bc_inf(System_of_eqs& syst, const char* eq, int nused = -1, Array<int>** pused = nullptr);

        void add_eq_int_inf(System_of_eqs& syst, const char* eq);

        void add_eq_int_bh(System_of_eqs& syst, const char* eq);

        void add_eq_zero_mode_inf(System_of_eqs& syst, const char* eq, int l_mode, int m_mode);

        void save(BinarySink& sink) const override; ///< Modern API.

        int nbr_unknowns_from_variable_domains() const override;

        void affecte_coef_to_variable_domains(int& position, int start_index, Array<int>& affected_doms) const override;

        void xx_to_ders_variable_domains(const Array<double>& solution_array, int& position) const override;

        void xx_to_vars_variable_domains(System_of_eqs* system, const Array<double>& solution_array,
                                                 int& position) const override;

        Array<int> get_indices_matching_non_std(int dom_index, int bound_index) const override;
    };

} // namespace Kadath
