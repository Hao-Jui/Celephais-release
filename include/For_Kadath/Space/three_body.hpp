/*
    Copyright 2026 Philippe Grandclement

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

#pragma once

#include "For_Kadath/Domain/bispheric_nosym.hpp"
#include "For_Kadath/Domain/spheric_adapted_nosym.hpp"
#include "For_Kadath/Domain/spheric_nosym.hpp"
#include "For_Kadath/IO/binary_sink.hpp"
#include "For_Kadath/IO/binary_source.hpp"
#include "For_Kadath/List_comp/list_comp.hpp"
#include "space.hpp"
#include <vector>

namespace Kadath
{
    /**
     * Three-body nested bispheric space.
     *
     * The layout is collinear. A parent five-domain bispheric aggregate wraps a
     * direct spherical body and the spherical exterior of a child five-domain
     * bispheric aggregate. The child aggregate wraps two spherical bodies.
     *
     * Per-body bounds use the adapted-NS layout:
     * [rin, rmid, optional outer shell bounds..., rout].
     */
    class Space_three_body : public Space
    {
      protected:
        int n_shells_body{0};
        int n_shells_child1{0};
        int n_shells_child2{0};
        int n_shells_outer{0};
        double child_origin_x{0.};

      public:
        int BODY{0};
        int ADAPTED_BODY{1};
        int CHILD1{3};
        int ADAPTED_CHILD1{4};
        int CHILD2{6};
        int ADAPTED_CHILD2{7};
        int CHILD_BI{9};
        int PARENT_BI{14};
        int OUTER{19};

        Space_three_body(int ttype, double parent_dist, double child_dist, const std::vector<double>& body_bounds,
                         const std::vector<double>& child1_bounds, const std::vector<double>& child2_bounds,
                         double child_outer_radius, const std::vector<double>& outer_bounds, int nr);

        Space_three_body(BinarySource& source);

        ~Space_three_body() override;
        void save(BinarySink& sink) const override;

        void add_eq(System_of_eqs& sys, const char* eq, const char* rac, const char* rac_der, int nused = -1,
                    Array<int>** pused = nullptr);
        void add_eq(System_of_eqs& sys, const char* eq, const char* rac, const char* rac_der,
                    const List_comp& used);

        void add_eq_int_inf(System_of_eqs& sys, const char* nom);
        void add_eq_int_volume(System_of_eqs& sys, int dmin, int dmax, const char* nom);

        void add_bc_body(System_of_eqs& sys, const char* name, int nused = -1, Array<int>** pused = nullptr);
        void add_bc_child_one(System_of_eqs& sys, const char* name, int nused = -1, Array<int>** pused = nullptr);
        void add_bc_child_two(System_of_eqs& sys, const char* name, int nused = -1, Array<int>** pused = nullptr);
        void add_bc_child_outer(System_of_eqs& sys, const char* name, int nused = -1, Array<int>** pused = nullptr);
        void add_bc_outer(System_of_eqs& sys, const char* name, int nused = -1, Array<int>** pused = nullptr);

        int nbr_unknowns_from_variable_domains() const override;
        void affecte_coef_to_variable_domains(int& pos, int cc, Array<int>& zedoms) const override;
        void xx_to_ders_variable_domains(const Array<double>& xx, int& conte) const override;
        void xx_to_vars_variable_domains(System_of_eqs* sys, const Array<double>& xx, int& pos) const override;
        Array<int> get_indices_matching_non_std(int dom, int bound) const override;

        int get_n_shells_outer() const { return n_shells_outer; }
        double get_child_origin_x() const { return child_origin_x; }

      private:
        int body_outer_domain() const { return ADAPTED_BODY + 1 + n_shells_body; }
        int child1_outer_domain() const { return ADAPTED_CHILD1 + 1 + n_shells_child1; }
        int child2_outer_domain() const { return ADAPTED_CHILD2 + 1 + n_shells_child2; }
        int outer_shell_or_compact_domain() const { return OUTER; }
        int compact_domain() const { return OUTER + n_shells_outer; }

        void set_domain_indices();
        void set_child_bispheric_origin() const;
        void update_adapted_mappings() const;
        void delete_adapted_derivatives() const;
    };
} // namespace Kadath
