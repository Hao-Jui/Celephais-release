
#pragma once

#include <vector>
#include "adapted_polar.hpp"

namespace Kadath
{

    class Domain_polar_shell_inner_homothetic : public Domain_polar_shell_inner_adapted
    {
      public:
        Domain_polar_shell_inner_homothetic(const Space& sp, int num, int ttype, double rin, double rout,
                                            const Point& cr, const Dim_array& nbr);
        Domain_polar_shell_inner_homothetic(const Domain_polar_shell_inner_homothetic& so);
        Domain_polar_shell_inner_homothetic(const Space& sp, const Domain_polar_shell_inner_homothetic& so);
        Domain_polar_shell_inner_homothetic(const Space& sp, int num, BinarySource& source); ///< Modern API.

        ~Domain_polar_shell_inner_homothetic() override;

        int nbr_unknowns_from_adapted() const override;
        void affecte_coef(int& conte, int cc, bool& found) const override;
        // un-hide the base double-bound overloads hidden by the const Val_domain& overrides below (-Woverloaded-virtual)
        using Domain::xx_to_vars_from_adapted;
        using Domain::update_constante;
        void xx_to_vars_from_adapted(Val_domain& new_inner_radius, const Array<double>& xx, int& pos) const override;
        void xx_to_ders_from_adapted(const Array<double>& xx, int& pos) const override;
        void update_constante(const Val_domain& cor_inner_radius, const Scalar& old, Scalar& res) const override;
        double integ(const Val_domain& so, int bound) const override;
        Term_eq integ_term_eq(const Term_eq& so, int bound) const override;
        ostream& print(ostream& o) const override;
    };

    class Domain_polar_shell_outer_homothetic : public Domain_polar_shell_outer_adapted
    {
      public:
        Domain_polar_shell_outer_homothetic(const Space& sp, int num, int ttype, double rin, double rout,
                                            const Point& cr, const Dim_array& nbr);
        Domain_polar_shell_outer_homothetic(const Domain_polar_shell_outer_homothetic& so);
        Domain_polar_shell_outer_homothetic(const Space& sp, const Domain_polar_shell_outer_homothetic& so);
        Domain_polar_shell_outer_homothetic(const Space& sp, int num, BinarySource& source); ///< Modern API.

        ~Domain_polar_shell_outer_homothetic() override;

        int nbr_unknowns_from_adapted() const override;
        void affecte_coef(int& conte, int cc, bool& found) const override;
        // un-hide the base double-bound overloads hidden by the const Val_domain& overrides below (-Woverloaded-virtual)
        using Domain::xx_to_vars_from_adapted;
        using Domain::update_constante;
        void xx_to_vars_from_adapted(Val_domain& new_outer_radius, const Array<double>& xx, int& pos) const override;
        void xx_to_ders_from_adapted(const Array<double>& xx, int& pos) const override;
        void update_constante(const Val_domain& cor_outer_radius, const Scalar& old, Scalar& res) const override;
        Tensor import(int numdom, int bound, int n_ope, const Array<int>& zedoms, Tensor** parts) const override;
        ostream& print(ostream& o) const override;
    };

} // namespace Kadath
