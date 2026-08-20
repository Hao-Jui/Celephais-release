/*
    Copyright 2018 Philippe Grandclement

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

/*
 * Modifications (Celephais):
 *   2026-06-16  Modified for the Celephais tree; see
 *               PATCHES-KADATH-UPSTREAM.md and LICENSE_SOURCE_AUDIT.tsv.
 *   2026-08-06  RAII/span modernization.
 */

#include "For_Kadath/System_of_eqs/system_of_eqs.hpp"
#include "For_Kadath/Ope_eq/ope_eq.hpp"
#include "For_Kadath/Term_eq/term_eq.hpp"
#include "For_Kadath/Scalar/scalar.hpp"
#include "For_Kadath/Utilities/name_tools.hpp"
namespace Kadath
{
    namespace
    {
        int count_exported_tau_rows(const Space& space, const Scalar& residual, int domain)
        {
            const Dim_array nbr_coefs = space.get_domain(domain)->get_nbr_coefs();
            int capacity = 1;
            for (int dim = 0; dim < nbr_coefs.get_ndim(); ++dim)
                capacity *= nbr_coefs(dim);

            Array<double> scratch(capacity);
            scratch = 0.0;
            Array<int> dry_ncond(1);
            dry_ncond.set(0) = capacity;
            int pos = 0;
            space.get_domain(domain)->export_tau(residual, domain, 0, scratch, pos, dry_ncond);
            return pos;
        }
    } // namespace

    Eq_first_integral::Eq_first_integral(const System_of_eqs* syst, const Domain* innerdom, int dommin, int dommax,
                                         const char* integ_part, const char* const_part,
                                         bool same_sector)
        : Equation(innerdom, dommin, dommax - dommin + 2), dom_min(dommin),
          dom_max(dommax), same_reflection_sector(same_sector)
    {

        // First the constant part
        char auxicst[LMAX];
        trim_spaces(auxicst, const_part);
        parts[0].reset(syst->give_ope(dom_min, auxicst));

        // First all the integral parts
        char auxiinteg[LMAX];
        trim_spaces(auxiinteg, integ_part);

        for (int d = dom_min; d <= dom_max; d++)
            parts[d - dom_min + 1].reset(syst->give_ope(d, auxiinteg));
    }

    Eq_first_integral::~Eq_first_integral() {}

    void Eq_first_integral::apply(int& conte, Term_eq** res)
    {
        const int old_conte = conte;
        for (int i = 0; i < n_ope; i++) {
            if (res[conte] != nullptr)
                *res[conte] = parts[i]->action();
            else
                res[conte] = new Term_eq(parts[i]->action());
            conte++;
        }

        if (!called) {
            const Tensor& constant_value = *res[old_conte]->get_p_val_t();
            const Space& space = constant_value.get_space();
            Index pori(space.get_domain(dom_min)->get_nbr_points());

            const Tensor& inner_value = *res[old_conte + 1]->get_p_val_t();
            const Val_domain& inner_domain_value = inner_value()(dom_min);
            const double val_ori = inner_domain_value.check_if_zero() ? 0.0 : inner_domain_value(pori);

            Array<int> counts(dom_max - dom_min + 1);

            Scalar auximin(space);
            auximin.set_domain(dom_min) = inner_domain_value - val_ori;
            counts.set(0) = count_exported_tau_rows(space, auximin, dom_min);

            for (int d = dom_min + 1; d <= dom_max; d++) {
                const Tensor& domain_value = *res[old_conte + d - dom_min + 1]->get_p_val_t();
                Scalar auxi(space);
                auxi.set_domain(d) = domain_value()(d) - val_ori;
                counts.set(d - dom_min) = count_exported_tau_rows(space, auxi, d);
            }

            n_cond = new Array<int>(counts);
            n_comp = n_cond->get_size(0);
            n_cond_tot = 0;
            for (int i = 0; i < n_cond->get_size(0); i++)
                n_cond_tot += (*n_cond)(i);
            called = true;
        }
    }

    void Eq_first_integral::export_val(int& conte, Term_eq** residus, Array<double>& sec, int& pos_res) const
    {

        // Standard parts
        assert(residus[conte]->get_type_data() == TERM_T);
        // Recover pointer on the space
        const Tensor& constant_value = *residus[conte]->get_p_val_t();
        const Space& space = constant_value.get_space();

        // Get the constant part
        Index pori(space.get_domain(dom_min)->get_nbr_points());
        const Val_domain& constant_domain_value = constant_value()(dom_min);
        double val_cst_part = constant_domain_value.check_if_zero() ? 0.0 : constant_domain_value(pori);
        conte++;

        // Recover origin value of first integral
        const Tensor& inner_value = *residus[conte]->get_p_val_t();
        const Val_domain& inner_domain_value = inner_value()(dom_min);
        double val_ori = inner_domain_value.check_if_zero() ? 0.0 : inner_domain_value(pori);

        // Inner domain
        // Remove value at the origin
        Scalar auximin(space);
        auximin.set_domain(dom_min) = inner_value()(dom_min)-val_ori;
        auximin.set_domain(dom_min).coef_i();

        // At the origin put the cst_part (no continuity but does it matter ?)
        Index pos(space.get_domain(dom_min)->get_nbr_points());
        do {
            if (pos(0) == 0)
                auximin.set_domain(dom_min).set(pos) = val_cst_part;
        } while (pos.inc());

        {
            Array<int> dom_ncond(1);
            dom_ncond.set(0) = (*n_cond)(0);
            space.get_domain(dom_min)->export_tau(auximin, dom_min, 0, sec, pos_res, dom_ncond);
        }
        conte++;

        // Export all the other domains
        for (int d = dom_min + 1; d <= dom_max; d++) {
            const Tensor& domain_value = *residus[conte]->get_p_val_t();
            Scalar auxi(space);
            auxi.set_domain(d) = domain_value()(d)-val_ori;
            Array<int> dom_ncond(1);
            dom_ncond.set(0) = (*n_cond)(d - dom_min);
            space.get_domain(d)->export_tau(auxi, d, 0, sec, pos_res, dom_ncond);
            conte++;
        }
    }

    void Eq_first_integral::export_der(int& conte, Term_eq** residus, Array<double>& sec, int& pos_res) const
    {
        // Standard parts
        assert(residus[conte]->get_type_data() == TERM_T);
        // Recover pointer on the space
        const Tensor& constant_value = *residus[conte]->get_p_val_t();
        const Tensor& constant_derivative = *residus[conte]->get_p_der_t();
        const Space& space = constant_value.get_space();

        // Get the constant part
        Index pori(space.get_domain(dom_min)->get_nbr_points());
        const Val_domain& constant_domain_derivative = constant_derivative()(dom_min);
        double val_cst_part = constant_domain_derivative.check_if_zero() ? 0.0 : constant_domain_derivative(pori);
        conte++;

        // Recover origin value of first integral
        const Tensor& inner_derivative = *residus[conte]->get_p_der_t();
        const Val_domain& inner_domain_derivative = inner_derivative()(dom_min);
        double val_ori = 0.;
        if (!inner_domain_derivative.check_if_zero())
            val_ori = inner_domain_derivative(pori);

        // Inner domain
        // Remove value at the origin
        Scalar auximin(space);
        auximin.set_domain(dom_min) = inner_derivative()(dom_min)-val_ori;
        auximin.set_domain(dom_min).coef_i();

        // At the origin put the cst_part (no continuity but does it matter ?)
        Index pos(space.get_domain(dom_min)->get_nbr_points());
        do {
            if (pos(0) == 0)
                auximin.set_domain(dom_min).set(pos) = val_cst_part;
        } while (pos.inc());

        {
            Array<int> dom_ncond(1);
            dom_ncond.set(0) = (*n_cond)(0);
            space.get_domain(dom_min)->export_tau(auximin, dom_min, 0, sec, pos_res, dom_ncond);
        }
        conte++;

        // Export all the other domains
        for (int d = dom_min + 1; d <= dom_max; d++) {
            const Tensor& domain_derivative = *residus[conte]->get_p_der_t();
            Scalar auxi(space);
            auxi.set_domain(d) = domain_derivative()(d)-val_ori;
            Array<int> dom_ncond(1);
            dom_ncond.set(0) = (*n_cond)(d - dom_min);
            space.get_domain(d)->export_tau(auxi, d, 0, sec, pos_res, dom_ncond);
            conte++;
        }
    }

    bool Eq_first_integral::describe_residual_rows(
        int& conte, Term_eq** residuals, int equation_index,
        std::vector<ResidualRowDescriptor>& descriptors) const
    {
        descriptors.clear();
        const int start = conte;
        conte += n_ope;
        if (!called || !same_reflection_sector || residuals == nullptr) {
            return false;
        }
        for (int part = 0; part < n_ope; ++part) {
            if (residuals[start + part] == nullptr ||
                residuals[start + part]->get_type_data() != TERM_T) {
                return false;
            }
        }

        descriptors.reserve(static_cast<std::size_t>(n_cond_tot));
        for (int domain_index = dom_min; domain_index <= dom_max;
             ++domain_index) {
            const int count_index = domain_index - dom_min;
            const Tensor& domain_residual =
                *residuals[start + count_index + 1]->get_p_val_t();
            Array<int> domain_counts(1);
            domain_counts.set(0) = (*n_cond)(count_index);
            std::vector<ResidualRowDescriptor> domain_rows;
            const Domain* row_domain =
                domain_residual.get_space().get_domain(domain_index);
            if (!row_domain->describe_volume_residual_rows(
                    domain_residual, domain_index, 0, domain_counts, -1,
                    nullptr, domain_rows) ||
                domain_rows.size() !=
                    static_cast<std::size_t>((*n_cond)(count_index))) {
                descriptors.clear();
                return false;
            }
            for (ResidualRowDescriptor& descriptor : domain_rows) {
                if (descriptor.family != ResidualRowEquationFamily::Unavailable ||
                    descriptor.available || descriptor.equation_index != -1 ||
                    descriptor.explicit_sector != 0 ||
                    descriptor.sides.size() != 1) {
                    descriptors.clear();
                    return false;
                }
                const ResidualRowCoordinate& coordinate =
                    descriptor.sides.front();
                if (coordinate.domain != domain_index ||
                    coordinate.component != 0 || coordinate.phi_index < 0 ||
                    coordinate.phi_basis == 0 ||
                    coordinate.phi_index >= row_domain->get_nbr_coefs()(2)) {
                    descriptors.clear();
                    return false;
                }
                descriptor.family = ResidualRowEquationFamily::Field;
                descriptor.equation_index = equation_index;
                descriptor.available = true;
                descriptors.push_back(std::move(descriptor));
            }
        }
        if (descriptors.size() != static_cast<std::size_t>(n_cond_tot)) {
            descriptors.clear();
            return false;
        }
        return true;
    }

    Array<int> Eq_first_integral::do_nbr_conditions(const Tensor& tt) const
    {

        Array<int> res(dom_max - dom_min + 1);
        const Space& space = tt.get_space();
        Scalar count_scalar(space);
        for (int d = dom_min; d <= dom_max; d++) {
            count_scalar.set_domain(d) = tt()(d);
            res.set(d - dom_min) = space.get_domain(d)->nbr_conditions(count_scalar, d, 0)(0);
        }
        return res;
    }

    bool Eq_first_integral::take_into_account(int target) const
    {
        if ((target >= dom_min) && (target <= dom_max))
            return true;
        else
            return false;
    }

} // namespace Kadath
