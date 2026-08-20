/*
    Copyright 2017 Gregoire Martinon

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
 *   2026-08-07  Removed the public GSL permutation dependency.
 */

#pragma once

#include "metric.hpp"
#include "For_Kadath/Scalar/scalar.hpp"
#include "For_Kadath/Space/space.hpp"
#include "For_Kadath/System_of_eqs/system_of_eqs.hpp"
#include "For_Kadath/Tensor/tensor.hpp"
#include "For_Kadath/Term_eq/term_eq.hpp"
#include "For_Kadath/Utilities/name_tools.hpp"
#include "For_Kadath/Tensor/metric_tensor.hpp"
#include "For_Kadath/Base_tensor/base_tensor.hpp"
#include "For_Kadath/Tensor/vector.hpp"
#include <cmath>
#include <vector>
#include <string>

namespace Kadath
{
    using std::vector;

    /**
     * Class to manage anti de Sitter metrics.
     * \ingroup metric
     */
    class Metric_ADS : public Metric
    {
      protected:
        int m_nd;      ///< number of physical domains (usually the last compaxtified domain is discarded).
        double m_L;    ///<  AdS length
        Scalar m_rho2; ///< \c Scalar containing \f$ r^2/L^2 \f$
        Scalar m_eps;  ///< \c Scalar containing the conformal factor \f$\epsilon = (1 - r^2/L^2)/(1 + r^2/L^2)\f$.
        Base_tensor* p_basis; ///< Pointer on the tensorial basis (Cartesian basis only).

      public:
        Metric_ADS(const Space& space);   ///< Standard constructor.
        Metric_ADS(const Metric_ADS& so); ///< Copy constructor
        virtual void manipulate_ind(Term_eq& so, int ind) const;
        virtual Term_eq derive(int type_der, char ind_der, const Term_eq&) const;
        /**
         * Associate the metric to a given system of equations.
         * @param syst : the \c System_of_eqs.
         * @param name : name by which the metric will be known in the system (like "g", "f"...)
         */
        virtual void set_system(System_of_eqs& syst, const char* name);
        virtual ~Metric_ADS();

      protected:
        virtual void compute_cov(int d) const;
        virtual void compute_con(int d) const;
        virtual void compute_christo(int d) const;
        virtual void compute_riemann(int d) const;
        virtual void compute_ricci_tensor(int d) const;
        virtual void compute_ricci_scalar(int d) const;
        virtual void update(int d);
        virtual void update();

        /**
         * Put the covariant and contravariant metric, dans the Ricci background into the \c System_of_eqs, as
         * constants.
         * @param syst : the \c System_of_eqs.
         * @param name_back_cov : name of the background convariant metric
         * @param name_back_con : name of the background contravariant metric
         * @param name_back_ricci : name of the background Ricci tensor.
         */
        void
        init_system(System_of_eqs& ss, const char* name_back_cov, const char* name_back_con,
                    const char* name_back_ricci); // put the covariant and contravariant metric in cst of the system
                                                  /**
                                                   * Put the covariant and contravariant metric, dans the Ricci background into the \c System_of_eqs, as
                                                   * constants.
                                                   * @param syst : the \c System_of_eqs.
                                                   * @param name_back_cov : name of the background convariant metric
                                                   * @param name_back_con : name of the background contravariant metric
                                                   * @param name_back_gam : name of the background Christoffels
                                                   * @param name_back_ricci : name of the background Ricci tensor.
                                                   */
        void
        init_system(System_of_eqs& ss, const char* name_back_cov, const char* name_back_con, const char* name_back_gam,
                    const char* name_back_ricci); // put the covariant and contravariant metric in cst of the system

      private:
        /**
         * Computes the covariant derivative. The name of the index is not defined (hence no inner summation is
         * performed).
         * @param so : the \c Term_eq to be derived.
         * @return : the covariant derivative.
         */
        Term_eq derive_simple(const Term_eq& so) const;

        /**
         * Division by the conformal factor \f$\epsilon^n\f$.
         * In the \c Domain containing the ADS boundary, the computation is performed in the coefficient space.
         * @param so : the input \c Term_eq.
         * @param n : power of the division.
         * @return the result.
         */
        Term_eq div_eps(Term_eq& so, int n) const;

        /**
         * Multiplication by the conformal factor \f$\epsilon^n\f$.
         * @param so : the input \c Term_eq.
         * @param n : power of the multiplication.
         * @return the result.
         */
        Term_eq mul_eps(Term_eq& so, int n) const;

        friend class Metric_AADS;
    };

    /**
     * Class to manage asymptotically anti de Sitter metrics.
     * \ingroup metric
     */
    class Metric_AADS : public Metric
    {
      protected:
        int m_nd;             ///< number of physical domains (usually the last compaxtified domain is discarded).
        int m_place_syst;     ///< Where is the unknown h in the system of equations.
        double m_L;           ///< AdS length
        mutable bool m_doder; ///< true if metric variation allowed to be computed
        Metric_ADS m_ads;     ///< Background metric(i.e. ADS metric).
        Metric_tensor*
            p_hmet; ///< \f$h_{ij}\f$ : the discrepancy between covariant metric and contravariant background.
        Vector*
            p_der_eps; ///< Pointer on the vector containing the derivative of the conformal factor \f$ D_i \epsilon\f$
        Base_tensor* p_basis; ///< Pointer on the tensorial basis (Cartesian basis only).

      private:
        mutable std::vector<std::unique_ptr<Term_eq>>
            p_h; ///< Pointers on the differences between the covariant metric and the covariant background
        mutable std::vector<std::unique_ptr<Term_eq>>
            p_k; ///< Pointers on the differences between the contravariant metric and the contravariant background

        /**
         * Computes the covariant derivative. The name of the index is not defined (hence no inner summation is
         * performed).
         * @param so : the \c Term_eq to be derived.
         * @return : the covariant derivative.
         */
        Term_eq derive_simple(const Term_eq& so) const;

      public:
        /**
         * Standard constructor.
         * @param space : the associated \c Space.
         * @param hmet : the difference between the covariant metric and the covariant background
         */
        Metric_AADS(const Space& space, Metric_tensor& hmet);
        Metric_AADS(const Metric_AADS& so); ///< Copy constructor.
        virtual void manipulate_ind(Term_eq& so, int ind) const;
        virtual Term_eq derive(int type_der, char ind_der, const Term_eq& so) const;

        /**
         * Associates the metric to a given system of equations.
         * @param syst : the \c System_of_eqs.
         * @param name_met : name by which the metric will be known in the system (like "g", "f"...)
         * @param name_hmet : name of \f$ h_{ij} \f$, difference with the background.
         */
        void set_system(System_of_eqs& syst, const char* name_met, const char* name_hmet);
        /**
         * Associates the metric to a given system of equations. It also sets the background quantities as constants.
         * @param syst : the \c System_of_eqs.
         * @param name_met : name by which the metric will be known in the system (like "g", "f"...)
         * @param name_hmet : name of \f$ h_{ij} \f$, difference with the background.
         * @param name_back_cov : name of the background convariant metric
         * @param name_back_con : name of the background contravariant description.
         * @param name_back_ricci : name of the background Ricci tensor.
         */
        void set_system(System_of_eqs& ss, const char* name_met, const char* name_hmet, const char* name_back_cov,
                        const char* name_back_con, const char* name_back_ricci);
        /**
         * Associates the metric to a given system of equations. It also sets the background quantities as constants.
         * @param syst : the \c System_of_eqs.
         * @param name_met : name by which the metric will be known in the system (like "g", "f"...)
         * @param name_hmet : name of \f$ h_{ij} \f$, difference with the background.
         * @param name_back_cov : name of the  convariant background metric
         * @param name_back_con : name of the contravariant background metric.
         * @param name_back_gam : name of the background Christoffels.
         * @param name_back_ricci : name of the backgroundRicci tensor.
         */
        void set_system(System_of_eqs& ss, const char* name_met, const char* name_hmet, const char* name_back_cov,
                        const char* name_back_con, const char* name_back_gam, const char* name_back_ricci);

        virtual const Metric* get_background() const; ///< @return the pointer on the background metric.
        virtual ~Metric_AADS();

      protected:
        virtual void compute_cov(int d) const;
        virtual void compute_con(int d) const;
        virtual void compute_christo(int d) const;
        virtual void compute_ricci_tensor(int d) const;
        virtual void compute_ricci_scalar(int d) const;

        virtual void update(int d);
        virtual void update();
    };
} // namespace Kadath
