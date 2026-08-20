/*
    Copyright 2017 Philippe Grandclement

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
 */

#pragma once
#include "metric.hpp"
#include "For_Kadath/Scalar/scalar.hpp"
#include "For_Kadath/Tensor/vector.hpp"

namespace Kadath
{
    class Space;
    class Tensor;
    class Term_eq;
    class Metric_tensor;
    class System_of_eqs;
    class Base_tensor;

    /**
     * Class that deals with flat metric assuming a axisymmetric setting (nothing depends on \f$varphi\f$).
     * \ingroup metric
     */
    class Metric_flat_nophi : public Metric
    {

      protected:
        const Base_tensor& basis; ///< The tensorial basis used.

      public:
        /**
         * Standard constructor
         * @param sp : the associated \c Space.
         */
        Metric_flat_nophi(const Space& sp, const Base_tensor& bb);
        Metric_flat_nophi(const Metric_flat_nophi&); ///< Copy constructor

      protected:
        virtual void compute_con(int) const;
        virtual void compute_cov(int) const;
        virtual void compute_christo(int) const;

      private:
        /**
         * Computes the partial derivative part of the covariant derivative, in orthonormal spherical coordinates.
         *  The \f$varphi\f$ component is set to zero.
         * The result is \f$ partial_r, 1/rpartial_theta, 0\f$.
         * If need be inner summation of the result is performed and index manipulated.
         * @param tder : type of the result (COV or CON)
         * @param indder : name of the index corresponding to the derivative.
         * @param so : the field to be derived.
         * @return : the result as a \c Term_eq.
         */
        Term_eq derive_partial_spher(int tder, char indder, const Term_eq& so) const;

        /**
         * Computes the flat covariant derivative, in orthonormal spherical coordinates.
         * If need be inner summation of the result is performed and index manipulated.
         * @param tder : type of the result (COV or CON)
         * @param indder : name of the index corresponding to the derivative.
         * @param so : the field to be derived.
         * @return : the result as a \c Term_eq.
         */
        Term_eq derive_spher(int tder, char indder, const Term_eq& so) const;

        /**
         * Computes the flat covariant derivative, in orthonormal spherical coordinates.
         * If need be inner summation of the result is performed.
         * If the CON derivative is asked for, the index is NOT manipulated by the flat metric but by an arbitrary one.
         * @param tder : type of the result (COV or CON)
         * @param indder : name of the index corresponding to the derivative.
         * @param so : the field to be derived.
         * @param othermet : pointer on the \c Metric used for index manipulation.
         * @return : the result as a \c Term_eq.
         */
        Term_eq derive_with_other_spher(int tder, char indder, const Term_eq& so, const Metric* othermet) const;

      public:
        virtual ~Metric_flat_nophi();
        virtual void update();
        virtual void update(int);
        virtual void manipulate_ind(Term_eq&, int) const;
        virtual Term_eq derive_partial(int, char, const Term_eq&) const;
        virtual Term_eq derive(int, char, const Term_eq&) const;

        /**
         * Computes the flat covariant derivative.
         * If need be inner summation of the result is performed.
         * If the CON derivative is asked for, the index is NOT manipulated by the flat metric but by an arbitrary one.
         * @param tder : type of the result (COV or CON)
         * @param indder : name of the index corresponding to the derivative.
         * @param so : the field to be derived.
         * @param othermet : pointer on the \c Metric used for index manipulation.
         * @return : the result as a \c Term_eq.
         */
        Term_eq derive_with_other(int tder, char indder, const Term_eq& so, const Metric* othermet) const;

        /**
         * Associate the metric to a given system of equations.
         * @param syst : the \c System_of_eqs.
         * @param name : name by which the metric will be known in the system (like "g", "f"...)
         */
        virtual void set_system(System_of_eqs& syst, const char* name);

        virtual int give_type(int) const;
    };

    /**
     * Class to deal with a metric independant of \f$varphi\f$.
     * \ingroup metric
     */
    class Metric_nophi : public Metric
    {

      protected:
        Metric_tensor* p_met;     ///< Pointer on the \c Metric_tensor describing the metric.
        const Base_tensor& basis; ///< The tensorial basis used.
        Metric_flat_nophi fmet;   ///< Associated flat metric.
        int place_syst; ///< Gives the location of the metric amongst the various unknowns of the associated \c
                        ///< System_of_eqs.

      public:
        Metric_nophi(Metric_tensor&);      ///< Constructor from a \c Metric_tensor.
        Metric_nophi(const Metric_nophi&); ///< Copy constructor.

      protected:
        virtual void compute_con(int) const;
        virtual void compute_cov(int) const;
        virtual void compute_christo(int) const;
        virtual void compute_riemann(int) const;
        virtual void compute_ricci_tensor(int) const;

      public:
        virtual ~Metric_nophi();
        virtual Term_eq derive(int, char, const Term_eq&) const;
        /**
         * Associate the metric to a given system of equations.
         * @param syst : the \c System_of_eqs.
         * @param name : name by which the metric will be known in the system (like "g", "f"...)
         */
        virtual void set_system(System_of_eqs& syst, const char* name);
        virtual Term_eq derive_flat(int, char, const Term_eq&) const;

        virtual int give_type(int) const;
    };

    /**
     * Class to deal with a metric independant of \f$varphi\f$ and constant. By this one means that it is a given and is
     * not modified by the \c System_of_eqs.
     * \ingroup metric
     */
    class Metric_nophi_const : public Metric_nophi
    {

      public:
        Metric_nophi_const(Metric_tensor&);            ///< Constructor from a \c Metric_tensor.
        Metric_nophi_const(const Metric_nophi_const&); ///< Copy constructor

      protected:
        virtual void compute_con(int) const;
        virtual void compute_cov(int) const;

      public:
        virtual ~Metric_nophi_const();

        /**
         * Associate the metric to a given system of equations.
         * @param syst : the \c System_of_eqs.
         * @param name : name by which the metric will be known in the system (like "g", "f"...)
         */
        virtual void set_system(System_of_eqs& syst, const char* name);
    };

    /**
     * Class to deal with a metric independant of \f$varphi\f$ with a conformal decomposition (mainly used for AADS
     * spacetimes) The true metric and the conformal are related via
     * \f$gamma_{ij} = 1/Omega^2gamma_tilde_{ij}\f$.
     * The conformal factor vanishes at some boundary so that the various quantities (Christoffels) are multiplied by
     * appropriate factors of $Omega$ to ensure regularity.
     * \ingroup metric
     */
    class Metric_nophi_AADS : public Metric
    {

      protected:
        Metric_tensor* p_met;     ///< Pointer on the \c Metric_tensor describing the coformal metric.
        const Base_tensor& basis; ///< The tensorial basis used.
        Metric_flat_nophi fmet;   ///< Associated flat metric.
        Scalar conformal;         ///< The conformal factor $Omega$ (must be a purely radial function)
        Scalar der_conf;          ///< Radial derivative of the conformal factor
        int place_syst; ///< Gives the location of the metric amongst the various unknowns of the associated \c
                        ///< System_of_eqs.

      public:
        Metric_nophi_AADS(Metric_tensor&,
                          const Scalar& conf);       ///< Constructor from a \c Metric_tensor and a conformal factor.
        Metric_nophi_AADS(const Metric_nophi_AADS&); ///< Copy constructor.

      protected:
        virtual void compute_con(int) const;
        virtual void compute_cov(int) const;
        virtual void compute_christo(int) const;
        virtual void compute_riemann(int) const;
        virtual void compute_ricci_tensor(int) const;

      public:
        virtual ~Metric_nophi_AADS();
        virtual Term_eq derive(int, char, const Term_eq&) const;
        /**
         * Associate the metric to a given system of equations.
         * @param syst : the \c System_of_eqs.
         * @param name : name by which the metric will be known in the system (like "g", "f"...)
         */
        virtual void set_system(System_of_eqs& syst, const char* name);
        virtual Term_eq derive_flat(int, char, const Term_eq&) const;

        virtual int give_type(int) const;
    };

    /**
     * Class to deal with a metric independant of \f$varphi\f$ with a conformal decomposition and constant. By this one
     * means that it is a given and is not modified by the \c System_of_eqs.
     * \ingroup metric
     */
    class Metric_nophi_AADS_const : public Metric_nophi_AADS
    {

      public:
        Metric_nophi_AADS_const(Metric_tensor&, const Scalar&);  ///< Constructor from a \c Metric_tensor.
        Metric_nophi_AADS_const(const Metric_nophi_AADS_const&); ///< Copy constructor

      protected:
        virtual void compute_con(int) const;
        virtual void compute_cov(int) const;

      public:
        virtual ~Metric_nophi_AADS_const();

        /**
         * Associate the metric to a given system of equations.
         * @param syst : the \c System_of_eqs.
         * @param name : name by which the metric will be known in the system (like "g", "f"...)
         */
        virtual void set_system(System_of_eqs& syst, const char* name);
    };

} // namespace Kadath
