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
 *   2026-08-09  Added direct one-domain scalar-result storage.
 *   2026-08-10  Added inline storage for a single component pointer.
 *   2026-08-11  Added a size-neutral direct-domain Scalar storage tag.
 */

#pragma once

#include <cstdint>

constexpr int COV = -1;
constexpr int CON = +1;

#include "For_Kadath/Base_tensor/base_tensor.hpp"
#include "For_Kadath/Array/array.hpp"
#include "For_Kadath/Space/space.hpp"
#include "For_Kadath/Val_domain/val_domain.hpp"
#include "For_Kadath/Array/memory.hpp"

namespace Kadath
{

    class Index;
    class Tensor;
    class Vector;
    class Ope_id;
    class Metric_tensor;
    class Param_tensor;
    class Term_eq;
    class Scalar;

    /** Internal tag for results whose values are meaningful in one domain only. */
    struct OneDomainStorageTag
    {
        explicit constexpr OneDomainStorageTag() = default;
    };
    inline constexpr OneDomainStorageTag one_domain_storage{};

    Tensor operator+(const Tensor&);
    Tensor operator-(const Tensor&);
    Tensor operator+(const Tensor&, const Tensor&);
    Tensor operator-(const Tensor&, const Tensor&);
    Scalar operator+(const Tensor&, const Scalar&);
    Scalar operator+(const Scalar&, const Tensor&);
    Tensor operator+(const Tensor&, double);
    Tensor operator+(double, const Tensor&);
    Scalar operator-(const Tensor&, const Scalar&);
    Scalar operator-(const Scalar&, const Tensor&);
    Tensor operator-(const Tensor&, double);
    Tensor operator-(double, const Tensor&);
    Tensor operator*(const Scalar&, const Tensor&);
    Tensor operator*(const Tensor&, const Scalar&);
    Tensor operator*(double, const Tensor&);
    Tensor operator*(const Tensor&, double);
    Tensor operator*(int, const Tensor&);
    Tensor operator*(const Tensor&, int);
    Tensor operator*(const Tensor&, const Tensor&);
    Tensor operator/(const Tensor&, const Scalar&);
    Tensor operator/(const Tensor&, double);
    Tensor operator/(const Tensor&, int);
    double maxval(const Tensor&);
    double minval(const Tensor&);

    void affecte_one_dom(int, Tensor*, const Tensor*);
    Tensor add_one_dom(int, const Tensor&, const Tensor&);
    Tensor add_one_dom(int, const Tensor&, double);
    Tensor add_one_dom(int, double, const Tensor&);
    Tensor sub_one_dom(int, const Tensor&, const Tensor&);
    Tensor sub_one_dom(int, const Tensor&, double);
    Tensor sub_one_dom(int, double, const Tensor&);
    Tensor mult_one_dom(int, const Tensor&, const Tensor&);
    Tensor mult_one_dom(int, const Tensor&, double);
    Tensor mult_one_dom(int, double, const Tensor&);
    Tensor mult_one_dom(int, const Tensor&, int);
    Tensor mult_one_dom(int, int, const Tensor&);
    Tensor div_one_dom(int, const Tensor&, const Tensor&);
    Tensor div_one_dom(int, const Tensor&, double);
    Tensor div_one_dom(int, double, const Tensor&);
    Tensor scal_one_dom(int, const Tensor&, const Tensor&);
    Tensor partial_one_dom(int, char, const Tensor&);
    Tensor sqrt_one_dom(int, const Tensor&);

    //-------------------------//
    //       class Tensor      //
    //-------------------------//

    int add_m_quant(const Param_tensor*, const Param_tensor*);
    int mult_m_quant(const Param_tensor*, const Param_tensor*);
    int inv_m_quant(const Param_tensor*);

    /**
     * Class for handling additional parameters for some \c Tensor.
     * It can, for instance, store the winding number of the scalar field of a boson star.
     * \ingroup fields.
     */
    class Param_tensor : public MemoryMappable
    {

      protected:
        bool m_order_affected; ///< States if the parameter \f$m_{order}\f$ is affected.
        int m_order;           ///< The value of \f$m_{order}\f$, if affected.
        bool m_quant_affected; ///< States if the parameter \f$m_{quant}\f$ is affected.
        int m_quant;           ///< The value of \f$m_{quant}\f$, if affected.
      public:
        Param_tensor();                    ///< Constructor
        Param_tensor(const Param_tensor&); ///< Copy constructor
        ~Param_tensor();                   ///< Destructor

      public:
        int get_m_order() const; ///< Returns \f$m_{order}\f$.
        int& set_m_order();      ///< Sets \f$m_{order}\f$.

        int get_m_quant() const; ///< Returns \f$m_{quant}\f$.
        int& set_m_quant();      ///< Sets \f$m_{quant}\f$.

        friend int add_m_quant(const Param_tensor*, const Param_tensor*);
        friend int mult_m_quant(const Param_tensor*, const Param_tensor*);
        friend int inv_m_quant(const Param_tensor*);

        friend class Tensor;
    };

    /**
     * Tensor handling. It consists mainly of an array of \c Scalar and some informations about the tensor (valence,
     * type and name of indices, tensorial basis of decomposition). The indices range from 1 to ndim (the dimension)
     * (i.e. they do NOT start from 0).
     * \ingroup fields
     */
    class Tensor : public MemoryMappable
    {

        // Data :
        // -----
      protected:
        const Space& espace; ///< The \c Space
        int ndom;            ///< The number of \d Domain
        int ndim;            ///< The dimension/
        int valence;         ///< Valence of the tensor (0 = scalar, 1 = vector, etc...)

        /** Tensorial basis with respect to which the tensor
         *  components are defined.
         */
        Base_tensor basis;

        /** 1D array of integers of size \c valence
         *  containing the type of each index:
         *  \c COV  for a covariant one and \c CON  for a contravariant one.
         *
         */
        Array<int> type_indice;

        bool name_affected; ///< Indicator that states if the indices have been given names.
        /** Scalar-only storage tag: zero is a domain table, 1..255 is direct domain 0..254. */
        mutable std::uint8_t scalar_domain_storage_tag = 0U;
        int n_comp;         ///< Number of stored components, depending on the symmetry.
        /**
         * If the indices haves names they are stored here. Each index is associated with a character.
         */
        char* name_indice;

        Scalar* inline_cmp; ///< Owned pointer-table storage when \c n_comp is one.
        Scalar** cmp;       ///< Array of size \c n_comp of pointers onto the components.

        /** Select inline single-pointer storage or the MemoryMapper fallback. */
        void allocate_cmp_storage();
        /** Release only MemoryMapper-owned component-pointer storage. */
        void release_cmp_storage() noexcept;

        Param_tensor* parameters; ///< Possible additional parameters relevant for the current \c Tensor.

        int (*give_place_array)(const Array<int>&,
                                int); ///< Pointer on the function that gives the storage location corresponding to a
                                      ///< set of indices values. (\c Array version)
        int (*give_place_index)(const Index&, int); ///< Pointer on the function that gives the storage location
                                                    ///< corresponding to a set of indices values. (\c Index version)
        Array<int> (*give_indices)(
            int, int,
            int); ///< Pointer on the function that gives the indices corresponding to a give storage location.

        // Constructors - Destructor :
        // -------------------------
      public:
        /**
         * Constructor
         * @param sp : the \c Space.
         * @param val : valence.
         * @param tipe : \c Array containing the types of each index (COV vs CON).
         * @param ba : the tensorial basis used.
         */
        Tensor(const Space& sp, int val, const Array<int>& tipe, const Base_tensor& ba);
        /**
         * Constructor where all the indices are of the same type
         * @param sp : the \c Space.
         * @param val : valence.
         * @param tipe : the type of all the indices (COV vs CON).
         * @param ba : the tensorial basis used.
         */
        Tensor(const Space& sp, int val, int tipe, const Base_tensor&);

        /**
         * Constructor assuming the dimension of the space and the tensor is different (for dealing with symmetries)
         * @param sp : the \c Space.
         * @param val : valence.
         * @param tipe : \c Array containing the types of each index (COV vs CON).
         * @param ba : the tensorial basis used.
         * @param dim : dimension of the tensor.
         */
        Tensor(const Space& sp, int val, const Array<int>& tipe, const Base_tensor& ba, int dim);

        /**
         * Constructor where all the indices are of the same type.
         * The dimension of the space and the tensor is different (for dealing with symmetries)
         * @param sp : the \c Space.
         * @param val : valence.
         * @param tipe : the type of all the indices (COV vs CON).
         * @param ba : the tensorial basis used.
         * @param dim : dimension of the tensor.
         */
        Tensor(const Space& sp, int val, int tipe, const Base_tensor&, int dim);

        /**
         * Constructor by copy
         * @param so : the input \c Tensor.
         * @param copie : if false only the property of the tensor are copied (valence etc...) not the values of the
         * field that are left undefined.
         */
        Tensor(const Tensor&, bool copie = true);
        /**
         * Copy a tensor's component layout while materializing storage only
         * for one domain. If copy is true, that domain's values are copied;
         * otherwise only its spectral bases and tensor metadata are copied.
         */
        Tensor(OneDomainStorageTag, int active_domain, const Tensor& source,
               bool copy = true);
        /**
         * Move constructor. Transfers the component, index-name and parameter
         * storage, leaving the source empty and safe to destroy.
         *
         * A \c Scalar rvalue selects the dedicated copying overload because
         * its sole component pointer refers back to the \c Scalar object
         * itself and cannot be transferred.
         */
        Tensor(Tensor&&);
        Tensor(Scalar&&);
        Tensor(const Space& sp, BinarySource&);  ///< Modern API: from BinarySource.
        Tensor(const Space& sp, int dim, BinarySource&); ///< Modern API with explicit dim.

      protected:
        /**
         *  Constructor for a scalar field: to be used only by the derived
         *  class \c Scalar .
         * @param sp : the only parameter the \c Space.
         */
        explicit Tensor(const Space& sp);
        /** Construct a plain valence-zero Tensor with storage in one domain. */
        Tensor(OneDomainStorageTag, int active_domain, const Space& sp);
        /**
         * Construct only the Scalar/Val_domain storage needed by a one-domain
         * algebra result. Inactive domains remain intentionally absent.
         */
        Tensor(OneDomainStorageTag, int active_domain, const Space& sp, int val,
               const Array<int>& tipe, const Base_tensor& ba);
        /**
         * Constructor where the number of components is prescribed.
         * @param sp : the \c Space.
         * @param val : valence.
         * @param tipe : \c Array containing the types of each index (COV vs CON).
         * @param n_compi : number of components.
         * @param ba : the tensorial basis used.
         */
        Tensor(const Space& sp, int val, const Array<int>& tipe, int n_compi, const Base_tensor&);
        Tensor(OneDomainStorageTag, int active_domain, const Space& sp, int val,
               const Array<int>& tipe, int n_compi, const Base_tensor&);

        /**
         * Constructor where the number of components is prescribed (all the indices are of the same type).
         * @param sp : the \c Space.
         * @param val : valence.
         * @param tipe : the type of all the indices (COV vs CON).
         * @param n_compi : number of components.
         * @param ba : the tensorial basis used.
         */
        Tensor(const Space& sp, int val, int tipe, int n_compi, const Base_tensor& ba);
        /**
         * Constructor where the number of components is prescribed.
         * The dimension of space and the tensor can be different (to deal with symmetries)
         * @param sp : the \c Space.
         * @param val : valence.
         * @param tipe : \c Array containing the types of each index (COV vs CON).
         * @param n_compi : number of components.
         * @param ba : the tensorial basis used.
         * @param dim : dimension of the tensor
         */
        Tensor(const Space& sp, int val, const Array<int>& tipe, int n_compi, const Base_tensor&, int dim);

        /**
         * Constructor where the number of components is prescribed (all the indices are of the same type).
         * The dimension of space and the tensor can be different (to deal with symmetries)
         * @param sp : the \c Space.
         * @param val : valence.
         * @param tipe : the type of all the indices (COV vs CON).
         * @param n_compi : number of components.
         * @param ba : the tensorial basis used.
         * @param dim : the dimension of the tensor
         */
        Tensor(const Space& sp, int val, int tipe, int n_compi, const Base_tensor& ba, int dim);

      public:
        virtual ~Tensor();              ///< Destructor
        virtual void save(BinarySink&) const; ///< Modern API.

        // Mutators / assignment
        // ---------------------
      public:
        const Param_tensor* get_parameters() const; ///< Returns a pointer on the possible additional parameters.
        Param_tensor*& set_parameters();            ///< Read/write of the parameters.
        void affect_parameters();                   ///< Affects the additional parameters.
        bool
        is_m_order_affected() const; ///< Checks whether the additional parameter order is affected (not very used).
        bool is_m_quant_affected()
            const; ///< Checks whether the additional parameter \f$m\f$ is affected (used for boson stars for instance).

        /** Assigns a new tensorial basis in a given domain
         * @param dd : the index of the \c Domain.
         */
        int& set_basis(int dd);

        virtual void operator=(const Tensor&); ///< Assignment to a \c Tensor
        virtual void operator=(double xx); ///< Assignment to a double (the same value for all the components at all the
                                           ///< collocation points).
        virtual void annule_hard();        ///< Sets the \c Tensor to zero (hard version ; no logical state used).

        /** Returns the value of a component (read/write version).
         *
         * @param ind  \c Array  of size \c valence  containing the
         *		values of each index specifing the component,  with the
         *		following storage convention:
         *			\li \c ind(0)  : value of the first index
         *			\li \c ind(1)  : value of the second index
         *			\li and so on...
         * @return modifiable reference on the component specified by \c ind
         *
         */
        Scalar& set(const Array<int>& ind);
        /** Returns the value of a component (read/write version).
         *
         * @param ind  1-D \c Index  of size \c valence  containing the
         *		values of each index specifing the component,  with the
         *		following storage convention:
         *			\li \c ind(0)  : value of the first index
         *			\li \c ind(1)  : value of the second index
         *			\li and so on...
         * @return modifiable reference on the component specified by \c ind
         *
         */
        Scalar& set(const Index& ind);
        Scalar& set(); ///< Read/write for a \c Scalar.
        /** Returns the value of a component for a tensor of valence 1
         *  (read/write version).
         *
         * @param i1  value of the first index
         * @return modifiable reference on the component specified by \c (i1)
         *
         */
        Scalar& set(int i);
        /** Returns the value of a component for a tensor of valence 2
         *  (read/write version).
         *
         * @param i1  value of the first index
         * @param i2  value of the second index
         *
         * @return modifiable reference on the component specified by \c (i1,i2)
         *
         */
        Scalar& set(int i1, int i2);

        /** Returns the value of a component for a tensor of valence 3
         *  (read/write version).
         *
         * @param i1  value of the first index
         * @param i2  value of the second index
         * @param i3  value of the third index
         *
         * @return modifiable reference on the component specified by \c (i1,i2,i3)
         *
         */
        Scalar& set(int i1, int i2, int i3);

        /** Returns the value of a component for a tensor of valence 4
         *  (read/write version).
         *
         * @param i1  value of the first index
         * @param i2  value of the second index
         * @param i3  value of the third index
         * @param i4  value of the fourth index
         *
         * @return modifiable reference on the component specified by \c (i1,i2,i3,i4)
         *
         */
        Scalar& set(int i1, int i2, int i3, int i4);

        /**
         * Sets the name of one index ; the names must have been affected first.
         * @param dd : which index ?
         * @param name : the name.
         */
        void set_name_ind(int dd, char name);
        /**
         * @returns the names of all the indices.
         */
        char* get_name_ind() const { return name_indice; };
        /**
         * Check whether the names of the indices have been affected.
         */
        bool is_name_affected() const { return name_affected; };
        /**
         * Affects the name of the indices.
         * They have to be given values afterwards.
         */
        void set_name_affected() { name_affected = true; };

        /**
         * Does the inner contraction of the \c Tensor.
         * It assumes exactly two indices of different types have the same name.
         * @returns : the contracted \c Tensor (with valence -2).
         */
        Tensor do_summation() const;

        /**
         * Does the inner contraction of the \c Tensor in a given domain. The values in the other domains are undefined.
         * It assumes exactly two indices of different types have the same name.
         * @param dd : the \c Domain where the contraction is performed.
         * @returns : the contracted \c Tensor (with valence -2).
         */
        Tensor do_summation_one_dom(int dd) const;

        Tensor grad() const; ///< Computes the flat gradient, in Cartesian coordinates.

        /**
         * Sets the standard spectal bases of decomposition for each component.
         * To be used only with \c valence  lower than or equal 2.
         */
        virtual void std_base();

        // Accessors
        // ---------
      public:
        /**
         * Gives the location of a given component in the array used for storage (\C Array version).
         * @param ind : values of the indices.
         * @returns : the storage location.
         */
        virtual int position(const Array<int>& ind) const;
        /**
         * Gives the location of a given component in the array used for storage (\C Index version).
         * @param ind : values of the indices.
         * @returns : the storage location.
         */
        virtual int position(const Index& ind) const;
        /**
         * Gives the values of the indices corresponding to a location in the array used for storage of the components.
         * @param pos : the storage location.
         * @returns : the values of all the indices.
         */
        virtual Array<int> indices(int pos) const;

      private:
        /**
         * Checks whether the current \c Tensor and \c tt have compatible indices (i.e. same names and types, possibly
         * in a different order).
         * @param tt : the \c Tensor used for the comparison.
         * @param output_ind : if the indices are compatible, it contains the permutation of the indices.
         * @returns : true if the indices are compatible, false otherwise.
         */
        bool find_indices(const Tensor& tt, Array<int>& output_ind) const;

      public:
        /**
         * Returns the \c Space.
         */
        const Space& get_space() const { return espace; };

        /** Returns the vectorial basis (triad) on which the components
         *  are defined.
         */
        const Base_tensor& get_basis() const { return basis; };

        /**
         * Returns the valence.
         */
        int get_valence() const { return valence; };

        /**
         * Returns the number of stored components.
         */
        int get_n_comp() const { return n_comp; };
        /**
         * Returns the number dimension.
         */
        int get_ndim() const { return ndim; };

        /**
         *  Gives the type (covariant or contravariant) of a given index.
         *  @param i : the index number (>=1)
         *  @return COV for a covariant index, CON for a contravariant one.
         */
        int get_index_type(int i) const { return type_indice(i); };

        /**
         *  @return The types of all the indices.
         */
        Array<int> get_index_type() const { return type_indice; };

        /**
         *  Sets the type of the index number
         *  @param i : the index number (>=1)
         *  @return set to COV  or CON.
         */
        int& set_index_type(int i) { return type_indice.set(i); };

        /**
         * Sets the types of all the indices.
         *
         *  @return a reference on an array describing the types of all the indices (CON or COV).
         */
        Array<int>& set_index_type() { return type_indice; };

        /** Returns the value of a component (read only version).
         *
         * @param ind  \c Array  of size \c valence  containing the
         *		values of each index specifing the component,  with the
         *		following storage convention:
         *			\li \c ind(0)  : value of the first index
         *			\li \c ind(1)  : value of the second index
         *			\li and so on...
         * @return  the component specified by \c ind
         *
         */
        const Scalar& operator()(const Array<int>& ind) const;
        /** Returns the value of a component (read only version).
         *
         * @param ind  \cIndex  of size \c valence  containing the
         *		values of each index specifing the component,  with the
         *		following storage convention:
         *			\li \c ind(0)  : value of the first index
         *			\li \c ind(1)  : value of the second index
         *			\li and so on...
         * @return  the component specified by \c ind
         *
         */
        const Scalar& operator()(const Index& ind) const;

        const Scalar& operator()() const; ///< Read only for a \c Scalar.
        /** Returns the value of a component for a tensor of valence 1
         *  (read only version).
         *
         * @param i1  value of the first index
         * @return the component specified by \c (i1)
         *
         */
        const Scalar& operator()(int i) const;
        /** Returns the value of a component for a tensor of valence 2
         *  (read only version).
         *
         * @param i1  value of the first index
         * @param i2  value of the second index
         *
         * @return the component specified by \c (i1,i2)
         *
         */
        const Scalar& operator()(int i1, int i2) const;
        const Scalar& at(int i1, int i2) const;

        /** Returns the value of a component for a tensor of valence 3
         *  (read only version).
         *
         * @param i1  value of the first index
         * @param i2  value of the second index
         * @param i3  value of the third index
         *
         * @return the component specified by \c (i1,i2,i3)
         *
         */
        const Scalar& operator()(int i1, int i2, int i3) const;

        /** Returns the value of a component for a tensor of valence 4
         *  (read only version).
         *
         * @param i1  value of the first index
         * @param i2  value of the second index
         * @param i3  value of the third index
         * @param i4  value of the fourth index
         *
         * @return the component specified by \c (i1,i2,i3,i4)
         */
        const Scalar& operator()(int i1, int i2, int i3, int i4) const;

        void change_basis_spher_to_cart(); ///< Changes the tensorial basis from orthonormal spherical to Cartesian.
        void change_basis_cart_to_spher(); ///< Changes the tensorial basis from Cartesian to orthonormal spherical.

        /**
         * Sets all the coefficients below a given treshold, to zero (maintaining regularity).
         * @param tre : the threshold.
         */
        void filter(double tre);

        // Member arithmetics
        // ------------------
      public:
        void coef() const;              ///< Computes the coefficients.
        void coef_i() const;            ///< Computes the values in the configuration space.
        void operator+=(const Tensor&); ///< += Tensor
        void operator-=(const Tensor&); ///< -= Tensor

        /**
         * Sets to zero all the coefficients above a given order, for the \f$ varphi\f$ coefficients, in a gicen \c
         * Domain. Takes into account the various Galerkin basis to maintain regularity.
         * @param dom : the \c Domain where the filter is applied.
         * @param ncf : the coefficients which index is above this are set to zero.
         */
        void filter_phi(int dom, int ncf);

        // Outputs
        // -------
      public:
        friend ostream& operator<<(ostream&, const Tensor&); ///< Display

        // Friend classes
        // ---------------
        friend class Index;
        friend class Domain;
        friend class Scalar;
        friend class Vector;
        friend class System_of_eqs;
        friend class Eq_matching_non_std;
        friend class Ope_id;
        friend class Metric_tensor;
        friend class Space_spheric_adapted;
        friend class Space_spheric_homothetic;
        friend class Space_trumpet;
        friend class Space_polar_adapted;
        friend class Space_polar_bilateral_adapted;
        friend class Space_adapted_bh_polar;
        friend class Space_bin_ns;
        friend class Space_bin_ns_nosym;
        friend class Space_bhns;
        friend class Space_bhns_nosym;
        friend class Space_bin_bh;
        friend class Space_polar_periodic;
        friend class Space_adapted_bh;
        friend class Space_adapted_bh_nosym;
        friend class Space_KerrSchild_bh;
        friend class Space_bbh;
        friend class Space_Kerr_bbh;
        friend class Space_three_body;

        // Mathematical operators
        // ----------------------
        friend Tensor operator+(const Tensor&);
        friend Tensor operator-(const Tensor&);
        friend Tensor operator+(const Tensor&, const Tensor&);
        friend Scalar operator+(const Tensor&, const Scalar&);
        friend Scalar operator+(const Scalar&, const Tensor&);
        friend Tensor operator+(const Tensor&, double);
        friend Tensor operator+(double, const Tensor&);
        friend Tensor operator-(const Tensor&, const Tensor&);
        friend Scalar operator-(const Tensor&, const Scalar&);
        friend Scalar operator-(const Scalar&, const Tensor&);
        friend Tensor operator-(const Tensor&, double);
        friend Tensor operator-(double, const Tensor&);
        friend Tensor operator*(const Scalar&, const Tensor&);
        friend Tensor operator*(const Tensor&, const Scalar&);
        friend Tensor operator*(double, const Tensor&);
        friend Tensor operator*(const Tensor&, double);
        friend Tensor operator*(int, const Tensor&);
        friend Tensor operator*(const Tensor&, int);
        friend Tensor operator*(const Tensor&,
                                const Tensor&); ///< Tensor multiplication ; if need be contractions are performed.
        friend Tensor operator/(const Tensor&, const Scalar&);
        friend Tensor operator/(const Tensor&, double);
        friend Tensor operator/(const Tensor&, int);
        friend double
        maxval(const Tensor&); ///< Gives the maximum value amongst all the components, at all the collocation points.
        friend double
        minval(const Tensor&); ///< Gives the minimum value amongst all the components, at all the collocation points.

        friend void affecte_one_dom(int, Tensor*, const Tensor*);
        friend Tensor add_one_dom(int, const Tensor&, const Tensor&);
        friend Tensor add_one_dom(int, const Tensor&, double);
        friend Tensor add_one_dom(int, double, const Tensor&);
        friend Tensor sub_one_dom(int, const Tensor&, const Tensor&);
        friend Tensor sub_one_dom(int, const Tensor&, double);
        friend Tensor sub_one_dom(int, double, const Tensor&);
        friend Tensor mult_one_dom(int, const Tensor&, const Tensor&);
        friend Tensor mult_one_dom_general(int, const Tensor&, const Tensor&);
        friend Tensor mult_one_dom(int, const Tensor&, double);
        friend Tensor mult_one_dom(int, double, const Tensor&);
        friend Tensor mult_one_dom(int, const Tensor&, int);
        friend Tensor mult_one_dom(int, int, const Tensor&);
        friend Tensor div_one_dom(int, const Tensor&, const Tensor&);
        friend Tensor div_one_dom(int, const Tensor&, double);
        friend Tensor div_one_dom(int, double, const Tensor&);
        friend Tensor scal_one_dom(int, const Tensor&, const Tensor&);
        friend Tensor partial_one_dom(int, char, const Tensor&);
        friend Tensor sqrt_one_dom(int, const Tensor&);

        friend class Domain_nucleus;
        friend class Domain_shell;
        friend class Domain_bispheric_chi_first;
        friend class Domain_bispheric_rect;
        friend class Domain_bispheric_eta_first;
        friend class Domain_bispheric_chi_first_nosym;
        friend class Domain_bispheric_rect_nosym;
        friend class Domain_bispheric_eta_first_nosym;
        friend class Domain_shell_outer_adapted;
        friend class Domain_shell_inner_adapted;
        friend class Domain_polar_shell_outer_adapted;
        friend class Domain_polar_shell_inner_adapted;
        friend class Domain_polar_shell_bilateral_adapted;
        friend class Domain_compact;
        friend class Domain_polar_periodic_nucleus;
        friend class Domain_polar_periodic_shell;
        friend class Domain_nucleus_nosym;
        friend class Domain_shell_nosym;
        friend class Domain_compact_nosym;
        friend class Domain_shell_inner_adapted_nosym;
        friend class Domain_shell_outer_adapted_nosym;
        friend class Space_spheric_adapted_nosym;
        friend class Space_bin_ns_nosym;
        friend class Space_three_body;
    };
} // namespace Kadath
