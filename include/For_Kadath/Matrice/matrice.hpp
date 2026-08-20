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
// fichiers includes
#include "For_Kadath/Array/headcpp.hpp"
#include "For_Kadath/Array/array.hpp"
#include "For_Kadath/Array/memory.hpp"

namespace Kadath
{
    /**
     * Matrix handling.
     * The matrix can be stored in the usual way in \c std,  in a band-way by
     * \c band and on a LU-decomposition by the two arrays \c lu and
     * \c permute. All the storage conventions are those af \b LAPACK which is
     * used to make the LU-decomposition,  the inversion and to compute the
     * eigenvalues of the matrix. All those representations are redondant, that means
     * that doing the LU-decomposition, for example,  does NOT destroy
     * previously calculated type of storage.
     * \ingroup util
     */

    class Matrice : public MemoryMappable
    {
        // Elements
      private:
        /**
         * \c Dim_array of dimension 2 containing the size of the matrix.
         */
        Dim_array sizes;
        Array<double>* std; ///< Pointer on the array of the standard representation.

        mutable int ku; ///< Number of upper-diagonals in the band representation.
        mutable int kl; ///< Number of lower-diagonals in the band representation.

        /**
         * Pointer on the array of the band representation of a square matrix.
         * To be precise, \f$ A(i, j)\f$ is stored in \c band
         * \f$[ku+1+i-j, j]\f$ for \f$\mathrm {max}(1, j-ku) <= i <=
         * \mathrm{min} (n, j+kl)\f$, \e n being the size of the matrix.
         */
        mutable Array<double>* band;

        mutable Array<double>* lu;   ///< Pointer on the first array of the LU-representation.
        mutable Array<int>* permute; ///< Pointer on the second array of the LU-representation.

        // Constructeurs destructeurs
      public:
        /**
         * Standard constructor.
         * @param size1 [input] number of lines.
         * @param size2 [input] number of columns.
         */
        Matrice(int size1, int size2);

        Matrice(const Matrice&); ///< Constructor by copy.

        /**
         * Constructor from a \c Tbl.
         * @param tab [input]  2-dimension or 1-dimension array
         *
         * If \c tab is a 1-dimension \c Tbl, a single-column matrix is created,
         *  otherwise \c *std is simply constructed by a \c Tbl copy of \c tab.
         *
         */
        Matrice(const Array<double>& tab);

        ~Matrice(); ///< Destructor

        // Gestion memoire
        /**
         * Logical destructor : dellocates the memory of the various used
         * representations.
         *
         */
      private:
        void del_deriv(); ///< Deletes the (mutable) derived members: band, lu, permute

      public:
        /**
         * Sets all the coeficients of the \c *std to \e 0. The other representations are destroyed.
         */
        void annule();

      public:
        /**
         * Returns the dimension of the matrix.
         * @param i [input] if \e i =0 returns the number of lines and if \e i =2
         * returns the number of columns.
         */
        int get_dim(int i) const { return sizes(i); };

        Array<double>* get_lu() { return lu; };        ///< Returns a pointer  on the \c lu decomposition
        Array<int>* get_permute() { return permute; }; ///< Returns a pointer  on the permutation array.

        const Array<double>& get_array() const { return *std; }; ///< Returns the array of matrix elements

        // affectation
      public:
        /**
         * Sets all the element of \c *std to \e x.
         * The other representations are destroyed.
         */
        void operator=(double x);

        void operator=(const Matrice&);       ///< Assignement to another \c Matrice.
        void operator=(const Array<double>&); ///< Assignement to an array.

        // Impression
        friend ostream& operator<<(ostream&, const Matrice&);

        // extraction d'un element :
      public:
        /**
         * Read/write of a particuliar element.
         * This is done in \c *std and all the other representations are no
         * longer valid.
         * @param j [input] line coordinate.
         * @param i [input] column coordinate.
         *
         */
        double& set(int, int);

        /**
         * Copies the elements of \c so inside the matrice, starting at the position \f$ (i,j) \f$.
         **/
        void copy_inside(int i, int j, const Matrice& so);

        /**
         * Read-only of a particuliar element.
         * @param j [input] line coordinate.
         * @param i [input] column coordinate.
         */
        double operator()(int, int) const;

        // Passage matrice a bande
        /**
         * Calculate the band storage of \c *std.
         * Please note that this function does NOT check if \c *std
         * represents a real band-matrix.
         * @param up [input] number of upper-diagonals.
         * @param low [input] number of lower-diagonals.
         */
        void set_band(int up, int low) const;

        // Decomposition LU
        /**
         * Calculate the LU-representation,  assuming the band-storage has been
         * done. The calculus is done using \b LAPACK.
         */
        void set_lu() const;

        // Inversion de la matrice
        /**
         * Solves the linear system represented by the matrix.
         * The calculus assumes the the LU-decomposition has been done and is
         * conducted using \b LAPACK.
         * @param sec_membre [input] the right-hand side of the system.
         */
        [[nodiscard]] Array<double> solve(const Array<double>& sec_membre) const;

        // Les valeurs propres :
        /**
         * Returns the eigenvalues of the matrix, calculated using \b LAPACK.
         * @return contains the real and the imaginary parts of the
         * eigenvalues. The real parts are in \c Array<double> \e (0, *) and
         * the imaginary parts in \c Array<double> \e (1, *).
         */
        Array<double> val_propre() const;
        /**
         * Returns the eigenvectors of the matrix, calculated using \b LAPACK.
         *
         */
        Matrice vect_propre() const;

        /**
         * Computes the transpose matrix
         */
        Matrice transpose() const;

        // Member arithmetics
        // ------------------
      public:
        void operator+=(const Matrice&); ///< Operator +=
        void operator+=(double);         ///< Operator +=
        void operator-=(const Matrice&); ///< Operator -=
        void operator-=(double);         ///< Operator -=
        void operator*=(double);         ///< Operator *=
        void operator/=(double);         ///< Operator /=

        // Operateurs amis
        friend Matrice operator+(const Matrice&, const Matrice&);
        friend Matrice operator-(const Matrice&, const Matrice&);
        friend Matrice operator-(const Matrice&);
        friend Matrice operator*(const Matrice&, double);
        friend Matrice operator*(double, const Matrice&);
        friend Matrice operator*(const Matrice&, const Matrice&);
        friend Matrice operator/(const Matrice&, double);
    };
    ostream& operator<<(ostream&, const Matrice&);

    Matrice operator+(const Matrice&, const Matrice&);
    Matrice operator-(const Matrice&, const Matrice&);
    Matrice operator-(const Matrice&);
    Matrice operator*(const Matrice&, double);
    Matrice operator*(double, const Matrice&);
    Matrice operator*(const Matrice&, const Matrice&);
    Matrice operator/(const Matrice&, double);

// Lapack routines :
#define F77_dswap dswap_
#define F77_dgbtrf dgbtrf_
#define F77_dgbtrs dgbtrs_
#define F77_dgetrf dgetrf_
#define F77_dgetrs dgetrs_
#define F77_dgeev dgeev_
#define F77_dgeqp3 dgeqp3_

    extern "C" {
    void F77_dgbtrf(int*, int*, int*, int*, double[], int*, int[], int*);
    void F77_dgbtrs(char*, int*, int*, int*, int*, double[], int*, int[], double[], int*, int*);
    void F77_dgetrf(int*, int*, double[], int*, int[], int*);
    void F77_dgetrs(char*, int*, int*, double[], int*, int[], double[], int*, int*);
    void F77_dgeev(char*, char*, int*, double[], int*, double[], double[], double[], int*, double[], int*, double[],
                   int*, int*);
    void F77_dgeqp3(int*, int*, double[], int*, int[], double[], double[], int*, int*);
    }

} // namespace Kadath
