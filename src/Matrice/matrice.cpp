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
 *   2026-08-06  RAII/span modernization.
 */

#include "For_Kadath/Array/headcpp.hpp"
#include "For_Kadath/Array/exceptions.hpp"

#include <memory>
#include <sstream>
#include "For_Kadath/Matrice/matrice.hpp"
#include "For_Kadath/Array/array.hpp"
namespace Kadath
{
    // Destructeur des quantites derivees

    void Matrice::del_deriv()
    {
        if (band != nullptr)
            delete band;
        if (lu != nullptr)
            delete lu;
        if (permute != nullptr)
            delete permute;
        band = nullptr;
        lu = nullptr;
        permute = nullptr;
    }

    // sets to zero in std
    void Matrice::annule()
    {
        del_deriv();
        Index index(sizes);
        do
            std->set(index) = 0;
        while (index.inc());
    }

    // Standard Constructor
    Matrice::Matrice(int i, int j) : sizes(2)
    {
        sizes.set(0) = i;
        sizes.set(1) = j;
        std = new Array<double>(sizes);
        kl = 0;
        ku = 0;
        band = nullptr;
        lu = nullptr;
        permute = nullptr;
    }

    // Copy constructor
    Matrice::Matrice(const Matrice& source) : sizes(source.sizes)
    {
        kl = source.kl;
        ku = source.ku;
        std = new Array<double>(*source.std);
        if (source.band != nullptr)
            band = new Array<double>(*source.band);
        else
            band = nullptr;
        if (source.lu != nullptr)
            lu = new Array<double>(*source.lu);
        else
            lu = nullptr;
        if (source.permute != nullptr)
            permute = new Array<int>(*source.permute);
        else
            permute = nullptr;
    }

    // Copy from a 2 dimensional array
    Matrice::Matrice(const Array<double>& source) : sizes(source.get_dimensions())
    {
        std = new Array<double>(source);
        assert(sizes.get_ndim() == 2);
        kl = 0;
        ku = 0;
        band = nullptr;
        lu = nullptr;
        permute = nullptr;
    }

    // destructor
    Matrice::~Matrice()
    {
        del_deriv();
        delete std;
    }

    // Assignement to a double in the std representation
    void Matrice::operator=(double x)
    {
        del_deriv();
        *std = x;
    }

    // Assignement to another matrix
    void Matrice::operator=(const Matrice& source)
    {

        assert(sizes == source.sizes);
        del_deriv();
        delete std;
        std = new Array<double>(*source.std);
        kl = source.kl;
        ku = source.ku;
        if (source.band != nullptr)
            band = new Array<double>(*source.band);
        if (source.lu != nullptr)
            lu = new Array<double>(*source.lu);
        if (source.permute != nullptr)
            permute = new Array<int>(*source.permute);
    }

    // Assignement to a 2dimensional array
    void Matrice::operator=(const Array<double>& source)
    {

        assert(sizes == source.get_dimensions());
        del_deriv();
        delete std;
        std = new Array<double>(source);
        kl = 0;
        ku = 0;
    }

    // Read/write of an element
    double& Matrice::set(int i, int j)
    {
        del_deriv();
        Index index(sizes);
        index.set(0) = i;
        index.set(1) = j;
        return std->set(index);
    }

    void Matrice::copy_inside(int i, int j, const Matrice& so)
    {
        del_deriv();
        Index index_so(so.sizes);
        Index index(sizes);
        do {
            index.set(0) = index_so(0) + i;
            index.set(1) = index_so(1) + j;
            std->set(index) = (*so.std)(index_so);
        } while (index_so.inc());
    }

    // Read only of an element
    double Matrice::operator()(int i, int j) const
    {
        Index index(sizes);
        index.set(0) = i;
        index.set(1) = j;
        return (*std)(index);
    }

    // Display
    ostream& operator<<(ostream& flux, const Matrice& source)
    {

        flux << "Matrix " << source.sizes(0) << " * " << source.sizes(1) << endl;
        for (int i = 0; i < source.sizes(0); i++) {
            for (int j = 0; j < source.sizes(1); j++)
                flux << source(i, j) << "  ";
            flux << endl;
        }
        flux << endl;
        return flux;
    }

    // Computes the banded representation : LAPACK storage
    void Matrice::set_band(int u, int l) const
    {
        if (band != nullptr)
            return;
        else {
            int n = sizes(0);
            assert(n == sizes(1));

            ku = u;
            kl = l;
            int ldab = 2 * l + u + 1;
            band = new Array<double>(ldab * n);
            *band = 0;

            for (int i = 0; i < u; i++)
                for (int j = u - i; j < n; j++)
                    band->set(j * ldab + i + l) = (*this)(j - u + i, j);

            for (int j = 0; j < n; j++)
                band->set(j * ldab + u + l) = (*this)(j, j);

            for (int i = u + 1; i < u + l + 1; i++)
                for (int j = 0; j < n - i + u; j++)
                    band->set(j * ldab + i + l) = (*this)(i + j - u, j);
        }
        return;
    }

    // LU decomposition : LAPACK storage
    void Matrice::set_lu() const
    {
        if (lu != nullptr) {
            assert(permute != nullptr);
            return;
        } else {
            // LU decomposition
            int n = sizes(0);
            int ldab, info;
            permute = new Array<int>(n);

            // Case of a banded matrix
            if (band != nullptr) {
                ldab = 2 * kl + ku + 1;
                lu = new Array<double>(*band);

                F77_dgbtrf(&n, &n, &kl, &ku, lu->data, &ldab, permute->data, &info);
                if (info != 0) {
                    std::ostringstream oss;
                    oss << "LAPACK F77_dgbtrf failed with info=" << info << endl;
                    KADATH_THROW(oss.str());
                }
            } else { // General matrix
                ldab = n;
                lu = new Array<double>(*std);

                F77_dgetrf(&n, &n, lu->data, &ldab, permute->data, &info);
                if (info != 0) {
                    std::ostringstream oss;
                    oss << "LAPACK F77_dgetrf failed with info=" << info << endl;
                    KADATH_THROW(oss.str());
                }
            }
        }
        return;
    }

    // Solution of Ax = B : use LAPACK et the LU decomposition.
    Array<double> Matrice::solve(const Array<double>& source) const
    {

        assert(lu != nullptr);
        assert(permute != nullptr);

        int n = source.get_size(0);
        assert(sizes(1) == n);
        int ldab, info;
        char trans;
        int nrhs = 1;
        int ldb = n;

        Array<double> res(source);

        if (band != nullptr) { // Banded matrix
            ldab = 2 * kl + ku + 1;
            trans = 'N';
            F77_dgbtrs(&trans, &n, &kl, &ku, &nrhs, lu->data, &ldab, permute->data, res.data, &ldb, &info);
            if (info != 0) {
                std::ostringstream oss;
                oss << "LAPACK F77_dgbtrs failed with info=" << info << endl;
                KADATH_THROW(oss.str());
            }
        } else { // General case
            ldab = n;
            trans = 'T';
            F77_dgetrs(&trans, &n, &nrhs, lu->data, &ldab, permute->data, res.data, &ldb, &info);
            if (info != 0) {
                std::ostringstream oss;
                oss << "LAPACK F77_dgetrs failed with info=" << info << endl;
                KADATH_THROW(oss.str());
            }
        }

        return res;
    }

    // Eighenvalues of the matrix, use of LAPACK :
    Array<double> Matrice::val_propre() const
    {

        char jobvl = 'N';
        char jobvr = 'N';

        int n = sizes(0);
        assert(n == sizes(1));

        auto a = std::make_unique_for_overwrite<double[]>(n * n);
        Index index(sizes);
        for (int i = 0; i < n * n; i++) {
            a[i] = (*std)(index);
            index.inc();
        }

        int lda = n;
        auto wr = std::make_unique_for_overwrite<double[]>(n);
        auto wi = std::make_unique_for_overwrite<double[]>(n);

        int ldvl = 1;
        double* vl = nullptr;
        int ldvr = 1;
        double* vr = nullptr;

        int ldwork = 3 * n;
        auto work = std::make_unique_for_overwrite<double[]>(ldwork);

        int info;

        F77_dgeev(&jobvl, &jobvr, &n, a.get(), &lda, wr.get(), wi.get(), vl, &ldvl, vr, &ldvr, work.get(),
                  &ldwork, &info);
        if (info != 0) {
            std::ostringstream oss;
            oss << "LAPACK F77_dgeev failed with info=" << info << endl;
            KADATH_THROW(oss.str());
        }

        Dim_array res_out(2);
        res_out.set(0) = 2;
        res_out.set(1) = n;
        Array<double> result(res_out);
        Index index_out(res_out);
        for (int i = 0; i < n; i++) {
            result.set(index_out) = wr[n - i - 1];
            index_out.inc();
            result.set(index_out) = wi[n - i - 1];
            index_out.inc();
        }

        return result;
    }

    // Eighenvectors of the matrix (use of LAPACK) :
    Matrice Matrice::vect_propre() const
    {

        assert(std != nullptr);

        char jobvl = 'V';
        char jobvr = 'V';

        int n = sizes(0);
        assert(n == sizes(1));

        auto a = std::make_unique_for_overwrite<double[]>(n * n);
        Index index(sizes);
        for (int i = 0; i < n * n; i++) {
            a[i] = (*std)(index);
            index.inc();
        }

        int lda = n;
        auto wr = std::make_unique_for_overwrite<double[]>(n);
        auto wi = std::make_unique_for_overwrite<double[]>(n);

        int ldvl = n;
        auto vl = std::make_unique_for_overwrite<double[]>(ldvl * ldvl);
        int ldvr = n;
        auto vr = std::make_unique_for_overwrite<double[]>(ldvl * ldvl);

        int ldwork = 4 * n;
        auto work = std::make_unique_for_overwrite<double[]>(ldwork);

        int info;

        F77_dgeev(&jobvl, &jobvr, &n, a.get(), &lda, wr.get(), wi.get(), vl.get(), &ldvl, vr.get(), &ldvr,
                  work.get(), &ldwork, &info);
        if (info != 0) {
            std::ostringstream oss;
            oss << "LAPACK F77_dgeev failed with info=" << info << endl;
            KADATH_THROW(oss.str());
        }

        Matrice res(n, n);

        int conte = 0;
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++) {
                res.set(j, n - i - 1) = vr[conte];
                conte++;
            }

        return res;
    }

    // Transposed matrix
    Matrice Matrice::transpose() const
    {

        int nbl = sizes(1);
        int nbc = sizes(0);

        Matrice resu(nbc, nbl);

        Index index(std->get_dimensions());
        for (int i = 0; i < nbc; i++)
            for (int j = 0; j < nbl; j++) {
                index.set(0) = j;
                index.set(1) = i;
                resu.set(i, j) = (*std)(index);
            }
        return resu;
    }

    void Matrice::operator+=(const Matrice& a)
    {
        assert((std != nullptr) && (a.std != nullptr));
        std->operator+=(*a.std);
    }

    void Matrice::operator-=(const Matrice& a)
    {
        assert((std != nullptr) && (a.std != nullptr));
        std->operator-=(*a.std);
    }

    void Matrice::operator+=(double x)
    {
        assert(std != nullptr);
        std->operator+=(x);
    }

    void Matrice::operator-=(double x)
    {
        assert(std != nullptr);
        std->operator-=(x);
    }

    void Matrice::operator*=(double x)
    {
        assert(std != nullptr);
        std->operator*=(x);
    }

    void Matrice::operator/=(double x)
    {
        assert(std != nullptr);
        assert(x != 0);
        std->operator/=(x);
    }

    Matrice operator+(const Matrice& a, const Matrice& b)
    {
        assert((a.std != nullptr) && (b.std != nullptr));
        Matrice res(*a.std + *b.std);
        return res;
    }

    Matrice operator-(const Matrice& a, const Matrice& b)
    {
        assert((a.std != nullptr) && (b.std != nullptr));
        Matrice res(*a.std - *b.std);
        return res;
    }

    Matrice operator-(const Matrice& a)
    {
        assert(a.std != nullptr);
        Matrice res(-*a.std);
        return res;
    }

    Matrice operator*(const Matrice& a, double x)
    {
        assert(a.std != nullptr);
        Matrice res(*a.std * x);
        return res;
    }

    Matrice operator*(double x, const Matrice& a)
    {
        assert(a.std != nullptr);
        Matrice res(*a.std * x);
        return res;
    }

    Matrice operator*(const Matrice& aa, const Matrice& bb)
    {

        int nbla = aa.sizes(1);
        int nbca = aa.sizes(0);
        int nbcb = bb.sizes(0);

        assert(nbca == bb.sizes(1));

        Matrice resu(nbla, nbcb);

        for (int i = 0; i < nbla; i++)
            for (int j = 0; j < nbcb; j++) {
                double sum = 0;
                for (int k = 0; k < nbca; k++) {
                    sum += aa(i, k) * bb(k, j);
                }
                resu.set(i, j) = sum;
            }
        return resu;
    }

    Matrice operator/(const Matrice& a, double x)
    {
        assert(x != 0);
        assert(a.std != nullptr);
        Matrice res(*a.std / x);
        return res;
    }
} // namespace Kadath
