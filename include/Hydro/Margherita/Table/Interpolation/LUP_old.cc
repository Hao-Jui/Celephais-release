//
// This file is part of Margherita, the light-weight EOS framework
//
//  Copyright (C) 2017, Elias Roland Most
//                      <emost@th.physik.uni-frankfurt.de>
//
//  This program is free software: you can redistribute it and/or modify
//  it under the terms of the GNU General Public License as published by
//  the Free Software Foundation, either version 3 of the License, or
//  (at your option) any later version.
//
//  This program is distributed in the hope that it will be useful,
//  but WITHOUT ANY WARRANTY; without even the implied warranty of
//  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//  GNU General Public License for more details.
//
//  You should have received a copy of the GNU General Public License
//  along with this program.  If not, see <http://www.gnu.org/licenses/>.
//
//

/*
 * Modifications (Celephais):
 *   2026-06-16  Modified for the Celephais tree; see
 *               PATCHES-KADATH-UPSTREAM.md and LICENSE_SOURCE_AUDIT.tsv.
 */

#pragma once

#include <array>
#include <cmath>
#include <vector>

namespace LUP_old {

template<typename T>
using  matrix_ptr = double * __restrict__ * __restrict__ ;
template<typename T>
using vec_ptr = double * __restrict__;
typedef int * __restrict__ int_ptr;


static constexpr int polish_iterations = 3;
template<typename T>
int LUdecompose(int dim, matrix_ptr<T> A,matrix_ptr<T> LU,int_ptr index){

  const double eps = 1.0e-40;
  std::vector<T> largest_row_element(dim);
  // initialise LU with A
  for (int i = 0; i < dim; i++) {
    index[i] = i;
    for (int j = 0; j < dim; j++)
      LU[i][j] = A[i][j];
  }
  // Compute largest element of each row
  // This is later used for the pivoting
  for (int i = 0; i < dim; i++) {
    largest_row_element[i] = 0.0;
    for (int j = 0; j < dim; j++) {
      if (fabs(A[i][j]) > largest_row_element[i])
        largest_row_element[i] = fabs(A[i][j]);
    }
  }
  // Now we need to iterate over all columns
  for (int k = 0; k < dim; k++) {
    int row_with_largest_element = k; // No swap
    bool swap = false;
    // Need to compute the largest element in the column
    // to decide whether to swap rows or not.
    T largest_column_element_weight = 0.0;
    for (int i = k; i < dim; i++) {
      T current_element_weight = fabs(LU[i][k]) / largest_row_element[i];
      if (current_element_weight > largest_column_element_weight) {
        largest_column_element_weight = current_element_weight;
        row_with_largest_element = i;
        swap = true;
      }
    }
    // Now we need to interchange rows to get the largest weighted row element
    // as pivot
    if (swap) {
      for (int j = 0; j < dim; j++) {
        // swap elementwise
        T tmp = LU[row_with_largest_element][j];
        LU[row_with_largest_element][j] = LU[k][j];
        LU[k][j] = tmp;
      }
      largest_row_element[row_with_largest_element] = largest_row_element[k];
    }
    // This helps preventing a division by zero.
    if (LU[k][k] == 0.0)
      LU[k][k] = eps;
    // Compute decomposition
    index[k] = row_with_largest_element;
    for (int i = k + 1; i < dim; i++) {
      LU[i][k] /= LU[k][k];
      for (int j = k + 1; j < dim; j++) {
        LU[i][j] -= LU[i][k] * LU[k][j];
      }
    }
  }
  return 0;
}

template<typename T>
inline int LUsolve(int dim, matrix_ptr<T> LU,vec_ptr<T> b,vec_ptr<T> x,int_ptr index){
  for (int i = 0; i < dim; i++)
    x[i] = b[i];
  // solve U*y=b
  for (int i = 0; i < dim; i++) {
    if (i == 0) {
      T tmp = x[i];
      x[i] = x[index[i]];
      x[index[i]] = tmp;
      continue;
    }
    double tmp = x[index[i]];
    x[index[i]] = x[i];
    x[i] = tmp;
    for (int j = i - 1; j < i; j++) {
      x[i] -= LU[i][j] * x[j];
    }
  }
  // Solve Lx=y
  for (int i = dim - 1; i >= 0; i--) {
    for (int j = i + 1; j < dim; j++) {
      x[i] -= LU[i][j] * x[j];
    }
    x[i] /= LU[i][i];
  }
  return 0;
}

template<typename T>
inline void MatrixMultiply(int dim, matrix_ptr<T> &A, vec_ptr<T> &x,
                           vec_ptr<T> &y) {
  for (int i = 0; i < dim; i++) {
    y[i] = 0.0;
    for (int j = 0; j < dim; j++)
      y[i] += A[i][j] * x[j];
  }
}

template <typename T, int iterations>
int LUpolish(int dim, matrix_ptr<T> A,matrix_ptr<T> LU, vec_ptr<T> x,
             vec_ptr<T> b,vec_ptr<T> rhs,vec_ptr<T> dx,int_ptr index){
  for (int it = 0; it < iterations; it++) {
    MatrixMultiply<T>(dim, A, x, rhs);
    for (int i = 0; i < dim; i++)
      rhs[i] -= b[i];
    LUsolve<T>(dim, LU, rhs, dx, index);
    for (int i = 0; i < dim; i++)
      x[i] -= dx[i];
  }
  return 0;
}
}

