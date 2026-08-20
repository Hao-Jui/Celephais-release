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

#include "For_Kadath/Tensor/tensor.hpp"
#include "For_Kadath/Array/exceptions.hpp"
namespace Kadath
{
    Param_tensor::Param_tensor() : m_order_affected(false), m_order(0), m_quant_affected(false), m_quant(0) {}

    Param_tensor::Param_tensor(const Param_tensor& so)
        : m_order_affected(so.m_order_affected), m_order(so.m_order), m_quant_affected(so.m_quant_affected),
          m_quant(so.m_quant)
    {
    }

    Param_tensor::~Param_tensor() {}

    int Param_tensor::get_m_order() const
    {
        if (!m_order_affected) {
            KADATH_THROW("m_order not affected in Param_tensor");
        } else
            return m_order;
    }

    int Param_tensor::get_m_quant() const
    {
        if (!m_quant_affected) {
            KADATH_THROW("m_quant not affected in Param_tensor");
        } else
            return m_quant;
    }

    int& Param_tensor::set_m_quant()
    {
        m_quant_affected = true;
        return m_quant;
    }

    int& Param_tensor::set_m_order()
    {
        m_order_affected = true;
        return m_order;
    }

    int add_m_quant(const Param_tensor* aa, const Param_tensor* bb)
    {
        int res = 0;
        if ((aa == nullptr) || (bb == nullptr))
            res = 0;
        else {
            int val_a = (aa->m_quant_affected) ? aa->m_quant : 0;
            int val_b = (bb->m_quant_affected) ? bb->m_quant : 0;
            res = (val_a < val_b) ? val_a : val_b;
        }
        return res;
    }

    int inv_m_quant(const Param_tensor* aa)
    {
        if (aa == nullptr)
            return 0;
        int val_a = (aa->m_quant_affected) ? aa->m_quant : 0;
        return (-val_a);
    }

    int mult_m_quant(const Param_tensor* aa, const Param_tensor* bb)
    {
        int res = 0;
        if ((aa == nullptr) && (bb == nullptr))
            return 0;
        if (aa == nullptr) {
            res = (bb->m_quant_affected) ? bb->m_quant : 0;
            return res;
        }
        if (bb == nullptr) {
            res = (aa->m_quant_affected) ? aa->m_quant : 0;
            return res;
        }
        int val_a = (aa->m_quant_affected) ? aa->m_quant : 0;
        int val_b = (bb->m_quant_affected) ? bb->m_quant : 0;
        return (val_a + val_b);
    }
} // namespace Kadath
