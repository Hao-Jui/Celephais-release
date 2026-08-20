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

#include "For_Kadath/Array/headcpp.hpp"
#include "For_Kadath/Array/exceptions.hpp"

#include <sstream>
#include "For_Kadath/Domain/bispheric.hpp"
#include "For_Kadath/Array/array.hpp"
#include "For_Kadath/Param/param.hpp"
#include "For_Kadath/Val_domain/val_domain.hpp"

namespace Kadath
{
    void Domain_bispheric_rect::find_other_dom(int dom, int bound, int& other_dom, int& other_bound) const
    {
        switch (bound) {
            case CHI_ONE_BC:
                if (eta_minus < 0)
                    other_dom = dom - 1;
                else
                    other_dom = dom + 1;
                other_bound = CHI_ONE_BC;
                break;
            case ETA_PLUS_BC:
                if (eta_minus < 0) {
                    other_dom = dom + 1;
                    other_bound = ETA_MINUS_BC;
                } else {
                    other_dom = dom - 1;
                    other_bound = ETA_PLUS_BC;
                }
                break;
            default:
                KADATH_THROW("Unknown boundary case in Domain_bispheric_rect::find_other_dom");
        }
    }

    double Domain_bispheric_rect::val_boundary(int bound, const Val_domain& so, const Index& pos_cf) const
    {

        if (so.check_if_zero())
            return 0.;

        else {
            so.coef();
            double res = 0;
            Index copie_pos(pos_cf);
            switch (bound) {
                case INNER_BC:
                    for (int i = 0; i < nbr_coefs(0); i++) {
                        copie_pos.set(0) = i;
                        if (i % 2 == 0)
                            res += so.get_coef(copie_pos);
                        else
                            res -= so.get_coef(copie_pos);
                    }
                    break;
                case ETA_PLUS_BC:
                    for (int i = 0; i < nbr_coefs(0); i++) {
                        copie_pos.set(0) = i;
                        res += so.get_coef(copie_pos);
                    }
                    break;
                case CHI_ONE_BC:
                    for (int j = 0; j < nbr_coefs(1); j++) {
                        copie_pos.set(1) = j;
                        res += so.get_coef(copie_pos);
                    }
                    break;
                case OUTER_BC:
                    for (int j = 0; j < nbr_coefs(1); j++) {
                        copie_pos.set(1) = j;
                        for (int i = 0; i < nbr_coefs(0); i++) {
                            copie_pos.set(0) = i;
                            res += so.get_coef(copie_pos);
                        }
                    }
                    break;
                default:
                    KADATH_THROW("Unknown boundary type in Domain_bispheric_rect::val_boundary");
            }
            return res;
        }
    }

    int Domain_bispheric_rect::nbr_points_boundary(int bound, const Base_spectral& bb) const
    {
        int basep = (*bb.bases_1d[2])(0);
        int nbrphi = (basep == COS) ? nbr_points(2) : nbr_points(2) - 2;
        int res;
        switch (bound) {
            case INNER_BC:
                res = nbrphi * (nbr_points(1) - 1);
                if (basep == COS)
                    res++;
                break;
            case OUTER_BC:
                res = nbrphi;
                break;
            default:
                std::ostringstream oss;
                oss << "Domain_bispheric_rect::nbr_points_boundary not yet implemented for boundary " << bound << endl;
                KADATH_THROW(oss.str());
        }

        return res;
    }

    void Domain_bispheric_rect::do_which_points_boundary(int bound, const Base_spectral& bb, Index** which_coef,
                                                         int start) const
    {
        int pos_which = start;
        Index pos(nbr_points);
        int basep = (*bb.bases_1d[2])(0);
        // Look at the symetrie :
        int mink = (basep == COS) ? 0 : 1;
        int maxk = (basep == COS) ? nbr_points(2) : nbr_points(2) - 1;
        int k, j;
        switch (bound) {
            case INNER_BC:
                pos.set(0) = 0;
                if (basep == COS) {
                    pos.set(2) = 0;
                    pos.set(1) = 0;
                    which_coef[pos_which] = new Index(pos);
                    pos_which++;
                }
                for (k = mink; k < maxk; k++) {
                    pos.set(2) = k;
                    for (j = 1; j < nbr_points(1); j++) {
                        pos.set(1) = j;
                        which_coef[pos_which] = new Index(pos);
                        pos_which++;
                    }
                }
                break;
            case OUTER_BC:
                pos.set(0) = nbr_points(0) - 1;
                pos.set(1) = nbr_points(1) - 1;
                for (k = mink; k < maxk; k++) {
                    pos.set(2) = k;
                    which_coef[pos_which] = new Index(pos);
                    pos_which++;
                }
                break;
            default:
                KADATH_THROW("Unknown boundary in Domain_bispheric_rect::do_which_points_inside");
        }
    }
} // namespace Kadath
