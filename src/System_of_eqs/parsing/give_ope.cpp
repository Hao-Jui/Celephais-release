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

#include "For_Kadath/System_of_eqs/system_of_eqs.hpp"
#include "equation_parser.hpp"
#include "For_Kadath/Array/exceptions.hpp"

#include <sstream>
#include "For_Kadath/Utilities/name_tools.hpp"
#include "For_Kadath/Term_eq/term_eq.hpp"
#include "For_Kadath/Base_tensor/base_tensor.hpp"
namespace Kadath
{
    bool System_of_eqs::isvar_double(const char* name, int& which) const
    {
        bool res = false;
        char auxi[LMAX];
        trim_spaces(auxi, name);
        for (int i = 0; i < nvar_double; i++) {
            if (!res)
                if (strcmp(auxi, names_var_double[i].c_str()) == 0) {
                    res = true;
                    which = i;
                }
        }
        return res;
    }

    bool System_of_eqs::isvar(const char* name, int& which, int& valence, char*& name_ind, Array<int>*& type_ind) const
    {
        bool res = false;
        for (int i = 0; i < nvar; i++) {
            if (name_ind != nullptr) {
                delete[] name_ind;
                name_ind = nullptr;
            }
            if (type_ind != nullptr) {
                delete type_ind;
                type_ind = nullptr;
            }
            res = is_tensor(name, names_var[i].c_str(), valence, name_ind, type_ind);
            if (res) {
                which = i;
                if (valence != var[which]->get_valence()) {
                    std::ostringstream oss;
                    oss << "Bad valence for " << name << endl;
                    KADATH_THROW(oss.str());
                }
                break;
            }
        }
        return res;
    }

    bool System_of_eqs::iscst(const char* name, int& which, int& valence, char*& name_ind, Array<int>*& type_ind) const
    {
        bool res = false;
        for (int i = 0; i < ncst; i++) {

            // Type of constant
            int ind = i * ndom;
            int type_cst = cst[ind]->get_type_data();

            switch (type_cst) {
                case TERM_T:
                    if (name_ind != nullptr) {
                        delete[] name_ind;
                        name_ind = nullptr;
                    }
                    if (type_ind != nullptr) {
                        delete type_ind;
                        type_ind = nullptr;
                    }
                    res = is_tensor(name, names_cst[i].c_str(), valence, name_ind, type_ind);
                    if (res) {
                        which = i;
                        if (valence != cst[ind]->get_val_t().get_valence()) {
                            std::ostringstream oss;
                            oss << "Bad valence for " << name << endl;
                            KADATH_THROW(oss.str());
                        }
                    }
                    break;
                case TERM_D:
                    if (strcmp(names_cst[i].c_str(), name) == 0) {
                        res = true;
                        which = i;
                    }
                    break;
                default:
                    KADATH_THROW("Unknown type of data in System_of_eqs::iscst");
            }
            if (res)
                break;
        }
        return res;
    }

    bool System_of_eqs::isdef(int dd, const char* name, int& which, int& valence, char*& name_ind,
                              Array<int>*& type_ind) const
    {
        bool res = false;
        for (int i = 0; i < ndef; i++) {
            if (name_ind != nullptr) {
                delete[] name_ind;
                name_ind = nullptr;
            }
            if (type_ind != nullptr) {
                delete type_ind;
                type_ind = nullptr;
            }
            res = is_tensor(name, names_def[i].c_str(), valence, name_ind, type_ind);
            if ((res) && (def[i]->get_dom() == dd)) {
                which = i;
                break;
            }
            res = false;
        }
        return res;
    }

    bool System_of_eqs::isdef_glob(int dd, const char* name, int& which) const
    {
        bool res = false;
        for (int i = 0; i < ndef_glob; i++) {
            if ((strcmp(names_def_glob[i].c_str(), name) == 0) && (def_glob[i]->get_dom() == dd)) {
                res = true;
                which = i;
                break;
            }
        }
        return res;
    }

    bool System_of_eqs::isdouble(const char* name, double& val) const
    {
        char* error;
        val = strtod(name, &error);
        bool res = ((*error != ' ') || (strlen(error) > 1)) ? false : true;
        return res;
    }

    bool System_of_eqs::ismet(const char* name, char*& name_ind, int& type_indice) const
    {
        if (met == nullptr)
            return false;
        int valence;
        Array<int>* type_ind = nullptr;
        bool res = is_tensor(name, name_met.c_str(), valence, name_ind, type_ind);
        if (res) {
            if (valence != 2) {
                std::ostringstream oss;
                oss << "Bad valence for the metric " << name_met << " in " << name << endl;
                KADATH_THROW(oss.str());
            }

            if ((*type_ind)(0) != (*type_ind)(1)) {
                KADATH_THROW("Indices of the metric must be of the same type");
            }

            type_indice = (*type_ind)(0);
            delete type_ind;
        }
        return res;
    }

    bool System_of_eqs::ismet(const char* name) const
    {

        char auxi[LMAX];
        trim_spaces(auxi, name);

        if (met == nullptr)
            return false;
        int same = strcmp(auxi, name_met.c_str());

        if (same != 0)
            return false;
        else
            return true;
    }

    bool System_of_eqs::ischristo(const char* name, char*& name_ind, Array<int>*& type_ind) const
    {

        if (met == nullptr)
            return false;
        int valence;
        bool res = is_tensor(name, "Gam ", valence, name_ind, type_ind);
        if (res) {
            if (valence != 3) {
                std::ostringstream oss;
                oss << "Bad valence for Christoffel symbol in " << name << endl;
                KADATH_THROW(oss.str());
            }

            if (((*type_ind)(0) != COV) || ((*type_ind)(1) != COV) || ((*type_ind)(2) != CON)) {
                std::ostringstream oss;
                oss << "Indices of the wrong type in " << name << endl;
                KADATH_THROW(oss.str());
            }
        }
        return res;
    }

    bool System_of_eqs::isriemann(const char* name, char*& name_ind, Array<int>*& type_ind) const
    {

        if (met == nullptr)
            return false;
        int valence;
        bool res = is_tensor(name, "R ", valence, name_ind, type_ind);
        if (res)
            if (valence != 4)
                res = false;
        if (res) {

            if (((*type_ind)(0) != CON) || ((*type_ind)(1) != COV) || ((*type_ind)(2) != COV) ||
                ((*type_ind)(3) != COV)) {
                delete[] name_ind;
                delete type_ind;
                res = false;
            }
        }

        if (!res) {
            if (name_ind != nullptr)
                delete[] name_ind;
            if (type_ind != nullptr)
                delete type_ind;
        }
        return res;
    }

    bool System_of_eqs::isricci_tensor(const char* name, char*& name_ind, Array<int>*& type_ind) const
    {

        if (met == nullptr)
            return false;
        int valence;
        bool res = is_tensor(name, "R ", valence, name_ind, type_ind);
        if ((res) && (valence != 2)) {
            delete[] name_ind;
            delete type_ind;
            res = false;
        }

        if (!res) {
            if (name_ind != nullptr)
                delete[] name_ind;
            if (type_ind != nullptr)
                delete type_ind;
        }
        return res;
    }

    bool System_of_eqs::isricci_scalar(const char* name, char*& name_ind, Array<int>*& type_ind) const
    {

        if (met == nullptr)
            return false;
        int valence;
        bool res = is_tensor(name, "R ", valence, name_ind, type_ind);
        if ((res) && (valence != 0)) {
            delete[] name_ind;
            delete type_ind;
            res = false;
        }

        if (!res) {
            if (name_ind != nullptr)
                delete[] name_ind;
            if (type_ind != nullptr)
                delete type_ind;
        }
        return res;
    }

    // is_ope_bin remains on System_of_eqs because Space/*/*_add_eq.cpp call
    // sys.is_ope_bin(..., '=') to split equation strings. Other is_ope_*
    // helpers live on EquationParser only.
    bool System_of_eqs::is_ope_bin(const char* name, char* part1, char* part2, char cible) const
    {
        return EquationParser{*this}.is_ope_bin(name, part1, part2, cible);
    }

    Ope_eq* System_of_eqs::find_ope(int dd, const char* name, int bound) const
    {
        return EquationParser{*this}.find_ope(dd, name, bound);
    }

    Ope_eq* System_of_eqs::give_ope(int dd, const char* name, int bound) const
    {
        auto p_ope = find_ope(dd, name, bound);
        if (p_ope == nullptr) {
            std::ostringstream oss;
            oss << "Unknown operator " << name << endl;
            KADATH_THROW(oss.str());
        }
        return p_ope;
    }

} // namespace Kadath
