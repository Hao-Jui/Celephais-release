// Recursive give_ope calls go through System_of_eqs::give_ope so the
// throw-on-unknown wrapper stays in one place.

#include "equation_parser.hpp"

#include "For_Kadath/System_of_eqs/system_of_eqs.hpp"
#include "For_Kadath/Array/exceptions.hpp"
#include "For_Kadath/Utilities/name_tools.hpp"
#include "For_Kadath/Ope_eq/ope_eq.hpp"
#include "For_Kadath/Term_eq/term_eq.hpp"
#include "For_Kadath/Base_tensor/base_tensor.hpp"
#include "For_Kadath/Metric/metric.hpp"
#include "For_Kadath/Space/space.hpp"

#include <cassert>
#include <cstring>
#include <cstdlib>

namespace Kadath {

bool EquationParser::is_ope_bin(const char* name, char* part1, char* part2, char cible) const
{
    assert((cible == '+') || (cible == '-') || (cible == '*') || (cible == '/') || (cible == '='));

    bool res = false;
    int nbr_cible = nbr_char(name, cible);
    int conte = 0;

    while ((!res) && (conte < nbr_cible)) {
        get_parts(name, part1, part2, cible, conte);
        if ((part1[0] != '\0') && (part2[0] != '\0'))
            if ((nbr_char(part1, '(') == nbr_char(part1, ')')) && (nbr_char(part2, '(') == nbr_char(part2, ')')))
                res = true;
        conte++;
    }
    return res;
}

bool EquationParser::is_ope_minus(const char* name, char* part) const
{
    char auxi[LMAX];
    trim_spaces(auxi, name);
    int len = static_cast<int>(strlen(auxi));

    // Bare "-" (len==1 after trim) is not a unary-minus expression — must
    // bail before the buffer manipulation below dereferences occi[len - 2]
    // == occi[-1].
    if (len < 2)
        return false;

    bool res = true;
    if (auxi[0] != '-')
        res = false;
    if (res) {
        char occi[LMAX];
        for (int i = 1; i < len - 1; i++)
            occi[i - 1] = auxi[i];
        occi[len - 2] = ' ';
        occi[len - 1] = '\0';
        trim_spaces(part, occi);
    }
    return res;
}

bool EquationParser::is_ope_uni(const char* name, char* part, const char* zeope) const
{
    char auxi[LMAX];
    trim_spaces(auxi, zeope);
    int len = static_cast<int>(strlen(auxi));

    bool res = true;
    for (int i = 0; i < len - 1; i++)
        if (name[i] != auxi[i]) {
            res = false;
            break;
        }

    if (res)
        if (name[len - 1] != '(')
            res = false;

    if (res) {
        int lentot = static_cast<int>(strlen(name));
        assert(name[lentot - 1] == ' ');
        if (name[lentot - 2] != ')')
            res = false;

        for (int i = len; i < lentot - 2; i++)
            part[i - len] = name[i];

        part[lentot - len - 2] = ' ';
        part[lentot - len - 1] = '\0';
    }

    return res;
}

bool EquationParser::is_ope_uni(const char* name, char* p1, char* p2, const char* zeope) const
{
    char auxi[LMAX];
    trim_spaces(auxi, zeope);
    int len = static_cast<int>(strlen(auxi));
    bool res = true;
    for (int i = 0; i < len - 2; i++)
        if (name[i] != auxi[i]) {
            res = false;
            break;
        }
    if (res)
        if (name[len - 1] != '(')
            res = false;

    if (res) {
        char part[LMAX];
        int lentot = static_cast<int>(strlen(name));
        assert(name[lentot - 1] == ' ');
        if (name[lentot - 2] != ')')
            res = false;

        for (int i = len; i < lentot - 2; i++)
            part[i - len] = name[i];

        part[lentot - len - 2] = ' ';
        part[lentot - len - 1] = '\0';

        get_parts(part, p1, p2, ',');
    }
    return res;
}

bool EquationParser::is_ope_deriv(const char* nn, char* part, int& type_der, char& name_ind) const
{
    char name[LMAX];
    trim_spaces(name, nn);
    bool res = true;
    int len = static_cast<int>(strlen(name));
    if (name[0] != 'D')
        res = false;
    if (res) {
        if (name[1] == '_')
            type_der = COV;
        else {
            if (name[1] == '^')
                type_der = CON;
            else
                res = false;
        }
    }

    if (res) {
        name_ind = name[2];

        char auxi[LMAX];
        for (int i = 3; i < len; i++)
            auxi[i - 3] = name[i];
        auxi[len - 3] = '\0';
        trim_spaces(part, auxi);
    }
    return res;
}

bool EquationParser::is_ope_deriv_flat(const char* nn, char* part, int& type_der, char& name_ind) const
{
    char name[LMAX];
    trim_spaces(name, nn);
    bool res = true;
    int len = static_cast<int>(strlen(name));
    if (name[0] != 'D')
        res = false;
    if (res && name[1] != 'F')
        res = false;
    if (res) {
        if (name[2] == '_')
            type_der = COV;
        else {
            if (name[2] == '^')
                type_der = CON;
            else
                res = false;
        }
    }

    if (res) {
        name_ind = name[3];

        char auxi[LMAX];
        for (int i = 4; i < len; i++)
            auxi[i - 4] = name[i];
        auxi[len - 4] = '\0';
        trim_spaces(part, auxi);
    }
    return res;
}

bool EquationParser::is_ope_deriv_background(const char* nn, char* part, int& type_der, char& name_ind) const
{
    char name[LMAX];
    trim_spaces(name, nn);
    bool res = true;
    int len = static_cast<int>(strlen(name));
    if (name[0] != 'D')
        res = false;
    if (res && name[1] != 'B')
        res = false;
    if (res) {
        if (name[2] == '_')
            type_der = COV;
        else {
            if (name[2] == '^')
                type_der = CON;
            else
                res = false;
        }
    }

    if (res) {
        name_ind = name[3];

        char auxi[LMAX];
        for (int i = 4; i < len; i++)
            auxi[i - 4] = name[i];
        auxi[len - 4] = '\0';
        trim_spaces(part, auxi);
    }
    return res;
}

bool EquationParser::is_ope_partial(const char* nn, char* part, char& name_ind) const
{
    char name[LMAX];
    trim_spaces(name, nn);
    bool res = true;
    int len = static_cast<int>(strlen(name));

    if (strncmp(name, "partial", 7) != 0)
        res = false;
    if (res)
        if (name[7] != '_')
            res = false;

    if (res) {
        name_ind = name[8];

        char auxi[LMAX];
        for (int i = 9; i < len; i++)
            auxi[i - 9] = name[i];
        auxi[len - 9] = '\0';
        trim_spaces(part, auxi);
    }
    return res;
}

bool EquationParser::is_ope_pow(const char* name, char* part, int& expo) const
{
    char auxi[LMAX];
    trim_spaces(auxi, name);

    bool res = (nbr_char(auxi, '^') >= 1) ? true : false;

    if (res) {
        char p2[LMAX];
        get_parts(auxi, part, p2, '^');

        char* error;
        expo = static_cast<int>(strtol(p2, &error, 0));
        res = ((*error != ' ') || (strlen(error) > 1)) ? false : true;
    }

    return res;
}

bool EquationParser::is_ope_der_var(int dd, const char* name, char* part, int& numvar) const
{
    char auxi[LMAX];
    trim_spaces(auxi, name);

    bool res = (nbr_char(name, ',') == 1) ? true : false;
    if (res) {
        char p2[LMAX];
        get_parts(auxi, part, p2, ',');

        int isitvar = system_.espace.get_domain(dd)->give_place_var(p2);
        if (isitvar == -1)
            res = false;
        else
            numvar = isitvar;
    }
    return res;
}

Ope_eq* EquationParser::find_ope(int dd, const char* name, int bound) const
{
    Ope_eq* p_ope = nullptr;
    bool indic;
    int which = -1;
    char p1[LMAX];
    char p2[LMAX];

    indic = is_ope_bin(name, p1, p2, '+');
    if (indic) {
        p_ope = new Ope_add(&system_, system_.give_ope(dd, p1, bound), system_.give_ope(dd, p2, bound));
        return p_ope;
    }

    indic = is_ope_bin(name, p1, p2, '-');
    if (indic) {
        p_ope = new Ope_sub(&system_, system_.give_ope(dd, p1, bound), system_.give_ope(dd, p2, bound));
        return p_ope;
    }

    indic = is_ope_minus(name, p1);
    if (indic) {
        p_ope = new Ope_minus(&system_, system_.give_ope(dd, p1, bound));
        return p_ope;
    }

    indic = is_ope_bin(name, p1, p2, '*');
    if (indic) {
        p_ope = new Ope_mult(&system_, system_.give_ope(dd, p1, bound), system_.give_ope(dd, p2, bound));
        return p_ope;
    }

    indic = is_ope_bin(name, p1, p2, '/');
    if (indic) {
        p_ope = new Ope_div(&system_, system_.give_ope(dd, p1, bound), system_.give_ope(dd, p2, bound));
        return p_ope;
    }

    indic = is_ope_uni(name, p1, "Lap");
    if (indic) {
        p_ope = new Ope_lap(&system_, system_.give_ope(dd, p1, bound));
        return p_ope;
    }

    indic = is_ope_uni(name, p1, "lap");
    if (indic) {
        p_ope = new Ope_lap(&system_, system_.give_ope(dd, p1, bound));
        return p_ope;
    }

    indic = is_ope_uni(name, p1, "Lap2");
    if (indic) {
        p_ope = new Ope_lap2(&system_, system_.give_ope(dd, p1, bound));
        return p_ope;
    }

    indic = is_ope_uni(name, p1, "lap2");
    if (indic) {
        p_ope = new Ope_lap2(&system_, system_.give_ope(dd, p1, bound));
        return p_ope;
    }

    indic = is_ope_uni(name, p1, "dn");
    if (indic) {
        p_ope = new Ope_dn(&system_, bound, system_.give_ope(dd, p1, bound));
        return p_ope;
    }

    indic = is_ope_uni(name, p1, "multx");
    if (indic) {
        p_ope = new Ope_mult_x(&system_, system_.give_ope(dd, p1, bound));
        return p_ope;
    }

    indic = is_ope_uni(name, p1, "multr");
    if (indic) {
        p_ope = new Ope_mult_r(&system_, system_.give_ope(dd, p1, bound));
        return p_ope;
    }

    indic = is_ope_uni(name, p1, "mult1mrsL");
    if (indic) {
        p_ope = new Ope_mult_1mrsL(&system_, system_.give_ope(dd, p1, bound));
        return p_ope;
    }

    indic = is_ope_uni(name, p1, "div1mrsL");
    if (indic) {
        p_ope = new Ope_div_1mrsL(&system_, system_.give_ope(dd, p1, bound));
        return p_ope;
    }

    indic = is_ope_uni(name, p1, "valori");
    if (indic) {
        p_ope = new Ope_val_ori(&system_, dd, system_.give_ope(0, p1, bound));
        return p_ope;
    }

    indic = is_ope_uni(name, p1, "srdr");
    if (indic) {
        p_ope = new Ope_srdr(&system_, system_.give_ope(dd, p1, bound));
        return p_ope;
    }

    indic = is_ope_uni(name, p1, "ddp");
    if (indic) {
        p_ope = new Ope_ddp(&system_, system_.give_ope(dd, p1, bound));
        return p_ope;
    }

    indic = is_ope_uni(name, p1, "ddt");
    if (indic) {
        p_ope = new Ope_ddt(&system_, system_.give_ope(dd, p1, bound));
        return p_ope;
    }

    indic = is_ope_uni(name, p1, "dt");
    if (indic) {
        p_ope = new Ope_dt(&system_, system_.give_ope(dd, p1, bound));
        return p_ope;
    }

    indic = is_ope_uni(name, p1, "dtime");
    if (indic) {
        p_ope = new Ope_dtime(&system_, system_.give_ope(dd, p1, bound));
        return p_ope;
    }

    indic = is_ope_uni(name, p1, "ddtime");
    if (indic) {
        p_ope = new Ope_ddtime(&system_, system_.give_ope(dd, p1, bound));
        return p_ope;
    }

    indic = is_ope_uni(name, p1, "ddr");
    if (indic) {
        p_ope = new Ope_ddr(&system_, system_.give_ope(dd, p1, bound));
        return p_ope;
    }

    indic = is_ope_uni(name, p1, "dr");
    if (indic) {
        p_ope = new Ope_dr(&system_, system_.give_ope(dd, p1, bound));
        return p_ope;
    }

    indic = is_ope_uni(name, p1, "divr");
    if (indic) {
        p_ope = new Ope_div_r(&system_, system_.give_ope(dd, p1, bound));
        return p_ope;
    }

    indic = is_ope_uni(name, p1, "multrsint");
    if (indic) {
        p_ope = new Ope_mult_rsint(&system_, system_.give_ope(dd, p1, bound));
        return p_ope;
    }

    indic = is_ope_uni(name, p1, "divrsint");
    if (indic) {
        p_ope = new Ope_div_rsint(&system_, system_.give_ope(dd, p1, bound));
        return p_ope;
    }

    indic = is_ope_uni(name, p1, "divsint");
    if (indic) {
        p_ope = new Ope_div_sint(&system_, system_.give_ope(dd, p1, bound));
        return p_ope;
    }

    indic = is_ope_uni(name, p1, "divcost");
    if (indic) {
        p_ope = new Ope_div_cost(&system_, system_.give_ope(dd, p1, bound));
        return p_ope;
    }

    indic = is_ope_uni(name, p1, "multsint");
    if (indic) {
        p_ope = new Ope_mult_sint(&system_, system_.give_ope(dd, p1, bound));
        return p_ope;
    }

    indic = is_ope_uni(name, p1, "multcosp");
    if (indic) {
        p_ope = new Ope_mult_cosp(&system_, system_.give_ope(dd, p1, bound));
        return p_ope;
    }

    indic = is_ope_uni(name, p1, "divxpone");
    if (indic) {
        p_ope = new Ope_div_xpone(&system_, system_.give_ope(dd, p1, bound));
        return p_ope;
    }

    indic = is_ope_uni(name, p1, "div1mx2");
    if (indic) {
        p_ope = new Ope_div_1mx2(&system_, system_.give_ope(dd, p1, bound));
        return p_ope;
    }

    indic = is_ope_uni(name, p1, "integ");
    if (indic) {
        p_ope = new Ope_int(&system_, bound, system_.give_ope(dd, p1, bound));
        return p_ope;
    }

    indic = is_ope_uni(name, p1, "integvolume");
    if (indic) {
        p_ope = new Ope_int_volume(&system_, system_.give_ope(dd, p1, bound));
        return p_ope;
    }

    indic = is_ope_uni(name, p1, "grad");
    if (indic) {
        p_ope = new Ope_grad(&system_, system_.give_ope(dd, p1, bound));
        return p_ope;
    }

    indic = is_ope_uni(name, p1, "sqrt");
    if (indic) {
        p_ope = new Ope_sqrt(&system_, system_.give_ope(dd, p1, bound));
        return p_ope;
    }

    indic = is_ope_uni(name, p1, "sqrtrho");
    if (indic) {
        p_ope = new Ope_sqrt_nonstd(&system_, system_.give_ope(dd, p1, bound));
        return p_ope;
    }

    indic = is_ope_uni(name, p1, "sqrtanti");
    if (indic) {
        p_ope = new Ope_sqrt_anti(&system_, system_.give_ope(dd, p1, bound));
        return p_ope;
    }

    indic = is_ope_uni(name, p1, "exp");
    if (indic) {
        p_ope = new Ope_exp(&system_, system_.give_ope(dd, p1, bound));
        return p_ope;
    }

    indic = is_ope_uni(name, p1, "log");
    if (indic) {
        p_ope = new Ope_log(&system_, system_.give_ope(dd, p1, bound));
        return p_ope;
    }

    indic = is_ope_uni(name, p1, "atanh");
    if (indic) {
        p_ope = new Ope_atanh(&system_, system_.give_ope(dd, p1, bound));
        return p_ope;
    }

    indic = is_ope_uni(name, p1, "atan");
    if (indic) {
        p_ope = new Ope_atan(&system_, system_.give_ope(dd, p1, bound));
        return p_ope;
    }

    indic = is_ope_uni(name, p1, "cos");
    if (indic) {
        p_ope = new Ope_cos(&system_, system_.give_ope(dd, p1, bound));
        return p_ope;
    }

    indic = is_ope_uni(name, p1, "sin");
    if (indic) {
        p_ope = new Ope_sin(&system_, system_.give_ope(dd, p1, bound));
        return p_ope;
    }

    indic = is_ope_uni(name, p1, "cosh");
    if (indic) {
        p_ope = new Ope_cosh(&system_, system_.give_ope(dd, p1, bound));
        return p_ope;
    }

    indic = is_ope_uni(name, p1, "sinh");
    if (indic) {
        p_ope = new Ope_sinh(&system_, system_.give_ope(dd, p1, bound));
        return p_ope;
    }

    indic = is_ope_uni(name, p1, p2, "scal");
    if (indic) {
        p_ope = new Ope_scal(&system_, system_.give_ope(dd, p1, bound), system_.give_ope(dd, p2, bound));
        return p_ope;
    }

    indic = is_ope_uni(name, p1, "determinant");
    if (indic) {
        p_ope = new Ope_determinant(&system_, system_.give_ope(dd, p1, bound));
        return p_ope;
    }

    indic = is_ope_uni(name, p1, "inverse");
    if (indic) {
        p_ope = new Ope_inverse(&system_, system_.give_ope(dd, p1, bound));
        return p_ope;
    }

    indic = is_ope_uni(name, p1, "inversenodet");
    if (indic) {
        p_ope = new Ope_inverse_nodet(&system_, system_.give_ope(dd, p1, bound));
        return p_ope;
    }

    indic = is_ope_uni(name, p1, p2, "fitwaves");
    if (indic) {
        p_ope = new Ope_fit_waves(&system_, system_.give_ope(dd, p1, bound), system_.give_ope(dd, p2, bound));
        return p_ope;
    }

    indic = is_ope_uni(name, p1, "import");
    if (indic) {
        p_ope = new Ope_import(&system_, dd, bound, p1);
        return p_ope;
    }

    indic = is_ope_uni(name, p1, "tospherical");
    if (indic) {
        p_ope = new Ope_change_basis(&system_, SPHERICAL_BASIS, system_.give_ope(dd, p1, bound));
        return p_ope;
    }

    indic = is_ope_uni(name, p1, "tocartesian");
    if (indic) {
        p_ope = new Ope_change_basis(&system_, CARTESIAN_BASIS, system_.give_ope(dd, p1, bound));
        return p_ope;
    }

    int type_der;
    char ind_der;
    indic = is_ope_deriv(name, p1, type_der, ind_der);
    if (indic) {
        p_ope = new Ope_der(&system_, type_der, ind_der, system_.give_ope(dd, p1, bound));
        return p_ope;
    }

    indic = is_ope_deriv_flat(name, p1, type_der, ind_der);
    if (indic) {
        p_ope = new Ope_der_flat(&system_, type_der, ind_der, system_.give_ope(dd, p1, bound));
        return p_ope;
    }

    indic = is_ope_deriv_background(name, p1, type_der, ind_der);
    if (indic) {
        p_ope = new Ope_der_background(&system_, type_der, ind_der, system_.give_ope(dd, p1, bound));
        return p_ope;
    }

    indic = is_ope_partial(name, p1, ind_der);
    if (indic) {
        p_ope = new Ope_partial(&system_, ind_der, system_.give_ope(dd, p1, bound));
        return p_ope;
    }

    int ind_var;
    indic = is_ope_der_var(dd, name, p1, ind_var);
    if (indic) {
        p_ope = new Ope_partial_var(&system_, ind_var, system_.give_ope(dd, p1, bound));
        return p_ope;
    }

    int val_exp;
    indic = is_ope_pow(name, p1, val_exp);
    if (indic) {
        p_ope = new Ope_pow(&system_, val_exp, system_.give_ope(dd, p1, bound));
        return p_ope;
    }

    indic = system_.isvar_double(name, which);
    if (indic) {
        p_ope = new Ope_id(&system_, system_.give_term_double(which, dd));
        return p_ope;
    }

    int valence;
    char* name_ind = nullptr;
    Array<int>* type_ind = nullptr;
    indic = system_.isvar(name, which, valence, name_ind, type_ind);
    if (indic) {
        p_ope = new Ope_id(&system_, system_.give_term(which, dd), valence, name_ind, type_ind);
        return p_ope;
    }

    name_ind = nullptr;
    int type_indice;
    indic = system_.ismet(name, name_ind, type_indice);
    if (indic) {
        type_ind = new Array<int>(2);
        type_ind->set(0) = type_indice;
        type_ind->set(1) = type_indice;
        p_ope = new Ope_id(&system_, system_.met->give_term(dd, type_indice), 2, name_ind, type_ind);
        return p_ope;
    }

    name_ind = nullptr;
    type_ind = nullptr;
    indic = system_.ischristo(name, name_ind, type_ind);
    if (indic) {
        p_ope = new Ope_id(&system_, system_.met->give_christo(dd), 3, name_ind, type_ind);
        return p_ope;
    }

    name_ind = nullptr;
    type_ind = nullptr;
    indic = system_.isriemann(name, name_ind, type_ind);
    if (indic) {
        p_ope = new Ope_id(&system_, system_.met->give_riemann(dd), 4, name_ind, type_ind);
        return p_ope;
    }

    name_ind = nullptr;
    type_ind = nullptr;
    indic = system_.isricci_tensor(name, name_ind, type_ind);
    if (indic) {
        p_ope = new Ope_id(&system_, system_.met->give_ricci_tensor(dd), 2, name_ind, type_ind);
        return p_ope;
    }

    name_ind = nullptr;
    type_ind = nullptr;
    indic = system_.isricci_scalar(name, name_ind, type_ind);
    if (indic) {
        p_ope = new Ope_id(&system_, system_.met->give_ricci_scalar(dd));
        return p_ope;
    }

    name_ind = nullptr;
    type_ind = nullptr;
    indic = is_tensor(name, "dirac ", valence, name_ind, type_ind);
    if (indic) {
        if (valence != 1) {
            KADATH_THROW("Dirac operator must be of valence one");
        }
        p_ope = new Ope_id(&system_, system_.met->give_dirac(dd), valence, name_ind, type_ind);
        return p_ope;
    }

    name_ind = nullptr;
    type_ind = nullptr;
    indic = is_tensor(name, "normal ", valence, name_ind, type_ind);
    if (indic) {
        if (valence != 1) {
            KADATH_THROW("Normal vector operator must be of valence one");
        }
        if (system_.met == nullptr) {
            KADATH_THROW("Metric must be passed to call normal");
        }
        int typebase = system_.met->give_type(dd);
        p_ope = new Ope_id(&system_, system_.espace.get_domain(dd)->give_normal(bound, typebase), valence, name_ind, type_ind);
        return p_ope;
    }

    name_ind = nullptr;
    type_ind = nullptr;
    indic = system_.iscst(name, which, valence, name_ind, type_ind);
    if (indic) {
        if (name_ind == nullptr)
            p_ope = new Ope_id(&system_, system_.give_cst(which, dd));
        else
            p_ope = new Ope_id(&system_, system_.give_cst(which, dd), valence, name_ind, type_ind);
        return p_ope;
    }

    double val = 0.;
    indic = system_.isdouble(name, val);
    if (indic) {
        valence = 0;
        p_ope = new Ope_id(&system_, system_.give_cst_hard(val, dd));
        return p_ope;
    }

    name_ind = nullptr;
    type_ind = nullptr;
    indic = system_.isdef(dd, name, which, valence, name_ind, type_ind);
    if (indic) {
        p_ope = new Ope_id(&system_, system_.give_def(which)->get_res(), valence, name_ind, type_ind);
        return p_ope;
    }

    indic = system_.isdef_glob(dd, name, which);
    if (indic) {
        p_ope = new Ope_id(&system_, system_.give_def_glob(which)->get_res());
        return p_ope;
    }

    for (int i = 0; i < system_.nopeuser; i++) {
        indic = is_ope_uni(name, p1, system_.names_opeuser[i].c_str());
        if (indic) {
            p_ope = new Ope_user(&system_, system_.opeuser[i], system_.paruser[i], system_.give_ope(dd, p1, bound));
            return p_ope;
        }
    }

    for (int i = 0; i < system_.nopeuser_bin; i++) {
        indic = is_ope_uni(name, p1, p2, system_.names_opeuser_bin[i].c_str());
        if (indic) {
            p_ope = new Ope_user_bin(&system_, system_.opeuser_bin[i], system_.paruser_bin[i],
                                     system_.give_ope(dd, p1, bound), system_.give_ope(dd, p2, bound));
            return p_ope;
        }
    }
    return p_ope;
}

} // namespace Kadath
