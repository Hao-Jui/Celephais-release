#pragma once

// Private implementation seam for the equation DSL parser. Holds the
// is_ope_* helpers and the find_ope decision chain; reaches back into
// System_of_eqs (var/cst/def lookups, met, user-operator registry,
// recursive give_ope) through a const reference and the friend grant.

namespace Kadath {

class System_of_eqs;
class Ope_eq;

class EquationParser {
 public:
    explicit EquationParser(const System_of_eqs& system) : system_(system) {}

    EquationParser(const EquationParser&) = delete;
    EquationParser& operator=(const EquationParser&) = delete;

    bool is_ope_bin(const char* name, char* part1, char* part2, char cible) const;
    bool is_ope_minus(const char* name, char* part) const;
    bool is_ope_uni(const char* name, char* part, const char* zeope) const;
    bool is_ope_uni(const char* name, char* p1, char* p2, const char* zeope) const;
    bool is_ope_deriv(const char* nn, char* part, int& type_der, char& name_ind) const;
    bool is_ope_deriv_flat(const char* nn, char* part, int& type_der, char& name_ind) const;
    bool is_ope_deriv_background(const char* nn, char* part, int& type_der, char& name_ind) const;
    bool is_ope_partial(const char* nn, char* part, char& name_ind) const;
    bool is_ope_pow(const char* name, char* part, int& expo) const;
    bool is_ope_der_var(int dd, const char* name, char* part, int& numvar) const;

    // Returns nullptr on unknown operator. Direct callers must handle that
    // case; System_of_eqs::give_ope wraps with throw-on-unknown so the throw
    // lives in one place.
    Ope_eq* find_ope(int dd, const char* name, int bound) const;

 private:
    const System_of_eqs& system_;
};

} // namespace Kadath
