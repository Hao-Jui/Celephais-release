#pragma once

namespace sacra_hydro {

struct ColdHydroState {
    double pressure;
    double density;
};

inline bool has_matter(double log_enthalpy)
{
    return log_enthalpy > 0.0;
}

inline double zero_outside_matter(double log_enthalpy, double value)
{
    return has_matter(log_enthalpy) ? value : 0.0;
}

inline ColdHydroState zero_outside_matter(double log_enthalpy,
                                          ColdHydroState state)
{
    if (has_matter(log_enthalpy))
        return state;
    return {0.0, 0.0};
}

} // namespace sacra_hydro
