# Equations of State (EOS)

The FUKA solvers support both piecewise polytropic and tabulated EOS in generating ID for Neutron Stars.  For both cases, a file is required
which includes either the tabulated EOS or describes the piecewise polytrope, as described below.

## Piecewise Polytropic Equations of State

The file format for storing a piecewise polytrope is quite basic.  All lines starting with a `#` or are blank are ignored.  
Most importantly, the order of parameters is strictly defined

```
#h4 pwpolytrope
num_pieces: 4
rhomin: 1.e-19
rhomax: 1.0
ktab0: 3.99873692e-8
Ptab0: 0

gamma_tab
1.35692395 2.909 2.246 2.144

rho_tab
0.0 8.87824e+14 5.01187234e+14 1.0e+15

#units can be:
#geometrised
#cgs
#cgs_cgs_over_c2
units: cgs_cgs_over_c2
```

1. in the first block, the number of pieces, the value range of the density, as well as `K_0` (constant factor at the initial boundary and `P0` (pressure at the initial boundary) must be defined in this order
2. secondly the values of the polytropic exponent for each piece is defined in a single row
3. thirdly the values of the density are defined for each piece within a single row
4. finally the units must be specified for the input values

## Tabulated Equations of State

The format for tabulated EOS' that are currently supported is the well known format used by the [LORENE](https://lorene.obspm.fr/) spectral solver.  
Details on the format of the table can be found [here](https://lorene.obspm.fr/Refguide/classLorene_1_1Eos__tabul.html)

## EOS Framework

The backend that handles the EOS is constructed in two parts

### 1. EOS Management

The construction and evaluation of the EOS is handled by a standalone version of 
the Margherita EOS framework developed by [Elias R. Most](emost@th.physik.uni-frankfurt.de)
For code details see `$HOME_KADATH/include/Hydro/Margherita/`

### 2. EOS Interface

To interface between KADATH and Margherita, a user-defined module was written by Samuel Tootle and L. Jens Papenfort allowing the use of Margherita within KADATH's System_of_equations framework. For details see `$HOME_KADATH/include/hydro/EOS.hh`.

To initialize Margherita, the following parameters need to be set within the [config](https://bitbucket.org/fukaws/fuka/src/fukav2//include/Configurator/) file

```
  {"eostype",EOSTYPE}, // Options are currently: Cold_PWPoly, Cold_Table
  {"eosfile",EOSFILE}, // Polytrope or Tabulated EOS file
  {"h_cut",HCUT}, // value to cut the specific enthalpy (0 is default)
  {"interpolation_pts", INTERP_PTS} // number of points to use for interpolating the table
```

Regarding the `eos_file`, if no path is specified, e.g. `eos_file togashi.lorene`, the EOS framework will only search the `$HOME_KADATH/eos/` directory for the EOS file specified (in this repo: `data/eos/`, with a symlink `eos → data/eos` for backward compatibility).  In the event you want to specify an EOS not located in the default directory, a path must be included.  For example, to read an EOS from the current directory, one would need to write `eos_file ./togashi.lorene`.
