// ns_3d_xcts solver stage aggregation. #includes the individual stage bodies.
// Body fragment included by solver.hpp; not a standalone translation unit.

// Individual stage bodies (each a body fragment, not a standalone TU).
#include "norot_stage.ipp"        // norot_stage (static TOV)
#include "uniform_rot_stage.ipp"  // uniform_rot_stage (UNIROT)
#include "binary_boost_stage.ipp" // binary_boost_stage (BIN_BOOST)

namespace Kadath {


} // namespace Kadath