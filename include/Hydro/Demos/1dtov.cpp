/*
 * Modifications (Celephais):
 *   2026-06-16  Modified for the Celephais tree; see
 *               PATCHES-KADATH-UPSTREAM.md and LICENSE_SOURCE_AUDIT.tsv.
 */
/*
 * =====================================================================================
 *
 *       Filename:  1dtov.cpp
 *
 *    Description:  Simple TOV test code to test tov.hh
 *
 *        Version:  1.0
 *        Created:  1/07/2021
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Samuel David Tootle, tootle@itp.uni-frankfurt.de
 *   Organization:  Goethe University Frankfurt
 *   Notes: this is a minimal test for the tov.hh code
 *
 * =====================================================================================
 */

/*
 * Modifications (Celephais):
 *   2026-06-16  Modified for the Celephais tree; see
 *               PATCHES-KADATH-UPSTREAM.md and LICENSE_SOURCE_AUDIT.tsv.
 */

#define DEBUG
#include "../margherita/table/cold_table.hh"
#include "../margherita/table/cold_table_implementation.hh"
#include "../margherita/table/setup_cold_table.hh"
#include "../margherita/pwp/cold_pwpoly.hh"
#include "../margherita/pwp/cold_pwpoly_implementation.hh"
#include "../margherita/pwp/setup_polytrope.hh"
#include "../margherita/tov_mass_search.hh"
#include <memory>

int main()
{
    using namespace Kadath::Margherita;

    setup_Cold_Table("togashi.lorene", 2000);
    auto tov = std::make_unique<MargheritaTOV<Cold_Table>>();
    // Margherita_setup_polytrope("mpa1.polytrope");
    // auto tov = std::make_unique<MargheritaTOV<Cold_PWPoly>>();

    // tov->solve(1.37e-3);
    solve_tov_for_adm_mass(*tov, 1.35);

    // creating a reference state to the TOV solution’s internal state.
    auto& state = tov->state;
    std::cout << *tov << std::endl;

    // since tov is a pointer to a TOV object, so cout << tov will
    // reture the address of the pointer, e.g., 0x7ffee42982b0
    return 0;
}
