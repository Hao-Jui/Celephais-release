#include <catch2/catch_test_macros.hpp>
#include "For_Kadath/Utilities/name_tools.hpp"
#include <cstring>

using namespace Kadath;

TEST_CASE("trim_spaces removes whitespace", "[name_tools]") {
    char dest[64];
    trim_spaces(dest, "  hello  ");
    // trim_spaces always appends a trailing space (used for internal parsing)
    REQUIRE(std::strcmp(dest, "hello ") == 0);
}

TEST_CASE("trim_spaces handles no spaces", "[name_tools]") {
    char dest[64];
    trim_spaces(dest, "world");
    REQUIRE(std::strcmp(dest, "world ") == 0);
}

TEST_CASE("trim_spaces collapses consecutive spaces", "[name_tools]") {
    char dest[64];
    trim_spaces(dest, "a  b   c");
    // trim_spaces collapses runs of spaces to single space, appends trailing space
    REQUIRE(std::strcmp(dest, "a b c ") == 0);
}

TEST_CASE("nbr_char counts occurrences", "[name_tools]") {
    int count = nbr_char("hello world", 'l');
    REQUIRE(count == 3);
}

// ---- Modern parser helpers (parity with legacy is_ope_bin / get_term /
// get_parts / is_tensor). Each test pins behavior; full call site migration
// is a separate commit.

TEST_CASE("split_binary_operator finds top-level operator", "[name_tools][modern]") {
    auto split = split_binary_operator("a+b ", '+');
    REQUIRE(split.has_value());
    REQUIRE(split->lhs == "a ");
    REQUIRE(split->rhs == "b ");
}

TEST_CASE("split_binary_operator skips parens-imbalanced splits", "[name_tools][modern]") {
    // For `(a+b)*c+d` with op='+', the rightmost '+' (between c and d) splits
    // into `(a+b)*c ` and `d ` — both balanced. That is the result.
    auto split = split_binary_operator("(a+b)*c+d ", '+');
    REQUIRE(split.has_value());
    REQUIRE(split->lhs == "(a+b)*c ");
    REQUIRE(split->rhs == "d ");
}

TEST_CASE("split_binary_operator returns nullopt when op only inside parens", "[name_tools][modern]") {
    auto split = split_binary_operator("(a+b) ", '+');
    REQUIRE_FALSE(split.has_value());
}

TEST_CASE("split_binary_operator returns nullopt when op missing", "[name_tools][modern]") {
    auto split = split_binary_operator("abc ", '+');
    REQUIRE_FALSE(split.has_value());
}

TEST_CASE("split_first leftmost with trim", "[name_tools][modern]") {
    auto [lhs, rhs] = split_first("i,j,k ", ',');
    REQUIRE(lhs == "i ");
    REQUIRE(rhs == "j,k ");
}

TEST_CASE("split_first missing separator returns input + empty", "[name_tools][modern]") {
    auto [lhs, rhs] = split_first("abc ", ',');
    REQUIRE(lhs == "abc ");
    REQUIRE(rhs.empty());
}

TEST_CASE("split_nth occurrence=0 is last separator", "[name_tools][modern]") {
    // input "a,b,c " has separators at positions 1 and 3.
    // occurrence=0 (right-to-left first hit) splits at the rightmost ','.
    auto split = split_nth("a,b,c ", ',', 0);
    REQUIRE(split.has_value());
    REQUIRE(split->lhs == "a,b ");
    REQUIRE(split->rhs == "c ");
}

TEST_CASE("split_nth occurrence=1 is second-to-last separator", "[name_tools][modern]") {
    auto split = split_nth("a,b,c ", ',', 1);
    REQUIRE(split.has_value());
    REQUIRE(split->lhs == "a ");
    REQUIRE(split->rhs == "b,c ");
}

TEST_CASE("split_nth occurrence too large returns nullopt", "[name_tools][modern]") {
    auto split = split_nth("a,b ", ',', 5);
    REQUIRE_FALSE(split.has_value());
}

TEST_CASE("parse_tensor scalar with no indices", "[name_tools][modern]") {
    auto p = parse_tensor("g ");
    REQUIRE(p.has_value());
    REQUIRE(p->base_name == "g ");
    REQUIRE(p->valence == 0);
    REQUIRE(p->indices.empty());
    REQUIRE(p->types.empty());
}

TEST_CASE("parse_tensor covariant indices", "[name_tools][modern]") {
    auto p = parse_tensor("g_ij ");
    REQUIRE(p.has_value());
    REQUIRE(p->base_name == "g ");
    REQUIRE(p->valence == 2);
    REQUIRE(p->indices == "ij");
    REQUIRE(p->types == std::vector<int>{0, 0});  // COV, COV
}

TEST_CASE("parse_tensor contravariant index", "[name_tools][modern]") {
    auto p = parse_tensor("v^a ");
    REQUIRE(p.has_value());
    REQUIRE(p->base_name == "v ");
    REQUIRE(p->valence == 1);
    REQUIRE(p->indices == "a");
    REQUIRE(p->types == std::vector<int>{1});  // CON
}

TEST_CASE("parse_tensor mixed COV/CON indices", "[name_tools][modern]") {
    auto p = parse_tensor("g_i^j ");
    REQUIRE(p.has_value());
    REQUIRE(p->base_name == "g ");
    REQUIRE(p->valence == 2);
    REQUIRE(p->indices == "ij");
    REQUIRE(p->types == std::vector<int>{0, 1});  // COV, CON
}

TEST_CASE("parse_tensor missing trailing space rejects", "[name_tools][modern]") {
    auto p = parse_tensor("badnospace");
    REQUIRE_FALSE(p.has_value());
}

TEST_CASE("split_binary_operator handles inputs longer than LMAX", "[name_tools][modern][cap]") {
    // Demonstrates the cap-removal payoff: legacy is_ope_bin would be UB on
    // inputs > LMAX (caller buffers were sized LMAX). Modern API allocates as
    // needed.
    std::string lhs(LMAX + 100, 'x');
    std::string rhs(LMAX + 50, 'y');
    std::string input = lhs + "+" + rhs + " ";
    auto split = split_binary_operator(input, '+');
    REQUIRE(split.has_value());
    REQUIRE(split->lhs == lhs + " ");
    REQUIRE(split->rhs == rhs + " ");
}
