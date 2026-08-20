#include <catch2/catch_test_macros.hpp>

#include "For_Kadath/Array/array.hpp"
#include "For_Kadath/Array/exceptions.hpp"
#include "For_Kadath/Base_spectral/base_r2hc.hpp"
#include "For_Kadath/Base_spectral/base_spectral.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <limits>
#include <numbers>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace Kadath {
void coef_1d(int base, Array<double>& tab);
bool coef_1d(int base, const double* src, double* dst,
             int nbr_in, int nbr_out, int stride);
void coef_i_1d(int base, Array<double>& tab);
bool coef_i_1d(int base, const double* src, double* dst,
               int nbr_in, int nbr_out, int stride);
r2hc_precomp_t& coef_1d_r2hc(int n);
r2hc_precomp_t& coef_i_1d_hc2r(int n);
}

namespace {

static_assert(!std::is_copy_constructible_v<Kadath::r2hc_precomp_t>);
static_assert(!std::is_copy_assignable_v<Kadath::r2hc_precomp_t>);

void hash_value(std::uint64_t& hash, double value)
{
    const auto bits = std::bit_cast<std::uint64_t>(value);
    for (unsigned int shift = 0; shift < 64; shift += 8) {
        hash ^= (bits >> shift) & 0xffu;
        hash *= 1099511628211ULL;
    }
}

std::uint64_t coefficient_transform_hash()
{
    constexpr std::array bases{
        CHEB,
        CHEB_EVEN,
        CHEB_ODD,
        COS,
        SIN,
        COS_EVEN,
        COS_ODD,
        SIN_EVEN,
        SIN_ODD,
    };
    constexpr int size = 7;
    std::uint64_t hash = 1469598103934665603ULL;

    for (int base : bases) {
        Kadath::Array<double> values(size);
        for (int i = 0; i < size; ++i)
            values.set(i) = (i + 1) * 0.125 - (i % 3) * 0.03125;
        Kadath::coef_1d(base, values);
        for (int i = 0; i < size; ++i)
            hash_value(hash, values(i));

        Kadath::Array<double> coefficients(size);
        for (int i = 0; i < size; ++i)
            coefficients.set(i) = (i + 2) * -0.0625 + (i % 2) * 0.015625;
        Kadath::coef_i_1d(base, coefficients);
        for (int i = 0; i < size; ++i)
            hash_value(hash, coefficients(i));
    }
    return hash;
}

struct native_cos_sin_family_case
{
    Kadath::native_spectral_family family;
    int base;
    std::size_t hash_index;
};

constexpr std::array native_cos_sin_transform_sizes{6, 8, 10, 12, 14, 16, 18, 20};
constexpr std::array native_cos_sin_families{
    native_cos_sin_family_case{Kadath::native_spectral_family::cos, COS, 0},
    native_cos_sin_family_case{Kadath::native_spectral_family::sin, SIN, 1},
    native_cos_sin_family_case{Kadath::native_spectral_family::cos_even, COS_EVEN, 2},
    native_cos_sin_family_case{Kadath::native_spectral_family::cos_odd, COS_ODD, 3},
    native_cos_sin_family_case{Kadath::native_spectral_family::sin_even, SIN_EVEN, 4},
    native_cos_sin_family_case{Kadath::native_spectral_family::sin_odd, SIN_ODD, 5},
};
constexpr std::array native_cos_sin_strides{1, 3};

bool native_backend_selected()
{
    const char* const backend = std::getenv("CELEPHAIS_FFT_BACKEND");
    return backend == nullptr || backend[0] == '\0'
           || std::strcmp(backend, "native") == 0;
}

bool native_cos_sin_fused_expected(const int transform_size,
                                   const Kadath::native_spectral_family family,
                                   const Kadath::r2hc_direction direction)
{
    if (direction == Kadath::r2hc_direction::inverse)
        return transform_size <= 16
               || family == Kadath::native_spectral_family::cos_odd
               || family == Kadath::native_spectral_family::sin_odd;
    return transform_size <= 18
           || family == Kadath::native_spectral_family::sin
           || family == Kadath::native_spectral_family::sin_even
           || family == Kadath::native_spectral_family::cos_odd
           || family == Kadath::native_spectral_family::sin_odd;
}

#if defined(__APPLE__) && defined(__aarch64__) && defined(__clang__)
constexpr std::uint64_t native_coefficient_transform_hash = 0x176f55c6eac05c0eULL;

// FNV-1a checksums frozen from the 64150eb5 native buffer route on
// AppleClang/macOS/arm64. Entries are ordered by N, family, direction
// (forward then inverse), and stride (1 then 3).
constexpr std::array<std::uint64_t, 192> native_cos_sin_buffer_hashes{
    0xf1c949924942cfa5ULL, 0x3a96900532f93df5ULL, 0xe9b0710bc7e125aeULL, 0xa79cc1b6e2588197ULL,
    0x3f2f59a08667c6d5ULL, 0xec35cd037a1bf82aULL, 0xa9125208ae2e6ca1ULL, 0x18d5591f604ff119ULL,
    0xb3dde03fbe4bf715ULL, 0x4f3c4ffc31a087b6ULL, 0x531fdbecfe9cc373ULL, 0x3cefdcf42f35d22fULL,
    0xb5ebca3829ec30c8ULL, 0xfa8ed68ab0939f9eULL, 0x6592da1d200d7b7cULL, 0x7b2f78271acc8845ULL,
    0x50b78c09c66a8500ULL, 0xcb2c6c8ee06fbc9eULL, 0xeef162e4c7ec85d9ULL, 0xfa968e6c8c100a31ULL,
    0xf2773ef7a848bc0eULL, 0x4e3d0178974db442ULL, 0x985d5907bca1ad5cULL, 0x552a228f01668299ULL,
    0xc381de3efd524c8eULL, 0x8d638ce7f31b50ecULL, 0x1fb9bb175ce4c3d3ULL, 0xe603ce32c96fd009ULL,
    0x3bca83368900ed30ULL, 0xea25ed8ee5992c54ULL, 0x316c911d745399ebULL, 0x94e506db49499d12ULL,
    0x964614b22705e6e9ULL, 0x2f5e06c4ae3ceb03ULL, 0x75f298ce19a54056ULL, 0x7c5b41da232bfa99ULL,
    0xa0818ad56ac5e95bULL, 0x209274fce5141caaULL, 0xaff47508f487ad56ULL, 0x47309cd1bc8063dbULL,
    0xc0d60c719fb520dbULL, 0x360e3f547132b09cULL, 0x1b257b100489f7a7ULL, 0xba33a20dde971700ULL,
    0xae003721407389f9ULL, 0x3a076c171746ab90ULL, 0x9a3ba84aa7e0db10ULL, 0x275fb8adab0284cdULL,
    0xa8c10693f8de1c12ULL, 0x4d3859e95452b076ULL, 0xf65d3d542a51b2dcULL, 0x7066791c3afd55bfULL,
    0x65918bd8bae2c9a3ULL, 0x8c6ef3d87b58ed69ULL, 0xc8c2c0966e31bd3aULL, 0xf758b537edad4531ULL,
    0x35694bee2460d85aULL, 0x608381918abba126ULL, 0xa7d1572a1c540fe6ULL, 0x44cbf7cf79ccde7fULL,
    0xef9d3fa87eedaf68ULL, 0xe751143a41668c42ULL, 0x0bef3e4b6512549eULL, 0xb1a32b9147f29801ULL,
    0xc03a0c90cd5e539fULL, 0x096ef027c8a19262ULL, 0xd98888fbc55dab48ULL, 0xa6a182ffb9714153ULL,
    0x6d1c3f763a595012ULL, 0xca68b467d58c62c0ULL, 0xd5a5bc45307195eaULL, 0xf2808e33808334c6ULL,
    0x6d063644aa12044eULL, 0x8ca42b9d79cca253ULL, 0xd033d56b47c20f44ULL, 0xe04ee8aead199827ULL,
    0xbf76c963c52eac28ULL, 0x9c3e22fcca4a2b15ULL, 0x8da814ee15152af2ULL, 0xd38a6417787d0fbaULL,
    0x1323bb0fd6ff29a2ULL, 0xdf0cfcda3a709f13ULL, 0xd16a99d9bfda6a8fULL, 0x5e4c883960d8e396ULL,
    0xd47358330a805310ULL, 0x64b8eb41f50766b8ULL, 0xd0596b66290d4f76ULL, 0x38748234b6efa6d3ULL,
    0x6d55dcf3f6c3e030ULL, 0xb7d3bab5a0140841ULL, 0xc5dcb1e96a7701e6ULL, 0x413402cdaf14d07aULL,
    0x3e5dbc9cdcc34079ULL, 0x4d26caf588d89410ULL, 0x2d0d260970a9a2e4ULL, 0x87193e81a924672dULL,
    0x0cb9bc91b9cf215dULL, 0x96ff9eed1b02cbcaULL, 0x311dfc501eb700baULL, 0x2ca4daa4f210a06fULL,
    0x6562ee74df7d2293ULL, 0xf53f4f379eaf87afULL, 0x88dfb7d60499cdfcULL, 0xbef39cbc8475a973ULL,
    0xb90c68c498d92ae9ULL, 0x569f3ffba3a7ae6eULL, 0x0cf3f11a70f38b28ULL, 0xc4a89008a1da1befULL,
    0x7e8f5750c9c70825ULL, 0x01db325d369f240fULL, 0x75cd90043ef69c5bULL, 0xed27285c59079ff0ULL,
    0x1461ec52c308fe4bULL, 0xd97b20f61272693dULL, 0xa85e2c4d17b711faULL, 0x6b198a534046e8c1ULL,
    0x7b30d6054ec2d43fULL, 0xcb527ad0bf301d94ULL, 0x2a10307b224beeddULL, 0xcb54399586800a09ULL,
    0x211d83082450a9b3ULL, 0x4a7712309db7169fULL, 0x86657dfb310dbe51ULL, 0xab3b67d0d4b77d85ULL,
    0x12dcf3ac0a347ab1ULL, 0xbffb0de9e3644482ULL, 0xb4a72a98e309c0ceULL, 0x2bbb67662cf70244ULL,
    0xe28ecde77c0cbd7eULL, 0x2490566dfe07d99dULL, 0xaa9a806030ce199cULL, 0x2df75251c2661302ULL,
    0x8880c0e3a589ae65ULL, 0xa228cc1dcde0bb6fULL, 0x4b679660427e5b64ULL, 0xab915e44900cf37dULL,
    0xd5dffa51e3371394ULL, 0x46261595882e16a2ULL, 0x3dea80db378e8e17ULL, 0xd5cabdad0bb39130ULL,
    0xbc7f26b06f006623ULL, 0xcd3385494c7d2a40ULL, 0x78f3dbfbbd6786e7ULL, 0x5b806877c7dd8646ULL,
    0xba085b5f93c90412ULL, 0x9775c736efac2fadULL, 0x23e441f62944e683ULL, 0x1e8f8559afe01237ULL,
    0x05da85b8db498321ULL, 0xc2e4fb4efb2b03ffULL, 0x1cee0ee2609fc28cULL, 0x558bc632366a9504ULL,
    0x6fa95a3e1407c974ULL, 0x746c7fcb18d84697ULL, 0x63f1b159ee108e8fULL, 0x87d8be0a4e791e71ULL,
    0x7cc56495d99c2045ULL, 0x2920dd29e96a64e1ULL, 0x5f600d4da106e902ULL, 0x2a6309a22088a62eULL,
    0x7f3945cf2b51a84fULL, 0x8adc036998b9f457ULL, 0x3281c1b2165bd92eULL, 0x885d9ec45a62b14aULL,
    0xb92e5035a4ab2a30ULL, 0x5083288e16df904cULL, 0x13905e6fecefe4cdULL, 0x53d2d87956e22ea9ULL,
    0x01b8dade3a3c6ae7ULL, 0x5505a7c95f84bd9dULL, 0x7e0a1791ac1b66a5ULL, 0xd29b4efc872461b5ULL,
    0x863a3a6b412e44a5ULL, 0x0b6cf44d098b2d27ULL, 0x9be70ef2065c358dULL, 0xb2663507f06a6fa3ULL,
    0x9f26f4e887c1beaaULL, 0xcb54e279d35fecbcULL, 0x3c3d60aac9d2cb86ULL, 0xa1c231411edbf518ULL,
    0x11ef62c64909d5cdULL, 0x94f77582b92e98e7ULL, 0x416813bf9d520c91ULL, 0xfe80ad58e25d7f89ULL,
    0x1e5b046f5ca8a2d7ULL, 0xf5631a84bf2aaf4dULL, 0x2fcecb337d2b27b9ULL, 0x517a8bd20b4537f0ULL,
    0x33f6926c8419b900ULL, 0x3bde7279fc98bffeULL, 0xac075fb35dc3ab18ULL, 0x9e88668c6c2b0770ULL,
};
#elif defined(__linux__) && defined(__x86_64__) && defined(__GNUC__) \
    && !defined(__clang__)
constexpr std::uint64_t native_coefficient_transform_hash = 0xcf5cacc9d6a8356eULL;

// GCC/Linux/x86-64 corpus generated by native_cossin_family_hash.cpp
// (SHA-256 ce1817f0d776b253f26c7feaa55f33b3e7f0407635abcf705338b845316d5e66).
// The 193-line CSV has SHA-256
// e92721f0b01f18b711e0d17458e268c62364213d68e971ecc87bc9a451133241.
// It was admitted only after the independent long-double DFT/roundtrip oracle
// passed 800 vectors with maximum scaled native error 0.003139 of tolerance.
constexpr std::array<std::uint64_t, 192> native_cos_sin_buffer_hashes{
    0x5b8b789d5ff18cdeULL, 0x0c1f372561d978c8ULL, 0x30820786e40c7d7fULL, 0x3999a91a510795aeULL,
    0xcaeccc30b670870cULL, 0xb46a3f8dc3b3913fULL, 0xd852756fc67f3c96ULL, 0xe7e014be2af50247ULL,
    0x9c320d74ce5d3f58ULL, 0xc734f34f18ef4b6fULL, 0x0831df55b373c8c2ULL, 0x2debd359d3053037ULL,
    0x1f027d0c6e455e1aULL, 0xaa758dea6ce75016ULL, 0x8eee8a8b0922c9b9ULL, 0xd0af22adf652f444ULL,
    0x8c12e0e43e6059b4ULL, 0x36c6c8b48c94afbaULL, 0x2c4da50bc7e9c557ULL, 0xad76d5715c18dd56ULL,
    0xd751b4e62628b34aULL, 0x8211440440bff49eULL, 0x30f13bf476b553b6ULL, 0xff0f3df70689d7efULL,
    0xc381de3efd524c8eULL, 0x8d638ce7f31b50ecULL, 0x1fb9bb175ce4c3d3ULL, 0xe603ce32c96fd009ULL,
    0xa2578c0192039f13ULL, 0x451ff6c143ae4a5fULL, 0x316c911d745399ebULL, 0x94e506db49499d12ULL,
    0x964614b22705e6e9ULL, 0x2f5e06c4ae3ceb03ULL, 0x75f298ce19a54056ULL, 0x7c5b41da232bfa99ULL,
    0xa0818ad56ac5e95bULL, 0x209274fce5141caaULL, 0xaff47508f487ad56ULL, 0x47309cd1bc8063dbULL,
    0x7d2c86a23f85411bULL, 0x42aa6c1d655d67e3ULL, 0x1b257b100489f7a7ULL, 0xba33a20dde971700ULL,
    0xae003721407389f9ULL, 0x3a076c171746ab90ULL, 0x9a3ba84aa7e0db10ULL, 0x275fb8adab0284cdULL,
    0xa8c10693f8de1c12ULL, 0x4d3859e95452b076ULL, 0xf65d3d542a51b2dcULL, 0x7066791c3afd55bfULL,
    0x475552f17e85f418ULL, 0x8a310dc4bd435196ULL, 0xc8c2c0966e31bd3aULL, 0xf758b537edad4531ULL,
    0x35694bee2460d85aULL, 0x608381918abba126ULL, 0xa7d1572a1c540fe6ULL, 0x44cbf7cf79ccde7fULL,
    0x550fb9b4aa9c37a3ULL, 0x640e9f0784005565ULL, 0x2951433778e7a3a2ULL, 0xd09863dfb6c4f7ddULL,
    0x59a6d3811e5dcd6fULL, 0xf7011b277e0292d0ULL, 0xd98888fbc55dab48ULL, 0xa6a182ffb9714153ULL,
    0xf508abd97a3df85eULL, 0x29d4392aea54f534ULL, 0x0dc6631f33611df0ULL, 0x5453d2497f7b29caULL,
    0x6fd16fdbb1145ed6ULL, 0x3c15441ed4efe69bULL, 0xe31f0e52dcfc55ffULL, 0xb647a47533c02900ULL,
    0xe8b28b93f71f2450ULL, 0xe2c6805492219092ULL, 0xf0b1b6a94983de9dULL, 0xc1c602ae6825f5a1ULL,
    0xa64242916c963472ULL, 0x659b6f6de2d90a00ULL, 0x03dbed188e24d7efULL, 0xc8539ea7e59aeca8ULL,
    0x60023d9c52e71213ULL, 0xd89cd7aedb3242b0ULL, 0x65a96cec1117b96dULL, 0xe2326edee88f3710ULL,
    0x858f8a74ca0934dfULL, 0x1ef571474fde2f4aULL, 0xa06dae8ca086b500ULL, 0xc5d9accece3cabbdULL,
    0xb947df4b077750a3ULL, 0xf306b76482a44adbULL, 0xda6991645473191bULL, 0xfddfddbce44cf72cULL,
    0x0cb9bc91b9cf215dULL, 0x96ff9eed1b02cbcaULL, 0x311dfc501eb700baULL, 0x2ca4daa4f210a06fULL,
    0xb61000bf9c6afb01ULL, 0x8d48226707901002ULL, 0x88dfb7d60499cdfcULL, 0xbef39cbc8475a973ULL,
    0xb90c68c498d92ae9ULL, 0x569f3ffba3a7ae6eULL, 0x0cf3f11a70f38b28ULL, 0xc4a89008a1da1befULL,
    0x7b1fb8efc46fbf90ULL, 0xe0021cc415b26bf7ULL, 0xa71bfaa378c71133ULL, 0x03b80cc1daf7cb54ULL,
    0xeb784846194541bcULL, 0xa7a531a30faf1402ULL, 0xa85e2c4d17b711faULL, 0x6b198a534046e8c1ULL,
    0x008f18e0b5d5edb6ULL, 0x95c494a22df1aa8dULL, 0x01a36389514e8bbfULL, 0xe94c306f6c071cabULL,
    0x211d83082450a9b3ULL, 0x4a7712309db7169fULL, 0x86657dfb310dbe51ULL, 0xab3b67d0d4b77d85ULL,
    0xd8817969a704bc49ULL, 0x363b6919079fbc01ULL, 0xb4a72a98e309c0ceULL, 0x2bbb67662cf70244ULL,
    0xe28ecde77c0cbd7eULL, 0x2490566dfe07d99dULL, 0xaa9a806030ce199cULL, 0x2df75251c2661302ULL,
    0x6fbd105b8494bfaeULL, 0x8d0a2e7086cb4e51ULL, 0x3a0ddce81cfeff5eULL, 0x65b24fa8129087b8ULL,
    0xf726df5c97b87882ULL, 0xae6d2cc155279d0dULL, 0x3dea80db378e8e17ULL, 0xd5cabdad0bb39130ULL,
    0x9caa3ad51b0f4182ULL, 0x726356d357b33f94ULL, 0x69fe258079a56c8fULL, 0xf3efda25dc529fb1ULL,
    0xba085b5f93c90412ULL, 0x9775c736efac2fadULL, 0x23e441f62944e683ULL, 0x1e8f8559afe01237ULL,
    0xb89bc3823b41dfdbULL, 0xb53814b9a8f7a841ULL, 0x1cee0ee2609fc28cULL, 0x558bc632366a9504ULL,
    0x6fa95a3e1407c974ULL, 0x746c7fcb18d84697ULL, 0x63f1b159ee108e8fULL, 0x87d8be0a4e791e71ULL,
    0x7cc56495d99c2045ULL, 0x2920dd29e96a64e1ULL, 0x5f600d4da106e902ULL, 0x2a6309a22088a62eULL,
    0xc353e903170616caULL, 0xf2a167dd2d09493dULL, 0x3281c1b2165bd92eULL, 0x885d9ec45a62b14aULL,
    0xb92e5035a4ab2a30ULL, 0x5083288e16df904cULL, 0x13905e6fecefe4cdULL, 0x53d2d87956e22ea9ULL,
    0x85aae9abf0e9c74fULL, 0xe970113a641a1db4ULL, 0xd62161d3731b8971ULL, 0x353ea0d2064ec976ULL,
    0x34463fd77e0b1051ULL, 0x2fff67ec1cbcf2b9ULL, 0x013930160f0e3978ULL, 0xc6987a9525c388dbULL,
    0xadec8cac801243d7ULL, 0x72c236e8ff21133bULL, 0xc7003f8508123768ULL, 0xf8b306431ca5ec2fULL,
    0x80187464909b7b9bULL, 0x5a49f4ab4e1dc765ULL, 0xc8c37b1e655b0135ULL, 0x43fefce506c2ad80ULL,
    0xf46397cd1c97b94cULL, 0x3a31458d48099854ULL, 0x8578c788511fcaf7ULL, 0x843b17da00fc7efcULL,
    0xbb983f760c760f92ULL, 0xad2ce15143ced841ULL, 0x47440f5589c87762ULL, 0x318a35f716b71db0ULL,
};
#endif

constexpr std::array direct_real_transform_sizes{
    2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32,
};

std::vector<long double> reference_r2hc(const std::vector<double>& input)
{
    const int n = static_cast<int>(input.size());
    std::vector<long double> expected(static_cast<std::size_t>(n));
    for (int k = 0; k <= n / 2; ++k) {
        long double real = 0.;
        long double imaginary = 0.;
        for (int j = 0; j < n; ++j) {
            const long double angle =
                2.L * std::numbers::pi_v<long double> * k * j / n;
            real += static_cast<long double>(input[static_cast<std::size_t>(j)])
                    * std::cos(angle);
            imaginary -= static_cast<long double>(input[static_cast<std::size_t>(j)])
                         * std::sin(angle);
        }
        expected[static_cast<std::size_t>(k)] = real;
        if (k != 0 && 2 * k != n)
            expected[static_cast<std::size_t>(n - k)] = imaginary;
    }
    return expected;
}

std::vector<long double> reference_hc2r(const std::vector<double>& halfcomplex)
{
    const int n = static_cast<int>(halfcomplex.size());
    std::vector<long double> expected(static_cast<std::size_t>(n));
    for (int j = 0; j < n; ++j) {
        long double value = halfcomplex[0];
        value += (j & 1) ? -halfcomplex[static_cast<std::size_t>(n / 2)]
                         : halfcomplex[static_cast<std::size_t>(n / 2)];
        for (int k = 1; k < n / 2; ++k) {
            const long double angle =
                2.L * std::numbers::pi_v<long double> * k * j / n;
            value += 2.L
                     * (static_cast<long double>(halfcomplex[static_cast<std::size_t>(k)])
                            * std::cos(angle)
                        - static_cast<long double>(
                              halfcomplex[static_cast<std::size_t>(n - k)])
                              * std::sin(angle));
        }
        expected[static_cast<std::size_t>(j)] = value;
    }
    return expected;
}

void require_matches_reference(const double* const actual,
                               const std::vector<long double>& expected)
{
    const auto n = static_cast<int>(expected.size());
    for (int i = 0; i < n; ++i) {
        CAPTURE(i);
        const long double scale =
            128.L * std::numeric_limits<double>::epsilon() * n
            * std::max(1.L, std::abs(expected[static_cast<std::size_t>(i)]));
        REQUIRE(std::abs(static_cast<long double>(actual[i])
                         - expected[static_cast<std::size_t>(i)])
                <= scale);
    }
}

double next_deterministic_sample(std::uint64_t& state)
{
    state = state * 6364136223846793005ULL + 1442695040888963407ULL;
    constexpr double inverse_53_bits = 1. / 9007199254740992.;
    return 2. * static_cast<double>(state >> 11) * inverse_53_bits - 1.;
}

template <typename Operation>
std::string invalid_argument_message(Operation&& operation)
{
    try {
        operation();
    }
    catch (const std::invalid_argument& error) {
        return error.what();
    }
    FAIL("expected std::invalid_argument");
    return {};
}

} // namespace

TEST_CASE("coefficient transforms are deterministic within a process", "[coef-precomp]")
{
    const std::uint64_t first = coefficient_transform_hash();
    REQUIRE(first == coefficient_transform_hash());
    const char* const configured_backend = std::getenv("CELEPHAIS_FFT_BACKEND");
#ifdef CELEPHAIS_ENABLE_FFTW_ORACLE
    if (configured_backend != nullptr && std::strcmp(configured_backend, "fftw") == 0) {
        REQUIRE(first == 0xcbf1902f7ab0c4feULL);
    }
    else
#else
    (void)configured_backend;
#endif
    {
#if (defined(__APPLE__) && defined(__aarch64__) && defined(__clang__)) \
    || (defined(__linux__) && defined(__x86_64__) && defined(__GNUC__) \
        && !defined(__clang__))
        REQUIRE(first == native_coefficient_transform_hash);
#else
        SKIP("no exact native checksum fixture is qualified for this platform/compiler");
#endif
    }
}

TEST_CASE("configured FFT backend obeys the build policy",
          "[coef-precomp][fft-backend-policy]")
{
    const char* const configured_backend = std::getenv("CELEPHAIS_FFT_BACKEND");
    if (configured_backend != nullptr && std::strcmp(configured_backend, "fftw") == 0) {
#ifdef CELEPHAIS_ENABLE_FFTW_ORACLE
        Kadath::r2hc_precomp_t transform(6, Kadath::r2hc_direction::forward);
        REQUIRE(transform.backend() == Kadath::r2hc_backend::fftw);
        REQUIRE(reinterpret_cast<std::uintptr_t>(transform.buffer) % 64 == 0);
#else
        REQUIRE(invalid_argument_message([] {
                    Kadath::r2hc_precomp_t(6, Kadath::r2hc_direction::forward);
                })
                == "CELEPHAIS_FFT_BACKEND=fftw requested a test-only oracle unavailable "
                   "in this production build; use 'native' or leave it unset");
#endif
    }
    else {
        Kadath::r2hc_precomp_t transform(6, Kadath::r2hc_direction::forward);
        REQUIRE(transform.backend() == Kadath::r2hc_backend::native);
        REQUIRE(reinterpret_cast<std::uintptr_t>(transform.buffer) % 64 == 0);
    }
}

TEST_CASE("native half-complex transforms match the discrete Fourier definition",
          "[coef-precomp][native-r2hc]")
{
    constexpr std::array sizes{
        2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32,
    };

    for (int n : sizes) {
        CAPTURE(n);
        Kadath::r2hc_precomp_t forward(n, Kadath::r2hc_direction::forward,
                                      Kadath::r2hc_backend::native);
        Kadath::r2hc_precomp_t inverse(n, Kadath::r2hc_direction::inverse,
                                      Kadath::r2hc_backend::native);
        std::vector<double> input(static_cast<std::size_t>(n));
        for (int j = 0; j < n; ++j) {
            input[static_cast<std::size_t>(j)] =
                0.125 * (j - 3) + std::sin(0.37 * j) - 0.25 * std::cos(0.19 * j);
            forward.buffer[j] = input[static_cast<std::size_t>(j)];
        }

        forward.execute_r2hc();
        for (int k = 0; k <= n / 2; ++k) {
            long double real = 0.;
            long double imaginary = 0.;
            for (int j = 0; j < n; ++j) {
                const long double angle =
                    2.L * std::numbers::pi_v<long double> * k * j / n;
                real += input[static_cast<std::size_t>(j)] * std::cos(angle);
                imaginary -= input[static_cast<std::size_t>(j)] * std::sin(angle);
            }
            const double scale = 2.e-12 * n * std::max(1.L, std::abs(real));
            REQUIRE(std::abs(forward.buffer[k] - static_cast<double>(real)) <= scale);
            if (k != 0 && 2 * k != n) {
                const double imaginary_scale =
                    2.e-12 * n * std::max(1.L, std::abs(imaginary));
                REQUIRE(std::abs(forward.buffer[n - k] - static_cast<double>(imaginary))
                        <= imaginary_scale);
            }
        }

        std::copy_n(forward.buffer, n, inverse.buffer);
        inverse.execute_hc2r();
        for (int j = 0; j < n; ++j) {
            const double expected = n * input[static_cast<std::size_t>(j)];
            const double scale = 4.e-12 * n * std::max(1., std::abs(expected));
            REQUIRE(std::abs(inverse.buffer[j] - expected) <= scale);
        }
    }
}

#ifdef CELEPHAIS_ENABLE_FFTW_ORACLE
TEST_CASE("native and test-only FFTW half-complex transforms match independent real-input definitions",
          "[coef-precomp][native-r2hc][fftw-differential][direct-real-oracle]")
{
    for (const int n : direct_real_transform_sizes) {
        CAPTURE(n);
        Kadath::r2hc_precomp_t native_forward(n, Kadath::r2hc_direction::forward,
                                             Kadath::r2hc_backend::native);
        Kadath::r2hc_precomp_t fftw_forward(n, Kadath::r2hc_direction::forward,
                                           Kadath::r2hc_backend::fftw);
        Kadath::r2hc_precomp_t native_inverse(n, Kadath::r2hc_direction::inverse,
                                             Kadath::r2hc_backend::native);
        Kadath::r2hc_precomp_t fftw_inverse(n, Kadath::r2hc_direction::inverse,
                                           Kadath::r2hc_backend::fftw);

        const double* const native_forward_storage = native_forward.buffer;
        const double* const fftw_forward_storage = fftw_forward.buffer;
        const double* const native_inverse_storage = native_inverse.buffer;
        const double* const fftw_inverse_storage = fftw_inverse.buffer;

        // Every real-input impulse exercises one complete column of the direct
        // R2HC transform, including the sign and n-k placement of imaginary modes.
        for (int impulse = 0; impulse < n; ++impulse) {
            CAPTURE(impulse);
            std::vector<double> input(static_cast<std::size_t>(n), 0.);
            input[static_cast<std::size_t>(impulse)] = 1.;
            const auto expected = reference_r2hc(input);
            std::copy(input.begin(), input.end(), native_forward.buffer);
            std::copy(input.begin(), input.end(), fftw_forward.buffer);
            native_forward.execute_r2hc();
            fftw_forward.execute_r2hc();
            require_matches_reference(native_forward.buffer, expected);
            require_matches_reference(fftw_forward.buffer, expected);
        }

        // Arbitrary halfcomplex impulses exercise the inverse convention without
        // constraining its input to values produced by either forward implementation.
        for (int impulse = 0; impulse < n; ++impulse) {
            CAPTURE(impulse);
            std::vector<double> halfcomplex(static_cast<std::size_t>(n), 0.);
            halfcomplex[static_cast<std::size_t>(impulse)] = 1.;
            const auto expected = reference_hc2r(halfcomplex);
            std::copy(halfcomplex.begin(), halfcomplex.end(), native_inverse.buffer);
            std::copy(halfcomplex.begin(), halfcomplex.end(), fftw_inverse.buffer);
            native_inverse.execute_hc2r();
            fftw_inverse.execute_hc2r();
            require_matches_reference(native_inverse.buffer, expected);
            require_matches_reference(fftw_inverse.buffer, expected);
        }

        std::uint64_t state = 0xd1b54a32d192ed03ULL ^ static_cast<std::uint64_t>(n);
        for (int fixture = 0; fixture < 8; ++fixture) {
            CAPTURE(fixture);
            std::vector<double> input(static_cast<std::size_t>(n));
            std::vector<double> halfcomplex(static_cast<std::size_t>(n));
            for (int j = 0; j < n; ++j) {
                input[static_cast<std::size_t>(j)] = next_deterministic_sample(state);
                halfcomplex[static_cast<std::size_t>(j)] = next_deterministic_sample(state);
            }

            const auto forward_expected = reference_r2hc(input);
            std::copy(input.begin(), input.end(), native_forward.buffer);
            std::copy(input.begin(), input.end(), fftw_forward.buffer);
            native_forward.execute_r2hc();
            fftw_forward.execute_r2hc();
            require_matches_reference(native_forward.buffer, forward_expected);
            require_matches_reference(fftw_forward.buffer, forward_expected);

            const auto inverse_expected = reference_hc2r(halfcomplex);
            std::copy(halfcomplex.begin(), halfcomplex.end(), native_inverse.buffer);
            std::copy(halfcomplex.begin(), halfcomplex.end(), fftw_inverse.buffer);
            native_inverse.execute_hc2r();
            fftw_inverse.execute_hc2r();
            require_matches_reference(native_inverse.buffer, inverse_expected);
            require_matches_reference(fftw_inverse.buffer, inverse_expected);
        }

        // Both public transform records are deliberately in-place; execution must
        // not swap the storage object as an implementation shortcut.
        REQUIRE(native_forward.buffer == native_forward_storage);
        REQUIRE(fftw_forward.buffer == fftw_forward_storage);
        REQUIRE(native_inverse.buffer == native_inverse_storage);
        REQUIRE(fftw_inverse.buffer == fftw_inverse_storage);
    }
}
#endif

TEST_CASE("transform records are indexed, exact, and reject unsupported sizes",
          "[coef-precomp][native-r2hc]")
{
    constexpr int transform_size = 6;
    auto& forward = Kadath::coef_1d_r2hc(transform_size);
    auto& inverse = Kadath::coef_i_1d_hc2r(transform_size);

    REQUIRE(&Kadath::coef_1d_r2hc(transform_size) == &forward);
    REQUIRE(&Kadath::coef_i_1d_hc2r(transform_size) == &inverse);
    REQUIRE(&forward != &inverse);
    REQUIRE(forward.transform_size == transform_size);
    REQUIRE(inverse.transform_size == transform_size);
    const char* const configured_backend = std::getenv("CELEPHAIS_FFT_BACKEND");
#ifdef CELEPHAIS_ENABLE_FFTW_ORACLE
    const auto expected_backend = configured_backend != nullptr
                                          && std::strcmp(configured_backend, "fftw") == 0
                                      ? Kadath::r2hc_backend::fftw
                                      : Kadath::r2hc_backend::native;
    REQUIRE(forward.backend() == expected_backend);
    REQUIRE(inverse.backend() == expected_backend);
#else
    REQUIRE((configured_backend == nullptr || configured_backend[0] == '\0'
             || std::strcmp(configured_backend, "native") == 0));
    REQUIRE(forward.backend() == Kadath::r2hc_backend::native);
    REQUIRE(inverse.backend() == Kadath::r2hc_backend::native);
#endif
    REQUIRE(reinterpret_cast<std::uintptr_t>(forward.buffer) % 64 == 0);
    REQUIRE(reinterpret_cast<std::uintptr_t>(inverse.buffer) % 64 == 0);

    Kadath::coef_1d_r2hc(transform_size * 2);
    Kadath::coef_i_1d_hc2r(transform_size * 2);
    REQUIRE(&Kadath::coef_1d_r2hc(transform_size) == &forward);
    REQUIRE(&Kadath::coef_i_1d_hc2r(transform_size) == &inverse);

    for (int i = 0; i <= transform_size; ++i) {
        const auto sin_pi_bits = std::bit_cast<std::uint64_t>(
            std::sin(std::numbers::pi_v<double> * i / transform_size));
        const auto sin_half_right_bits = std::bit_cast<std::uint64_t>(
            std::sin(std::numbers::pi_v<double> * i / 2. / transform_size));
        const auto sin_half_left_bits = std::bit_cast<std::uint64_t>(
            std::sin(std::numbers::pi_v<double> / 2. * i / transform_size));

        REQUIRE(std::bit_cast<std::uint64_t>(forward.sin_pi_over_n[i]) == sin_pi_bits);
        REQUIRE(std::bit_cast<std::uint64_t>(forward.sin_pi_i_over_2n[i]) == sin_half_right_bits);
        REQUIRE(std::bit_cast<std::uint64_t>(forward.sin_half_pi_i_over_n[i]) == sin_half_left_bits);
        REQUIRE(std::bit_cast<std::uint64_t>(inverse.sin_pi_over_n[i]) == sin_pi_bits);
        REQUIRE(std::bit_cast<std::uint64_t>(inverse.sin_pi_i_over_2n[i]) == sin_half_right_bits);
        REQUIRE(std::bit_cast<std::uint64_t>(inverse.sin_half_pi_i_over_n[i]) == sin_half_left_bits);
    }

    const auto quarter_bits = std::bit_cast<std::uint64_t>(
        std::sin(std::numbers::pi_v<double> * transform_size / 4 / transform_size));
    REQUIRE(std::bit_cast<std::uint64_t>(forward.sin_pi_quarter) == quarter_bits);
    REQUIRE(std::bit_cast<std::uint64_t>(inverse.sin_pi_quarter) == quarter_bits);

    constexpr std::array invalid_sizes{-1, 0, 1, 3, 31, 33, 34, 1'000'000'000};
    constexpr std::array backends{Kadath::r2hc_backend::native, Kadath::r2hc_backend::fftw};
    constexpr std::array directions{Kadath::r2hc_direction::forward,
                                    Kadath::r2hc_direction::inverse};
    for (const int invalid_size : invalid_sizes) {
        CAPTURE(invalid_size);
        REQUIRE_THROWS_AS(Kadath::coef_1d_r2hc(invalid_size), Kadath::KadathError);
        REQUIRE_THROWS_AS(Kadath::coef_i_1d_hc2r(invalid_size), Kadath::KadathError);
        for (const auto backend : backends) {
            for (const auto direction : directions) {
                CAPTURE(backend, direction);
                REQUIRE_THROWS_AS(Kadath::r2hc_precomp_t(invalid_size, direction, backend),
                                  std::invalid_argument);
            }
        }
    }

    Kadath::r2hc_precomp_t native_forward(transform_size, Kadath::r2hc_direction::forward,
                                          Kadath::r2hc_backend::native);
    Kadath::r2hc_precomp_t native_inverse(transform_size, Kadath::r2hc_direction::inverse,
                                          Kadath::r2hc_backend::native);
    REQUIRE_THROWS_AS(native_forward.execute_hc2r(), std::logic_error);
    REQUIRE_THROWS_AS(native_inverse.execute_r2hc(), std::logic_error);
#ifdef CELEPHAIS_ENABLE_FFTW_ORACLE
    Kadath::r2hc_precomp_t fftw_forward(transform_size, Kadath::r2hc_direction::forward,
                                        Kadath::r2hc_backend::fftw);
    Kadath::r2hc_precomp_t fftw_inverse(transform_size, Kadath::r2hc_direction::inverse,
                                        Kadath::r2hc_backend::fftw);
    REQUIRE_THROWS_AS(fftw_forward.execute_hc2r(), std::logic_error);
    REQUIRE_THROWS_AS(fftw_inverse.execute_r2hc(), std::logic_error);
#else
    REQUIRE(invalid_argument_message([&] {
                Kadath::r2hc_precomp_t(transform_size, Kadath::r2hc_direction::forward,
                                      Kadath::r2hc_backend::fftw);
            })
            == "The FFTW backend is unavailable in this production build; use "
               "CELEPHAIS_FFT_BACKEND=native or leave CELEPHAIS_FFT_BACKEND unset");
#endif
}

TEST_CASE("real transforms propagate non-finite inputs at every selected size",
          "[coef-precomp][native-r2hc][direct-real-oracle]")
{
#ifdef CELEPHAIS_ENABLE_FFTW_ORACLE
    constexpr std::array backends{Kadath::r2hc_backend::native, Kadath::r2hc_backend::fftw};
#else
    constexpr std::array backends{Kadath::r2hc_backend::native};
#endif
    constexpr std::array non_finite_inputs{std::numeric_limits<double>::quiet_NaN(),
                                           std::numeric_limits<double>::infinity()};
    for (const int n : direct_real_transform_sizes) {
        for (const auto backend : backends) {
            for (const double non_finite : non_finite_inputs) {
                CAPTURE(n, backend, non_finite);
                Kadath::r2hc_precomp_t forward(n, Kadath::r2hc_direction::forward, backend);
                Kadath::r2hc_precomp_t inverse(n, Kadath::r2hc_direction::inverse, backend);

                std::fill_n(forward.buffer, n, 0.);
                forward.buffer[n / 3] = non_finite;
                forward.execute_r2hc();
                REQUIRE(std::any_of(forward.buffer, forward.buffer + n,
                                    [](double value) { return !std::isfinite(value); }));

                std::fill_n(inverse.buffer, n, 0.);
                inverse.buffer[n - 1] = non_finite;
                inverse.execute_hc2r();
                REQUIRE(std::any_of(inverse.buffer, inverse.buffer + n,
                                    [](double value) { return !std::isfinite(value); }));
            }
        }
    }
}

TEST_CASE("fused native spectral lines cover every selected transform size",
          "[coef-precomp][native-r2hc][native-fused-spectral]")
{
    struct family_case
    {
        Kadath::native_spectral_family family;
        int base;
    };
    constexpr std::array transform_sizes{6, 8, 10, 12, 14, 16, 18, 20};
    constexpr std::array families{
        family_case{Kadath::native_spectral_family::cheb, CHEB},
        family_case{Kadath::native_spectral_family::cheb_even, CHEB_EVEN},
        family_case{Kadath::native_spectral_family::cheb_odd, CHEB_ODD},
        family_case{Kadath::native_spectral_family::cossin, COSSIN},
    };
    constexpr std::array strides{1, 3};
    constexpr double guard = -0x1.5a5a5a5a5a5a5p+17;

    const char* const fused_flag = std::getenv("CELEPHAIS_NATIVE_FUSED");
    const bool fused_disabled = fused_flag != nullptr && std::strcmp(fused_flag, "0") == 0;

    for (const int transform_size : transform_sizes) {
        Kadath::r2hc_precomp_t forward(transform_size, Kadath::r2hc_direction::forward,
                                      Kadath::r2hc_backend::native);
        Kadath::r2hc_precomp_t inverse(transform_size, Kadath::r2hc_direction::inverse,
                                      Kadath::r2hc_backend::native);

        for (const family_case test : families) {
            const bool cossin = test.family == Kadath::native_spectral_family::cossin;
            const int forward_input_size = cossin ? transform_size : transform_size + 1;
            const int forward_output_size = cossin ? transform_size + 2 : transform_size + 1;
            const int inverse_input_size = forward_output_size;
            const int inverse_output_size = forward_input_size;

            for (const int stride : strides) {
                CAPTURE(transform_size, test.base, stride);
#if defined(__linux__) && defined(__x86_64__) && defined(__GNUC__) \
    && !defined(__clang__)
                const bool measured_slower_forward = cossin && transform_size >= 18;
#else
                const bool measured_slower_forward = cossin && transform_size >= 10;
#endif
                const bool measured_slower_inverse =
                    measured_slower_forward
                    || ((transform_size == 18 || transform_size == 20)
                        && (test.family == Kadath::native_spectral_family::cheb
                            || test.family == Kadath::native_spectral_family::cheb_even));
                const int extent = std::max(forward_output_size, inverse_input_size);
                const auto storage_size = static_cast<std::size_t>((extent - 1) * stride + 1);

                std::vector<double> forward_input(storage_size, guard);
                std::vector<double> fused_forward(storage_size, guard);
                std::vector<double> fused_in_place_forward(storage_size, guard);
                for (int i = 0; i < forward_input_size; ++i)
                    forward_input[static_cast<std::size_t>(i * stride)] =
                        0.125 * (i - 3) + std::sin(0.37 * i) - 0.25 * std::cos(0.19 * i);
                fused_in_place_forward = forward_input;

                const bool accepted_forward = forward.try_execute_fused_forward(
                    test.family, forward_input.data(), fused_forward.data(),
                    forward_input_size, forward_output_size, stride);
                const bool accepted_in_place_forward = forward.try_execute_fused_forward(
                    test.family, fused_in_place_forward.data(), fused_in_place_forward.data(),
                    forward_input_size, forward_output_size, stride);
                if (fused_disabled || measured_slower_forward) {
                    REQUIRE_FALSE(accepted_forward);
                    REQUIRE_FALSE(accepted_in_place_forward);
                }
                else {
                    REQUIRE(accepted_forward);
                    REQUIRE(accepted_in_place_forward);

                    std::vector<double> public_forward(storage_size, guard);
                    std::vector<double> in_place_forward = forward_input;
                    REQUIRE(Kadath::coef_1d(test.base, forward_input.data(), public_forward.data(),
                                            forward_input_size, forward_output_size, stride));
                    REQUIRE(Kadath::coef_1d(test.base, in_place_forward.data(),
                                            in_place_forward.data(), forward_input_size,
                                            forward_output_size, stride));
                    for (int i = 0; i < forward_output_size; ++i) {
                        const auto index = static_cast<std::size_t>(i * stride);
                        const double scale = 2.e-11 * transform_size
                                             * std::max({1., std::abs(fused_forward[index]),
                                                         std::abs(public_forward[index])});
                        REQUIRE(std::abs(fused_forward[index] - public_forward[index]) <= scale);
                        REQUIRE(std::bit_cast<std::uint64_t>(fused_in_place_forward[index])
                                == std::bit_cast<std::uint64_t>(fused_forward[index]));
                        REQUIRE(std::bit_cast<std::uint64_t>(in_place_forward[index])
                                == std::bit_cast<std::uint64_t>(public_forward[index]));
                    }
#if defined(__linux__) && defined(__x86_64__) && defined(__GNUC__) \
    && !defined(__clang__)
                    if (cossin && transform_size >= 10 && transform_size <= 16
                        && stride == 1) {
                        std::vector<double> workspace_alias_forward(
                            storage_size, guard);
                        std::copy_n(forward_input.data(), transform_size,
                                    forward.buffer);
                        REQUIRE(forward.try_execute_fused_forward(
                            test.family, forward.buffer,
                            workspace_alias_forward.data(), forward_input_size,
                            forward_output_size, stride));
                        for (int i = 0; i < forward_output_size; ++i)
                            REQUIRE(std::bit_cast<std::uint64_t>(
                                        workspace_alias_forward[
                                            static_cast<std::size_t>(i)])
                                    == std::bit_cast<std::uint64_t>(
                                        fused_forward[
                                            static_cast<std::size_t>(i)]));
                    }
#endif
                    if (stride > 1) {
                        for (std::size_t i = 0; i < storage_size; ++i)
                            if (i % static_cast<std::size_t>(stride) != 0) {
                                REQUIRE(fused_forward[i] == guard);
                                REQUIRE(fused_in_place_forward[i] == guard);
                                REQUIRE(public_forward[i] == guard);
                                REQUIRE(in_place_forward[i] == guard);
                            }
                    }
                }

                std::vector<double> inverse_input(storage_size, guard);
                std::vector<double> fused_inverse(storage_size, guard);
                std::vector<double> fused_in_place_inverse(storage_size, guard);
                for (int i = 0; i < inverse_input_size; ++i)
                    inverse_input[static_cast<std::size_t>(i * stride)] =
                        -0.0625 * (i + 2) + 0.015625 * (i % 2) + std::cos(0.23 * i);
                fused_in_place_inverse = inverse_input;

                const bool accepted_inverse = inverse.try_execute_fused_inverse(
                    test.family, inverse_input.data(), fused_inverse.data(),
                    inverse_input_size, inverse_output_size, stride);
                const bool accepted_in_place_inverse = inverse.try_execute_fused_inverse(
                    test.family, fused_in_place_inverse.data(), fused_in_place_inverse.data(),
                    inverse_input_size, inverse_output_size, stride);
                if (fused_disabled || measured_slower_inverse) {
                    REQUIRE_FALSE(accepted_inverse);
                    REQUIRE_FALSE(accepted_in_place_inverse);
                    continue;
                }
                REQUIRE(accepted_inverse);
                REQUIRE(accepted_in_place_inverse);

                std::vector<double> public_inverse(storage_size, guard);
                std::vector<double> in_place_inverse = inverse_input;
                REQUIRE(Kadath::coef_i_1d(test.base, inverse_input.data(), public_inverse.data(),
                                          inverse_input_size, inverse_output_size, stride));
                REQUIRE(Kadath::coef_i_1d(test.base, in_place_inverse.data(),
                                          in_place_inverse.data(), inverse_input_size,
                                          inverse_output_size, stride));
                for (int i = 0; i < inverse_output_size; ++i) {
                    const auto index = static_cast<std::size_t>(i * stride);
                    const double scale = 2.e-11 * transform_size
                                         * std::max({1., std::abs(fused_inverse[index]),
                                                     std::abs(public_inverse[index])});
                    REQUIRE(std::abs(fused_inverse[index] - public_inverse[index]) <= scale);
                    REQUIRE(std::bit_cast<std::uint64_t>(fused_in_place_inverse[index])
                            == std::bit_cast<std::uint64_t>(fused_inverse[index]));
                    REQUIRE(std::bit_cast<std::uint64_t>(in_place_inverse[index])
                            == std::bit_cast<std::uint64_t>(public_inverse[index]));
                }
                if (cossin) {
                    for (int i = inverse_output_size; i < inverse_input_size; ++i) {
                        const auto index = static_cast<std::size_t>(i * stride);
                        REQUIRE(fused_inverse[index] == guard);
                        REQUIRE(public_inverse[index] == guard);
                        REQUIRE(std::bit_cast<std::uint64_t>(fused_in_place_inverse[index])
                                == std::bit_cast<std::uint64_t>(inverse_input[index]));
                        REQUIRE(std::bit_cast<std::uint64_t>(in_place_inverse[index])
                                == std::bit_cast<std::uint64_t>(inverse_input[index]));
                    }
                }
                if (stride > 1) {
                    for (std::size_t i = 0; i < storage_size; ++i)
                        if (i % static_cast<std::size_t>(stride) != 0) {
                            REQUIRE(fused_inverse[i] == guard);
                            REQUIRE(fused_in_place_inverse[i] == guard);
                            REQUIRE(public_inverse[i] == guard);
                            REQUIRE(in_place_inverse[i] == guard);
                        }
                }
            }
        }
    }

    if (native_backend_selected()) {
        for (const int transform_size : native_cos_sin_transform_sizes) {
            Kadath::r2hc_precomp_t forward(transform_size,
                                           Kadath::r2hc_direction::forward,
                                           Kadath::r2hc_backend::native);
            Kadath::r2hc_precomp_t inverse(transform_size,
                                           Kadath::r2hc_direction::inverse,
                                           Kadath::r2hc_backend::native);
            const int extent = transform_size + 1;

            for (const native_cos_sin_family_case test : native_cos_sin_families) {
                for (const int stride : native_cos_sin_strides) {
                    CAPTURE(transform_size, test.base, stride);
                    const auto storage_size =
                        static_cast<std::size_t>((extent - 1) * stride + 1);

                    std::vector<double> forward_input(storage_size, guard);
                    for (int i = 0; i < extent; ++i)
                        forward_input[static_cast<std::size_t>(i * stride)] =
                            0.125 * (i - 3) + std::sin(0.37 * i)
                            - 0.25 * std::cos(0.19 * i);
                    std::vector<double> fused_forward(storage_size, guard);
                    std::vector<double> fused_in_place_forward = forward_input;
                    const bool accepted_forward = forward.try_execute_fused_forward(
                        test.family, forward_input.data(), fused_forward.data(),
                        extent, extent, stride);
                    const bool accepted_in_place_forward =
                        forward.try_execute_fused_forward(
                            test.family, fused_in_place_forward.data(),
                            fused_in_place_forward.data(), extent, extent, stride);

                    const bool forward_enabled = native_cos_sin_fused_expected(
                        transform_size, test.family, Kadath::r2hc_direction::forward);

                    if (fused_disabled || !forward_enabled) {
                        REQUIRE_FALSE(accepted_forward);
                        REQUIRE_FALSE(accepted_in_place_forward);
                    }
                    else {
                        REQUIRE(accepted_forward);
                        REQUIRE(accepted_in_place_forward);
                        std::vector<double> public_forward(storage_size, guard);
                        std::vector<double> public_in_place_forward = forward_input;
                        REQUIRE(Kadath::coef_1d(test.base, forward_input.data(),
                                                public_forward.data(), extent, extent,
                                                stride));
                        REQUIRE(Kadath::coef_1d(test.base,
                                                public_in_place_forward.data(),
                                                public_in_place_forward.data(), extent,
                                                extent, stride));
                        for (int i = 0; i < extent; ++i) {
                            const auto index = static_cast<std::size_t>(i * stride);
                            REQUIRE(std::bit_cast<std::uint64_t>(fused_forward[index])
                                    == std::bit_cast<std::uint64_t>(
                                        public_forward[index]));
                            REQUIRE(std::bit_cast<std::uint64_t>(
                                        fused_in_place_forward[index])
                                    == std::bit_cast<std::uint64_t>(
                                        fused_forward[index]));
                            REQUIRE(std::bit_cast<std::uint64_t>(
                                        public_in_place_forward[index])
                                    == std::bit_cast<std::uint64_t>(
                                        public_forward[index]));
                        }
                        if (stride > 1) {
                            for (std::size_t i = 0; i < storage_size; ++i)
                                if (i % static_cast<std::size_t>(stride) != 0) {
                                    REQUIRE(fused_forward[i] == guard);
                                    REQUIRE(fused_in_place_forward[i] == guard);
                                    REQUIRE(public_forward[i] == guard);
                                    REQUIRE(public_in_place_forward[i] == guard);
                                }
                        }
                    }

                    std::vector<double> inverse_input(storage_size, guard);
                    for (int i = 0; i < extent; ++i)
                        inverse_input[static_cast<std::size_t>(i * stride)] =
                            -0.0625 * (i + 2) + 0.015625 * (i % 2)
                            + std::cos(0.23 * i);
                    std::vector<double> fused_inverse(storage_size, guard);
                    std::vector<double> fused_in_place_inverse = inverse_input;
                    const bool accepted_inverse = inverse.try_execute_fused_inverse(
                        test.family, inverse_input.data(), fused_inverse.data(),
                        extent, extent, stride);
                    const bool accepted_in_place_inverse =
                        inverse.try_execute_fused_inverse(
                            test.family, fused_in_place_inverse.data(),
                            fused_in_place_inverse.data(), extent, extent, stride);
                    const bool inverse_enabled = native_cos_sin_fused_expected(
                        transform_size, test.family, Kadath::r2hc_direction::inverse);

                    if (fused_disabled || !inverse_enabled) {
                        REQUIRE_FALSE(accepted_inverse);
                        REQUIRE_FALSE(accepted_in_place_inverse);
                        continue;
                    }
                    REQUIRE(accepted_inverse);
                    REQUIRE(accepted_in_place_inverse);
                    std::vector<double> public_inverse(storage_size, guard);
                    std::vector<double> public_in_place_inverse = inverse_input;
                    REQUIRE(Kadath::coef_i_1d(test.base, inverse_input.data(),
                                              public_inverse.data(), extent, extent,
                                              stride));
                    REQUIRE(Kadath::coef_i_1d(test.base,
                                              public_in_place_inverse.data(),
                                              public_in_place_inverse.data(), extent,
                                              extent, stride));
                    for (int i = 0; i < extent; ++i) {
                        const auto index = static_cast<std::size_t>(i * stride);
                        REQUIRE(std::bit_cast<std::uint64_t>(fused_inverse[index])
                                == std::bit_cast<std::uint64_t>(
                                    public_inverse[index]));
                        REQUIRE(std::bit_cast<std::uint64_t>(
                                    fused_in_place_inverse[index])
                                == std::bit_cast<std::uint64_t>(
                                    fused_inverse[index]));
                        REQUIRE(std::bit_cast<std::uint64_t>(
                                    public_in_place_inverse[index])
                                == std::bit_cast<std::uint64_t>(
                                    public_inverse[index]));
                    }
                    if (stride > 1) {
                        for (std::size_t i = 0; i < storage_size; ++i)
                            if (i % static_cast<std::size_t>(stride) != 0) {
                                REQUIRE(fused_inverse[i] == guard);
                                REQUIRE(fused_in_place_inverse[i] == guard);
                                REQUIRE(public_inverse[i] == guard);
                                REQUIRE(public_in_place_inverse[i] == guard);
                            }
                    }
                }
            }
        }

#if (defined(__APPLE__) && defined(__aarch64__) && defined(__clang__)) \
    || (defined(__linux__) && defined(__x86_64__) && defined(__GNUC__) \
        && !defined(__clang__))
        for (std::size_t size_index = 0;
             size_index < native_cos_sin_transform_sizes.size(); ++size_index) {
            const int transform_size = native_cos_sin_transform_sizes[size_index];
            const int extent = transform_size + 1;
            for (const native_cos_sin_family_case test : native_cos_sin_families) {
                for (const bool inverse : {false, true}) {
                    for (std::size_t stride_index = 0;
                         stride_index < native_cos_sin_strides.size(); ++stride_index) {
                        const int stride = native_cos_sin_strides[stride_index];
                        CAPTURE(transform_size, test.base, inverse, stride);
                        const auto storage_size =
                            static_cast<std::size_t>((extent - 1) * stride + 1);
                        std::uint64_t hash = 1469598103934665603ULL;
                        bool all_accepted = true;
                        for (int salt = 0; salt < 128; ++salt) {
                            const int fixture_salt = transform_size + stride
                                                     + 3 * test.base
                                                     + (inverse ? 53 : 41) + salt;
                            std::vector<double> input(storage_size, -91.25);
                            for (int i = 0; i < extent; ++i)
                                input[static_cast<std::size_t>(i * stride)] =
                                    static_cast<double>((17 * i + 11 * fixture_salt) % 37
                                                        - 18)
                                        / 16.
                                    + 0.03125;
                            std::vector<double> output(storage_size, -77.5);
                            const bool accepted = inverse
                                ? Kadath::coef_i_1d(test.base, input.data(), output.data(),
                                                    extent, extent, stride)
                                : Kadath::coef_1d(test.base, input.data(), output.data(),
                                                  extent, extent, stride);
                            all_accepted = all_accepted && accepted;
                            if (!accepted)
                                break;
                            for (int i = 0; i < extent; ++i)
                                hash_value(hash,
                                           output[static_cast<std::size_t>(i * stride)]);
                        }
                        REQUIRE(all_accepted);
                        const auto expected_index =
                            (((size_index * native_cos_sin_families.size()
                               + test.hash_index)
                              * 2
                              + static_cast<std::size_t>(inverse))
                             * native_cos_sin_strides.size())
                            + stride_index;
                        REQUIRE(hash == native_cos_sin_buffer_hashes[expected_index]);
                    }
                }
            }
        }
#else
        SKIP("no exact native COS/SIN corpus is qualified for this platform/compiler");
#endif
    }
}

#if defined(__linux__) && defined(__x86_64__) && defined(__GNUC__) \
    && !defined(__clang__)
TEST_CASE("GCC cached COSSIN forward plan matches the public fallback",
          "[coef-precomp][native-r2hc][native-cossin-cached]")
{
    if (!native_backend_selected())
        SKIP("the cached COSSIN route is native-only");
    const char* const fused_flag = std::getenv("CELEPHAIS_NATIVE_FUSED");
    if (fused_flag != nullptr && std::strcmp(fused_flag, "0") == 0)
        SKIP("the cached COSSIN route is disabled by configuration");

    constexpr std::array target_sizes{10, 12, 14, 16};
    constexpr std::array declined_sizes{6, 8, 18, 20};
    constexpr std::array strides{1, 2, 3, 5};
    constexpr double guard = -0x1.6b6b6b6b6b6b6p+18;

    const auto require_guards = [guard](const std::vector<double>& values,
                                        const int line_size,
                                        const int stride) {
        for (std::size_t index = 0; index < values.size(); ++index) {
            const bool line_slot = index % static_cast<std::size_t>(stride) == 0
                                   && index / static_cast<std::size_t>(stride)
                                          < static_cast<std::size_t>(line_size);
            if (!line_slot)
                REQUIRE(values[index] == guard);
        }
    };

    for (const int transform_size : target_sizes) {
        auto& transform = Kadath::coef_1d_r2hc(transform_size);
        const int input_size = transform_size;
        const int output_size = transform_size + 2;

        for (const int stride : strides) {
            CAPTURE(transform_size, stride);
            const auto storage_size =
                static_cast<std::size_t>((output_size + 2) * stride + 3);
            std::vector<double> input(storage_size, guard);
            for (int i = 0; i < input_size; ++i)
                input[static_cast<std::size_t>(i * stride)] =
                    0.1875 * (i - 4) + std::sin(0.31 * i)
                    - 0.125 * std::cos(0.17 * i);

            std::vector<double> direct(storage_size, guard);
            std::vector<double> public_result(storage_size, guard);
            std::vector<double> direct_in_place = input;
            std::vector<double> public_in_place = input;
            REQUIRE(transform.try_execute_cached_cossin_forward(
                input.data(), direct.data(), input_size, output_size, stride));
            REQUIRE(Kadath::coef_1d(COSSIN, input.data(), public_result.data(),
                                    input_size, output_size, stride));
            REQUIRE(transform.try_execute_cached_cossin_forward(
                direct_in_place.data(), direct_in_place.data(), input_size,
                output_size, stride));
            REQUIRE(Kadath::coef_1d(COSSIN, public_in_place.data(),
                                    public_in_place.data(), input_size,
                                    output_size, stride));

            for (int i = 0; i < output_size; ++i) {
                const auto index = static_cast<std::size_t>(i * stride);
                REQUIRE(std::bit_cast<std::uint64_t>(public_result[index])
                        == std::bit_cast<std::uint64_t>(direct[index]));
                REQUIRE(std::bit_cast<std::uint64_t>(direct_in_place[index])
                        == std::bit_cast<std::uint64_t>(direct[index]));
                REQUIRE(std::bit_cast<std::uint64_t>(public_in_place[index])
                        == std::bit_cast<std::uint64_t>(direct[index]));
            }
            require_guards(direct, output_size, stride);
            require_guards(public_result, output_size, stride);
            require_guards(direct_in_place, output_size, stride);
            require_guards(public_in_place, output_size, stride);

            std::vector<double> refused(storage_size, guard);
            const std::vector<double> refused_before = refused;
            REQUIRE_FALSE(transform.try_execute_cached_cossin_forward(
                input.data(), refused.data(), input_size, output_size - 1,
                stride));
            REQUIRE(refused == refused_before);

            if (stride == 1) {
                std::vector<double> direct_workspace(storage_size, guard);
                std::vector<double> public_workspace(storage_size, guard);
                std::copy_n(input.data(), input_size, transform.buffer);
                REQUIRE(transform.try_execute_cached_cossin_forward(
                    transform.buffer, direct_workspace.data(), input_size,
                    output_size, stride));
                std::copy_n(input.data(), input_size, transform.buffer);
                REQUIRE(Kadath::coef_1d(
                    COSSIN, transform.buffer, public_workspace.data(),
                    input_size, output_size, stride));
                for (int i = 0; i < output_size; ++i) {
                    const auto index = static_cast<std::size_t>(i);
                    REQUIRE(std::bit_cast<std::uint64_t>(
                                direct_workspace[index])
                            == std::bit_cast<std::uint64_t>(direct[index]));
                    REQUIRE(std::bit_cast<std::uint64_t>(
                                public_workspace[index])
                            == std::bit_cast<std::uint64_t>(direct[index]));
                }
                require_guards(direct_workspace, output_size, stride);
                require_guards(public_workspace, output_size, stride);
            }
        }
    }

    for (const int transform_size : declined_sizes) {
        auto& transform = Kadath::coef_1d_r2hc(transform_size);
        const int input_size = transform_size;
        const int output_size = transform_size + 2;

        for (const int stride : strides) {
            CAPTURE(transform_size, stride);
            const auto storage_size =
                static_cast<std::size_t>((output_size + 2) * stride + 3);
            std::vector<double> input(storage_size, guard);
            for (int i = 0; i < input_size; ++i)
                input[static_cast<std::size_t>(i * stride)] =
                    -0.09375 * (i + 1) + std::cos(0.29 * i);

            std::vector<double> refused(storage_size, guard);
            const std::vector<double> refused_before = refused;
            REQUIRE_FALSE(transform.try_execute_cached_cossin_forward(
                input.data(), refused.data(), input_size, output_size, stride));
            REQUIRE(refused == refused_before);

            std::vector<double> public_result(storage_size, guard);
            std::vector<double> public_in_place = input;
            REQUIRE(Kadath::coef_1d(COSSIN, input.data(), public_result.data(),
                                    input_size, output_size, stride));
            REQUIRE(Kadath::coef_1d(COSSIN, public_in_place.data(),
                                    public_in_place.data(), input_size,
                                    output_size, stride));
            for (int i = 0; i < output_size; ++i) {
                const auto index = static_cast<std::size_t>(i * stride);
                REQUIRE(std::isfinite(public_result[index]));
                REQUIRE(std::bit_cast<std::uint64_t>(public_in_place[index])
                        == std::bit_cast<std::uint64_t>(public_result[index]));
            }
            require_guards(public_result, output_size, stride);
            require_guards(public_in_place, output_size, stride);
        }
    }
}
#endif

TEST_CASE("fused native spectral lines refuse unsupported calls and propagate NaNs",
          "[coef-precomp][native-r2hc][native-fused-spectral]")
{
    constexpr std::array transform_sizes{6, 8, 10, 12, 14, 16, 18, 20};
    constexpr std::array families{
        Kadath::native_spectral_family::cheb,
        Kadath::native_spectral_family::cheb_even,
        Kadath::native_spectral_family::cheb_odd,
        Kadath::native_spectral_family::cossin,
    };
    const char* const fused_flag = std::getenv("CELEPHAIS_NATIVE_FUSED");
    const bool fused_disabled = fused_flag != nullptr && std::strcmp(fused_flag, "0") == 0;

    std::array<double, 24> source{};
    std::array<double, 24> destination{};
    for (const int transform_size : transform_sizes) {
        Kadath::r2hc_precomp_t forward(transform_size, Kadath::r2hc_direction::forward,
                                      Kadath::r2hc_backend::native);
        Kadath::r2hc_precomp_t inverse(transform_size, Kadath::r2hc_direction::inverse,
                                      Kadath::r2hc_backend::native);
        for (const auto family : families) {
            CAPTURE(transform_size, family);
            const bool cossin = family == Kadath::native_spectral_family::cossin;
#if defined(__linux__) && defined(__x86_64__) && defined(__GNUC__) \
    && !defined(__clang__)
            const bool measured_slower_forward = cossin && transform_size >= 18;
#else
            const bool measured_slower_forward = cossin && transform_size >= 10;
#endif
            const bool measured_slower_inverse =
                measured_slower_forward
                || ((transform_size == 18 || transform_size == 20)
                    && (family == Kadath::native_spectral_family::cheb
                        || family == Kadath::native_spectral_family::cheb_even));
            const int forward_input_size = cossin ? transform_size : transform_size + 1;
            const int forward_output_size = cossin ? transform_size + 2 : transform_size + 1;

            source.fill(0.);
            destination.fill(0.);
            source[2] = std::numeric_limits<double>::quiet_NaN();
            const bool accepted_forward = forward.try_execute_fused_forward(
                family, source.data(), destination.data(),
                forward_input_size, forward_output_size, 1);
            if (fused_disabled || measured_slower_forward) {
                REQUIRE_FALSE(accepted_forward);
            }
            else {
                REQUIRE(accepted_forward);
                REQUIRE(std::any_of(destination.begin(),
                                    destination.begin() + forward_output_size,
                                    [](double value) { return !std::isfinite(value); }));
            }

            source.fill(0.);
            destination.fill(0.);
            source[2] = std::numeric_limits<double>::quiet_NaN();
            const bool accepted_inverse = inverse.try_execute_fused_inverse(
                family, source.data(), destination.data(),
                forward_output_size, forward_input_size, 1);
            if (fused_disabled || measured_slower_inverse) {
                REQUIRE_FALSE(accepted_inverse);
            }
            else {
                REQUIRE(accepted_inverse);
                REQUIRE(std::any_of(destination.begin(),
                                    destination.begin() + forward_input_size,
                                    [](double value) { return !std::isfinite(value); }));
            }
        }
    }

    // N=22 is a supported raw native transform but deliberately remains outside
    // the N<=20 fused spectral-line contract.
    Kadath::r2hc_precomp_t unsupported_fused_forward(22, Kadath::r2hc_direction::forward,
                                                     Kadath::r2hc_backend::native);
    Kadath::r2hc_precomp_t unsupported_fused_inverse(22, Kadath::r2hc_direction::inverse,
                                                     Kadath::r2hc_backend::native);
    Kadath::r2hc_precomp_t native_forward(6, Kadath::r2hc_direction::forward,
                                         Kadath::r2hc_backend::native);
    Kadath::r2hc_precomp_t native_inverse(6, Kadath::r2hc_direction::inverse,
                                         Kadath::r2hc_backend::native);
#ifdef CELEPHAIS_ENABLE_FFTW_ORACLE
    Kadath::r2hc_precomp_t test_oracle_fftw(6, Kadath::r2hc_direction::forward,
                                           Kadath::r2hc_backend::fftw);
#endif

    REQUIRE_FALSE(unsupported_fused_forward.try_execute_fused_forward(
        Kadath::native_spectral_family::cheb, source.data(), destination.data(), 23, 23, 1));
    REQUIRE_FALSE(unsupported_fused_inverse.try_execute_fused_inverse(
        Kadath::native_spectral_family::cheb, source.data(), destination.data(), 23, 23, 1));
#ifdef CELEPHAIS_ENABLE_FFTW_ORACLE
    REQUIRE_FALSE(test_oracle_fftw.try_execute_fused_forward(
        Kadath::native_spectral_family::cheb, source.data(), destination.data(), 7, 7, 1));
#endif
    REQUIRE_FALSE(native_forward.try_execute_fused_inverse(
        Kadath::native_spectral_family::cheb, source.data(), destination.data(), 7, 7, 1));
    REQUIRE_FALSE(native_inverse.try_execute_fused_forward(
        Kadath::native_spectral_family::cheb, source.data(), destination.data(), 7, 7, 1));
    REQUIRE_FALSE(native_forward.try_execute_fused_forward(
        Kadath::native_spectral_family::cheb, source.data(), destination.data(), 6, 7, 1));
    REQUIRE_FALSE(native_forward.try_execute_fused_forward(
        Kadath::native_spectral_family::cossin, source.data(), destination.data(), 6, 7, 1));
    REQUIRE_FALSE(native_inverse.try_execute_fused_inverse(
        Kadath::native_spectral_family::cossin, source.data(), destination.data(), 7, 6, 1));

    if (native_backend_selected()) {
        constexpr double new_family_guard = -0x1.6b6b6b6b6b6b6p+18;
        for (const int transform_size : native_cos_sin_transform_sizes) {
            Kadath::r2hc_precomp_t forward(transform_size,
                                           Kadath::r2hc_direction::forward,
                                           Kadath::r2hc_backend::native);
            Kadath::r2hc_precomp_t inverse(transform_size,
                                           Kadath::r2hc_direction::inverse,
                                           Kadath::r2hc_backend::native);
            const int extent = transform_size + 1;

            for (const native_cos_sin_family_case test : native_cos_sin_families) {
                for (const int stride : native_cos_sin_strides) {
                    CAPTURE(transform_size, test.base, stride);
                    const auto storage_size =
                        static_cast<std::size_t>((extent - 1) * stride + 1);
                    std::vector<double> nan_source(storage_size, new_family_guard);
                    std::vector<double> nan_destination(storage_size, new_family_guard);
                    for (int i = 0; i < extent; ++i)
                        nan_source[static_cast<std::size_t>(i * stride)] = 0.;
                    nan_source[static_cast<std::size_t>(2 * stride)] =
                        std::numeric_limits<double>::quiet_NaN();

                    const bool accepted_forward = forward.try_execute_fused_forward(
                        test.family, nan_source.data(), nan_destination.data(),
                        extent, extent, stride);
                    const bool forward_enabled = native_cos_sin_fused_expected(
                        transform_size, test.family, Kadath::r2hc_direction::forward);
                    if (fused_disabled || !forward_enabled) {
                        REQUIRE_FALSE(accepted_forward);
                    }
                    else {
                        REQUIRE(accepted_forward);
                        bool found_non_finite = false;
                        for (int i = 0; i < extent; ++i)
                            found_non_finite =
                                found_non_finite
                                || !std::isfinite(nan_destination[
                                    static_cast<std::size_t>(i * stride)]);
                        REQUIRE(found_non_finite);
                    }

                    std::fill(nan_destination.begin(), nan_destination.end(),
                              new_family_guard);
                    const bool accepted_inverse = inverse.try_execute_fused_inverse(
                        test.family, nan_source.data(), nan_destination.data(),
                        extent, extent, stride);
                    const bool inverse_enabled = native_cos_sin_fused_expected(
                        transform_size, test.family, Kadath::r2hc_direction::inverse);
                    if (fused_disabled || !inverse_enabled) {
                        REQUIRE_FALSE(accepted_inverse);
                    }
                    else {
                        REQUIRE(accepted_inverse);
                        bool found_non_finite = false;
                        for (int i = 0; i < extent; ++i)
                            found_non_finite =
                                found_non_finite
                                || !std::isfinite(nan_destination[
                                    static_cast<std::size_t>(i * stride)]);
                        REQUIRE(found_non_finite);
                    }

                    std::vector<double> mismatch_source(storage_size, 0.25);
                    std::vector<double> mismatch_destination(storage_size,
                                                             new_family_guard);
                    const std::vector<double> unchanged_destination =
                        mismatch_destination;
                    const auto require_unchanged = [&]() {
                        REQUIRE(std::memcmp(mismatch_destination.data(),
                                            unchanged_destination.data(),
                                            storage_size * sizeof(double)) == 0);
                        mismatch_destination = unchanged_destination;
                    };

                    REQUIRE_FALSE(forward.try_execute_fused_forward(
                        test.family, mismatch_source.data(),
                        mismatch_destination.data(), extent - 1, extent, stride));
                    require_unchanged();
                    REQUIRE_FALSE(forward.try_execute_fused_forward(
                        test.family, mismatch_source.data(),
                        mismatch_destination.data(), extent, extent - 1, stride));
                    require_unchanged();
                    REQUIRE_FALSE(inverse.try_execute_fused_inverse(
                        test.family, mismatch_source.data(),
                        mismatch_destination.data(), extent - 1, extent, stride));
                    require_unchanged();
                    REQUIRE_FALSE(inverse.try_execute_fused_inverse(
                        test.family, mismatch_source.data(),
                        mismatch_destination.data(), extent, extent - 1, stride));
                    require_unchanged();
                }
            }
        }
    }
}

TEST_CASE("doubled COSSIN parity wrappers preserve fused N=18 and N=20 contracts",
          "[coef-precomp][native-r2hc][native-fused-spectral]")
{
    constexpr std::array line_sizes{11, 12};
    constexpr std::array bases{COSSIN_EVEN, COSSIN_ODD};
    constexpr std::array strides{1, 3};
    constexpr double guard = -0x1.3d3d3d3d3d3d3p+19;

    for (const int line_size : line_sizes) {
        for (const int base : bases) {
            for (const int stride : strides) {
                CAPTURE(line_size, base, stride);
                const auto storage_size =
                    static_cast<std::size_t>((line_size - 1) * stride + 1);
                std::vector<double> input(storage_size, guard);
                for (int i = 0; i < line_size - 2; ++i)
                    input[static_cast<std::size_t>(i * stride)] =
                        0.1 * (i - 3) + std::sin(0.2 * i);
                input[static_cast<std::size_t>((line_size - 2) * stride)] = 0.;
                input[static_cast<std::size_t>((line_size - 1) * stride)] = 0.;

                std::vector<double> coefficients(storage_size, guard);
                std::vector<double> in_place = input;
                REQUIRE(Kadath::coef_1d(base, input.data(), coefficients.data(),
                                        line_size, line_size, stride));
                REQUIRE(Kadath::coef_1d(base, in_place.data(), in_place.data(),
                                        line_size, line_size, stride));
                for (int i = 0; i < line_size; ++i) {
                    const auto index = static_cast<std::size_t>(i * stride);
                    REQUIRE(std::isfinite(coefficients[index]));
                    REQUIRE(std::bit_cast<std::uint64_t>(in_place[index])
                            == std::bit_cast<std::uint64_t>(coefficients[index]));
                }

                std::vector<double> round_trip(storage_size, guard);
                REQUIRE(Kadath::coef_i_1d(base, coefficients.data(), round_trip.data(),
                                          line_size, line_size, stride));
                const bool exact_length = line_size == 12 || base == COSSIN_EVEN;
                if (exact_length) {
                    for (int i = 0; i < line_size; ++i) {
                        const auto index = static_cast<std::size_t>(i * stride);
                        const double scale = 2.e-11 * line_size
                                             * std::max(1., std::abs(input[index]));
                        REQUIRE(std::abs(round_trip[index] - input[index]) <= scale);
                    }
                }
                else {
                    for (int i = 0; i < line_size; ++i)
                        REQUIRE(std::isfinite(
                            round_trip[static_cast<std::size_t>(i * stride)]));
                }
                REQUIRE(round_trip[static_cast<std::size_t>((line_size - 2) * stride)] == 0.);
                REQUIRE(round_trip[static_cast<std::size_t>((line_size - 1) * stride)] == 0.);

                if (stride > 1) {
                    for (std::size_t i = 0; i < storage_size; ++i) {
                        if (i % static_cast<std::size_t>(stride) == 0)
                            continue;
                        REQUIRE(coefficients[i] == guard);
                        REQUIRE(in_place[i] == guard);
                        REQUIRE(round_trip[i] == guard);
                    }
                }
            }
        }
    }
}
