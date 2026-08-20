#pragma once

#include "For_Kadath/IO/binary_sink.hpp"

#include <cstdint>
#include <fstream>
#include <string>

namespace Kadath
{
    // The exact unknown count (get_nbr_unknowns) of the most recent Newton solve.
    // The Newton driver stashes it here so save_to_file can stamp it into the
    // dataset; the reader reads it back instead of reconstructing the unknown set,
    // which is stage-dependent (e.g. the velocity potential is a var in some stages
    // and fixed in others) and not recoverable from the saved fields alone.
    // -1 (initial) means no solve has run in this process yet -> reader prints "n/a".
    inline int& last_solved_system_dof()
    {
        static int dof = -1;
        return dof;
    }

    // Magic tag written immediately before the DOF so a reader can tell a stamped
    // dataset from a pre-feature one (no trailing bytes -> read throws -> absent)
    // or from a misaligned read (tag mismatch -> absent). ASCII "DOF!".
    inline constexpr int kSystemDofMagic = 0x444F4621;

    // Append "<magic><dof>" after the fields. Called by save_to_file. The stash is
    // not cleared: a stage may save more than once, and the DOF of the solve that
    // produced the state is the same across those saves.
    inline void write_system_dof(BinarySink& sink)
    {
        sink.write<int>(kSystemDofMagic);
        sink.write<int>(last_solved_system_dof());
    }

    // Read the DOF stamped at the end of a saved dataset, addressed from the file
    // tail so it is independent of how many fields a given reader loads. Returns -1
    // when the file is too short, the tag is absent (a dataset saved before this
    // feature), or the read fails. BeFileSink always writes big-endian, so the two
    // trailing ints are decoded big-endian regardless of host.
    inline int read_system_dof_from_file(const std::string& path)
    {
        std::ifstream file(path, std::ios::binary | std::ios::ate);
        if (!file)
            return -1;
        const std::streamoff size = file.tellg();
        if (size < 8)
            return -1;
        file.seekg(size - 8);
        unsigned char bytes[8];
        if (!file.read(reinterpret_cast<char*>(bytes), 8))
            return -1;
        auto decode_be = [](const unsigned char* p) {
            return static_cast<int>((static_cast<std::uint32_t>(p[0]) << 24)
                                    | (static_cast<std::uint32_t>(p[1]) << 16)
                                    | (static_cast<std::uint32_t>(p[2]) << 8)
                                    | static_cast<std::uint32_t>(p[3]));
        };
        if (decode_be(bytes) != kSystemDofMagic)
            return -1;
        return decode_be(bytes + 4);
    }
} // namespace Kadath
