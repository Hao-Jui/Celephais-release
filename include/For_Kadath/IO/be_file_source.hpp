#pragma once
#include "binary_source.hpp"

#include <cstdio>
#include <string>

namespace Kadath
{

// RAII over FILE* "rb" with big-endian byte swap on little-endian hosts.
// Dispatch by element size: 4 → fread_be(std::span<int>), 8 → fread_be(std::span<double>),
// otherwise raw fread. Throws KadathError on open or short read.
class BeFileSource : public BinarySource
{
  public:
    explicit BeFileSource(const std::string& path);
    ~BeFileSource() override;

    BeFileSource(const BeFileSource&) = delete;
    BeFileSource& operator=(const BeFileSource&) = delete;

    void read_bytes(void* dest, std::size_t size, std::size_t count) override;

  private:
    FILE* file_;
    std::string path_;
};

} // namespace Kadath
