#pragma once
#include "binary_sink.hpp"

#include <cstdio>
#include <string>

namespace Kadath
{

// RAII over FILE* "wb" with big-endian byte swap on little-endian hosts.
// Dispatch by element size: 4 → fwrite_be(std::span<const int>),
// 8 → fwrite_be(std::span<const double>),
// otherwise raw fwrite. Throws KadathError on open failure or short write.
class BeFileSink : public BinarySink
{
  public:
    explicit BeFileSink(const std::string& path);
    ~BeFileSink() override;

    BeFileSink(const BeFileSink&) = delete;
    BeFileSink& operator=(const BeFileSink&) = delete;

    void write_bytes(const void* data, std::size_t size, std::size_t count) override;

  private:
    FILE* file_;
    std::string path_;
};

} // namespace Kadath
