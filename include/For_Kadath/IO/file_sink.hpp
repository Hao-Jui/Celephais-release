#pragma once
#include "binary_sink.hpp"

#include <cstdio>
#include <string>

namespace Kadath
{

// RAII wrapper around FILE* opened in "wb" mode. Throws KadathError on open
// failure or short write.
class FileSink : public BinarySink
{
  public:
    explicit FileSink(const std::string& path);
    ~FileSink() override;

    FileSink(const FileSink&) = delete;
    FileSink& operator=(const FileSink&) = delete;

    void write_bytes(const void* data, std::size_t size, std::size_t count) override;

  private:
    FILE* file_;
    std::string path_;
};

} // namespace Kadath
