#pragma once
#include "binary_source.hpp"

#include <cstdio>
#include <string>

namespace Kadath
{

// RAII wrapper around FILE* opened in "rb" mode. Throws KadathError on open
// failure or short read.
class FileSource : public BinarySource
{
  public:
    explicit FileSource(const std::string& path);
    ~FileSource() override;

    FileSource(const FileSource&) = delete;
    FileSource& operator=(const FileSource&) = delete;

    void read_bytes(void* dest, std::size_t size, std::size_t count) override;

  private:
    FILE* file_;
    std::string path_;
};

} // namespace Kadath
