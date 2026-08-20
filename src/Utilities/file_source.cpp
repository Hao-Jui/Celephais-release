#include "For_Kadath/IO/file_source.hpp"
#include "For_Kadath/Array/exceptions.hpp"

#include <sstream>

namespace Kadath
{

FileSource::FileSource(const std::string& path) : file_{std::fopen(path.c_str(), "rb")}, path_{path}
{
    if (file_ == nullptr) {
        std::ostringstream oss;
        oss << "FileSource: failed to open '" << path << "' for reading";
        KADATH_THROW(oss.str());
    }
}

FileSource::~FileSource()
{
    if (file_ != nullptr) {
        std::fclose(file_);
        file_ = nullptr;
    }
}

void FileSource::read_bytes(void* dest, std::size_t size, std::size_t count)
{
    const std::size_t got = std::fread(dest, size, count, file_);
    if (got != count) {
        std::ostringstream oss;
        oss << "FileSource: short read from '" << path_ << "' (got " << got << " of "
            << count << " elements of " << size << " bytes)";
        KADATH_THROW(oss.str());
    }
}

} // namespace Kadath
