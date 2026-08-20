#include "For_Kadath/IO/file_sink.hpp"
#include "For_Kadath/Array/exceptions.hpp"

#include <sstream>

namespace Kadath
{

FileSink::FileSink(const std::string& path) : file_{std::fopen(path.c_str(), "wb")}, path_{path}
{
    if (file_ == nullptr) {
        std::ostringstream oss;
        oss << "FileSink: failed to open '" << path << "' for writing";
        KADATH_THROW(oss.str());
    }
}

FileSink::~FileSink()
{
    if (file_ != nullptr) {
        std::fclose(file_);
        file_ = nullptr;
    }
}

void FileSink::write_bytes(const void* data, std::size_t size, std::size_t count)
{
    const std::size_t written = std::fwrite(data, size, count, file_);
    if (written != count) {
        std::ostringstream oss;
        oss << "FileSink: short write to '" << path_ << "' (wrote " << written << " of "
            << count << " elements of " << size << " bytes)";
        KADATH_THROW(oss.str());
    }
}

} // namespace Kadath
