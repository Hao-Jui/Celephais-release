#include "For_Kadath/IO/be_file_source.hpp"
#include "For_Kadath/Array/exceptions.hpp"
#include "For_Kadath/Utilities/utilities.hpp"

#include <sstream>
#include <span>

namespace Kadath
{

BeFileSource::BeFileSource(const std::string& path)
    : file_{std::fopen(path.c_str(), "rb")}, path_{path}
{
    if (file_ == nullptr) {
        std::ostringstream oss;
        oss << "BeFileSource: failed to open '" << path << "' for reading";
        KADATH_THROW(oss.str());
    }
}

BeFileSource::~BeFileSource()
{
    if (file_ != nullptr) {
        std::fclose(file_);
        file_ = nullptr;
    }
}

void BeFileSource::read_bytes(void* dest, std::size_t size, std::size_t count)
{
    int got = 0;
    if (size == sizeof(int)) {
        got = fread_be(std::span<int>{static_cast<int*>(dest), count}, file_);
    } else if (size == sizeof(double)) {
        got = fread_be(std::span<double>{static_cast<double*>(dest), count}, file_);
    } else {
        got = static_cast<int>(std::fread(dest, size, count, file_));
    }
    if (static_cast<std::size_t>(got) != count) {
        std::ostringstream oss;
        oss << "BeFileSource: short read from '" << path_ << "' (got " << got
            << " of " << count << " elements of " << size << " bytes)";
        KADATH_THROW(oss.str());
    }
}

} // namespace Kadath
