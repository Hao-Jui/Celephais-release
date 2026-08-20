#include "For_Kadath/IO/be_file_sink.hpp"
#include "For_Kadath/Array/exceptions.hpp"
#include "For_Kadath/Utilities/utilities.hpp"

#include <sstream>
#include <span>

namespace Kadath
{

BeFileSink::BeFileSink(const std::string& path)
    : file_{std::fopen(path.c_str(), "wb")}, path_{path}
{
    if (file_ == nullptr) {
        std::ostringstream oss;
        oss << "BeFileSink: failed to open '" << path << "' for writing";
        KADATH_THROW(oss.str());
    }
}

BeFileSink::~BeFileSink()
{
    if (file_ != nullptr) {
        std::fclose(file_);
        file_ = nullptr;
    }
}

void BeFileSink::write_bytes(const void* data, std::size_t size, std::size_t count)
{
    int written = 0;
    if (size == sizeof(int)) {
        written = fwrite_be(std::span<const int>{static_cast<const int*>(data), count}, file_);
    } else if (size == sizeof(double)) {
        written = fwrite_be(std::span<const double>{static_cast<const double*>(data), count}, file_);
    } else {
        written = static_cast<int>(std::fwrite(data, size, count, file_));
    }
    if (static_cast<std::size_t>(written) != count) {
        std::ostringstream oss;
        oss << "BeFileSink: short write to '" << path_ << "' (wrote " << written
            << " of " << count << " elements of " << size << " bytes)";
        KADATH_THROW(oss.str());
    }
}

} // namespace Kadath
