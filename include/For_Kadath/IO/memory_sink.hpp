#pragma once
#include "binary_sink.hpp"
#include "binary_source.hpp"
#include "For_Kadath/Array/exceptions.hpp"

#include <cstring>
#include <span>
#include <sstream>
#include <vector>

namespace Kadath
{

// std::vector-backed sink for unit tests. Captures writes into an in-memory
// buffer; pair with MemorySource to exercise round-trip serialization
// without touching disk.
class MemorySink : public BinarySink
{
  public:
    void write_bytes(const void* data, std::size_t size, std::size_t count) override
    {
        const std::size_t total = size * count;
        const std::size_t old = buffer_.size();
        buffer_.resize(old + total);
        std::memcpy(buffer_.data() + old, data, total);
    }

    const std::vector<unsigned char>& buffer() const { return buffer_; }
    std::span<const unsigned char> bytes() const { return {buffer_.data(), buffer_.size()}; }

  private:
    std::vector<unsigned char> buffer_;
};

// Cursor-based reader over a byte vector. Pair with MemorySink for tests.
class MemorySource : public BinarySource
{
  public:
    explicit MemorySource(std::vector<unsigned char> buffer)
        : buffer_{std::move(buffer)}, cursor_{0}
    {
    }

    void read_bytes(void* dest, std::size_t size, std::size_t count) override
    {
        const std::size_t total = size * count;
        if (cursor_ + total > buffer_.size()) {
            std::ostringstream oss;
            oss << "MemorySource: read past end (cursor=" << cursor_ << " want=" << total
                << " size=" << buffer_.size() << ")";
            KADATH_THROW(oss.str());
        }
        std::memcpy(dest, buffer_.data() + cursor_, total);
        cursor_ += total;
    }

    std::size_t cursor() const { return cursor_; }

  private:
    std::vector<unsigned char> buffer_;
    std::size_t cursor_;
};

} // namespace Kadath
