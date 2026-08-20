#pragma once
#include <cstddef>

namespace Kadath
{

// Type-agnostic byte source. Concrete adapters: FileSource (RAII over
// FILE*), MemorySource (cursor over std::vector for tests).
class BinarySource
{
  public:
    virtual ~BinarySource() = default;

    BinarySource() = default;
    BinarySource(const BinarySource&) = delete;
    BinarySource& operator=(const BinarySource&) = delete;

    virtual void read_bytes(void* dest, std::size_t size, std::size_t count) = 0;

    template <class T>
    T read()
    {
        T value;
        read_bytes(&value, sizeof(T), 1);
        return value;
    }

    template <class T>
    void read_array(T* dest, std::size_t count)
    {
        read_bytes(dest, sizeof(T), count);
    }
};

} // namespace Kadath
