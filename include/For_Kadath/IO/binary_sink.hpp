#pragma once
#include <cstddef>
#include <cstring>

namespace Kadath
{

// Type-agnostic byte sink. Concrete adapters: FileSink (RAII over FILE*),
// MemorySink (std::vector backing for tests). Caller-side templated
// write<T> + write_array<T> dispatch through the byte-level virtual.
class BinarySink
{
  public:
    virtual ~BinarySink() = default;

    BinarySink() = default;
    BinarySink(const BinarySink&) = delete;
    BinarySink& operator=(const BinarySink&) = delete;

    virtual void write_bytes(const void* data, std::size_t size, std::size_t count) = 0;

    template <class T>
    void write(const T& value)
    {
        write_bytes(&value, sizeof(T), 1);
    }

    template <class T>
    void write_array(const T* data, std::size_t count)
    {
        write_bytes(data, sizeof(T), count);
    }
};

} // namespace Kadath
