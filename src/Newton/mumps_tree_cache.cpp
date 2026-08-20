/*
    Copyright 2026 Kadath contributors

    This file is part of Kadath.

    Kadath is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    Kadath is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with Kadath.  If not, see <http://www.gnu.org/licenses/>.
*/

#include "Linear_algebra/mumps_tree_cache.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cerrno>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <sys/stat.h>
#include <system_error>
#include <unistd.h>
#include <vector>

namespace Kadath
{
    namespace
    {

        std::filesystem::path active_cache_path;

        // Schema 1 is a fixed-width big-endian binary cache:
        //
        //   byte  0: magic "KADMUMP\0"                         (8 bytes)
        //   byte  8: schema version                            (u32)
        //   byte 12: flags; bit 0 means matching was applied   (u32)
        //   byte 16: endian marker 0x01020304                  (u32)
        //   byte 20: header size, exactly 64                   (u32)
        //   byte 24: matrix dimension n                       (u64)
        //   byte 32: source-pattern nnz provenance            (u64)
        //   byte 40: UNS_PERM count, exactly n                (u64)
        //   byte 48: SYM_PERM count, exactly n                (u64)
        //   byte 56: total file bytes, exactly 104 + 8*n      (u64)
        //   byte 64: n one-based UNS_PERM entries             (i32 each)
        //             n one-based SYM_PERM entries             (i32 each)
        //             SHA-256(header + both arrays)            (32 bytes)
        //             footer "MUMPEND\0"                       (8 bytes)
        //
        // The stored nnz is deliberately provenance, not an eligibility key:
        // error-scaled dropping may slightly change a refresh pattern while the
        // two n-dimensional permutations remain reusable.
        constexpr std::array<std::uint8_t, 8> kMagic{{'K', 'A', 'D', 'M', 'U', 'M', 'P', '\0'}};
        constexpr std::array<std::uint8_t, 8> kFooter{{'M', 'U', 'M', 'P', 'E', 'N', 'D', '\0'}};
        constexpr std::uint32_t kSchemaVersion = 1;
        constexpr std::uint32_t kMatchingAppliedFlag = 0x1;
        constexpr std::uint32_t kAllowedFlags = kMatchingAppliedFlag;
        constexpr std::uint32_t kEndianMarker = 0x01020304;
        constexpr std::uint32_t kHeaderBytes = 64;
        constexpr std::uint64_t kHashBytes = 32;
        constexpr std::uint64_t kFooterBytes = 8;
        constexpr std::uint64_t kMaximumDimension = 10'000'000;
        constexpr std::uint64_t kMaximumFileBytes = kHeaderBytes + 8 * kMaximumDimension + kHashBytes + kFooterBytes;

        using ArchiveHash = std::array<std::uint8_t, 32>;

        [[noreturn]] void fail(const std::string& message)
        {
            throw std::runtime_error("mumpstree cache: " + message);
        }

        std::uint64_t checked_add(std::uint64_t left, std::uint64_t right, std::string_view what)
        {
            if (right > std::numeric_limits<std::uint64_t>::max() - left)
                fail(std::string(what) + " byte count overflows uint64");
            return left + right;
        }

        std::uint64_t checked_multiply(std::uint64_t left, std::uint64_t right, std::string_view what)
        {
            if (left != 0 && right > std::numeric_limits<std::uint64_t>::max() / left)
                fail(std::string(what) + " byte count overflows uint64");
            return left * right;
        }

        std::uint64_t expected_file_bytes(std::uint64_t dimension)
        {
            std::uint64_t bytes = kHeaderBytes;
            bytes = checked_add(bytes, checked_multiply(dimension, 8, "permutation payload"), "permutation payload");
            bytes = checked_add(bytes, kHashBytes, "archive hash");
            return checked_add(bytes, kFooterBytes, "footer");
        }

        std::array<std::uint8_t, 4> encode_u32(std::uint32_t value)
        {
            return {{static_cast<std::uint8_t>(value >> 24), static_cast<std::uint8_t>(value >> 16),
                     static_cast<std::uint8_t>(value >> 8), static_cast<std::uint8_t>(value)}};
        }

        std::array<std::uint8_t, 8> encode_u64(std::uint64_t value)
        {
            return {{static_cast<std::uint8_t>(value >> 56), static_cast<std::uint8_t>(value >> 48),
                     static_cast<std::uint8_t>(value >> 40), static_cast<std::uint8_t>(value >> 32),
                     static_cast<std::uint8_t>(value >> 24), static_cast<std::uint8_t>(value >> 16),
                     static_cast<std::uint8_t>(value >> 8), static_cast<std::uint8_t>(value)}};
        }

        std::uint32_t decode_u32(const std::uint8_t* bytes)
        {
            return (static_cast<std::uint32_t>(bytes[0]) << 24) | (static_cast<std::uint32_t>(bytes[1]) << 16) |
                   (static_cast<std::uint32_t>(bytes[2]) << 8) | static_cast<std::uint32_t>(bytes[3]);
        }

        std::uint64_t decode_u64(const std::uint8_t* bytes)
        {
            std::uint64_t value = 0;
            for (int index = 0; index < 8; ++index)
                value = (value << 8) | bytes[index];
            return value;
        }

        class Sha256
        {
          public:
            Sha256()
                : state_{{0x6a09e667U, 0xbb67ae85U, 0x3c6ef372U, 0xa54ff53aU, 0x510e527fU, 0x9b05688cU, 0x1f83d9abU,
                          0x5be0cd19U}}
            {
            }

            void update(const void* input, std::size_t size)
            {
                if (finished_)
                    fail("internal SHA-256 update after finish");
                if (size > std::numeric_limits<std::uint64_t>::max() - byte_count_)
                    fail("SHA-256 input length overflows uint64");
                byte_count_ += static_cast<std::uint64_t>(size);
                const auto* bytes = static_cast<const std::uint8_t*>(input);
                while (size != 0) {
                    const std::size_t take = std::min(size, block_.size() - block_size_);
                    std::memcpy(block_.data() + block_size_, bytes, take);
                    block_size_ += take;
                    bytes += take;
                    size -= take;
                    if (block_size_ == block_.size()) {
                        compress(block_.data());
                        block_size_ = 0;
                    }
                }
            }

            ArchiveHash finish()
            {
                if (finished_)
                    fail("internal SHA-256 finish called twice");
                if (byte_count_ > std::numeric_limits<std::uint64_t>::max() / 8)
                    fail("SHA-256 bit length overflows uint64");
                const std::uint64_t bit_count = byte_count_ * 8;
                block_[block_size_++] = 0x80;
                if (block_size_ > 56) {
                    std::fill(block_.begin() + static_cast<std::ptrdiff_t>(block_size_), block_.end(), 0);
                    compress(block_.data());
                    block_size_ = 0;
                }
                std::fill(block_.begin() + static_cast<std::ptrdiff_t>(block_size_), block_.begin() + 56, 0);
                const auto encoded_count = encode_u64(bit_count);
                std::copy(encoded_count.begin(), encoded_count.end(), block_.begin() + 56);
                compress(block_.data());
                finished_ = true;

                ArchiveHash digest{};
                for (std::size_t word = 0; word < state_.size(); ++word) {
                    const auto encoded = encode_u32(state_[word]);
                    std::copy(encoded.begin(), encoded.end(), digest.begin() + static_cast<std::ptrdiff_t>(4 * word));
                }
                return digest;
            }

          private:
            static constexpr std::array<std::uint32_t, 64> constants_{{
                0x428a2f98U, 0x71374491U, 0xb5c0fbcfU, 0xe9b5dba5U, 0x3956c25bU, 0x59f111f1U, 0x923f82a4U, 0xab1c5ed5U,
                0xd807aa98U, 0x12835b01U, 0x243185beU, 0x550c7dc3U, 0x72be5d74U, 0x80deb1feU, 0x9bdc06a7U, 0xc19bf174U,
                0xe49b69c1U, 0xefbe4786U, 0x0fc19dc6U, 0x240ca1ccU, 0x2de92c6fU, 0x4a7484aaU, 0x5cb0a9dcU, 0x76f988daU,
                0x983e5152U, 0xa831c66dU, 0xb00327c8U, 0xbf597fc7U, 0xc6e00bf3U, 0xd5a79147U, 0x06ca6351U, 0x14292967U,
                0x27b70a85U, 0x2e1b2138U, 0x4d2c6dfcU, 0x53380d13U, 0x650a7354U, 0x766a0abbU, 0x81c2c92eU, 0x92722c85U,
                0xa2bfe8a1U, 0xa81a664bU, 0xc24b8b70U, 0xc76c51a3U, 0xd192e819U, 0xd6990624U, 0xf40e3585U, 0x106aa070U,
                0x19a4c116U, 0x1e376c08U, 0x2748774cU, 0x34b0bcb5U, 0x391c0cb3U, 0x4ed8aa4aU, 0x5b9cca4fU, 0x682e6ff3U,
                0x748f82eeU, 0x78a5636fU, 0x84c87814U, 0x8cc70208U, 0x90befffaU, 0xa4506cebU, 0xbef9a3f7U, 0xc67178f2U,
            }};

            static std::uint32_t choose(std::uint32_t x, std::uint32_t y, std::uint32_t z)
            {
                return (x & y) ^ (~x & z);
            }

            static std::uint32_t majority(std::uint32_t x, std::uint32_t y, std::uint32_t z)
            {
                return (x & y) ^ (x & z) ^ (y & z);
            }

            void compress(const std::uint8_t* block)
            {
                std::array<std::uint32_t, 64> words{};
                for (std::size_t index = 0; index < 16; ++index)
                    words[index] = decode_u32(block + 4 * index);
                for (std::size_t index = 16; index < words.size(); ++index) {
                    const std::uint32_t s0 =
                        std::rotr(words[index - 15], 7) ^ std::rotr(words[index - 15], 18) ^ (words[index - 15] >> 3);
                    const std::uint32_t s1 =
                        std::rotr(words[index - 2], 17) ^ std::rotr(words[index - 2], 19) ^ (words[index - 2] >> 10);
                    words[index] = words[index - 16] + s0 + words[index - 7] + s1;
                }

                std::uint32_t a = state_[0];
                std::uint32_t b = state_[1];
                std::uint32_t c = state_[2];
                std::uint32_t d = state_[3];
                std::uint32_t e = state_[4];
                std::uint32_t f = state_[5];
                std::uint32_t g = state_[6];
                std::uint32_t h = state_[7];
                for (std::size_t index = 0; index < words.size(); ++index) {
                    const std::uint32_t sum1 = std::rotr(e, 6) ^ std::rotr(e, 11) ^ std::rotr(e, 25);
                    const std::uint32_t temporary1 = h + sum1 + choose(e, f, g) + constants_[index] + words[index];
                    const std::uint32_t sum0 = std::rotr(a, 2) ^ std::rotr(a, 13) ^ std::rotr(a, 22);
                    const std::uint32_t temporary2 = sum0 + majority(a, b, c);
                    h = g;
                    g = f;
                    f = e;
                    e = d + temporary1;
                    d = c;
                    c = b;
                    b = a;
                    a = temporary1 + temporary2;
                }
                state_[0] += a;
                state_[1] += b;
                state_[2] += c;
                state_[3] += d;
                state_[4] += e;
                state_[5] += f;
                state_[6] += g;
                state_[7] += h;
            }

            std::array<std::uint32_t, 8> state_{};
            std::array<std::uint8_t, 64> block_{};
            std::size_t block_size_ = 0;
            std::uint64_t byte_count_ = 0;
            bool finished_ = false;
        };

        void validate_extension(const std::filesystem::path& path, std::string_view operation)
        {
            if (path.empty() || path.extension() != ".mumpstree")
                fail(std::string(operation) + " path must have a .mumpstree extension");
        }

        void validate_bijection(std::span<const int> permutation, int dimension, std::string_view name)
        {
            if (permutation.size() != static_cast<std::size_t>(dimension))
                fail(std::string(name) + " length does not match dimension");
            std::vector<bool> seen(static_cast<std::size_t>(dimension), false);
            for (int value : permutation) {
                if (value < 1 || value > dimension)
                    fail(std::string(name) + " contains an out-of-range position");
                const std::size_t index = static_cast<std::size_t>(value - 1);
                if (seen[index])
                    fail(std::string(name) + " is not a bijection");
                seen[index] = true;
            }
        }

        void validate_view(const MumpsTreeCacheView& cache)
        {
            static_assert(sizeof(int) == sizeof(std::int32_t));
            if (cache.dimension < 1 || static_cast<std::uint64_t>(cache.dimension) > kMaximumDimension)
                fail("dimension is outside the supported range [1,10000000]");
            if (cache.pattern_nnz < 1)
                fail("pattern nnz provenance must be positive");
            validate_bijection(cache.column_permutation_1based, cache.dimension, "UNS_PERM");
            validate_bijection(cache.symmetric_permutation_1based, cache.dimension, "SYM_PERM");
            if (!cache.matching_applied) {
                for (int index = 0; index < cache.dimension; ++index) {
                    if (cache.column_permutation_1based[static_cast<std::size_t>(index)] != index + 1)
                        fail("UNS_PERM must be identity when matching was not applied");
                }
            }
            (void)expected_file_bytes(static_cast<std::uint64_t>(cache.dimension));
        }

        MumpsTreeCacheView owned_view(const MumpsTreeCache& cache)
        {
            return {cache.dimension, cache.pattern_nnz, cache.matching_applied, cache.column_permutation_1based,
                    cache.symmetric_permutation_1based};
        }

        class AtomicFile
        {
          public:
            explicit AtomicFile(const std::filesystem::path& target) : target_(target)
            {
                const std::filesystem::path parent =
                    target.parent_path().empty() ? std::filesystem::path{"."} : target.parent_path();
                static std::atomic<unsigned long long> sequence{0};
                for (int attempt = 0; attempt < 100; ++attempt) {
                    temporary_ = parent / ("." + target.filename().string() + ".tmp." +
                                           std::to_string(static_cast<long long>(::getpid())) + "." +
                                           std::to_string(sequence.fetch_add(1, std::memory_order_relaxed)));
                    int flags = O_WRONLY | O_CREAT | O_EXCL;
#ifdef O_CLOEXEC
                    flags |= O_CLOEXEC;
#endif
                    fd_ = ::open(temporary_.c_str(), flags, 0600);
                    if (fd_ >= 0)
                        return;
                    if (errno != EEXIST)
                        fail("cannot create temporary file '" + temporary_.string() +
                             "': " + std::system_category().message(errno));
                }
                fail("could not allocate an exclusive temporary filename");
            }

            AtomicFile(const AtomicFile&) = delete;
            AtomicFile& operator=(const AtomicFile&) = delete;

            ~AtomicFile()
            {
                if (fd_ >= 0)
                    ::close(fd_);
                if (!committed_) {
                    std::error_code ignored;
                    std::filesystem::remove(temporary_, ignored);
                }
            }

            void write_prefix(const void* data, std::size_t size)
            {
                write_raw(data, size);
                archive_hash_.update(data, size);
            }

            void write_tail(const void* data, std::size_t size) { write_raw(data, size); }

            void write_prefix_u32(std::uint32_t value)
            {
                const auto bytes = encode_u32(value);
                write_prefix(bytes.data(), bytes.size());
            }

            void write_prefix_i32(std::int32_t value) { write_prefix_u32(std::bit_cast<std::uint32_t>(value)); }

            void write_prefix_u64(std::uint64_t value)
            {
                const auto bytes = encode_u64(value);
                write_prefix(bytes.data(), bytes.size());
            }

            ArchiveHash finish_archive_hash() { return archive_hash_.finish(); }

            void commit()
            {
                flush_buffer();
                if (::fsync(fd_) != 0)
                    fail("fsync failed for temporary file: " + std::system_category().message(errno));
                if (::close(fd_) != 0) {
                    fd_ = -1;
                    fail("close failed for temporary file: " + std::system_category().message(errno));
                }
                fd_ = -1;
                if (::link(temporary_.c_str(), target_.c_str()) != 0)
                    fail("atomic no-replace publish failed for '" + target_.string() +
                         "': " + std::system_category().message(errno));
                if (::unlink(temporary_.c_str()) != 0)
                    fail_after_publish("cannot unlink published temporary file: " +
                                       std::system_category().message(errno));

                const std::filesystem::path parent =
                    target_.parent_path().empty() ? std::filesystem::path{"."} : target_.parent_path();
                int directory_flags = O_RDONLY;
#ifdef O_DIRECTORY
                directory_flags |= O_DIRECTORY;
#endif
#ifdef O_CLOEXEC
                directory_flags |= O_CLOEXEC;
#endif
                const int directory_fd = ::open(parent.c_str(), directory_flags);
                if (directory_fd < 0)
                    fail_after_publish("cannot open output directory for fsync: " +
                                       std::system_category().message(errno));
                if (::fsync(directory_fd) != 0) {
                    const std::string message =
                        "output directory fsync failed: " + std::system_category().message(errno);
                    (void)::close(directory_fd);
                    fail_after_publish(message);
                }
                if (::close(directory_fd) != 0)
                    fail_after_publish("output directory close failed: " + std::system_category().message(errno));
                committed_ = true;
            }

          private:
            [[noreturn]] void fail_after_publish(const std::string& message)
            {
                if (::unlink(target_.c_str()) != 0)
                    fail(message + "; rollback unlink also failed: " + std::system_category().message(errno));
                fail(message + "; published output was rolled back");
            }

            void flush_buffer()
            {
                std::size_t offset = 0;
                while (offset != output_size_) {
                    const ssize_t written = ::write(fd_, output_buffer_.data() + offset, output_size_ - offset);
                    if (written < 0) {
                        if (errno == EINTR)
                            continue;
                        fail("write failed for temporary file: " + std::system_category().message(errno));
                    }
                    if (written == 0)
                        fail("zero-length write to temporary file");
                    offset += static_cast<std::size_t>(written);
                }
                output_size_ = 0;
            }

            void write_raw(const void* data, std::size_t size)
            {
                const auto* bytes = static_cast<const std::uint8_t*>(data);
                while (size != 0) {
                    if (output_size_ == output_buffer_.size())
                        flush_buffer();
                    const std::size_t count = std::min(size, output_buffer_.size() - output_size_);
                    std::memcpy(output_buffer_.data() + output_size_, bytes, count);
                    output_size_ += count;
                    bytes += count;
                    size -= count;
                }
            }

            std::filesystem::path target_;
            std::filesystem::path temporary_;
            int fd_ = -1;
            bool committed_ = false;
            std::array<std::uint8_t, 64 * 1024> output_buffer_{};
            std::size_t output_size_ = 0;
            Sha256 archive_hash_;
        };

        class FileReader
        {
          public:
            explicit FileReader(const std::filesystem::path& path) : fd_(open_cache(path))
            {
                if (fd_ < 0)
                    fail("cannot open '" + path.string() + "': " + std::system_category().message(errno));
                struct stat status{};
                if (::fstat(fd_, &status) != 0)
                    fail_open("cannot stat '" + path.string() + "': " + std::system_category().message(errno));
                if (!S_ISREG(status.st_mode))
                    fail_open("cache path is not a regular file");
                if (status.st_size < 0)
                    fail_open("cache file has a negative size");
                size_ = static_cast<std::uint64_t>(status.st_size);
            }

            FileReader(const FileReader&) = delete;
            FileReader& operator=(const FileReader&) = delete;

            ~FileReader()
            {
                if (fd_ >= 0)
                    ::close(fd_);
            }

            std::uint64_t size() const { return size_; }
            std::uint64_t position() const { return position_; }

            void begin_archive_hash()
            {
                if (hashing_)
                    fail("internal reader hash already active");
                hashing_ = true;
            }

            ArchiveHash finish_archive_hash()
            {
                if (!hashing_)
                    fail("internal reader hash is not active");
                hashing_ = false;
                return archive_hash_.finish();
            }

            void seek(std::uint64_t offset)
            {
                if (offset > static_cast<std::uint64_t>(std::numeric_limits<off_t>::max()))
                    fail("file offset does not fit off_t");
                if (::lseek(fd_, static_cast<off_t>(offset), SEEK_SET) < 0)
                    fail("seek failed: " + std::system_category().message(errno));
                position_ = offset;
            }

            void read_exact(void* destination, std::size_t size)
            {
                auto* bytes = static_cast<std::uint8_t*>(destination);
                while (size != 0) {
                    const ssize_t count = ::read(fd_, bytes, size);
                    if (count < 0) {
                        if (errno == EINTR)
                            continue;
                        fail("read failed: " + std::system_category().message(errno));
                    }
                    if (count == 0)
                        fail("truncated file");
                    if (hashing_)
                        archive_hash_.update(bytes, static_cast<std::size_t>(count));
                    bytes += count;
                    size -= static_cast<std::size_t>(count);
                    position_ += static_cast<std::uint64_t>(count);
                }
            }

            std::uint32_t read_u32()
            {
                std::array<std::uint8_t, 4> bytes{};
                read_exact(bytes.data(), bytes.size());
                return decode_u32(bytes.data());
            }

            std::int32_t read_i32() { return std::bit_cast<std::int32_t>(read_u32()); }

            std::uint64_t read_u64()
            {
                std::array<std::uint8_t, 8> bytes{};
                read_exact(bytes.data(), bytes.size());
                return decode_u64(bytes.data());
            }

          private:
            [[noreturn]] void fail_open(const std::string& message)
            {
                (void)::close(fd_);
                fd_ = -1;
                fail(message);
            }

            static int open_cache(const std::filesystem::path& path)
            {
                int flags = O_RDONLY;
#ifdef O_NOFOLLOW
                flags |= O_NOFOLLOW;
#endif
#ifdef O_CLOEXEC
                flags |= O_CLOEXEC;
#endif
                return ::open(path.c_str(), flags);
            }

            int fd_ = -1;
            std::uint64_t size_ = 0;
            std::uint64_t position_ = 0;
            Sha256 archive_hash_;
            bool hashing_ = false;
        };

        struct Header {
            std::uint32_t flags = 0;
            std::uint64_t dimension = 0;
            std::uint64_t pattern_nnz = 0;
            std::uint64_t column_permutation_count = 0;
            std::uint64_t symmetric_permutation_count = 0;
            std::uint64_t total_bytes = 0;

            bool operator==(const Header&) const = default;
        };

        Header read_header(FileReader& file)
        {
            std::array<std::uint8_t, 8> magic{};
            file.read_exact(magic.data(), magic.size());
            if (magic != kMagic)
                fail("bad magic");
            if (file.read_u32() != kSchemaVersion)
                fail("unsupported schema version");
            Header header;
            header.flags = file.read_u32();
            if ((header.flags & ~kAllowedFlags) != 0)
                fail("unsupported flags");
            if (file.read_u32() != kEndianMarker)
                fail("bad endian marker");
            if (file.read_u32() != kHeaderBytes)
                fail("unexpected header size");
            header.dimension = file.read_u64();
            header.pattern_nnz = file.read_u64();
            header.column_permutation_count = file.read_u64();
            header.symmetric_permutation_count = file.read_u64();
            header.total_bytes = file.read_u64();
            if (file.position() != kHeaderBytes)
                fail("internal header size mismatch");
            return header;
        }

        void validate_header(const Header& header, std::uint64_t actual_bytes, int expected_dimension)
        {
            if (header.dimension == 0 || header.dimension > kMaximumDimension ||
                header.dimension > static_cast<std::uint64_t>(INT_MAX))
                fail("stored dimension is outside the supported range [1,10000000]");
            if (header.dimension != static_cast<std::uint64_t>(expected_dimension))
                fail("stored dimension does not match expected dimension");
            if (header.pattern_nnz == 0 || header.pattern_nnz > static_cast<std::uint64_t>(LLONG_MAX))
                fail("stored pattern nnz provenance is outside the supported range");
            if (header.column_permutation_count != header.dimension ||
                header.symmetric_permutation_count != header.dimension)
                fail("stored permutation counts do not match dimension");
            const std::uint64_t expected = expected_file_bytes(header.dimension);
            if (expected > kMaximumFileBytes)
                fail("declared cache size exceeds the supported maximum");
            if (header.total_bytes != expected)
                fail("declared file size does not match section counts");
            if (actual_bytes < expected)
                fail("truncated file");
            if (actual_bytes > expected)
                fail("trailing bytes after cache footer");
        }

        void write_header(AtomicFile& file, const MumpsTreeCacheView& cache, std::uint64_t total_bytes)
        {
            const std::uint32_t flags = cache.matching_applied ? kMatchingAppliedFlag : 0;
            file.write_prefix(kMagic.data(), kMagic.size());
            file.write_prefix_u32(kSchemaVersion);
            file.write_prefix_u32(flags);
            file.write_prefix_u32(kEndianMarker);
            file.write_prefix_u32(kHeaderBytes);
            file.write_prefix_u64(static_cast<std::uint64_t>(cache.dimension));
            file.write_prefix_u64(static_cast<std::uint64_t>(cache.pattern_nnz));
            file.write_prefix_u64(cache.column_permutation_1based.size());
            file.write_prefix_u64(cache.symmetric_permutation_1based.size());
            file.write_prefix_u64(total_bytes);
        }

        void sync_parent_after_delete(const std::filesystem::path& path)
        {
            const std::filesystem::path parent =
                path.parent_path().empty() ? std::filesystem::path{"."} : path.parent_path();
            int directory_flags = O_RDONLY;
#ifdef O_DIRECTORY
            directory_flags |= O_DIRECTORY;
#endif
#ifdef O_CLOEXEC
            directory_flags |= O_CLOEXEC;
#endif
            const int directory_fd = ::open(parent.c_str(), directory_flags);
            if (directory_fd < 0)
                fail("cache was deleted but its directory cannot be opened for fsync: " +
                     std::system_category().message(errno));
            if (::fsync(directory_fd) != 0) {
                const std::string message =
                    "cache was deleted but directory fsync failed: " + std::system_category().message(errno);
                (void)::close(directory_fd);
                fail(message);
            }
            if (::close(directory_fd) != 0)
                fail("cache was deleted but directory close failed: " + std::system_category().message(errno));
        }

    } // namespace

    void write_mumps_tree_cache(const std::filesystem::path& path, const MumpsTreeCacheView& cache)
    {
        validate_extension(path, "output");
        std::error_code status_error;
        const std::filesystem::file_status status = std::filesystem::symlink_status(path, status_error);
        if (status_error && status_error != std::errc::no_such_file_or_directory)
            fail("cannot inspect output path: " + status_error.message());
        if (!status_error && status.type() != std::filesystem::file_type::not_found)
            fail("refusing to overwrite existing output path");

        validate_view(cache);
        const std::uint64_t total_bytes = expected_file_bytes(static_cast<std::uint64_t>(cache.dimension));
        AtomicFile file(path);
        write_header(file, cache, total_bytes);
        for (int value : cache.column_permutation_1based)
            file.write_prefix_i32(value);
        for (int value : cache.symmetric_permutation_1based)
            file.write_prefix_i32(value);
        const ArchiveHash archive_hash = file.finish_archive_hash();
        file.write_tail(archive_hash.data(), archive_hash.size());
        file.write_tail(kFooter.data(), kFooter.size());
        file.commit();
    }

    MumpsTreeCache read_mumps_tree_cache(const std::filesystem::path& path, int expected_dimension)
    {
        validate_extension(path, "input");
        if (expected_dimension < 1 || static_cast<std::uint64_t>(expected_dimension) > kMaximumDimension)
            fail("expected dimension is outside the supported range [1,10000000]");

        std::error_code status_error;
        const std::filesystem::file_status status = std::filesystem::symlink_status(path, status_error);
        if (status_error)
            fail("cannot inspect input path: " + status_error.message());
        if (status.type() == std::filesystem::file_type::symlink)
            fail("refusing to read a symbolic-link cache path");
        if (status.type() != std::filesystem::file_type::regular)
            fail("cache path is not a regular file");

        FileReader file(path);
        if (file.size() > kMaximumFileBytes)
            fail("cache file exceeds the supported maximum size");
        Header header = read_header(file);
        validate_header(header, file.size(), expected_dimension);

        file.seek(0);
        file.begin_archive_hash();
        const Header decoded_header = read_header(file);
        if (decoded_header != header)
            fail("cache header changed between verification and decode");

        MumpsTreeCache cache;
        cache.dimension = static_cast<int>(header.dimension);
        cache.pattern_nnz = static_cast<long long>(header.pattern_nnz);
        cache.matching_applied = (header.flags & kMatchingAppliedFlag) != 0;
        cache.column_permutation_1based.resize(static_cast<std::size_t>(cache.dimension));
        cache.symmetric_permutation_1based.resize(static_cast<std::size_t>(cache.dimension));
        for (int& value : cache.column_permutation_1based)
            value = file.read_i32();
        for (int& value : cache.symmetric_permutation_1based)
            value = file.read_i32();
        const ArchiveHash decoded_archive_hash = file.finish_archive_hash();
        ArchiveHash stored_archive_hash{};
        file.read_exact(stored_archive_hash.data(), stored_archive_hash.size());
        std::array<std::uint8_t, 8> footer{};
        file.read_exact(footer.data(), footer.size());
        if (footer != kFooter || file.position() != file.size())
            fail("cache footer or final length is invalid");
        if (stored_archive_hash != decoded_archive_hash)
            fail("archive hash mismatch");

        validate_view(owned_view(cache));
        return cache;
    }

    bool remove_mumps_tree_cache(const std::filesystem::path& path)
    {
        validate_extension(path, "delete");
        std::error_code status_error;
        const std::filesystem::file_status status = std::filesystem::symlink_status(path, status_error);
        if (status_error == std::errc::no_such_file_or_directory)
            return false;
        if (status_error)
            fail("cannot inspect delete path: " + status_error.message());
        if (status.type() == std::filesystem::file_type::not_found)
            return false;
        if (status.type() == std::filesystem::file_type::symlink)
            fail("refusing to delete a symbolic-link cache path");
        if (status.type() != std::filesystem::file_type::regular)
            fail("refusing to delete a non-regular cache path");
        if (::unlink(path.c_str()) != 0)
            fail("cannot delete cache path: " + std::system_category().message(errno));
        sync_parent_after_delete(path);
        return true;
    }

    std::vector<int> compose_mumps_tree_column_indices_1based(
        std::span<const int> column_indices_1based,
        std::span<const int> column_permutation_1based,
        int dimension)
    {
        if (dimension < 1 ||
            column_permutation_1based.size() != static_cast<std::size_t>(dimension))
            fail("composition permutation must contain exactly n entries");

        std::vector<int> inverse(static_cast<std::size_t>(dimension) + 1, 0);
        for (int position = 1; position <= dimension; ++position) {
            const int original_column =
                column_permutation_1based[static_cast<std::size_t>(position - 1)];
            if (original_column < 1 || original_column > dimension ||
                inverse[static_cast<std::size_t>(original_column)] != 0)
                fail("composition permutation is not a bijection over [1,n]");
            inverse[static_cast<std::size_t>(original_column)] = position;
        }

        std::vector<int> composed;
        composed.reserve(column_indices_1based.size());
        for (const int original_column : column_indices_1based) {
            if (original_column < 1 || original_column > dimension)
                fail("composition column index is outside [1,n]");
            composed.push_back(inverse[static_cast<std::size_t>(original_column)]);
        }
        return composed;
    }

    void set_active_mumps_tree_cache_path(const std::filesystem::path& path)
    {
        validate_extension(path, "active cache");
        active_cache_path = path;
    }

    std::filesystem::path active_mumps_tree_cache_path()
    {
        return active_cache_path;
    }

    void clear_active_mumps_tree_cache_path() noexcept
    {
        active_cache_path.clear();
    }

} // namespace Kadath
