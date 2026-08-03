#include "ggml.h"
#include "ggml-quants.h"

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fcntl.h>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <vector>

static void read_exact(int fd, void * data, size_t size, off_t offset) {
    auto * out = static_cast<uint8_t *>(data);
    size_t done = 0;
    while (done < size) {
        const ssize_t count = pread(fd, out + done, size - done, offset + done);
        if (count <= 0) {
            throw std::runtime_error("pread failed at " + std::to_string(offset + done));
        }
        done += static_cast<size_t>(count);
    }
}

static void write_exact(int fd, const void * data, size_t size, off_t offset) {
    const auto * in = static_cast<const uint8_t *>(data);
    size_t done = 0;
    while (done < size) {
        const ssize_t count = pwrite(fd, in + done, size - done, offset + done);
        if (count <= 0) {
            throw std::runtime_error("pwrite failed at " + std::to_string(offset + done));
        }
        done += static_cast<size_t>(count);
    }
}

int main(int argc, char ** argv) {
    if (argc != 6) {
        std::fprintf(
            stderr,
            "usage: %s MODEL OFFSET N_ROWS N_COLS CHUNK_ROWS\n",
            argv[0]);
        return 2;
    }

    const std::string path = argv[1];
    const uint64_t base = std::stoull(argv[2]);
    const int64_t n_rows = std::stoll(argv[3]);
    const int64_t n_cols = std::stoll(argv[4]);
    const int64_t chunk_rows = std::stoll(argv[5]);
    if (n_rows <= 0 || n_cols <= 0 || chunk_rows <= 0 || n_cols % 256 != 0) {
        std::fprintf(stderr, "invalid dimensions or chunk size\n");
        return 2;
    }

    const size_t q4_row = ggml_row_size(GGML_TYPE_Q4_0, n_cols);
    const size_t q2_row = ggml_row_size(GGML_TYPE_Q2_K, n_cols);
    if (q2_row >= q4_row) {
        std::fprintf(stderr, "destination row is not smaller than source row\n");
        return 2;
    }

    const int fd = open(path.c_str(), O_RDWR);
    if (fd < 0) {
        std::perror("open");
        return 1;
    }

    try {
        for (int64_t start = 0; start < n_rows; start += chunk_rows) {
            const int64_t rows = std::min(chunk_rows, n_rows - start);
            std::vector<uint8_t> q4(static_cast<size_t>(rows) * q4_row);
            std::vector<float> f32(static_cast<size_t>(rows) * n_cols);
            std::vector<uint8_t> q2(static_cast<size_t>(rows) * q2_row);

            read_exact(
                fd,
                q4.data(),
                q4.size(),
                static_cast<off_t>(base + static_cast<uint64_t>(start) * q4_row));
            dequantize_row_q4_0(
                reinterpret_cast<const block_q4_0 *>(q4.data()),
                f32.data(),
                rows * n_cols);
            const size_t quantized = ggml_quantize_chunk(
                GGML_TYPE_Q2_K,
                f32.data(),
                q2.data(),
                0,
                rows,
                n_cols,
                nullptr);
            if (quantized != q2.size()) {
                throw std::runtime_error(
                    "unexpected Q2_K size " + std::to_string(quantized) +
                    ", expected " + std::to_string(q2.size()));
            }
            write_exact(
                fd,
                q2.data(),
                q2.size(),
                static_cast<off_t>(base + static_cast<uint64_t>(start) * q2_row));
        }
        if (fsync(fd) != 0) {
            throw std::runtime_error("fsync failed");
        }
    } catch (const std::exception & error) {
        std::fprintf(stderr, "%s\n", error.what());
        close(fd);
        return 1;
    }

    close(fd);
    std::printf(
        "{\"rows\":%lld,\"cols\":%lld,\"q4_bytes\":%llu,\"q2_bytes\":%llu}\n",
        static_cast<long long>(n_rows),
        static_cast<long long>(n_cols),
        static_cast<unsigned long long>(n_rows * q4_row),
        static_cast<unsigned long long>(n_rows * q2_row));
    return 0;
}
