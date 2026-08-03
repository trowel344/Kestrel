#include "ggml.h"
#include "ggml-quants.h"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <fcntl.h>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
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
    if (argc != 8) {
        std::fprintf(
            stderr,
            "usage: %s MODEL OFFSET N_ROWS N_COLS CHUNK_ROWS SOURCE_TYPE THREADS\n",
            argv[0]);
        return 2;
    }
    const std::string path = argv[1];
    const uint64_t base = std::stoull(argv[2]);
    const int64_t n_rows = std::stoll(argv[3]);
    const int64_t n_cols = std::stoll(argv[4]);
    const int64_t chunk_rows = std::stoll(argv[5]);
    const std::string source_name = argv[6];
    const int n_threads = std::stoi(argv[7]);
    if (n_rows <= 0 || n_cols <= 0 || chunk_rows <= 0 || n_threads <= 0 ||
        n_cols % 128 != 0) {
        std::fprintf(stderr, "invalid dimensions, chunk size, or thread count\n");
        return 2;
    }

    const ggml_type source_type =
        source_name == "q4_0" ? GGML_TYPE_Q4_0 :
        source_name == "q2_k" ? GGML_TYPE_Q2_K :
        GGML_TYPE_COUNT;
    if (source_type == GGML_TYPE_COUNT) {
        std::fprintf(stderr, "SOURCE_TYPE must be q4_0 or q2_k\n");
        return 2;
    }
    const size_t source_row = ggml_row_size(source_type, n_cols);
    const size_t q1_row = ggml_row_size(GGML_TYPE_Q1_0, n_cols);
    if (q1_row >= source_row) {
        std::fprintf(stderr, "destination row is not smaller than source row\n");
        return 2;
    }
    const int64_t n_chunks = (n_rows + chunk_rows - 1) / chunk_rows;
    const int fd = open(path.c_str(), O_RDWR);
    if (fd < 0) {
        std::perror("open");
        return 1;
    }

    // Snapshot the complete source tensor before compacting it in place.
    // Q1_0 rows are smaller than Q2_K/Q4_0 rows, so an out-of-order worker
    // could otherwise overwrite source rows that another worker has not read.
    std::vector<uint8_t> source_all(static_cast<size_t>(n_rows) * source_row);
    try {
        read_exact(fd, source_all.data(), source_all.size(), static_cast<off_t>(base));
    } catch (const std::exception & error) {
        std::fprintf(stderr, "%s\n", error.what());
        close(fd);
        return 1;
    }

    std::atomic<int64_t> next_chunk{0};
    std::atomic<bool> failed{false};
    std::mutex error_mu;
    std::string error_message;
    auto worker = [&]() {
        while (!failed.load(std::memory_order_relaxed)) {
            const int64_t chunk = next_chunk.fetch_add(1);
            if (chunk >= n_chunks) {
                return;
            }
            const int64_t start = chunk * chunk_rows;
            const int64_t rows = std::min(chunk_rows, n_rows - start);
            try {
                std::vector<float> f32(static_cast<size_t>(rows) * n_cols);
                std::vector<uint8_t> q1(static_cast<size_t>(rows) * q1_row);
                const uint8_t * source =
                    source_all.data() + static_cast<size_t>(start) * source_row;
                if (source_type == GGML_TYPE_Q4_0) {
                    dequantize_row_q4_0(
                        reinterpret_cast<const block_q4_0 *>(source),
                        f32.data(),
                        rows * n_cols);
                } else {
                    dequantize_row_q2_K(
                        reinterpret_cast<const block_q2_K *>(source),
                        f32.data(),
                        rows * n_cols);
                }
                const size_t quantized = ggml_quantize_chunk(
                    GGML_TYPE_Q1_0,
                    f32.data(),
                    q1.data(),
                    0,
                    rows,
                    n_cols,
                    nullptr);
                if (quantized != q1.size()) {
                    throw std::runtime_error(
                        "unexpected Q1_0 size " + std::to_string(quantized) +
                        ", expected " + std::to_string(q1.size()));
                }
                write_exact(
                    fd,
                    q1.data(),
                    q1.size(),
                    static_cast<off_t>(base + static_cast<uint64_t>(start) * q1_row));
            } catch (const std::exception & error) {
                failed.store(true);
                std::lock_guard<std::mutex> lock(error_mu);
                if (error_message.empty()) {
                    error_message = error.what();
                }
                return;
            }
        }
    };

    std::vector<std::thread> workers;
    for (int i = 0; i < std::min<int64_t>(n_threads, n_chunks); i++) {
        workers.emplace_back(worker);
    }
    for (std::thread & thread : workers) {
        thread.join();
    }
    if (failed || fsync(fd) != 0) {
        std::fprintf(stderr, "%s\n", failed ? error_message.c_str() : "fsync failed");
        close(fd);
        return 1;
    }
    close(fd);
    std::printf(
        "{\"rows\":%lld,\"cols\":%lld,\"source\":\"%s\",\"source_bytes\":%llu,\"q1_bytes\":%llu}\n",
        static_cast<long long>(n_rows),
        static_cast<long long>(n_cols),
        source_name.c_str(),
        static_cast<unsigned long long>(n_rows * source_row),
        static_cast<unsigned long long>(n_rows * q1_row));
    return 0;
}
