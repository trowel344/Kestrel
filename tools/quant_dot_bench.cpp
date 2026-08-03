#include "ggml.h"
#include "ggml-quants.h"
#include "quants.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <random>
#include <thread>
#include <vector>

template<typename Fn>
static double run_rows(
        Fn fn,
        int n,
        int rows,
        int threads,
        int repeats,
        const uint8_t * weights,
        size_t row_bytes,
        const uint8_t * activation,
        std::vector<float> & output) {
    const auto started = std::chrono::steady_clock::now();
    for (int repeat = 0; repeat < repeats; ++repeat) {
        std::vector<std::thread> workers;
        for (int worker = 0; worker < threads; ++worker) {
            workers.emplace_back([&, worker]() {
                for (int row = worker; row < rows; row += threads) {
                    fn(
                        n,
                        &output[row],
                        0,
                        weights + static_cast<size_t>(row) * row_bytes,
                        0,
                        activation,
                        0,
                        1);
                }
            });
        }
        for (auto & worker : workers) {
            worker.join();
        }
    }
    return std::chrono::duration<double>(
        std::chrono::steady_clock::now() - started).count();
}

int main(int argc, char ** argv) {
    const int n = argc > 1 ? std::stoi(argv[1]) : 3072;
    const int rows = argc > 2 ? std::stoi(argv[2]) : 16384;
    const int threads = argc > 3 ? std::stoi(argv[3]) : 14;
    const int repeats = argc > 4 ? std::stoi(argv[4]) : 20;
    if (n <= 0 || n % 256 != 0 || rows <= 0 || threads <= 0 || repeats <= 0) {
        std::fprintf(stderr, "usage: %s [n_multiple_256] [rows] [threads] [repeats]\n", argv[0]);
        return 2;
    }

    std::mt19937 rng(42);
    std::normal_distribution<float> normal(0.0f, 0.25f);
    std::vector<float> source(static_cast<size_t>(rows) * n);
    std::vector<float> input(n);
    for (float & value : source) value = normal(rng);
    for (float & value : input) value = normal(rng);

    const size_t q1_row = ggml_row_size(GGML_TYPE_Q1_0, n);
    const size_t q2_row = ggml_row_size(GGML_TYPE_Q2_K, n);
    const size_t q8_0_row = ggml_row_size(GGML_TYPE_Q8_0, n);
    const size_t q8_k_row = ggml_row_size(GGML_TYPE_Q8_K, n);
    std::vector<uint8_t> q1(static_cast<size_t>(rows) * q1_row);
    std::vector<uint8_t> q2(static_cast<size_t>(rows) * q2_row);
    std::vector<uint8_t> q8_0(q8_0_row);
    std::vector<uint8_t> q8_k(q8_k_row);
    for (int row = 0; row < rows; ++row) {
        quantize_row_q1_0(source.data() + static_cast<size_t>(row) * n, q1.data() + static_cast<size_t>(row) * q1_row, n);
        quantize_row_q2_K(source.data() + static_cast<size_t>(row) * n, q2.data() + static_cast<size_t>(row) * q2_row, n);
    }
    quantize_row_q8_0(input.data(), q8_0.data(), n);
    quantize_row_q8_K(input.data(), q8_k.data(), n);

    std::vector<float> output(rows);
    const double q1_seconds = run_rows(
        ggml_vec_dot_q1_0_q8_0, n, rows, threads, repeats,
        q1.data(), q1_row, q8_0.data(), output);
    const double q1_checksum = std::accumulate(output.begin(), output.end(), 0.0);
    const double q2_seconds = run_rows(
        ggml_vec_dot_q2_K_q8_K, n, rows, threads, repeats,
        q2.data(), q2_row, q8_k.data(), output);
    const double q2_checksum = std::accumulate(output.begin(), output.end(), 0.0);

    const double dots = static_cast<double>(rows) * repeats;
    std::printf(
        "n=%d rows=%d threads=%d repeats=%d\n"
        "q1_0: %.3f ms, %.0f dots/s, %.2f GiB/s, checksum %.6g\n"
        "q2_K: %.3f ms, %.0f dots/s, %.2f GiB/s, checksum %.6g\n"
        "q1/q2 time: %.3fx\n",
        n, rows, threads, repeats,
        q1_seconds * 1000.0, dots / q1_seconds,
        dots * q1_row / q1_seconds / (1024.0 * 1024.0 * 1024.0), q1_checksum,
        q2_seconds * 1000.0, dots / q2_seconds,
        dots * q2_row / q2_seconds / (1024.0 * 1024.0 * 1024.0), q2_checksum,
        q1_seconds / q2_seconds);
    return std::isfinite(q1_checksum + q2_checksum) ? 0 : 1;
}
