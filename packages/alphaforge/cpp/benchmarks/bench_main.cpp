// Native latency benchmark for the AlphaForge order book — no Python overhead.
//
// Workload: seeded random flow around a drifting mid price,
//   55% add_limit / 30% cancel / 15% market order,
// which keeps the book at a realistic resting depth. Reports throughput and
// sampled per-op latency percentiles.
//
// Build:  cmake -S cpp -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build
// Run:    ./build/bench_orderbook [n_ops]

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

#include "alphaforge/order_book.hpp"

using Clock = std::chrono::steady_clock;
using alphaforge::OrderBook;
using alphaforge::Side;

int main(int argc, char** argv) {
    const std::size_t n_ops = argc > 1 ? std::strtoull(argv[1], nullptr, 10) : 2'000'000;
    const std::size_t sample_every = 16;

    std::mt19937_64 rng(42);
    std::uniform_int_distribution<int> op_dist(0, 99);
    std::uniform_int_distribution<std::int64_t> px_offset(-50, 50);
    std::uniform_int_distribution<std::int64_t> qty_dist(1, 500);

    OrderBook book;
    std::vector<std::uint64_t> live;
    live.reserve(n_ops / 2);
    std::vector<double> samples;
    samples.reserve(n_ops / sample_every + 1);
    const std::int64_t mid = 100'000;

    // Warm-up: seed resting depth on both sides.
    for (int i = 0; i < 5'000; ++i) {
        live.push_back(book.add_limit(Side::Buy, mid - 1 - (i % 50), qty_dist(rng)));
        live.push_back(book.add_limit(Side::Sell, mid + 1 + (i % 50), qty_dist(rng)));
    }
    book.take_fills();

    const auto t0 = Clock::now();
    for (std::size_t i = 0; i < n_ops; ++i) {
        const int op = op_dist(rng);
        const bool sampled = (i % sample_every) == 0;
        const auto s0 = sampled ? Clock::now() : Clock::time_point{};

        if (op < 55) {
            const Side side = (op & 1) ? Side::Buy : Side::Sell;
            const std::int64_t px = mid + px_offset(rng);
            live.push_back(book.add_limit(side, px, qty_dist(rng)));
        } else if (op < 85 && !live.empty()) {
            std::uniform_int_distribution<std::size_t> pick(0, live.size() - 1);
            const std::size_t j = pick(rng);
            book.cancel(live[j]);
            live[j] = live.back();
            live.pop_back();
        } else {
            book.add_market((op & 1) ? Side::Buy : Side::Sell, qty_dist(rng));
        }

        if (sampled) {
            samples.push_back(
                std::chrono::duration<double, std::nano>(Clock::now() - s0).count());
        }
        if ((i & 0xFFFF) == 0) book.take_fills();  // keep the fill buffer bounded
    }
    const double total_s = std::chrono::duration<double>(Clock::now() - t0).count();

    std::sort(samples.begin(), samples.end());
    const auto pct = [&](double p) {
        return samples[static_cast<std::size_t>(p * (samples.size() - 1))];
    };
    std::printf("ops           : %zu\n", n_ops);
    std::printf("elapsed       : %.3f s\n", total_s);
    std::printf("throughput    : %.2f M ops/s\n", n_ops / total_s / 1e6);
    std::printf("latency p50   : %.0f ns\n", pct(0.50));
    std::printf("latency p90   : %.0f ns\n", pct(0.90));
    std::printf("latency p99   : %.0f ns\n", pct(0.99));
    std::printf("latency p99.9 : %.0f ns\n", pct(0.999));
    std::printf("open orders   : %zu\n", book.open_orders());
    return 0;
}
