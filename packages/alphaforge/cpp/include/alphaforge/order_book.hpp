// AlphaForge native execution core: price-time-priority limit order book.
//
// Design goals:
//  - O(log P) insertion at a new price level, O(1) amortized matching per fill,
//    O(1) cancel via an id -> iterator index (std::list iterators are stable).
//  - Integer tick prices and integer quantities: no floating point in the
//    matching path, so behaviour is exactly reproducible across platforms and
//    bit-identical to the pure-Python reference implementation.
//  - Header-only, no dependencies beyond the standard library.
//
// Semantics (mirrored by alphaforge/execution/orderbook_py.py and enforced by
// tests/test_orderbook.py parity tests):
//  - Limit orders match while crossed, fill at the resting (maker) price,
//    and any remainder rests in the book.
//  - Market orders walk the opposite side; any unfilled remainder is dropped.
//  - Fills are recorded as (maker_id, taker_id, price, qty) in match order.
//  - No self-match prevention (single-strategy simulation does not need it).

#pragma once

#include <cstdint>
#include <list>
#include <map>
#include <optional>
#include <unordered_map>
#include <utility>
#include <vector>

namespace alphaforge {

enum class Side : std::uint8_t { Buy = 0, Sell = 1 };

struct Order {
    std::uint64_t id;
    Side side;
    std::int64_t price;
    std::int64_t qty;
};

struct Fill {
    std::uint64_t maker_id;
    std::uint64_t taker_id;
    std::int64_t price;
    std::int64_t qty;
};

class OrderBook {
  public:
    // Returns the order id. The order may have fully or partially executed on
    // arrival; the remainder (if any) rests at `price`.
    std::uint64_t add_limit(Side side, std::int64_t price, std::int64_t qty) {
        const std::uint64_t id = next_id_++;
        if (qty <= 0) return id;
        std::int64_t remaining = qty;
        if (side == Side::Buy) {
            remaining = match_(asks_, id, price, remaining,
                               [price](std::int64_t best) { return best <= price; });
            if (remaining > 0) rest_(bids_, Side::Buy, id, price, remaining);
        } else {
            remaining = match_(bids_, id, price, remaining,
                               [price](std::int64_t best) { return best >= price; });
            if (remaining > 0) rest_(asks_, Side::Sell, id, price, remaining);
        }
        return id;
    }

    // Executes immediately against the opposite side; unfilled quantity is
    // dropped. Returns the filled quantity.
    std::int64_t add_market(Side side, std::int64_t qty) {
        const std::uint64_t id = next_id_++;
        if (qty <= 0) return 0;
        std::int64_t remaining;
        if (side == Side::Buy) {
            remaining = match_(asks_, id, 0, qty, [](std::int64_t) { return true; });
        } else {
            remaining = match_(bids_, id, 0, qty, [](std::int64_t) { return true; });
        }
        return qty - remaining;
    }

    bool cancel(std::uint64_t order_id) {
        auto found = index_.find(order_id);
        if (found == index_.end()) return false;
        const Locator& loc = found->second;
        if (loc.side == Side::Buy) {
            erase_(bids_, loc);
        } else {
            erase_(asks_, loc);
        }
        index_.erase(found);
        return true;
    }

    std::optional<std::int64_t> best_bid() const {
        if (bids_.empty()) return std::nullopt;
        return bids_.begin()->first;
    }

    std::optional<std::int64_t> best_ask() const {
        if (asks_.empty()) return std::nullopt;
        return asks_.begin()->first;
    }

    // Top `levels` of (price, total_qty), best first.
    std::vector<std::pair<std::int64_t, std::int64_t>> bid_depth(std::size_t levels) const {
        return depth_(bids_, levels);
    }

    std::vector<std::pair<std::int64_t, std::int64_t>> ask_depth(std::size_t levels) const {
        return depth_(asks_, levels);
    }

    std::int64_t bid_volume() const { return volume_(bids_); }
    std::int64_t ask_volume() const { return volume_(asks_); }
    std::size_t open_orders() const { return index_.size(); }

    // Drains and returns fills accumulated since the last call.
    std::vector<Fill> take_fills() {
        std::vector<Fill> out;
        out.swap(fills_);
        return out;
    }

    void clear() {
        bids_.clear();
        asks_.clear();
        index_.clear();
        fills_.clear();
    }

  private:
    struct Level {
        std::int64_t total = 0;
        std::list<Order> orders;
    };

    using BidMap = std::map<std::int64_t, Level, std::greater<std::int64_t>>;
    using AskMap = std::map<std::int64_t, Level, std::less<std::int64_t>>;

    struct Locator {
        Side side;
        std::int64_t price;
        std::list<Order>::iterator it;
    };

    template <typename Book, typename Crossed>
    std::int64_t match_(Book& book, std::uint64_t taker_id, std::int64_t /*limit*/,
                        std::int64_t remaining, Crossed crossed) {
        while (remaining > 0 && !book.empty()) {
            auto level_it = book.begin();
            if (!crossed(level_it->first)) break;
            Level& level = level_it->second;
            while (remaining > 0 && !level.orders.empty()) {
                Order& maker = level.orders.front();
                const std::int64_t traded = maker.qty < remaining ? maker.qty : remaining;
                fills_.push_back({maker.id, taker_id, maker.price, traded});
                maker.qty -= traded;
                level.total -= traded;
                remaining -= traded;
                if (maker.qty == 0) {
                    index_.erase(maker.id);
                    level.orders.pop_front();
                }
            }
            if (level.orders.empty()) book.erase(level_it);
        }
        return remaining;
    }

    template <typename Book>
    void rest_(Book& book, Side side, std::uint64_t id, std::int64_t price, std::int64_t qty) {
        Level& level = book[price];
        level.orders.push_back({id, side, price, qty});
        level.total += qty;
        index_[id] = {side, price, std::prev(level.orders.end())};
    }

    template <typename Book>
    void erase_(Book& book, const Locator& loc) {
        auto level_it = book.find(loc.price);
        if (level_it == book.end()) return;
        level_it->second.total -= loc.it->qty;
        level_it->second.orders.erase(loc.it);
        if (level_it->second.orders.empty()) book.erase(level_it);
    }

    template <typename Book>
    static std::vector<std::pair<std::int64_t, std::int64_t>> depth_(const Book& book,
                                                                     std::size_t levels) {
        std::vector<std::pair<std::int64_t, std::int64_t>> out;
        out.reserve(levels);
        for (auto it = book.begin(); it != book.end() && out.size() < levels; ++it) {
            out.emplace_back(it->first, it->second.total);
        }
        return out;
    }

    template <typename Book>
    static std::int64_t volume_(const Book& book) {
        std::int64_t total = 0;
        for (const auto& [price, level] : book) total += level.total;
        return total;
    }

    BidMap bids_;
    AskMap asks_;
    std::unordered_map<std::uint64_t, Locator> index_;
    std::vector<Fill> fills_;
    std::uint64_t next_id_ = 1;
};

// Depth-aware fill simulation used by the paper-trading simulator: build a
// synthetic book around `mid` and walk it with a market order. Returns
// (avg_fill_price_ticks, filled_qty). Slippage emerges from consuming levels
// rather than from a flat bps assumption.
inline std::pair<double, std::int64_t> simulate_fill(Side side, std::int64_t qty,
                                                     std::int64_t mid, std::int64_t half_spread,
                                                     std::int64_t tick, std::size_t n_levels,
                                                     std::int64_t qty_per_level) {
    if (qty <= 0 || qty_per_level <= 0 || n_levels == 0) return {0.0, 0};
    OrderBook book;
    for (std::size_t i = 0; i < n_levels; ++i) {
        const auto offset = static_cast<std::int64_t>(i) * tick;
        book.add_limit(Side::Sell, mid + half_spread + offset, qty_per_level);
        book.add_limit(Side::Buy, mid - half_spread - offset, qty_per_level);
    }
    const std::int64_t filled = book.add_market(side, qty);
    if (filled == 0) return {0.0, 0};
    double notional = 0.0;
    for (const Fill& f : book.take_fills()) {
        if (f.taker_id != 0) notional += static_cast<double>(f.price) * static_cast<double>(f.qty);
    }
    return {notional / static_cast<double>(filled), filled};
}

}  // namespace alphaforge
