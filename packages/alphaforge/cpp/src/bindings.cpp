// pybind11 bindings for the AlphaForge native execution core.
//
// Built as alphaforge/alphaforge_native.<abi>.so so it imports as
// `from alphaforge import alphaforge_native`. See scripts/build_native.py.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "alphaforge/order_book.hpp"

namespace py = pybind11;
using alphaforge::Fill;
using alphaforge::OrderBook;
using alphaforge::Side;

PYBIND11_MODULE(alphaforge_native, m) {
    m.doc() = "AlphaForge native execution core (C++17 limit order book)";
    // This version identifies the stable Python/native boundary, not the
    // independently released AlphaForge package.
    m.attr("__abi_version__") = "1";

    py::enum_<Side>(m, "Side")
        .value("BUY", Side::Buy)
        .value("SELL", Side::Sell);

    py::class_<Fill>(m, "Fill")
        .def_readonly("maker_id", &Fill::maker_id)
        .def_readonly("taker_id", &Fill::taker_id)
        .def_readonly("price", &Fill::price)
        .def_readonly("qty", &Fill::qty)
        .def("__repr__",
             [](const Fill& f) {
                 return "Fill(maker_id=" + std::to_string(f.maker_id) +
                        ", taker_id=" + std::to_string(f.taker_id) +
                        ", price=" + std::to_string(f.price) +
                        ", qty=" + std::to_string(f.qty) + ")";
             })
        .def("as_tuple", [](const Fill& f) {
            return py::make_tuple(f.maker_id, f.taker_id, f.price, f.qty);
        });

    py::class_<OrderBook>(m, "OrderBook")
        .def(py::init<>())
        .def("add_limit", &OrderBook::add_limit, py::arg("side"), py::arg("price"),
             py::arg("qty"))
        .def("add_market", &OrderBook::add_market, py::arg("side"), py::arg("qty"))
        .def("cancel", &OrderBook::cancel, py::arg("order_id"))
        .def("best_bid", &OrderBook::best_bid)
        .def("best_ask", &OrderBook::best_ask)
        .def("bid_depth", &OrderBook::bid_depth, py::arg("levels") = 10)
        .def("ask_depth", &OrderBook::ask_depth, py::arg("levels") = 10)
        .def("bid_volume", &OrderBook::bid_volume)
        .def("ask_volume", &OrderBook::ask_volume)
        .def("open_orders", &OrderBook::open_orders)
        .def("take_fills", &OrderBook::take_fills)
        .def("clear", &OrderBook::clear);

    m.def("simulate_fill", &alphaforge::simulate_fill, py::arg("side"), py::arg("qty"),
          py::arg("mid"), py::arg("half_spread"), py::arg("tick") = 1,
          py::arg("n_levels") = 10, py::arg("qty_per_level") = 1000,
          "Depth-aware market-order fill against a synthetic book: "
          "returns (avg_fill_price_ticks, filled_qty).");
}
