import copy
import io
import json
from collections import defaultdict
from contextlib import redirect_stdout
from typing import Any

import constants
import util
from tqdm import tqdm

from datamodel import Order, OrderDepth, Trade, TradingState


class Backtester:
    output: str

    def __init__(self, trader_fname: str, data_fname: str):
        _, self.raw_market_data, _ = util._parse_data(data_fname)
        self.trade_states = util.get_trade_states(data_fname)
        self.trader = util.get_trader(trader_fname)
        self.products = sorted(list(self.trade_states[0].listings.keys()))

        self.cur_pos = {product: 0 for product in self.products}
        self.pnl_hist = []
        self.pnl = {product: 0 for product in self.products}
        self.cash = {product: 0 for product in self.products}
        self.trades = []
        self.sb_logs = []
        self.trader_data = ""

        self.run()
        self._export()

    def _export(self):
        output = ""
        output += "Sandbox logs:\n"
        for i in self.sb_logs:
            output += json.dumps(i, indent=2) + "\n"

        output += "\n\n\nActivities log:\n"
        self.raw_market_data.sort_values(["timestamp", "product"], inplace=True)
        self.raw_market_data["profit_and_loss"] = self.pnl_hist
        market_data_csv = self.raw_market_data.to_csv(sep=";")
        market_data_csv = market_data_csv.replace("\r\n", "\n")
        output += market_data_csv

        output += "\n\n\n\nTrade History:\n"
        output += json.dumps(self.trades, indent=2)
        self.output = output

    def run(self):
        self.own_trades = defaultdict(list)
        self.market_trades = defaultdict(list)

        for ts in tqdm(self.trade_states, desc="Processing trade states"):
            self.step(ts)

    def step(self, state: TradingState):
        captured_output = io.StringIO()
        order_depths1 = copy.deepcopy(state.order_depths)
        order_depths2 = copy.deepcopy(state.order_depths)
        order_depths3 = copy.deepcopy(state.order_depths)

        with redirect_stdout(captured_output):
            orders, conversions, trader_data = self.trader.run(
                TradingState(
                    traderData=self.trader_data,
                    timestamp=state.timestamp,
                    listings=state.listings,
                    order_depths=order_depths1,
                    own_trades=dict(self.own_trades),
                    market_trades=dict(self.market_trades),
                    position=self.cur_pos,
                    observations=state.observations,
                )
            )
            self.trader_data = trader_data
        captured_output = captured_output.getvalue()

        for product in state.listings:
            new_trades = []
            for order in orders.get(product, []):
                trades_done, captured_output = self._execute_order(
                    state.timestamp,
                    order,
                    order_depths2,
                    state.market_trades,
                    captured_output,
                )
                new_trades.extend(trades_done)
            if len(new_trades) > 0:
                self.own_trades[product] = new_trades

        self.sb_logs.append(
            {
                "sandboxLog": captured_output,
                "lambdaLog": "",
                "timestamp": state.timestamp,
            }
        )

        for product in self.products:
            self._mark_pnl(
                order_depths3,
                product,
            )
            self.pnl_hist.append(self.pnl[product])
        self._add_trades(self.own_trades, self.market_trades)

    def _trade_to_dict(self, trade: Trade) -> dict[str, Any]:
        return {
            "timestamp": trade.timestamp,
            "buyer": trade.buyer,
            "seller": trade.seller,
            "symbol": trade.symbol,
            "currency": "SEASHELLS",
            "price": trade.price,
            "quantity": trade.quantity,
        }

    def _add_trades(
        self, own_trades: dict[str, list[Trade]], market_trades: dict[str, list[Trade]]
    ):
        products = set(own_trades.keys()) | set(market_trades.keys())
        for product in products:
            self.trades.extend(
                [self._trade_to_dict(trade) for trade in own_trades.get(product, [])]
            )
        for product in products:
            self.trades.extend(
                [self._trade_to_dict(trade) for trade in market_trades.get(product, [])]
            )

    def _mark_pnl(self, order_depths: OrderDepth, product):
        order_depth = order_depths[product]
        best_ask = min(order_depth.sell_orders.keys())
        best_bid = max(order_depth.buy_orders.keys())
        mid = (best_ask + best_bid) / 2

        fair = (
            constants.FAIR_MKT_VALUE[product](order_depth)
            if product in constants.FAIR_MKT_VALUE
            else mid
        )

        self.pnl[product] = self.cash[product] + fair * self.cur_pos[product]

    def _execute_buy_order(
        self,
        timestamp: int,
        order: Order,
        order_depths: dict,
        trade_history_dict: dict[str, list[Trade]],
        sandboxLog: str,
    ):
        trades = []
        order_depth = order_depths[order.symbol]

        for price, volume in list(order_depth.sell_orders.items()):
            if price > order.price or order.quantity == 0:
                break

            trade_volume = min(abs(order.quantity), abs(volume))
            if abs(trade_volume + self.cur_pos[order.symbol]) <= int(
                constants.POSITION_LIMITS[order.symbol]
            ):
                trades.append(
                    Trade(
                        order.symbol, price, trade_volume, "SUBMISSION", "", timestamp
                    )
                )
                self.cur_pos[order.symbol] += trade_volume
                self.cash[order.symbol] -= price * trade_volume
                order_depth.sell_orders[price] += trade_volume
                order.quantity -= trade_volume
            else:
                sandboxLog += f"\nOrders for product {order.symbol} exceeded limit of {constants.POSITION_LIMITS[order.symbol]} set"

            if order_depth.sell_orders[price] == 0:
                del order_depth.sell_orders[price]

        trades_at_timestamp = trade_history_dict[order.symbol]
        new_trades_at_timestamp = []
        for trade in trades_at_timestamp:
            if trade.price >= order.price:
                new_trades_at_timestamp.append(trade)
                continue

            trade_volume = min(abs(order.quantity), abs(trade.quantity))
            trades.append(
                Trade(
                    order.symbol,
                    order.price,
                    trade_volume,
                    "SUBMISSION",
                    "",
                    timestamp,
                )
            )

            order.quantity -= trade_volume
            self.cur_pos[order.symbol] += trade_volume
            self.cash[order.symbol] -= order.price * trade_volume

            if trade_volume != abs(trade.quantity):
                new_quantity = trade.quantity - trade_volume
                new_trades_at_timestamp.append(
                    Trade(
                        order.symbol,
                        order.price,
                        new_quantity,
                        "",
                        "",
                        timestamp,
                    )
                )

        if len(new_trades_at_timestamp) > 0:
            trade_history_dict[order.symbol] = new_trades_at_timestamp

        return trades, sandboxLog

    def _execute_sell_order(
        self,
        timestamp: int,
        order: Order,
        order_depths: dict,
        trade_history_dict: dict[str, list[Trade]],
        sandboxLog: str,
    ):
        trades = []
        order_depth = order_depths[order.symbol]

        for price, volume in sorted(order_depth.buy_orders.items(), reverse=True):
            if price < order.price or order.quantity == 0:
                break

            trade_volume = min(abs(order.quantity), abs(volume))
            if abs(self.cur_pos[order.symbol] - trade_volume) <= int(
                constants.POSITION_LIMITS[order.symbol]
            ):
                trades.append(
                    Trade(
                        order.symbol, price, trade_volume, "", "SUBMISSION", timestamp
                    )
                )
                self.cur_pos[order.symbol] -= trade_volume
                self.cash[order.symbol] += price * abs(trade_volume)
                order_depth.buy_orders[price] -= abs(trade_volume)
                order.quantity += trade_volume
            else:
                sandboxLog += f"\nOrders for product {order.symbol} exceeded limit of {constants.POSITION_LIMITS[order.symbol]} set"

            if order_depth.buy_orders[price] == 0:
                del order_depth.buy_orders[price]

        trades_at_timestamp = trade_history_dict[order.symbol]
        new_trades_at_timestamp = []
        for trade in trades_at_timestamp:
            if trade.price <= order.price:
                new_trades_at_timestamp.append(trade)
                continue

            trade_volume = min(abs(order.quantity), abs(trade.quantity))
            trades.append(
                Trade(
                    order.symbol,
                    order.price,
                    trade_volume,
                    "",
                    "SUBMISSION",
                    timestamp,
                )
            )

            order.quantity += trade_volume
            self.cur_pos[order.symbol] -= trade_volume
            self.cash[order.symbol] += order.price * trade_volume

            if trade_volume != abs(trade.quantity):
                new_quantity = trade.quantity - trade_volume
                new_trades_at_timestamp.append(
                    Trade(
                        order.symbol,
                        order.price,
                        new_quantity,
                        "",
                        "",
                        timestamp,
                    )
                )

        if len(new_trades_at_timestamp) > 0:
            trade_history_dict[order.symbol] = new_trades_at_timestamp

        return trades, sandboxLog

    def _execute_order(
        self,
        timestamp: int,
        order: Order,
        order_depths: dict,
        trade_history_dict: dict[str, list[Trade]],
        sandboxLog: str,
    ):
        if order.quantity == 0:
            return [], ""

        method = (
            self._execute_buy_order if order.quantity > 0 else self._execute_sell_order
        )

        return method(
            timestamp,
            order,
            order_depths,
            trade_history_dict,
            sandboxLog,
        )
