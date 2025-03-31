import copy
import io
import json
import os
import pickle
from collections import defaultdict
from contextlib import redirect_stdout
from typing import Any, Dict, List, Literal

import constants
import pandas as pd
import util

from datamodel import Listing, Observation, OrderDepth, Trade, TradingState


class Backtester:
    market_data: pd.DataFrame
    trade_history: pd.DataFrame
    output: str

    def __init__(
        self,
        trader_fname: str,
        data_fname: str,
        bot_type: Literal["neq", "nop", "eq"] = "neq",
    ):
        _, market_data, trade_history = util._parse_data(data_fname)
        self.trader = util.get_trader(trader_fname)
        self.trade_history = trade_history.sort_values(by=["timestamp", "symbol"])
        self.market_data = market_data

        cache_file = f"cache/{data_fname}"
        if os.path.exists(cache_file):
            with open(cache_file, "rb") as f:
                self.cache_order_depths = pickle.load(f)
        else:
            print("Cached file not found, generating order depths...")
            self.cache_order_depths = {}
            for timestamp, group in self.market_data.groupby("timestamp"):
                self.cache_order_depths[timestamp] = self._construct_order_depths(group)
            with open(cache_file, "wb") as f:
                pickle.dump(self.cache_order_depths, f)

        self.bot_type = bot_type

        self.fair_marks = constants.FAIR_MKT_VALUE
        self.position_limit = constants.POSITION_LIMITS

        self.listings: Dict[str, Listing] = {
            product: Listing(product, product, "SEASHELLS")
            for product in self.market_data["product"].unique()
        }

        self.current_position = {product: 0 for product in self.listings.keys()}
        self.pnl_history = []
        self.pnl = {product: 0 for product in self.listings.keys()}
        self.cash = {product: 0 for product in self.listings.keys()}
        self.trades = []
        self.sandbox_logs = []
        self.run_times = []

        self.trades = []

        self.run()
        self._log_trades()

    def run(self):
        traderData = ""

        timestamp_group_md = self.market_data.groupby("timestamp")
        timestamp_group_th = self.trade_history.groupby("timestamp")

        trade_history_dict = {}
        for timestamp, group in timestamp_group_th:
            trades = []
            for _, row in group.iterrows():
                symbol = row["symbol"]
                price = row["price"]
                quantity = row["quantity"]
                buyer = row["buyer"] if pd.notnull(row["buyer"]) else ""
                seller = row["seller"] if pd.notnull(row["seller"]) else ""

                trade = Trade(
                    symbol, int(price), int(quantity), buyer, seller, timestamp
                )

                trades.append(trade)
            trade_history_dict[timestamp] = trades

        for timestamp, group in timestamp_group_md:
            own_trades = defaultdict(list)
            market_trades = defaultdict(list)

            order_depths = self.cache_order_depths[timestamp]
            order_depths_matching = copy.deepcopy(order_depths)
            order_depths_pnl = copy.deepcopy(order_depths)

            state = self._construct_trading_state(
                traderData,
                timestamp,
                self.listings,
                order_depths,
                dict(own_trades),
                dict(market_trades),
                self.current_position,
                Observation({}, {}),
            )

            # start_time = time.time()
            captured_output = io.StringIO()
            with redirect_stdout(captured_output):
                orders, conversions, traderData = self.trader.run(state)
            # end_time = time.time()
            # self.run_times.append(end_time - start_time)

            products = group["product"].tolist()
            sandboxLog = ""
            trades_at_timestamp = trade_history_dict.get(timestamp, [])

            for product in products:
                new_trades = []

                for order in orders.get(product, []):
                    executed_orders = self._execute_order(
                        timestamp,
                        order,
                        order_depths_matching,
                        self.current_position,
                        self.cash,
                        trade_history_dict,
                        sandboxLog,
                    )
                    if len(executed_orders) > 0:
                        trades_done, sandboxLog = executed_orders
                        new_trades.extend(trades_done)
                if len(new_trades) > 0:
                    own_trades[product] = new_trades

            self.sandbox_logs.append(
                {
                    "sandboxLog": sandboxLog,
                    "lambdaLog": captured_output.getvalue(),
                    "timestamp": timestamp,
                }
            )

            trades_at_timestamp = trade_history_dict.get(timestamp, [])
            if trades_at_timestamp:
                for trade in trades_at_timestamp:
                    product = trade.symbol
                    market_trades[product].append(trade)
            else:
                for product in products:
                    market_trades[product] = []

            for product in products:
                self._mark_pnl(
                    self.cash,
                    self.current_position,
                    order_depths_pnl,
                    self.pnl,
                    product,
                )
                self.pnl_history.append(self.pnl[product])

            self._add_trades(own_trades, market_trades)
            # if np.mean(self.run_times) * 1000 > 900:
            #     print(f"Mean Run time: {np.mean(self.run_times) * 1000} ms")

    def _log_trades(self):
        self.market_data["profit_and_loss"] = self.pnl_history

        output = ""
        output += "Sandbox logs:\n"
        for i in self.sandbox_logs:
            output += json.dumps(i, indent=2) + "\n"

        output += "\n\n\n\nActivities log:\n"
        market_data_csv = self.market_data.to_csv(sep=";")
        market_data_csv = market_data_csv.replace("\r\n", "\n")
        output += market_data_csv

        output += "\n\n\n\nTrade History:\n"
        output += json.dumps(self.trades, indent=2)

        self.output = output

    def _add_trades(
        self, own_trades: Dict[str, List[Trade]], market_trades: Dict[str, List[Trade]]
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

    def _construct_trading_state(
        self,
        traderData,
        timestamp,
        listings,
        order_depths,
        own_trades,
        market_trades,
        position,
        observations,
    ):
        state = TradingState(
            traderData,
            timestamp,
            listings,
            order_depths,
            own_trades,
            market_trades,
            position,
            observations,
        )
        return state

    def _construct_order_depths(self, group):
        order_depths = {}
        for idx, row in group.iterrows():
            product = row["product"]
            order_depth = OrderDepth()
            for i in range(1, 4):
                if f"bid_price_{i}" in row and f"bid_volume_{i}" in row:
                    bid_price = row[f"bid_price_{i}"]
                    bid_volume = row[f"bid_volume_{i}"]
                    if not pd.isna(bid_price) and not pd.isna(bid_volume):
                        order_depth.buy_orders[int(bid_price)] = int(bid_volume)
                if f"ask_price_{i}" in row and f"ask_volume_{i}" in row:
                    ask_price = row[f"ask_price_{i}"]
                    ask_volume = row[f"ask_volume_{i}"]
                    if not pd.isna(ask_price) and not pd.isna(ask_volume):
                        order_depth.sell_orders[int(ask_price)] = -int(ask_volume)
            order_depths[product] = order_depth
        return order_depths

    def _execute_buy_order(
        self,
        timestamp,
        order,
        order_depths,
        position,
        cash,
        trade_history_dict,
        sandboxLog,
    ):
        trades = []
        order_depth = order_depths[order.symbol]

        for price, volume in list(order_depth.sell_orders.items()):
            if price > order.price or order.quantity == 0:
                break

            trade_volume = min(abs(order.quantity), abs(volume))
            if abs(trade_volume + position[order.symbol]) <= int(
                self.position_limit[order.symbol]
            ):
                trades.append(
                    Trade(
                        order.symbol, price, trade_volume, "SUBMISSION", "", timestamp
                    )
                )
                position[order.symbol] += trade_volume
                self.cash[order.symbol] -= price * trade_volume
                order_depth.sell_orders[price] += trade_volume
                order.quantity -= trade_volume
            else:
                sandboxLog += f"\nOrders for product {order.symbol} exceeded limit of {self.position_limit[order.symbol]} set"

            if order_depth.sell_orders[price] == 0:
                del order_depth.sell_orders[price]

        trades_at_timestamp = trade_history_dict.get(timestamp, [])
        new_trades_at_timestamp = []
        for trade in trades_at_timestamp:
            if trade.symbol == order.symbol:
                if trade.price <= order.price:
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
                    position[order.symbol] += trade_volume
                    self.cash[order.symbol] -= order.price * trade_volume
                    if trade_volume == abs(trade.quantity):
                        continue
                    else:
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
                        continue
            new_trades_at_timestamp.append(trade)

        if len(new_trades_at_timestamp) > 0:
            trade_history_dict[timestamp] = new_trades_at_timestamp

        return trades, sandboxLog

    def _execute_sell_order(
        self,
        timestamp,
        order,
        order_depths,
        position,
        cash,
        trade_history_dict,
        sandboxLog,
    ):
        trades = []
        order_depth = order_depths[order.symbol]

        for price, volume in sorted(order_depth.buy_orders.items(), reverse=True):
            if price < order.price or order.quantity == 0:
                break

            trade_volume = min(abs(order.quantity), abs(volume))
            if abs(position[order.symbol] - trade_volume) <= int(
                self.position_limit[order.symbol]
            ):
                trades.append(
                    Trade(
                        order.symbol, price, trade_volume, "", "SUBMISSION", timestamp
                    )
                )
                position[order.symbol] -= trade_volume
                self.cash[order.symbol] += price * abs(trade_volume)
                order_depth.buy_orders[price] -= abs(trade_volume)
                order.quantity += trade_volume
            else:
                sandboxLog += f"\nOrders for product {order.symbol} exceeded limit of {self.position_limit[order.symbol]} set"

            if order_depth.buy_orders[price] == 0:
                del order_depth.buy_orders[price]

        trades_at_timestamp = trade_history_dict.get(timestamp, [])
        new_trades_at_timestamp = []
        for trade in trades_at_timestamp:
            if trade.symbol == order.symbol:
                if trade.price >= order.price:
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
                    position[order.symbol] -= trade_volume
                    self.cash[order.symbol] += order.price * trade_volume
                    if trade_volume == abs(trade.quantity):
                        continue
                    else:
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
                        continue
            new_trades_at_timestamp.append(trade)

        if len(new_trades_at_timestamp) > 0:
            trade_history_dict[timestamp] = new_trades_at_timestamp

        return trades, sandboxLog

    def _execute_order(
        self,
        timestamp,
        order,
        order_depths,
        position,
        cash,
        trades_at_timestamp,
        sandboxLog,
    ):
        if order.quantity == 0:
            return []
        if order.quantity > 0:
            return self._execute_buy_order(
                timestamp,
                order,
                order_depths,
                position,
                cash,
                trades_at_timestamp,
                sandboxLog,
            )
        else:
            return self._execute_sell_order(
                timestamp,
                order,
                order_depths,
                position,
                cash,
                trades_at_timestamp,
                sandboxLog,
            )

    # def _execute_conversion(
    #     self, conversions, order_depths, position, cash, observation
    # ):
    #     implied_bid = observation.implied_bid
    #     implied_ask = observation.implied_ask
    #     if conversions > 0:
    #         position["ORCHIDS"] += abs(conversions)
    #         cash["ORCHIDS"] -= implied_ask * abs(conversions)
    #     else:
    #         position["ORCHIDS"] -= abs(conversions)
    #         cash["ORCHIDS"] += implied_bid * abs(conversions)

    def _mark_pnl(self, cash, position, order_depths, pnl, product):
        order_depth = order_depths[product]
        best_ask = min(order_depth.sell_orders.keys())
        best_bid = max(order_depth.buy_orders.keys())
        mid = (best_ask + best_bid) / 2

        fair = (
            self.fair_marks[product](order_depth) if product in self.fair_marks else mid
        )

        self.pnl[product] = self.cash[product] + fair * self.current_position[product]
