import importlib.util
import io
import os
import pickle
from contextlib import redirect_stdout
from dataclasses import dataclass

import numpy as np
import pandas as pd
from tqdm import tqdm

from datamodel import (
    ConversionObservation,
    Listing,
    Observation,
    Order,
    OrderDepth,
    Trade,
    TradingState,
)


def get_trade_states(dir: str) -> list[TradingState]:
    if os.path.exists(f"cache/{dir}"):
        print("Loading cached trade states...")
        with open(f"cache/{dir}", "rb") as f:
            return pickle.load(f)

    print("Cached file not found, generating trade states...")
    book_df = pd.read_csv(f"data/{dir}/spread.csv", sep=";").set_index("timestamp")
    trades_df = pd.read_csv(f"data/{dir}/trades.csv", sep=";").set_index("timestamp")

    trade_states = []
    for timestamp, group in tqdm(
        book_df.groupby("timestamp"), desc="Processing trade states"
    ):
        trade_state = _get_trade_state(
            timestamp, group, trades_df.loc[trades_df.index.intersection([timestamp])]
        )
        trade_states.append(trade_state)

    with open(f"cache/{dir}", "wb") as f:
        pickle.dump(trade_states, f)

    return trade_states


def _get_trade_state(
    timestamp: int, group: pd.DataFrame, mkt_trades: pd.DataFrame
) -> TradingState:
    products = group["product"].unique()
    listings = {
        product: Listing(symbol=product, product=product, denomination="USD")
        for product in products
    }

    order_depths = {
        product: OrderDepth(
            buy_orders={
                row[f"bid_price_{i}"]: row[f"bid_volume_{i}"]
                for i in range(1, 4)
                if not pd.isnull(row[f"bid_price_{i}"])
                and not pd.isnull(row[f"bid_volume_{i}"])
            },
            sell_orders={
                row[f"ask_price_{i}"]: -row[f"ask_volume_{i}"]
                for i in range(1, 4)
                if not pd.isnull(row[f"ask_price_{i}"])
                and not pd.isnull(row[f"ask_volume_{i}"])
            },
            _mid_price=row["mid_price"],
        )
        for product, rows in group.groupby("product")
        for _, row in rows.iterrows()
    }

    own_trades = {product: [] for product in products}

    market_trades = {
        product: [
            Trade(
                symbol=product,
                price=row.price,
                quantity=row.quantity,
                timestamp=timestamp,
                buyer=row.buyer,
                seller=row.seller,
            )
            for row in mkt_trades[mkt_trades["symbol"] == product].itertuples()
        ]
        for product in products
    }
    position = {product: 0 for product in products}

    observations = Observation(
        plainValueObservations={product: 0 for product in products},
        conversionObservations={
            product: ConversionObservation(
                bidPrice=0,
                askPrice=0,
                transportFees=0,
                exportTariff=0,
                importTariff=0,
                sugarPrice=0,
                sunlightIndex=0,
            )
            for product in products
        },
    )

    trader_data = ""
    state = TradingState(
        traderData=trader_data,
        timestamp=timestamp,
        listings=listings,
        order_depths=order_depths,
        own_trades=own_trades,
        market_trades=market_trades,
        position=position,
        observations=observations,
    )

    return state


def get_trader(trader: str):
    trader_path = os.path.join("traders", f"{trader}.py")
    if not os.path.exists(trader_path):
        raise FileNotFoundError(f"Trader file {trader_path} not found.")

    spec = importlib.util.spec_from_file_location(trader, trader_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Trader()


@dataclass
class TestState:
    market_trades: dict[str, list[Trade]]
    own_trades: dict[str, list[Trade]]
    position: dict[str, int]
    cash_position: dict[str, int]
    trader_data: str


@dataclass
class StepResult:
    own_trades: list[Trade]
    trader_data: str


class Backtest:
    bt_state: TestState
    history: pd.DataFrame
    trade_history: list[Trade]

    def __init__(self, dir: str, trader: str):
        self.trade_states = get_trade_states(dir)
        self.trader = get_trader(trader)
        self.position_limits = {
            "KELP": 50,
            "RAINFOREST_RESIN": 50,
        }
        self.products = list(self.trade_states[0].listings.keys())
        self.bt_state = TestState(
            trader_data="",
            own_trades={product: [] for product in self.products},
            market_trades={product: [] for product in self.products},
            position={product: 0 for product in self.products},
            cash_position={product: 0 for product in self.products},
        )

        self.history = pd.DataFrame(
            np.nan,
            columns=["pnl", "logs"]
            + [f"{product}_position" for product in self.products]
            + [f"{product}_pnl" for product in self.products],
            index=[t.timestamp for t in self.trade_states],
        )
        self.history["logs"] = self.history["logs"].astype(str)
        self.trade_history = []

        self.run()

    def run(self):
        for state in tqdm(self.trade_states, desc="Running backtest"):
            next_mkt_trades = state.market_trades
            state.market_trades = self.bt_state.market_trades  # replace with prev
            state.position = self.bt_state.position
            state.traderData = self.bt_state.trader_data
            state.own_trades = self.bt_state.own_trades

            own_trades, trader_data, remaining_orders = self.step(state)

            self.fill_orders(remaining_orders, next_mkt_trades, own_trades)

            self.bt_state.own_trades = own_trades
            self.bt_state.trader_data = trader_data

    def fill_orders(
        self,
        remaining_orders: dict[str, tuple[list[Order], list[Order]]],
        next_mkt_trades: dict[str, list[Trade]],
        own_trades: list,
    ):
        for product in self.products:
            print(remaining_orders[product])

        return

    def step(self, state: TradingState):
        captured_output = io.StringIO()
        with redirect_stdout(captured_output):
            result, conversions, trader_data = self.trader.run(state)
        own_trades = []

        remaining_orders = {}
        for product in self.products:
            orders: list[Order] = result.get(product, [])

            buy_orders = sorted(
                [o for o in orders if o.quantity > 0], key=lambda x: -x.price
            )
            sell_orders = sorted(
                [o for o in orders if o.quantity < 0], key=lambda x: x.price
            )
            book = state.order_depths[product]

            if (
                sum([o.quantity for o in buy_orders]) + self.bt_state.position[product]
                > self.position_limits[product]
                or sum([o.quantity for o in sell_orders])
                + self.bt_state.position[product]
                < -self.position_limits[product]
            ):
                print(
                    f"Position limit exceeded for {product} at timestamp {state.timestamp}."
                )
                break

            # Match buy orders
            for sell_price in sorted(book.sell_orders.keys()):
                if len(buy_orders) == 0:
                    break
                if sell_price > buy_orders[0].price:
                    break

                book_sell = book.sell_orders[sell_price]  # is negative
                trader_buy = buy_orders[0].quantity

                if book_sell + trader_buy > 0:
                    buy_orders[0].quantity += book_sell
                    book.sell_orders[sell_price] = 0
                    qty = -book_sell

                else:
                    book.sell_orders[sell_price] += trader_buy
                    buy_orders.pop(0)
                    qty = trader_buy

                own_trades.append(
                    Trade(
                        symbol=product,
                        price=sell_price,
                        quantity=-qty,
                        buyer="SUBMISSION",
                        seller="",
                        timestamp=state.timestamp,
                    )
                )
                self.bt_state.position[product] += qty
                self.bt_state.cash_position[product] -= qty * sell_price

            # Match sell orders
            for buy_price in sorted(book.buy_orders.keys(), reverse=True):
                if len(sell_orders) == 0:
                    break
                if buy_price < sell_orders[0].price:
                    break

                book_buy = book.buy_orders[buy_price]
                trader_sell = sell_orders[0].quantity  # is negative

                if book_buy + trader_sell < 0:
                    sell_orders[0].quantity += book_buy
                    book.buy_orders[buy_price] = 0
                    qty = book_buy
                else:
                    book.buy_orders[buy_price] += trader_sell
                    sell_orders.pop(0)
                    qty = -trader_sell

                own_trades.append(
                    Trade(
                        symbol=product,
                        price=buy_price,
                        quantity=qty,
                        buyer="",
                        seller="SUBMISSION",
                        timestamp=state.timestamp,
                    )
                )
                self.bt_state.position[product] -= qty
                self.bt_state.cash_position[product] += qty * buy_price

            remaining_orders[product] = (buy_orders, sell_orders)

        self.trade_history.extend(own_trades)

        for product in self.products:
            self.history.loc[state.timestamp, f"{product}_position"] = (
                self.bt_state.position[product]
            )
            self.history.loc[state.timestamp, f"{product}_pnl"] = (
                self.bt_state.cash_position[product]
                + state.order_depths[product]._vwap * self.bt_state.position[product]
            )

        self.history.loc[state.timestamp, "pnl"] = sum(
            self.history.loc[state.timestamp, f"{product}_pnl"]
            for product in self.products
        )

        self.history.loc[state.timestamp, "logs"] = captured_output.getvalue()

        return own_trades, trader_data, remaining_orders
