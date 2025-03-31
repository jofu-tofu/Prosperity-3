import importlib
import io
import os
import re

import pandas as pd


def get_trader(trader_fname):
    if isinstance(trader_fname, io.StringIO):
        trader_fname.seek(0)
        content = trader_fname.read()
        cache_path = os.path.join("cache", "tmp_trader.py")
        with open(cache_path, "w") as tmp_file:
            tmp_file.write(content)
        trader_path = cache_path
    else:
        trader_path = os.path.join("traders", trader_fname)
        if not os.path.exists(trader_path):
            raise FileNotFoundError(f"Trader file {trader_path} not found.")

    spec = importlib.util.spec_from_file_location("trader", trader_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Trader()


def _parse_data(data_fname):
    if isinstance(data_fname, io.StringIO):
        data_fname.seek(0)
        content = data_fname.read()
    else:
        data_path = os.path.join("data", data_fname)
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file {data_fname} not found.")
        with open(data_path, "r") as file:
            content = file.read()

    sbox_logs, split = content.split("\n\n\n\nActivities log:")
    sbox_logs = (
        "[" + re.sub(r"}\n{", "},\n{", sbox_logs.replace("Sandbox logs:\n", "")) + "]"
    )
    sbox_logs = pd.read_json(io.StringIO(sbox_logs)).set_index("timestamp")

    market_csv, trade_json = split.split("\n\n\n\nTrade History:")
    market_data = pd.read_csv(io.StringIO(market_csv), sep=";").set_index("timestamp")
    trade_history = pd.read_json(io.StringIO(trade_json)).set_index("timestamp")
    return sbox_logs, market_data, trade_history


# def get_trade_states(data_fname: str) -> list[TradingState]:
#     if os.path.exists(f"cache/{data_fname}"):
#         print("Loading cached trade states...")
#         with open(f"cache/{data_fname}", "rb") as f:
#             return pickle.load(f)

#     print("Cached file not found, generating trade states...")
#     _, book_df, trades_df = _parse_data(data_fname)

#     trade_states = []
#     for timestamp, group in tqdm(
#         book_df.groupby("timestamp"), desc="Processing trade states"
#     ):
#         trade_state = _get_trade_state(
#             timestamp, group, trades_df.loc[trades_df.index.intersection([timestamp])]
#         )
#         trade_states.append(trade_state)

#     with open(f"cache/{data_fname}", "wb") as f:
#         pickle.dump(trade_states, f)

#     return trade_states


# def _get_trade_state(
#     timestamp: int, group: pd.DataFrame, mkt_trades: pd.DataFrame
# ) -> TradingState:
#     products = group["product"].unique()
#     listings = {
#         product: Listing(symbol=product, product=product, denomination="USD")
#         for product in products
#     }

#     order_depths = {
#         product: OrderDepth(
#             buy_orders={
#                 row[f"bid_price_{i}"]: row[f"bid_volume_{i}"]
#                 for i in range(1, 4)
#                 if not pd.isnull(row[f"bid_price_{i}"])
#                 and not pd.isnull(row[f"bid_volume_{i}"])
#             },
#             sell_orders={
#                 row[f"ask_price_{i}"]: -row[f"ask_volume_{i}"]
#                 for i in range(1, 4)
#                 if not pd.isnull(row[f"ask_price_{i}"])
#                 and not pd.isnull(row[f"ask_volume_{i}"])
#             },
#             _mid_price=row["mid_price"],
#         )
#         for product, rows in group.groupby("product")
#         for _, row in rows.iterrows()
#     }

#     own_trades = {product: [] for product in products}

#     market_trades = {
#         product: [
#             Trade(
#                 symbol=product,
#                 price=row.price,
#                 quantity=row.quantity,
#                 timestamp=timestamp,
#                 buyer=row.buyer,
#                 seller=row.seller,
#             )
#             for row in mkt_trades[mkt_trades["symbol"] == product].itertuples()
#         ]
#         for product in products
#     }
#     position = {product: 0 for product in products}

#     observations = Observation({}, {})

#     trader_data = ""
#     state = TradingState(
#         traderData=trader_data,
#         timestamp=timestamp,
#         listings=listings,
#         order_depths=order_depths,
#         own_trades=own_trades,
#         market_trades=market_trades,
#         position=position,
#         observations=observations,
#     )

#     return state
