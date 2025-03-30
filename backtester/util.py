import pandas as pd

from datamodel import (
    ConversionObservation,
    Listing,
    Observation,
    OrderDepth,
    TradingState,
)


def get_trade_state(timestamp: int, group: pd.DataFrame) -> TradingState:
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
    market_trades = {product: [] for product in products}

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
