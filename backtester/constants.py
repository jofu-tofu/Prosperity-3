POSITION_LIMITS = {
    "KELP": 50,
    "RAINFOREST_RESIN": 50,
}

FAIR_MKT_VALUE = {
    "KELP": lambda x: (
        (
            sum(p * x.buy_orders[p] for p in x.buy_orders)
            - sum(p * x.sell_orders[p] for p in x.sell_orders)
        )
        / (sum(x.buy_orders.values()) - sum(x.sell_orders.values()))
    )
    if x.buy_orders and x.sell_orders
    else 0,
    "RAINFOREST_RESIN": lambda _: 10000,
}

BLANK_TRADER = """from datamodel import Order, TradingState

class Trader:
    '''
    [TradingState properties]
    traderData: str,
    timestamp: Time,
    listings: Dict[Symbol, Listing],
    order_depths: Dict[Symbol, OrderDepth],
    own_trades: Dict[Symbol, List[Trade]],
    market_trades: Dict[Symbol, List[Trade]],
    position: Dict[Product, Position],
    observations: Observation,
    '''
    def run(self, state: TradingState) -> tuple[dict[str, Order], int, str]:
        # your code here
        orders = {}
        conversions = 1
        traderData = "example"
        return orders, conversions, traderData"""
