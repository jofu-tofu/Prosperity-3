from datamodel import Order, TradingState

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
        return orders, conversions, traderData awd awdawd 