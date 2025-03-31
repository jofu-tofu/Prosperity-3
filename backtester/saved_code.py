from datamodel import Order

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
    def run(self, state: TradingState) -> tuple[list[Order], int, str]:
        # your code here
        
        return orders, conversions, traderData
        