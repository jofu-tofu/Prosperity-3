from datamodel import Order, Trade


def nop(product: str, market_trades: list[Trade], order: Order):
    trades_made = []
    new_market_trades = market_trades
    
    return trades_made, new_market_trades

def match_market_eq(product: str, market_trades: list[Trade], order: Order):
    trades_made = []
    new_market_trades = []
    
    if order.quantity > 0:
        

    # Check if the order is a market order
    if order.order_type == 'market':
        # Check if the order is a buy or sell order
        if order.side == 'buy':
            # Match with the best available sell orders in the market
            for trade in market_trades:
                if trade.side == 'sell' and trade.price <= order.price:
                    trades_made.append(trade)
                    new_market_trades.remove(trade)
                    break
        elif order.side == 'sell':
            # Match with the best available buy orders in the market
            for trade in market_trades:
                if trade.side == 'buy' and trade.price >= order.price:
                    trades_made.append(trade)
                    new_market_trades.remove(trade)
                    break

    return trades_made, new_market_trades