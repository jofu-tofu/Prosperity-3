from datamodel import Order, Trade


def nop(timestamp: int, product: str, market_trades: list[Trade], order: Order):
    trades_made = []
    new_market_trades = market_trades

    return trades_made, new_market_trades


def match_market_eq(
    timestamp: int, product: str, market_trades: list[Trade], order: Order
):
    trades_made = []
    new_market_trades = []

    if order.quantity > 0:
        for trade in market_trades:
            if trade.price > order.price:
                new_market_trades.append(trade)
                continue

            trade_volume = min(abs(order.quantity), abs(trade.quantity))
            trades_made.append(
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

            if trade_volume != abs(trade.quantity):
                new_quantity = trade.quantity - trade_volume
                new_market_trades.append(
                    Trade(
                        order.symbol,
                        order.price,
                        new_quantity,
                        "",
                        "",
                        timestamp,
                    )
                )
        return trades_made, new_market_trades

    for trade in market_trades:
        if trade.price < order.price:
            new_market_trades.append(trade)
            continue

        trade_volume = min(abs(order.quantity), abs(trade.quantity))
        trades_made.append(
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

        if trade_volume != abs(trade.quantity):
            new_quantity = trade.quantity - trade_volume
            new_market_trades.append(
                Trade(
                    order.symbol,
                    order.price,
                    new_quantity,
                    "",
                    "",
                    timestamp,
                )
            )

    return trades_made, new_market_trades


def match_market_neq(
    timestamp: int, product: str, market_trades: list[Trade], order: Order
):
    trades_made = []
    new_market_trades = []

    if order.quantity > 0:
        for trade in market_trades:
            if trade.price >= order.price:
                new_market_trades.append(trade)
                continue

            trade_volume = min(abs(order.quantity), abs(trade.quantity))
            trades_made.append(
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

            if trade_volume != abs(trade.quantity):
                new_quantity = trade.quantity - trade_volume
                new_market_trades.append(
                    Trade(
                        order.symbol,
                        order.price,
                        new_quantity,
                        "",
                        "",
                        timestamp,
                    )
                )
        return trades_made, new_market_trades

    for trade in market_trades:
        if trade.price <= order.price:
            new_market_trades.append(trade)
            continue

        trade_volume = min(abs(order.quantity), abs(trade.quantity))
        trades_made.append(
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

        if trade_volume != abs(trade.quantity):
            new_quantity = trade.quantity - trade_volume
            new_market_trades.append(
                Trade(
                    order.symbol,
                    order.price,
                    new_quantity,
                    "",
                    "",
                    timestamp,
                )
            )

    return trades_made, new_market_trades


BOT_BEHAVIOR = {
    "nop": nop,
    "eq": match_market_eq,
    "neq": match_market_neq,
}
