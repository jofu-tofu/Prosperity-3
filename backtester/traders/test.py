from datamodel import Order, TradingState


class Trader:
    def run(self, state: TradingState):
        result = {}
        current_time = state.timestamp
        best_ask, best_ask_amount = list(
            state.order_depths["KELP"].sell_orders.items()
        )[0]
        best_bid, best_bid_amount = list(state.order_depths["KELP"].buy_orders.items())[
            0
        ]
        if current_time % 5000 == 0:
            result["KELP"] = [Order("KELP", best_ask-1, 1), Order("KELP", best_bid+1, -1)]
        traderData = "SAMPLE"
        return result, None, traderData
