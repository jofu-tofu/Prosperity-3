from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List
import string

class Trader:
    def run(self, state: TradingState):
        print("Current Positions: " + str(state.position))

        result = {}
        current_time = state.timestamp
        best_ask, best_ask_amount = list(state.order_depths['KELP'].sell_orders.items())[0]
        best_bid, best_bid_amount = list(state.order_depths['KELP'].buy_orders.items())[0]
        if current_time == 0:
            result['KELP'] = [Order('KELP', best_ask, 1)]
        traderData = "SAMPLE" 
        
				# Sample conversion request. Check more details below. 
        conversions = 1
        return result, conversions, traderData
    