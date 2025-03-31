from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List
import string
import jsonpickle

class Trader:
    MAX_KELP_POSITION = 50
    
    def run(self, state: TradingState):
        print("Current Positions: " + str(state.position))
        traderData = {}
        if state.traderData:
            traderData = jsonpickle.decode(state.traderData)

        result = {}
        kelp_orders = []
        current_time = state.timestamp
        order_depth: OrderDepth = state.order_depths['KELP']
        mm_ask, mm_ask_amount = list(order_depth.sell_orders.items())[-1]
        mm_bid, mm_bid_amount = list(order_depth.buy_orders.items())[-1]
        current_kelp_pos = state.position['KELP'] if 'KELP' in state.position else 0
        last_vwap = traderData['kelp_vwap'] if 'kelp_vwap' in traderData else 0
        total_dolvol = 0
        total_vol = 0
        for ask, ask_amount in list(order_depth.sell_orders.items()):
            ask_amount = abs(ask_amount)
            total_dolvol += ask * ask_amount
            total_vol += ask_amount
        for bid, bid_amount in list(order_depth.buy_orders.items()):
            total_dolvol += bid * bid_amount
            total_vol += bid_amount
        current_vwap = total_dolvol / total_vol if total_vol > 0 else (mm_ask+mm_bid) / 2
        rounded_vwap = round(current_vwap)
        print("Current VWAP: " + str(current_vwap))
        print("Current VWAP rounded: " + str(rounded_vwap))
        max_sell = -current_kelp_pos-self.MAX_KELP_POSITION
        max_buy = -current_kelp_pos+self.MAX_KELP_POSITION
        print("Max Sell: " + str(max_sell))
        print("Max Buy: " + str(max_buy))
        kelp_orders.append(Order('KELP', rounded_vwap+1, max_sell))
        kelp_orders.append(Order('KELP', rounded_vwap-1, max_buy))

        traderData['kelp_vwap'] = current_vwap
        traderData = jsonpickle.encode(traderData)
        conversions = 1
        result['KELP'] = kelp_orders
        return result, conversions, traderData
    