from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List
import string
import numpy as np
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
        last_vwap = traderData['kelp_vwap'] if 'kelp_vwap' in traderData else current_vwap
        rounded_vwap = round(current_vwap)
        print("Current VWAP: " + str(current_vwap))
        changein_vwap = current_vwap - last_vwap
        print("Change in VWAP: " + str(changein_vwap))
        predicted_next_vwap = current_vwap + changein_vwap*-0.2
        print("Predicted Next VWAP: " + str(predicted_next_vwap))
        rounded_predicted_next_vwap = round(predicted_next_vwap)
        neutralize = False
        if abs(rounded_predicted_next_vwap - predicted_next_vwap) < 0.25:
            neutralize = True
        print("Neutralize: " + str(neutralize))
       
        neutralize_factor = 4
        neutralize_amount = -current_kelp_pos//neutralize_factor
        if not neutralize:
            neutralize_amount = 0
        adj_curr_pos = current_kelp_pos + neutralize_amount
        max_sell = -current_kelp_pos-self.MAX_KELP_POSITION 
        max_buy = -current_kelp_pos+self.MAX_KELP_POSITION
        max_sell2 = -adj_curr_pos-self.MAX_KELP_POSITION
        max_buy2 = -adj_curr_pos+self.MAX_KELP_POSITION
        max_sell = max(max_sell, max_sell2)
        max_buy = min(max_buy, max_buy2)
        print("Max Sell: " + str(max_sell))
        print("Max Buy: " + str(max_buy))
        print('Neutralize Amount: ' + str(neutralize_amount))

        move_bid = 1
        move_ask = -1

        if current_kelp_pos > 30:
            move_ask -= 1
        elif current_kelp_pos < -30:
            move_bid += 1
        if neutralize_amount != 0:
            kelp_orders.append(Order('KELP', rounded_predicted_next_vwap, neutralize_amount))
        if not neutralize:
            kelp_orders.append(Order('KELP', int(np.ceil(predicted_next_vwap)) + move_bid - 1, max_sell))
            kelp_orders.append(Order('KELP', int(np.floor(predicted_next_vwap)) + move_ask + 1, max_buy))
        else:
            kelp_orders.append(Order('KELP', rounded_vwap + move_bid, max_sell))
            kelp_orders.append(Order('KELP', rounded_vwap + move_ask, max_buy))
        print("Orders: " + str(kelp_orders))

        traderData['kelp_vwap'] = current_vwap
        traderData = jsonpickle.encode(traderData)
        conversions = 1
        result['KELP'] = kelp_orders
        return result, conversions, traderData
    