from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List
import string
import numpy as np
import jsonpickle

class Trader:
    MAX_KELP_POSITION = 50
    MAX_RESIN_POSITION = 50
    products = ['KELP', 'RAINFOREST_RESIN']
    
    def run(self, state: TradingState):
        print("Current Positions: " + str(state.position))
        if state.traderData:
            traderData = jsonpickle.decode(state.traderData)
        else:
            traderData = {}
        result = {}
        kelp_orders = []
        for product in self.products:
            order_depth: OrderDepth = state.order_depths[product]
            print(product + ' SELL ORDERS: ' + str(order_depth.sell_orders))
            print(product + ' BUY ORDERS: ' + str(order_depth.buy_orders))
            current_pos = state.position[product] if product in state.position else 0
            total_dolvol = 0
            total_vol = 0
            for ask, ask_amount in list(order_depth.sell_orders.items()):
                ask_amount = abs(ask_amount)
                total_dolvol += ask * ask_amount
                total_vol += ask_amount
            for bid, bid_amount in list(order_depth.buy_orders.items()):
                total_dolvol += bid * bid_amount
                total_vol += bid_amount
            current_vwap = total_dolvol / total_vol
            rounded_vwap = round(current_vwap)
            orders = []
            kelp_orders.append(Order(product, rounded_vwap, -current_pos)) # Sell current position at VWAP
            max_buy = -current_pos + self.MAX_KELP_POSITION
            max_sell = -current_pos - self.MAX_KELP_POSITION
            orders.append(Order(product, rounded_vwap-2, min(max_buy, 30)))
            orders.append(Order(product, rounded_vwap+2, max(max_sell, -30))) # Buy at VWAP-1 if possible
            result[product] = orders
        conversions = 1
        traderData = jsonpickle.encode(traderData)
        return result, conversions, traderData
    