from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List
import string

class Trader:
    
    def run(self, state: TradingState):
        print("traderData: " + state.traderData)
        print("Observations: " + str(state.observations))

				
        result = {}
        for product in state.order_depths:
            if product == 'RAINFOREST_RESIN':
                order_depth: OrderDepth = state.order_depths[product]
                orders: List[Order] = []
                position: int = state.position.get(product, 0)
                max_buy = 50 - position
                max_sell = -50 - position

                print("POSITION", position)
                total_amount = 0
                acceptable_price = 0
                if len(order_depth.sell_orders) != 0:
                  for ask, amount in list(order_depth.sell_orders.items()):
                    acceptable_price += abs(ask * amount)
                    total_amount += abs(amount)
                if len(order_depth.buy_orders) != 0:
                  for bid, amount in list(order_depth.buy_orders.items()):
                    acceptable_price += abs(bid * amount)
                    total_amount += abs(amount)
                acceptable_price = 10000
    
                if len(order_depth.sell_orders) != 0:
                  for ask, amount in list(order_depth.sell_orders.items()):
                    if int(ask) < acceptable_price:
                      buy_volume = max(min(-amount, max_buy), 0)
                      print("BUY", str(-buy_volume) + "x", ask)
                      orders.append(Order(product, ask, buy_volume))
                      max_buy = max_buy - buy_volume
                      position = position + buy_volume
    
                    if ask == 10000 and position < 0:
                      buy_volume = min(-amount, -position)
                      print("BUY", str(-buy_volume) + "x", ask)
                      orders.append(Order(product, ask, buy_volume))
                      max_buy = max_buy - buy_volume
                      position = position + buy_volume
    
                if len(order_depth.buy_orders) != 0:
                  for bid, amount in list(order_depth.buy_orders.items()):
                    if int(bid) > acceptable_price:
                      sell_volume = min(max(-amount, max_sell), 0)
                      print("SELL", str(sell_volume) + "x", bid)
                      orders.append(Order(product, bid, sell_volume))
                      max_sell = max_sell - sell_volume
                      position = position + sell_volume

                    if bid == 10000 and position > 0:
                      sell_volume = max(-amount, -position)
                      print("SELL", str(sell_volume) + "x", bid)
                      orders.append(Order(product, bid, sell_volume))
                      max_sell = max_sell - sell_volume
                      position = position + sell_volume
    
                if len(order_depth.sell_orders) != 0 and list(order_depth.sell_orders.items())[0][0] == 10005 or (list(order_depth.sell_orders.items())[0][0] == 10004 and list(order_depth.sell_orders.items())[0][1] == -1):
                  orders.append(Order(product, acceptable_price + 4, max_sell))
                elif len(order_depth.sell_orders) != 0 and list(order_depth.sell_orders.items())[0][0] == 10003:
                  orders.append(Order(product, acceptable_price + 2, max_sell))
                elif len(order_depth.sell_orders) != 0 and list(order_depth.sell_orders.items())[0][0] == 10002:
                  orders.append(Order(product, acceptable_price + 1, max_sell))
                elif len(order_depth.sell_orders) != 0 and list(order_depth.sell_orders.items())[0][0] == 10000 and position > 0:
                  orders.append(Order(product, acceptable_price, -position))
                else:
                  orders.append(Order(product, acceptable_price + 3, max_sell))
    
                if len(order_depth.buy_orders) != 0 and list(order_depth.buy_orders.items())[0][0] == 9995 or (list(order_depth.buy_orders.items())[0][0] == 9996 and list(order_depth.buy_orders.items())[0][1] == 1):
                  orders.append(Order(product, acceptable_price - 4, max_buy))
                elif len(order_depth.buy_orders) != 0 and list(order_depth.buy_orders.items())[0][0] == 9997:
                  orders.append(Order(product, acceptable_price - 2, max_buy))
                elif len(order_depth.buy_orders) != 0 and list(order_depth.buy_orders.items())[0][0] == 9998:
                  orders.append(Order(product, acceptable_price - 1, max_buy))
                elif len(order_depth.buy_orders) != 0 and list(order_depth.buy_orders.items())[0][0] == 10000 and position < 0:
                  orders.append(Order(product, acceptable_price, -position))
                else:
                  orders.append(Order(product, acceptable_price - 3, max_buy))
    
                result[product] = orders
    
        traderData = "SAMPLE" 
        
        conversions = 1
        
        return result, conversions, traderData

