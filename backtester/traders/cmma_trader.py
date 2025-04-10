from backtester.datamodel import Order, OrderDepth, TradingState
from typing import Dict, List, Tuple
import numpy as np
from collections import deque
import jsonpickle

class Trader:
    def __init__(self):
        # CMMA parameters
        self.lookback = 20
        self.upper_threshold = 0.7
        self.lower_threshold = 0.3
        self.exponent = 2.0
        self.max_position = 50  # Maximum allowed position

        # Price history for CMMA calculation
        self.price_history: Dict[str, deque] = {}
        self.products = ["SQUID_INK"]  # Add more products as needed

        for product in self.products:
            self.price_history[product] = deque(maxlen=self.lookback + 1)

    def calculate_cmma(self, prices: deque) -> float:
        """
        Compute Cumulative Moving Average Momentum (CMMA)
        """
        if len(prices) < self.lookback:
            return 0.5  # Default to neutral when insufficient data

        price_list = list(prices)
        current_price = price_list[-1]
        ma = sum(price_list[-self.lookback-1:-1]) / self.lookback

        # Calculate raw CMMA
        raw_cmma = (current_price - ma) / np.sqrt(self.lookback + 1)

        # Normalize using sigmoid function
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        return sigmoid(raw_cmma)

    def calculate_position_size(self, cmma: float) -> int:
        """
        Calculate desired position size based on CMMA value
        """
        # Center CMMA around 0.5 and scale to [-1, 1]
        cmma_scaled = 2 * (cmma - 0.5)

        # Calculate base position using exponential scaling
        if cmma_scaled > 0:
            # High CMMA -> short position
            base_position = -min(abs(cmma_scaled) ** self.exponent, 1.0)
        else:
            # Low CMMA -> long position
            base_position = min(abs(cmma_scaled) ** self.exponent, 1.0)

        # Scale to max position size and round to integer
        target_position = int(base_position * self.max_position)

        return target_position

    def get_mid_price(self, order_depth: OrderDepth) -> float:
        """
        Calculate mid price from order depth
        """
        mm_ask = max(order_depth.sell_orders.keys()) if len(order_depth.sell_orders) > 0 else None
        mm_bid = min(order_depth.buy_orders.keys()) if len(order_depth.buy_orders) > 0 else None

        if mm_ask is None or mm_bid is None:
            return None

        return (mm_ask + mm_bid) / 2

    def run(self, state: TradingState) -> Tuple[Dict[str, List[Order]], int, str]:
        """
        Main trading logic
        """
        # Initialize trader data from state or create new if none exists
        trader_data = {}
        if state.traderData:
            try:
                trader_data = jsonpickle.decode(state.traderData)
            except:
                trader_data = {}

        # Store CMMA values for each product in trader_data
        if 'cmma_values' not in trader_data:
            trader_data['cmma_values'] = {}

        result = {}

        for product in self.products:
            if product in state.order_depths:
                order_depth = state.order_depths[product]
                mid_price = self.get_mid_price(order_depth)
                if mid_price is None:
                    continue
                self.price_history[product].append(mid_price)
                if len(self.price_history[product]) < self.lookback:
                    continue
                cmma = self.calculate_cmma(self.price_history[product])

                # Store CMMA value in trader_data
                trader_data['cmma_values'][product] = cmma

                current_position = state.position.get(product, 0)
                target_position = self.calculate_position_size(cmma)
                position_difference = target_position - current_position
                orders: List[Order] = []
                if position_difference > 0:  # Need to buy
                    for price, volume in order_depth.sell_orders.items():
                        if position_difference <= 0:
                            break
                        volume_to_trade = min(abs(volume), position_difference)
                        orders.append(Order(product, price, volume_to_trade))
                        position_difference -= volume_to_trade

                elif position_difference < 0:  # Need to sell
                    for price, volume in order_depth.buy_orders.items():
                        if position_difference >= 0:
                            break
                        volume_to_trade = min(abs(volume), abs(position_difference))
                        orders.append(Order(product, price, -volume_to_trade))
                        position_difference += volume_to_trade

                if orders:
                    result[product] = orders

        # Store timestamp for reference
        trader_data['last_timestamp'] = state.timestamp

        # Encode trader_data to string
        serialized_trader_data = jsonpickle.encode(trader_data)

        # No conversions needed for SQUID_INK
        conversions = 0

        return result, conversions, serialized_trader_data
