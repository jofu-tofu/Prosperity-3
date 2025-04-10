from datamodel import Order, OrderDepth, TradingState
from typing import Dict, List, Tuple
import numpy as np
from collections import deque
import jsonpickle

class Trader:
    """
    CMMA Rolling Deviation Trader for SQUID_INK

    This trader uses Cumulative Moving Average Momentum (CMMA) with rolling deviation
    to determine position direction and only trades at or near the midprice.

    Key parameters:
    - lookback: Number of periods to use for CMMA moving average calculation (default: 10)
    - dev_lookback: Number of periods to use for mean absolute log price change calculation (default: 20)
      The raw CMMA is divided by this mean absolute change to normalize for volatility
    - upper_threshold: Upper threshold for CMMA (default: 0.7)
    - lower_threshold: Lower threshold for CMMA (default: 0.3)
    - max_position: Maximum allowed position size (default: 50)
    - max_spread: Maximum spread willing to pay (0, 1, or 2) (default: 1)
    - fair_price: Fair price for SQUID_INK (default: 2000)
    - allow_counter_fair: If True, allows positions against fair price direction (default: False)
      When False, prevents going long above fair price or short below fair price
    """
    def __init__(self):
        # CMMA parameters
        self.lookback = 2  # Lookback for moving average calculation (default: 10)
        self.dev_lookback = 10  # Lookback for rolling deviation calculation (default: 20)
        self.upper_threshold = 0.6
        self.lower_threshold = 0.4
        self.max_position = 5  # Maximum allowed position

        # Trading parameters
        self.max_spread = 0  # Maximum spread willing to pay (0, 1, or 2)
        self.fair_price = 2000  # Fair price for SQUID_INK
        self.allow_counter_fair = True  # If True, allows positions against fair price direction

        # Price history for CMMA calculation
        self.price_history = {}
        self.log_price_history = {}
        self.products = ["SQUID_INK"]  # Add more products as needed

        # Use the larger of the two lookbacks for the price history (dev_lookback + 1 for calculating changes)
        max_lookback = max(self.lookback, self.dev_lookback + 1)
        for product in self.products:
            self.price_history[product] = deque(maxlen=max_lookback + 1)
            self.log_price_history[product] = deque(maxlen=max_lookback + 1)

    def calculate_cmma(self, prices: deque, log_prices: deque) -> float:
        """
        Compute Cumulative Moving Average Momentum (CMMA) normalized by mean absolute log price change
        """
        # Need enough data for both lookback periods (dev_lookback + 1 for calculating changes)
        required_data = max(self.lookback, self.dev_lookback + 1)
        if len(prices) < required_data:
            return 0.5  # Default to neutral when insufficient data

        # Convert to lists for easier manipulation
        price_list = list(prices)
        log_price_list = list(log_prices)

        # Calculate EMA for CMMA
        if len(price_list) >= self.lookback + 1:
            # Use exponential moving average for smoother results
            alpha = 2 / (self.lookback + 1)
            ema = price_list[-self.lookback-1]
            for i in range(-self.lookback, 0):
                ema = alpha * price_list[i] + (1 - alpha) * ema
        else:
            # Not enough data for EMA, use simple average
            ema = sum(price_list[:-1]) / (len(price_list) - 1) if len(price_list) > 1 else price_list[0]

        current_price = price_list[-1]

        # Calculate raw CMMA
        raw_cmma = (current_price - ema) / np.sqrt(self.lookback + 1)

        # Calculate mean absolute log price change if we have enough data
        if len(log_price_list) >= self.dev_lookback + 1:  # Need at least dev_lookback + 1 points to calculate changes
            # Calculate log price changes (shifted by 1)
            recent_log_prices = log_price_list[-(self.dev_lookback+1):]
            log_price_changes = [abs(recent_log_prices[i] - recent_log_prices[i-1]) for i in range(1, len(recent_log_prices))]

            # Calculate mean of absolute changes
            mean_abs_change = sum(log_price_changes) / len(log_price_changes) if log_price_changes else 0.01
        else:
            # Not enough data for mean abs change, use a default value
            mean_abs_change = 0.01  # Default value

        # Ensure mean_abs_change is not too small to avoid division by zero
        mean_abs_change = max(mean_abs_change, 0.0001)

        # Normalize raw CMMA by mean absolute log price change
        normalized_cmma = raw_cmma / mean_abs_change

        # Normalize using sigmoid function
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        return sigmoid(normalized_cmma)

    def calculate_position_size(self, cmma: float, current_price: float) -> int:
        """
        Calculate desired position size based on CMMA value and current price
        """
        # Determine position direction based on CMMA thresholds
        if cmma > self.upper_threshold:
            # High CMMA -> short position
            direction = -1
            # Don't short below fair price unless allowed
            if current_price < self.fair_price and not self.allow_counter_fair:
                return 0
        elif cmma < self.lower_threshold:
            # Low CMMA -> long position
            direction = 1
            # Don't go long above fair price unless allowed
            if current_price > self.fair_price and not self.allow_counter_fair:
                return 0
        else:
            # CMMA in neutral zone -> no position
            return 0

        # Calculate position size based on distance from threshold
        if direction == 1:  # Long position
            distance = (self.lower_threshold - cmma) / self.lower_threshold
        else:  # Short position
            distance = (cmma - self.upper_threshold) / (1 - self.upper_threshold)

        # Ensure distance is between 0 and 1
        distance = max(0, min(1, distance))

        # Calculate target position
        target_position = int(direction * distance * self.max_position)

        return target_position

    def get_mid_price(self, order_depth: OrderDepth) -> float:
        """
        Calculate mid price from order depth
        """
        if not order_depth.sell_orders or not order_depth.buy_orders:
            return None

        mm_ask = min(order_depth.sell_orders.keys())
        mm_bid = max(order_depth.buy_orders.keys())

        # Calculate midprice and round to nearest integer
        return round((mm_ask + mm_bid) / 2)

    def get_adjusted_prices(self, order_depth: OrderDepth) -> tuple:
        """
        Calculate adjusted prices based on midprice and max spread
        """
        mid_price = self.get_mid_price(order_depth)
        if mid_price is None:
            return None, None

        # Adjust prices based on max spread parameter
        buy_price = mid_price - self.max_spread
        sell_price = mid_price + self.max_spread

        return buy_price, sell_price

    def run(self, state: TradingState) -> Tuple[Dict[str, List[Order]], int, str]:
        """
        Main trading logic - only trades at midprice with adjusted spread
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

        # Store rolling std values for each product in trader_data
        if 'rolling_std' not in trader_data:
            trader_data['rolling_std'] = {}

        result = {}

        for product in self.products:
            if product in state.order_depths:
                order_depth = state.order_depths[product]
                mid_price = self.get_mid_price(order_depth)
                if mid_price is None:
                    continue

                # Get adjusted prices based on spread parameter
                buy_price, sell_price = self.get_adjusted_prices(order_depth)
                if buy_price is None or sell_price is None:
                    continue

                # Store prices in trader_data
                if 'prices' not in trader_data:
                    trader_data['prices'] = {}
                trader_data['prices'][product] = {
                    'mid_price': mid_price,
                    'buy_price': buy_price,
                    'sell_price': sell_price
                }

                # Update price history
                if product not in self.price_history:
                    max_lookback = max(self.lookback, self.dev_lookback + 1)  # +1 for calculating changes
                    self.price_history[product] = deque(maxlen=max_lookback + 1)
                    self.log_price_history[product] = deque(maxlen=max_lookback + 1)

                self.price_history[product].append(mid_price)
                # Calculate log price and update log price history
                log_price = np.log(mid_price) if mid_price > 0 else 0
                self.log_price_history[product].append(log_price)

                # Skip if we don't have enough data (dev_lookback + 1 for calculating changes)
                if len(self.price_history[product]) < max(self.lookback, self.dev_lookback + 1):
                    continue

                # Calculate CMMA with mean absolute log price change normalization
                cmma = self.calculate_cmma(self.price_history[product], self.log_price_history[product])

                # Store CMMA value in trader_data
                trader_data['cmma_values'][product] = cmma

                # Calculate mean absolute log price change for reference
                if len(self.log_price_history[product]) >= self.dev_lookback + 1:
                    log_price_list = list(self.log_price_history[product])
                    recent_log_prices = log_price_list[-(self.dev_lookback+1):]
                    log_price_changes = [abs(recent_log_prices[i] - recent_log_prices[i-1]) for i in range(1, len(recent_log_prices))]
                    mean_abs_change = sum(log_price_changes) / len(log_price_changes) if log_price_changes else 0.01
                else:
                    mean_abs_change = 0.01  # Default value

                # Store mean absolute log price change
                trader_data['rolling_std'][product] = mean_abs_change  # Keep the same key for backward compatibility

                # Calculate target position
                current_position = state.position.get(product, 0)
                target_position = self.calculate_position_size(cmma, mid_price)
                position_difference = target_position - current_position

                # Create orders at adjusted prices based on direction
                orders: List[Order] = []
                if position_difference > 0:  # Need to buy
                    orders.append(Order(product, buy_price, position_difference))
                elif position_difference < 0:  # Need to sell
                    orders.append(Order(product, sell_price, position_difference))

                if orders:
                    result[product] = orders

        # Store parameters and timestamp for reference
        trader_data['last_timestamp'] = state.timestamp
        trader_data['parameters'] = {
            'lookback': self.lookback,
            'dev_lookback': self.dev_lookback,
            'upper_threshold': self.upper_threshold,
            'lower_threshold': self.lower_threshold,
            'max_position': self.max_position,
            'max_spread': self.max_spread,
            'fair_price': self.fair_price,
            'allow_counter_fair': self.allow_counter_fair
        }

        # Encode trader_data to string
        serialized_trader_data = jsonpickle.encode(trader_data)

        # No conversions needed for SQUID_INK
        conversions = 0

        return result, conversions, serialized_trader_data
