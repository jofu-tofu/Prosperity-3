from backtester.datamodel import Order, OrderDepth, TradingState
from typing import Dict, List, Tuple
import numpy as np
from collections import deque
import jsonpickle

class Trader:
    """
    CMMA Rolling Deviation Trader for SQUID_INK

    This trader uses Cumulative Moving Average Momentum (CMMA) with rolling deviation
    to determine position direction and only trades at or near the midprice.
    Positions can only move in increments of 0.2 (20% of max position).

    Key parameters:
    - lookback: Number of periods to use for CMMA moving average calculation (default: 10)
    - dev_lookback: Number of periods to use for rolling deviation calculation (default: 20)
      The raw CMMA is divided by this rolling deviation to normalize for volatility
    - upper_threshold: Upper threshold for CMMA (default: 0.7)
    - lower_threshold: Lower threshold for CMMA (default: 0.3)
    - max_position: Maximum allowed position size (default: 50)
    - max_spread: Maximum spread willing to pay (0, 1, or 2) (default: 1)
    - fair_price: Fair price for SQUID_INK (default: 2000)
    - allow_counter_fair: If True, allows positions against fair price direction (default: False)
      When False, prevents going long above fair price or short below fair price

    Position sizing features:
    - Positions are quantized to increments of 0.2 * max_position (e.g., 0, 10, 20, 30, 40, 50 if max_position=50)
    - Position changes are limited to at most 0.2 * max_position in either direction per update
    - This prevents large position swings and provides more gradual position adjustments
    """
    def __init__(self):
        # CMMA parameters
        self.lookback = 10  # Lookback for moving average calculation (default: 10)
        self.dev_lookback = 20  # Lookback for rolling deviation calculation (default: 20)
        self.upper_threshold = 0.7
        self.lower_threshold = 0.3
        self.max_position = 50  # Maximum allowed position

        # Trading parameters
        self.max_spread = 1  # Maximum spread willing to pay (0, 1, or 2)
        self.fair_price = 2000  # Fair price for SQUID_INK
        self.allow_counter_fair = False  # If True, allows positions against fair price direction

        # Price history for CMMA calculation
        self.price_history: Dict[str, deque] = {}
        self.log_price_history: Dict[str, deque] = {}
        self.products = ["SQUID_INK"]  # Add more products as needed

        # Use the larger of the two lookbacks for the price history
        max_lookback = max(self.lookback, self.dev_lookback)
        for product in self.products:
            self.price_history[product] = deque(maxlen=max_lookback + 1)
            self.log_price_history[product] = deque(maxlen=max_lookback + 1)

    def calculate_cmma(self, prices: deque, log_prices: deque) -> float:
        """
        Compute Cumulative Moving Average Momentum (CMMA) with rolling deviation normalization

        Parameters:
            prices: deque of price history
            log_prices: deque of log price history

        Returns:
            float: CMMA indicator (0-1 range)
        """
        # Need enough data for both lookback periods
        required_data = max(self.lookback, self.dev_lookback)
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

        # Calculate rolling standard deviation if we have enough data
        if len(log_price_list) >= self.dev_lookback:
            # Use log prices for standard deviation calculation
            recent_log_prices = log_price_list[-self.dev_lookback:]
            mean_log_price = sum(recent_log_prices) / len(recent_log_prices)
            squared_diffs = [(p - mean_log_price) ** 2 for p in recent_log_prices]
            variance = sum(squared_diffs) / len(squared_diffs)
            rolling_std = np.sqrt(variance) if variance > 0 else 0.0001
        else:
            # Not enough data for rolling std, use a default value
            rolling_std = 0.01  # Default value

        # Ensure rolling_std is not too small to avoid division by zero
        rolling_std = max(rolling_std, 0.0001)

        # Normalize raw CMMA by rolling standard deviation
        normalized_cmma = raw_cmma / rolling_std

        # Normalize using sigmoid function
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        return sigmoid(normalized_cmma)

    def calculate_position_size(self, cmma: float, current_price: float) -> int:
        """
        Calculate desired position size based on CMMA value and current price

        Parameters:
            cmma: CMMA indicator value (0-1 range)
            current_price: Current price of the asset

        Returns:
            int: Target position size
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

        # Calculate position size in increments of 0.2 (20% of max position)
        # First, convert distance to a scale of 0 to 5 (0, 0.2, 0.4, 0.6, 0.8, 1.0 of max position)
        position_level = round(distance * 5) / 5

        # Calculate target position
        target_position = int(direction * position_level * self.max_position)

        return target_position

    def get_mid_price(self, order_depth: OrderDepth) -> float:
        """
        Calculate mid price from order depth

        Parameters:
            order_depth: OrderDepth object

        Returns:
            float: Mid price or None if not available
        """
        mm_ask = min(order_depth.sell_orders.keys()) if len(order_depth.sell_orders) > 0 else None
        mm_bid = max(order_depth.buy_orders.keys()) if len(order_depth.buy_orders) > 0 else None

        if mm_ask is None or mm_bid is None:
            return None

        # Calculate midprice and round to nearest integer
        return round((mm_ask + mm_bid) / 2)

    def get_adjusted_prices(self, order_depth: OrderDepth) -> tuple:
        """
        Calculate adjusted prices based on midprice and max spread

        Parameters:
            order_depth: OrderDepth object

        Returns:
            tuple: (buy_price, sell_price) or (None, None) if not available
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

        Parameters:
            state: TradingState object

        Returns:
            tuple: (orders, conversions, trader_data)
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
                self.price_history[product].append(mid_price)
                # Calculate log price and update log price history
                log_price = np.log(mid_price) if mid_price > 0 else 0
                self.log_price_history[product].append(log_price)

                # Skip if we don't have enough data
                if len(self.price_history[product]) < max(self.lookback, self.dev_lookback):
                    continue

                # Calculate CMMA with rolling deviation
                cmma = self.calculate_cmma(self.price_history[product], self.log_price_history[product])

                # Store CMMA value in trader_data
                trader_data['cmma_values'][product] = cmma

                # Calculate rolling standard deviation for reference
                if len(self.log_price_history[product]) >= self.dev_lookback:
                    log_price_list = list(self.log_price_history[product])
                    recent_log_prices = log_price_list[-self.dev_lookback:]
                    mean_log_price = sum(recent_log_prices) / len(recent_log_prices)
                    squared_diffs = [(p - mean_log_price) ** 2 for p in recent_log_prices]
                    variance = sum(squared_diffs) / len(squared_diffs)
                    rolling_std = np.sqrt(variance) if variance > 0 else 0.0001
                else:
                    rolling_std = 0.01  # Default value

                # Store rolling standard deviation
                trader_data['rolling_std'][product] = rolling_std

                # Calculate target position
                current_position = state.position.get(product, 0)
                target_position = self.calculate_position_size(cmma, mid_price)

                # Ensure current position is also in increments of 0.2 * max_position
                position_increment = int(0.2 * self.max_position)
                normalized_current = round(current_position / position_increment) * position_increment

                # Calculate position difference
                position_difference = target_position - normalized_current

                # Limit position change to at most 0.2 * max_position in either direction
                if position_difference > position_increment:
                    position_difference = position_increment
                elif position_difference < -position_increment:
                    position_difference = -position_increment

                # Store position information in trader_data
                if 'positions' not in trader_data:
                    trader_data['positions'] = {}
                trader_data['positions'][product] = {
                    'current': current_position,
                    'normalized_current': normalized_current,
                    'target': target_position,
                    'difference': position_difference
                }

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
