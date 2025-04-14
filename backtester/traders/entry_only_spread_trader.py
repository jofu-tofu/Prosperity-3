import json
import jsonpickle
import numpy as np
from typing import Dict, List, Tuple, Any, Deque
from collections import deque
from datamodel import OrderDepth, TradingState, Order, ProsperityEncoder

class Logger:
    def __init__(self) -> None:
        self.logs = ""

    def print(self, *objects, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(str(obj) for obj in objects) + end

    def flush(self, state: TradingState, orders: Dict[str, List[Order]], conversions: int) -> None:
        print(self.logs)
        self.logs = ""

    def compress_state(self, state: TradingState, trader_data: str) -> list[Any]:
        return [
            state.timestamp,
            trader_data,
            self.compress_listings(state.listings),
            self.compress_order_depths(state.order_depths),
            self.compress_trades(state.own_trades),
            self.compress_trades(state.market_trades),
            state.position,
            self.compress_observations(state.observations),
        ]

    def compress_listings(self, listings: dict[str, Any]) -> list[list[Any]]:
        compressed = []
        for listing in listings.values():
            compressed.append([listing.symbol, listing.product, listing.denomination])

        return compressed

    def compress_order_depths(self, order_depths: dict[str, OrderDepth]) -> list[list[Any]]:
        compressed = []
        for symbol, order_depth in order_depths.items():
            compressed.append([symbol, order_depth.buy_orders, order_depth.sell_orders])

        return compressed

    def compress_trades(self, trades: dict[str, List[Any]]) -> list[list[Any]]:
        compressed = []
        for symbol, trade_list in trades.items():
            for trade in trade_list:
                compressed.append(
                    [
                        symbol,
                        trade.price,
                        trade.quantity,
                        trade.buyer,
                        trade.seller,
                        trade.timestamp,
                    ]
                )

        return compressed

    def compress_observations(self, observations: Any) -> list[Any]:
        return observations

    def to_json(self, value: Any) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        if len(value) <= max_length:
            return value

        return value[: max_length - 3] + "..."


logger = Logger()

class Trader:
    """
    Entry-Only Spread Trader that exclusively trades the custom spread (3×PB2 + 2×DJEMBES - 2×PB1)
    based on entry thresholds only, with no exit thresholds.

    Trading logic:
    - If spread is above entry threshold, sell the spread
    - If spread is below negative entry threshold, buy the spread
    - If spread is between thresholds, maintain current position
    - Only change position when crossing to the opposite side of the threshold

    Uses rolling standard deviation for z-score calculation.
    """
    def __init__(self):
        # Products to trade
        self.products = ["PICNIC_BASKET1", "PICNIC_BASKET2", "DJEMBES"]

        # Position limits for each product
        self.position_limits = {
            "PICNIC_BASKET1": 60,
            "PICNIC_BASKET2": 100,
            "DJEMBES": 60
        }

        # Custom spread components
        self.custom_spread_components = {
            "PICNIC_BASKET2": 3,
            "DJEMBES": 2,
            "PICNIC_BASKET1": -2
        }

        # Entry threshold for trading (no exit threshold)
        self.entry_threshold = 4

        # Maximum position size for the spread
        self.max_spread_position = 50

        # Initialize spread position
        self.spread_position = 0

        # Window size for rolling standard deviation calculation
        self.window_size = 200

        # Initialize spread history for rolling calculations
        self.spread_history = deque(maxlen=self.window_size + 1)

        # Minimum standard deviation to avoid division by zero
        self.min_std_dev = 1.0

        # Price adjustment parameters (how much we're willing to pay to enter/exit)
        # These values adjust the price we're willing to pay relative to the mid price
        # For mid price ending in .5:
        #   - adjustment of 0 means +0.5/-0.5 from mid price
        #   - adjustment of 1 means +1.5/-1.5 from mid price
        # For integer mid price:
        #   - adjustment is applied directly
        self.price_adjustment = 4  # Willing to pay 4 units worse than mid price

        # Track previous z-score to detect threshold crossings
        self.previous_z_score = 0

    def run(self, state: TradingState) -> Tuple[Dict[str, List[Order]], int, str]:
        """
        Main trading logic

        Parameters:
            state: TradingState object

        Returns:
            tuple: (orders, conversions, trader_data)
        """
        # Initialize result dictionary for orders
        result = {}
        logger.print("Entry-Only Spread Trader starting...")

        # Initialize trader data from state or create new if none exists
        trader_data = {}
        if state.traderData:
            try:
                trader_data = jsonpickle.decode(state.traderData)
            except:
                trader_data = {}

        # Store spread position, previous z-score, and spread history in trader_data
        if 'spread_position' not in trader_data:
            trader_data['spread_position'] = self.spread_position
        else:
            self.spread_position = trader_data['spread_position']

        if 'previous_z_score' not in trader_data:
            trader_data['previous_z_score'] = self.previous_z_score
        else:
            self.previous_z_score = trader_data['previous_z_score']

        if 'spread_history' not in trader_data:
            trader_data['spread_history'] = list(self.spread_history)
        else:
            # Convert list back to deque with maxlen
            self.spread_history = deque(trader_data['spread_history'], maxlen=self.window_size + 1)

        # Calculate current prices for each product
        current_prices = {}
        for product in self.products:
            if product in state.order_depths:
                order_depth = state.order_depths[product]
                if len(order_depth.buy_orders) > 0 and len(order_depth.sell_orders) > 0:
                    # Calculate mid price
                    best_bid = min(order_depth.buy_orders.keys())
                    best_ask = max(order_depth.sell_orders.keys())
                    mid_price = (best_bid + best_ask) / 2
                    current_prices[product] = mid_price
                elif len(order_depth.buy_orders) > 0:
                    # Only buy orders available
                    best_bid = max(order_depth.buy_orders.keys())
                    current_prices[product] = best_bid
                elif len(order_depth.sell_orders) > 0:
                    # Only sell orders available
                    best_ask = min(order_depth.sell_orders.keys())
                    current_prices[product] = best_ask

        # Check if we have prices for all products
        if len(current_prices) != len(self.products):
            logger.print("Missing prices for some products. Skipping trading.")
            trader_data['spread_position'] = self.spread_position
            trader_data['previous_z_score'] = self.previous_z_score
            return result, 0, jsonpickle.encode(trader_data)

        # Calculate custom spread
        custom_spread = (
            self.custom_spread_components["PICNIC_BASKET2"] * current_prices["PICNIC_BASKET2"] +
            self.custom_spread_components["DJEMBES"] * current_prices["DJEMBES"] +
            self.custom_spread_components["PICNIC_BASKET1"] * current_prices["PICNIC_BASKET1"]
        )

        # Add current spread to history
        self.spread_history.append(custom_spread)

        logger.print(f"Custom spread: {custom_spread}")
        logger.print(f"Current positions: {state.position}")
        logger.print(f"Current spread position: {self.spread_position}")
        logger.print(f"Spread history length: {len(self.spread_history)}")

        # Calculate z-score using rolling standard deviation
        z_score = 0
        if len(self.spread_history) > 1:
            # Calculate standard deviation of spread (excluding the current price)
            if len(self.spread_history) > self.window_size:
                # Use full window excluding the current price
                spread_values = list(self.spread_history)[-self.window_size-1:-1]
            else:
                # Use available data excluding the current price
                spread_values = list(self.spread_history)[:-1]
                

            # Calculate standard deviation
            std_dev = np.std(spread_values)*np.sqrt(self.window_size)

            # Ensure minimum standard deviation to avoid division by zero
            std_dev = max(std_dev, self.min_std_dev)

            # Calculate z-score
            z_score = custom_spread / std_dev

        logger.print(f"Z-score: {z_score}")
        logger.print(f"Previous Z-score: {self.previous_z_score}")

        # Get current positions
        current_positions = {product: 0 for product in self.products}
        for product in self.products:
            if product in state.position:
                current_positions[product] = state.position[product]

        # Determine trade signal for custom spread based on entry thresholds and threshold crossings
        spread_signal = None

        # Check if we've crossed from below threshold to above threshold
        if self.previous_z_score <= self.entry_threshold and z_score > self.entry_threshold:
            # Sell the spread if it crosses above the entry threshold
            spread_signal = -self.max_spread_position
            logger.print(f"Crossed above entry threshold. Selling spread.")

        # Check if we've crossed from above negative threshold to below negative threshold
        elif self.previous_z_score >= -self.entry_threshold and z_score < -self.entry_threshold:
            # Buy the spread if it crosses below the negative entry threshold
            spread_signal = self.max_spread_position
            logger.print(f"Crossed below negative entry threshold. Buying spread.")

        # Check if we've crossed from above threshold to below threshold
        elif self.previous_z_score > self.entry_threshold and z_score <= self.entry_threshold:
            # Neutralize position if it crosses back into the neutral zone from above
            if self.spread_position < 0:
                spread_signal = 0
                logger.print(f"Crossed back into neutral zone from above. Neutralizing position.")

        # Check if we've crossed from below negative threshold to above negative threshold
        elif self.previous_z_score < -self.entry_threshold and z_score >= -self.entry_threshold:
            # Neutralize position if it crosses back into the neutral zone from below
            if self.spread_position > 0:
                spread_signal = 0
                logger.print(f"Crossed back into neutral zone from below. Neutralizing position.")

        # If we're already in a position and cross to the opposite side, reverse position
        elif (self.spread_position > 0 and z_score > self.entry_threshold) or \
             (self.spread_position < 0 and z_score < -self.entry_threshold):
            # Reverse position if we cross to the opposite side
            spread_signal = -self.spread_position
            logger.print(f"Crossed to opposite side. Reversing position.")

        else:
            # No threshold crossing, maintain current position
            spread_signal = None
            logger.print(f"No threshold crossing. Maintaining current position.")

        logger.print(f"Spread signal: {spread_signal}")

        # Calculate target position for custom spread
        if spread_signal is None:
            # If signal is None, maintain current position (no change)
            target_spread_position = self.spread_position
        else:
            # Otherwise, use the trade signal as the target position
            target_spread_position = int(spread_signal)

        # Calculate position change for custom spread
        spread_change = int(target_spread_position - self.spread_position)

        logger.print(f"Spread change: {spread_change}")

        # Calculate component quantities for custom spread change
        buy_components = {product: 0 for product in self.products}
        sell_components = {product: 0 for product in self.products}

        if spread_change != 0:
            # Custom spread = 3×PB2 + 2×DJEMBES - 2×PB1
            pb2_qty = abs(spread_change * self.custom_spread_components["PICNIC_BASKET2"])
            djembes_qty = abs(spread_change * self.custom_spread_components["DJEMBES"])
            pb1_qty = abs(spread_change * self.custom_spread_components["PICNIC_BASKET1"])

            if spread_change > 0:  # Buying the custom spread
                # Buy positive components, sell negative components
                buy_components["PICNIC_BASKET2"] = pb2_qty
                buy_components["DJEMBES"] = djembes_qty
                sell_components["PICNIC_BASKET1"] = pb1_qty
            else:  # Selling the custom spread
                # Sell positive components, buy negative components
                sell_components["PICNIC_BASKET2"] = pb2_qty
                sell_components["DJEMBES"] = djembes_qty
                buy_components["PICNIC_BASKET1"] = pb1_qty

        # Check position limits and adjust quantities if needed
        for product in self.products:
            current_position = current_positions[product]
            position_limit = self.position_limits[product]

            # Adjust buy quantities
            if buy_components[product] > 0:
                max_buy = position_limit - current_position
                if buy_components[product] > max_buy:
                    buy_components[product] = max_buy

            # Adjust sell quantities
            if sell_components[product] > 0:
                max_sell = current_position + position_limit
                if sell_components[product] > max_sell:
                    sell_components[product] = max_sell

        # Place orders for each product
        for product in self.products:
            if product in state.order_depths:
                order_depth = state.order_depths[product]
                orders = []

                # Process buy orders
                if buy_components[product] > 0:
                    # Find available sell orders
                    if len(order_depth.sell_orders) > 0:
                        # Sort sell orders by price (ascending)
                        sorted_asks = sorted(order_depth.sell_orders.items())

                        # Calculate mid price for price adjustment
                        if len(order_depth.buy_orders) > 0:
                            best_bid = max(order_depth.buy_orders.keys())
                            best_ask = min(order_depth.sell_orders.keys())
                            mid_price = (best_bid + best_ask) / 2

                            # Calculate limit price with adjustment
                            if mid_price % 1 == 0.5:  # Mid price ends in .5
                                limit_price = int(mid_price + 0.5) + self.price_adjustment
                            else:  # Integer mid price
                                limit_price = int(mid_price) + self.price_adjustment

                            # Ensure limit price is not higher than best ask
                            limit_price = min(limit_price, best_ask)
                        else:
                            # If no buy orders, use best ask as limit price
                            limit_price = sorted_asks[0][0]

                        # Place buy order
                        orders.append(Order(product, limit_price, buy_components[product]))
                        logger.print(f"Placing buy order for {product}: {buy_components[product]} @ {limit_price}")

                # Process sell orders
                if sell_components[product] > 0:
                    # Find available buy orders
                    if len(order_depth.buy_orders) > 0:
                        # Sort buy orders by price (descending)
                        sorted_bids = sorted(order_depth.buy_orders.items(), reverse=True)

                        # Calculate mid price for price adjustment
                        if len(order_depth.sell_orders) > 0:
                            best_bid = max(order_depth.buy_orders.keys())
                            best_ask = min(order_depth.sell_orders.keys())
                            mid_price = (best_bid + best_ask) / 2

                            # Calculate limit price with adjustment
                            if mid_price % 1 == 0.5:  # Mid price ends in .5
                                limit_price = int(mid_price - 0.5) - self.price_adjustment
                            else:  # Integer mid price
                                limit_price = int(mid_price) - self.price_adjustment

                            # Ensure limit price is not lower than best bid
                            limit_price = max(limit_price, best_bid)
                        else:
                            # If no sell orders, use best bid as limit price
                            limit_price = sorted_bids[0][0]

                        # Place sell order
                        orders.append(Order(product, limit_price, -sell_components[product]))
                        logger.print(f"Placing sell order for {product}: {-sell_components[product]} @ {limit_price}")

                # Add orders to result
                if orders:
                    result[product] = orders

        # Update spread position if orders were placed
        if spread_change != 0:
            self.spread_position = target_spread_position

        # Update previous z-score for next iteration
        self.previous_z_score = z_score

        # Update trader data
        trader_data['spread_position'] = self.spread_position
        trader_data['previous_z_score'] = self.previous_z_score
        trader_data['spread_history'] = list(self.spread_history)  # Convert deque to list for serialization

        # Log final state
        logger.print(f"Final spread position: {self.spread_position}")
        logger.print(f"Standard deviation used: {std_dev if 'std_dev' in locals() else 'N/A'}")
        logger.print(f"Number of data points used for std dev: {len(spread_values) if 'spread_values' in locals() else 0}")
        logger.print(f"Current spread excluded from std dev calculation")
        logger.flush(state, result, 0)

        return result, 0, jsonpickle.encode(trader_data)
