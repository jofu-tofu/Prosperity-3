from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
from typing import Dict, List, Tuple, Any
import numpy as np
import jsonpickle
import json

class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
        base_length = len(
            self.to_json(
                [
                    self.compress_state(state, ""),
                    self.compress_orders(orders),
                    conversions,
                    "",
                    "",
                ]
            )
        )

        # We truncate state.traderData, trader_data, and self.logs to the same max. length to fit the log limit
        max_item_length = (self.max_log_length - base_length) // 3

        print(
            self.to_json(
                [
                    self.compress_state(state, self.truncate(state.traderData, max_item_length)),
                    self.compress_orders(orders),
                    conversions,
                    self.truncate(trader_data, max_item_length),
                    self.truncate(self.logs, max_item_length),
                ]
            )
        )

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

    def compress_listings(self, listings: dict[Symbol, Listing]) -> list[list[Any]]:
        compressed = []
        for listing in listings.values():
            compressed.append([listing.symbol, listing.product, listing.denomination])

        return compressed

    def compress_order_depths(self, order_depths: dict[Symbol, OrderDepth]) -> dict[Symbol, list[Any]]:
        compressed = {}
        for symbol, order_depth in order_depths.items():
            compressed[symbol] = [order_depth.buy_orders, order_depth.sell_orders]

        return compressed

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
        compressed = []
        for arr in trades.values():
            for trade in arr:
                compressed.append(
                    [
                        trade.symbol,
                        trade.price,
                        trade.quantity,
                        trade.buyer,
                        trade.seller,
                        trade.timestamp,
                    ]
                )

        return compressed

    def compress_observations(self, observations: Observation) -> list[Any]:
        conversion_observations = {}
        for product, observation in observations.conversionObservations.items():
            conversion_observations[product] = [
                observation.bidPrice,
                observation.askPrice,
                observation.transportFees,
                observation.exportTariff,
                observation.importTariff,
                observation.sugarPrice,
                observation.sunlightIndex,
            ]

        return [observations.plainValueObservations, conversion_observations]

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        compressed = []
        for arr in orders.values():
            for order in arr:
                compressed.append([order.symbol, order.price, order.quantity])

        return compressed

    def to_json(self, value: Any) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        if len(value) <= max_length:
            return value

        return value[: max_length - 3] + "..."


logger = Logger()

class Trader:
    """
    Priority Basket Trader that prioritizes trading the custom spread first,
    then trades PB1 and PB2 spreads with remaining position capacity.

    Trading logic:
    1. Trade the custom spread (3×PB2 + 2×DJEMBES - 2×PB1) based on entry/exit thresholds
    2. Calculate remaining position capacity
    3. Trade PB1 and PB2 spreads with remaining capacity, with preference to PB2
    4. When buying custom spread, only sell PB2 spread and buy PB1 spread
    """
    def __init__(self):
        # Products to trade
        self.products = ["PICNIC_BASKET1", "PICNIC_BASKET2", "CROISSANTS", "JAMS", "DJEMBES"]

        # Position limits for each product
        self.position_limits = {
            "CROISSANTS": 250,
            "JAMS": 350,
            "DJEMBES": 60,
            "PICNIC_BASKET1": 60,
            "PICNIC_BASKET2": 100
        }

        # Spread components
        self.pb1_components = {
            "CROISSANTS": 6,
            "JAMS": 3,
            "DJEMBES": 1
        }

        self.pb2_components = {
            "CROISSANTS": 4,
            "JAMS": 2
        }

        self.custom_spread_components = {
            "PICNIC_BASKET2": 3,
            "DJEMBES": 2,
            "PICNIC_BASKET1": -2
        }

        # Thresholds for trading
        self.entry_thresholds = {
            "pb1_spread": 40.0,
            "pb2_spread": 30.0,
            "custom_spread": 10.0
        }

        self.exit_thresholds = {
            "pb1_spread": 5.0,
            "pb2_spread": 5.0,
            "custom_spread": 0
        }

        # Maximum position size for each spread
        self.max_spread_positions = {
            "pb1_spread": 50,
            "pb2_spread": 80,
            "custom_spread": 50
        }

        # Initialize price history for calculating z-scores
        self.price_history = {product: [] for product in self.products}
        self.spread_history = {
            "pb1_spread": [],
            "pb2_spread": [],
            "custom_spread": []
        }

        # Initialize spread positions
        self.spread_positions = {
            "pb1_spread": 0,
            "pb2_spread": 0,
            "custom_spread": 0
        }

        # Window size for calculating standard deviation
        self.window_size = 100

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
        logger.print("Priority Basket Trader starting...")

        # Initialize trader data from state or create new if none exists
        trader_data = {}
        if state.traderData:
            try:
                trader_data = jsonpickle.decode(state.traderData)
            except:
                trader_data = {}

        # Store spread positions in trader_data
        if 'spread_positions' not in trader_data:
            trader_data['spread_positions'] = self.spread_positions
        else:
            self.spread_positions = trader_data['spread_positions']

        # Store spread history in trader_data
        if 'spread_history' not in trader_data:
            trader_data['spread_history'] = self.spread_history
        else:
            self.spread_history = trader_data['spread_history']

        # Get current prices for all products
        current_prices = {}
        for product in self.products:
            if product in state.order_depths:
                order_depth = state.order_depths[product]
                if len(order_depth.buy_orders) > 0 and len(order_depth.sell_orders) > 0:
                    # Calculate mid price
                    best_bid = max(order_depth.buy_orders.keys())
                    best_ask = min(order_depth.sell_orders.keys())
                    mid_price = (best_bid + best_ask) / 2
                    current_prices[product] = mid_price

        # Check if we have prices for all products
        if len(current_prices) != len(self.products):
            # Not enough data to trade
            logger.print("Missing prices for some products, skipping trading")
            return result, 0, jsonpickle.encode(trader_data)

        # Update price history
        for product, price in current_prices.items():
            self.price_history[product].append(price)
            # Keep only the last 200 prices
            if len(self.price_history[product]) > 200:
                self.price_history[product] = self.price_history[product][-200:]

        # Calculate theoretical basket values
        theoretical_pb1 = (
            self.pb1_components["CROISSANTS"] * current_prices["CROISSANTS"] +
            self.pb1_components["JAMS"] * current_prices["JAMS"] +
            self.pb1_components["DJEMBES"] * current_prices["DJEMBES"]
        )

        theoretical_pb2 = (
            self.pb2_components["CROISSANTS"] * current_prices["CROISSANTS"] +
            self.pb2_components["JAMS"] * current_prices["JAMS"]
        )

        # Calculate spreads
        pb1_spread = current_prices["PICNIC_BASKET1"] - theoretical_pb1
        pb2_spread = current_prices["PICNIC_BASKET2"] - theoretical_pb2
        custom_spread = (
            self.custom_spread_components["PICNIC_BASKET2"] * current_prices["PICNIC_BASKET2"] +
            self.custom_spread_components["DJEMBES"] * current_prices["DJEMBES"] +
            self.custom_spread_components["PICNIC_BASKET1"] * current_prices["PICNIC_BASKET1"]
        )

        logger.print(f"PB1 spread: {pb1_spread}, PB2 spread: {pb2_spread}, Custom spread: {custom_spread}")
        logger.print(f"Current positions: {state.position}")
        logger.print(f"Current spread positions: {self.spread_positions}")

        # Update spread history
        self.spread_history["pb1_spread"].append(pb1_spread)
        self.spread_history["pb2_spread"].append(pb2_spread)
        self.spread_history["custom_spread"].append(custom_spread)

        # Keep only the last 200 spread values
        for spread_name in self.spread_history:
            if len(self.spread_history[spread_name]) > 200:
                self.spread_history[spread_name] = self.spread_history[spread_name][-200:]

        # Calculate z-scores for spreads
        z_scores = {}
        for spread_name, spread_values in self.spread_history.items():
            if len(spread_values) > self.window_size:
                # Calculate rolling standard deviation of spread differences
                spread_diffs = np.diff(spread_values[-self.window_size-1:])
                rolling_std = np.std(spread_diffs)

                # Avoid division by zero
                if rolling_std > 0:
                    current_spread = spread_values[-1]
                    z_scores[spread_name] = current_spread / rolling_std
                else:
                    z_scores[spread_name] = 0
            else:
                z_scores[spread_name] = 0

        logger.print(f"Z-scores: {z_scores}")

        # Get current positions
        current_positions = {product: 0 for product in self.products}
        for product in self.products:
            if product in state.position:
                current_positions[product] = state.position[product]

        # Step 1: Determine trade signal for custom spread
        custom_spread_signal = None

        if z_scores["custom_spread"] > self.entry_thresholds["custom_spread"]:
            # Sell the custom spread if it's above the entry threshold
            custom_spread_signal = -self.max_spread_positions["custom_spread"]
        elif z_scores["custom_spread"] < -self.entry_thresholds["custom_spread"]:
            # Buy the custom spread if it's below the negative entry threshold
            custom_spread_signal = self.max_spread_positions["custom_spread"]
        elif abs(z_scores["custom_spread"]) < self.exit_thresholds["custom_spread"]:
            # Neutralize position if spread is within exit thresholds
            custom_spread_signal = 0
        else:
            # In the middle zone - neither entry nor exit - hold current position without trading
            custom_spread_signal = None

        logger.print(f"Custom spread signal: {custom_spread_signal}")

        # Calculate target position for custom spread
        if custom_spread_signal is None:
            # If signal is None, maintain current position (no change)
            target_custom_spread_position = self.spread_positions["custom_spread"]
        else:
            # Otherwise, use the trade signal as the target position
            target_custom_spread_position = int(custom_spread_signal)

        # Calculate position change for custom spread
        custom_spread_change = int(target_custom_spread_position - self.spread_positions["custom_spread"])

        logger.print(f"Custom spread change: {custom_spread_change}")

        # Calculate component quantities for custom spread change
        custom_spread_buy_components = {product: 0 for product in self.products}
        custom_spread_sell_components = {product: 0 for product in self.products}

        if custom_spread_change != 0:
            # Custom spread = 3×PB2 + 2×DJEMBES - 2×PB1
            pb2_qty = abs(custom_spread_change * self.custom_spread_components["PICNIC_BASKET2"])
            djembes_qty = abs(custom_spread_change * self.custom_spread_components["DJEMBES"])
            pb1_qty = abs(custom_spread_change * self.custom_spread_components["PICNIC_BASKET1"])

            if custom_spread_change > 0:  # Buying the custom spread
                # Buy positive components, sell negative components
                custom_spread_buy_components["PICNIC_BASKET2"] = pb2_qty
                custom_spread_buy_components["DJEMBES"] = djembes_qty
                custom_spread_sell_components["PICNIC_BASKET1"] = pb1_qty
            else:  # Selling the custom spread
                # Sell positive components, buy negative components
                custom_spread_sell_components["PICNIC_BASKET2"] = pb2_qty
                custom_spread_sell_components["DJEMBES"] = djembes_qty
                custom_spread_buy_components["PICNIC_BASKET1"] = pb1_qty

        logger.print(f"Custom spread buy components: {custom_spread_buy_components}")
        logger.print(f"Custom spread sell components: {custom_spread_sell_components}")

        # Step 2: Calculate remaining position capacity after custom spread
        remaining_capacity = {}
        for product in self.products:
            # Calculate how much capacity we have left after the custom spread trade
            buy_capacity = self.position_limits[product] - current_positions[product] - custom_spread_buy_components[product]
            sell_capacity = self.position_limits[product] + current_positions[product] - custom_spread_sell_components[product]
            remaining_capacity[product] = {"buy": max(0, buy_capacity), "sell": max(0, sell_capacity)}

        logger.print(f"Remaining capacity: {remaining_capacity}")

        # Step 3: Determine trade signals for PB1 and PB2 spreads
        pb1_spread_signal = None
        pb2_spread_signal = None

        # Check if we're buying or selling the custom spread
        buying_custom_spread = custom_spread_change > 0
        selling_custom_spread = custom_spread_change < 0

        # PB1 spread trading logic - only buy if selling custom spread, only sell if buying custom spread
        if z_scores["pb1_spread"] > self.entry_thresholds["pb1_spread"] and selling_custom_spread:
            # Sell the spread if it's above the entry threshold and we're buying custom spread
            pb1_spread_signal = -self.max_spread_positions["pb1_spread"]
        elif z_scores["pb1_spread"] < -self.entry_thresholds["pb1_spread"] and buying_custom_spread:
            # Buy the spread if it's below the negative entry threshold and we're selling custom spread
            pb1_spread_signal = self.max_spread_positions["pb1_spread"]
        elif abs(z_scores["pb1_spread"]) < self.exit_thresholds["pb1_spread"]:
            # Neutralize position if spread is within exit thresholds
            pb1_spread_signal = 0
        else:
            # In the middle zone - neither entry nor exit - hold current position without trading
            pb1_spread_signal = None

        # PB2 spread trading logic - only buy if selling custom spread, only sell if buying custom spread
        if z_scores["pb2_spread"] > self.entry_thresholds["pb2_spread"] and buying_custom_spread:
            # Sell the spread if it's above the entry threshold and we're buying custom spread
            pb2_spread_signal = -self.max_spread_positions["pb2_spread"]
        elif z_scores["pb2_spread"] < -self.entry_thresholds["pb2_spread"] and selling_custom_spread:
            # Buy the spread if it's below the negative entry threshold and we're selling custom spread
            pb2_spread_signal = self.max_spread_positions["pb2_spread"]
        elif abs(z_scores["pb2_spread"]) < self.exit_thresholds["pb2_spread"]:
            # Neutralize position if spread is within exit thresholds
            pb2_spread_signal = 0
        else:
            # In the middle zone - neither entry nor exit - hold current position without trading
            pb2_spread_signal = None

        logger.print(f"PB1 spread signal: {pb1_spread_signal}, PB2 spread signal: {pb2_spread_signal}")

        # Calculate target positions for PB1 and PB2 spreads
        if pb1_spread_signal is None:
            # If signal is None, maintain current position (no change)
            target_pb1_spread_position = self.spread_positions["pb1_spread"]
        else:
            # Otherwise, use the trade signal as the target position
            target_pb1_spread_position = int(pb1_spread_signal)

        if pb2_spread_signal is None:
            # If signal is None, maintain current position (no change)
            target_pb2_spread_position = self.spread_positions["pb2_spread"]
        else:
            # Otherwise, use the trade signal as the target position
            target_pb2_spread_position = int(pb2_spread_signal)

        # Calculate position changes for PB1 and PB2 spreads
        pb1_spread_change = int(target_pb1_spread_position - self.spread_positions["pb1_spread"])
        pb2_spread_change = int(target_pb2_spread_position - self.spread_positions["pb2_spread"])

        logger.print(f"PB1 spread change: {pb1_spread_change}, PB2 spread change: {pb2_spread_change}")

        # Calculate component quantities for PB1 and PB2 spread changes
        pb_spread_buy_components = {product: 0 for product in self.products}
        pb_spread_sell_components = {product: 0 for product in self.products}

        # Process PB2 spread first (preference)
        if pb2_spread_change != 0:
            # PB2 spread = PICNIC_BASKET2 - (4×CROISSANTS + 2×JAMS)
            pb2_qty = abs(pb2_spread_change)  # Quantity of PICNIC_BASKET2
            croissants_qty = pb2_qty * self.pb2_components["CROISSANTS"]
            jams_qty = pb2_qty * self.pb2_components["JAMS"]

            if pb2_spread_change > 0:  # Buying the spread (buy basket, sell components)
                pb_spread_buy_components["PICNIC_BASKET2"] += pb2_qty
                pb_spread_sell_components["CROISSANTS"] += croissants_qty
                pb_spread_sell_components["JAMS"] += jams_qty
            else:  # Selling the spread (sell basket, buy components)
                pb_spread_sell_components["PICNIC_BASKET2"] += pb2_qty
                pb_spread_buy_components["CROISSANTS"] += croissants_qty
                pb_spread_buy_components["JAMS"] += jams_qty

        # Then process PB1 spread
        if pb1_spread_change != 0:
            # PB1 spread = PICNIC_BASKET1 - (6×CROISSANTS + 3×JAMS + 1×DJEMBE)
            pb1_qty = abs(pb1_spread_change)  # Quantity of PICNIC_BASKET1
            croissants_qty = pb1_qty * self.pb1_components["CROISSANTS"]
            jams_qty = pb1_qty * self.pb1_components["JAMS"]
            djembes_qty = pb1_qty * self.pb1_components["DJEMBES"]

            if pb1_spread_change > 0:  # Buying the spread (buy basket, sell components)
                pb_spread_buy_components["PICNIC_BASKET1"] += pb1_qty
                pb_spread_sell_components["CROISSANTS"] += croissants_qty
                pb_spread_sell_components["JAMS"] += jams_qty
                pb_spread_sell_components["DJEMBES"] += djembes_qty
            else:  # Selling the spread (sell basket, buy components)
                pb_spread_sell_components["PICNIC_BASKET1"] += pb1_qty
                pb_spread_buy_components["CROISSANTS"] += croissants_qty
                pb_spread_buy_components["JAMS"] += jams_qty
                pb_spread_buy_components["DJEMBES"] += djembes_qty

        logger.print(f"PB spread buy components: {pb_spread_buy_components}")
        logger.print(f"PB spread sell components: {pb_spread_sell_components}")

        # Step 4: Check if PB spread trades would exceed remaining capacity
        pb_scale_factor = 1.0

        # Check buy capacity
        for product, buy_qty in pb_spread_buy_components.items():
            if buy_qty > 0 and buy_qty > remaining_capacity[product]["buy"]:
                product_scale_factor = remaining_capacity[product]["buy"] / buy_qty if buy_qty > 0 else 1.0
                pb_scale_factor = min(pb_scale_factor, product_scale_factor)
                logger.print(f"Buy capacity would be exceeded for {product}: buy={buy_qty}, capacity={remaining_capacity[product]['buy']}, scale_factor={product_scale_factor}")

        # Check sell capacity
        for product, sell_qty in pb_spread_sell_components.items():
            if sell_qty > 0 and sell_qty > remaining_capacity[product]["sell"]:
                product_scale_factor = remaining_capacity[product]["sell"] / sell_qty if sell_qty > 0 else 1.0
                pb_scale_factor = min(pb_scale_factor, product_scale_factor)
                logger.print(f"Sell capacity would be exceeded for {product}: sell={sell_qty}, capacity={remaining_capacity[product]['sell']}, scale_factor={product_scale_factor}")

        # Scale PB spread trades if necessary
        if pb_scale_factor < 1.0:
            logger.print(f"Scaling PB spread trades by factor: {pb_scale_factor}")

            # Scale the spread position changes
            pb1_spread_change = int(pb1_spread_change * pb_scale_factor)
            pb2_spread_change = int(pb2_spread_change * pb_scale_factor)

            # Recalculate component changes with scaled spread positions
            pb_spread_buy_components = {product: 0 for product in self.products}
            pb_spread_sell_components = {product: 0 for product in self.products}

            # Process PB2 spread first (preference)
            if pb2_spread_change != 0:
                # PB2 spread = PICNIC_BASKET2 - (4×CROISSANTS + 2×JAMS)
                pb2_qty = abs(pb2_spread_change)  # Quantity of PICNIC_BASKET2
                croissants_qty = pb2_qty * self.pb2_components["CROISSANTS"]
                jams_qty = pb2_qty * self.pb2_components["JAMS"]

                if pb2_spread_change > 0:  # Buying the spread (buy basket, sell components)
                    pb_spread_buy_components["PICNIC_BASKET2"] += pb2_qty
                    pb_spread_sell_components["CROISSANTS"] += croissants_qty
                    pb_spread_sell_components["JAMS"] += jams_qty
                else:  # Selling the spread (sell basket, buy components)
                    pb_spread_sell_components["PICNIC_BASKET2"] += pb2_qty
                    pb_spread_buy_components["CROISSANTS"] += croissants_qty
                    pb_spread_buy_components["JAMS"] += jams_qty

            # Then process PB1 spread
            if pb1_spread_change != 0:
                # PB1 spread = PICNIC_BASKET1 - (6×CROISSANTS + 3×JAMS + 1×DJEMBE)
                pb1_qty = abs(pb1_spread_change)  # Quantity of PICNIC_BASKET1
                croissants_qty = pb1_qty * self.pb1_components["CROISSANTS"]
                jams_qty = pb1_qty * self.pb1_components["JAMS"]
                djembes_qty = pb1_qty * self.pb1_components["DJEMBES"]

                if pb1_spread_change > 0:  # Buying the spread (buy basket, sell components)
                    pb_spread_buy_components["PICNIC_BASKET1"] += pb1_qty
                    pb_spread_sell_components["CROISSANTS"] += croissants_qty
                    pb_spread_sell_components["JAMS"] += jams_qty
                    pb_spread_sell_components["DJEMBES"] += djembes_qty
                else:  # Selling the spread (sell basket, buy components)
                    pb_spread_sell_components["PICNIC_BASKET1"] += pb1_qty
                    pb_spread_buy_components["CROISSANTS"] += croissants_qty
                    pb_spread_buy_components["JAMS"] += jams_qty
                    pb_spread_buy_components["DJEMBES"] += djembes_qty

            logger.print(f"After scaling - PB spread buy components: {pb_spread_buy_components}")
            logger.print(f"After scaling - PB spread sell components: {pb_spread_sell_components}")

        # Step 5: Combine all component changes
        total_buy_components = {product: 0 for product in self.products}
        total_sell_components = {product: 0 for product in self.products}

        # Add custom spread components
        for product in self.products:
            total_buy_components[product] += custom_spread_buy_components[product]
            total_sell_components[product] += custom_spread_sell_components[product]

        # Add PB spread components
        for product in self.products:
            total_buy_components[product] += pb_spread_buy_components[product]
            total_sell_components[product] += pb_spread_sell_components[product]

        logger.print(f"Total buy components: {total_buy_components}")
        logger.print(f"Total sell components: {total_sell_components}")

        # Step 6: Execute orders with proper liquidity taking
        actual_buys = {product: 0 for product in self.products}
        actual_sells = {product: 0 for product in self.products}

        # Process buy orders
        for product, buy_qty in total_buy_components.items():
            if buy_qty > 0:
                if product in state.order_depths:
                    order_depth = state.order_depths[product]

                    # Find available sell orders
                    if len(order_depth.sell_orders) > 0:
                        # Sort sell orders by price (ascending)
                        sorted_asks = sorted(order_depth.sell_orders.items())

                        # Take liquidity from the order book
                        remaining_to_buy = buy_qty
                        for ask_price, ask_volume in sorted_asks:
                            # Volume is negative in sell_orders
                            available_volume = abs(ask_volume)
                            executable_volume = min(remaining_to_buy, available_volume)

                            # Create buy order with integer volume
                            executable_volume_int = int(executable_volume)
                            if executable_volume_int > 0:  # Only create orders with non-zero volume
                                if product not in result:
                                    result[product] = []
                                result[product].append(Order(product, ask_price, executable_volume_int))

                                # Update actual buys with the integer volume
                                actual_buys[product] += executable_volume_int

                                # Update remaining to buy with the integer volume
                                remaining_to_buy -= executable_volume_int

                                # If we've filled the entire order, break
                                if remaining_to_buy <= 0:
                                    break

        # Process sell orders
        for product, sell_qty in total_sell_components.items():
            if sell_qty > 0:
                if product in state.order_depths:
                    order_depth = state.order_depths[product]

                    # Find available buy orders
                    if len(order_depth.buy_orders) > 0:
                        # Sort buy orders by price (descending)
                        sorted_bids = sorted(order_depth.buy_orders.items(), reverse=True)

                        # Take liquidity from the order book
                        remaining_to_sell = sell_qty
                        for bid_price, bid_volume in sorted_bids:
                            available_volume = bid_volume
                            executable_volume = min(remaining_to_sell, available_volume)

                            # Create sell order with integer volume
                            executable_volume_int = int(executable_volume)
                            if executable_volume_int > 0:  # Only create orders with non-zero volume
                                if product not in result:
                                    result[product] = []
                                result[product].append(Order(product, bid_price, -executable_volume_int))

                                # Update actual sells with the integer volume
                                actual_sells[product] += executable_volume_int

                                # Update remaining to sell with the integer volume
                                remaining_to_sell -= executable_volume_int

                                # If we've filled the entire order, break
                                if remaining_to_sell <= 0:
                                    break

        logger.print(f"Actual buys: {actual_buys}")
        logger.print(f"Actual sells: {actual_sells}")

        # Step 7: Update spread positions based on actual executions
        # Calculate execution ratios for custom spread
        custom_spread_execution_ratio = 1.0
        if custom_spread_change != 0:
            # Determine the limiting factor for the custom spread execution
            if custom_spread_change > 0:  # Buying the custom spread
                for product, buy_qty in custom_spread_buy_components.items():
                    if buy_qty > 0:
                        execution_ratio = actual_buys[product] / buy_qty if buy_qty > 0 else 1.0
                        custom_spread_execution_ratio = min(custom_spread_execution_ratio, execution_ratio)

                for product, sell_qty in custom_spread_sell_components.items():
                    if sell_qty > 0:
                        execution_ratio = actual_sells[product] / sell_qty if sell_qty > 0 else 1.0
                        custom_spread_execution_ratio = min(custom_spread_execution_ratio, execution_ratio)
            else:  # Selling the custom spread
                for product, buy_qty in custom_spread_buy_components.items():
                    if buy_qty > 0:
                        execution_ratio = actual_buys[product] / buy_qty if buy_qty > 0 else 1.0
                        custom_spread_execution_ratio = min(custom_spread_execution_ratio, execution_ratio)

                for product, sell_qty in custom_spread_sell_components.items():
                    if sell_qty > 0:
                        execution_ratio = actual_sells[product] / sell_qty if sell_qty > 0 else 1.0
                        custom_spread_execution_ratio = min(custom_spread_execution_ratio, execution_ratio)

        # Calculate execution ratios for PB1 and PB2 spreads
        pb1_spread_execution_ratio = 1.0
        if pb1_spread_change != 0:
            # Determine the limiting factor for the PB1 spread execution
            if pb1_spread_change > 0:  # Buying the PB1 spread
                for product, buy_qty in pb_spread_buy_components.items():
                    if buy_qty > 0 and product in ["PICNIC_BASKET1", "CROISSANTS", "JAMS", "DJEMBES"]:
                        execution_ratio = actual_buys[product] / buy_qty if buy_qty > 0 else 1.0
                        pb1_spread_execution_ratio = min(pb1_spread_execution_ratio, execution_ratio)

                for product, sell_qty in pb_spread_sell_components.items():
                    if sell_qty > 0 and product in ["PICNIC_BASKET1", "CROISSANTS", "JAMS", "DJEMBES"]:
                        execution_ratio = actual_sells[product] / sell_qty if sell_qty > 0 else 1.0
                        pb1_spread_execution_ratio = min(pb1_spread_execution_ratio, execution_ratio)
            else:  # Selling the PB1 spread
                for product, buy_qty in pb_spread_buy_components.items():
                    if buy_qty > 0 and product in ["PICNIC_BASKET1", "CROISSANTS", "JAMS", "DJEMBES"]:
                        execution_ratio = actual_buys[product] / buy_qty if buy_qty > 0 else 1.0
                        pb1_spread_execution_ratio = min(pb1_spread_execution_ratio, execution_ratio)

                for product, sell_qty in pb_spread_sell_components.items():
                    if sell_qty > 0 and product in ["PICNIC_BASKET1", "CROISSANTS", "JAMS", "DJEMBES"]:
                        execution_ratio = actual_sells[product] / sell_qty if sell_qty > 0 else 1.0
                        pb1_spread_execution_ratio = min(pb1_spread_execution_ratio, execution_ratio)

        pb2_spread_execution_ratio = 1.0
        if pb2_spread_change != 0:
            # Determine the limiting factor for the PB2 spread execution
            if pb2_spread_change > 0:  # Buying the PB2 spread
                for product, buy_qty in pb_spread_buy_components.items():
                    if buy_qty > 0 and product in ["PICNIC_BASKET2", "CROISSANTS", "JAMS"]:
                        execution_ratio = actual_buys[product] / buy_qty if buy_qty > 0 else 1.0
                        pb2_spread_execution_ratio = min(pb2_spread_execution_ratio, execution_ratio)

                for product, sell_qty in pb_spread_sell_components.items():
                    if sell_qty > 0 and product in ["PICNIC_BASKET2", "CROISSANTS", "JAMS"]:
                        execution_ratio = actual_sells[product] / sell_qty if sell_qty > 0 else 1.0
                        pb2_spread_execution_ratio = min(pb2_spread_execution_ratio, execution_ratio)
            else:  # Selling the PB2 spread
                for product, buy_qty in pb_spread_buy_components.items():
                    if buy_qty > 0 and product in ["PICNIC_BASKET2", "CROISSANTS", "JAMS"]:
                        execution_ratio = actual_buys[product] / buy_qty if buy_qty > 0 else 1.0
                        pb2_spread_execution_ratio = min(pb2_spread_execution_ratio, execution_ratio)

                for product, sell_qty in pb_spread_sell_components.items():
                    if sell_qty > 0 and product in ["PICNIC_BASKET2", "CROISSANTS", "JAMS"]:
                        execution_ratio = actual_sells[product] / sell_qty if sell_qty > 0 else 1.0
                        pb2_spread_execution_ratio = min(pb2_spread_execution_ratio, execution_ratio)

        # Calculate actual spread position changes
        actual_custom_spread_change = int(custom_spread_change * custom_spread_execution_ratio)
        actual_pb1_spread_change = int(pb1_spread_change * pb1_spread_execution_ratio)
        actual_pb2_spread_change = int(pb2_spread_change * pb2_spread_execution_ratio)

        logger.print(f"Custom spread execution ratio: {custom_spread_execution_ratio}, actual change: {actual_custom_spread_change}")
        logger.print(f"PB1 spread execution ratio: {pb1_spread_execution_ratio}, actual change: {actual_pb1_spread_change}")
        logger.print(f"PB2 spread execution ratio: {pb2_spread_execution_ratio}, actual change: {actual_pb2_spread_change}")

        # Update spread positions
        self.spread_positions["custom_spread"] = int(self.spread_positions["custom_spread"] + actual_custom_spread_change)
        self.spread_positions["pb1_spread"] = int(self.spread_positions["pb1_spread"] + actual_pb1_spread_change)
        self.spread_positions["pb2_spread"] = int(self.spread_positions["pb2_spread"] + actual_pb2_spread_change)

        logger.print(f"Updated spread positions: {self.spread_positions}")

        # Update trader data
        trader_data['spread_positions'] = self.spread_positions
        trader_data['spread_history'] = self.spread_history

        logger.flush(state, result, 0, jsonpickle.encode(trader_data))
        return result, 0, jsonpickle.encode(trader_data)
