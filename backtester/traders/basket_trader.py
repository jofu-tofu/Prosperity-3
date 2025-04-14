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
    Basket Trader for trading PICNIC_BASKET1, PICNIC_BASKET2, CROISSANTS, JAMS, and DJEMBES.

    This trader calculates three spreads:
    1. pb1_spread = PICNIC_BASKET1 - (6×CROISSANTS + 3×JAMS + 1×DJEMBE)
    2. pb2_spread = PICNIC_BASKET2 - (4×CROISSANTS + 2×JAMS)
    3. custom_spread = 3×PICNIC_BASKET2 + 2×DJEMBES - 2×PICNIC_BASKET1

    Trading logic:
    - If spread is above threshold, sell the spread
    - If spread is below negative threshold, buy the spread
    - If spread is within a second defined threshold, neutralize position
    - Special logic for the third spread
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
            "pb1_spread": 25.0,
            "pb2_spread": 25.0,
            "custom_spread": 10.0
        }

        self.exit_thresholds = {
            "pb1_spread": 5.0,
            "pb2_spread": 5.0,
            "custom_spread": 0.0
        }

        # Maximum position size for each spread
        self.max_spread_position = 10

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
        logger.print("Hello.")

        """
        Main trading logic

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

        # Initialize result dictionary for orders
        result = {}

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
            return result, 0, jsonpickle.encode(trader_data)
        # Update price history
        for product, price in current_prices.items():
            self.price_history[product].append(price)
            # Keep only the last 200 prices
            if len(self.price_history[product]) > 200:
                self.price_history[product] = self.price_history[product][-200:]
        # Calculate theoretical basket values
        theoretical_pb1 = (
            6 * current_prices["CROISSANTS"] +
            3 * current_prices["JAMS"] +
            1 * current_prices["DJEMBES"]
        )

        theoretical_pb2 = (
            4 * current_prices["CROISSANTS"] +
            2 * current_prices["JAMS"]
        )

        # Calculate spreads
        pb1_spread = current_prices["PICNIC_BASKET1"] - theoretical_pb1
        pb2_spread = current_prices["PICNIC_BASKET2"] - theoretical_pb2
        custom_spread = (
            3 * current_prices["PICNIC_BASKET2"] +
            2 * current_prices["DJEMBES"] -
            2 * current_prices["PICNIC_BASKET1"]
        )
        logger.print(f"pb1_spread: {pb1_spread}, pb2_spread: {pb2_spread}, custom_spread: {custom_spread}")

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

        # Get current positions
        current_positions = {product: 0 for product in self.products}
        for product in self.products:
            if product in state.position:
                current_positions[product] = state.position[product]

        # Determine trade signals for each spread
        trade_signals = {}

        # PB1 spread trading logic
        if z_scores["pb1_spread"] > self.entry_thresholds["pb1_spread"]:
            # Sell the spread if it's above the entry threshold
            trade_signals["pb1_spread"] = -self.max_spread_position
        elif z_scores["pb1_spread"] < -self.entry_thresholds["pb1_spread"]:
            # Buy the spread if it's below the negative entry threshold
            trade_signals["pb1_spread"] = self.max_spread_position
        elif abs(z_scores["pb1_spread"]) < self.exit_thresholds["pb1_spread"]:
            # Neutralize position if spread is within exit thresholds
            trade_signals["pb1_spread"] = 0
        else:
            # Maintain current position
            trade_signals["pb1_spread"] = self.spread_positions["pb1_spread"]

        # PB2 spread trading logic
        if z_scores["pb2_spread"] > self.entry_thresholds["pb2_spread"]:
            # Sell the spread if it's above the entry threshold
            trade_signals["pb2_spread"] = -self.max_spread_position
        elif z_scores["pb2_spread"] < -self.entry_thresholds["pb2_spread"]:
            # Buy the spread if it's below the negative entry threshold
            trade_signals["pb2_spread"] = self.max_spread_position
        elif abs(z_scores["pb2_spread"]) < self.exit_thresholds["pb2_spread"]:
            # Neutralize position if spread is within exit thresholds
            trade_signals["pb2_spread"] = 0
        else:
            # Maintain current position
            trade_signals["pb2_spread"] = self.spread_positions["pb2_spread"]

        # Custom spread trading logic with special conditions
        if z_scores["custom_spread"] > self.entry_thresholds["custom_spread"]:
            # Sell the spread if it's above the entry threshold
            # But only if one of the basket spreads is not being traded
            # and if pb2 spread is being bought or pb1 spread is being sold
            if (trade_signals["pb1_spread"] == 0 or trade_signals["pb2_spread"] == 0) and \
               (trade_signals["pb2_spread"] > 0 or trade_signals["pb1_spread"] < 0):
                trade_signals["custom_spread"] = -self.max_spread_position
            else:
                trade_signals["custom_spread"] = self.spread_positions["custom_spread"]
        elif z_scores["custom_spread"] < -self.entry_thresholds["custom_spread"]:
            # Buy the spread if it's below the negative entry threshold
            # But only if one of the basket spreads is not being traded
            # and if pb2 spread is being sold or pb1 spread is being bought
            if (trade_signals["pb1_spread"] == 0 or trade_signals["pb2_spread"] == 0) and \
               (trade_signals["pb2_spread"] < 0 or trade_signals["pb1_spread"] > 0):
                trade_signals["custom_spread"] = self.max_spread_position
            else:
                trade_signals["custom_spread"] = self.spread_positions["custom_spread"]
        elif abs(z_scores["custom_spread"]) < self.exit_thresholds["custom_spread"]:
            # Neutralize position if spread is within exit thresholds
            trade_signals["custom_spread"] = 0
        else:
            # Maintain current position
            trade_signals["custom_spread"] = self.spread_positions["custom_spread"]

        # Calculate target positions for each spread
        target_spread_positions = {
            spread_name: trade_signal
            for spread_name, trade_signal in trade_signals.items()
        }

        # Calculate position changes for each spread
        spread_position_changes = {
            spread_name: target_position - self.spread_positions[spread_name]
            for spread_name, target_position in target_spread_positions.items()
        }

        # Separate buy and sell component changes for each product and spread
        buy_component_changes = {product: 0 for product in self.products}
        sell_component_changes = {product: 0 for product in self.products}

        # Track which spread contributes to which component change
        spread_to_component_map = {}

        # Process each spread separately
        for spread_name, change in spread_position_changes.items():
            if change == 0:
                continue

            # Initialize component changes for this spread
            spread_to_component_map[spread_name] = {}

            if spread_name == "pb1_spread":
                # PB1 spread = PICNIC_BASKET1 - (6×CROISSANTS + 3×JAMS + 1×DJEMBE)
                # When spread is positive (PICNIC_BASKET1 > theoretical value):
                # - If we buy the spread: we buy PICNIC_BASKET1, sell components
                # - If we sell the spread: we sell PICNIC_BASKET1, buy components

                # Calculate component quantities for this spread
                pb1_qty = abs(change)  # Quantity of PICNIC_BASKET1
                croissants_qty = pb1_qty * self.pb1_components["CROISSANTS"]
                jams_qty = pb1_qty * self.pb1_components["JAMS"]
                djembes_qty = pb1_qty * self.pb1_components["DJEMBES"]

                if change > 0:  # Buying the spread (buy basket, sell components)
                    buy_component_changes["PICNIC_BASKET1"] += pb1_qty
                    sell_component_changes["CROISSANTS"] += croissants_qty
                    sell_component_changes["JAMS"] += jams_qty
                    sell_component_changes["DJEMBES"] += djembes_qty
                else:  # Selling the spread (sell basket, buy components)
                    sell_component_changes["PICNIC_BASKET1"] += pb1_qty
                    buy_component_changes["CROISSANTS"] += croissants_qty
                    buy_component_changes["JAMS"] += jams_qty
                    buy_component_changes["DJEMBES"] += djembes_qty

                # Track components for this spread
                spread_to_component_map[spread_name]["PICNIC_BASKET1"] = change
                spread_to_component_map[spread_name]["CROISSANTS"] = -change * self.pb1_components["CROISSANTS"]
                spread_to_component_map[spread_name]["JAMS"] = -change * self.pb1_components["JAMS"]
                spread_to_component_map[spread_name]["DJEMBES"] = -change * self.pb1_components["DJEMBES"]

            elif spread_name == "pb2_spread":
                # PB2 spread = PICNIC_BASKET2 - (4×CROISSANTS + 2×JAMS)
                # When spread is positive (PICNIC_BASKET2 > theoretical value):
                # - If we buy the spread: we buy PICNIC_BASKET2, sell components
                # - If we sell the spread: we sell PICNIC_BASKET2, buy components

                # Calculate component quantities for this spread
                pb2_qty = abs(change)  # Quantity of PICNIC_BASKET2
                croissants_qty = pb2_qty * self.pb2_components["CROISSANTS"]
                jams_qty = pb2_qty * self.pb2_components["JAMS"]

                if change > 0:  # Buying the spread (buy basket, sell components)
                    buy_component_changes["PICNIC_BASKET2"] += pb2_qty
                    sell_component_changes["CROISSANTS"] += croissants_qty
                    sell_component_changes["JAMS"] += jams_qty
                else:  # Selling the spread (sell basket, buy components)
                    sell_component_changes["PICNIC_BASKET2"] += pb2_qty
                    buy_component_changes["CROISSANTS"] += croissants_qty
                    buy_component_changes["JAMS"] += jams_qty

                # Track components for this spread
                spread_to_component_map[spread_name]["PICNIC_BASKET2"] = change
                spread_to_component_map[spread_name]["CROISSANTS"] = -change * self.pb2_components["CROISSANTS"]
                spread_to_component_map[spread_name]["JAMS"] = -change * self.pb2_components["JAMS"]

            elif spread_name == "custom_spread":
                # Custom spread = 3×PICNIC_BASKET2 + 2×DJEMBES - 2×PICNIC_BASKET1
                # When spread is positive (3×PB2 + 2×DJEMBES > 2×PB1):
                # - If we buy the spread: we buy PB2 & DJEMBES, sell PB1
                # - If we sell the spread: we sell PB2 & DJEMBES, buy PB1

                # Calculate component quantities for this spread
                pb2_qty = abs(change * self.custom_spread_components["PICNIC_BASKET2"])
                djembes_qty = abs(change * self.custom_spread_components["DJEMBES"])
                pb1_qty = abs(change * self.custom_spread_components["PICNIC_BASKET1"])

                if change > 0:  # Buying the spread
                    # Buy positive components, sell negative components
                    buy_component_changes["PICNIC_BASKET2"] += pb2_qty
                    buy_component_changes["DJEMBES"] += djembes_qty
                    sell_component_changes["PICNIC_BASKET1"] += pb1_qty
                else:  # Selling the spread
                    # Sell positive components, buy negative components
                    sell_component_changes["PICNIC_BASKET2"] += pb2_qty
                    sell_component_changes["DJEMBES"] += djembes_qty
                    buy_component_changes["PICNIC_BASKET1"] += pb1_qty

                # Track components for this spread
                spread_to_component_map[spread_name]["PICNIC_BASKET2"] = change * self.custom_spread_components["PICNIC_BASKET2"]
                spread_to_component_map[spread_name]["DJEMBES"] = change * self.custom_spread_components["DJEMBES"]
                spread_to_component_map[spread_name]["PICNIC_BASKET1"] = change * self.custom_spread_components["PICNIC_BASKET1"]

        logger.print(f"Buy component changes: {buy_component_changes}")
        logger.print(f"Sell component changes: {sell_component_changes}")

        # Check position limits for buys (current position + buys cannot exceed max position)
        buy_scale_factor = 1.0
        for product, buy_qty in buy_component_changes.items():
            if buy_qty > 0:  # Only check products we're buying
                max_allowed_buy = self.position_limits[product] - current_positions[product]
                if buy_qty > max_allowed_buy:
                    product_scale_factor = max_allowed_buy / buy_qty if buy_qty > 0 else 1.0
                    buy_scale_factor = min(buy_scale_factor, product_scale_factor)
                    logger.print(f"Buy limit would be violated for {product}: current={current_positions[product]}, buy={buy_qty}, max_allowed={max_allowed_buy}, scale_factor={product_scale_factor}")

        # Check position limits for sells (current position - sells cannot go below negative max position)
        sell_scale_factor = 1.0
        for product, sell_qty in sell_component_changes.items():
            if sell_qty > 0:  # Only check products we're selling
                max_allowed_sell = current_positions[product] + self.position_limits[product]
                if sell_qty > max_allowed_sell:
                    product_scale_factor = max_allowed_sell / sell_qty if sell_qty > 0 else 1.0
                    sell_scale_factor = min(sell_scale_factor, product_scale_factor)
                    logger.print(f"Sell limit would be violated for {product}: current={current_positions[product]}, sell={sell_qty}, max_allowed={max_allowed_sell}, scale_factor={product_scale_factor}")

        # Apply scaling to spread position changes
        min_scale_factor = min(buy_scale_factor, sell_scale_factor)
        if min_scale_factor < 1.0:
            logger.print(f"Scaling all changes by factor: {min_scale_factor}")

            # Scale the spread position changes
            for spread_name in spread_position_changes:
                spread_position_changes[spread_name] *= min_scale_factor

            # Recalculate component changes with scaled spread positions
            buy_component_changes = {product: 0 for product in self.products}
            sell_component_changes = {product: 0 for product in self.products}

            # Process each spread again with scaled changes
            for spread_name, change in spread_position_changes.items():
                if change == 0:
                    continue

                if spread_name == "pb1_spread":
                    # PB1 spread = PICNIC_BASKET1 - (6×CROISSANTS + 3×JAMS + 1×DJEMBE)
                    # Calculate component quantities for this spread
                    pb1_qty = abs(change)  # Quantity of PICNIC_BASKET1
                    croissants_qty = pb1_qty * self.pb1_components["CROISSANTS"]
                    jams_qty = pb1_qty * self.pb1_components["JAMS"]
                    djembes_qty = pb1_qty * self.pb1_components["DJEMBES"]

                    if change > 0:  # Buying the spread (buy basket, sell components)
                        buy_component_changes["PICNIC_BASKET1"] += pb1_qty
                        sell_component_changes["CROISSANTS"] += croissants_qty
                        sell_component_changes["JAMS"] += jams_qty
                        sell_component_changes["DJEMBES"] += djembes_qty
                    else:  # Selling the spread (sell basket, buy components)
                        sell_component_changes["PICNIC_BASKET1"] += pb1_qty
                        buy_component_changes["CROISSANTS"] += croissants_qty
                        buy_component_changes["JAMS"] += jams_qty
                        buy_component_changes["DJEMBES"] += djembes_qty

                elif spread_name == "pb2_spread":
                    # PB2 spread = PICNIC_BASKET2 - (4×CROISSANTS + 2×JAMS)
                    # Calculate component quantities for this spread
                    pb2_qty = abs(change)  # Quantity of PICNIC_BASKET2
                    croissants_qty = pb2_qty * self.pb2_components["CROISSANTS"]
                    jams_qty = pb2_qty * self.pb2_components["JAMS"]

                    if change > 0:  # Buying the spread (buy basket, sell components)
                        buy_component_changes["PICNIC_BASKET2"] += pb2_qty
                        sell_component_changes["CROISSANTS"] += croissants_qty
                        sell_component_changes["JAMS"] += jams_qty
                    else:  # Selling the spread (sell basket, buy components)
                        sell_component_changes["PICNIC_BASKET2"] += pb2_qty
                        buy_component_changes["CROISSANTS"] += croissants_qty
                        buy_component_changes["JAMS"] += jams_qty

                elif spread_name == "custom_spread":
                    # Custom spread = 3×PICNIC_BASKET2 + 2×DJEMBES - 2×PICNIC_BASKET1
                    # Calculate component quantities for this spread
                    pb2_qty = abs(change * self.custom_spread_components["PICNIC_BASKET2"])
                    djembes_qty = abs(change * self.custom_spread_components["DJEMBES"])
                    pb1_qty = abs(change * self.custom_spread_components["PICNIC_BASKET1"])

                    if change > 0:  # Buying the spread
                        # Buy positive components, sell negative components
                        buy_component_changes["PICNIC_BASKET2"] += pb2_qty
                        buy_component_changes["DJEMBES"] += djembes_qty
                        sell_component_changes["PICNIC_BASKET1"] += pb1_qty
                    else:  # Selling the spread
                        # Sell positive components, buy negative components
                        sell_component_changes["PICNIC_BASKET2"] += pb2_qty
                        sell_component_changes["DJEMBES"] += djembes_qty
                        buy_component_changes["PICNIC_BASKET1"] += pb1_qty

            logger.print(f"After scaling - Buy component changes: {buy_component_changes}")
            logger.print(f"After scaling - Sell component changes: {sell_component_changes}")

        # Combine buy and sell changes into final component changes
        component_changes = {product: 0 for product in self.products}
        for product in self.products:
            if buy_component_changes[product] > 0:
                component_changes[product] += buy_component_changes[product]
            if sell_component_changes[product] > 0:
                component_changes[product] -= sell_component_changes[product]

        # Generate orders based on buy and sell component changes and available liquidity
        actual_buys = {product: 0 for product in self.products}
        actual_sells = {product: 0 for product in self.products}

        # Process buy orders
        for product, buy_qty in buy_component_changes.items():
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

                            if executable_volume > 0:
                                # Create buy order
                                if product not in result:
                                    result[product] = []
                                result[product].append(Order(product, ask_price, executable_volume))

                                # Update actual buys
                                actual_buys[product] += executable_volume
                                remaining_to_buy -= executable_volume

                                # If we've filled the entire order, break
                                if remaining_to_buy <= 0:
                                    break

        # Process sell orders
        for product, sell_qty in sell_component_changes.items():
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

                            if executable_volume > 0:
                                # Create sell order
                                if product not in result:
                                    result[product] = []
                                result[product].append(Order(product, bid_price, -executable_volume))

                                # Update actual sells
                                actual_sells[product] += executable_volume
                                remaining_to_sell -= executable_volume

                                # If we've filled the entire order, break
                                if remaining_to_sell <= 0:
                                    break

        logger.print(f"Actual buys: {actual_buys}")
        logger.print(f"Actual sells: {actual_sells}")

        # Calculate actual component changes from buys and sells
        actual_component_changes = {product: 0 for product in self.products}
        for product in self.products:
            actual_component_changes[product] = actual_buys[product] - actual_sells[product]

        # Calculate actual spread position changes based on executed component changes
        actual_spread_position_changes = {spread_name: 0 for spread_name in self.spread_positions}

        # Create a mapping of which products are involved in which spreads
        product_to_spreads = {}
        for product in self.products:
            product_to_spreads[product] = []
            for spread_name, components in spread_to_component_map.items():
                if product in components:
                    product_to_spreads[product].append(spread_name)

        # Calculate actual executed quantities for each spread
        # We need to determine how much of each spread was actually executed based on the component executions

        # First, calculate the execution ratio for each product (how much of the intended order was executed)
        execution_ratios = {}
        for product in self.products:
            # Calculate buy execution ratio
            if buy_component_changes[product] > 0:
                buy_ratio_key = f"{product}_buy"
                execution_ratios[buy_ratio_key] = actual_buys[product] / buy_component_changes[product]
                logger.print(f"{product} buy execution ratio: {execution_ratios[buy_ratio_key]} (executed {actual_buys[product]} of {buy_component_changes[product]})")
            else:
                execution_ratios[f"{product}_buy"] = 1.0

            # Calculate sell execution ratio
            if sell_component_changes[product] > 0:
                sell_ratio_key = f"{product}_sell"
                execution_ratios[sell_ratio_key] = actual_sells[product] / sell_component_changes[product]
                logger.print(f"{product} sell execution ratio: {execution_ratios[sell_ratio_key]} (executed {actual_sells[product]} of {sell_component_changes[product]})")
            else:
                execution_ratios[f"{product}_sell"] = 1.0

        # For each spread, find the minimum execution ratio across all its components
        # This represents the maximum amount of the spread that could have been executed
        spread_execution_ratios = {}
        for spread_name, components in spread_to_component_map.items():
            min_ratio = 1.0
            limiting_product = ""

            for product, change in components.items():
                # Determine if we're buying or selling this product for this spread
                is_buying = False

                if spread_name == "pb1_spread":
                    # For pb1_spread, we buy PICNIC_BASKET1 and sell components when change > 0
                    if product == "PICNIC_BASKET1":
                        is_buying = (change > 0)  # Buy basket when change > 0
                    else:
                        is_buying = (change < 0)  # Buy components when change < 0

                elif spread_name == "pb2_spread":
                    # For pb2_spread, we buy PICNIC_BASKET2 and sell components when change > 0
                    if product == "PICNIC_BASKET2":
                        is_buying = (change > 0)  # Buy basket when change > 0
                    else:
                        is_buying = (change < 0)  # Buy components when change < 0

                elif spread_name == "custom_spread":
                    # For custom_spread = 3×PB2 + 2×DJEMBES - 2×PB1
                    # When change > 0, we buy positive components (PB2, DJEMBES) and sell negative components (PB1)
                    if product in ["PICNIC_BASKET2", "DJEMBES"]:
                        is_buying = (change > 0)  # Buy PB2 & DJEMBES when change > 0
                    else:  # PICNIC_BASKET1
                        is_buying = (change < 0)  # Buy PB1 when change < 0

                # Get the execution ratio for this product
                ratio_key = f"{product}_buy" if is_buying else f"{product}_sell"
                product_ratio = execution_ratios[ratio_key]

                # Update the minimum ratio if this product has a lower execution ratio
                if product_ratio < min_ratio:
                    min_ratio = product_ratio
                    limiting_product = product

            spread_execution_ratios[spread_name] = min_ratio
            if min_ratio < 1.0:
                logger.print(f"{spread_name} execution limited by {limiting_product} with ratio {min_ratio}")

        # Calculate actual spread position changes based on execution ratios
        for spread_name, change in spread_position_changes.items():
            actual_spread_position_changes[spread_name] = change * spread_execution_ratios[spread_name]

        logger.print(f"Spread execution ratios: {spread_execution_ratios}")
        logger.print(f"Actual spread position changes: {actual_spread_position_changes}")

        # Update spread positions based on actual changes
        for spread_name, change in actual_spread_position_changes.items():
            self.spread_positions[spread_name] += change

        logger.print(f"Actual component changes: {actual_component_changes}")
        logger.print(f"Actual spread position changes: {actual_spread_position_changes}")
        logger.print(f"Updated spread positions: {self.spread_positions}")

        # Update trader data
        trader_data['spread_positions'] = self.spread_positions
        trader_data['spread_history'] = self.spread_history
        logger.flush(state, result, 0, trader_data)

        return result, 0, jsonpickle.encode(trader_data)
