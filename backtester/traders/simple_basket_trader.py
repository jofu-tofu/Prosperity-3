from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
from typing import Dict, List, Tuple, Any
import numpy as np  # Used for calculating standard deviation
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
    Simple Basket Trader that only trades PB1 and PB2 spreads.

    This trader calculates two spreads:
    1. pb1_spread = PICNIC_BASKET1 - (6×CROISSANTS + 3×JAMS + 1×DJEMBE)
    2. pb2_spread = PICNIC_BASKET2 - (4×CROISSANTS + 2×JAMS)

    Trading logic:
    - If spread is above threshold, sell the spread
    - If spread is below negative threshold, buy the spread
    - If spread is within exit threshold, neutralize position
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

        # Thresholds for trading
        self.entry_thresholds = {
            "pb1_spread": 50.0,
            "pb2_spread": 30.0
        }

        self.exit_thresholds = {
            "pb1_spread": 5.0,
            "pb2_spread": 5.0
        }

        # Maximum position size for each spread (as integers)
        self.max_spread_positions = {
            "pb1_spread": 60,  # Max position size for PB1 spread
            "pb2_spread": 100   # Max position size for PB2 spread
        }

        # Initialize price history for calculating z-scores
        self.price_history = {product: [] for product in self.products}
        self.spread_history = {
            "pb1_spread": [],
            "pb2_spread": []
        }

        # Initialize spread positions (as integers)
        self.spread_positions = {
            "pb1_spread": 0,
            "pb2_spread": 0
        }

        # Window size for calculating standard deviation
        self.window_size = 50

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
        logger.print("Simple Basket Trader starting...")

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

        logger.print(f"PB1 spread: {pb1_spread}, PB2 spread: {pb2_spread}")
        logger.print(f"Current positions: {state.position}")
        logger.print(f"Current spread positions: {self.spread_positions}")

        # Update spread history
        self.spread_history["pb1_spread"].append(pb1_spread)
        self.spread_history["pb2_spread"].append(pb2_spread)

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

        # Determine trade signals for each spread
        trade_signals = {}

        # PB1 spread trading logic
        if z_scores["pb1_spread"] > self.entry_thresholds["pb1_spread"]:
            # Sell the spread if it's above the entry threshold
            trade_signals["pb1_spread"] = -self.max_spread_positions["pb1_spread"]
        elif z_scores["pb1_spread"] < -self.entry_thresholds["pb1_spread"]:
            # Buy the spread if it's below the negative entry threshold
            trade_signals["pb1_spread"] = self.max_spread_positions["pb1_spread"]
        elif abs(z_scores["pb1_spread"]) < self.exit_thresholds["pb1_spread"]:
            # Neutralize position if spread is within exit thresholds
            trade_signals["pb1_spread"] = 0
        else:
            # In the middle zone - neither entry nor exit - hold current position without trading
            # We set the target position equal to the current position, which means no new orders
            trade_signals["pb1_spread"] = None  # None means "do nothing"

        # PB2 spread trading logic
        if z_scores["pb2_spread"] > self.entry_thresholds["pb2_spread"]:
            # Sell the spread if it's above the entry threshold
            trade_signals["pb2_spread"] = -self.max_spread_positions["pb2_spread"]
        elif z_scores["pb2_spread"] < -self.entry_thresholds["pb2_spread"]:
            # Buy the spread if it's below the negative entry threshold
            trade_signals["pb2_spread"] = self.max_spread_positions["pb2_spread"]
        elif abs(z_scores["pb2_spread"]) < self.exit_thresholds["pb2_spread"]:
            # Neutralize position if spread is within exit thresholds
            trade_signals["pb2_spread"] = 0
        else:
            # In the middle zone - neither entry nor exit - hold current position without trading
            # We set the target position equal to the current position, which means no new orders
            trade_signals["pb2_spread"] = None  # None means "do nothing"

        logger.print(f"Trade signals: {trade_signals}")

        # Calculate target positions for each spread
        target_spread_positions = {}
        for spread_name, trade_signal in trade_signals.items():
            if trade_signal is None:
                # If signal is None, maintain current position (no change)
                target_spread_positions[spread_name] = self.spread_positions[spread_name]
            else:
                # Otherwise, use the trade signal as the target position
                target_spread_positions[spread_name] = int(trade_signal)

        # Calculate position changes for each spread
        spread_position_changes = {}
        for spread_name, target_position in target_spread_positions.items():
            # Calculate the change from current position to target position
            change = int(target_position - self.spread_positions[spread_name])
            spread_position_changes[spread_name] = change

        logger.print(f"Spread position changes: {spread_position_changes}")

        # Separate buy and sell component changes for each product and spread
        buy_component_changes = {product: 0 for product in self.products}
        sell_component_changes = {product: 0 for product in self.products}

        # Process each spread separately
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

        # logger.print(f"Buy component changes: {buy_component_changes}")
        # logger.print(f"Sell component changes: {sell_component_changes}")

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

            # logger.print(f"After scaling - Buy component changes: {buy_component_changes}")
            # logger.print(f"After scaling - Sell component changes: {sell_component_changes}")

        # Combine buy and sell changes into final component changes
        component_changes = {product: 0 for product in self.products}
        for product in self.products:
            if buy_component_changes[product] > 0:
                component_changes[product] += buy_component_changes[product]
            if sell_component_changes[product] > 0:
                component_changes[product] -= sell_component_changes[product]

        # logger.print(f"Final component changes: {component_changes}")

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
                                # Create buy order with integer volume
                                executable_volume_int = int(executable_volume)
                                if executable_volume_int > 0:  # Only create orders with non-zero volume
                                    if product not in result:
                                        result[product] = []
                                    result[product].append(Order(product, ask_price, executable_volume_int))

                                    # Update remaining to buy with the integer volume
                                    remaining_to_buy -= executable_volume_int

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

        # logger.print(f"Actual buys: {actual_buys}")
        # logger.print(f"Actual sells: {actual_sells}")

        # Calculate actual component changes from buys and sells
        actual_component_changes = {product: 0 for product in self.products}
        for product in self.products:
            actual_component_changes[product] = actual_buys[product] - actual_sells[product]

        # logger.print(f"Actual component changes: {actual_component_changes}")

        # Calculate actual spread position changes based on executed orders
        actual_spread_position_changes = {spread_name: 0 for spread_name in self.spread_positions}

        # For PB1 spread, the position change is based on the PICNIC_BASKET1 execution
        if spread_position_changes["pb1_spread"] != 0:
            if spread_position_changes["pb1_spread"] > 0:  # Buying the spread
                execution_ratio = actual_buys["PICNIC_BASKET1"] / buy_component_changes["PICNIC_BASKET1"] if buy_component_changes["PICNIC_BASKET1"] > 0 else 0
            else:  # Selling the spread
                execution_ratio = actual_sells["PICNIC_BASKET1"] / sell_component_changes["PICNIC_BASKET1"] if sell_component_changes["PICNIC_BASKET1"] > 0 else 0

            actual_spread_position_changes["pb1_spread"] = int(spread_position_changes["pb1_spread"] * execution_ratio)

        # For PB2 spread, the position change is based on the PICNIC_BASKET2 execution
        if spread_position_changes["pb2_spread"] != 0:
            if spread_position_changes["pb2_spread"] > 0:  # Buying the spread
                execution_ratio = actual_buys["PICNIC_BASKET2"] / buy_component_changes["PICNIC_BASKET2"] if buy_component_changes["PICNIC_BASKET2"] > 0 else 0
            else:  # Selling the spread
                execution_ratio = actual_sells["PICNIC_BASKET2"] / sell_component_changes["PICNIC_BASKET2"] if sell_component_changes["PICNIC_BASKET2"] > 0 else 0

            actual_spread_position_changes["pb2_spread"] = int(spread_position_changes["pb2_spread"] * execution_ratio)

        logger.print(f"Actual spread position changes: {actual_spread_position_changes}")

        # Update spread positions based on actual changes (ensure they are integers)
        for spread_name, change in actual_spread_position_changes.items():
            self.spread_positions[spread_name] = int(self.spread_positions[spread_name] + change)

        logger.print(f"Updated spread positions: {self.spread_positions}")

        # Update trader data
        trader_data['spread_positions'] = self.spread_positions
        trader_data['spread_history'] = self.spread_history

        logger.flush(state, result, 0, jsonpickle.encode(trader_data))
        return result, 0, jsonpickle.encode(trader_data)
