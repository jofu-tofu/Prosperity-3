from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
from typing import Dict, List, Tuple, Any
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
    Multi-Spread Tracker that tracks all three spreads (PB1, PB2, and custom spread)
    but only trades the custom spread (3×PB2 + 2×DJEMBES - 2×PB1).
    """
    def __init__(self):
        # Products to trade
        self.products = ["PICNIC_BASKET1", "PICNIC_BASKET2", "CROISSANTS", "JAMS", "DJEMBES"]

        # Position limits for each product
        self.position_limits = {
            "PICNIC_BASKET1": 60,
            "PICNIC_BASKET2": 100,
            "CROISSANTS": 250,
            "JAMS": 350,
            "DJEMBES": 60
        }

        # Spread components
        # PB1 spread = PICNIC_BASKET1 - (6×CROISSANTS + 3×JAMS + 1×DJEMBE)
        self.pb1_components = {
            "PICNIC_BASKET1": 1,
            "CROISSANTS": -6,
            "JAMS": -3,
            "DJEMBES": -1
        }

        # PB2 spread = PICNIC_BASKET2 - (4×CROISSANTS + 2×JAMS)
        self.pb2_components = {
            "PICNIC_BASKET2": 1,
            "CROISSANTS": -4,
            "JAMS": -2
        }

        # Custom spread = 3×PB2 + 2×DJEMBES - 2×PB1
        self.custom_spread_components = {
            "PICNIC_BASKET2": 3,
            "DJEMBES": 2,
            "PICNIC_BASKET1": -2
        }

        # Thresholds for trading each spread
        self.entry_thresholds = {
            "pb1_spread": 5.5,
            "pb2_spread": 5,
            "custom_spread": 5.5
        }

        self.exit_thresholds = {
            "pb1_spread": 1.0,
            "pb2_spread": 1,
            "custom_spread": 1.25
        }

        # Maximum position size for each spread
        self.max_spread_positions = {
            "pb1_spread": 0,  # Set to non-zero to enable trading PB1 spread
            "pb2_spread": 20,  # Set to 0 to disable trading PB2 spread
            "custom_spread": 45
        }

        # Initialize spread positions
        self.spread_positions = {
            "pb1_spread": 0,
            "pb2_spread": 0,
            "custom_spread": 0
        }

        # Fixed standard deviation value to use for z-score calculation
        self.fixed_std_dev = 25

        # Price adjustment parameters (how much we're willing to pay to enter/exit)
        # These values adjust the price we're willing to pay relative to the mid price
        # For mid price ending in .5:
        #   - adjustment of 0 means +0.5/-0.5 from mid price
        #   - adjustment of 1 means +1.5/-1.5 from mid price
        # For integer mid price:
        #   - adjustment is applied directly
        #
        # When to use each adjustment:
        # - entry_price_adjustment: Used when |z_score| >= entry_threshold (entering a new position)
        # - exit_price_adjustment: Used when |z_score| < exit_threshold (exiting a position)
        # - For z-scores in between, use exit_price_adjustment when closing positions

        # Entry price adjustments for each spread
        self.entry_price_adjustments = {
            "pb1_spread": 1,  # Willing to pay 0 units worse than mid price to enter a position
            "pb2_spread": 1,  # Willing to pay 0 units worse than mid price to enter a position
            "custom_spread": 5   # Willing to pay 0 units worse than mid price to enter a position
        }

        # Exit price adjustments for each spread
        self.exit_price_adjustments = {
            "pb1_spread": 1,  # Willing to pay 1 unit worse than mid price to exit a position
            "pb2_spread": 0,  # Willing to pay 1 unit worse than mid price to exit a position
            "custom_spread": 1   # Willing to pay 1 unit worse than mid price to exit a position
        }

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
        logger.print("Multi-Spread Tracker starting...")

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

        # Get current prices for all products
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

        # Check if we have prices for all products
        if len(current_prices) != len(self.products):
            # Not enough data to trade
            logger.print("Missing prices for some products, skipping trading")
            return result, 0, jsonpickle.encode(trader_data)

        # Calculate theoretical basket values
        theoretical_pb1 = sum(qty * current_prices[product] for product, qty in self.pb1_components.items())
        theoretical_pb2 = sum(qty * current_prices[product] for product, qty in self.pb2_components.items())

        # Calculate spreads
        pb1_spread = theoretical_pb1
        pb2_spread = theoretical_pb2
        custom_spread = sum(qty * current_prices[product] for product, qty in self.custom_spread_components.items())

        # Calculate z-scores using fixed standard deviation
        z_scores = {
            "pb1_spread": pb1_spread / self.fixed_std_dev,
            "pb2_spread": pb2_spread / self.fixed_std_dev,
            "custom_spread": custom_spread / self.fixed_std_dev
        }

        logger.print(f"PB1 spread: {pb1_spread}, z-score: {z_scores['pb1_spread']}")
        logger.print(f"PB2 spread: {pb2_spread}, z-score: {z_scores['pb2_spread']}")
        logger.print(f"Custom spread: {custom_spread}, z-score: {z_scores['custom_spread']}")
        logger.print(f"Current positions: {state.position}")
        logger.print(f"Current spread positions: {self.spread_positions}")

        # Get current positions
        current_positions = {product: 0 for product in self.products}
        for product in self.products:
            if product in state.position:
                current_positions[product] = state.position[product]

        # Only generate trading signals for the custom spread
        custom_z_score = z_scores["custom_spread"]
        spread_signal = None

        # Get thresholds for the custom spread
        custom_entry_threshold = self.entry_thresholds["custom_spread"]
        custom_exit_threshold = self.exit_thresholds["custom_spread"]
        custom_max_position = self.max_spread_positions["custom_spread"]

        if custom_z_score > custom_entry_threshold:
            # Sell the spread if it's above the entry threshold
            spread_signal = -custom_max_position
        elif custom_z_score < -custom_entry_threshold:
            # Buy the spread if it's below the negative entry threshold
            spread_signal = custom_max_position
        elif abs(custom_z_score) < custom_exit_threshold:
            # Neutralize position if spread is within exit thresholds
            spread_signal = 0
        else:
            # In the middle zone - neither entry nor exit - hold current position without trading
            spread_signal = None

        logger.print(f"Custom spread signal: {spread_signal}")

        # Calculate target position for custom spread
        if spread_signal is None:
            # If signal is None, maintain current position (no change)
            target_spread_position = self.spread_positions["custom_spread"]
        else:
            # Otherwise, use the trade signal as the target position
            target_spread_position = int(spread_signal)

        # Calculate position change for custom spread
        spread_change = int(target_spread_position - self.spread_positions["custom_spread"])

        logger.print(f"Custom spread change: {spread_change}")

        # If no change in position, return early
        if spread_change == 0:
            logger.print("No change in custom spread position, skipping trading")
            return result, 0, jsonpickle.encode(trader_data)

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

        logger.print(f"Buy components: {buy_components}")
        logger.print(f"Sell components: {sell_components}")

        # Step 1: Check if position limits would be violated
        scale_factor = 1.0

        # First, calculate the minimum scale factor needed to respect all position limits
        for product in self.products:
            # Check buy limit
            if buy_components[product] > 0:
                max_buy = self.position_limits[product] - current_positions[product]
                if buy_components[product] > max_buy:
                    product_scale_factor = max_buy / buy_components[product] if buy_components[product] > 0 else 0
                    scale_factor = min(scale_factor, product_scale_factor)
                    logger.print(f"Buy limit for {product}: need scale factor {product_scale_factor}")

            # Check sell limit
            if sell_components[product] > 0:
                max_sell = current_positions[product] + self.position_limits[product]
                if sell_components[product] > max_sell:
                    product_scale_factor = max_sell / sell_components[product] if sell_components[product] > 0 else 0
                    scale_factor = min(scale_factor, product_scale_factor)
                    logger.print(f"Sell limit for {product}: need scale factor {product_scale_factor}")

        # If we need to scale down, calculate the integer number of spread units we can trade
        if scale_factor < 1.0:
            # Calculate how many integer spread units we can trade
            max_spread_units = int(abs(spread_change) * scale_factor)

            # Recalculate scale factor based on integer spread units
            if abs(spread_change) > 0:
                scale_factor = max_spread_units / abs(spread_change)

            logger.print(f"Scaling down to {max_spread_units} spread units (scale factor: {scale_factor})")

            # Scale the spread change to an integer number of units
            if spread_change > 0:
                spread_change = max_spread_units
            else:
                spread_change = -max_spread_units

            # Recalculate component quantities based on the integer spread change
            buy_components = {product: 0 for product in self.products}
            sell_components = {product: 0 for product in self.products}

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

        logger.print(f"After position limit check - Buy components: {buy_components}")
        logger.print(f"After position limit check - Sell components: {sell_components}")

        # Step 2: Check available liquidity in the orderbook
        liquidity_scale_factor = 1.0

        # Check buy liquidity
        for product, buy_qty in buy_components.items():
            if buy_qty > 0:
                if product in state.order_depths:
                    order_depth = state.order_depths[product]

                    # Find available sell orders
                    if len(order_depth.sell_orders) > 0:
                        # Sort sell orders by price (ascending)
                        sorted_asks = sorted(order_depth.sell_orders.items())

                        # Determine price adjustment based on z-score thresholds
                        # Get thresholds and price adjustments for the custom spread
                        custom_entry_threshold = self.entry_thresholds["custom_spread"]
                        custom_exit_threshold = self.exit_thresholds["custom_spread"]
                        custom_entry_price_adjustment = self.entry_price_adjustments["custom_spread"]
                        custom_exit_price_adjustment = self.exit_price_adjustments["custom_spread"]

                        if abs(custom_z_score) >= custom_entry_threshold:  # Entering a position
                            price_adjustment = custom_entry_price_adjustment
                        elif abs(custom_z_score) < custom_exit_threshold:  # Exiting a position
                            price_adjustment = custom_exit_price_adjustment
                        else:  # In the middle zone
                            # Use exit price adjustment when closing positions
                            if (custom_z_score > 0 and self.spread_positions["custom_spread"] > 0) or \
                               (custom_z_score < 0 and self.spread_positions["custom_spread"] < 0):
                                price_adjustment = custom_exit_price_adjustment
                            else:
                                price_adjustment = custom_entry_price_adjustment

                        # Calculate the maximum price we're willing to pay
                        mid_price = (max(order_depth.buy_orders.keys()) + min(order_depth.sell_orders.keys())) / 2

                        # Handle special case when mid price ends in .5
                        if mid_price % 1 == 0.5:
                            # For mid price ending in .5, adjustment of 0 means +0.5, adjustment of 1 means +1.5
                            max_buy_price = int(mid_price + 0.5) + price_adjustment
                        else:
                            # For integer mid price, adjustment is applied directly
                            max_buy_price = int(mid_price) + price_adjustment

                        # Calculate how much we can buy at or below our max price
                        available_to_buy = 0
                        for ask_price, ask_volume in sorted_asks:
                            if ask_price <= max_buy_price:
                                available_to_buy += abs(ask_volume)  # Volume is negative in sell_orders
                            else:
                                break  # Stop once we exceed our max price

                        # Calculate scale factor based on available liquidity
                        if available_to_buy < buy_qty:
                            product_liquidity_factor = available_to_buy / buy_qty if buy_qty > 0 else 0
                            liquidity_scale_factor = min(liquidity_scale_factor, product_liquidity_factor)
                            logger.print(f"Buy liquidity for {product}: need scale factor {product_liquidity_factor} (available: {available_to_buy}, needed: {buy_qty})")
                    else:
                        # No sell orders available
                        logger.print(f"No sell orders available for {product}")
                        liquidity_scale_factor = 0
                else:
                    # Product not in order depths
                    logger.print(f"No order depth available for {product}")
                    liquidity_scale_factor = 0

        # Check sell liquidity
        for product, sell_qty in sell_components.items():
            if sell_qty > 0:
                if product in state.order_depths:
                    order_depth = state.order_depths[product]

                    # Find available buy orders
                    if len(order_depth.buy_orders) > 0:
                        # Sort buy orders by price (descending)
                        sorted_bids = sorted(order_depth.buy_orders.items(), reverse=True)

                        # Determine price adjustment based on z-score thresholds
                        # Get thresholds and price adjustments for the custom spread
                        custom_entry_threshold = self.entry_thresholds["custom_spread"]
                        custom_exit_threshold = self.exit_thresholds["custom_spread"]
                        custom_entry_price_adjustment = self.entry_price_adjustments["custom_spread"]
                        custom_exit_price_adjustment = self.exit_price_adjustments["custom_spread"]

                        if abs(custom_z_score) >= custom_entry_threshold:  # Entering a position
                            price_adjustment = custom_entry_price_adjustment
                        elif abs(custom_z_score) < custom_exit_threshold:  # Exiting a position
                            price_adjustment = custom_exit_price_adjustment
                        else:  # In the middle zone
                            # Use exit price adjustment when closing positions
                            if (custom_z_score > 0 and self.spread_positions["custom_spread"] > 0) or \
                               (custom_z_score < 0 and self.spread_positions["custom_spread"] < 0):
                                price_adjustment = custom_exit_price_adjustment
                            else:
                                price_adjustment = custom_entry_price_adjustment

                        # Calculate the minimum price we're willing to accept
                        mid_price = (max(order_depth.buy_orders.keys()) + min(order_depth.sell_orders.keys())) / 2

                        # Handle special case when mid price ends in .5
                        if mid_price % 1 == 0.5:
                            # For mid price ending in .5, adjustment of 0 means -0.5, adjustment of 1 means -1.5
                            min_sell_price = int(mid_price - 0.5) - price_adjustment
                        else:
                            # For integer mid price, adjustment is applied directly
                            min_sell_price = int(mid_price) - price_adjustment

                        # Calculate how much we can sell at or above our min price
                        available_to_sell = 0
                        for bid_price, bid_volume in sorted_bids:
                            if bid_price >= min_sell_price:
                                available_to_sell += bid_volume
                            else:
                                break  # Stop once we go below our min price

                        # Calculate scale factor based on available liquidity
                        if available_to_sell < sell_qty:
                            product_liquidity_factor = available_to_sell / sell_qty if sell_qty > 0 else 0
                            liquidity_scale_factor = min(liquidity_scale_factor, product_liquidity_factor)
                            logger.print(f"Sell liquidity for {product}: need scale factor {product_liquidity_factor} (available: {available_to_sell}, needed: {sell_qty})")
                    else:
                        # No buy orders available
                        logger.print(f"No buy orders available for {product}")
                        liquidity_scale_factor = 0
                else:
                    # Product not in order depths
                    logger.print(f"No order depth available for {product}")
                    liquidity_scale_factor = 0

        # If we need to scale down due to liquidity constraints
        if liquidity_scale_factor < 1.0:
            # Calculate how many integer spread units we can trade based on liquidity
            max_liquidity_spread_units = int(abs(spread_change) * liquidity_scale_factor)

            # Recalculate scale factor based on integer spread units
            if abs(spread_change) > 0:
                liquidity_scale_factor = max_liquidity_spread_units / abs(spread_change)

            logger.print(f"Scaling down due to liquidity constraints to {max_liquidity_spread_units} spread units (scale factor: {liquidity_scale_factor})")

            # Scale the spread change to an integer number of units
            if spread_change > 0:
                spread_change = max_liquidity_spread_units
            else:
                spread_change = -max_liquidity_spread_units

            # Recalculate component quantities based on the integer spread change
            buy_components = {product: 0 for product in self.products}
            sell_components = {product: 0 for product in self.products}

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

        logger.print(f"After liquidity check - Buy components: {buy_components}")
        logger.print(f"After liquidity check - Sell components: {sell_components}")

        # Execute orders with proper liquidity taking
        actual_buys = {product: 0 for product in self.products}
        actual_sells = {product: 0 for product in self.products}

        # Process buy orders
        for product, buy_qty in buy_components.items():
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
                                # We've already checked that this order is within our price limits during the liquidity check
                                # So we can just use the ask_price directly
                                # Ensure the price is an integer
                                adjusted_price = int(ask_price)

                                if product not in result:
                                    result[product] = []
                                result[product].append(Order(product, adjusted_price, executable_volume_int))

                                # Update actual buys with the integer volume
                                actual_buys[product] += executable_volume_int

                                # Update remaining to buy with the integer volume
                                remaining_to_buy -= executable_volume_int

                                # If we've filled the entire order, break
                                if remaining_to_buy <= 0:
                                    break

        # Process sell orders
        for product, sell_qty in sell_components.items():
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
                                # We've already checked that this order is within our price limits during the liquidity check
                                # So we can just use the bid_price directly
                                # Ensure the price is an integer
                                adjusted_price = int(bid_price)

                                if product not in result:
                                    result[product] = []
                                result[product].append(Order(product, adjusted_price, -executable_volume_int))

                                # Update actual sells with the integer volume
                                actual_sells[product] += executable_volume_int

                                # Update remaining to sell with the integer volume
                                remaining_to_sell -= executable_volume_int

                                # If we've filled the entire order, break
                                if remaining_to_sell <= 0:
                                    break

        logger.print(f"Actual buys: {actual_buys}")
        logger.print(f"Actual sells: {actual_sells}")

        # Calculate execution ratio for the spread
        spread_execution_ratio = 1.0
        if spread_change != 0:
            # Determine the limiting factor for the spread execution
            if spread_change > 0:  # Buying the spread
                for product, buy_qty in buy_components.items():
                    if buy_qty > 0:
                        execution_ratio = actual_buys[product] / buy_qty if buy_qty > 0 else 1.0
                        spread_execution_ratio = min(spread_execution_ratio, execution_ratio)

                for product, sell_qty in sell_components.items():
                    if sell_qty > 0:
                        execution_ratio = actual_sells[product] / sell_qty if sell_qty > 0 else 1.0
                        spread_execution_ratio = min(spread_execution_ratio, execution_ratio)
            else:  # Selling the spread
                for product, buy_qty in buy_components.items():
                    if buy_qty > 0:
                        execution_ratio = actual_buys[product] / buy_qty if buy_qty > 0 else 1.0
                        spread_execution_ratio = min(spread_execution_ratio, execution_ratio)

                for product, sell_qty in sell_components.items():
                    if sell_qty > 0:
                        execution_ratio = actual_sells[product] / sell_qty if sell_qty > 0 else 1.0
                        spread_execution_ratio = min(spread_execution_ratio, execution_ratio)

        # Calculate actual spread position change
        # We need to ensure the actual spread change is an integer number of spread units
        if abs(spread_change) > 0:
            # Calculate the integer number of spread units that were executed
            executed_spread_units = int(abs(spread_change) * spread_execution_ratio)

            # Set the actual spread change with the correct sign
            if spread_change > 0:
                actual_spread_change = executed_spread_units
            else:
                actual_spread_change = -executed_spread_units
        else:
            actual_spread_change = 0

        logger.print(f"Spread execution ratio: {spread_execution_ratio}, actual change: {actual_spread_change}")

        # Update spread position
        self.spread_positions["custom_spread"] = int(self.spread_positions["custom_spread"] + actual_spread_change)

        # Determine if we're currently trading the custom spread
        is_custom_spread_trading = actual_spread_change != 0

        # Determine the direction of the custom spread trade
        custom_spread_direction = 0
        if actual_spread_change > 0:
            custom_spread_direction = 1  # Buying the custom spread
        elif actual_spread_change < 0:
            custom_spread_direction = -1  # Selling the custom spread

        logger.print(f"Custom spread trading: {is_custom_spread_trading}, direction: {custom_spread_direction}")

        # Check if we need to neutralize PB1 or PB2 spread positions that are trading against the custom spread
        spreads_to_neutralize = []

        if is_custom_spread_trading:
            if custom_spread_direction > 0:  # Buying the custom spread
                # We should neutralize any positive PB2 spread positions and any negative PB1 spread positions
                if self.spread_positions["pb2_spread"] > 0:
                    logger.print(f"Need to neutralize positive PB2 spread position: {self.spread_positions['pb2_spread']}")
                    spreads_to_neutralize.append(("pb2_spread", 0))

                if self.spread_positions["pb1_spread"] < 0:
                    logger.print(f"Need to neutralize negative PB1 spread position: {self.spread_positions['pb1_spread']}")
                    spreads_to_neutralize.append(("pb1_spread", 0))

            elif custom_spread_direction < 0:  # Selling the custom spread
                # We should neutralize any negative PB2 spread positions and any positive PB1 spread positions
                if self.spread_positions["pb2_spread"] < 0:
                    logger.print(f"Need to neutralize negative PB2 spread position: {self.spread_positions['pb2_spread']}")
                    spreads_to_neutralize.append(("pb2_spread", 0))

                if self.spread_positions["pb1_spread"] > 0:
                    logger.print(f"Need to neutralize positive PB1 spread position: {self.spread_positions['pb1_spread']}")
                    spreads_to_neutralize.append(("pb1_spread", 0))

        # Process spreads that need to be neutralized
        for spread_name, target_position in spreads_to_neutralize:
            logger.print(f"Neutralizing {spread_name} position from {self.spread_positions[spread_name]} to {target_position}")

            # Calculate the spread change needed to reach the target position
            spread_change = target_position - self.spread_positions[spread_name]

            if spread_name == "pb2_spread":
                # Calculate component quantities for PB2 spread change
                pb2_buy_components = {product: 0 for product in self.products}
                pb2_sell_components = {product: 0 for product in self.products}

                # PB2 spread = PICNIC_BASKET2 - (4×CROISSANTS + 2×JAMS)
                pb2_basket_qty = abs(spread_change * self.pb2_components["PICNIC_BASKET2"])
                croissants_qty = abs(spread_change * self.pb2_components["CROISSANTS"])
                jams_qty = abs(spread_change * self.pb2_components["JAMS"])

                if spread_change > 0:  # Buying the PB2 spread
                    # Buy positive components, sell negative components
                    pb2_buy_components["PICNIC_BASKET2"] = pb2_basket_qty
                    pb2_sell_components["CROISSANTS"] = croissants_qty
                    pb2_sell_components["JAMS"] = jams_qty
                else:  # Selling the PB2 spread
                    # Sell positive components, buy negative components
                    pb2_sell_components["PICNIC_BASKET2"] = pb2_basket_qty
                    pb2_buy_components["CROISSANTS"] = croissants_qty
                    pb2_buy_components["JAMS"] = jams_qty

                logger.print(f"PB2 neutralization - Buy components: {pb2_buy_components}")
                logger.print(f"PB2 neutralization - Sell components: {pb2_sell_components}")

                # Execute the neutralization trade with full trading logic
                # Step 1: Check if position limits would be violated
                pb2_scale_factor = 1.0

                # First, calculate the minimum scale factor needed to respect all position limits
                for product in ["PICNIC_BASKET2", "CROISSANTS", "JAMS"]:
                    # Check buy limit
                    if pb2_buy_components[product] > 0:
                        max_buy = self.position_limits[product] - current_positions[product]
                        if pb2_buy_components[product] > max_buy:
                            product_scale_factor = max_buy / pb2_buy_components[product] if pb2_buy_components[product] > 0 else 0
                            pb2_scale_factor = min(pb2_scale_factor, product_scale_factor)
                            logger.print(f"PB2 neutralization - Buy limit for {product}: need scale factor {product_scale_factor}")

                    # Check sell limit
                    if pb2_sell_components[product] > 0:
                        max_sell = current_positions[product] + self.position_limits[product]
                        if pb2_sell_components[product] > max_sell:
                            product_scale_factor = max_sell / pb2_sell_components[product] if pb2_sell_components[product] > 0 else 0
                            pb2_scale_factor = min(pb2_scale_factor, product_scale_factor)
                            logger.print(f"PB2 neutralization - Sell limit for {product}: need scale factor {product_scale_factor}")

                # If we need to scale down, calculate the integer number of spread units we can trade
                if pb2_scale_factor < 1.0:
                    # Calculate how many integer spread units we can trade
                    max_pb2_spread_units = int(abs(spread_change) * pb2_scale_factor)

                    # Recalculate scale factor based on integer spread units
                    if abs(spread_change) > 0:
                        pb2_scale_factor = max_pb2_spread_units / abs(spread_change)

                    logger.print(f"PB2 neutralization - Scaling down to {max_pb2_spread_units} spread units (scale factor: {pb2_scale_factor})")

                    # Scale the spread change to an integer number of units
                    if spread_change > 0:
                        spread_change = max_pb2_spread_units
                    else:
                        spread_change = -max_pb2_spread_units

                    # Recalculate component quantities based on the integer spread change
                    pb2_buy_components = {product: 0 for product in self.products}
                    pb2_sell_components = {product: 0 for product in self.products}

                    # PB2 spread = PICNIC_BASKET2 - (4×CROISSANTS + 2×JAMS)
                    pb2_basket_qty = abs(spread_change * self.pb2_components["PICNIC_BASKET2"])
                    croissants_qty = abs(spread_change * self.pb2_components["CROISSANTS"])
                    jams_qty = abs(spread_change * self.pb2_components["JAMS"])

                    if spread_change > 0:  # Buying the PB2 spread
                        # Buy positive components, sell negative components
                        pb2_buy_components["PICNIC_BASKET2"] = pb2_basket_qty
                        pb2_sell_components["CROISSANTS"] = croissants_qty
                        pb2_sell_components["JAMS"] = jams_qty
                    else:  # Selling the PB2 spread
                        # Sell positive components, buy negative components
                        pb2_sell_components["PICNIC_BASKET2"] = pb2_basket_qty
                        pb2_buy_components["CROISSANTS"] = croissants_qty
                        pb2_buy_components["JAMS"] = jams_qty

                logger.print(f"PB2 neutralization - After position limit check - Buy components: {pb2_buy_components}")
                logger.print(f"PB2 neutralization - After position limit check - Sell components: {pb2_sell_components}")

                # Step 2: Check available liquidity in the orderbook
                pb2_liquidity_scale_factor = 1.0

                # Check buy liquidity
                for product, buy_qty in pb2_buy_components.items():
                    if buy_qty > 0:
                        if product in state.order_depths:
                            order_depth = state.order_depths[product]

                            # Find available sell orders
                            if len(order_depth.sell_orders) > 0:
                                # Sort sell orders by price (ascending)
                                sorted_asks = sorted(order_depth.sell_orders.items())

                                # Always use exit price adjustment for neutralization
                                price_adjustment = self.exit_price_adjustments["pb2_spread"]

                                # Calculate the maximum price we're willing to pay
                                mid_price = (max(order_depth.buy_orders.keys()) + min(order_depth.sell_orders.keys())) / 2

                                # Handle special case when mid price ends in .5
                                if mid_price % 1 == 0.5:
                                    # For mid price ending in .5, adjustment of 0 means +0.5, adjustment of 1 means +1.5
                                    max_buy_price = int(mid_price + 0.5) + price_adjustment
                                else:
                                    # For integer mid price, adjustment is applied directly
                                    max_buy_price = int(mid_price) + price_adjustment

                                # Calculate how much we can buy at or below our max price
                                available_to_buy = 0
                                for ask_price, ask_volume in sorted_asks:
                                    if ask_price <= max_buy_price:
                                        available_to_buy += abs(ask_volume)  # Volume is negative in sell_orders
                                    else:
                                        break  # Stop once we exceed our max price

                                # Calculate scale factor based on available liquidity
                                if available_to_buy < buy_qty:
                                    product_liquidity_factor = available_to_buy / buy_qty if buy_qty > 0 else 0
                                    pb2_liquidity_scale_factor = min(pb2_liquidity_scale_factor, product_liquidity_factor)
                                    logger.print(f"PB2 neutralization - Buy liquidity for {product}: need scale factor {product_liquidity_factor} (available: {available_to_buy}, needed: {buy_qty})")
                            else:
                                # No sell orders available
                                logger.print(f"PB2 neutralization - No sell orders available for {product}")
                                pb2_liquidity_scale_factor = 0
                        else:
                            # Product not in order depths
                            logger.print(f"PB2 neutralization - No order depth available for {product}")
                            pb2_liquidity_scale_factor = 0

                # Check sell liquidity
                for product, sell_qty in pb2_sell_components.items():
                    if sell_qty > 0:
                        if product in state.order_depths:
                            order_depth = state.order_depths[product]

                            # Find available buy orders
                            if len(order_depth.buy_orders) > 0:
                                # Sort buy orders by price (descending)
                                sorted_bids = sorted(order_depth.buy_orders.items(), reverse=True)

                                # Always use exit price adjustment for neutralization
                                price_adjustment = self.exit_price_adjustments["pb2_spread"]

                                # Calculate the minimum price we're willing to accept
                                mid_price = (max(order_depth.buy_orders.keys()) + min(order_depth.sell_orders.keys())) / 2

                                # Handle special case when mid price ends in .5
                                if mid_price % 1 == 0.5:
                                    # For mid price ending in .5, adjustment of 0 means -0.5, adjustment of 1 means -1.5
                                    min_sell_price = int(mid_price - 0.5) - price_adjustment
                                else:
                                    # For integer mid price, adjustment is applied directly
                                    min_sell_price = int(mid_price) - price_adjustment

                                # Calculate how much we can sell at or above our min price
                                available_to_sell = 0
                                for bid_price, bid_volume in sorted_bids:
                                    if bid_price >= min_sell_price:
                                        available_to_sell += bid_volume
                                    else:
                                        break  # Stop once we go below our min price

                                # Calculate scale factor based on available liquidity
                                if available_to_sell < sell_qty:
                                    product_liquidity_factor = available_to_sell / sell_qty if sell_qty > 0 else 0
                                    pb2_liquidity_scale_factor = min(pb2_liquidity_scale_factor, product_liquidity_factor)
                                    logger.print(f"PB2 neutralization - Sell liquidity for {product}: need scale factor {product_liquidity_factor} (available: {available_to_sell}, needed: {sell_qty})")
                            else:
                                # No buy orders available
                                logger.print(f"PB2 neutralization - No buy orders available for {product}")
                                pb2_liquidity_scale_factor = 0
                        else:
                            # Product not in order depths
                            logger.print(f"PB2 neutralization - No order depth available for {product}")
                            pb2_liquidity_scale_factor = 0

                # If we need to scale down due to liquidity constraints
                if pb2_liquidity_scale_factor < 1.0:
                    # Calculate how many integer spread units we can trade based on liquidity
                    max_pb2_liquidity_spread_units = int(abs(spread_change) * pb2_liquidity_scale_factor)

                    # Recalculate scale factor based on integer spread units
                    if abs(spread_change) > 0:
                        pb2_liquidity_scale_factor = max_pb2_liquidity_spread_units / abs(spread_change)

                    logger.print(f"PB2 neutralization - Scaling down due to liquidity constraints to {max_pb2_liquidity_spread_units} spread units (scale factor: {pb2_liquidity_scale_factor})")

                    # Scale the spread change to an integer number of units
                    if spread_change > 0:
                        spread_change = max_pb2_liquidity_spread_units
                    else:
                        spread_change = -max_pb2_liquidity_spread_units

                    # Recalculate component quantities based on the integer spread change
                    pb2_buy_components = {product: 0 for product in self.products}
                    pb2_sell_components = {product: 0 for product in self.products}

                    # PB2 spread = PICNIC_BASKET2 - (4×CROISSANTS + 2×JAMS)
                    pb2_basket_qty = abs(spread_change * self.pb2_components["PICNIC_BASKET2"])
                    croissants_qty = abs(spread_change * self.pb2_components["CROISSANTS"])
                    jams_qty = abs(spread_change * self.pb2_components["JAMS"])

                    if spread_change > 0:  # Buying the PB2 spread
                        # Buy positive components, sell negative components
                        pb2_buy_components["PICNIC_BASKET2"] = pb2_basket_qty
                        pb2_sell_components["CROISSANTS"] = croissants_qty
                        pb2_sell_components["JAMS"] = jams_qty
                    else:  # Selling the PB2 spread
                        # Sell positive components, buy negative components
                        pb2_sell_components["PICNIC_BASKET2"] = pb2_basket_qty
                        pb2_buy_components["CROISSANTS"] = croissants_qty
                        pb2_buy_components["JAMS"] = jams_qty

                logger.print(f"PB2 neutralization - After liquidity check - Buy components: {pb2_buy_components}")
                logger.print(f"PB2 neutralization - After liquidity check - Sell components: {pb2_sell_components}")

                # Execute orders with proper liquidity taking
                pb2_actual_buys = {product: 0 for product in self.products}
                pb2_actual_sells = {product: 0 for product in self.products}

                # Process buy orders
                for product, buy_qty in pb2_buy_components.items():
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
                                        # We've already checked that this order is within our price limits during the liquidity check
                                        # So we can just use the ask_price directly
                                        # Ensure the price is an integer
                                        adjusted_price = int(ask_price)

                                        if product not in result:
                                            result[product] = []
                                        result[product].append(Order(product, adjusted_price, executable_volume_int))

                                        # Update actual buys with the integer volume
                                        pb2_actual_buys[product] += executable_volume_int

                                        # Update remaining to buy with the integer volume
                                        remaining_to_buy -= executable_volume_int

                                        # If we've filled the entire order, break
                                        if remaining_to_buy <= 0:
                                            break

                # Process sell orders
                for product, sell_qty in pb2_sell_components.items():
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
                                        # We've already checked that this order is within our price limits during the liquidity check
                                        # So we can just use the bid_price directly
                                        # Ensure the price is an integer
                                        adjusted_price = int(bid_price)

                                        if product not in result:
                                            result[product] = []
                                        result[product].append(Order(product, adjusted_price, -executable_volume_int))

                                        # Update actual sells with the integer volume
                                        pb2_actual_sells[product] += executable_volume_int

                                        # Update remaining to sell with the integer volume
                                        remaining_to_sell -= executable_volume_int

                                        # If we've filled the entire order, break
                                        if remaining_to_sell <= 0:
                                            break

                logger.print(f"PB2 neutralization - Actual buys: {pb2_actual_buys}")
                logger.print(f"PB2 neutralization - Actual sells: {pb2_actual_sells}")

                # Calculate execution ratio for the spread
                pb2_spread_execution_ratio = 1.0
                if spread_change != 0:
                    # Determine the limiting factor for the spread execution
                    if spread_change > 0:  # Buying the spread
                        for product, buy_qty in pb2_buy_components.items():
                            if buy_qty > 0:
                                execution_ratio = pb2_actual_buys[product] / buy_qty if buy_qty > 0 else 1.0
                                pb2_spread_execution_ratio = min(pb2_spread_execution_ratio, execution_ratio)

                        for product, sell_qty in pb2_sell_components.items():
                            if sell_qty > 0:
                                execution_ratio = pb2_actual_sells[product] / sell_qty if sell_qty > 0 else 1.0
                                pb2_spread_execution_ratio = min(pb2_spread_execution_ratio, execution_ratio)
                    else:  # Selling the spread
                        for product, buy_qty in pb2_buy_components.items():
                            if buy_qty > 0:
                                execution_ratio = pb2_actual_buys[product] / buy_qty if buy_qty > 0 else 1.0
                                pb2_spread_execution_ratio = min(pb2_spread_execution_ratio, execution_ratio)

                        for product, sell_qty in pb2_sell_components.items():
                            if sell_qty > 0:
                                execution_ratio = pb2_actual_sells[product] / sell_qty if sell_qty > 0 else 1.0
                                pb2_spread_execution_ratio = min(pb2_spread_execution_ratio, execution_ratio)

                # Calculate actual spread position change
                # We need to ensure the actual spread change is an integer number of spread units
                if abs(spread_change) > 0:
                    # Calculate the integer number of spread units that were executed
                    executed_pb2_spread_units = int(abs(spread_change) * pb2_spread_execution_ratio)

                    # Set the actual spread change with the correct sign
                    if spread_change > 0:
                        pb2_actual_spread_change = executed_pb2_spread_units
                    else:
                        pb2_actual_spread_change = -executed_pb2_spread_units
                else:
                    pb2_actual_spread_change = 0

                logger.print(f"PB2 neutralization - Spread execution ratio: {pb2_spread_execution_ratio}, actual change: {pb2_actual_spread_change}")

                # Update spread position
                self.spread_positions["pb2_spread"] = int(self.spread_positions["pb2_spread"] + pb2_actual_spread_change)

                # Update current positions for subsequent trades
                for product in self.products:
                    current_positions[product] += pb2_actual_buys[product] - pb2_actual_sells[product]

                logger.print(f"Neutralized PB2 spread position, new position: {self.spread_positions['pb2_spread']}")

            elif spread_name == "pb1_spread":
                # Calculate component quantities for PB1 spread change
                pb1_buy_components = {product: 0 for product in self.products}
                pb1_sell_components = {product: 0 for product in self.products}

                # PB1 spread = PICNIC_BASKET1 - (6×CROISSANTS + 3×JAMS + 1×DJEMBE)
                pb1_basket_qty = abs(spread_change * self.pb1_components["PICNIC_BASKET1"])
                croissants_qty = abs(spread_change * self.pb1_components["CROISSANTS"])
                jams_qty = abs(spread_change * self.pb1_components["JAMS"])
                djembes_qty = abs(spread_change * self.pb1_components["DJEMBES"])

                if spread_change > 0:  # Buying the PB1 spread
                    # Buy positive components, sell negative components
                    pb1_buy_components["PICNIC_BASKET1"] = pb1_basket_qty
                    pb1_sell_components["CROISSANTS"] = croissants_qty
                    pb1_sell_components["JAMS"] = jams_qty
                    pb1_sell_components["DJEMBES"] = djembes_qty
                else:  # Selling the PB1 spread
                    # Sell positive components, buy negative components
                    pb1_sell_components["PICNIC_BASKET1"] = pb1_basket_qty
                    pb1_buy_components["CROISSANTS"] = croissants_qty
                    pb1_buy_components["JAMS"] = jams_qty
                    pb1_buy_components["DJEMBES"] = djembes_qty

                logger.print(f"PB1 neutralization - Buy components: {pb1_buy_components}")
                logger.print(f"PB1 neutralization - Sell components: {pb1_sell_components}")

                # Execute the neutralization trade with full trading logic
                # Step 1: Check if position limits would be violated
                pb1_scale_factor = 1.0

                # First, calculate the minimum scale factor needed to respect all position limits
                for product in ["PICNIC_BASKET1", "CROISSANTS", "JAMS", "DJEMBES"]:
                    # Check buy limit
                    if pb1_buy_components[product] > 0:
                        max_buy = self.position_limits[product] - current_positions[product]
                        if pb1_buy_components[product] > max_buy:
                            product_scale_factor = max_buy / pb1_buy_components[product] if pb1_buy_components[product] > 0 else 0
                            pb1_scale_factor = min(pb1_scale_factor, product_scale_factor)
                            logger.print(f"PB1 neutralization - Buy limit for {product}: need scale factor {product_scale_factor}")

                    # Check sell limit
                    if pb1_sell_components[product] > 0:
                        max_sell = current_positions[product] + self.position_limits[product]
                        if pb1_sell_components[product] > max_sell:
                            product_scale_factor = max_sell / pb1_sell_components[product] if pb1_sell_components[product] > 0 else 0
                            pb1_scale_factor = min(pb1_scale_factor, product_scale_factor)
                            logger.print(f"PB1 neutralization - Sell limit for {product}: need scale factor {product_scale_factor}")

                # If we need to scale down, calculate the integer number of spread units we can trade
                if pb1_scale_factor < 1.0:
                    # Calculate how many integer spread units we can trade
                    max_pb1_spread_units = int(abs(spread_change) * pb1_scale_factor)

                    # Recalculate scale factor based on integer spread units
                    if abs(spread_change) > 0:
                        pb1_scale_factor = max_pb1_spread_units / abs(spread_change)

                    logger.print(f"PB1 neutralization - Scaling down to {max_pb1_spread_units} spread units (scale factor: {pb1_scale_factor})")

                    # Scale the spread change to an integer number of units
                    if spread_change > 0:
                        spread_change = max_pb1_spread_units
                    else:
                        spread_change = -max_pb1_spread_units

                    # Recalculate component quantities based on the integer spread change
                    pb1_buy_components = {product: 0 for product in self.products}
                    pb1_sell_components = {product: 0 for product in self.products}

                    # PB1 spread = PICNIC_BASKET1 - (6×CROISSANTS + 3×JAMS + 1×DJEMBE)
                    pb1_basket_qty = abs(spread_change * self.pb1_components["PICNIC_BASKET1"])
                    croissants_qty = abs(spread_change * self.pb1_components["CROISSANTS"])
                    jams_qty = abs(spread_change * self.pb1_components["JAMS"])
                    djembes_qty = abs(spread_change * self.pb1_components["DJEMBES"])

                    if spread_change > 0:  # Buying the PB1 spread
                        # Buy positive components, sell negative components
                        pb1_buy_components["PICNIC_BASKET1"] = pb1_basket_qty
                        pb1_sell_components["CROISSANTS"] = croissants_qty
                        pb1_sell_components["JAMS"] = jams_qty
                        pb1_sell_components["DJEMBES"] = djembes_qty
                    else:  # Selling the PB1 spread
                        # Sell positive components, buy negative components
                        pb1_sell_components["PICNIC_BASKET1"] = pb1_basket_qty
                        pb1_buy_components["CROISSANTS"] = croissants_qty
                        pb1_buy_components["JAMS"] = jams_qty
                        pb1_buy_components["DJEMBES"] = djembes_qty

                logger.print(f"PB1 neutralization - After position limit check - Buy components: {pb1_buy_components}")
                logger.print(f"PB1 neutralization - After position limit check - Sell components: {pb1_sell_components}")

                # Step 2: Check available liquidity in the orderbook
                pb1_liquidity_scale_factor = 1.0

                # Check buy liquidity
                for product, buy_qty in pb1_buy_components.items():
                    if buy_qty > 0:
                        if product in state.order_depths:
                            order_depth = state.order_depths[product]

                            # Find available sell orders
                            if len(order_depth.sell_orders) > 0:
                                # Sort sell orders by price (ascending)
                                sorted_asks = sorted(order_depth.sell_orders.items())

                                # Always use exit price adjustment for neutralization
                                price_adjustment = self.exit_price_adjustments["pb1_spread"]

                                # Calculate the maximum price we're willing to pay
                                mid_price = (max(order_depth.buy_orders.keys()) + min(order_depth.sell_orders.keys())) / 2

                                # Handle special case when mid price ends in .5
                                if mid_price % 1 == 0.5:
                                    # For mid price ending in .5, adjustment of 0 means +0.5, adjustment of 1 means +1.5
                                    max_buy_price = int(mid_price + 0.5) + price_adjustment
                                else:
                                    # For integer mid price, adjustment is applied directly
                                    max_buy_price = int(mid_price) + price_adjustment

                                # Calculate how much we can buy at or below our max price
                                available_to_buy = 0
                                for ask_price, ask_volume in sorted_asks:
                                    if ask_price <= max_buy_price:
                                        available_to_buy += abs(ask_volume)  # Volume is negative in sell_orders
                                    else:
                                        break  # Stop once we exceed our max price

                                # Calculate scale factor based on available liquidity
                                if available_to_buy < buy_qty:
                                    product_liquidity_factor = available_to_buy / buy_qty if buy_qty > 0 else 0
                                    pb1_liquidity_scale_factor = min(pb1_liquidity_scale_factor, product_liquidity_factor)
                                    logger.print(f"PB1 neutralization - Buy liquidity for {product}: need scale factor {product_liquidity_factor} (available: {available_to_buy}, needed: {buy_qty})")
                            else:
                                # No sell orders available
                                logger.print(f"PB1 neutralization - No sell orders available for {product}")
                                pb1_liquidity_scale_factor = 0
                        else:
                            # Product not in order depths
                            logger.print(f"PB1 neutralization - No order depth available for {product}")
                            pb1_liquidity_scale_factor = 0

                # Check sell liquidity
                for product, sell_qty in pb1_sell_components.items():
                    if sell_qty > 0:
                        if product in state.order_depths:
                            order_depth = state.order_depths[product]

                            # Find available buy orders
                            if len(order_depth.buy_orders) > 0:
                                # Sort buy orders by price (descending)
                                sorted_bids = sorted(order_depth.buy_orders.items(), reverse=True)

                                # Always use exit price adjustment for neutralization
                                price_adjustment = self.exit_price_adjustments["pb1_spread"]

                                # Calculate the minimum price we're willing to accept
                                mid_price = (max(order_depth.buy_orders.keys()) + min(order_depth.sell_orders.keys())) / 2

                                # Handle special case when mid price ends in .5
                                if mid_price % 1 == 0.5:
                                    # For mid price ending in .5, adjustment of 0 means -0.5, adjustment of 1 means -1.5
                                    min_sell_price = int(mid_price - 0.5) - price_adjustment
                                else:
                                    # For integer mid price, adjustment is applied directly
                                    min_sell_price = int(mid_price) - price_adjustment

                                # Calculate how much we can sell at or above our min price
                                available_to_sell = 0
                                for bid_price, bid_volume in sorted_bids:
                                    if bid_price >= min_sell_price:
                                        available_to_sell += bid_volume
                                    else:
                                        break  # Stop once we go below our min price

                                # Calculate scale factor based on available liquidity
                                if available_to_sell < sell_qty:
                                    product_liquidity_factor = available_to_sell / sell_qty if sell_qty > 0 else 0
                                    pb1_liquidity_scale_factor = min(pb1_liquidity_scale_factor, product_liquidity_factor)
                                    logger.print(f"PB1 neutralization - Sell liquidity for {product}: need scale factor {product_liquidity_factor} (available: {available_to_sell}, needed: {sell_qty})")
                            else:
                                # No buy orders available
                                logger.print(f"PB1 neutralization - No buy orders available for {product}")
                                pb1_liquidity_scale_factor = 0
                        else:
                            # Product not in order depths
                            logger.print(f"PB1 neutralization - No order depth available for {product}")
                            pb1_liquidity_scale_factor = 0

                # If we need to scale down due to liquidity constraints
                if pb1_liquidity_scale_factor < 1.0:
                    # Calculate how many integer spread units we can trade based on liquidity
                    max_pb1_liquidity_spread_units = int(abs(spread_change) * pb1_liquidity_scale_factor)

                    # Recalculate scale factor based on integer spread units
                    if abs(spread_change) > 0:
                        pb1_liquidity_scale_factor = max_pb1_liquidity_spread_units / abs(spread_change)

                    logger.print(f"PB1 neutralization - Scaling down due to liquidity constraints to {max_pb1_liquidity_spread_units} spread units (scale factor: {pb1_liquidity_scale_factor})")

                    # Scale the spread change to an integer number of units
                    if spread_change > 0:
                        spread_change = max_pb1_liquidity_spread_units
                    else:
                        spread_change = -max_pb1_liquidity_spread_units

                    # Recalculate component quantities based on the integer spread change
                    pb1_buy_components = {product: 0 for product in self.products}
                    pb1_sell_components = {product: 0 for product in self.products}

                    # PB1 spread = PICNIC_BASKET1 - (6×CROISSANTS + 3×JAMS + 1×DJEMBE)
                    pb1_basket_qty = abs(spread_change * self.pb1_components["PICNIC_BASKET1"])
                    croissants_qty = abs(spread_change * self.pb1_components["CROISSANTS"])
                    jams_qty = abs(spread_change * self.pb1_components["JAMS"])
                    djembes_qty = abs(spread_change * self.pb1_components["DJEMBES"])

                    if spread_change > 0:  # Buying the PB1 spread
                        # Buy positive components, sell negative components
                        pb1_buy_components["PICNIC_BASKET1"] = pb1_basket_qty
                        pb1_sell_components["CROISSANTS"] = croissants_qty
                        pb1_sell_components["JAMS"] = jams_qty
                        pb1_sell_components["DJEMBES"] = djembes_qty
                    else:  # Selling the PB1 spread
                        # Sell positive components, buy negative components
                        pb1_sell_components["PICNIC_BASKET1"] = pb1_basket_qty
                        pb1_buy_components["CROISSANTS"] = croissants_qty
                        pb1_buy_components["JAMS"] = jams_qty
                        pb1_buy_components["DJEMBES"] = djembes_qty

                logger.print(f"PB1 neutralization - After liquidity check - Buy components: {pb1_buy_components}")
                logger.print(f"PB1 neutralization - After liquidity check - Sell components: {pb1_sell_components}")

                # Execute orders with proper liquidity taking
                pb1_actual_buys = {product: 0 for product in self.products}
                pb1_actual_sells = {product: 0 for product in self.products}

                # Process buy orders
                for product, buy_qty in pb1_buy_components.items():
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
                                        # We've already checked that this order is within our price limits during the liquidity check
                                        # So we can just use the ask_price directly
                                        # Ensure the price is an integer
                                        adjusted_price = int(ask_price)

                                        if product not in result:
                                            result[product] = []
                                        result[product].append(Order(product, adjusted_price, executable_volume_int))

                                        # Update actual buys with the integer volume
                                        pb1_actual_buys[product] += executable_volume_int

                                        # Update remaining to buy with the integer volume
                                        remaining_to_buy -= executable_volume_int

                                        # If we've filled the entire order, break
                                        if remaining_to_buy <= 0:
                                            break

                # Process sell orders
                for product, sell_qty in pb1_sell_components.items():
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
                                        # We've already checked that this order is within our price limits during the liquidity check
                                        # So we can just use the bid_price directly
                                        # Ensure the price is an integer
                                        adjusted_price = int(bid_price)

                                        if product not in result:
                                            result[product] = []
                                        result[product].append(Order(product, adjusted_price, -executable_volume_int))

                                        # Update actual sells with the integer volume
                                        pb1_actual_sells[product] += executable_volume_int

                                        # Update remaining to sell with the integer volume
                                        remaining_to_sell -= executable_volume_int

                                        # If we've filled the entire order, break
                                        if remaining_to_sell <= 0:
                                            break

                logger.print(f"PB1 neutralization - Actual buys: {pb1_actual_buys}")
                logger.print(f"PB1 neutralization - Actual sells: {pb1_actual_sells}")

                # Calculate execution ratio for the spread
                pb1_spread_execution_ratio = 1.0
                if spread_change != 0:
                    # Determine the limiting factor for the spread execution
                    if spread_change > 0:  # Buying the spread
                        for product, buy_qty in pb1_buy_components.items():
                            if buy_qty > 0:
                                execution_ratio = pb1_actual_buys[product] / buy_qty if buy_qty > 0 else 1.0
                                pb1_spread_execution_ratio = min(pb1_spread_execution_ratio, execution_ratio)

                        for product, sell_qty in pb1_sell_components.items():
                            if sell_qty > 0:
                                execution_ratio = pb1_actual_sells[product] / sell_qty if sell_qty > 0 else 1.0
                                pb1_spread_execution_ratio = min(pb1_spread_execution_ratio, execution_ratio)
                    else:  # Selling the spread
                        for product, buy_qty in pb1_buy_components.items():
                            if buy_qty > 0:
                                execution_ratio = pb1_actual_buys[product] / buy_qty if buy_qty > 0 else 1.0
                                pb1_spread_execution_ratio = min(pb1_spread_execution_ratio, execution_ratio)

                        for product, sell_qty in pb1_sell_components.items():
                            if sell_qty > 0:
                                execution_ratio = pb1_actual_sells[product] / sell_qty if sell_qty > 0 else 1.0
                                pb1_spread_execution_ratio = min(pb1_spread_execution_ratio, execution_ratio)

                # Calculate actual spread position change
                # We need to ensure the actual spread change is an integer number of spread units
                if abs(spread_change) > 0:
                    # Calculate the integer number of spread units that were executed
                    executed_pb1_spread_units = int(abs(spread_change) * pb1_spread_execution_ratio)

                    # Set the actual spread change with the correct sign
                    if spread_change > 0:
                        pb1_actual_spread_change = executed_pb1_spread_units
                    else:
                        pb1_actual_spread_change = -executed_pb1_spread_units
                else:
                    pb1_actual_spread_change = 0

                logger.print(f"PB1 neutralization - Spread execution ratio: {pb1_spread_execution_ratio}, actual change: {pb1_actual_spread_change}")

                # Update spread position
                self.spread_positions["pb1_spread"] = int(self.spread_positions["pb1_spread"] + pb1_actual_spread_change)

                # Update current positions for subsequent trades
                for product in self.products:
                    current_positions[product] += pb1_actual_buys[product] - pb1_actual_sells[product]

                logger.print(f"Neutralized PB1 spread position, new position: {self.spread_positions['pb1_spread']}")

        # Calculate theoretical PB1 and PB2 spread positions based on component positions
        # This is just for tracking purposes, we don't trade these spreads
        theoretical_pb1_position = 0
        theoretical_pb2_position = 0

        # Get updated positions after trading
        updated_positions = {product: current_positions[product] for product in self.products}
        for product in self.products:
            updated_positions[product] += actual_buys[product] - actual_sells[product]

        # Calculate theoretical PB1 spread position
        # PB1 spread = PICNIC_BASKET1 - (6×CROISSANTS + 3×JAMS + 1×DJEMBE)
        if "PICNIC_BASKET1" in updated_positions and "CROISSANTS" in updated_positions and \
           "JAMS" in updated_positions and "DJEMBES" in updated_positions:
            pb1_basket_qty = updated_positions["PICNIC_BASKET1"]
            pb1_components_qty = (6 * updated_positions["CROISSANTS"] +
                                 3 * updated_positions["JAMS"] +
                                 1 * updated_positions["DJEMBES"]) / 10  # Scale to match basket size
            theoretical_pb1_position = min(pb1_basket_qty, pb1_components_qty)

        # Calculate theoretical PB2 spread position
        # PB2 spread = PICNIC_BASKET2 - (4×CROISSANTS + 2×JAMS)
        if "PICNIC_BASKET2" in updated_positions and "CROISSANTS" in updated_positions and \
           "JAMS" in updated_positions:
            pb2_basket_qty = updated_positions["PICNIC_BASKET2"]
            pb2_components_qty = (4 * updated_positions["CROISSANTS"] +
                                 2 * updated_positions["JAMS"]) / 6  # Scale to match basket size
            theoretical_pb2_position = min(pb2_basket_qty, pb2_components_qty)

        # Calculate theoretical PB1 and PB2 spread positions for tracking
        theoretical_pb1_spread_position = int(theoretical_pb1_position)
        theoretical_pb2_spread_position = int(theoretical_pb2_position)

        logger.print(f"Theoretical PB1 spread position: {theoretical_pb1_spread_position}")
        logger.print(f"Theoretical PB2 spread position: {theoretical_pb2_spread_position}")

        # Update the spread positions based on z-scores and thresholds
        # Only update if max_spread_positions is > 0
        # The position should only be updated if the z-score is beyond the entry threshold
        # Whether we trade on it depends on the state of the custom spread
        pb1_z_score = z_scores["pb1_spread"]
        pb1_entry_threshold = self.entry_thresholds["pb1_spread"]
        pb1_exit_threshold = self.exit_thresholds["pb1_spread"]

        pb2_z_score = z_scores["pb2_spread"]
        pb2_entry_threshold = self.entry_thresholds["pb2_spread"]
        pb2_exit_threshold = self.exit_thresholds["pb2_spread"]

        # Calculate theoretical spread positions based on z-scores and entry thresholds
        # For PB1 spread
        if self.max_spread_positions["pb1_spread"] > 0:
            # Only set a target position if the z-score is beyond the entry threshold
            if pb1_z_score <= -pb1_entry_threshold:  # Below negative entry threshold -> Buy
                self.spread_positions["pb1_spread"] = self.max_spread_positions["pb1_spread"]
                logger.print(f"Set PB1 spread position to {self.spread_positions['pb1_spread']} (buy) based on z-score {pb1_z_score} < -{pb1_entry_threshold}")
            elif pb1_z_score >= pb1_entry_threshold:  # Above entry threshold -> Sell
                self.spread_positions["pb1_spread"] = -self.max_spread_positions["pb1_spread"]
                logger.print(f"Set PB1 spread position to {self.spread_positions['pb1_spread']} (sell) based on z-score {pb1_z_score} > {pb1_entry_threshold}")
            elif abs(pb1_z_score) < pb1_exit_threshold:  # Within exit threshold -> Exit
                self.spread_positions["pb1_spread"] = 0
                logger.print(f"Set PB1 spread position to 0 (exit) based on z-score {pb1_z_score} within exit threshold {pb1_exit_threshold}")
            else:
                logger.print(f"Maintaining current PB1 spread position {self.spread_positions['pb1_spread']} because z-score {pb1_z_score} is between exit threshold {pb1_exit_threshold} and entry threshold {pb1_entry_threshold}")

        # For PB2 spread
        if self.max_spread_positions["pb2_spread"] > 0:
            # Only set a target position if the z-score is beyond the entry threshold
            if pb2_z_score <= -pb2_entry_threshold:  # Below negative entry threshold -> Buy
                self.spread_positions["pb2_spread"] = self.max_spread_positions["pb2_spread"]
                logger.print(f"Set PB2 spread position to {self.spread_positions['pb2_spread']} (buy) based on z-score {pb2_z_score} < -{pb2_entry_threshold}")
            elif pb2_z_score >= pb2_entry_threshold:  # Above entry threshold -> Sell
                self.spread_positions["pb2_spread"] = -self.max_spread_positions["pb2_spread"]
                logger.print(f"Set PB2 spread position to {self.spread_positions['pb2_spread']} (sell) based on z-score {pb2_z_score} > {pb2_entry_threshold}")
            elif abs(pb2_z_score) < pb2_exit_threshold:  # Within exit threshold -> Exit
                self.spread_positions["pb2_spread"] = 0
                logger.print(f"Set PB2 spread position to 0 (exit) based on z-score {pb2_z_score} within exit threshold {pb2_exit_threshold}")
            else:
                logger.print(f"Maintaining current PB2 spread position {self.spread_positions['pb2_spread']} because z-score {pb2_z_score} is between exit threshold {pb2_exit_threshold} and entry threshold {pb2_entry_threshold}")

        logger.print(f"Updated spread positions: {self.spread_positions}")

        # Get max positions for both spreads
        pb1_max_position = self.max_spread_positions["pb1_spread"]
        pb2_max_position = self.max_spread_positions["pb2_spread"]

        # Check if either spread is within exit threshold
        pb1_within_exit = abs(pb1_z_score) < pb1_exit_threshold
        pb2_within_exit = abs(pb2_z_score) < pb2_exit_threshold

        logger.print(f"PB1 z-score: {pb1_z_score}, entry threshold: {pb1_entry_threshold}, exit threshold: {pb1_exit_threshold}, within exit: {pb1_within_exit}")
        logger.print(f"PB2 z-score: {pb2_z_score}, entry threshold: {pb2_entry_threshold}, exit threshold: {pb2_exit_threshold}, within exit: {pb2_within_exit}")

        # First, check if we need to exit any positions in PB1 or PB2 spreads
        # We always exit if the spread is within exit threshold, regardless of custom spread trading
        spreads_to_exit = []

        if pb2_within_exit and self.spread_positions["pb2_spread"] != 0 and self.max_spread_positions["pb2_spread"] > 0:
            spreads_to_exit.append("pb2_spread")
            logger.print(f"PB2 spread within exit threshold, will exit position: {self.spread_positions['pb2_spread']}")

        if pb1_within_exit and self.spread_positions["pb1_spread"] != 0 and self.max_spread_positions["pb1_spread"] > 0:
            spreads_to_exit.append("pb1_spread")
            logger.print(f"PB1 spread within exit threshold, will exit position: {self.spread_positions['pb1_spread']}")

        # Process exits first
        for spread_name in spreads_to_exit:
            if spread_name == "pb2_spread":
                # Exit PB2 spread position
                pb2_target_position = 0
                pb2_spread_change = int(pb2_target_position - self.spread_positions["pb2_spread"])

                logger.print(f"Exiting PB2 spread position: {self.spread_positions['pb2_spread']} -> 0, change: {pb2_spread_change}")

                # If there's a change in position, trade the PB2 spread
                if pb2_spread_change != 0:
                    # Calculate component quantities for PB2 spread change
                    pb2_buy_components = {product: 0 for product in self.products}
                    pb2_sell_components = {product: 0 for product in self.products}

                    # PB2 spread = PICNIC_BASKET2 - (4×CROISSANTS + 2×JAMS)
                    pb2_basket_qty = abs(pb2_spread_change * self.pb2_components["PICNIC_BASKET2"])
                    croissants_qty = abs(pb2_spread_change * self.pb2_components["CROISSANTS"])
                    jams_qty = abs(pb2_spread_change * self.pb2_components["JAMS"])

                    if pb2_spread_change > 0:  # Buying the PB2 spread
                        # Buy positive components, sell negative components
                        pb2_buy_components["PICNIC_BASKET2"] = pb2_basket_qty
                        pb2_sell_components["CROISSANTS"] = croissants_qty
                        pb2_sell_components["JAMS"] = jams_qty
                    else:  # Selling the PB2 spread
                        # Sell positive components, buy negative components
                        pb2_sell_components["PICNIC_BASKET2"] = pb2_basket_qty
                        pb2_buy_components["CROISSANTS"] = croissants_qty
                        pb2_buy_components["JAMS"] = jams_qty

                    # Execute the exit trade with full trading logic
                    # Step 1: Check if position limits would be violated
                    pb2_scale_factor = 1.0

                    # First, calculate the minimum scale factor needed to respect all position limits
                    for product in ["PICNIC_BASKET2", "CROISSANTS", "JAMS"]:
                        # Check buy limit
                        if pb2_buy_components[product] > 0:
                            max_buy = self.position_limits[product] - current_positions[product]
                            if pb2_buy_components[product] > max_buy:
                                product_scale_factor = max_buy / pb2_buy_components[product] if pb2_buy_components[product] > 0 else 0
                                pb2_scale_factor = min(pb2_scale_factor, product_scale_factor)
                                logger.print(f"PB2 - Buy limit for {product}: need scale factor {product_scale_factor}")

                        # Check sell limit
                        if pb2_sell_components[product] > 0:
                            max_sell = current_positions[product] + self.position_limits[product]
                            if pb2_sell_components[product] > max_sell:
                                product_scale_factor = max_sell / pb2_sell_components[product] if pb2_sell_components[product] > 0 else 0
                                pb2_scale_factor = min(pb2_scale_factor, product_scale_factor)
                                logger.print(f"PB2 - Sell limit for {product}: need scale factor {product_scale_factor}")

                    # If we need to scale down, calculate the integer number of spread units we can trade
                    if pb2_scale_factor < 1.0:
                        # Calculate how many integer spread units we can trade
                        max_pb2_spread_units = int(abs(pb2_spread_change) * pb2_scale_factor)

                        # Recalculate scale factor based on integer spread units
                        if abs(pb2_spread_change) > 0:
                            pb2_scale_factor = max_pb2_spread_units / abs(pb2_spread_change)

                        logger.print(f"PB2 - Scaling down to {max_pb2_spread_units} spread units (scale factor: {pb2_scale_factor})")

                        # Scale the spread change to an integer number of units
                        if pb2_spread_change > 0:
                            pb2_spread_change = max_pb2_spread_units
                        else:
                            pb2_spread_change = -max_pb2_spread_units

                        # Recalculate component quantities based on the integer spread change
                        pb2_buy_components = {product: 0 for product in self.products}
                        pb2_sell_components = {product: 0 for product in self.products}

                        # PB2 spread = PICNIC_BASKET2 - (4×CROISSANTS + 2×JAMS)
                        pb2_basket_qty = abs(pb2_spread_change * self.pb2_components["PICNIC_BASKET2"])
                        croissants_qty = abs(pb2_spread_change * self.pb2_components["CROISSANTS"])
                        jams_qty = abs(pb2_spread_change * self.pb2_components["JAMS"])

                        if pb2_spread_change > 0:  # Buying the PB2 spread
                            # Buy positive components, sell negative components
                            pb2_buy_components["PICNIC_BASKET2"] = pb2_basket_qty
                            pb2_sell_components["CROISSANTS"] = croissants_qty
                            pb2_sell_components["JAMS"] = jams_qty
                        else:  # Selling the PB2 spread
                            # Sell positive components, buy negative components
                            pb2_sell_components["PICNIC_BASKET2"] = pb2_basket_qty
                            pb2_buy_components["CROISSANTS"] = croissants_qty
                            pb2_buy_components["JAMS"] = jams_qty

                    logger.print(f"PB2 - After position limit check - Buy components: {pb2_buy_components}")
                    logger.print(f"PB2 - After position limit check - Sell components: {pb2_sell_components}")

                    # Step 2: Check available liquidity in the orderbook
                    pb2_liquidity_scale_factor = 1.0

                    # Check buy liquidity
                    for product, buy_qty in pb2_buy_components.items():
                        if buy_qty > 0:
                            if product in state.order_depths:
                                order_depth = state.order_depths[product]

                                # Find available sell orders
                                if len(order_depth.sell_orders) > 0:
                                    # Sort sell orders by price (ascending)
                                    sorted_asks = sorted(order_depth.sell_orders.items())

                                    # Determine price adjustment based on z-score thresholds
                                    # Get thresholds and price adjustments for the PB2 spread
                                    pb2_entry_price_adjustment = self.entry_price_adjustments["pb2_spread"]
                                    pb2_exit_price_adjustment = self.exit_price_adjustments["pb2_spread"]

                                    # For exits, always use exit price adjustment
                                    price_adjustment = pb2_exit_price_adjustment

                                    # Calculate the maximum price we're willing to pay
                                    mid_price = (max(order_depth.buy_orders.keys()) + min(order_depth.sell_orders.keys())) / 2

                                    # Handle special case when mid price ends in .5
                                    if mid_price % 1 == 0.5:
                                        # For mid price ending in .5, adjustment of 0 means +0.5, adjustment of 1 means +1.5
                                        max_buy_price = int(mid_price + 0.5) + price_adjustment
                                    else:
                                        # For integer mid price, adjustment is applied directly
                                        max_buy_price = int(mid_price) + price_adjustment

                                    # Calculate how much we can buy at or below our max price
                                    available_to_buy = 0
                                    for ask_price, ask_volume in sorted_asks:
                                        if ask_price <= max_buy_price:
                                            available_to_buy += abs(ask_volume)  # Volume is negative in sell_orders
                                        else:
                                            break  # Stop once we exceed our max price

                                    # Calculate scale factor based on available liquidity
                                    if available_to_buy < buy_qty:
                                        product_liquidity_factor = available_to_buy / buy_qty if buy_qty > 0 else 0
                                        pb2_liquidity_scale_factor = min(pb2_liquidity_scale_factor, product_liquidity_factor)
                                        logger.print(f"PB2 - Buy liquidity for {product}: need scale factor {product_liquidity_factor} (available: {available_to_buy}, needed: {buy_qty})")
                                else:
                                    # No sell orders available
                                    logger.print(f"PB2 - No sell orders available for {product}")
                                    pb2_liquidity_scale_factor = 0
                            else:
                                # Product not in order depths
                                logger.print(f"PB2 - No order depth available for {product}")
                                pb2_liquidity_scale_factor = 0

                    # Check sell liquidity
                    for product, sell_qty in pb2_sell_components.items():
                        if sell_qty > 0:
                            if product in state.order_depths:
                                order_depth = state.order_depths[product]

                                # Find available buy orders
                                if len(order_depth.buy_orders) > 0:
                                    # Sort buy orders by price (descending)
                                    sorted_bids = sorted(order_depth.buy_orders.items(), reverse=True)

                                    # Determine price adjustment based on z-score thresholds
                                    # Get thresholds and price adjustments for the PB2 spread
                                    pb2_entry_price_adjustment = self.entry_price_adjustments["pb2_spread"]
                                    pb2_exit_price_adjustment = self.exit_price_adjustments["pb2_spread"]

                                    # For exits, always use exit price adjustment
                                    price_adjustment = pb2_exit_price_adjustment

                                    # Calculate the minimum price we're willing to accept
                                    mid_price = (max(order_depth.buy_orders.keys()) + min(order_depth.sell_orders.keys())) / 2

                                    # Handle special case when mid price ends in .5
                                    if mid_price % 1 == 0.5:
                                        # For mid price ending in .5, adjustment of 0 means -0.5, adjustment of 1 means -1.5
                                        min_sell_price = int(mid_price - 0.5) - price_adjustment
                                    else:
                                        # For integer mid price, adjustment is applied directly
                                        min_sell_price = int(mid_price) - price_adjustment

                                    # Calculate how much we can sell at or above our min price
                                    available_to_sell = 0
                                    for bid_price, bid_volume in sorted_bids:
                                        if bid_price >= min_sell_price:
                                            available_to_sell += bid_volume
                                        else:
                                            break  # Stop once we go below our min price

                                    # Calculate scale factor based on available liquidity
                                    if available_to_sell < sell_qty:
                                        product_liquidity_factor = available_to_sell / sell_qty if sell_qty > 0 else 0
                                        pb2_liquidity_scale_factor = min(pb2_liquidity_scale_factor, product_liquidity_factor)
                                        logger.print(f"PB2 - Sell liquidity for {product}: need scale factor {product_liquidity_factor} (available: {available_to_sell}, needed: {sell_qty})")
                                else:
                                    # No buy orders available
                                    logger.print(f"PB2 - No buy orders available for {product}")
                                    pb2_liquidity_scale_factor = 0
                            else:
                                # Product not in order depths
                                logger.print(f"PB2 - No order depth available for {product}")
                                pb2_liquidity_scale_factor = 0

                    # If we need to scale down due to liquidity constraints
                    if pb2_liquidity_scale_factor < 1.0:
                        # Calculate how many integer spread units we can trade based on liquidity
                        max_pb2_liquidity_spread_units = int(abs(pb2_spread_change) * pb2_liquidity_scale_factor)

                        # Recalculate scale factor based on integer spread units
                        if abs(pb2_spread_change) > 0:
                            pb2_liquidity_scale_factor = max_pb2_liquidity_spread_units / abs(pb2_spread_change)

                        logger.print(f"PB2 - Scaling down due to liquidity constraints to {max_pb2_liquidity_spread_units} spread units (scale factor: {pb2_liquidity_scale_factor})")

                        # Scale the spread change to an integer number of units
                        if pb2_spread_change > 0:
                            pb2_spread_change = max_pb2_liquidity_spread_units
                        else:
                            pb2_spread_change = -max_pb2_liquidity_spread_units

                        # Recalculate component quantities based on the integer spread change
                        pb2_buy_components = {product: 0 for product in self.products}
                        pb2_sell_components = {product: 0 for product in self.products}

                        # PB2 spread = PICNIC_BASKET2 - (4×CROISSANTS + 2×JAMS)
                        pb2_basket_qty = abs(pb2_spread_change * self.pb2_components["PICNIC_BASKET2"])
                        croissants_qty = abs(pb2_spread_change * self.pb2_components["CROISSANTS"])
                        jams_qty = abs(pb2_spread_change * self.pb2_components["JAMS"])

                        if pb2_spread_change > 0:  # Buying the PB2 spread
                            # Buy positive components, sell negative components
                            pb2_buy_components["PICNIC_BASKET2"] = pb2_basket_qty
                            pb2_sell_components["CROISSANTS"] = croissants_qty
                            pb2_sell_components["JAMS"] = jams_qty
                        else:  # Selling the PB2 spread
                            # Sell positive components, buy negative components
                            pb2_sell_components["PICNIC_BASKET2"] = pb2_basket_qty
                            pb2_buy_components["CROISSANTS"] = croissants_qty
                            pb2_buy_components["JAMS"] = jams_qty

                    logger.print(f"PB2 - After liquidity check - Buy components: {pb2_buy_components}")
                    logger.print(f"PB2 - After liquidity check - Sell components: {pb2_sell_components}")

                    # Execute orders with proper liquidity taking
                    pb2_actual_buys = {product: 0 for product in self.products}
                    pb2_actual_sells = {product: 0 for product in self.products}

                    # Process buy orders
                    for product, buy_qty in pb2_buy_components.items():
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
                                            # We've already checked that this order is within our price limits during the liquidity check
                                            # So we can just use the ask_price directly
                                            # Ensure the price is an integer
                                            adjusted_price = int(ask_price)

                                            if product not in result:
                                                result[product] = []
                                            result[product].append(Order(product, adjusted_price, executable_volume_int))

                                            # Update actual buys with the integer volume
                                            pb2_actual_buys[product] += executable_volume_int

                                            # Update remaining to buy with the integer volume
                                            remaining_to_buy -= executable_volume_int

                                            # If we've filled the entire order, break
                                            if remaining_to_buy <= 0:
                                                break

                    # Process sell orders
                    for product, sell_qty in pb2_sell_components.items():
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
                                            # We've already checked that this order is within our price limits during the liquidity check
                                            # So we can just use the bid_price directly
                                            # Ensure the price is an integer
                                            adjusted_price = int(bid_price)

                                            if product not in result:
                                                result[product] = []
                                            result[product].append(Order(product, adjusted_price, -executable_volume_int))

                                            # Update actual sells with the integer volume
                                            pb2_actual_sells[product] += executable_volume_int

                                            # Update remaining to sell with the integer volume
                                            remaining_to_sell -= executable_volume_int

                                            # If we've filled the entire order, break
                                            if remaining_to_sell <= 0:
                                                break

                    logger.print(f"PB2 - Actual buys: {pb2_actual_buys}")
                    logger.print(f"PB2 - Actual sells: {pb2_actual_sells}")

                    # Calculate execution ratio for the spread
                    pb2_spread_execution_ratio = 1.0
                    if pb2_spread_change != 0:
                        # Determine the limiting factor for the spread execution
                        if pb2_spread_change > 0:  # Buying the spread
                            for product, buy_qty in pb2_buy_components.items():
                                if buy_qty > 0:
                                    execution_ratio = pb2_actual_buys[product] / buy_qty if buy_qty > 0 else 1.0
                                    pb2_spread_execution_ratio = min(pb2_spread_execution_ratio, execution_ratio)

                            for product, sell_qty in pb2_sell_components.items():
                                if sell_qty > 0:
                                    execution_ratio = pb2_actual_sells[product] / sell_qty if sell_qty > 0 else 1.0
                                    pb2_spread_execution_ratio = min(pb2_spread_execution_ratio, execution_ratio)
                        else:  # Selling the spread
                            for product, buy_qty in pb2_buy_components.items():
                                if buy_qty > 0:
                                    execution_ratio = pb2_actual_buys[product] / buy_qty if buy_qty > 0 else 1.0
                                    pb2_spread_execution_ratio = min(pb2_spread_execution_ratio, execution_ratio)

                            for product, sell_qty in pb2_sell_components.items():
                                if sell_qty > 0:
                                    execution_ratio = pb2_actual_sells[product] / sell_qty if sell_qty > 0 else 1.0
                                    pb2_spread_execution_ratio = min(pb2_spread_execution_ratio, execution_ratio)

                    # Calculate actual spread position change
                    # We need to ensure the actual spread change is an integer number of spread units
                    if abs(pb2_spread_change) > 0:
                        # Calculate the integer number of spread units that were executed
                        executed_pb2_spread_units = int(abs(pb2_spread_change) * pb2_spread_execution_ratio)

                        # Set the actual spread change with the correct sign
                        if pb2_spread_change > 0:
                            pb2_actual_spread_change = executed_pb2_spread_units
                        else:
                            pb2_actual_spread_change = -executed_pb2_spread_units
                    else:
                        pb2_actual_spread_change = 0

                    logger.print(f"PB2 - Spread execution ratio: {pb2_spread_execution_ratio}, actual change: {pb2_actual_spread_change}")

                    # Update spread position
                    self.spread_positions["pb2_spread"] = int(self.spread_positions["pb2_spread"] + pb2_actual_spread_change)

                    # Update current positions for subsequent trades
                    for product in self.products:
                        current_positions[product] += pb2_actual_buys[product] - pb2_actual_sells[product]

                    logger.print(f"Exited PB2 spread position, new position: {self.spread_positions['pb2_spread']}")

            elif spread_name == "pb1_spread":
                # Exit PB1 spread position
                pb1_target_position = 0
                pb1_spread_change = int(pb1_target_position - self.spread_positions["pb1_spread"])

                logger.print(f"Exiting PB1 spread position: {self.spread_positions['pb1_spread']} -> 0, change: {pb1_spread_change}")

                # If there's a change in position, trade the PB1 spread
                if pb1_spread_change != 0:
                    # Calculate component quantities for PB1 spread change
                    pb1_buy_components = {product: 0 for product in self.products}
                    pb1_sell_components = {product: 0 for product in self.products}

                    # PB1 spread = PICNIC_BASKET1 - (6×CROISSANTS + 3×JAMS + 1×DJEMBE)
                    pb1_basket_qty = abs(pb1_spread_change * self.pb1_components["PICNIC_BASKET1"])
                    croissants_qty = abs(pb1_spread_change * self.pb1_components["CROISSANTS"])
                    jams_qty = abs(pb1_spread_change * self.pb1_components["JAMS"])
                    djembes_qty = abs(pb1_spread_change * self.pb1_components["DJEMBES"])

                    if pb1_spread_change > 0:  # Buying the PB1 spread
                        # Buy positive components, sell negative components
                        pb1_buy_components["PICNIC_BASKET1"] = pb1_basket_qty
                        pb1_sell_components["CROISSANTS"] = croissants_qty
                        pb1_sell_components["JAMS"] = jams_qty
                        pb1_sell_components["DJEMBES"] = djembes_qty
                    else:  # Selling the PB1 spread
                        # Sell positive components, buy negative components
                        pb1_sell_components["PICNIC_BASKET1"] = pb1_basket_qty
                        pb1_buy_components["CROISSANTS"] = croissants_qty
                        pb1_buy_components["JAMS"] = jams_qty
                        pb1_buy_components["DJEMBES"] = djembes_qty

                    # Execute the exit trade with full trading logic
                    # Step 1: Check if position limits would be violated
                    pb1_scale_factor = 1.0

                    # First, calculate the minimum scale factor needed to respect all position limits
                    for product in ["PICNIC_BASKET1", "CROISSANTS", "JAMS", "DJEMBES"]:
                        # Check buy limit
                        if pb1_buy_components[product] > 0:
                            max_buy = self.position_limits[product] - current_positions[product]
                            if pb1_buy_components[product] > max_buy:
                                product_scale_factor = max_buy / pb1_buy_components[product] if pb1_buy_components[product] > 0 else 0
                                pb1_scale_factor = min(pb1_scale_factor, product_scale_factor)
                                logger.print(f"PB1 - Buy limit for {product}: need scale factor {product_scale_factor}")

                        # Check sell limit
                        if pb1_sell_components[product] > 0:
                            max_sell = current_positions[product] + self.position_limits[product]
                            if pb1_sell_components[product] > max_sell:
                                product_scale_factor = max_sell / pb1_sell_components[product] if pb1_sell_components[product] > 0 else 0
                                pb1_scale_factor = min(pb1_scale_factor, product_scale_factor)
                                logger.print(f"PB1 - Sell limit for {product}: need scale factor {product_scale_factor}")

                    # If we need to scale down, calculate the integer number of spread units we can trade
                    if pb1_scale_factor < 1.0:
                        # Calculate how many integer spread units we can trade
                        max_pb1_spread_units = int(abs(pb1_spread_change) * pb1_scale_factor)

                        # Recalculate scale factor based on integer spread units
                        if abs(pb1_spread_change) > 0:
                            pb1_scale_factor = max_pb1_spread_units / abs(pb1_spread_change)

                        logger.print(f"PB1 - Scaling down to {max_pb1_spread_units} spread units (scale factor: {pb1_scale_factor})")

                        # Scale the spread change to an integer number of units
                        if pb1_spread_change > 0:
                            pb1_spread_change = max_pb1_spread_units
                        else:
                            pb1_spread_change = -max_pb1_spread_units

                        # Recalculate component quantities based on the integer spread change
                        pb1_buy_components = {product: 0 for product in self.products}
                        pb1_sell_components = {product: 0 for product in self.products}

                        # PB1 spread = PICNIC_BASKET1 - (6×CROISSANTS + 3×JAMS + 1×DJEMBE)
                        pb1_basket_qty = abs(pb1_spread_change * self.pb1_components["PICNIC_BASKET1"])
                        croissants_qty = abs(pb1_spread_change * self.pb1_components["CROISSANTS"])
                        jams_qty = abs(pb1_spread_change * self.pb1_components["JAMS"])
                        djembes_qty = abs(pb1_spread_change * self.pb1_components["DJEMBES"])

                        if pb1_spread_change > 0:  # Buying the PB1 spread
                            # Buy positive components, sell negative components
                            pb1_buy_components["PICNIC_BASKET1"] = pb1_basket_qty
                            pb1_sell_components["CROISSANTS"] = croissants_qty
                            pb1_sell_components["JAMS"] = jams_qty
                            pb1_sell_components["DJEMBES"] = djembes_qty
                        else:  # Selling the PB1 spread
                            # Sell positive components, buy negative components
                            pb1_sell_components["PICNIC_BASKET1"] = pb1_basket_qty
                            pb1_buy_components["CROISSANTS"] = croissants_qty
                            pb1_buy_components["JAMS"] = jams_qty
                            pb1_buy_components["DJEMBES"] = djembes_qty

                    logger.print(f"PB1 - After position limit check - Buy components: {pb1_buy_components}")
                    logger.print(f"PB1 - After position limit check - Sell components: {pb1_sell_components}")

                    # Step 2: Check available liquidity in the orderbook
                    pb1_liquidity_scale_factor = 1.0

                    # Check buy liquidity
                    for product, buy_qty in pb1_buy_components.items():
                        if buy_qty > 0:
                            if product in state.order_depths:
                                order_depth = state.order_depths[product]

                                # Find available sell orders
                                if len(order_depth.sell_orders) > 0:
                                    # Sort sell orders by price (ascending)
                                    sorted_asks = sorted(order_depth.sell_orders.items())

                                    # Determine price adjustment based on z-score thresholds
                                    # Get thresholds and price adjustments for the PB1 spread
                                    pb1_entry_price_adjustment = self.entry_price_adjustments["pb1_spread"]
                                    pb1_exit_price_adjustment = self.exit_price_adjustments["pb1_spread"]

                                    # For exits, always use exit price adjustment
                                    price_adjustment = pb1_exit_price_adjustment

                                    # Calculate the maximum price we're willing to pay
                                    mid_price = (max(order_depth.buy_orders.keys()) + min(order_depth.sell_orders.keys())) / 2

                                    # Handle special case when mid price ends in .5
                                    if mid_price % 1 == 0.5:
                                        # For mid price ending in .5, adjustment of 0 means +0.5, adjustment of 1 means +1.5
                                        max_buy_price = int(mid_price + 0.5) + price_adjustment
                                    else:
                                        # For integer mid price, adjustment is applied directly
                                        max_buy_price = int(mid_price) + price_adjustment

                                    # Calculate how much we can buy at or below our max price
                                    available_to_buy = 0
                                    for ask_price, ask_volume in sorted_asks:
                                        if ask_price <= max_buy_price:
                                            available_to_buy += abs(ask_volume)  # Volume is negative in sell_orders
                                        else:
                                            break  # Stop once we exceed our max price

                                    # Calculate scale factor based on available liquidity
                                    if available_to_buy < buy_qty:
                                        product_liquidity_factor = available_to_buy / buy_qty if buy_qty > 0 else 0
                                        pb1_liquidity_scale_factor = min(pb1_liquidity_scale_factor, product_liquidity_factor)
                                        logger.print(f"PB1 - Buy liquidity for {product}: need scale factor {product_liquidity_factor} (available: {available_to_buy}, needed: {buy_qty})")
                                else:
                                    # No sell orders available
                                    logger.print(f"PB1 - No sell orders available for {product}")
                                    pb1_liquidity_scale_factor = 0
                            else:
                                # Product not in order depths
                                logger.print(f"PB1 - No order depth available for {product}")
                                pb1_liquidity_scale_factor = 0

                    # Check sell liquidity
                    for product, sell_qty in pb1_sell_components.items():
                        if sell_qty > 0:
                            if product in state.order_depths:
                                order_depth = state.order_depths[product]

                                # Find available buy orders
                                if len(order_depth.buy_orders) > 0:
                                    # Sort buy orders by price (descending)
                                    sorted_bids = sorted(order_depth.buy_orders.items(), reverse=True)

                                    # Determine price adjustment based on z-score thresholds
                                    # Get thresholds and price adjustments for the PB1 spread
                                    pb1_entry_price_adjustment = self.entry_price_adjustments["pb1_spread"]
                                    pb1_exit_price_adjustment = self.exit_price_adjustments["pb1_spread"]

                                    # For exits, always use exit price adjustment
                                    price_adjustment = pb1_exit_price_adjustment

                                    # Calculate the minimum price we're willing to accept
                                    mid_price = (max(order_depth.buy_orders.keys()) + min(order_depth.sell_orders.keys())) / 2

                                    # Handle special case when mid price ends in .5
                                    if mid_price % 1 == 0.5:
                                        # For mid price ending in .5, adjustment of 0 means -0.5, adjustment of 1 means -1.5
                                        min_sell_price = int(mid_price - 0.5) - price_adjustment
                                    else:
                                        # For integer mid price, adjustment is applied directly
                                        min_sell_price = int(mid_price) - price_adjustment

                                    # Calculate how much we can sell at or above our min price
                                    available_to_sell = 0
                                    for bid_price, bid_volume in sorted_bids:
                                        if bid_price >= min_sell_price:
                                            available_to_sell += bid_volume
                                        else:
                                            break  # Stop once we go below our min price

                                    # Calculate scale factor based on available liquidity
                                    if available_to_sell < sell_qty:
                                        product_liquidity_factor = available_to_sell / sell_qty if sell_qty > 0 else 0
                                        pb1_liquidity_scale_factor = min(pb1_liquidity_scale_factor, product_liquidity_factor)
                                        logger.print(f"PB1 - Sell liquidity for {product}: need scale factor {product_liquidity_factor} (available: {available_to_sell}, needed: {sell_qty})")
                                else:
                                    # No buy orders available
                                    logger.print(f"PB1 - No buy orders available for {product}")
                                    pb1_liquidity_scale_factor = 0
                            else:
                                # Product not in order depths
                                logger.print(f"PB1 - No order depth available for {product}")
                                pb1_liquidity_scale_factor = 0

                    # If we need to scale down due to liquidity constraints
                    if pb1_liquidity_scale_factor < 1.0:
                        # Calculate how many integer spread units we can trade based on liquidity
                        max_pb1_liquidity_spread_units = int(abs(pb1_spread_change) * pb1_liquidity_scale_factor)

                        # Recalculate scale factor based on integer spread units
                        if abs(pb1_spread_change) > 0:
                            pb1_liquidity_scale_factor = max_pb1_liquidity_spread_units / abs(pb1_spread_change)

                        logger.print(f"PB1 - Scaling down due to liquidity constraints to {max_pb1_liquidity_spread_units} spread units (scale factor: {pb1_liquidity_scale_factor})")

                        # Scale the spread change to an integer number of units
                        if pb1_spread_change > 0:
                            pb1_spread_change = max_pb1_liquidity_spread_units
                        else:
                            pb1_spread_change = -max_pb1_liquidity_spread_units

                        # Recalculate component quantities based on the integer spread change
                        pb1_buy_components = {product: 0 for product in self.products}
                        pb1_sell_components = {product: 0 for product in self.products}

                        # PB1 spread = PICNIC_BASKET1 - (6×CROISSANTS + 3×JAMS + 1×DJEMBE)
                        pb1_basket_qty = abs(pb1_spread_change * self.pb1_components["PICNIC_BASKET1"])
                        croissants_qty = abs(pb1_spread_change * self.pb1_components["CROISSANTS"])
                        jams_qty = abs(pb1_spread_change * self.pb1_components["JAMS"])
                        djembes_qty = abs(pb1_spread_change * self.pb1_components["DJEMBES"])

                        if pb1_spread_change > 0:  # Buying the PB1 spread
                            # Buy positive components, sell negative components
                            pb1_buy_components["PICNIC_BASKET1"] = pb1_basket_qty
                            pb1_sell_components["CROISSANTS"] = croissants_qty
                            pb1_sell_components["JAMS"] = jams_qty
                            pb1_sell_components["DJEMBES"] = djembes_qty
                        else:  # Selling the PB1 spread
                            # Sell positive components, buy negative components
                            pb1_sell_components["PICNIC_BASKET1"] = pb1_basket_qty
                            pb1_buy_components["CROISSANTS"] = croissants_qty
                            pb1_buy_components["JAMS"] = jams_qty
                            pb1_buy_components["DJEMBES"] = djembes_qty

                    logger.print(f"PB1 - After liquidity check - Buy components: {pb1_buy_components}")
                    logger.print(f"PB1 - After liquidity check - Sell components: {pb1_sell_components}")

                    # Execute orders with proper liquidity taking
                    pb1_actual_buys = {product: 0 for product in self.products}
                    pb1_actual_sells = {product: 0 for product in self.products}

                    # Process buy orders
                    for product, buy_qty in pb1_buy_components.items():
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
                                            # We've already checked that this order is within our price limits during the liquidity check
                                            # So we can just use the ask_price directly
                                            # Ensure the price is an integer
                                            adjusted_price = int(ask_price)

                                            if product not in result:
                                                result[product] = []
                                            result[product].append(Order(product, adjusted_price, executable_volume_int))

                                            # Update actual buys with the integer volume
                                            pb1_actual_buys[product] += executable_volume_int

                                            # Update remaining to buy with the integer volume
                                            remaining_to_buy -= executable_volume_int

                                            # If we've filled the entire order, break
                                            if remaining_to_buy <= 0:
                                                break

                    # Process sell orders
                    for product, sell_qty in pb1_sell_components.items():
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
                                            # We've already checked that this order is within our price limits during the liquidity check
                                            # So we can just use the bid_price directly
                                            # Ensure the price is an integer
                                            adjusted_price = int(bid_price)

                                            if product not in result:
                                                result[product] = []
                                            result[product].append(Order(product, adjusted_price, -executable_volume_int))

                                            # Update actual sells with the integer volume
                                            pb1_actual_sells[product] += executable_volume_int

                                            # Update remaining to sell with the integer volume
                                            remaining_to_sell -= executable_volume_int

                                            # If we've filled the entire order, break
                                            if remaining_to_sell <= 0:
                                                break

                    logger.print(f"PB1 - Actual buys: {pb1_actual_buys}")
                    logger.print(f"PB1 - Actual sells: {pb1_actual_sells}")

                    # Calculate execution ratio for the spread
                    pb1_spread_execution_ratio = 1.0
                    if pb1_spread_change != 0:
                        # Determine the limiting factor for the spread execution
                        if pb1_spread_change > 0:  # Buying the spread
                            for product, buy_qty in pb1_buy_components.items():
                                if buy_qty > 0:
                                    execution_ratio = pb1_actual_buys[product] / buy_qty if buy_qty > 0 else 1.0
                                    pb1_spread_execution_ratio = min(pb1_spread_execution_ratio, execution_ratio)

                            for product, sell_qty in pb1_sell_components.items():
                                if sell_qty > 0:
                                    execution_ratio = pb1_actual_sells[product] / sell_qty if sell_qty > 0 else 1.0
                                    pb1_spread_execution_ratio = min(pb1_spread_execution_ratio, execution_ratio)
                        else:  # Selling the spread
                            for product, buy_qty in pb1_buy_components.items():
                                if buy_qty > 0:
                                    execution_ratio = pb1_actual_buys[product] / buy_qty if buy_qty > 0 else 1.0
                                    pb1_spread_execution_ratio = min(pb1_spread_execution_ratio, execution_ratio)

                            for product, sell_qty in pb1_sell_components.items():
                                if sell_qty > 0:
                                    execution_ratio = pb1_actual_sells[product] / sell_qty if sell_qty > 0 else 1.0
                                    pb1_spread_execution_ratio = min(pb1_spread_execution_ratio, execution_ratio)

                    # Calculate actual spread position change
                    # We need to ensure the actual spread change is an integer number of spread units
                    if abs(pb1_spread_change) > 0:
                        # Calculate the integer number of spread units that were executed
                        executed_pb1_spread_units = int(abs(pb1_spread_change) * pb1_spread_execution_ratio)

                        # Set the actual spread change with the correct sign
                        if pb1_spread_change > 0:
                            pb1_actual_spread_change = executed_pb1_spread_units
                        else:
                            pb1_actual_spread_change = -executed_pb1_spread_units
                    else:
                        pb1_actual_spread_change = 0

                    logger.print(f"PB1 - Spread execution ratio: {pb1_spread_execution_ratio}, actual change: {pb1_actual_spread_change}")

                    # Update spread position
                    self.spread_positions["pb1_spread"] = int(self.spread_positions["pb1_spread"] + pb1_actual_spread_change)

                    # Update current positions for subsequent trades
                    for product in self.products:
                        current_positions[product] += pb1_actual_buys[product] - pb1_actual_sells[product]

                    logger.print(f"Exited PB1 spread position, new position: {self.spread_positions['pb1_spread']}")

        # Now consider trading PB2 and PB1 spreads based on the custom spread direction
        # We'll only enter new positions if we're trading the custom spread and their max positions are not 0
        # Also, we'll only trade if the entry thresholds are reachable (not set to an unreasonably high value)
        if is_custom_spread_trading and (self.max_spread_positions["pb1_spread"] > 0 or self.max_spread_positions["pb2_spread"] > 0):
            # We've already defined these variables above

            # Determine if we can trade each spread based on custom spread direction
            can_trade_pb1 = False
            pb1_target_direction = 0

            can_trade_pb2 = False
            pb2_target_direction = 0

            if custom_spread_direction > 0:  # Buying custom spread
                # We're allowed to buy PB1 spread and sell PB2 spread
                can_trade_pb1 = True
                pb1_target_direction = 1

                can_trade_pb2 = True
                pb2_target_direction = -1

                logger.print(f"Custom spread direction: {custom_spread_direction} (buying) - Can trade PB1: {can_trade_pb1} (direction: {pb1_target_direction}), Can trade PB2: {can_trade_pb2} (direction: {pb2_target_direction})")
            elif custom_spread_direction < 0:  # Selling custom spread
                # We're allowed to sell PB1 spread and buy PB2 spread
                can_trade_pb1 = True
                pb1_target_direction = -1

                can_trade_pb2 = True
                pb2_target_direction = 1

                logger.print(f"Custom spread direction: {custom_spread_direction} (selling) - Can trade PB1: {can_trade_pb1} (direction: {pb1_target_direction}), Can trade PB2: {can_trade_pb2} (direction: {pb2_target_direction})")
            else:
                logger.print(f"Custom spread direction: {custom_spread_direction} (neutral) - Cannot trade PB1 or PB2")

            # Create a list of spreads to process for entry signals
            # We've already handled exits above
            spreads_to_process = []

            # Only add spreads that are not within exit threshold and can be traded
            if not pb2_within_exit and can_trade_pb2 and self.max_spread_positions["pb2_spread"] > 0:
                spreads_to_process.append("pb2_spread")

            if not pb1_within_exit and can_trade_pb1 and self.max_spread_positions["pb1_spread"] > 0:
                spreads_to_process.append("pb1_spread")

            logger.print(f"Spreads to process in order: {spreads_to_process}")

            # Process each spread in the determined order
            for spread_name in spreads_to_process:
                if spread_name == "pb2_spread":
                    # Process PB2 spread
                    pb2_signal = None

                    if can_trade_pb2:
                        if pb2_target_direction > 0 and pb2_z_score < -pb2_entry_threshold:
                            # Buy the spread if it's below the negative entry threshold and we're allowed to buy
                            # Scale the signal based on how far the z-score is from the entry threshold
                            # This ensures that the signal is proportional to the z-score
                            z_score_ratio = min(abs(pb2_z_score) / pb2_entry_threshold, 2.0)  # Cap at 2.0 to avoid extreme positions
                            pb2_signal = int(pb2_max_position * z_score_ratio)
                            logger.print(f"PB2 - Buy signal generated: target direction {pb2_target_direction}, z-score {pb2_z_score} < -{pb2_entry_threshold}, z-score ratio: {z_score_ratio}, signal: {pb2_signal}")
                        elif pb2_target_direction < 0 and pb2_z_score > pb2_entry_threshold:
                            # Sell the spread if it's above the entry threshold and we're allowed to sell
                            # Scale the signal based on how far the z-score is from the entry threshold
                            # This ensures that the signal is proportional to the z-score
                            z_score_ratio = min(abs(pb2_z_score) / pb2_entry_threshold, 2.0)  # Cap at 2.0 to avoid extreme positions
                            pb2_signal = -int(pb2_max_position * z_score_ratio)
                            logger.print(f"PB2 - Sell signal generated: target direction {pb2_target_direction}, z-score {pb2_z_score} > {pb2_entry_threshold}, z-score ratio: {z_score_ratio}, signal: {pb2_signal}")
                        else:
                            logger.print(f"PB2 - No signal generated: target direction {pb2_target_direction}, z-score {pb2_z_score}, entry threshold: {pb2_entry_threshold}")

                    logger.print(f"PB2 spread signal: {pb2_signal}, can trade: {can_trade_pb2}, target direction: {pb2_target_direction}, within exit: {pb2_within_exit}")

                    # Calculate target position for PB2 spread
                    if pb2_signal is not None:
                        # Calculate position change for PB2 spread
                        pb2_target_position = int(pb2_signal)
                        pb2_spread_change = int(pb2_target_position - self.spread_positions["pb2_spread"])

                        logger.print(f"PB2 spread change: {pb2_spread_change}")

                        # If there's a change in position, trade the PB2 spread
                        if pb2_spread_change != 0:
                            # Calculate component quantities for PB2 spread change
                            pb2_buy_components = {product: 0 for product in self.products}
                            pb2_sell_components = {product: 0 for product in self.products}

                            # PB2 spread = PICNIC_BASKET2 - (4×CROISSANTS + 2×JAMS)
                            pb2_basket_qty = abs(pb2_spread_change * self.pb2_components["PICNIC_BASKET2"])
                            croissants_qty = abs(pb2_spread_change * self.pb2_components["CROISSANTS"])
                            jams_qty = abs(pb2_spread_change * self.pb2_components["JAMS"])

                            if pb2_spread_change > 0:  # Buying the PB2 spread
                                # Buy positive components, sell negative components
                                pb2_buy_components["PICNIC_BASKET2"] = pb2_basket_qty
                                pb2_sell_components["CROISSANTS"] = croissants_qty
                                pb2_sell_components["JAMS"] = jams_qty
                            else:  # Selling the PB2 spread
                                # Sell positive components, buy negative components
                                pb2_sell_components["PICNIC_BASKET2"] = pb2_basket_qty
                                pb2_buy_components["CROISSANTS"] = croissants_qty
                                pb2_buy_components["JAMS"] = jams_qty

                            logger.print(f"PB2 spread - Buy components: {pb2_buy_components}")
                            logger.print(f"PB2 spread - Sell components: {pb2_sell_components}")

                            # Step 1: Check if position limits would be violated
                            pb2_scale_factor = 1.0

                            # First, calculate the minimum scale factor needed to respect all position limits
                            for product in ["PICNIC_BASKET2", "CROISSANTS", "JAMS"]:
                                # Check buy limit
                                if pb2_buy_components[product] > 0:
                                    max_buy = self.position_limits[product] - current_positions[product]
                                    if pb2_buy_components[product] > max_buy:
                                        product_scale_factor = max_buy / pb2_buy_components[product] if pb2_buy_components[product] > 0 else 0
                                        pb2_scale_factor = min(pb2_scale_factor, product_scale_factor)
                                        logger.print(f"PB2 - Buy limit for {product}: need scale factor {product_scale_factor}")

                                # Check sell limit
                                if pb2_sell_components[product] > 0:
                                    max_sell = current_positions[product] + self.position_limits[product]
                                    if pb2_sell_components[product] > max_sell:
                                        product_scale_factor = max_sell / pb2_sell_components[product] if pb2_sell_components[product] > 0 else 0
                                        pb2_scale_factor = min(pb2_scale_factor, product_scale_factor)
                                        logger.print(f"PB2 - Sell limit for {product}: need scale factor {product_scale_factor}")

                            # If we need to scale down, calculate the integer number of spread units we can trade
                            if pb2_scale_factor < 1.0:
                                # Calculate how many integer spread units we can trade
                                max_pb2_spread_units = int(abs(pb2_spread_change) * pb2_scale_factor)

                                # Recalculate scale factor based on integer spread units
                                if abs(pb2_spread_change) > 0:
                                    pb2_scale_factor = max_pb2_spread_units / abs(pb2_spread_change)

                                logger.print(f"PB2 - Scaling down to {max_pb2_spread_units} spread units (scale factor: {pb2_scale_factor})")

                                # Scale the spread change to an integer number of units
                                if pb2_spread_change > 0:
                                    pb2_spread_change = max_pb2_spread_units
                                else:
                                    pb2_spread_change = -max_pb2_spread_units

                                # Recalculate component quantities based on the integer spread change
                                pb2_buy_components = {product: 0 for product in self.products}
                                pb2_sell_components = {product: 0 for product in self.products}

                                # PB2 spread = PICNIC_BASKET2 - (4×CROISSANTS + 2×JAMS)
                                pb2_basket_qty = abs(pb2_spread_change * self.pb2_components["PICNIC_BASKET2"])
                                croissants_qty = abs(pb2_spread_change * self.pb2_components["CROISSANTS"])
                                jams_qty = abs(pb2_spread_change * self.pb2_components["JAMS"])

                                if pb2_spread_change > 0:  # Buying the PB2 spread
                                    # Buy positive components, sell negative components
                                    pb2_buy_components["PICNIC_BASKET2"] = pb2_basket_qty
                                    pb2_sell_components["CROISSANTS"] = croissants_qty
                                    pb2_sell_components["JAMS"] = jams_qty
                                else:  # Selling the PB2 spread
                                    # Sell positive components, buy negative components
                                    pb2_sell_components["PICNIC_BASKET2"] = pb2_basket_qty
                                    pb2_buy_components["CROISSANTS"] = croissants_qty
                                    pb2_buy_components["JAMS"] = jams_qty

                            logger.print(f"PB2 - After position limit check - Buy components: {pb2_buy_components}")
                            logger.print(f"PB2 - After position limit check - Sell components: {pb2_sell_components}")

                            # Step 2: Check available liquidity in the orderbook
                            pb2_liquidity_scale_factor = 1.0

                            # Check buy liquidity
                            for product, buy_qty in pb2_buy_components.items():
                                if buy_qty > 0:
                                    if product in state.order_depths:
                                        order_depth = state.order_depths[product]

                                        # Find available sell orders
                                        if len(order_depth.sell_orders) > 0:
                                            # Sort sell orders by price (ascending)
                                            sorted_asks = sorted(order_depth.sell_orders.items())

                                            # Determine price adjustment based on z-score thresholds
                                            # Get thresholds and price adjustments for the PB2 spread
                                            pb2_entry_price_adjustment = self.entry_price_adjustments["pb2_spread"]
                                            pb2_exit_price_adjustment = self.exit_price_adjustments["pb2_spread"]

                                            if pb2_within_exit:
                                                price_adjustment = pb2_exit_price_adjustment
                                            else:
                                                price_adjustment = pb2_entry_price_adjustment

                                            # Calculate the maximum price we're willing to pay
                                            mid_price = (max(order_depth.buy_orders.keys()) + min(order_depth.sell_orders.keys())) / 2

                                            # Handle special case when mid price ends in .5
                                            if mid_price % 1 == 0.5:
                                                # For mid price ending in .5, adjustment of 0 means +0.5, adjustment of 1 means +1.5
                                                max_buy_price = int(mid_price + 0.5) + price_adjustment
                                            else:
                                                # For integer mid price, adjustment is applied directly
                                                max_buy_price = int(mid_price) + price_adjustment

                                            # Calculate how much we can buy at or below our max price
                                            available_to_buy = 0
                                            for ask_price, ask_volume in sorted_asks:
                                                if ask_price <= max_buy_price:
                                                    available_to_buy += abs(ask_volume)  # Volume is negative in sell_orders
                                                else:
                                                    break  # Stop once we exceed our max price

                                            # Calculate scale factor based on available liquidity
                                            if available_to_buy < buy_qty:
                                                product_liquidity_factor = available_to_buy / buy_qty if buy_qty > 0 else 0
                                                pb2_liquidity_scale_factor = min(pb2_liquidity_scale_factor, product_liquidity_factor)
                                                logger.print(f"PB2 - Buy liquidity for {product}: need scale factor {product_liquidity_factor} (available: {available_to_buy}, needed: {buy_qty})")
                                        else:
                                            # No sell orders available
                                            logger.print(f"PB2 - No sell orders available for {product}")
                                            pb2_liquidity_scale_factor = 0
                                    else:
                                        # Product not in order depths
                                        logger.print(f"PB2 - No order depth available for {product}")
                                        pb2_liquidity_scale_factor = 0

                            # Check sell liquidity
                            for product, sell_qty in pb2_sell_components.items():
                                if sell_qty > 0:
                                    if product in state.order_depths:
                                        order_depth = state.order_depths[product]

                                        # Find available buy orders
                                        if len(order_depth.buy_orders) > 0:
                                            # Sort buy orders by price (descending)
                                            sorted_bids = sorted(order_depth.buy_orders.items(), reverse=True)

                                            # Determine price adjustment based on z-score thresholds
                                            # Get thresholds and price adjustments for the PB2 spread
                                            pb2_entry_price_adjustment = self.entry_price_adjustments["pb2_spread"]
                                            pb2_exit_price_adjustment = self.exit_price_adjustments["pb2_spread"]

                                            if pb2_within_exit:
                                                price_adjustment = pb2_exit_price_adjustment
                                            else:
                                                price_adjustment = pb2_entry_price_adjustment

                                            # Calculate the minimum price we're willing to accept
                                            mid_price = (max(order_depth.buy_orders.keys()) + min(order_depth.sell_orders.keys())) / 2

                                            # Handle special case when mid price ends in .5
                                            if mid_price % 1 == 0.5:
                                                # For mid price ending in .5, adjustment of 0 means -0.5, adjustment of 1 means -1.5
                                                min_sell_price = int(mid_price - 0.5) - price_adjustment
                                            else:
                                                # For integer mid price, adjustment is applied directly
                                                min_sell_price = int(mid_price) - price_adjustment

                                            # Calculate how much we can sell at or above our min price
                                            available_to_sell = 0
                                            for bid_price, bid_volume in sorted_bids:
                                                if bid_price >= min_sell_price:
                                                    available_to_sell += bid_volume
                                                else:
                                                    break  # Stop once we go below our min price

                                            # Calculate scale factor based on available liquidity
                                            if available_to_sell < sell_qty:
                                                product_liquidity_factor = available_to_sell / sell_qty if sell_qty > 0 else 0
                                                pb2_liquidity_scale_factor = min(pb2_liquidity_scale_factor, product_liquidity_factor)
                                                logger.print(f"PB2 - Sell liquidity for {product}: need scale factor {product_liquidity_factor} (available: {available_to_sell}, needed: {sell_qty})")
                                        else:
                                            # No buy orders available
                                            logger.print(f"PB2 - No buy orders available for {product}")
                                            pb2_liquidity_scale_factor = 0
                                    else:
                                        # Product not in order depths
                                        logger.print(f"PB2 - No order depth available for {product}")
                                        pb2_liquidity_scale_factor = 0

                            # If we need to scale down due to liquidity constraints
                            if pb2_liquidity_scale_factor < 1.0:
                                # Calculate how many integer spread units we can trade based on liquidity
                                max_pb2_liquidity_spread_units = int(abs(pb2_spread_change) * pb2_liquidity_scale_factor)

                                # Recalculate scale factor based on integer spread units
                                if abs(pb2_spread_change) > 0:
                                    pb2_liquidity_scale_factor = max_pb2_liquidity_spread_units / abs(pb2_spread_change)

                                logger.print(f"PB2 - Scaling down due to liquidity constraints to {max_pb2_liquidity_spread_units} spread units (scale factor: {pb2_liquidity_scale_factor})")

                                # Scale the spread change to an integer number of units
                                if pb2_spread_change > 0:
                                    pb2_spread_change = max_pb2_liquidity_spread_units
                                else:
                                    pb2_spread_change = -max_pb2_liquidity_spread_units

                                # Recalculate component quantities based on the integer spread change
                                pb2_buy_components = {product: 0 for product in self.products}
                                pb2_sell_components = {product: 0 for product in self.products}

                                # PB2 spread = PICNIC_BASKET2 - (4×CROISSANTS + 2×JAMS)
                                pb2_basket_qty = abs(pb2_spread_change * self.pb2_components["PICNIC_BASKET2"])
                                croissants_qty = abs(pb2_spread_change * self.pb2_components["CROISSANTS"])
                                jams_qty = abs(pb2_spread_change * self.pb2_components["JAMS"])

                                if pb2_spread_change > 0:  # Buying the PB2 spread
                                    # Buy positive components, sell negative components
                                    pb2_buy_components["PICNIC_BASKET2"] = pb2_basket_qty
                                    pb2_sell_components["CROISSANTS"] = croissants_qty
                                    pb2_sell_components["JAMS"] = jams_qty
                                else:  # Selling the PB2 spread
                                    # Sell positive components, buy negative components
                                    pb2_sell_components["PICNIC_BASKET2"] = pb2_basket_qty
                                    pb2_buy_components["CROISSANTS"] = croissants_qty
                                    pb2_buy_components["JAMS"] = jams_qty

                            logger.print(f"PB2 - After liquidity check - Buy components: {pb2_buy_components}")
                            logger.print(f"PB2 - After liquidity check - Sell components: {pb2_sell_components}")

                            # Execute orders with proper liquidity taking
                            pb2_actual_buys = {product: 0 for product in self.products}
                            pb2_actual_sells = {product: 0 for product in self.products}

                            # Process buy orders
                            for product, buy_qty in pb2_buy_components.items():
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
                                                    # We've already checked that this order is within our price limits during the liquidity check
                                                    # So we can just use the ask_price directly
                                                    # Ensure the price is an integer
                                                    adjusted_price = int(ask_price)

                                                    if product not in result:
                                                        result[product] = []
                                                    result[product].append(Order(product, adjusted_price, executable_volume_int))

                                                    # Update actual buys with the integer volume
                                                    pb2_actual_buys[product] += executable_volume_int

                                                    # Update remaining to buy with the integer volume
                                                    remaining_to_buy -= executable_volume_int

                                                    # If we've filled the entire order, break
                                                    if remaining_to_buy <= 0:
                                                        break

                            # Process sell orders
                            for product, sell_qty in pb2_sell_components.items():
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
                                                    # We've already checked that this order is within our price limits during the liquidity check
                                                    # So we can just use the bid_price directly
                                                    # Ensure the price is an integer
                                                    adjusted_price = int(bid_price)

                                                    if product not in result:
                                                        result[product] = []
                                                    result[product].append(Order(product, adjusted_price, -executable_volume_int))

                                                    # Update actual sells with the integer volume
                                                    pb2_actual_sells[product] += executable_volume_int

                                                    # Update remaining to sell with the integer volume
                                                    remaining_to_sell -= executable_volume_int

                                                    # If we've filled the entire order, break
                                                    if remaining_to_sell <= 0:
                                                        break

                            logger.print(f"PB2 - Actual buys: {pb2_actual_buys}")
                            logger.print(f"PB2 - Actual sells: {pb2_actual_sells}")

                            # Calculate execution ratio for the spread
                            pb2_spread_execution_ratio = 1.0
                            if pb2_spread_change != 0:
                                # Determine the limiting factor for the spread execution
                                if pb2_spread_change > 0:  # Buying the spread
                                    for product, buy_qty in pb2_buy_components.items():
                                        if buy_qty > 0:
                                            execution_ratio = pb2_actual_buys[product] / buy_qty if buy_qty > 0 else 1.0
                                            pb2_spread_execution_ratio = min(pb2_spread_execution_ratio, execution_ratio)

                                    for product, sell_qty in pb2_sell_components.items():
                                        if sell_qty > 0:
                                            execution_ratio = pb2_actual_sells[product] / sell_qty if sell_qty > 0 else 1.0
                                            pb2_spread_execution_ratio = min(pb2_spread_execution_ratio, execution_ratio)
                                else:  # Selling the spread
                                    for product, buy_qty in pb2_buy_components.items():
                                        if buy_qty > 0:
                                            execution_ratio = pb2_actual_buys[product] / buy_qty if buy_qty > 0 else 1.0
                                            pb2_spread_execution_ratio = min(pb2_spread_execution_ratio, execution_ratio)

                                    for product, sell_qty in pb2_sell_components.items():
                                        if sell_qty > 0:
                                            execution_ratio = pb2_actual_sells[product] / sell_qty if sell_qty > 0 else 1.0
                                            pb2_spread_execution_ratio = min(pb2_spread_execution_ratio, execution_ratio)

                            # Calculate actual spread position change
                            # We need to ensure the actual spread change is an integer number of spread units
                            if abs(pb2_spread_change) > 0:
                                # Calculate the integer number of spread units that were executed
                                executed_pb2_spread_units = int(abs(pb2_spread_change) * pb2_spread_execution_ratio)

                                # Set the actual spread change with the correct sign
                                if pb2_spread_change > 0:
                                    pb2_actual_spread_change = executed_pb2_spread_units
                                else:
                                    pb2_actual_spread_change = -executed_pb2_spread_units
                            else:
                                pb2_actual_spread_change = 0

                            logger.print(f"PB2 - Spread execution ratio: {pb2_spread_execution_ratio}, actual change: {pb2_actual_spread_change}")

                            # Update spread position
                            self.spread_positions["pb2_spread"] = int(self.spread_positions["pb2_spread"] + pb2_actual_spread_change)

                            # Update current positions for subsequent trades
                            for product in self.products:
                                current_positions[product] += pb2_actual_buys[product] - pb2_actual_sells[product]

                elif spread_name == "pb1_spread":
                    # Process PB1 spread
                    pb1_signal = None

                    if can_trade_pb1:
                        if pb1_target_direction > 0 and pb1_z_score < -pb1_entry_threshold:
                            # Buy the spread if it's below the negative entry threshold and we're allowed to buy
                            pb1_signal = pb1_max_position
                            logger.print(f"PB1 - Buy signal generated: target direction {pb1_target_direction}, z-score {pb1_z_score} < -{pb1_entry_threshold}, signal: {pb1_signal}")
                        elif pb1_target_direction < 0 and pb1_z_score > pb1_entry_threshold:
                            # Sell the spread if it's above the entry threshold and we're allowed to sell
                            pb1_signal = -pb1_max_position
                            logger.print(f"PB1 - Sell signal generated: target direction {pb1_target_direction}, z-score {pb1_z_score} > {pb1_entry_threshold}, signal: {pb1_signal}")
                        else:
                            logger.print(f"PB1 - No signal generated: target direction {pb1_target_direction}, z-score {pb1_z_score}, entry threshold: {pb1_entry_threshold}")

                    logger.print(f"PB1 spread signal: {pb1_signal}, can trade: {can_trade_pb1}, target direction: {pb1_target_direction}, within exit: {pb1_within_exit}")

                    # Calculate target position for PB1 spread
                    if pb1_signal is not None:
                        # Calculate position change for PB1 spread
                        pb1_target_position = int(pb1_signal)
                        pb1_spread_change = int(pb1_target_position - self.spread_positions["pb1_spread"])

                        logger.print(f"PB1 spread change: {pb1_spread_change}")

                        # If there's a change in position, trade the PB1 spread
                        if pb1_spread_change != 0:
                            # Calculate component quantities for PB1 spread change
                            pb1_buy_components = {product: 0 for product in self.products}
                            pb1_sell_components = {product: 0 for product in self.products}

                            # PB1 spread = PICNIC_BASKET1 - (6×CROISSANTS + 3×JAMS + 1×DJEMBE)
                            pb1_basket_qty = abs(pb1_spread_change * self.pb1_components["PICNIC_BASKET1"])
                            croissants_qty = abs(pb1_spread_change * self.pb1_components["CROISSANTS"])
                            jams_qty = abs(pb1_spread_change * self.pb1_components["JAMS"])
                            djembes_qty = abs(pb1_spread_change * self.pb1_components["DJEMBES"])

                            if pb1_spread_change > 0:  # Buying the PB1 spread
                                # Buy positive components, sell negative components
                                pb1_buy_components["PICNIC_BASKET1"] = pb1_basket_qty
                                pb1_sell_components["CROISSANTS"] = croissants_qty
                                pb1_sell_components["JAMS"] = jams_qty
                                pb1_sell_components["DJEMBES"] = djembes_qty
                            else:  # Selling the PB1 spread
                                # Sell positive components, buy negative components
                                pb1_sell_components["PICNIC_BASKET1"] = pb1_basket_qty
                                pb1_buy_components["CROISSANTS"] = croissants_qty
                                pb1_buy_components["JAMS"] = jams_qty
                                pb1_buy_components["DJEMBES"] = djembes_qty

                            logger.print(f"PB1 spread - Buy components: {pb1_buy_components}")
                            logger.print(f"PB1 spread - Sell components: {pb1_sell_components}")

                            # Step 1: Check if position limits would be violated
                            pb1_scale_factor = 1.0

                            # First, calculate the minimum scale factor needed to respect all position limits
                            for product in ["PICNIC_BASKET1", "CROISSANTS", "JAMS", "DJEMBES"]:
                                # Check buy limit
                                if pb1_buy_components[product] > 0:
                                    max_buy = self.position_limits[product] - current_positions[product]
                                    if pb1_buy_components[product] > max_buy:
                                        product_scale_factor = max_buy / pb1_buy_components[product] if pb1_buy_components[product] > 0 else 0
                                        pb1_scale_factor = min(pb1_scale_factor, product_scale_factor)
                                        logger.print(f"PB1 - Buy limit for {product}: need scale factor {product_scale_factor}")

                                # Check sell limit
                                if pb1_sell_components[product] > 0:
                                    max_sell = current_positions[product] + self.position_limits[product]
                                    if pb1_sell_components[product] > max_sell:
                                        product_scale_factor = max_sell / pb1_sell_components[product] if pb1_sell_components[product] > 0 else 0
                                        pb1_scale_factor = min(pb1_scale_factor, product_scale_factor)
                                        logger.print(f"PB1 - Sell limit for {product}: need scale factor {product_scale_factor}")

                            # If we need to scale down, calculate the integer number of spread units we can trade
                            if pb1_scale_factor < 1.0:
                                # Calculate how many integer spread units we can trade
                                max_pb1_spread_units = int(abs(pb1_spread_change) * pb1_scale_factor)

                                # Recalculate scale factor based on integer spread units
                                if abs(pb1_spread_change) > 0:
                                    pb1_scale_factor = max_pb1_spread_units / abs(pb1_spread_change)

                                logger.print(f"PB1 - Scaling down to {max_pb1_spread_units} spread units (scale factor: {pb1_scale_factor})")

                                # Scale the spread change to an integer number of units
                                if pb1_spread_change > 0:
                                    pb1_spread_change = max_pb1_spread_units
                                else:
                                    pb1_spread_change = -max_pb1_spread_units

                                # Recalculate component quantities based on the integer spread change
                                pb1_buy_components = {product: 0 for product in self.products}
                                pb1_sell_components = {product: 0 for product in self.products}

                                # PB1 spread = PICNIC_BASKET1 - (6×CROISSANTS + 3×JAMS + 1×DJEMBE)
                                pb1_basket_qty = abs(pb1_spread_change * self.pb1_components["PICNIC_BASKET1"])
                                croissants_qty = abs(pb1_spread_change * self.pb1_components["CROISSANTS"])
                                jams_qty = abs(pb1_spread_change * self.pb1_components["JAMS"])
                                djembes_qty = abs(pb1_spread_change * self.pb1_components["DJEMBES"])

                                if pb1_spread_change > 0:  # Buying the PB1 spread
                                    # Buy positive components, sell negative components
                                    pb1_buy_components["PICNIC_BASKET1"] = pb1_basket_qty
                                    pb1_sell_components["CROISSANTS"] = croissants_qty
                                    pb1_sell_components["JAMS"] = jams_qty
                                    pb1_sell_components["DJEMBES"] = djembes_qty
                                else:  # Selling the PB1 spread
                                    # Sell positive components, buy negative components
                                    pb1_sell_components["PICNIC_BASKET1"] = pb1_basket_qty
                                    pb1_buy_components["CROISSANTS"] = croissants_qty
                                    pb1_buy_components["JAMS"] = jams_qty
                                    pb1_buy_components["DJEMBES"] = djembes_qty

                            logger.print(f"PB1 - After position limit check - Buy components: {pb1_buy_components}")
                            logger.print(f"PB1 - After position limit check - Sell components: {pb1_sell_components}")

                            # Step 2: Check available liquidity in the orderbook
                            pb1_liquidity_scale_factor = 1.0

                            # Check buy liquidity
                            for product, buy_qty in pb1_buy_components.items():
                                if buy_qty > 0:
                                    if product in state.order_depths:
                                        order_depth = state.order_depths[product]

                                        # Find available sell orders
                                        if len(order_depth.sell_orders) > 0:
                                            # Sort sell orders by price (ascending)
                                            sorted_asks = sorted(order_depth.sell_orders.items())

                                            # Determine price adjustment based on z-score thresholds
                                            # Get thresholds and price adjustments for the PB1 spread
                                            pb1_entry_price_adjustment = self.entry_price_adjustments["pb1_spread"]
                                            pb1_exit_price_adjustment = self.exit_price_adjustments["pb1_spread"]

                                            if pb1_within_exit:
                                                price_adjustment = pb1_exit_price_adjustment
                                            else:
                                                price_adjustment = pb1_entry_price_adjustment

                                            # Calculate the maximum price we're willing to pay
                                            mid_price = (max(order_depth.buy_orders.keys()) + min(order_depth.sell_orders.keys())) / 2

                                            # Handle special case when mid price ends in .5
                                            if mid_price % 1 == 0.5:
                                                # For mid price ending in .5, adjustment of 0 means +0.5, adjustment of 1 means +1.5
                                                max_buy_price = int(mid_price + 0.5) + price_adjustment
                                            else:
                                                # For integer mid price, adjustment is applied directly
                                                max_buy_price = int(mid_price) + price_adjustment

                                            # Calculate how much we can buy at or below our max price
                                            available_to_buy = 0
                                            for ask_price, ask_volume in sorted_asks:
                                                if ask_price <= max_buy_price:
                                                    available_to_buy += abs(ask_volume)  # Volume is negative in sell_orders
                                                else:
                                                    break  # Stop once we exceed our max price

                                            # Calculate scale factor based on available liquidity
                                            if available_to_buy < buy_qty:
                                                product_liquidity_factor = available_to_buy / buy_qty if buy_qty > 0 else 0
                                                pb1_liquidity_scale_factor = min(pb1_liquidity_scale_factor, product_liquidity_factor)
                                                logger.print(f"PB1 - Buy liquidity for {product}: need scale factor {product_liquidity_factor} (available: {available_to_buy}, needed: {buy_qty})")
                                        else:
                                            # No sell orders available
                                            logger.print(f"PB1 - No sell orders available for {product}")
                                            pb1_liquidity_scale_factor = 0
                                    else:
                                        # Product not in order depths
                                        logger.print(f"PB1 - No order depth available for {product}")
                                        pb1_liquidity_scale_factor = 0

                            # Check sell liquidity
                            for product, sell_qty in pb1_sell_components.items():
                                if sell_qty > 0:
                                    if product in state.order_depths:
                                        order_depth = state.order_depths[product]

                                        # Find available buy orders
                                        if len(order_depth.buy_orders) > 0:
                                            # Sort buy orders by price (descending)
                                            sorted_bids = sorted(order_depth.buy_orders.items(), reverse=True)

                                            # Determine price adjustment based on z-score thresholds
                                            # Get thresholds and price adjustments for the PB1 spread
                                            pb1_entry_price_adjustment = self.entry_price_adjustments["pb1_spread"]
                                            pb1_exit_price_adjustment = self.exit_price_adjustments["pb1_spread"]

                                            if pb1_within_exit:
                                                price_adjustment = pb1_exit_price_adjustment
                                            else:
                                                price_adjustment = pb1_entry_price_adjustment

                                            # Calculate the minimum price we're willing to accept
                                            mid_price = (max(order_depth.buy_orders.keys()) + min(order_depth.sell_orders.keys())) / 2

                                            # Handle special case when mid price ends in .5
                                            if mid_price % 1 == 0.5:
                                                # For mid price ending in .5, adjustment of 0 means -0.5, adjustment of 1 means -1.5
                                                min_sell_price = int(mid_price - 0.5) - price_adjustment
                                            else:
                                                # For integer mid price, adjustment is applied directly
                                                min_sell_price = int(mid_price) - price_adjustment

                                            # Calculate how much we can sell at or above our min price
                                            available_to_sell = 0
                                            for bid_price, bid_volume in sorted_bids:
                                                if bid_price >= min_sell_price:
                                                    available_to_sell += bid_volume
                                                else:
                                                    break  # Stop once we go below our min price

                                            # Calculate scale factor based on available liquidity
                                            if available_to_sell < sell_qty:
                                                product_liquidity_factor = available_to_sell / sell_qty if sell_qty > 0 else 0
                                                pb1_liquidity_scale_factor = min(pb1_liquidity_scale_factor, product_liquidity_factor)
                                                logger.print(f"PB1 - Sell liquidity for {product}: need scale factor {product_liquidity_factor} (available: {available_to_sell}, needed: {sell_qty})")
                                        else:
                                            # No buy orders available
                                            logger.print(f"PB1 - No buy orders available for {product}")
                                            pb1_liquidity_scale_factor = 0
                                    else:
                                        # Product not in order depths
                                        logger.print(f"PB1 - No order depth available for {product}")
                                        pb1_liquidity_scale_factor = 0

                            # If we need to scale down due to liquidity constraints
                            if pb1_liquidity_scale_factor < 1.0:
                                # Calculate how many integer spread units we can trade based on liquidity
                                max_pb1_liquidity_spread_units = int(abs(pb1_spread_change) * pb1_liquidity_scale_factor)

                                # Recalculate scale factor based on integer spread units
                                if abs(pb1_spread_change) > 0:
                                    pb1_liquidity_scale_factor = max_pb1_liquidity_spread_units / abs(pb1_spread_change)

                                logger.print(f"PB1 - Scaling down due to liquidity constraints to {max_pb1_liquidity_spread_units} spread units (scale factor: {pb1_liquidity_scale_factor})")

                                # Scale the spread change to an integer number of units
                                if pb1_spread_change > 0:
                                    pb1_spread_change = max_pb1_liquidity_spread_units
                                else:
                                    pb1_spread_change = -max_pb1_liquidity_spread_units

                                # Recalculate component quantities based on the integer spread change
                                pb1_buy_components = {product: 0 for product in self.products}
                                pb1_sell_components = {product: 0 for product in self.products}

                                # PB1 spread = PICNIC_BASKET1 - (6×CROISSANTS + 3×JAMS + 1×DJEMBE)
                                pb1_basket_qty = abs(pb1_spread_change * self.pb1_components["PICNIC_BASKET1"])
                                croissants_qty = abs(pb1_spread_change * self.pb1_components["CROISSANTS"])
                                jams_qty = abs(pb1_spread_change * self.pb1_components["JAMS"])
                                djembes_qty = abs(pb1_spread_change * self.pb1_components["DJEMBES"])

                                if pb1_spread_change > 0:  # Buying the PB1 spread
                                    # Buy positive components, sell negative components
                                    pb1_buy_components["PICNIC_BASKET1"] = pb1_basket_qty
                                    pb1_sell_components["CROISSANTS"] = croissants_qty
                                    pb1_sell_components["JAMS"] = jams_qty
                                    pb1_sell_components["DJEMBES"] = djembes_qty
                                else:  # Selling the PB1 spread
                                    # Sell positive components, buy negative components
                                    pb1_sell_components["PICNIC_BASKET1"] = pb1_basket_qty
                                    pb1_buy_components["CROISSANTS"] = croissants_qty
                                    pb1_buy_components["JAMS"] = jams_qty
                                    pb1_buy_components["DJEMBES"] = djembes_qty

                            logger.print(f"PB1 - After liquidity check - Buy components: {pb1_buy_components}")
                            logger.print(f"PB1 - After liquidity check - Sell components: {pb1_sell_components}")

                            # Execute orders with proper liquidity taking
                            pb1_actual_buys = {product: 0 for product in self.products}
                            pb1_actual_sells = {product: 0 for product in self.products}

                            # Process buy orders
                            for product, buy_qty in pb1_buy_components.items():
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
                                                    # We've already checked that this order is within our price limits during the liquidity check
                                                    # So we can just use the ask_price directly
                                                    # Ensure the price is an integer
                                                    adjusted_price = int(ask_price)

                                                    if product not in result:
                                                        result[product] = []
                                                    result[product].append(Order(product, adjusted_price, executable_volume_int))

                                                    # Update actual buys with the integer volume
                                                    pb1_actual_buys[product] += executable_volume_int

                                                    # Update remaining to buy with the integer volume
                                                    remaining_to_buy -= executable_volume_int

                                                    # If we've filled the entire order, break
                                                    if remaining_to_buy <= 0:
                                                        break

                            # Process sell orders
                            for product, sell_qty in pb1_sell_components.items():
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
                                                    # We've already checked that this order is within our price limits during the liquidity check
                                                    # So we can just use the bid_price directly
                                                    # Ensure the price is an integer
                                                    adjusted_price = int(bid_price)

                                                    if product not in result:
                                                        result[product] = []
                                                    result[product].append(Order(product, adjusted_price, -executable_volume_int))

                                                    # Update actual sells with the integer volume
                                                    pb1_actual_sells[product] += executable_volume_int

                                                    # Update remaining to sell with the integer volume
                                                    remaining_to_sell -= executable_volume_int

                                                    # If we've filled the entire order, break
                                                    if remaining_to_sell <= 0:
                                                        break

                            logger.print(f"PB1 - Actual buys: {pb1_actual_buys}")
                            logger.print(f"PB1 - Actual sells: {pb1_actual_sells}")

                            # Calculate execution ratio for the spread
                            pb1_spread_execution_ratio = 1.0
                            if pb1_spread_change != 0:
                                # Determine the limiting factor for the spread execution
                                if pb1_spread_change > 0:  # Buying the spread
                                    for product, buy_qty in pb1_buy_components.items():
                                        if buy_qty > 0:
                                            execution_ratio = pb1_actual_buys[product] / buy_qty if buy_qty > 0 else 1.0
                                            pb1_spread_execution_ratio = min(pb1_spread_execution_ratio, execution_ratio)

                                    for product, sell_qty in pb1_sell_components.items():
                                        if sell_qty > 0:
                                            execution_ratio = pb1_actual_sells[product] / sell_qty if sell_qty > 0 else 1.0
                                            pb1_spread_execution_ratio = min(pb1_spread_execution_ratio, execution_ratio)
                                else:  # Selling the spread
                                    for product, buy_qty in pb1_buy_components.items():
                                        if buy_qty > 0:
                                            execution_ratio = pb1_actual_buys[product] / buy_qty if buy_qty > 0 else 1.0
                                            pb1_spread_execution_ratio = min(pb1_spread_execution_ratio, execution_ratio)

                                    for product, sell_qty in pb1_sell_components.items():
                                        if sell_qty > 0:
                                            execution_ratio = pb1_actual_sells[product] / sell_qty if sell_qty > 0 else 1.0
                                            pb1_spread_execution_ratio = min(pb1_spread_execution_ratio, execution_ratio)

                            # Calculate actual spread position change
                            # We need to ensure the actual spread change is an integer number of spread units
                            if abs(pb1_spread_change) > 0:
                                # Calculate the integer number of spread units that were executed
                                executed_pb1_spread_units = int(abs(pb1_spread_change) * pb1_spread_execution_ratio)

                                # Set the actual spread change with the correct sign
                                if pb1_spread_change > 0:
                                    pb1_actual_spread_change = executed_pb1_spread_units
                                else:
                                    pb1_actual_spread_change = -executed_pb1_spread_units
                            else:
                                pb1_actual_spread_change = 0

                            logger.print(f"PB1 - Spread execution ratio: {pb1_spread_execution_ratio}, actual change: {pb1_actual_spread_change}")

                            # Update spread position
                            self.spread_positions["pb1_spread"] = int(self.spread_positions["pb1_spread"] + pb1_actual_spread_change)

                            # Update current positions for subsequent trades
                            for product in self.products:
                                current_positions[product] += pb1_actual_buys[product] - pb1_actual_sells[product]

        # Update trader data
        trader_data['spread_positions'] = self.spread_positions

        logger.flush(state, result, 0, jsonpickle.encode(trader_data))
        return result, 0, jsonpickle.encode(trader_data)
