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
    PB2 Spread Trader that exclusively trades the PB2 spread
    (PICNIC_BASKET2 - (4×CROISSANTS + 2×JAMS))
    based on entry and exit thresholds.
    """
    def __init__(self):
        # Products to trade
        self.products = ["PICNIC_BASKET2", "CROISSANTS", "JAMS"]
        
        # Position limits for each product
        self.position_limits = {
            "PICNIC_BASKET2": 100,
            "CROISSANTS": 250,
            "JAMS": 350
        }
        
        # PB2 spread components
        # PB2 spread = PICNIC_BASKET2 - (4×CROISSANTS + 2×JAMS)
        self.pb2_components = {
            "PICNIC_BASKET2": 1,
            "CROISSANTS": -4,
            "JAMS": -2
        }
        
        # Thresholds for trading
        self.entry_threshold = 5.0
        self.exit_threshold = 1.0
        
        # Maximum position size for the spread
        self.max_spread_position = 100
        
        # Initialize spread position
        self.spread_position = 0
        
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
        self.entry_price_adjustment = 0  # Willing to pay 0 units worse than mid price to enter a position
        self.exit_price_adjustment = 1   # Willing to pay 1 unit worse than mid price to exit a position

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
        logger.print("PB2 Spread Trader starting...")
        
        # Initialize trader data from state or create new if none exists
        trader_data = {}
        if state.traderData:
            try:
                trader_data = jsonpickle.decode(state.traderData)
            except:
                trader_data = {}
        
        # Store spread position in trader_data
        if 'spread_position' not in trader_data:
            trader_data['spread_position'] = self.spread_position
        else:
            self.spread_position = trader_data['spread_position']
        
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
        
        # Calculate theoretical basket value
        theoretical_pb2 = sum(qty * current_prices[product] for product, qty in self.pb2_components.items())
        
        # Calculate spread
        pb2_spread = theoretical_pb2
        
        # Calculate z-score using fixed standard deviation
        z_score = pb2_spread / self.fixed_std_dev
        
        logger.print(f"PB2 spread: {pb2_spread}, z-score: {z_score}")
        logger.print(f"Current positions: {state.position}")
        logger.print(f"Current spread position: {self.spread_position}")
        
        # Get current positions
        current_positions = {product: 0 for product in self.products}
        for product in self.products:
            if product in state.position:
                current_positions[product] = state.position[product]
        
        # Generate trading signal for PB2 spread
        spread_signal = None
        
        if z_score > self.entry_threshold:
            # Sell the spread if it's above the entry threshold
            spread_signal = -self.max_spread_position
        elif z_score < -self.entry_threshold:
            # Buy the spread if it's below the negative entry threshold
            spread_signal = self.max_spread_position
        elif abs(z_score) < self.exit_threshold:
            # Neutralize position if spread is within exit thresholds
            spread_signal = 0
        else:
            # In the middle zone - neither entry nor exit - hold current position without trading
            spread_signal = None
        
        logger.print(f"PB2 spread signal: {spread_signal}")
        
        # Calculate target position for PB2 spread
        if spread_signal is None:
            # If signal is None, maintain current position (no change)
            target_spread_position = self.spread_position
        else:
            # Otherwise, use the trade signal as the target position
            target_spread_position = int(spread_signal)
        
        # Calculate position change for PB2 spread
        spread_change = int(target_spread_position - self.spread_position)
        
        logger.print(f"PB2 spread change: {spread_change}")
        
        # If no change in position, return early
        if spread_change == 0:
            logger.print("No change in PB2 spread position, skipping trading")
            return result, 0, jsonpickle.encode(trader_data)
        
        # Calculate component quantities for PB2 spread change
        buy_components = {product: 0 for product in self.products}
        sell_components = {product: 0 for product in self.products}
        
        if spread_change != 0:
            # PB2 spread = PICNIC_BASKET2 - (4×CROISSANTS + 2×JAMS)
            pb2_qty = abs(spread_change * self.pb2_components["PICNIC_BASKET2"])
            croissants_qty = abs(spread_change * self.pb2_components["CROISSANTS"])
            jams_qty = abs(spread_change * self.pb2_components["JAMS"])
            
            if spread_change > 0:  # Buying the PB2 spread
                # Buy positive components, sell negative components
                buy_components["PICNIC_BASKET2"] = pb2_qty
                sell_components["CROISSANTS"] = croissants_qty
                sell_components["JAMS"] = jams_qty
            else:  # Selling the PB2 spread
                # Sell positive components, buy negative components
                sell_components["PICNIC_BASKET2"] = pb2_qty
                buy_components["CROISSANTS"] = croissants_qty
                buy_components["JAMS"] = jams_qty
        
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
            
            # PB2 spread = PICNIC_BASKET2 - (4×CROISSANTS + 2×JAMS)
            pb2_qty = abs(spread_change * self.pb2_components["PICNIC_BASKET2"])
            croissants_qty = abs(spread_change * self.pb2_components["CROISSANTS"])
            jams_qty = abs(spread_change * self.pb2_components["JAMS"])
            
            if spread_change > 0:  # Buying the PB2 spread
                # Buy positive components, sell negative components
                buy_components["PICNIC_BASKET2"] = pb2_qty
                sell_components["CROISSANTS"] = croissants_qty
                sell_components["JAMS"] = jams_qty
            else:  # Selling the PB2 spread
                # Sell positive components, buy negative components
                sell_components["PICNIC_BASKET2"] = pb2_qty
                buy_components["CROISSANTS"] = croissants_qty
                buy_components["JAMS"] = jams_qty
        
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
                        if abs(z_score) >= self.entry_threshold:  # Entering a position
                            price_adjustment = self.entry_price_adjustment
                        elif abs(z_score) < self.exit_threshold:  # Exiting a position
                            price_adjustment = self.exit_price_adjustment
                        else:  # In the middle zone
                            # Use exit price adjustment when closing positions
                            if (z_score > 0 and self.spread_position > 0) or \
                               (z_score < 0 and self.spread_position < 0):
                                price_adjustment = self.exit_price_adjustment
                            else:
                                price_adjustment = self.entry_price_adjustment
                        
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
                        if abs(z_score) >= self.entry_threshold:  # Entering a position
                            price_adjustment = self.entry_price_adjustment
                        elif abs(z_score) < self.exit_threshold:  # Exiting a position
                            price_adjustment = self.exit_price_adjustment
                        else:  # In the middle zone
                            # Use exit price adjustment when closing positions
                            if (z_score > 0 and self.spread_position > 0) or \
                               (z_score < 0 and self.spread_position < 0):
                                price_adjustment = self.exit_price_adjustment
                            else:
                                price_adjustment = self.entry_price_adjustment
                        
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
            
            # PB2 spread = PICNIC_BASKET2 - (4×CROISSANTS + 2×JAMS)
            pb2_qty = abs(spread_change * self.pb2_components["PICNIC_BASKET2"])
            croissants_qty = abs(spread_change * self.pb2_components["CROISSANTS"])
            jams_qty = abs(spread_change * self.pb2_components["JAMS"])
            
            if spread_change > 0:  # Buying the PB2 spread
                # Buy positive components, sell negative components
                buy_components["PICNIC_BASKET2"] = pb2_qty
                sell_components["CROISSANTS"] = croissants_qty
                sell_components["JAMS"] = jams_qty
            else:  # Selling the PB2 spread
                # Sell positive components, buy negative components
                sell_components["PICNIC_BASKET2"] = pb2_qty
                buy_components["CROISSANTS"] = croissants_qty
                buy_components["JAMS"] = jams_qty
        
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
        self.spread_position = int(self.spread_position + actual_spread_change)
        
        logger.print(f"Updated spread position: {self.spread_position}")
        
        # Update trader data
        trader_data['spread_position'] = self.spread_position
        
        logger.flush(state, result, 0, jsonpickle.encode(trader_data))
        return result, 0, jsonpickle.encode(trader_data)
