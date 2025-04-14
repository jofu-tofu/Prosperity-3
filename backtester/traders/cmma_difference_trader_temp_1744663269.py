import json
from typing import Any, Dict, List, Tuple, Optional
import numpy as np
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
import jsonpickle


class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 10050

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
    Trader that trades on the CMMA difference between PB2+DJEMBES and PB1.
    Buys when CMMA difference is above threshold and sells when below negative threshold.
    Only flips position when crossing the opposite threshold.
    """

    def __init__(self):
        """
        Initialize the CMMA Difference Trader.
        """
        # Trading parameters
        self.threshold = 0.5
        self.lookback = 200  # Used as span for EWM
        self.atr_lookback = 5.0*self.lookback/4.0  # Used as span for EWM of absolute differences
        self.max_position = 30  # Maximum number of spread units to hold (not individual instruments)
        self.price_adjustment = 1

        # Products we'll be trading
        self.products = ['PICNIC_BASKET1', 'PICNIC_BASKET2', 'DJEMBES']

        # Track current position state
        self.current_position = 0

        # EWM state variables
        self.ewm_means = {}
        self.ewm_atr = {}
        self.last_prices = {}

        # These variables will be stored in trader_data and initialized at the start of each run

    def _calculate_spread_position(self, positions):
        """
        Calculate the current spread position based on individual instrument positions

        Parameters:
            positions: Dictionary of positions for each product

        Returns:
            int: Current spread position
        """
        # Spread multipliers - these define how many units of each product make up one spread unit
        pb1_multiplier = -2  # Sell 2 units of PB1 per spread when buying the spread
        pb2_multiplier = 3   # Buy 3 units of PB2 per spread when buying the spread
        dj_multiplier = 2    # Buy 2 units of DJ per spread when buying the spread

        # Get positions for each product (default to 0 if not present)
        pb1_position = positions.get('PICNIC_BASKET1', 0)
        pb2_position = positions.get('PICNIC_BASKET2', 0)
        dj_position = positions.get('DJEMBES', 0)

        return int(pb1_position/pb1_multiplier)
    
    def _save_parameters_to_trader_data(self, trader_data):
        """Helper function to save all parameters to trader_data"""
        trader_data['threshold'] = self.threshold
        trader_data['lookback'] = self.lookback
        trader_data['atr_lookback'] = self.atr_lookback
        trader_data['max_position'] = self.max_position
        trader_data['price_adjustment'] = self.price_adjustment
        trader_data['products'] = self.products
        trader_data['current_position'] = self.current_position
        trader_data['ewm_means'] = self.ewm_means
        trader_data['ewm_atr'] = self.ewm_atr
        trader_data['last_prices'] = self.last_prices
        return trader_data

    def update_ewm(self, product_key, current_log_price):
        """
        Update exponentially weighted moving averages for a product

        Parameters:
            product_key: Key to identify the product (e.g., 'pb1', 'pb2dj')
            current_log_price: Current log price value

        Returns:
            None (updates internal state)
        """
        # Initialize if this is the first update for this product
        if product_key not in self.ewm_means:
            self.ewm_means[product_key] = current_log_price
            self.ewm_atr[product_key] = 0.0001  # Small non-zero value to avoid division by zero
            self.last_prices[product_key] = current_log_price
            return

        # Get previous values
        last_price = self.last_prices[product_key]

        # Calculate price difference
        price_diff = current_log_price - last_price

        # Update EWM mean with span = lookback
        alpha_mean = 2.0 / (self.lookback + 1)
        self.ewm_means[product_key] = (1 - alpha_mean) * self.ewm_means[product_key] + alpha_mean * current_log_price

        # Update EWM ATR (using absolute difference) with span = atr_lookback
        alpha_atr = 2.0 / (self.atr_lookback + 1)
        abs_diff = abs(price_diff)
        self.ewm_atr[product_key] = (1 - alpha_atr) * self.ewm_atr[product_key] + alpha_atr * abs_diff

        # Update last price
        self.last_prices[product_key] = current_log_price

    def calculate_cmma(self, product_key, current_log_price):
        """
        Compute Cumulative Moving Average Momentum (CMMA) using EWM

        Parameters:
            product_key: Key to identify the product (e.g., 'pb1', 'pb2dj')
            current_log_price: Current log price value

        Returns:
            float: CMMA value
        """
        # Update EWM values first
        self.update_ewm(product_key, current_log_price)

        # Calculate CMMA using EWM values
        # Use a small epsilon to avoid division by zero
        epsilon = 1e-10
        raw_cmma = (current_log_price - self.ewm_means[product_key]) / (self.ewm_atr[product_key] * np.sqrt(self.lookback + 1) + epsilon)

        return raw_cmma

    def run(self, state: TradingState) -> Tuple[Dict[str, List[Order]], int, str]:
        """
        Main trading logic.

        Parameters:
            state: Current trader state
            market_state: Current market state

        Returns:
            List of orders to execute
        """
        # Initialize result dictionary for orders
        result = {}
        conversions = 0


        # Initialize trader data from state or create new if none exists
        trader_data = {}
        if state.traderData:
            try:
                trader_data = jsonpickle.decode(state.traderData)
            except:
                trader_data = {}

        # Initialize or retrieve all trader parameters from trader_data
        # Trading parameters
        if 'threshold' not in trader_data:
            trader_data['threshold'] = self.threshold
        else:
            self.threshold = 0.5

        if 'lookback' not in trader_data:
            trader_data['lookback'] = self.lookback
        else:
            self.lookback = 200  # Used as span for EWM

        if 'atr_lookback' not in trader_data:
            trader_data['atr_lookback'] = self.atr_lookback
        else:
            self.atr_lookback = 5.0*self.lookback/4.0  # Used as span for EWM of absolute differences

        if 'max_position' not in trader_data:
            trader_data['max_position'] = self.max_position
        else:
            self.max_position = 30  # Maximum number of spread units to hold (not individual instruments)

        if 'price_adjustment' not in trader_data:
            trader_data['price_adjustment'] = self.price_adjustment
        else:
            self.price_adjustment = 1

        # Products list
        if 'products' not in trader_data:
            trader_data['products'] = self.products
        else:
            self.products = trader_data['products']

        # Current position
        if 'current_position' not in trader_data:
            trader_data['current_position'] = self.current_position
        else:
            self.current_position = trader_data['current_position']

        # EWM state variables
        if 'ewm_means' not in trader_data:
            trader_data['ewm_means'] = {}
        else:
            self.ewm_means = trader_data['ewm_means']

        if 'ewm_atr' not in trader_data:
            trader_data['ewm_atr'] = {}
        else:
            self.ewm_atr = trader_data['ewm_atr']

        if 'last_prices' not in trader_data:
            trader_data['last_prices'] = {}
        else:
            self.last_prices = trader_data['last_prices']

        # Get current mid prices for each product
        current_prices = {}
        for product in self.products:
            if product in state.order_depths:
                order_depth = state.order_depths[product]
                if len(order_depth.buy_orders) > 0 and len(order_depth.sell_orders) > 0:
                    # Calculate mid price
                    best_bid = min(order_depth.buy_orders.keys())
                    best_ask = max(order_depth.sell_orders.keys())
                    current_prices[product] = (best_bid + best_ask) / 2

        # Check if we have prices for all products
        if len(current_prices) < len(self.products):
            # Not enough data yet
            trader_data = self._save_parameters_to_trader_data(trader_data)
            trader_data = jsonpickle.encode(trader_data)
            logger.flush(state, result, conversions, trader_data)
            return result, 0, trader_data

        # Calculate log prices for PB1 and PB2+DJEMBES
        log_pb1 = np.log(current_prices['PICNIC_BASKET1'])
        log_pb2dj = np.log(1.5 * current_prices['PICNIC_BASKET2'] + current_prices['DJEMBES'])

        logger.print("Current Spread Price: ", log_pb2dj-log_pb1)
        # Calculate CMMA for both log price series using EWM
        cmma_pb1 = self.calculate_cmma('pb1', log_pb1)
        cmma_pb2dj = self.calculate_cmma('pb2dj', log_pb2dj)
        # Calculate CMMA difference
        def tanh(x):
            return np.tanh(x)
        cmma_difference = tanh(cmma_pb2dj - cmma_pb1)
        logger.print(f"CMMA Difference: {cmma_difference}")
        logger.print(f"Current position: {self.current_position}")

        # Calculate the current spread position based on actual instrument positions
        self.current_position = self._calculate_spread_position(state.position)
        logger.print(f"Current spread position: {self.current_position}")

        # Determine target position based on CMMA difference
        target_position = 0

        # Only flip position when crossing opposite threshold
        # max_position represents the maximum number of spread units we want to hold
        if cmma_difference > self.threshold:
            target_position = -self.max_position  # Short max_position spread units
        elif cmma_difference < -self.threshold:
            target_position = self.max_position   # Long max_position spread units
        else:
            target_position = self.current_position  # Maintain current spread position

        logger.print(f"Target spread position: {target_position}")

        # Calculate position changes needed
        position_change = target_position - self.current_position

        # If no change needed, return empty orders list
        if position_change == 0:
            # Save all parameters back to trader_data
            trader_data = self._save_parameters_to_trader_data(trader_data)
            trader_data = jsonpickle.encode(trader_data)
            logger.flush(state, result, conversions, trader_data)
            return result, 0, trader_data

        # Use the single price adjustment parameter
        price_adjustment = self.price_adjustment

        # Execute trades for the spread
        orders = []
        logger.print(f"Position change needed: {position_change}")
        logger.print(f"Current instrument positions: {state.position}")
        if position_change > 0:
            # Buy the spread (buy PB2+DJ, sell PB1)
            new_orders = self.buy_spread(state.order_depths, position_change, price_adjustment)
            orders.extend(new_orders)
            logger.print(f"Buying {position_change} spread units, generated {len(new_orders)} orders")
        elif position_change < 0:
            # Sell the spread (sell PB2+DJ, buy PB1)
            new_orders = self.sell_spread(state.order_depths, abs(position_change), price_adjustment)
            orders.extend(new_orders)
            logger.print(f"Selling {abs(position_change)} spread units, generated {len(new_orders)} orders")
        else:
            logger.print("No position change needed")

        # We'll calculate the actual current position based on the state.position in the next iteration
        # Don't update self.current_position here, as not all orders may be executed

        # Save all parameters back to trader_data
        trader_data = self._save_parameters_to_trader_data(trader_data)

        # Organize orders by product
        for order in orders:
            if order.symbol not in result:
                result[order.symbol] = []
            result[order.symbol].append(order)
        logger.print(f"Orders: {orders}")
        trader_data = jsonpickle.encode(trader_data)
        logger.flush(state, result, conversions, trader_data)
        return result, 0, trader_data

    def _calculate_adjusted_price(self, price: float, is_buy: bool, price_adjustment: int) -> float:
        """
        Calculate the adjusted price based on whether we're buying or selling

        Parameters:
            price: The base price (best bid or best ask)
            is_buy: True if buying, False if selling
            price_adjustment: The price adjustment to apply

        Returns:
            The adjusted price
        """
        # For buying, we add the adjustment to the ask price
        # For selling, we subtract the adjustment from the bid price
        if is_buy:
            adjusted_price = price + price_adjustment
            if price % 1 == 0.5:
                adjusted_price = price + price_adjustment + 0.5
        else:
            adjusted_price = price - price_adjustment
            if price % 1 == 0.5:
                adjusted_price = price - price_adjustment - 0.5

        return adjusted_price

    def _get_available_volume(self, order_depth: OrderDepth, is_buy: bool, price_adjustment: int) -> tuple[float, float]:
        """
        Calculate the available volume at the adjusted price level

        Parameters:
            order_depth: The order depth for a product
            is_buy: True if buying, False if selling
            price_adjustment: The price adjustment to apply

        Returns:
            Tuple of (adjusted_price, available_volume)
        """
        if is_buy and len(order_depth.sell_orders) > 0:
            # Buying - check sell orders
            best_price = min(order_depth.sell_orders.keys())
            adjusted_price = self._calculate_adjusted_price(best_price, True, price_adjustment)

            # Count liquidity at or below our adjusted price
            available_volume = 0
            for price, volume in order_depth.sell_orders.items():
                if price <= adjusted_price:
                    available_volume += abs(volume)

            return adjusted_price, available_volume

        elif not is_buy and len(order_depth.buy_orders) > 0:
            # Selling - check buy orders
            best_price = max(order_depth.buy_orders.keys())
            adjusted_price = self._calculate_adjusted_price(best_price, False, price_adjustment)

            # Count liquidity at or above our adjusted price
            available_volume = 0
            for price, volume in order_depth.buy_orders.items():
                if price >= adjusted_price:
                    available_volume += abs(volume)

            return adjusted_price, available_volume

        return 0, 0

    def _create_spread_order(self, product: str, order_depth: OrderDepth, quantity: int, is_buy: bool, price_adjustment: int) -> Optional[Order]:
        """
        Create an order for a product in the spread

        Parameters:
            product: The product to trade
            order_depth: The order depth for the product
            quantity: The quantity to trade (positive for buy, negative for sell)
            is_buy: True if buying, False if selling
            price_adjustment: The price adjustment to apply

        Returns:
            The created order, or None if no order can be created
        """
        if quantity == 0:
            return None

        adjusted_price, _ = self._get_available_volume(order_depth, is_buy, price_adjustment)
        if adjusted_price == 0:
            return None

        return Order(product, adjusted_price, quantity)

    def buy_spread(self, order_depths: Dict[str, OrderDepth], quantity: int, price_adjustment: int) -> List[Order]:
        """
        Buy the spread: buy PB2+DJ, sell PB1

        Parameters:
            order_depths: Dictionary of order depths for each product
            quantity: Requested quantity of spread units to trade (not individual instruments)
            price_adjustment: Price adjustment for orders

        Returns:
            List of orders
        """
        orders = []

        # Spread multipliers - these define how many units of each product make up one spread unit
        pb1_multiplier = -2  # Sell 2 units of PB1 per spread
        pb2_multiplier = 3   # Buy 3 units of PB2 per spread
        dj_multiplier = 2    # Buy 2 units of DJ per spread

        # Calculate max spreads we can trade based on available liquidity at our price levels
        max_spreads_pb1 = 0
        max_spreads_pb2 = 0
        max_spreads_dj = 0

        # For PB1 (selling), check buy orders at or above our adjusted price
        if 'PICNIC_BASKET1' in order_depths:
            _, available_volume = self._get_available_volume(order_depths['PICNIC_BASKET1'], False, price_adjustment)
            max_spreads_pb1 = available_volume // abs(pb1_multiplier) if pb1_multiplier != 0 else float('inf')

        # For PB2 (buying), check sell orders at or below our adjusted price
        if 'PICNIC_BASKET2' in order_depths:
            _, available_volume = self._get_available_volume(order_depths['PICNIC_BASKET2'], True, price_adjustment)
            max_spreads_pb2 = available_volume // abs(pb2_multiplier) if pb2_multiplier != 0 else float('inf')

        # For DJEMBES (buying), check sell orders at or below our adjusted price
        if 'DJEMBES' in order_depths:
            _, available_volume = self._get_available_volume(order_depths['DJEMBES'], True, price_adjustment)
            max_spreads_dj = available_volume // abs(dj_multiplier) if dj_multiplier != 0 else float('inf')

        # Take the minimum to ensure we have enough liquidity for all components
        # The quantity parameter represents the number of spread units we want to trade
        max_spreads = min(max_spreads_pb1, max_spreads_pb2, max_spreads_dj, quantity)

        # Log the liquidity constraints
        logger.print(f"Buy spread liquidity constraints:")
        logger.print(f"  PB1: {max_spreads_pb1} spreads")
        logger.print(f"  PB2: {max_spreads_pb2} spreads")
        logger.print(f"  DJ: {max_spreads_dj} spreads")
        logger.print(f"  Requested: {quantity} spreads")
        logger.print(f"  Max possible: {max_spreads} spreads")

        # If we can't trade any spreads, return empty orders list
        if max_spreads <= 0:
            logger.print("Cannot buy any spreads due to liquidity constraints")
            return orders

        # Calculate final quantities to trade for each instrument
        # Each instrument's quantity is its multiplier times the number of spread units
        pb1_quantity = pb1_multiplier * max_spreads
        pb2_quantity = pb2_multiplier * max_spreads
        dj_quantity = dj_multiplier * max_spreads

        # Create orders with the adjusted prices we already calculated
        if 'PICNIC_BASKET1' in order_depths:
            order = self._create_spread_order('PICNIC_BASKET1', order_depths['PICNIC_BASKET1'],
                                             pb1_quantity, False, price_adjustment)
            if order:
                orders.append(order)

        if 'PICNIC_BASKET2' in order_depths:
            order = self._create_spread_order('PICNIC_BASKET2', order_depths['PICNIC_BASKET2'],
                                             pb2_quantity, True, price_adjustment)
            if order:
                orders.append(order)

        if 'DJEMBES' in order_depths:
            order = self._create_spread_order('DJEMBES', order_depths['DJEMBES'],
                                            dj_quantity, True, price_adjustment)
            if order:
                orders.append(order)

        return orders

    def sell_spread(self, order_depths: Dict[str, OrderDepth], quantity: int, price_adjustment: int) -> List[Order]:
        """
        Sell the spread: sell PB2+DJ, buy PB1

        Parameters:
            order_depths: Dictionary of order depths for each product
            quantity: Requested quantity of spread units to trade (not individual instruments)
            price_adjustment: Price adjustment for orders

        Returns:
            List of orders
        """
        orders = []

        # Spread multipliers - these define how many units of each product make up one spread unit
        pb1_multiplier = 2    # Buy 2 units of PB1 per spread
        pb2_multiplier = -3   # Sell 3 units of PB2 per spread
        dj_multiplier = -2    # Sell 2 units of DJ per spread

        # Calculate max spreads we can trade based on available liquidity at our price levels
        max_spreads_pb1 = 0
        max_spreads_pb2 = 0
        max_spreads_dj = 0

        # For PB1 (buying), check sell orders at or below our adjusted price
        if 'PICNIC_BASKET1' in order_depths:
            _, available_volume = self._get_available_volume(order_depths['PICNIC_BASKET1'], True, price_adjustment)
            max_spreads_pb1 = available_volume // abs(pb1_multiplier) if pb1_multiplier != 0 else float('inf')

        # For PB2 (selling), check buy orders at or above our adjusted price
        if 'PICNIC_BASKET2' in order_depths:
            _, available_volume = self._get_available_volume(order_depths['PICNIC_BASKET2'], False, price_adjustment)
            max_spreads_pb2 = available_volume // abs(pb2_multiplier) if pb2_multiplier != 0 else float('inf')

        # For DJEMBES (selling), check buy orders at or above our adjusted price
        if 'DJEMBES' in order_depths:
            _, available_volume = self._get_available_volume(order_depths['DJEMBES'], False, price_adjustment)
            max_spreads_dj = available_volume // abs(dj_multiplier) if dj_multiplier != 0 else float('inf')

        # Take the minimum to ensure we have enough liquidity for all components
        # The quantity parameter represents the number of spread units we want to trade
        max_spreads = min(max_spreads_pb1, max_spreads_pb2, max_spreads_dj, quantity)

        # Log the liquidity constraints
        logger.print(f"Sell spread liquidity constraints:")
        logger.print(f"  PB1: {max_spreads_pb1} spreads")
        logger.print(f"  PB2: {max_spreads_pb2} spreads")
        logger.print(f"  DJ: {max_spreads_dj} spreads")
        logger.print(f"  Requested: {quantity} spreads")
        logger.print(f"  Max possible: {max_spreads} spreads")

        # If we can't trade any spreads, return empty orders list
        if max_spreads <= 0:
            logger.print("Cannot sell any spreads due to liquidity constraints")
            return orders

        # Calculate final quantities to trade for each instrument
        # Each instrument's quantity is its multiplier times the number of spread units
        pb1_quantity = pb1_multiplier * max_spreads
        pb2_quantity = pb2_multiplier * max_spreads
        dj_quantity = dj_multiplier * max_spreads

        # Create orders with the adjusted prices we already calculated
        if 'PICNIC_BASKET1' in order_depths:
            order = self._create_spread_order('PICNIC_BASKET1', order_depths['PICNIC_BASKET1'],
                                             pb1_quantity, True, price_adjustment)
            if order:
                orders.append(order)

        if 'PICNIC_BASKET2' in order_depths:
            order = self._create_spread_order('PICNIC_BASKET2', order_depths['PICNIC_BASKET2'],
                                             pb2_quantity, False, price_adjustment)
            if order:
                orders.append(order)

        if 'DJEMBES' in order_depths:
            order = self._create_spread_order('DJEMBES', order_depths['DJEMBES'],
                                            dj_quantity, False, price_adjustment)
            if order:
                orders.append(order)

        return orders
