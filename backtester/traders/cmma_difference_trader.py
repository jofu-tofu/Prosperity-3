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
        self.lookback = 800
        self.atr_lookback = 1000
        self.max_position = 30
        self.price_adjustment = 5

        # Products we'll be trading
        self.products = ['PICNIC_BASKET1', 'PICNIC_BASKET2', 'DJEMBES']

        # Track current position state
        self.current_position = 0

        # These variables will be stored in trader_data and initialized at the start of each run

    def _save_parameters_to_trader_data(self, trader_data):
        """Helper function to save all parameters to trader_data"""
        trader_data['threshold'] = self.threshold
        trader_data['lookback'] = self.lookback
        trader_data['atr_lookback'] = self.atr_lookback
        trader_data['max_position'] = self.max_position
        trader_data['price_adjustment'] = self.price_adjustment
        trader_data['products'] = self.products
        trader_data['current_position'] = self.current_position
        return trader_data

    def calculate_cmma(self, log_prices, lookback=None, atr_lookback=None):
        """
        Compute Cumulative Moving Average Momentum (CMMA) as defined in basket_trading2.ipynb

        Parameters:
            log_prices: Numpy array of log prices
            lookback: Lookback period for CMMA calculation
            atr_lookback: Lookback period for ATR calculation

        Returns:
            float: CMMA value
        """
        if lookback is None:
            lookback = self.lookback
        if atr_lookback is None:
            atr_lookback = self.atr_lookback

        if len(log_prices) < max(lookback, atr_lookback) + 1:
            return 0  # Not enough data

        # Calculate log price differences using numpy
        diffs = np.diff(log_prices)

        # Calculate ATR (standard deviation of log price differences)
        if len(diffs) < atr_lookback:
            atr = np.std(diffs)
        else:
            atr = np.std(diffs[-atr_lookback:])

        # Calculate moving average
        if len(log_prices) <= lookback:
            ma = np.mean(log_prices[:-1])
        else:
            ma = np.mean(log_prices[-lookback-1:-1])

        # Calculate raw CMMA
        current_price = log_prices[-1]
        raw_cmma = (current_price - ma) / (atr * np.sqrt(lookback + 1))

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
            self.threshold = trader_data['threshold']

        if 'lookback' not in trader_data:
            trader_data['lookback'] = self.lookback
        else:
            self.lookback = trader_data['lookback']

        if 'atr_lookback' not in trader_data:
            trader_data['atr_lookback'] = self.atr_lookback
        else:
            self.atr_lookback = trader_data['atr_lookback']

        if 'max_position' not in trader_data:
            trader_data['max_position'] = self.max_position
        else:
            self.max_position = trader_data['max_position']

        if 'price_adjustment' not in trader_data:
            trader_data['price_adjustment'] = self.price_adjustment
        else:
            self.price_adjustment = trader_data['price_adjustment']

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

        # Initialize price history in trader_data if it doesn't exist
        if 'price_history' not in trader_data:
            trader_data['price_history'] = {}
            for product in self.products:
                trader_data['price_history'][product] = []

        # Update price history with latest mid prices
        for product in self.products:
            if product in state.order_depths:
                order_depth = state.order_depths[product]
                if len(order_depth.buy_orders) > 0 and len(order_depth.sell_orders) > 0:
                    # Calculate mid price
                    best_bid = min(order_depth.buy_orders.keys())
                    best_ask = max(order_depth.sell_orders.keys())
                    mid_price = (best_bid + best_ask) / 2

                    # Add mid price to price history
                    trader_data['price_history'][product].append(mid_price)

                    # Pop oldest price if exceeding max window size
                    max_window_size = max(self.lookback, self.atr_lookback) + 1
                    if len(trader_data['price_history'][product]) > max_window_size:
                        trader_data['price_history'][product].pop(0)

        # Check if we have enough data for all products
        for product in self.products:
            if len(trader_data['price_history'][product]) < max(self.lookback, self.atr_lookback) + 1:
                # Not enough data yet
                # Save all parameters back to trader_data
                trader_data = self._save_parameters_to_trader_data(trader_data)
                logger.flush(state, result, conversions, trader_data)
                return result, 0, jsonpickle.encode(trader_data)

        # Calculate log prices for PB1 and PB2+DJEMBES using numpy
        pb1_prices = np.array(trader_data['price_history']['PICNIC_BASKET1'])
        pb2_prices = np.array(trader_data['price_history']['PICNIC_BASKET2'])
        dj_prices = np.array(trader_data['price_history']['DJEMBES'])

        # Calculate log prices
        log_pb1 = np.log(pb1_prices)
        log_pb2dj = np.log(1.5 * pb2_prices + dj_prices)

        # Calculate CMMA for both log price series
        cmma_pb1 = self.calculate_cmma(log_pb1, self.lookback, self.atr_lookback)
        cmma_pb2dj = self.calculate_cmma(log_pb2dj, self.lookback, self.atr_lookback)

        # Calculate CMMA difference
        cmma_difference = cmma_pb2dj - cmma_pb1
        logger.print(f"CMMA Difference: {cmma_difference}")

        # Note: We're not using current positions for trading decisions
        # We only use the CMMA difference and the current_position state

        # Determine target position based on CMMA difference
        target_position = 0

        # Only flip position when crossing opposite threshold
        if cmma_difference > self.threshold and self.current_position <= 0:
            # Buy signal
            target_position = -self.max_position
        elif cmma_difference < -self.threshold and self.current_position >= 0:
            # Sell signal
            target_position = self.max_position
        else:
            # Maintain current position
            target_position = self.current_position

        # Calculate position changes needed
        position_change = target_position - self.current_position

        # If no change needed, return empty orders list
        if position_change == 0:
            # Save all parameters back to trader_data
            trader_data = self._save_parameters_to_trader_data(trader_data)
            return result, 0, jsonpickle.encode(trader_data)

        # Use the single price adjustment parameter
        price_adjustment = self.price_adjustment

        # Execute trades for the spread
        orders = []
        if position_change > 0:
            # Buy the spread (buy PB2+DJ, sell PB1)
            orders.extend(self.buy_spread(state.order_depths, position_change, price_adjustment))
        else:
            # Sell the spread (sell PB2+DJ, buy PB1)
            orders.extend(self.sell_spread(state.order_depths, abs(position_change), price_adjustment))

        # Update current position
        self.current_position = target_position

        # Save all parameters back to trader_data
        trader_data = self._save_parameters_to_trader_data(trader_data)

        # Organize orders by product
        for order in orders:
            if order.symbol not in result:
                result[order.symbol] = []
            result[order.symbol].append(order)
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
            quantity: Requested quantity to trade
            price_adjustment: Price adjustment for orders

        Returns:
            List of orders
        """
        orders = []

        # Spread multipliers
        pb1_multiplier = -2
        pb2_multiplier = 3
        dj_multiplier = 2

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
        max_spreads = min(max_spreads_pb1, max_spreads_pb2, max_spreads_dj, quantity)

        # If we can't trade any spreads, return empty orders list
        if max_spreads <= 0:
            return orders

        # Calculate final quantities to trade
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
            quantity: Requested quantity to trade
            price_adjustment: Price adjustment for orders

        Returns:
            List of orders
        """
        orders = []

        # Spread multipliers
        pb1_multiplier = 2
        pb2_multiplier = -3
        dj_multiplier = -2

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
        max_spreads = min(max_spreads_pb1, max_spreads_pb2, max_spreads_dj, quantity)

        # If we can't trade any spreads, return empty orders list
        if max_spreads <= 0:
            return orders

        # Calculate final quantities to trade
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
