import json
from typing import Any, Dict, List, Tuple, Optional
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
import jsonpickle
import math
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
    def __init__(self):
        self.product_max_positions = {
            'VOLCANIC_ROCK': 400,
            'VOLCANIC_ROCK_VOUCHER_9500': 200,
            'VOLCANIC_ROCK_VOUCHER_9750': 200,
            'VOLCANIC_ROCK_VOUCHER_10000': 200,
            'VOLCANIC_ROCK_VOUCHER_10250': 200,
            'VOLCANIC_ROCK_VOUCHER_10500': 200
        }
        self.voucher_names = ['VOLCANIC_ROCK_VOUCHER_9500', 'VOLCANIC_ROCK_VOUCHER_9750', 'VOLCANIC_ROCK_VOUCHER_10000', 'VOLCANIC_ROCK_VOUCHER_10250', 'VOLCANIC_ROCK_VOUCHER_10500']
        self.voucher_strikes = {
            'VOLCANIC_ROCK_VOUCHER_9500': 9500.0,
            'VOLCANIC_ROCK_VOUCHER_9750': 9750.0,
            'VOLCANIC_ROCK_VOUCHER_10000': 10000.0,
            'VOLCANIC_ROCK_VOUCHER_10250': 10250.0,
            'VOLCANIC_ROCK_VOUCHER_10500': 10500.0
        }
        self.current_positions = {
            'VOLCANIC_ROCK': 0,
            'VOLCANIC_ROCK_VOUCHER_9500': 0,
            'VOLCANIC_ROCK_VOUCHER_9750': 0,
            'VOLCANIC_ROCK_VOUCHER_10000': 0,
            'VOLCANIC_ROCK_VOUCHER_10250': 0,
            'VOLCANIC_ROCK_VOUCHER_10500': 0
        }
        self.T = -1
        self.DAYS_IN_YEAR = 365.0
        self.t0_rock_prices = [10503.0, 10516.0, 10218.5]
        self.day = 0
        self.iv_poly_params = {
            "ask": [0.15, 0, 0.3],
            "bid": [0.15, 0, 0.15]
        }

        # Track the current shorted option
        self.current_short_option = None

        # Timestep counter
        self.timestep_counter = 0

    def estimate_iv(self, moneyness, price_type):
        return self.iv_poly_params[price_type][0] + self.iv_poly_params[price_type][1] * moneyness + self.iv_poly_params[price_type][2] * moneyness**2

    def norm_pdf(self,x):
        """Standard normal probability density function."""
        return math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)

    def norm_cdf(self, x):
        """Standard normal cumulative distribution function."""
        # Using math.erf for the error function
        return 0.5 * (1 + math.erf(x / math.sqrt(2)))

    def black_scholes_price(self, option_type, S, K, T, r, sigma):
        """
        Computes the Black-Scholes price for European call or put options.
        Ensures price is never below intrinsic value.

        Parameters:
            option_type (str): 'c' for call, 'p' for put.
            S (float): Underlying asset price.
            K (float): Strike price.
            T (float): Time to expiration in years.
            r (float): Risk-free interest rate.
            sigma (float): Volatility of the underlying asset.

        Returns:
            float: Theoretical option price, never below intrinsic value.
        """
        # Calculate intrinsic value
        intrinsic_value = max(0, S - K) if option_type == 'c' else max(0, K - S)

        if T <= 0:
            return intrinsic_value

        # Avoid division by zero by ensuring sigma > 0.
        if sigma <= 0:
            raise ValueError("Sigma must be positive.")

        d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)

        if option_type == 'c':
            price = S * self.norm_cdf(d1) - K * math.exp(-r * T) * self.norm_cdf(d2)
        elif option_type == 'p':
            price = K * math.exp(-r * T) * self.norm_cdf(-d2) - S * self.norm_cdf(-d1)
        else:
            raise ValueError("option_type must be 'c' (call) or 'p' (put).")

        # Return maximum of theoretical price and intrinsic value
        return max(price, intrinsic_value)

    def black_scholes_vega(self, S, K, T, r, sigma):
        """
        Computes the Black-Scholes vega, the sensitivity of the option's price
        to changes in volatility.
        """
        d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
        return S * self.norm_pdf(d1) * math.sqrt(T)

    def implied_volatility(self, option_type, S, K, T, r, market_price, tol=1e-6, max_iter=100):
        """
        Computes the implied volatility using the Newton-Raphson method.
        Handles cases where market price is below intrinsic value.
        """
        # Calculate intrinsic value
        intrinsic_value = max(0, S - K) if option_type == 'c' else max(0, K - S)

        # If market price is below intrinsic value, return None or raise warning
        if market_price < intrinsic_value:
            logger.print(f"Warning: Market price {market_price} below intrinsic value {intrinsic_value}")
            return None

        # If market price equals intrinsic value, implied vol is effectively 0
        if abs(market_price - intrinsic_value) < tol:
            return 0.0001  # Return small positive value instead of 0

        sigma = 0.2  # initial guess
        for _ in range(max_iter):
            price = self.black_scholes_price(option_type, S, K, T, r, sigma)
            price_diff = market_price - price
            if abs(price_diff) < tol:
                return sigma
            vega = self.black_scholes_vega(S, K, T, r, sigma)
            if vega < 1e-8:
                break
            sigma += price_diff / vega

            # Add bounds check for sigma
            if sigma <= 0:
                sigma = 0.0001
            elif sigma > 5:  # Cap maximum volatility at 500%
                break

        return sigma

    def black_scholes_greeks(self,option_type, S, K, T, r, sigma):
        """
        Computes the main Black-Scholes Greeks for European options.

        Returns a dictionary with:
        - delta: rate of change of option price w.r.t. S,
        - gamma: rate of change of delta,
        - vega: sensitivity to volatility,
        - theta: time decay (annualized),
        - rho: sensitivity to interest rate.
        """
        d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)
        pdf_d1 = self.norm_pdf(d1)

        # Delta calculation
        if option_type == 'c':
            delta = self.norm_cdf(d1)
        elif option_type == 'p':
            delta = self.norm_cdf(d1) - 1
        else:
            raise ValueError("option_type must be 'c' (call) or 'p' (put).")

        # Gamma is the same for calls and puts
        gamma = pdf_d1 / (S * sigma * math.sqrt(T))

        # Vega: sensitivity of the option price to volatility changes.
        vega = S * pdf_d1 * math.sqrt(T)

        # Theta: time decay of the option value.
        if option_type == 'c':
            theta = (-S * pdf_d1 * sigma / (2 * math.sqrt(T))
                    - r * K * math.exp(-r * T) * self.norm_cdf(d2))
        else:  # put option
            theta = (-S * pdf_d1 * sigma / (2 * math.sqrt(T))
                    + r * K * math.exp(-r * T) * self.norm_cdf(-d2))
        # Annualize theta (per day decay assuming 365 days per year)
        theta /= 365.0

        # Rho: sensitivity to the interest rate.
        if option_type == 'c':
            rho = K * T * math.exp(-r * T) * self.norm_cdf(d2)
        else:
            rho = -K * T * math.exp(-r * T) * self.norm_cdf(-d2)

        return {'delta': delta, 'gamma': gamma, 'vega': vega, 'theta': theta, 'rho': rho}

    def fast_iv_and_greeks(self, option_type, S, K, T, r, market_price, tol=1e-6, max_iter=100):
        """
        Computes both the implied volatility and the Greeks for a European option.
        Handles cases where market price is below intrinsic value.
        """
        iv = self.implied_volatility(option_type, S, K, T, r, market_price, tol, max_iter)

        # If IV calculation failed due to price below intrinsic
        if iv is None:
            return {
                'implied_volatility': None,
                'delta': 1.0 if option_type == 'c' else -1.0,  # Deep ITM delta
                'gamma': 0.0,  # No gamma for pure intrinsic value
                'vega': 0.0,   # No vega for pure intrinsic value
                'theta': 0.0,  # No theta for pure intrinsic value
                'rho': T * K * math.exp(-r * T) if option_type == 'c' else -T * K * math.exp(-r * T)  # Rho for intrinsic value
            }

        greeks = self.black_scholes_greeks(option_type, S, K, T, r, iv)
        greeks['implied_volatility'] = iv
        return greeks


    def get_black_scholes_greeks_iv(self, S, K, timestamp, options_price, r = 0):
        T = ((8-self.day)*1000000-timestamp) / (self.DAYS_IN_YEAR*1000000)    # Time to expiration in years
        option_type = 'c'   # 'c' for call ('p' for put)
        result = self.fast_iv_and_greeks(option_type, S, K, T, r, options_price)
        return result

    def get_current_vwap(self, state: TradingState, product: str):
        try:
            if product not in state.order_depths:
                return 0

            order_depth: OrderDepth = state.order_depths[product]
            total_dolvol = 0
            total_vol = 0

            # Process sell orders (quantities are negative in the orderbook)
            for ask, ask_amount in list(order_depth.sell_orders.items()):
                ask_amount = abs(ask_amount)  # Take absolute value since sell quantities are negative
                total_dolvol += ask * ask_amount
                total_vol += ask_amount

            # Process buy orders (quantities are positive in the orderbook)
            for bid, bid_amount in list(order_depth.buy_orders.items()):
                total_dolvol += bid * bid_amount
                total_vol += bid_amount

            # Calculate VWAP
            current_vwap = total_dolvol / total_vol if total_vol > 0 else 0
            return current_vwap
        except Exception as e:
            logger.print(f"Error calculating VWAP for {product}: {e}")
            return 0

    # Get best bid and ask prices for a product
    def get_best_prices(self, state: TradingState, product: str):
        if product not in state.order_depths:
            return None, None

        order_depth: OrderDepth = state.order_depths[product]

        best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None
        best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None

        return best_bid, best_ask

    # Calculate moneyness for an option
    def calculate_moneyness(self, strike, underlying_price, T):
        return math.log(underlying_price / strike) / math.sqrt(T + 1e-10)

    # Find the most ATM option with highest theta
    def find_highest_theta_option(self, state: TradingState, underlying_price):
        """
        Finds the most at-the-money option with the highest theta value.

        Parameters:
            state (TradingState): Current trading state
            underlying_price (float): Current price of the underlying asset

        Returns:
            dict: Information about the selected option or None if no suitable option found
        """
        logger.print("Finding option with highest theta...")

        options_data = []

        # Calculate ATM-ness and theta for each option
        for voucher in self.voucher_names:
            if voucher not in state.order_depths:
                continue

            strike = self.voucher_strikes[voucher]
            best_bid, best_ask = self.get_best_prices(state, voucher)

            if not best_bid or not best_ask:
                continue

            # Calculate mid price
            mid_price = (best_bid + best_ask) / 2

            # Calculate ATM-ness (how close to at-the-money)
            atm_ness = abs(underlying_price - strike)

            # Get Greeks using mid price
            greeks = self.get_black_scholes_greeks_iv(underlying_price, strike, state.timestamp, mid_price)

            if not greeks or 'theta' not in greeks:
                continue

            # Get theta (negative value, so we take absolute value for comparison)
            theta = abs(greeks['theta'])
            delta = greeks['delta']

            options_data.append({
                'voucher': voucher,
                'strike': strike,
                'best_bid': best_bid,
                'best_ask': best_ask,
                'atm_ness': atm_ness,
                'theta': theta,
                'delta': delta,
                'greeks': greeks
            })

        if not options_data:
            logger.print("No valid options found")
            return None

        # Sort options by ATM-ness (ascending, so most ATM first)
        options_data.sort(key=lambda x: x['atm_ness'])

        # Take the top 3 most ATM options
        most_atm_options = options_data[:3] if len(options_data) >= 3 else options_data

        # From these, find the one with highest theta
        highest_theta_option = max(most_atm_options, key=lambda x: x['theta'])

        logger.print(f"Selected option {highest_theta_option['voucher']} with theta {highest_theta_option['theta']:.6f}")
        return highest_theta_option

    def manage_short_position(self, state: TradingState, underlying_price, result):
        """
        Manages the current short position - either opening a new one or closing an existing one.

        Parameters:
            state (TradingState): Current trading state
            underlying_price (float): Current price of the underlying asset
            result (dict): Dictionary to store orders

        Returns:
            bool: True if a position was opened or closed, False otherwise
        """
        # Initialize orders dictionary for each product if not already present
        for product in self.product_max_positions.keys():
            if product not in result:
                result[product] = []

        # Find the option with highest theta
        highest_theta_option = self.find_highest_theta_option(state, underlying_price)

        if highest_theta_option is None:
            logger.print("No suitable option found for shorting")
            return False

        # If we don't have a current short position, or if the highest theta option has changed
        if self.current_short_option is None or self.current_short_option['voucher'] != highest_theta_option['voucher']:
            # If we have an existing position in a different option, close it first
            if self.current_short_option is not None:
                old_voucher = self.current_short_option['voucher']
                logger.print(f"Current short position in {old_voucher} is no longer optimal, closing position")

                # Get current position in this option
                current_position = state.position.get(old_voucher, 0)

                # If we have a short position (negative position)
                if current_position < 0:
                    # Get best ask price to buy back
                    best_ask = self.get_best_prices(state, old_voucher)[1]

                    if best_ask is None:
                        logger.print(f"Cannot close position in {old_voucher} - No ask price available")
                    else:
                        # Calculate available volume at the ask price
                        available_volume = abs(state.order_depths[old_voucher].sell_orders[best_ask]) if best_ask in state.order_depths[old_voucher].sell_orders else 0

                        # Determine how much to buy back
                        buy_volume = min(abs(current_position), available_volume)

                        if buy_volume <= 0:
                            logger.print(f"Cannot close position in {old_voucher} - No volume available")
                        else:
                            logger.print(f"Buying back {buy_volume} {old_voucher} at {best_ask}")

                            # Add buy order to result (positive quantity means buy)
                            result[old_voucher].append(Order(old_voucher, best_ask, buy_volume))

                            # If we're buying back the full position, clear the current_short_option
                            if buy_volume == abs(current_position):
                                logger.print(f"Fully closing position in {old_voucher}")
                                self.current_short_option = None

            # Now open or add to a position in the highest theta option
            voucher = highest_theta_option['voucher']
            best_bid = highest_theta_option['best_bid']
            delta = highest_theta_option['delta']

            logger.print(f"Opening/adding to position in {voucher} with highest theta")

            # Calculate available volume to short
            available_volume = state.order_depths[voucher].buy_orders[best_bid] if best_bid in state.order_depths[voucher].buy_orders else 0
            max_position = self.product_max_positions[voucher]
            current_position = state.position.get(voucher, 0)

            # Calculate how much more we can short (considering current position)
            remaining_capacity = max_position + current_position  # For short positions (negative), this gives us how much more we can short

            # Determine how much to short (positive number)
            short_volume = min(available_volume, remaining_capacity)

            if short_volume <= 0:
                logger.print(f"Cannot short {voucher} - No volume available or position limit reached")
                return False

            logger.print(f"Shorting {short_volume} {voucher} at {best_bid}")

            # Add short order to result (negative quantity means sell/short)
            result[voucher].append(Order(voucher, best_bid, -short_volume))

            # Store information about the shorted option
            self.current_short_option = {
                'voucher': voucher,
                'strike': highest_theta_option['strike'],
                'short_price': best_bid,
                'short_volume': short_volume,
                'delta': delta
            }

            return True

        # If we already have a position in the highest theta option, check if we can add to it
        else:
            voucher = highest_theta_option['voucher']
            best_bid = highest_theta_option['best_bid']
            current_position = state.position.get(voucher, 0)
            max_position = self.product_max_positions[voucher]

            # If we haven't reached the max position yet
            if current_position > -max_position:
                logger.print(f"Current position in {voucher}: {current_position}, max: {-max_position}, adding to position")

                # Calculate available volume to short
                available_volume = state.order_depths[voucher].buy_orders[best_bid] if best_bid in state.order_depths[voucher].buy_orders else 0

                # Calculate how much more we can short
                remaining_capacity = max_position + current_position

                # Determine how much to short (positive number)
                short_volume = min(available_volume, remaining_capacity)

                if short_volume <= 0:
                    logger.print(f"Cannot add to position in {voucher} - No volume available")
                    return False

                logger.print(f"Adding {short_volume} to short position in {voucher} at {best_bid}")

                # Add short order to result (negative quantity means sell/short)
                result[voucher].append(Order(voucher, best_bid, -short_volume))

                # Update information about the shorted option
                self.current_short_option['short_volume'] += short_volume

                return True
            else:
                logger.print(f"Already at max position for {voucher}: {current_position}")

        return False

    def calculate_delta_hedge(self, state: TradingState, underlying_price, result):
        """
        Calculates and executes delta hedging based on current option positions.
        Only executes orders if the required change exceeds 10 units.

        Parameters:
            state (TradingState): Current trading state
            underlying_price (float): Current price of the underlying asset
            result (dict): Dictionary to store orders
        """
        logger.print("Calculating delta hedge...")

        # Initialize orders dictionary for VOLCANIC_ROCK if not already present
        if 'VOLCANIC_ROCK' not in result:
            result['VOLCANIC_ROCK'] = []

        # Calculate total delta exposure from all option positions
        total_delta_exposure = 0

        # Get current positions for all options
        for voucher in self.voucher_names:
            position = state.position.get(voucher, 0)

            if position == 0:
                continue

            # Get best bid and ask for this option
            best_bid, best_ask = self.get_best_prices(state, voucher)

            if not best_bid or not best_ask:
                logger.print(f"Skipping delta calculation for {voucher}: no prices available")
                continue

            # Calculate mid price
            mid_price = (best_bid + best_ask) / 2

            # Get strike price
            strike = self.voucher_strikes[voucher]

            # Calculate Greeks using mid price
            greeks = self.get_black_scholes_greeks_iv(underlying_price, strike, state.timestamp, mid_price)

            if not greeks or 'delta' not in greeks:
                logger.print(f"Skipping delta calculation for {voucher}: could not calculate Greeks")
                continue

            # Calculate delta exposure for this position
            delta_per_contract = greeks['delta']
            position_delta = delta_per_contract * position
            total_delta_exposure += position_delta

            logger.print(f"Delta for {voucher}: {position_delta:.2f} (delta_per_contract: {delta_per_contract:.4f}, position: {position})")

        logger.print(f"Total delta exposure: {total_delta_exposure:.2f}")

        # Calculate the position we need to take to be fully delta hedged
        # This is the negative of the total delta exposure
        target_rock_position = int(-total_delta_exposure)  # Round to nearest integer

        # Current rock position
        current_rock_position = state.position.get('VOLCANIC_ROCK', 0)

        # Calculate the order we need to place to reach the target position
        order_volume = target_rock_position - current_rock_position

        logger.print(f"Current VOLCANIC_ROCK position: {current_rock_position}")
        logger.print(f"Target VOLCANIC_ROCK position: {target_rock_position}")
        logger.print(f"Order volume needed: {order_volume}")

        # Only hedge if the volume is more than 10
        if abs(order_volume) > 10:
            logger.print(f"Delta hedge volume {abs(order_volume)} > 10, proceeding with hedging")

            # Get best bid and ask for the underlying
            rock_best_bid, rock_best_ask = self.get_best_prices(state, 'VOLCANIC_ROCK')

            if not rock_best_bid or not rock_best_ask:
                logger.print("Cannot execute delta hedge: no prices available for VOLCANIC_ROCK")
                return

            # Check if we have enough capacity to hedge the delta
            max_position = self.product_max_positions['VOLCANIC_ROCK']

            # Calculate available capacity based on direction
            if order_volume > 0:  # Buying
                # We can buy up to max_position - current_position
                available_capacity = max_position - current_rock_position
                if order_volume > available_capacity:
                    order_volume = available_capacity
                    logger.print(f"Adjusted buy volume to {order_volume} due to long position limit")
            else:  # Selling
                # We can sell up to max_position + current_position
                available_capacity = max_position + current_rock_position
                if abs(order_volume) > available_capacity:
                    order_volume = -available_capacity
                    logger.print(f"Adjusted sell volume to {order_volume} due to short position limit")

            if order_volume != 0:
                # Determine price based on direction
                if order_volume > 0:  # Buy
                    price = rock_best_ask
                else:  # Sell
                    price = rock_best_bid

                # Add order to result
                result['VOLCANIC_ROCK'].append(Order('VOLCANIC_ROCK', price, order_volume))
                logger.print(f"Delta hedging: {'Buying' if order_volume > 0 else 'Selling'} {abs(order_volume)} VOLCANIC_ROCK at {price}")
        else:
            logger.print(f"Delta hedge volume {abs(order_volume)} <= 10, skipping hedging")

    def run(self, state: TradingState):
        logger.print("Current Positions: " + str(state.position))

        # Initialize trader data
        trader_data = {}
        if state.traderData:
            try:
                trader_data = jsonpickle.decode(state.traderData)
                # Update current positions from trader data if available
                if 'current_positions' in trader_data:
                    self.current_positions = trader_data['current_positions']
                if 'current_short_option' in trader_data:
                    self.current_short_option = trader_data['current_short_option']
                if 'day' in trader_data:
                    self.day = trader_data['day']
                if 'timestep_counter' in trader_data:
                    self.timestep_counter = trader_data['timestep_counter']
            except Exception as e:
                logger.print(f"Error decoding trader data: {e}")

        # Initialize result dictionary for orders
        result = {}
        for product in self.product_max_positions.keys():
            result[product] = []

        # Detect day from initial price if timestamp is 0
        if state.timestamp == 0:
            rock_ob = state.order_depths['VOLCANIC_ROCK']
            best_bid = max(rock_ob.buy_orders.keys())
            best_ask = min(rock_ob.sell_orders.keys())
            mid_price = (best_bid + best_ask) / 2
            for hc_price in self.t0_rock_prices:
                if mid_price != hc_price:
                    self.day += 1
                    break
            logger.print(f'DAY: {self.day}')

        # Calculate time to expiry
        self.T = ((8-self.day)*1000000-state.timestamp)/(self.DAYS_IN_YEAR*1000000)

        # Update current positions from state
        for product in self.product_max_positions.keys():
            if product in state.position:
                self.current_positions[product] = state.position[product]

        # Get current VWAP for the underlying
        rock_vwap = self.get_current_vwap(state, 'VOLCANIC_ROCK')
        if not rock_vwap:
            # If no VWAP available, try to get mid price
            best_bid, best_ask = self.get_best_prices(state, 'VOLCANIC_ROCK')
            if best_bid and best_ask:
                rock_vwap = (best_bid + best_ask) / 2
            else:
                # If no prices available, use the last known price
                rock_vwap = self.t0_rock_prices[min(self.day - 1, 2)]

        logger.print(f"Underlying VWAP: {rock_vwap}")
        logger.print(f"Time to expiry (years): {self.T}")

        # Increment timestep counter
        self.timestep_counter += 1

        # Main trading logic

        # 1. Manage the short position (open a new one or close an existing one)
        self.manage_short_position(state, rock_vwap, result)

        # 2. Calculate and execute delta hedging
        self.calculate_delta_hedge(state, rock_vwap, result)

        # Update trader data
        trader_data = {
            'current_positions': self.current_positions,
            'current_short_option': self.current_short_option,
            'day': self.day,
            'timestep_counter': self.timestep_counter
        }

        # Remove empty order lists
        for product in list(result.keys()):
            if not result[product]:  # Remove empty order lists
                del result[product]

        # Flush logs
        logger.flush(state, result, 0, jsonpickle.encode(trader_data))

        return result, 0, jsonpickle.encode(trader_data)

