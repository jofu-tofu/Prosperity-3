import math
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
    def __init__(self):
        # Maximum positions for each product
        self.product_max_positions = {
            'VOLCANIC_ROCK': 400,
            'VOLCANIC_ROCK_VOUCHER_9500': 200,
            'VOLCANIC_ROCK_VOUCHER_9750': 200,
            'VOLCANIC_ROCK_VOUCHER_10000': 200,
            'VOLCANIC_ROCK_VOUCHER_10250': 200,
            'VOLCANIC_ROCK_VOUCHER_10500': 200
        }

        # Voucher names and strikes
        self.voucher_names = [
            'VOLCANIC_ROCK_VOUCHER_9500',
            'VOLCANIC_ROCK_VOUCHER_9750',
            'VOLCANIC_ROCK_VOUCHER_10000',
            'VOLCANIC_ROCK_VOUCHER_10250',
            'VOLCANIC_ROCK_VOUCHER_10500'
        ]

        self.voucher_strikes = {
            'VOLCANIC_ROCK_VOUCHER_9500': 9500.0,
            'VOLCANIC_ROCK_VOUCHER_9750': 9750.0,
            'VOLCANIC_ROCK_VOUCHER_10000': 10000.0,
            'VOLCANIC_ROCK_VOUCHER_10250': 10250.0,
            'VOLCANIC_ROCK_VOUCHER_10500': 10500.0
        }

        # Track current positions
        self.current_positions = {
            'VOLCANIC_ROCK': 0,
            'VOLCANIC_ROCK_VOUCHER_9500': 0,
            'VOLCANIC_ROCK_VOUCHER_9750': 0,
            'VOLCANIC_ROCK_VOUCHER_10000': 0,
            'VOLCANIC_ROCK_VOUCHER_10250': 0,
            'VOLCANIC_ROCK_VOUCHER_10500': 0
        }

        # Time-related variables
        self.T = -1
        self.DAYS_IN_YEAR = 365.0
        self.t0_rock_prices = [10503.0, 10516.0, 10218.5]
        self.day = 0

        # IV polynomial parameters
        self.iv_poly_params = {
            "ask": [0.15, 0, 0.3],
            "bid": [0.15, 0, 0.15]
        }

        # Timestep counter for recalculation
        self.timestep_counter = 0

        # Track options we've shorted
        self.shorted_options = {}

    # IV estimation function using polynomial parameters
    def estimate_iv(self, moneyness, price_type):
        return self.iv_poly_params[price_type][0] + self.iv_poly_params[price_type][1] * moneyness + self.iv_poly_params[price_type][2] * moneyness**2

    # Standard normal probability density function
    def norm_pdf(self, x):
        return math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)

    # Standard normal cumulative distribution function
    def norm_cdf(self, x):
        return 0.5 * (1 + math.erf(x / math.sqrt(2)))

    # Black-Scholes price calculation
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

    # Black-Scholes vega calculation
    def black_scholes_vega(self, S, K, T, r, sigma):
        """
        Computes the Black-Scholes vega, the sensitivity of the option's price
        to changes in volatility.
        """
        d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
        return S * self.norm_pdf(d1) * math.sqrt(T)

    # Implied volatility calculation
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

        # Initial guess for implied volatility
        sigma = 0.3  # Start with a reasonable guess

        for _ in range(max_iter):  # Use _ for unused loop variable
            price = self.black_scholes_price(option_type, S, K, T, r, sigma)
            price_diff = market_price - price

            # If the price difference is within tolerance, return the current sigma
            if abs(price_diff) < tol:
                return sigma

            # Calculate vega to update sigma
            vega = self.black_scholes_vega(S, K, T, r, sigma)

            # Avoid division by zero
            if abs(vega) < 1e-10:
                return sigma

            # Update sigma using Newton-Raphson
            sigma_change = price_diff / vega

            # Limit the change in sigma to avoid overshooting
            sigma_change = max(min(sigma_change, 0.5), -0.5)

            sigma += sigma_change

            # Ensure sigma remains positive
            if sigma <= 0:
                sigma = 0.0001

        # If we've reached the maximum iterations, return the current sigma
        return sigma

    # Black-Scholes Greeks calculation
    def black_scholes_greeks(self, option_type, S, K, T, r, sigma):
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

    # Fast IV and Greeks calculation
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

    # Get Black-Scholes Greeks and IV
    def get_black_scholes_greeks_iv(self, S, K, timestamp, options_price, r=0):
        T = ((8-self.day)*1000000-timestamp) / (self.DAYS_IN_YEAR*1000000)
        option_type = 'c'   # 'c' for call ('p' for put)
        result = self.fast_iv_and_greeks(option_type, S, K, T, r, options_price)
        return result

    # Calculate VWAP for a product
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

    # Calculate gamma/delta ratio for an option
    def calculate_gamma_delta_ratio(self, greeks):
        if greeks is None or 'gamma' not in greeks or 'delta' not in greeks:
            return 0

        delta = abs(greeks['delta'])
        gamma = greeks['gamma']

        # Avoid division by zero
        if delta < 0.001:
            delta = 0.001

        return gamma / delta

    # Calculate moneyness for an option
    def calculate_moneyness(self, strike, underlying_price, T):
        return math.log(underlying_price / strike) / math.sqrt(T + 1e-10)

    # Sort options by gamma/delta ratio
    def sort_options_by_gamma_delta_ratio(self, state: TradingState, underlying_price, timestamp):
        options_data = []

        for voucher in self.voucher_names:
            if voucher not in state.order_depths:
                continue

            strike = self.voucher_strikes[voucher]
            best_bid, best_ask = self.get_best_prices(state, voucher)

            if not best_bid or not best_ask:
                continue

            bid_greeks = self.get_black_scholes_greeks_iv(underlying_price, strike, timestamp, best_bid)
            ask_greeks = self.get_black_scholes_greeks_iv(underlying_price, strike, timestamp, best_ask)

            if not bid_greeks or not ask_greeks:
                continue

            gamma_delta_ratio = self.calculate_gamma_delta_ratio(bid_greeks)

            # Calculate moneyness for IV estimation
            moneyness = self.calculate_moneyness(strike, underlying_price, self.T)
            estimated_bid_iv = self.estimate_iv(moneyness, 'bid')
            estimated_ask_iv = self.estimate_iv(moneyness, 'ask')

            # Get actual IVs
            actual_bid_iv = bid_greeks['implied_volatility']
            actual_ask_iv = ask_greeks['implied_volatility']

            options_data.append({
                'voucher': voucher,
                'strike': strike,
                'best_bid': best_bid,
                'best_ask': best_ask,
                'bid_greeks': bid_greeks,
                'ask_greeks': ask_greeks,
                'gamma_delta_ratio': gamma_delta_ratio,
                'moneyness': moneyness,
                'estimated_bid_iv': estimated_bid_iv,
                'estimated_ask_iv': estimated_ask_iv,
                'actual_bid_iv': actual_bid_iv,
                'actual_ask_iv': actual_ask_iv
            })

        # Sort by gamma/delta ratio in descending order
        options_data.sort(key=lambda x: x['gamma_delta_ratio'], reverse=True)
        return options_data

    # Check and buy back options where ask IV is below predicted bid IV
    def check_and_buy_back_options(self, state: TradingState, underlying_price, result):
        # Skip if no options have been shorted
        if not self.shorted_options:
            return

        logger.print("Checking for options to buy back...")

        # Make a copy of shorted_options to avoid modifying during iteration
        shorted_options_copy = dict(self.shorted_options)

        for voucher, _ in shorted_options_copy.items():
            if voucher not in state.order_depths:
                continue

            strike = self.voucher_strikes[voucher]
            best_bid, best_ask = self.get_best_prices(state, voucher)

            if not best_bid or not best_ask:
                continue

            # Calculate moneyness for IV estimation
            moneyness = self.calculate_moneyness(strike, underlying_price, self.T)
            estimated_ask_iv = self.estimate_iv(moneyness, 'ask')

            # Get actual ask IV
            ask_greeks = self.get_black_scholes_greeks_iv(underlying_price, strike, state.timestamp, best_ask)
            if not ask_greeks or 'implied_volatility' not in ask_greeks:
                continue

            actual_ask_iv = ask_greeks['implied_volatility']

            # Check if ask IV is now below the predicted ask IV
            if actual_ask_iv is not None and estimated_ask_iv is not None and actual_ask_iv < estimated_ask_iv:
                logger.print(f"Buying back {voucher} - Ask IV: {actual_ask_iv:.4f} < Estimated Ask IV: {estimated_ask_iv:.4f}")

                # Calculate how much to buy back (close position to 0)
                current_position = self.current_positions[voucher]
                if current_position < 0:  # We have a short position
                    # Get available volume at the ask price
                    # Sell order quantities in the orderbook are negative, so we need to take the absolute value
                    available_volume = abs(state.order_depths[voucher].sell_orders[best_ask]) if best_ask in state.order_depths[voucher].sell_orders else 0
                    buy_volume = min(abs(current_position), available_volume)

                    if buy_volume > 0:
                        # Add buy order to result
                        # For our orders, positive quantity means buy
                        result[voucher].append(Order(voucher, best_ask, buy_volume))

                        # Update current position (will be fully updated in next timestep from state)
                        self.current_positions[voucher] += buy_volume

                        # If position is fully closed, remove from shorted_options
                        if self.current_positions[voucher] >= 0:
                            del self.shorted_options[voucher]
                            logger.print(f"Fully closed position in {voucher}")

    # Find sell opportunities based on gamma/delta ratio
    def find_sell_opportunities(self, state: TradingState, underlying_price, result):
        logger.print("Finding sell opportunities...")

        # Sort options by gamma/delta ratio
        sorted_options = self.sort_options_by_gamma_delta_ratio(state, underlying_price, state.timestamp)

        if not sorted_options:
            logger.print("No valid options found for gamma/delta ratio strategy")
            return

        # Process options in order of highest gamma/delta ratio

        # Iterate through options in order of highest gamma/delta ratio
        for option_data in sorted_options:
            voucher = option_data['voucher']
            best_bid = option_data['best_bid']
            actual_bid_iv = option_data['actual_bid_iv']
            estimated_ask_iv = option_data['estimated_ask_iv']
            bid_greeks = option_data['bid_greeks']

            # Check if bid IV is above the ask estimated IV
            if actual_bid_iv is not None and estimated_ask_iv is not None and actual_bid_iv > estimated_ask_iv:
                logger.print(f"Found opportunity in {voucher} - Bid IV: {actual_bid_iv:.4f} > Estimated Ask IV: {estimated_ask_iv:.4f}")

                # Calculate available volume to short
                # Buy order quantities in the orderbook are positive
                available_volume = state.order_depths[voucher].buy_orders[best_bid] if best_bid in state.order_depths[voucher].buy_orders else 0
                current_position = self.current_positions[voucher]
                max_position = self.product_max_positions[voucher]
                # For short positions, we need to calculate how much more we can short
                # If max_position is 200, and current_position is -50, we can short 150 more
                remaining_capacity = max_position + current_position

                # Determine how much to short (positive number)
                short_volume = min(available_volume, remaining_capacity)

                if short_volume <= 0:
                    logger.print(f"Cannot short {voucher} - No capacity or volume available")
                    continue

                logger.print(f"Available volume: {available_volume}, Remaining capacity: {remaining_capacity}, Short volume: {short_volume}")

                # Get delta for storing in shorted_options
                delta_per_contract = bid_greeks['delta']

                # Add short order to result
                # For our orders, negative quantity means sell/short
                result[voucher].append(Order(voucher, best_bid, -short_volume))
                logger.print(f"Shorting {short_volume} {voucher} at {best_bid}")

                # Store information about shorted option
                self.shorted_options[voucher] = {
                    'short_price': best_bid,
                    'short_volume': short_volume,
                    'delta_per_contract': delta_per_contract,
                    'timestamp': state.timestamp
                }

                # Update current position (will be fully updated in next timestep from state)
                self.current_positions[voucher] -= short_volume

        return

    # Calculate and execute delta hedging based on current positions and pending orders
    def calculate_and_execute_delta_hedging(self, state: TradingState, result, _):
        logger.print("Calculating delta hedging based on end-of-timestep positions...")

        # Get mid price for the underlying
        rock_best_bid, rock_best_ask = self.get_best_prices(state, 'VOLCANIC_ROCK')
        if not rock_best_bid or not rock_best_ask:
            logger.print("Cannot calculate delta hedging: no prices available for VOLCANIC_ROCK")
            return

        rock_mid_price = (rock_best_bid + rock_best_ask) / 2

        # Track positions we'll have at the end of this timestep
        end_positions = self.current_positions.copy()

        # Add pending orders to end positions
        for product, orders_list in result.items():
            for order in orders_list:
                if product in end_positions:
                    end_positions[product] += order.quantity

        logger.print(f"Current positions: {self.current_positions}")
        logger.print(f"Projected end positions: {end_positions}")

        # Calculate total delta exposure from all option positions
        total_delta_exposure = 0

        # Calculate delta for each option position
        for voucher in self.voucher_names:
            position = end_positions.get(voucher, 0)
            if position == 0:
                continue

            strike = self.voucher_strikes[voucher]

            # Calculate mid price IV for this option
            best_bid, best_ask = self.get_best_prices(state, voucher)
            if not best_bid or not best_ask:
                logger.print(f"Skipping delta calculation for {voucher}: no prices available")
                continue

            option_mid_price = (best_bid + best_ask) / 2

            # Calculate Greeks using mid prices
            greeks = self.get_black_scholes_greeks_iv(rock_mid_price, strike, state.timestamp, option_mid_price)
            if not greeks or 'delta' not in greeks:
                logger.print(f"Skipping delta calculation for {voucher}: could not calculate Greeks")
                continue

            # Calculate delta exposure for this position
            delta_per_contract = greeks['delta']
            position_delta = delta_per_contract * position
            total_delta_exposure += position_delta

            logger.print(f"Delta for {voucher}: {position_delta} (delta_per_contract: {delta_per_contract}, position: {position})")

        logger.print(f"Total delta exposure: {total_delta_exposure}")

        # Calculate the position we need to take to be fully delta hedged
        # This is the negative of the total delta exposure
        target_rock_position = int(-total_delta_exposure)  # Round to nearest integer

        # Current rock position (including pending orders)
        current_rock_position = end_positions.get('VOLCANIC_ROCK', 0)

        # Calculate the order we need to place to reach the target position
        order_volume = target_rock_position - current_rock_position

        logger.print(f"Current VOLCANIC_ROCK position: {current_rock_position}")
        logger.print(f"Target VOLCANIC_ROCK position: {target_rock_position}")
        logger.print(f"Order volume needed: {order_volume}")

        # Only hedge if the volume is more than 25
        if abs(order_volume) > 25:
            logger.print(f"Delta hedge volume {abs(order_volume)} > 25, proceeding with hedging")

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

                # For our orders, positive quantity means buy, negative means sell
                result['VOLCANIC_ROCK'].append(Order('VOLCANIC_ROCK', price, order_volume))
                logger.print(f"Delta hedging: {'Buying' if order_volume > 0 else 'Selling'} {abs(order_volume)} VOLCANIC_ROCK at {price}")

                # Update current position (will be fully updated in next timestep from state)
                self.current_positions['VOLCANIC_ROCK'] += order_volume
        else:
            logger.print(f"Delta hedge volume {abs(order_volume)} <= 25, skipping hedging")

    # Execute the gamma/delta ratio strategy (for backward compatibility)
    def execute_gamma_ratio_strategy(self, state: TradingState, underlying_price, result):
        # This method now just calls the two separate methods
        self.find_sell_opportunities(state, underlying_price, result)
        self.calculate_and_execute_delta_hedging(state, result, underlying_price)

    # Main run method
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
                if 'shorted_options' in trader_data:
                    self.shorted_options = trader_data['shorted_options']
                if 'day' in trader_data:
                    self.day = trader_data['day']
                if 'timestep_counter' in trader_data:
                    self.timestep_counter = trader_data['timestep_counter']
            except Exception as e:
                logger.print(f"Error decoding trader data: {e}")

        # Initialize result dictionary for orders
        result = {}

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
        logger.print(f"Timestep counter: {self.timestep_counter}")

        # Increment timestep counter
        self.timestep_counter += 1

        # Initialize orders for each product
        for product in self.product_max_positions.keys():
            result[product] = []

        # Log the order depths to understand what we're working with
        for product in state.order_depths:
            if product in self.product_max_positions:
                order_depth = state.order_depths[product]
                logger.print(f"{product} order depth - Buy orders: {order_depth.buy_orders}, Sell orders: {order_depth.sell_orders}")

        # Main trading logic

        # Check if rock position is near position limits
        rock_position = self.current_positions.get('VOLCANIC_ROCK', 0)
        max_rock_position = self.product_max_positions['VOLCANIC_ROCK']
        position_limit_threshold = 0.8  # 80% of max position
        near_position_limit = abs(rock_position) > position_limit_threshold * max_rock_position

        if near_position_limit:
            logger.print(f"VOLCANIC_ROCK position ({rock_position}) is near position limit ({max_rock_position}), skipping gamma strategy")

            # Only buy back options to reduce exposure
            self.check_and_buy_back_options(state, rock_vwap, result)
        else:
            # 1. Check if we need to buy back any options
            self.check_and_buy_back_options(state, rock_vwap, result)

            # 2. Look for sell opportunities
            self.find_sell_opportunities(state, rock_vwap, result)

        # 3. Calculate and execute delta hedging based on end-of-timestep positions
        self.calculate_and_execute_delta_hedging(state, result, rock_vwap)

        # Update trader data
        trader_data = {
            'current_positions': self.current_positions,
            'shorted_options': self.shorted_options,
            'day': self.day,
            'timestep_counter': self.timestep_counter
        }

        # Flush logs
        logger.flush(state, result, 0, jsonpickle.encode(trader_data))

        # The expected return format is a dictionary mapping product names to lists of orders
        # Make sure we're returning the correct format
        for product in list(result.keys()):
            if not result[product]:  # Remove empty order lists
                del result[product]

        logger.print(f"Final orders: {result}")

        return result, 0, jsonpickle.encode(trader_data)
