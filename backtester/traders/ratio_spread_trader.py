import json
from typing import Any, Dict, List
import numpy as np
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
import math

# Reuse the Logger class from arb_trader.py
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
        # Initialize product position limits
        self.product_max_positions = {
            'VOLCANIC_ROCK': 400,
            'VOLCANIC_ROCK_VOUCHER_9500': 200,
            'VOLCANIC_ROCK_VOUCHER_9750': 200,
            'VOLCANIC_ROCK_VOUCHER_10000': 200,
            'VOLCANIC_ROCK_VOUCHER_10250': 200,
            'VOLCANIC_ROCK_VOUCHER_10500': 200
        }

        # List of voucher names for easy iteration
        self.voucher_names = [
            'VOLCANIC_ROCK_VOUCHER_9500',
            'VOLCANIC_ROCK_VOUCHER_9750',
            'VOLCANIC_ROCK_VOUCHER_10000',
            'VOLCANIC_ROCK_VOUCHER_10250',
            'VOLCANIC_ROCK_VOUCHER_10500'
        ]

        # Strike prices for each voucher
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

        # Time to expiry and other constants
        self.T = -1
        self.DAYS_IN_YEAR = 365.0
        self.t0_rock_prices = [10503.0, 10516.0, 10218.5]
        self.day = 0

        # Polynomial parameters for IV estimation
        self.iv_poly_params = {
            "ask": [0.15, 0, 0.3],
            "bid": [0.15, 0, 0.15]
        }

        # Track active ratio spreads
        self.active_spreads = []

        # Track spread ratios
        self.spread_ratios = {}

        # Track current day's ATM option
        self.current_atm_option = None

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
        T = ((8-self.day)*1000000-timestamp) / self.DAYS_IN_YEAR    # Days to expiration in years
        option_type = 'c'   # 'c' for call ('p' for put)
        result = self.fast_iv_and_greeks(option_type, S, K, T, r, options_price)
        return result

    # Get current VWAP for a product
    def get_current_vwap(self, state: TradingState, product: str):
        try:
            if product not in state.order_depths:
                return 0

            order_depth: OrderDepth = state.order_depths[product]
            total_dolvol = 0
            total_vol = 0

            # Process sell orders
            for ask, ask_amount in list(order_depth.sell_orders.items()):
                ask_amount = abs(ask_amount)
                total_dolvol += ask * ask_amount
                total_vol += ask_amount

            # Process buy orders
            for bid, bid_amount in list(order_depth.buy_orders.items()):
                total_dolvol += bid * bid_amount
                total_vol += bid_amount

            # Calculate VWAP
            current_vwap = total_dolvol / total_vol if total_vol > 0 else 0
            return current_vwap
        except Exception as e:
            logger.print(f"Error calculating VWAP for {product}: {e}")
            return 0

    # Find the most ATM call option
    def find_atm_option(self, underlying_price):
        min_distance = float('inf')
        atm_option = None

        for voucher in self.voucher_names:
            strike = self.voucher_strikes[voucher]
            distance = abs(strike - underlying_price)

            if distance < min_distance:
                min_distance = distance
                atm_option = voucher

        return atm_option

    # Calculate moneyness for a voucher
    def calculate_moneyness(self, strike, underlying_price, time_to_expiry):
        # Ensure time_to_expiry is positive to avoid issues with sqrt
        time_to_expiry = max(time_to_expiry, 0.0001)
        # Ensure underlying_price and strike are positive to avoid issues with log
        underlying_price = max(underlying_price, 0.0001)
        strike = max(strike, 0.0001)
        return np.log(underlying_price / strike) / np.sqrt(time_to_expiry)

    # Check if a voucher is OTM
    def is_otm(self, strike, underlying_price):
        return strike > underlying_price

    # Get best bid and ask prices for a product
    def get_best_prices(self, state: TradingState, product: str):
        try:
            if product not in state.order_depths:
                return None, None

            order_depth: OrderDepth = state.order_depths[product]
            best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None
            best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None
            return best_bid, best_ask
        except Exception as e:
            logger.print(f"Error getting best prices for {product}: {e}")
            return None, None

    # Calculate the ratio for a ratio spread
    def calculate_ratio(self, atm_delta, otm_delta):
        # Avoid division by zero
        if otm_delta == 0:
            return 0

        # Calculate the ratio to be delta neutral
        ratio = abs(atm_delta / otm_delta)

        # Round to nearest integer for simplicity
        return round(ratio)

    # Check if we should close existing spreads
    def should_close_spreads(self, state: TradingState, underlying_price, timestamp):
        # If no active spreads, no need to close
        if not self.active_spreads:
            return False

        # Check if market conditions have changed significantly
        for spread in self.active_spreads:
            atm_option = spread['atm_option']
            otm_option = spread['otm_option']
            is_regular = spread['is_regular']

            # Get current prices
            atm_strike = self.voucher_strikes[atm_option]
            otm_strike = self.voucher_strikes[otm_option]

            # Calculate time to expiry (used in moneyness calculation below)
            T = ((8-self.day)*1000000-timestamp) / self.DAYS_IN_YEAR

            # Get best prices
            atm_best_bid, atm_best_ask = self.get_best_prices(state, atm_option)
            otm_best_bid, otm_best_ask = self.get_best_prices(state, otm_option)

            if not atm_best_bid or not atm_best_ask or not otm_best_bid or not otm_best_ask:
                continue

            # Calculate moneyness
            atm_moneyness = self.calculate_moneyness(atm_strike, underlying_price, T)
            otm_moneyness = self.calculate_moneyness(otm_strike, underlying_price, T)

            # Calculate IVs
            atm_est_bid_iv = self.estimate_iv(atm_moneyness, "bid")
            atm_est_ask_iv = self.estimate_iv(atm_moneyness, "ask")
            otm_est_bid_iv = self.estimate_iv(otm_moneyness, "bid")
            otm_est_ask_iv = self.estimate_iv(otm_moneyness, "ask")

            atm_actual_bid_iv = self.get_black_scholes_greeks_iv(underlying_price, atm_strike, timestamp, atm_best_bid)['implied_volatility']
            atm_actual_ask_iv = self.get_black_scholes_greeks_iv(underlying_price, atm_strike, timestamp, atm_best_ask)['implied_volatility']
            otm_actual_bid_iv = self.get_black_scholes_greeks_iv(underlying_price, otm_strike, timestamp, otm_best_bid)['implied_volatility']
            otm_actual_ask_iv = self.get_black_scholes_greeks_iv(underlying_price, otm_strike, timestamp, otm_best_ask)['implied_volatility']

            # Check if conditions are no longer favorable
            if is_regular:
                # For regular strategy, we want ATM ask IV < bid estimated IV and OTM bid IV > ask estimated IV
                if not (atm_actual_ask_iv is not None and atm_est_bid_iv is not None and atm_actual_ask_iv < atm_est_bid_iv) or \
                   not (otm_actual_bid_iv is not None and otm_est_ask_iv is not None and otm_actual_bid_iv > otm_est_ask_iv):
                    return True
            else:
                # For reverse strategy, we want ATM bid IV > ask estimated IV and OTM ask IV < bid estimated IV
                if not (atm_actual_bid_iv is not None and atm_est_ask_iv is not None and atm_actual_bid_iv > atm_est_ask_iv) or \
                   not (otm_actual_ask_iv is not None and otm_est_bid_iv is not None and otm_actual_ask_iv < otm_est_bid_iv):
                    return True

        return False

    # Close all existing ratio spreads
    def close_all_spreads(self, state: TradingState, orders: dict):
        for spread in self.active_spreads:
            atm_option = spread['atm_option']
            otm_option = spread['otm_option']
            atm_position = self.current_positions[atm_option]
            otm_position = self.current_positions[otm_option]

            # Close ATM position
            if atm_position > 0:
                # Sell to close
                best_bid, _ = self.get_best_prices(state, atm_option)
                if best_bid:
                    orders.setdefault(atm_option, []).append(Order(atm_option, best_bid, -atm_position))
            elif atm_position < 0:
                # Buy to close
                _, best_ask = self.get_best_prices(state, atm_option)
                if best_ask:
                    orders.setdefault(atm_option, []).append(Order(atm_option, best_ask, -atm_position))

            # Close OTM position
            if otm_position > 0:
                # Sell to close
                best_bid, _ = self.get_best_prices(state, otm_option)
                if best_bid:
                    orders.setdefault(otm_option, []).append(Order(otm_option, best_bid, -otm_position))
            elif otm_position < 0:
                # Buy to close
                _, best_ask = self.get_best_prices(state, otm_option)
                if best_ask:
                    orders.setdefault(otm_option, []).append(Order(otm_option, best_ask, -otm_position))

        # Clear active spreads
        self.active_spreads = []
        self.spread_ratios = {}

    # Check IV conditions for ratio spread
    def check_iv_conditions(self, state: TradingState, underlying_price, timestamp):
        atm_option = self.find_atm_option(underlying_price)
        if not atm_option:
            return None, None, False

        self.current_atm_option = atm_option
        atm_strike = self.voucher_strikes[atm_option]

        # Calculate time to expiry (used in moneyness calculation below)
        T = ((8-self.day)*1000000-timestamp) / self.DAYS_IN_YEAR

        # Calculate moneyness for ATM option
        atm_moneyness = self.calculate_moneyness(atm_strike, underlying_price, T)

        # Get best prices for ATM option
        atm_best_bid, atm_best_ask = self.get_best_prices(state, atm_option)
        if not atm_best_ask or not atm_best_bid:
            return None, None, False

        # Calculate estimated IV for ATM option
        atm_est_bid_iv = self.estimate_iv(atm_moneyness, "bid")
        atm_est_ask_iv = self.estimate_iv(atm_moneyness, "ask")

        # Calculate actual IV for ATM option
        atm_actual_bid_iv = self.get_black_scholes_greeks_iv(underlying_price, atm_strike, timestamp, atm_best_bid)['implied_volatility']
        atm_actual_ask_iv = self.get_black_scholes_greeks_iv(underlying_price, atm_strike, timestamp, atm_best_ask)['implied_volatility']

        # Check if ATM ask IV is below the bid estimated IV
        atm_condition = atm_actual_ask_iv is not None and atm_est_bid_iv is not None and atm_actual_ask_iv < atm_est_bid_iv

        # Find an OTM option with bid IV above estimated ask IV
        selected_otm = None
        for voucher in self.voucher_names:
            if voucher == atm_option or not self.is_otm(self.voucher_strikes[voucher], underlying_price):
                continue

            otm_strike = self.voucher_strikes[voucher]
            otm_moneyness = self.calculate_moneyness(otm_strike, underlying_price, T)

            otm_best_bid, _ = self.get_best_prices(state, voucher)
            if not otm_best_bid:
                continue

            otm_est_ask_iv = self.estimate_iv(otm_moneyness, "ask")
            otm_actual_bid_iv = self.get_black_scholes_greeks_iv(underlying_price, otm_strike, timestamp, otm_best_bid)['implied_volatility']

            if otm_actual_bid_iv is not None and otm_est_ask_iv is not None and otm_actual_bid_iv > otm_est_ask_iv:
                selected_otm = voucher
                break

        if atm_condition and selected_otm:
            return atm_option, selected_otm, True

        # Check for reverse condition (ATM bid IV > ask estimated IV and OTM ask IV < bid estimated IV)
        atm_reverse_condition = atm_actual_bid_iv is not None and atm_est_ask_iv is not None and atm_actual_bid_iv > atm_est_ask_iv

        selected_otm_reverse = None
        for voucher in self.voucher_names:
            if voucher == atm_option or not self.is_otm(self.voucher_strikes[voucher], underlying_price):
                continue

            otm_strike = self.voucher_strikes[voucher]
            otm_moneyness = self.calculate_moneyness(otm_strike, underlying_price, T)

            _, otm_best_ask = self.get_best_prices(state, voucher)
            if not otm_best_ask:
                continue

            otm_est_bid_iv = self.estimate_iv(otm_moneyness, "bid")
            otm_actual_ask_iv = self.get_black_scholes_greeks_iv(underlying_price, otm_strike, timestamp, otm_best_ask)['implied_volatility']

            if otm_actual_ask_iv is not None and otm_est_bid_iv is not None and otm_actual_ask_iv < otm_est_bid_iv:
                selected_otm_reverse = voucher
                break

        if atm_reverse_condition and selected_otm_reverse:
            return atm_option, selected_otm_reverse, False  # False indicates reverse strategy

        return None, None, False

    # Execute ratio spread strategy
    def execute_ratio_spread(self, state: TradingState, atm_option, otm_option, is_regular, underlying_price, timestamp, orders):
        # Get strikes
        atm_strike = self.voucher_strikes[atm_option]
        otm_strike = self.voucher_strikes[otm_option]

        # Calculate time to expiry (used in Greeks calculation inside get_black_scholes_greeks_iv)

        # Get best prices
        atm_best_bid, atm_best_ask = self.get_best_prices(state, atm_option)
        otm_best_bid, otm_best_ask = self.get_best_prices(state, otm_option)

        if not atm_best_bid or not atm_best_ask or not otm_best_bid or not otm_best_ask:
            return

        # Calculate Greeks for ATM and OTM options
        atm_greeks = self.get_black_scholes_greeks_iv(underlying_price, atm_strike, timestamp,
                                                     atm_best_ask if is_regular else atm_best_bid)
        otm_greeks = self.get_black_scholes_greeks_iv(underlying_price, otm_strike, timestamp,
                                                     otm_best_bid if is_regular else otm_best_ask)

        # Get deltas
        atm_delta = atm_greeks['delta']
        otm_delta = otm_greeks['delta']

        if atm_delta is None or otm_delta is None:
            return

        # Calculate ratio for delta neutrality
        ratio = self.calculate_ratio(atm_delta, otm_delta)
        if ratio <= 0:
            return

        # Store the ratio for this spread
        spread_key = f"{atm_option}_{otm_option}"
        self.spread_ratios[spread_key] = ratio

        # Calculate maximum positions based on position limits
        atm_max_position = self.product_max_positions[atm_option]
        otm_max_position = self.product_max_positions[otm_option]

        # Calculate current positions
        atm_current_position = self.current_positions[atm_option]
        otm_current_position = self.current_positions[otm_option]

        # Calculate available position capacity
        if is_regular:
            # For regular strategy: buy ATM, sell OTM
            atm_available = atm_max_position - atm_current_position
            otm_available = otm_current_position - (-otm_max_position)  # Convert to positive
        else:
            # For reverse strategy: sell ATM, buy OTM
            atm_available = atm_current_position - (-atm_max_position)  # Convert to positive
            otm_available = otm_max_position - otm_current_position

        # Determine the maximum number of spreads we can execute
        if is_regular:
            max_spreads = min(atm_available, otm_available // ratio)
        else:
            max_spreads = min(atm_available // ratio, otm_available)

        # Limit to a reasonable number of spreads per execution
        max_spreads = min(max_spreads, 10)

        if max_spreads <= 0:
            return

        # Calculate quantities for ATM and OTM options
        if is_regular:
            atm_quantity = max_spreads
            otm_quantity = -max_spreads * ratio
        else:
            atm_quantity = -max_spreads * ratio
            otm_quantity = max_spreads

        # Execute the trades
        if is_regular:
            # Buy ATM calls
            orders.setdefault(atm_option, []).append(Order(atm_option, atm_best_ask, atm_quantity))
            # Sell OTM calls
            orders.setdefault(otm_option, []).append(Order(otm_option, otm_best_bid, otm_quantity))
        else:
            # Sell ATM calls
            orders.setdefault(atm_option, []).append(Order(atm_option, atm_best_bid, atm_quantity))
            # Buy OTM calls
            orders.setdefault(otm_option, []).append(Order(otm_option, otm_best_ask, otm_quantity))

        # Record the spread
        self.active_spreads.append({
            'atm_option': atm_option,
            'otm_option': otm_option,
            'ratio': ratio,
            'is_regular': is_regular
        })

        # Calculate net delta for delta hedging
        net_delta = atm_quantity * atm_delta + otm_quantity * otm_delta

        # Delta hedge with the underlying
        self.delta_hedge(state, net_delta, orders)

    # Delta hedge with the underlying
    def delta_hedge(self, state: TradingState, net_delta, orders):
        # Calculate the quantity of underlying needed to hedge
        hedge_quantity = -round(net_delta * 100)  # Scale delta to match contract size

        # Check if we need to hedge
        if hedge_quantity == 0:
            return

        # Get current position in the underlying
        current_position = self.current_positions['VOLCANIC_ROCK']

        # Calculate new position after hedging
        new_position = current_position + hedge_quantity

        # Check position limits
        max_position = self.product_max_positions['VOLCANIC_ROCK']
        if abs(new_position) > max_position:
            # Scale down the hedge to respect position limits
            if new_position > 0:
                hedge_quantity = max_position - current_position
            else:
                hedge_quantity = -max_position - current_position

        # Execute the hedge if needed
        if hedge_quantity != 0:
            # Get best prices for the underlying
            best_bid, best_ask = self.get_best_prices(state, 'VOLCANIC_ROCK')

            if hedge_quantity > 0 and best_ask:
                # Buy underlying
                orders.setdefault('VOLCANIC_ROCK', []).append(Order('VOLCANIC_ROCK', best_ask, hedge_quantity))
            elif hedge_quantity < 0 and best_bid:
                # Sell underlying
                orders.setdefault('VOLCANIC_ROCK', []).append(Order('VOLCANIC_ROCK', best_bid, hedge_quantity))

    # Main trading function
    def run(self, state: TradingState) -> Dict[str, List[Order]]:
        # Initialize orders dictionary
        orders: Dict[str, List[Order]] = {}

        # Update current positions
        for product in self.product_max_positions.keys():
            if product in state.position:
                self.current_positions[product] = state.position[product]

        # Get timestamp and update day if needed
        timestamp = state.timestamp
        if timestamp < 1000000:
            self.day = 1
        elif timestamp < 2000000:
            self.day = 2
        elif timestamp < 3000000:
            self.day = 3
        elif timestamp < 4000000:
            self.day = 4
        elif timestamp < 5000000:
            self.day = 5
        elif timestamp < 6000000:
            self.day = 6
        elif timestamp < 7000000:
            self.day = 7
        else:
            self.day = 8

        # Get current price of the underlying
        underlying_price = self.get_current_vwap(state, 'VOLCANIC_ROCK')
        if not underlying_price:
            # If no VWAP available, try to get mid price
            best_bid, best_ask = self.get_best_prices(state, 'VOLCANIC_ROCK')
            if best_bid and best_ask:
                underlying_price = (best_bid + best_ask) / 2
            else:
                # If no prices available, use the last known price
                underlying_price = self.t0_rock_prices[min(self.day - 1, 2)]

        # Check if we should close existing spreads
        if self.should_close_spreads(state, underlying_price, timestamp):
            self.close_all_spreads(state, orders)

        # Check for new ratio spread opportunities
        atm_option, otm_option, is_regular = self.check_iv_conditions(state, underlying_price, timestamp)

        if atm_option and otm_option:
            # Execute the ratio spread strategy
            self.execute_ratio_spread(state, atm_option, otm_option, is_regular, underlying_price, timestamp, orders)

        # Convert trader data to JSON string
        trader_data = json.dumps({
            'day': self.day,
            'active_spreads': self.active_spreads,
            'spread_ratios': self.spread_ratios,
            'current_positions': self.current_positions
        })

        # Flush logs
        logger.flush(state, orders, 0, trader_data)

        return orders