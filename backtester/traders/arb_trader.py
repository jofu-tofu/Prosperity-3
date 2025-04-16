import json
from typing import Any, Dict, List, Tuple, Optional
import numpy as np
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
            "ask": [0.15, 0 , 0.3],
            "bid": [0.15, 0 , 0.15]
        }

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
        for i in range(max_iter):
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
        T = ((8-self.day)*1000000-timestamp) / self.DAYS_IN_YEAR    # 30 days to expiration in years
        option_type = 'c'   # 'c' for call ('p' for put)
        result = self.fast_iv_and_greeks(option_type, S, K, T, r, options_price)

        return result

    def get_current_vwap(self, state: TradingState, product: str):
        order_depth: OrderDepth = state.order_depths[product]
        total_dolvol = 0
        total_vol = 0
        for ask, ask_amount in list(order_depth.sell_orders.items()):
            ask_amount = abs(ask_amount)
            total_dolvol += ask * ask_amount
            total_vol += ask_amount
        for bid, bid_amount in list(order_depth.buy_orders.items()):
            total_dolvol += bid * bid_amount
            total_vol += bid_amount
        current_vwap = total_dolvol / total_vol if total_vol > 0 else 0
        return current_vwap

    def sort_by_theta_delta_ratio(self, greeks_by_strike):
        """
        Sorts options by absolute theta/delta ratio from highest to lowest.

        Parameters:
            greeks_by_strike (dict): Dictionary mapping strikes to their Greeks

        Returns:
            list: Sorted list of tuples (strike, theta/delta ratio)
        """
        ratios = []
        for strike, greeks in greeks_by_strike.items():
            delta = greeks['delta']
            theta = greeks['theta']

            # Avoid division by zero and tiny deltas
            if abs(delta) > 0.01:
                ratio = abs(theta/delta)
                ratios.append((strike, ratio))

        # Sort by ratio in descending order
        return sorted(ratios, key=lambda x: x[1], reverse=True)

    def run(self, state: TradingState):
        print("Current Positions: " + str(state.position))
        if state.traderData:
            traderData = jsonpickle.decode(state.traderData)
        else:
            traderData = {}
        result = {}
        rock_ob = state.order_depths['VOLCANIC_ROCK']

        # Check if rock order book has orders
        if not rock_ob.buy_orders or not rock_ob.sell_orders:
            return {}, 0, jsonpickle.encode(traderData)

        if state.timestamp == 0:
            best_bid = max(rock_ob.buy_orders.keys())
            best_ask = min(rock_ob.sell_orders.keys())
            mid_price = (best_bid + best_ask) / 2
            for hc_price in self.t0_rock_prices:
                if mid_price != hc_price:
                    self.day += 1
                    break
                logger.print(f'DAY: {self.day}')

        self.T = ((8-self.day)*1000000-state.timestamp)/(self.DAYS_IN_YEAR*1000000)
        rock_vwap = self.get_current_vwap(state, 'VOLCANIC_ROCK')

        voucher_ask_greeks = {}
        voucher_bid_greeks = {}
        top_two_vouchers = ['VOLCANIC_ROCK_VOUCHER_9500', 'VOLCANIC_ROCK_VOUCHER_9750']
        rock_orders = []

        # Get rock prices with safety checks
        rock_bid, rock_bid_vol = max(rock_ob.buy_orders.items(), key=lambda x: x[0])
        rock_ask, rock_ask_vol = min(rock_ob.sell_orders.items(), key=lambda x: x[0])

        for voucher in top_two_vouchers:
            voucher_od = state.order_depths[voucher]

            # Skip if voucher order book is empty
            if not voucher_od.buy_orders or not voucher_od.sell_orders:
                continue

            try:
                voucher_bid, voucher_bid_vol = max(voucher_od.buy_orders.items(), key=lambda x: x[0])
                voucher_ask, voucher_ask_vol = min(voucher_od.sell_orders.items(), key=lambda x: x[0])
            except ValueError:  # Handle empty order books
                continue

            strike = self.voucher_strikes[voucher]

            # Calculate intrinsic value using rock's bid price
            intrinsic_value = max(0, rock_bid - strike)

            # Check for arbitrage opportunity - option ask price below intrinsic value
            if voucher_ask < intrinsic_value:
                current_position = state.position.get(voucher, 0)
                max_buy = min(-voucher_ask_vol, self.product_max_positions[voucher] - current_position)
                if max_buy > 0:
                    result[voucher] = [Order(voucher, voucher_ask, max_buy)]
                    # Add corresponding hedge in the underlying
                continue  # Skip other checks if we found arbitrage

            bid_greeks = self.get_black_scholes_greeks_iv(rock_vwap, strike, state.timestamp, voucher_bid)
            ask_greeks = self.get_black_scholes_greeks_iv(rock_vwap, strike, state.timestamp, voucher_ask)

            if not bid_greeks or not ask_greeks:
                continue

            bid_vol = bid_greeks['implied_volatility']
            ask_vol = ask_greeks['implied_volatility']
            moneyness = np.log(rock_vwap/strike)/np.sqrt(self.T + 1e-10)
            estimated_bid_vol = self.estimate_iv(moneyness, 'bid')
            estimated_ask_vol = self.estimate_iv(moneyness, 'ask')

            # Add safety check for None values before comparison
            if bid_vol is not None and ask_vol is not None and estimated_ask_vol is not None:
                if ask_vol < 0.05:
                    result[voucher] = [Order(voucher, voucher_ask, voucher_ask_vol)]
                    ask_delta = ask_greeks['delta']
                # Sell when bid IV is above ask prediction
                elif bid_vol > estimated_ask_vol:
                    current_position = state.position.get(voucher, 0)
                    max_sell = min(current_position + self.product_max_positions[voucher], voucher_bid_vol)
                    result[voucher] = [Order(voucher, voucher_bid, max_sell)]
                    bid_delta = bid_greeks['delta']

        # Calculate total delta exposure from all voucher positions based ONLY on current mid delta
        total_delta_exposure = 0
        for voucher in top_two_vouchers:
            # Calculate mid price for Greeks regardless of current position
            voucher_od = state.order_depths[voucher]
            if voucher_od.buy_orders and voucher_od.sell_orders:
                voucher_bid = max(voucher_od.buy_orders.keys())
                voucher_ask = min(voucher_od.sell_orders.keys())
                mid_price = (voucher_bid + voucher_ask) / 2
                strike = self.voucher_strikes[voucher]
                mid_greeks = self.get_black_scholes_greeks_iv(rock_vwap, strike, state.timestamp, mid_price)
                if mid_greeks:
                    current_position = state.position.get(voucher, 0)
                    total_delta_exposure += current_position * mid_greeks['delta']

        # Hedge the total delta exposure
        if total_delta_exposure != 0:
            current_rock_position = state.position.get('VOLCANIC_ROCK', 0)
            hedge_amount = -int(total_delta_exposure) - current_rock_position
            if hedge_amount > 0:
                rock_orders.append(Order('VOLCANIC_ROCK', rock_ask, hedge_amount))
            elif hedge_amount < 0:
                rock_orders.append(Order('VOLCANIC_ROCK', rock_bid, hedge_amount))

        if rock_orders:
            result['VOLCANIC_ROCK'] = rock_orders

        logger.flush(state, result, 0, jsonpickle.encode(traderData))
        return result, 0, jsonpickle.encode(traderData)

