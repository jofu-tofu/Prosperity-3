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

        # Timestep counter
        self.timestep_counter = 0

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

    # Calculate moneyness for an option
    def calculate_moneyness(self, strike, underlying_price, T):
        return math.log(underlying_price / strike) / math.sqrt(T + 1e-10)

    # Find eligible options (ATM or OTM) sorted by theta
    def find_eligible_options(self, state: TradingState, underlying_price):
        """
        Finds options that are ATM or OTM (strike >= underlying_price) and sorts them by theta.
        
        Parameters:
            state (TradingState): Current trading state
            underlying_price (float): Current price of the underlying asset
            
        Returns:
            list: List of eligible options sorted by theta (highest first)
        """
        logger.print("Finding eligible options (ATM or OTM)...")
        
        eligible_options = []
        
        for voucher in self.voucher_names:
            if voucher not in state.order_depths:
                continue
                
            strike = self.voucher_strikes[voucher]
            
            # Only consider ATM or OTM options (strike >= underlying_price)
            if strike < underlying_price:
                logger.print(f"Skipping {voucher} - ITM (strike {strike} < underlying {underlying_price})")
                continue
                
            best_bid, best_ask = self.get_best_prices(state, voucher)
            
            if not best_bid or not best_ask:
                continue
                
            # Calculate mid price for Greeks
            mid_price = (best_bid + best_ask) / 2
            
            # Calculate moneyness for IV estimation
            moneyness = self.calculate_moneyness(strike, underlying_price, self.T)
            
            # Get estimated IVs
            estimated_bid_iv = self.estimate_iv(moneyness, 'bid')
            estimated_ask_iv = self.estimate_iv(moneyness, 'ask')
            
            # Get actual IVs and Greeks
            bid_greeks = self.get_black_scholes_greeks_iv(underlying_price, strike, state.timestamp, best_bid)
            ask_greeks = self.get_black_scholes_greeks_iv(underlying_price, strike, state.timestamp, best_ask)
            
            if not bid_greeks or not ask_greeks:
                continue
                
            # Get actual IVs
            actual_bid_iv = bid_greeks.get('implied_volatility')
            actual_ask_iv = ask_greeks.get('implied_volatility')
            
            if actual_bid_iv is None or actual_ask_iv is None:
                continue
                
            # Get theta (negative value, so we take absolute value for comparison)
            theta = abs(bid_greeks.get('theta', 0))
            
            eligible_options.append({
                'voucher': voucher,
                'strike': strike,
                'best_bid': best_bid,
                'best_ask': best_ask,
                'moneyness': moneyness,
                'estimated_bid_iv': estimated_bid_iv,
                'estimated_ask_iv': estimated_ask_iv,
                'actual_bid_iv': actual_bid_iv,
                'actual_ask_iv': actual_ask_iv,
                'theta': theta,
                'bid_greeks': bid_greeks,
                'ask_greeks': ask_greeks
            })
        
        if not eligible_options:
            logger.print("No eligible options found")
            return []
            
        # Sort by theta (highest first)
        eligible_options.sort(key=lambda x: x['theta'], reverse=True)
        
        for option in eligible_options:
            logger.print(f"{option['voucher']}: Strike={option['strike']}, Theta={option['theta']:.6f}, "
                        f"Bid IV={option['actual_bid_iv']:.4f}, Est Ask IV={option['estimated_ask_iv']:.4f}, "
                        f"Ask IV={option['actual_ask_iv']:.4f}, Est Bid IV={option['estimated_bid_iv']:.4f}")
            
        return eligible_options

    # Execute IV-based trading strategy
    def execute_iv_strategy(self, state: TradingState, underlying_price, result):
        """
        Executes the IV-based trading strategy:
        - Sell when bid IV > estimated ask IV
        - Buy when ask IV < estimated bid IV
        - Only trade ATM or OTM options
        - Prioritize options with highest theta
        
        Parameters:
            state (TradingState): Current trading state
            underlying_price (float): Current price of the underlying asset
            result (dict): Dictionary to store orders
        """
        logger.print("Executing IV-based trading strategy...")
        
        # Find eligible options
        eligible_options = self.find_eligible_options(state, underlying_price)
        
        if not eligible_options:
            return
        
        # Process each eligible option in order of highest theta
        for option in eligible_options:
            voucher = option['voucher']
            best_bid = option['best_bid']
            best_ask = option['best_ask']
            actual_bid_iv = option['actual_bid_iv']
            actual_ask_iv = option['actual_ask_iv']
            estimated_bid_iv = option['estimated_bid_iv']
            estimated_ask_iv = option['estimated_ask_iv']
            
            # Get current position
            current_position = state.position.get(voucher, 0)
            max_position = self.product_max_positions[voucher]
            
            # Check for sell opportunity (bid IV > estimated ask IV)
            if actual_bid_iv > estimated_ask_iv:
                logger.print(f"Sell opportunity in {voucher}: Bid IV {actual_bid_iv:.4f} > Est Ask IV {estimated_ask_iv:.4f}")
                
                # Calculate how much more we can short
                remaining_short_capacity = -max_position - current_position  # For short positions (negative), this gives us how much more we can short
                
                if remaining_short_capacity <= 0:
                    logger.print(f"Cannot short {voucher} - Position limit reached (current: {current_position}, max: {-max_position})")
                    continue
                
                # Calculate available volume to short
                available_volume = state.order_depths[voucher].buy_orders[best_bid] if best_bid in state.order_depths[voucher].buy_orders else 0
                
                # Determine how much to short (positive number)
                short_volume = min(available_volume, remaining_short_capacity)
                
                if short_volume <= 0:
                    logger.print(f"Cannot short {voucher} - No volume available")
                    continue
                
                logger.print(f"Shorting {short_volume} {voucher} at {best_bid}")
                
                # Add short order to result (negative quantity means sell/short)
                if voucher not in result:
                    result[voucher] = []
                result[voucher].append(Order(voucher, best_bid, -short_volume))
            
            # Check for buy opportunity (ask IV < estimated bid IV)
            elif actual_ask_iv < estimated_bid_iv:
                logger.print(f"Buy opportunity in {voucher}: Ask IV {actual_ask_iv:.4f} < Est Bid IV {estimated_bid_iv:.4f}")
                
                # Calculate how much more we can buy
                remaining_long_capacity = max_position - current_position
                
                if remaining_long_capacity <= 0:
                    logger.print(f"Cannot buy {voucher} - Position limit reached (current: {current_position}, max: {max_position})")
                    continue
                
                # Calculate available volume to buy
                available_volume = abs(state.order_depths[voucher].sell_orders[best_ask]) if best_ask in state.order_depths[voucher].sell_orders else 0
                
                # Determine how much to buy
                buy_volume = min(available_volume, remaining_long_capacity)
                
                if buy_volume <= 0:
                    logger.print(f"Cannot buy {voucher} - No volume available")
                    continue
                
                logger.print(f"Buying {buy_volume} {voucher} at {best_ask}")
                
                # Add buy order to result (positive quantity means buy)
                if voucher not in result:
                    result[voucher] = []
                result[voucher].append(Order(voucher, best_ask, buy_volume))
            
            else:
                logger.print(f"No trading opportunity in {voucher}: "
                           f"Bid IV {actual_bid_iv:.4f} <= Est Ask IV {estimated_ask_iv:.4f}, "
                           f"Ask IV {actual_ask_iv:.4f} >= Est Bid IV {estimated_bid_iv:.4f}")

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
        
        # Increment timestep counter
        self.timestep_counter += 1
        
        # Execute IV-based trading strategy
        self.execute_iv_strategy(state, rock_vwap, result)
        
        # Update trader data
        trader_data = {
            'current_positions': self.current_positions,
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
