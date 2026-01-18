"""
Black-Scholes and Black-76 Option Pricing Models

Implements the standard Black-Scholes model and its variant Black-76
for pricing vanilla options on cryptocurrencies.
"""

import numpy as np
from scipy.stats import norm
from typing import Tuple, Dict, Optional
from dataclasses import dataclass


@dataclass
class OptionPriceResult:
    """Container for option pricing results."""
    price: float
    delta: float
    gamma: float
    vega: float
    theta: float
    rho: float
    iv: float
    intrinsic: float
    time_value: float


class BlackScholesModel:
    """
    Standard Black-Scholes model for European options pricing.
    
    Assumes constant volatility and risk-free rate, lognormal price distribution.
    """
    
    def __init__(self, r: float = 0.0):
        """
        Initialize the Black-Scholes model.
        
        Args:
            r: Risk-free interest rate (annualized)
        """
        self.r = r
    
    def d1(self, S: float, K: float, T: float, sigma: float) -> float:
        """Calculate d1 parameter."""
        if T <= 0 or sigma <= 0:
            return np.inf if S > K else -np.inf
        return (np.log(S / K) + (self.r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    
    def d2(self, S: float, K: float, T: float, sigma: float) -> float:
        """Calculate d2 parameter."""
        return self.d1(S, K, T, sigma) - sigma * np.sqrt(T)
    
    def call_price(self, S: float, K: float, T: float, sigma: float) -> float:
        """
        Calculate European call option price.
        
        Args:
            S: Spot price
            K: Strike price
            T: Time to expiry in years
            sigma: Implied volatility (annualized)
            
        Returns:
            Call option price
        """
        if T <= 0:
            return max(S - K, 0)
        
        d1 = self.d1(S, K, T, sigma)
        d2 = self.d2(S, K, T, sigma)
        
        return S * norm.cdf(d1) - K * np.exp(-self.r * T) * norm.cdf(d2)
    
    def put_price(self, S: float, K: float, T: float, sigma: float) -> float:
        """
        Calculate European put option price.
        
        Args:
            S: Spot price
            K: Strike price
            T: Time to expiry in years
            sigma: Implied volatility (annualized)
            
        Returns:
            Put option price
        """
        if T <= 0:
            return max(K - S, 0)
        
        d1 = self.d1(S, K, T, sigma)
        d2 = self.d2(S, K, T, sigma)
        
        return K * np.exp(-self.r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    
    def call_delta(self, S: float, K: float, T: float, sigma: float) -> float:
        """
        Calculate call option delta.
        
        Delta represents the rate of change of option price with respect to
        the underlying price. It's also used as a proxy for probability of
        expiring in-the-money.
        
        Returns:
            Delta value between 0 and 1
        """
        if T <= 0:
            return 1.0 if S > K else 0.0
        
        d1 = self.d1(S, K, T, sigma)
        return norm.cdf(d1)
    
    def put_delta(self, S: float, K: float, T: float, sigma: float) -> float:
        """
        Calculate put option delta.
        
        Returns:
            Delta value between -1 and 0
        """
        if T <= 0:
            return -1.0 if S < K else 0.0
        
        d1 = self.d1(S, K, T, sigma)
        return norm.cdf(d1) - 1
    
    def gamma(self, S: float, K: float, T: float, sigma: float) -> float:
        """
        Calculate option gamma (same for calls and puts).
        
        Gamma represents the rate of change of delta.
        
        Returns:
            Gamma value
        """
        if T <= 0 or S <= 0:
            return 0.0
        
        d1 = self.d1(S, K, T, sigma)
        return norm.pdf(d1) / (S * sigma * np.sqrt(T))
    
    def vega(self, S: float, K: float, T: float, sigma: float) -> float:
        """
        Calculate option vega (same for calls and puts).
        
        Vega represents sensitivity to volatility changes.
        
        Returns:
            Vega value (per 1% vol change)
        """
        if T <= 0:
            return 0.0
        
        d1 = self.d1(S, K, T, sigma)
        return S * norm.pdf(d1) * np.sqrt(T) / 100  # Per 1% change
    
    def theta(self, S: float, K: float, T: float, sigma: float, is_call: bool) -> float:
        """
        Calculate option theta.
        
        Theta represents time decay - daily loss of option value.
        
        Returns:
            Theta value (per day)
        """
        if T <= 0:
            return 0.0
        
        d1 = self.d1(S, K, T, sigma)
        d2 = self.d2(S, K, T, sigma)
        
        term1 = -S * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
        
        if is_call:
            term2 = -self.r * K * np.exp(-self.r * T) * norm.cdf(d2)
        else:
            term2 = self.r * K * np.exp(-self.r * T) * norm.cdf(-d2)
        
        return (term1 + term2) / 365  # Per day
    
    def implied_volatility(
        self,
        option_price: float,
        S: float,
        K: float,
        T: float,
        is_call: bool,
        tolerance: float = 1e-6,
        max_iterations: int = 100
    ) -> float:
        """
        Calculate implied volatility using Newton-Raphson method.
        
        Args:
            option_price: Market price of the option
            S: Spot price
            K: Strike price
            T: Time to expiry in years
            is_call: True for call, False for put
            tolerance: Convergence tolerance
            max_iterations: Maximum iterations
            
        Returns:
            Implied volatility
        """
        if T <= 0:
            return 0.0
        
        # Initial guess based on simple heuristic
        sigma = 0.5
        
        for i in range(max_iterations):
            if is_call:
                price = self.call_price(S, K, T, sigma)
            else:
                price = self.put_price(S, K, T, sigma)
            
            diff = price - option_price
            
            if abs(diff) < tolerance:
                return sigma
            
            vega = self.vega(S, K, T, sigma) * 100  # Undo the /100
            
            if vega < 1e-10:
                break
            
            sigma = sigma - diff / vega
            sigma = max(0.01, min(5.0, sigma))
        
        return sigma
    
    def full_pricing(
        self,
        S: float,
        K: float,
        T: float,
        sigma: float,
        is_call: bool
    ) -> OptionPriceResult:
        """
        Calculate full option pricing with all Greeks.
        
        Args:
            S: Spot price
            K: Strike price
            T: Time to expiry
            sigma: Implied volatility
            is_call: True for call, False for put
            
        Returns:
            OptionPriceResult with all values
        """
        if is_call:
            price = self.call_price(S, K, T, sigma)
            delta = self.call_delta(S, K, T, sigma)
            intrinsic = max(S - K, 0)
        else:
            price = self.put_price(S, K, T, sigma)
            delta = self.put_delta(S, K, T, sigma)
            intrinsic = max(K - S, 0)
        
        return OptionPriceResult(
            price=price,
            delta=delta,
            gamma=self.gamma(S, K, T, sigma),
            vega=self.vega(S, K, T, sigma),
            theta=self.theta(S, K, T, sigma, is_call),
            rho=0.0,  # Simplified
            iv=sigma,
            intrinsic=intrinsic,
            time_value=price - intrinsic
        )


class Black76Model:
    """
    Black-76 model for options on futures/forwards.
    
    This is the standard model used by Deribit and other crypto
    options exchanges for pricing inverse options.
    
    Key difference from Black-Scholes: uses forward price F instead
    of spot price S, and assumes the underlying is a futures contract.
    """
    
    def __init__(self, r: float = 0.0):
        """
        Initialize Black-76 model.
        
        Args:
            r: Risk-free rate (Deribit uses 0)
        """
        self.r = r
    
    def d1(self, F: float, K: float, T: float, sigma: float) -> float:
        """Calculate d1 parameter using forward price."""
        if T <= 0 or sigma <= 0:
            return np.inf if F > K else -np.inf
        return (np.log(F / K) + 0.5 * sigma**2 * T) / (sigma * np.sqrt(T))
    
    def d2(self, F: float, K: float, T: float, sigma: float) -> float:
        """Calculate d2 parameter."""
        return self.d1(F, K, T, sigma) - sigma * np.sqrt(T)
    
    def call_price(self, F: float, K: float, T: float, sigma: float) -> float:
        """
        Calculate European call option price on a futures contract.
        
        Args:
            F: Forward/Futures price
            K: Strike price
            T: Time to expiry in years
            sigma: Implied volatility
            
        Returns:
            Call option price in USD
        """
        if T <= 0:
            return max(F - K, 0)
        
        d1 = self.d1(F, K, T, sigma)
        d2 = self.d2(F, K, T, sigma)
        
        discount = np.exp(-self.r * T)
        return discount * (F * norm.cdf(d1) - K * norm.cdf(d2))
    
    def put_price(self, F: float, K: float, T: float, sigma: float) -> float:
        """
        Calculate European put option price on a futures contract.
        
        Args:
            F: Forward/Futures price
            K: Strike price
            T: Time to expiry in years
            sigma: Implied volatility
            
        Returns:
            Put option price in USD
        """
        if T <= 0:
            return max(K - F, 0)
        
        d1 = self.d1(F, K, T, sigma)
        d2 = self.d2(F, K, T, sigma)
        
        discount = np.exp(-self.r * T)
        return discount * (K * norm.cdf(-d2) - F * norm.cdf(-d1))
    
    def call_delta(self, F: float, K: float, T: float, sigma: float) -> float:
        """
        Calculate call option delta (Black-76).
        
        Per the paper, delta is used as approximation for probability
        of expiring in-the-money for vanilla options.
        
        Returns:
            Delta value
        """
        if T <= 0:
            return 1.0 if F > K else 0.0
        
        d1 = self.d1(F, K, T, sigma)
        discount = np.exp(-self.r * T)
        return discount * norm.cdf(d1)
    
    def put_delta(self, F: float, K: float, T: float, sigma: float) -> float:
        """
        Calculate put option delta (Black-76).
        
        Returns:
            Delta value
        """
        if T <= 0:
            return -1.0 if F < K else 0.0
        
        d1 = self.d1(F, K, T, sigma)
        discount = np.exp(-self.r * T)
        return -discount * norm.cdf(-d1)
    
    def vega(self, F: float, K: float, T: float, sigma: float) -> float:
        """Calculate option vega (Black-76)."""
        if T <= 0:
            return 0.0
        
        d1 = self.d1(F, K, T, sigma)
        discount = np.exp(-self.r * T)
        return F * discount * norm.pdf(d1) * np.sqrt(T)
    
    def implied_volatility(
        self,
        option_price: float,
        F: float,
        K: float,
        T: float,
        is_call: bool,
        tolerance: float = 1e-6,
        max_iterations: int = 100
    ) -> float:
        """
        Calculate implied volatility using Newton-Raphson method.
        
        Args:
            option_price: Market price of the option
            F: Forward price
            K: Strike price
            T: Time to expiry
            is_call: True for call, False for put
            tolerance: Convergence tolerance
            max_iterations: Maximum iterations
            
        Returns:
            Implied volatility
        """
        if T <= 0:
            return 0.0
        
        sigma = 0.5
        
        for _ in range(max_iterations):
            if is_call:
                price = self.call_price(F, K, T, sigma)
            else:
                price = self.put_price(F, K, T, sigma)
            
            diff = price - option_price
            
            if abs(diff) < tolerance:
                return sigma
            
            v = self.vega(F, K, T, sigma)
            
            if v < 1e-10:
                break
            
            sigma = sigma - diff / v
            sigma = max(0.01, min(5.0, sigma))
        
        return sigma
    
    def inverse_option_price_btc(
        self,
        F: float,
        K: float,
        T: float,
        sigma: float,
        is_call: bool
    ) -> float:
        """
        Calculate inverse option price in BTC terms.
        
        Deribit options are settled in the underlying cryptocurrency,
        creating an inverse relationship. The payoff in BTC is:
        max(S_T - K, 0) / S_T for calls
        max(K - S_T, 0) / S_T for puts
        
        Args:
            F: Forward price
            K: Strike price
            T: Time to expiry
            sigma: Implied volatility
            is_call: True for call, False for put
            
        Returns:
            Option price in BTC
        """
        usd_price = self.call_price(F, K, T, sigma) if is_call else self.put_price(F, K, T, sigma)
        return usd_price / F
