"""
Performance Metrics Module

Calculates various performance metrics following the paper's methodology.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from scipy import stats


@dataclass
class PerformanceMetrics:
    """
    Calculator for strategy performance metrics.
    
    Following the paper's methodology:
    - Annualized metrics based on active days (Lucca and Moench, 2015)
    - Focus on negative volatility for downside risk
    - Per-trade analysis for arbitrage-style strategies
    """
    
    @staticmethod
    def calculate_annualized_metrics(
        returns: List[float],
        active_days: int,
        risk_free_rate: float = 0.0  # Paper assumes ~13.5% (1350 bps)
    ) -> Dict[str, float]:
        """
        Calculate annualized metrics following the paper's methodology.
        
        From the paper:
        "Given that our approach is grounded in a no-arbitrage condition
        and assumes a relatively low frequency of occurrence, we follow
        the annualization method of Lucca and Moench (2015)"
        
        μa = μ̄ × Na (average daily return × active days)
        σa = σ × √Na (volatility × sqrt of active days)
        
        Args:
            returns: List of trade returns (percentage)
            active_days: Number of days strategy was active
            risk_free_rate: Annualized risk-free rate (default per paper)
            
        Returns:
            Dictionary with annualized metrics
        """
        if not returns or active_days == 0:
            return {
                'annualized_return': 0,
                'annualized_volatility': 0,
                'sharpe_ratio': 0,
                'sortino_ratio': 0
            }
        
        returns_array = np.array(returns)
        
        # Average daily return (when active)
        avg_return = np.mean(returns_array)
        
        # Annualize using active days
        annualized_return = avg_return * active_days
        
        # Volatility (standard deviation of returns)
        volatility = np.std(returns_array)
        annualized_volatility = volatility * np.sqrt(active_days)
        
        # Negative volatility (downside only)
        negative_returns = returns_array[returns_array < 0]
        if len(negative_returns) > 0:
            negative_volatility = np.std(negative_returns) * np.sqrt(active_days)
        else:
            negative_volatility = 0
        
        # Sharpe ratio
        if annualized_volatility > 0:
            sharpe = (annualized_return - risk_free_rate) / annualized_volatility
        else:
            sharpe = 0
        
        # Sortino ratio (using negative volatility)
        if negative_volatility > 0:
            sortino = (annualized_return - risk_free_rate) / negative_volatility
        else:
            sortino = float('inf') if annualized_return > risk_free_rate else 0
        
        return {
            'annualized_return': annualized_return,
            'annualized_volatility': annualized_volatility,
            'negative_volatility': negative_volatility,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino
        }
    
    @staticmethod
    def calculate_trade_metrics(
        trade_returns: List[float]
    ) -> Dict[str, float]:
        """
        Calculate per-trade metrics as emphasized in the paper.
        
        From the paper's Table 2:
        - Average return per trade
        - Return volatility
        - Variance
        - Number of losing trades
        - Profit rate
        - Skewness
        - Kurtosis
        
        Args:
            trade_returns: List of returns per trade (percentage)
            
        Returns:
            Dictionary with per-trade metrics
        """
        if not trade_returns:
            return {
                'average_return': 0,
                'median_return': 0,
                'volatility': 0,
                'variance': 0,
                'losing_trades': 0,
                'profit_rate': 0,
                'skewness': 0,
                'kurtosis': 0,
                'min_return': 0,
                'max_return': 0
            }
        
        returns = np.array(trade_returns)
        
        # Basic statistics
        avg_return = np.mean(returns)
        median_return = np.median(returns)
        volatility = np.std(returns)
        variance = np.var(returns)
        
        # Trade classification
        losing_trades = np.sum(returns < 0)
        winning_trades = np.sum(returns > 0)
        neutral_trades = np.sum(returns == 0)
        
        profit_rate = winning_trades / len(returns) * 100 if len(returns) > 0 else 0
        
        # Higher moments
        skewness = stats.skew(returns) if len(returns) > 2 else 0
        kurtosis = stats.kurtosis(returns) if len(returns) > 3 else 0
        
        return {
            'average_return': avg_return,
            'median_return': median_return,
            'volatility': volatility,
            'variance': variance,
            'losing_trades': int(losing_trades),
            'winning_trades': int(winning_trades),
            'neutral_trades': int(neutral_trades),
            'profit_rate': profit_rate,
            'skewness': skewness,
            'kurtosis': kurtosis,
            'min_return': np.min(returns),
            'max_return': np.max(returns)
        }
    
    @staticmethod
    def calculate_drawdown(nav_series: pd.Series) -> Dict[str, float]:
        """
        Calculate drawdown metrics.
        
        Args:
            nav_series: Series of portfolio NAV values
            
        Returns:
            Dictionary with drawdown metrics
        """
        if nav_series.empty:
            return {
                'max_drawdown': 0,
                'avg_drawdown': 0,
                'max_drawdown_duration': 0
            }
        
        # Calculate running maximum
        running_max = nav_series.cummax()
        
        # Calculate drawdown
        drawdown = (nav_series - running_max) / running_max * 100
        
        # Maximum drawdown
        max_drawdown = drawdown.min()
        
        # Average drawdown
        avg_drawdown = drawdown.mean()
        
        # Maximum drawdown duration (consecutive days in drawdown)
        in_drawdown = drawdown < 0
        if in_drawdown.any():
            # Find consecutive sequences
            changes = in_drawdown.astype(int).diff().fillna(0)
            starts = changes[changes == 1].index.tolist()
            ends = changes[changes == -1].index.tolist()
            
            if len(starts) > 0:
                if len(ends) == 0 or (len(ends) > 0 and starts[-1] > ends[-1]):
                    ends.append(nav_series.index[-1])
                
                durations = []
                for i, start in enumerate(starts):
                    if i < len(ends):
                        duration = (ends[i] - start).days if hasattr(ends[i] - start, 'days') else 1
                        durations.append(duration)
                
                max_duration = max(durations) if durations else 0
            else:
                max_duration = 0
        else:
            max_duration = 0
        
        return {
            'max_drawdown': max_drawdown,
            'avg_drawdown': avg_drawdown,
            'max_drawdown_duration': max_duration
        }
    
    @staticmethod
    def decompose_returns(
        trade_results: List[Any]
    ) -> Dict[str, float]:
        """
        Decompose returns by leg (vanilla vs binary).
        
        Following the paper's Table 2, which shows decomposed returns
        for vanilla legs vs binary legs.
        
        Args:
            trade_results: List of TradeResult objects
            
        Returns:
            Dictionary with decomposed returns
        """
        if not trade_results:
            return {
                'vanilla_contribution': 0,
                'binary_contribution': 0,
                'total_return': 0
            }
        
        vanilla_payoffs = [t.vanilla_payoff for t in trade_results]
        binary_payoffs = [t.binary_payoff for t in trade_results]
        total_payoffs = [t.total_payoff for t in trade_results]
        
        return {
            'avg_vanilla_payoff': np.mean(vanilla_payoffs),
            'avg_binary_payoff': np.mean(binary_payoffs),
            'avg_total_payoff': np.mean(total_payoffs),
            'total_vanilla_pnl': sum(vanilla_payoffs),
            'total_binary_pnl': sum(binary_payoffs),
            'total_pnl': sum(total_payoffs),
            'vanilla_contribution_pct': sum(vanilla_payoffs) / sum(total_payoffs) * 100 if sum(total_payoffs) != 0 else 0,
            'binary_contribution_pct': sum(binary_payoffs) / sum(total_payoffs) * 100 if sum(total_payoffs) != 0 else 0
        }
    
    @staticmethod
    def calculate_correlation(
        returns: List[float],
        benchmark_returns: List[float]
    ) -> float:
        """
        Calculate correlation with benchmark (e.g., BTC price returns).
        
        Args:
            returns: Strategy returns
            benchmark_returns: Benchmark returns
            
        Returns:
            Correlation coefficient
        """
        if len(returns) != len(benchmark_returns) or len(returns) < 2:
            return 0.0
        
        return np.corrcoef(returns, benchmark_returns)[0, 1]
    
    @staticmethod
    def calculate_arbitrage_metrics(
        opportunities: List[Any],
        total_days: int
    ) -> Dict[str, float]:
        """
        Calculate arbitrage-specific metrics.
        
        Args:
            opportunities: List of ArbitrageOpportunity objects
            total_days: Total days in backtest period
            
        Returns:
            Dictionary with arbitrage metrics
        """
        if not opportunities or total_days == 0:
            return {
                'arbitrage_frequency': 0,
                'avg_time_to_expiry': 0,
                'avg_combined_probability': 0,
                'active_percentage': 0
            }
        
        time_to_expiries = [o.time_to_expiry for o in opportunities]
        combined_probs = [o.combined_itm_probability for o in opportunities]
        
        # Count unique active days
        active_dates = set(o.date.date() for o in opportunities)
        active_days = len(active_dates)
        
        return {
            'arbitrage_frequency': len(opportunities) / total_days * 100,
            'avg_time_to_expiry': np.mean(time_to_expiries) * 365,  # Convert to days
            'avg_combined_probability': np.mean(combined_probs) * 100,
            'active_percentage': active_days / total_days * 100
        }
