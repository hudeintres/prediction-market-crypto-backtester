"""
Backtesting Engine

Main engine for running backtests on arbitrage strategies.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
import os

from ..strategy.arbitrage import VanillaBinaryArbitrage, ArbitrageOpportunity, TradeResult
from ..strategy.portfolio import Portfolio, PositionType


@dataclass
class BacktestConfig:
    """Configuration for backtesting."""
    start_date: str
    end_date: str
    initial_capital: float = 10000.0
    transaction_fee: float = 0.0003  # 3 bps
    min_combined_probability: float = 0.0
    trade_at_time: str = "08:00"  # UTC, Deribit expiry time
    hold_to_expiry: bool = True  # Paper strategy holds to expiry
    max_positions_per_expiry: int = 1  # One position per expiry date
    verbose: bool = True


@dataclass
class BacktestResult:
    """Container for backtesting results."""
    config: BacktestConfig
    
    # Trade-level results
    trades: List[TradeResult]
    opportunities_found: int
    
    # Portfolio-level results
    portfolio_metrics: Dict[str, float]
    
    # Time series
    daily_values: pd.DataFrame
    
    # Summary
    total_return_pct: float
    annualized_return_pct: float
    max_drawdown_pct: float
    sharpe_ratio: float
    win_rate: float
    average_return_per_trade: float


class BacktestEngine:
    """
    Main backtesting engine for the vanilla-binary arbitrage strategy.
    
    Implements the strategy from the paper:
    - Daily check at 08:00 UTC for arbitrage opportunities
    - Hold positions until expiry
    - Calculate returns at expiration
    """
    
    def __init__(self, config: BacktestConfig):
        """
        Initialize the backtest engine.
        
        Args:
            config: Backtest configuration
        """
        self.config = config
        self.strategy = VanillaBinaryArbitrage(
            transaction_fee=config.transaction_fee,
            min_combined_probability=config.min_combined_probability,
            capital=config.initial_capital
        )
        self.portfolio = Portfolio(
            name="Vanilla-Binary Arbitrage",
            initial_capital=config.initial_capital
        )
        
        # Track state
        self.current_date: Optional[datetime] = None
        self.spot_prices: Optional[pd.DataFrame] = None
        self.vanilla_options: Optional[pd.DataFrame] = None
        self.binary_options: Optional[pd.DataFrame] = None
        
        # Results
        self.all_opportunities: List[ArbitrageOpportunity] = []
        self.active_trades: Dict[str, ArbitrageOpportunity] = {}  # expiry -> opportunity
        
    def load_data(
        self,
        spot_prices: pd.DataFrame,
        vanilla_options: pd.DataFrame,
        binary_options: pd.DataFrame
    ):
        """
        Load market data for backtesting.
        
        Args:
            spot_prices: BTC spot price history
            vanilla_options: Vanilla options chain
            binary_options: Binary options data
        """
        self.spot_prices = spot_prices.copy()
        self.spot_prices['date'] = pd.to_datetime(self.spot_prices['date'])
        
        self.vanilla_options = vanilla_options.copy()
        self.vanilla_options['date'] = pd.to_datetime(self.vanilla_options['date'])
        self.vanilla_options['expiry'] = pd.to_datetime(self.vanilla_options['expiry'])
        
        self.binary_options = binary_options.copy()
        self.binary_options['date'] = pd.to_datetime(self.binary_options['date'])
        self.binary_options['expiry'] = pd.to_datetime(self.binary_options['expiry'])
        
        if self.config.verbose:
            print(f"Loaded data:")
            print(f"  - Spot prices: {len(self.spot_prices)} days")
            print(f"  - Vanilla options: {len(self.vanilla_options)} records")
            print(f"  - Binary options: {len(self.binary_options)} records")
    
    def get_spot_price(self, date: datetime) -> Optional[float]:
        """Get spot price for a specific date."""
        date_filter = self.spot_prices['date'].dt.date == date.date()
        if date_filter.any():
            return self.spot_prices.loc[date_filter, 'close'].iloc[0]
        return None
    
    def get_options_for_date(self, date: datetime) -> tuple:
        """Get vanilla and binary options available on a specific date."""
        vanilla = self.vanilla_options[
            self.vanilla_options['date'].dt.date == date.date()
        ]
        binary = self.binary_options[
            self.binary_options['date'].dt.date == date.date()
        ]
        return vanilla, binary
    
    def run(self) -> BacktestResult:
        """
        Run the backtest.
        
        Returns:
            BacktestResult with all performance metrics
        """
        if self.spot_prices is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        start = pd.to_datetime(self.config.start_date)
        end = pd.to_datetime(self.config.end_date)
        
        # Filter to backtest period
        date_range = pd.date_range(start=start, end=end, freq='D')
        
        if self.config.verbose:
            print(f"\nRunning backtest from {start.date()} to {end.date()}")
            print(f"Total days: {len(date_range)}\n")
        
        trade_count = 0
        
        for date in date_range:
            self.current_date = date
            spot_price = self.get_spot_price(date)
            
            if spot_price is None:
                continue
            
            # 1. Close any expiring positions
            closed = self._process_expiring_positions(date, spot_price)
            trade_count += len(closed)
            
            # 2. Look for new arbitrage opportunities
            vanilla_opts, binary_opts = self.get_options_for_date(date)
            
            if not vanilla_opts.empty and not binary_opts.empty:
                opportunities = self.strategy.find_opportunities(
                    date=date,
                    spot_price=spot_price,
                    vanilla_options=vanilla_opts,
                    binary_options=binary_opts
                )
                
                # 3. Enter new positions for valid opportunities
                for opp in opportunities:
                    expiry_key = pd.to_datetime(opp.expiry).strftime('%Y%m%d')
                    
                    # Check if we already have a position for this expiry
                    if expiry_key not in self.active_trades:
                        self._enter_trade(opp)
                        self.all_opportunities.append(opp)
                        self.active_trades[expiry_key] = opp
                        
                        if self.config.verbose:
                            print(f"[{date.date()}] NEW TRADE:")
                            print(f"  Vanilla {opp.vanilla_type} @ {opp.vanilla_strike}")
                            print(f"  Binary {opp.binary_type} @ {opp.binary_strike}")
                            print(f"  Spot: ${spot_price:,.0f}, Expiry: {pd.to_datetime(opp.expiry).date()}")
                            print(f"  Combined ITM Prob: {opp.combined_itm_probability:.2%}")
                            print()
            
            # 4. Record daily portfolio value
            self.portfolio.record_value(date, spot_price)
        
        # Compile results
        return self._compile_results()
    
    def _process_expiring_positions(
        self,
        date: datetime,
        spot_price: float
    ) -> List[Dict]:
        """Process and close expiring positions."""
        closed_results = []
        
        # Find expiring trades
        expiring_keys = []
        for expiry_key, opp in self.active_trades.items():
            if pd.to_datetime(opp.expiry).date() == date.date():
                expiring_keys.append(expiry_key)
        
        for expiry_key in expiring_keys:
            opp = self.active_trades[expiry_key]
            
            # Execute trade at expiry
            result = self.strategy.execute_trade(opp, spot_price)
            
            if self.config.verbose:
                print(f"[{date.date()}] TRADE EXPIRED:")
                print(f"  Spot at expiry: ${spot_price:,.0f}")
                print(f"  Vanilla payoff: ${result.vanilla_payoff:,.2f}")
                print(f"  Binary payoff: ${result.binary_payoff:,.2f}")
                print(f"  Total payoff: ${result.total_payoff:,.2f}")
                print(f"  Return: {result.net_return_pct:.2f}%")
                print()
            
            closed_results.append(result)
            del self.active_trades[expiry_key]
        
        # Also process portfolio positions
        portfolio_closed = self.portfolio.close_expiring_positions(date, spot_price)
        
        return closed_results
    
    def _enter_trade(self, opportunity: ArbitrageOpportunity):
        """Enter a new arbitrage trade."""
        trade_id = f"arb_{pd.to_datetime(opportunity.expiry).strftime('%Y%m%d')}_{opportunity.vanilla_strike}"
        
        # Add vanilla position
        if opportunity.vanilla_type == 'call':
            pos_type = PositionType.VANILLA_CALL
        else:
            pos_type = PositionType.VANILLA_PUT
        
        self.portfolio.add_position(
            position_type=pos_type,
            entry_date=opportunity.date,
            expiry_date=opportunity.expiry,
            strike=opportunity.vanilla_strike,
            quantity=opportunity.vanilla_quantity,
            entry_price=opportunity.vanilla_price_usd,
            trade_id=trade_id
        )
        
        # Add binary position
        if opportunity.binary_type == 'yes':
            pos_type = PositionType.BINARY_YES
        else:
            pos_type = PositionType.BINARY_NO
        
        self.portfolio.add_position(
            position_type=pos_type,
            entry_date=opportunity.date,
            expiry_date=opportunity.expiry,
            strike=opportunity.binary_strike,
            quantity=opportunity.binary_quantity,
            entry_price=opportunity.binary_price,
            trade_id=trade_id
        )
    
    def _compile_results(self) -> BacktestResult:
        """Compile all results into BacktestResult object."""
        trade_summary = self.strategy.get_trade_summary()
        portfolio_metrics = self.portfolio.get_performance_metrics()
        
        # Create daily values DataFrame
        daily_df = pd.DataFrame(self.portfolio.value_history)
        
        # Calculate key metrics
        if self.strategy.trades:
            returns = [t.net_return_pct for t in self.strategy.trades]
            winning = [r for r in returns if r > 0]
            win_rate = len(winning) / len(returns) * 100 if returns else 0
            avg_return = np.mean(returns) if returns else 0
        else:
            win_rate = 0
            avg_return = 0
        
        result = BacktestResult(
            config=self.config,
            trades=self.strategy.trades,
            opportunities_found=len(self.all_opportunities),
            portfolio_metrics=portfolio_metrics,
            daily_values=daily_df,
            total_return_pct=portfolio_metrics.get('total_return_pct', 0),
            annualized_return_pct=portfolio_metrics.get('annualized_return_pct', 0),
            max_drawdown_pct=portfolio_metrics.get('max_drawdown_pct', 0),
            sharpe_ratio=portfolio_metrics.get('sharpe_ratio', 0),
            win_rate=win_rate,
            average_return_per_trade=avg_return
        )
        
        return result
    
    def print_summary(self, result: BacktestResult):
        """Print formatted summary of backtest results."""
        print("\n" + "="*60)
        print("BACKTEST RESULTS: Vanilla-Binary Arbitrage Strategy")
        print("="*60)
        
        print(f"\nPeriod: {self.config.start_date} to {self.config.end_date}")
        print(f"Initial Capital: ${self.config.initial_capital:,.2f}")
        
        print("\n--- Trade Summary ---")
        print(f"Total Opportunities Found: {result.opportunities_found}")
        print(f"Total Trades Executed: {len(result.trades)}")
        print(f"Win Rate: {result.win_rate:.1f}%")
        print(f"Average Return Per Trade: {result.average_return_per_trade:.2f}%")
        
        if result.trades:
            returns = [t.net_return_pct for t in result.trades]
            print(f"Best Trade: {max(returns):.2f}%")
            print(f"Worst Trade: {min(returns):.2f}%")
            losing = [r for r in returns if r < 0]
            print(f"Losing Trades: {len(losing)}")
        
        print("\n--- Portfolio Performance ---")
        print(f"Final NAV: ${result.portfolio_metrics.get('final_nav', self.config.initial_capital):,.2f}")
        print(f"Total Return: {result.total_return_pct:.2f}%")
        print(f"Annualized Return: {result.annualized_return_pct:.2f}%")
        print(f"Annualized Volatility: {result.portfolio_metrics.get('annualized_volatility', 0):.2f}%")
        print(f"Downside Volatility: {result.portfolio_metrics.get('downside_volatility', 0):.2f}%")
        print(f"Sharpe Ratio: {result.sharpe_ratio:.2f}")
        print(f"Max Drawdown: {result.max_drawdown_pct:.2f}%")
        print(f"Total PnL: ${result.portfolio_metrics.get('total_pnl', 0):,.2f}")
        
        if result.trades:
            trade_summary = self.strategy.get_trade_summary()
            print("\n--- Paper Metrics (Per-Trade Basis) ---")
            print(f"Skewness: {trade_summary.get('skewness', 0):.2f}")
            print(f"Kurtosis: {trade_summary.get('kurtosis', 0):.2f}")
            print(f"Both ITM Rate: {trade_summary.get('both_itm_rate', 0):.1f}%")
        
        print("\n" + "="*60)
