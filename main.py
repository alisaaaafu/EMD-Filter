# -*- coding: utf-8 -*-
"""
EMD-based trading system for Chinese stock index futures.

This script implements an adaptive trading system that uses Empirical Mode
Decomposition (EMD) to detect market regime changes and apply appropriate
trading strategies (trend-following or mean-reversion).

@author: Wanzhen Fu
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import math
import matplotlib.pyplot as plt
import os

from emd.decomposer import compute_noise_ratio, compute_signal_ratio
from strategies.trend_strategy import TrendStrategy
from strategies.mean_reversion_strategy import MeanReversionStrategy


class EMDTradingSystem:
    """EMD-based adaptive trading system."""
    
    def __init__(self, data_path, initial_capital=4000000, fee_rate=0.0002, 
                 contract_multiplier=100):
        """
        Initialize trading system.
        
        Args:
            data_path: Path to CSV file with market data
            initial_capital: Initial account balance
            fee_rate: Transaction fee rate
            contract_multiplier: Futures contract multiplier
        """
        self.data_path = data_path
        self.df = pd.read_csv(data_path)
        
        # Extract data
        self.timestamps = self._parse_timestamps()
        self.close = self.df.iloc[:, 4]
        
        # Trading parameters
        self.initial_capital = initial_capital
        self.fee_rate = fee_rate
        self.contract_multiplier = contract_multiplier
        
        # Initialize strategies
        self.trend_strategy = TrendStrategy(data_path)
        self.mean_reversion_strategy = MeanReversionStrategy(data_path)
        
        # Trading records
        self.returns = []
        self.market_values = [1.0]
        self.log_market_values = [0.0]
        self.trade_dates = [0]
        self.noise_ratios = []
        self.mean_noise_ratios = []
        self.continuity = []
    
    def _parse_timestamps(self):
        """Parse timestamp strings to datetime objects."""
        time_strings = np.array(self.df.iloc[:, 0])
        timestamps = [datetime.strptime(t, "%Y/%m/%d %H:%M") for t in time_strings]
        return np.array(timestamps)
    
    def get_calculation_window(self, start_idx):
        """
        Get calculation window (4.5 hours from start).
        
        Args:
            start_idx: Starting index
        
        Returns:
            Tuple of (start_idx, end_idx) or (start_idx, False) if not found
        """
        start_time = self.timestamps[start_idx]
        end_time = start_time + timedelta(hours=4.5)
        
        for i in range(start_idx, len(self.timestamps)):
            if end_time == self.timestamps[i]:
                return start_idx, i
        
        return start_idx, False
    
    def get_trading_time(self, start_idx):
        """
        Get trading entry time (1 minute after calculation window).
        
        Args:
            start_idx: Starting index
        
        Returns:
            Trading time index or False
        """
        if start_idx is False:
            return False
        
        _, end_idx = self.get_calculation_window(start_idx)
        
        if end_idx is False:
            return False
        
        end_time = self.timestamps[end_idx]
        buy_time = end_time + timedelta(seconds=60)
        
        # Check if still same trading day
        if buy_time.day != end_time.day:
            return False
        
        for i in range(end_idx, len(self.timestamps)):
            if (self.timestamps[i] == buy_time and 
                self.timestamps[i+1].day == buy_time.day):
                return i
        
        return False
    
    def get_closing_time(self, start_idx):
        """
        Get closing time (end of trading day).
        
        Args:
            start_idx: Starting index
        
        Returns:
            Closing time index or False
        """
        entry_idx = self.get_trading_time(start_idx)
        
        if entry_idx is False:
            return False
        
        for i in range(entry_idx, len(self.timestamps)):
            if self.timestamps[i].day == self.timestamps[entry_idx].day:
                if (i + 1 >= len(self.timestamps) or 
                    self.timestamps[i+1].day != self.timestamps[entry_idx].day):
                    return i
        
        return False
    
    def find_next_trading_day(self, start_idx):
        """
        Find start index of next trading day.
        
        Args:
            start_idx: Current index
        
        Returns:
            Next trading day index or False
        """
        current_day = self.timestamps[start_idx].day
        
        for i in range(start_idx, len(self.timestamps)):
            if self.timestamps[i].day != current_day:
                return i
        
        return False
    
    def run_backtest(self, initial_window_start=0, initial_window_end=5399,
                     first_trade_idx=5400):
        """
        Run backtest on historical data.
        
        Args:
            initial_window_start: Start of initial calibration window
            initial_window_end: End of initial calibration window
            first_trade_idx: First trading day index
        """
        # Initial calibration
        next_idx = first_trade_idx
        noise_window_start = initial_window_start
        noise_window_end = initial_window_end
        
        while next_idx is not False:
            start_idx, end_idx = self.get_calculation_window(next_idx)
            entry_idx = self.get_trading_time(start_idx)
            exit_idx = self.get_closing_time(start_idx)
            
            if (start_idx is not False and end_idx is not False and 
                entry_idx is not False and exit_idx is not False):
                
                # Calculate EMD ratios
                mean_noise_ratio = compute_noise_ratio(
                    self.data_path, noise_window_start, noise_window_end
                )
                current_noise_ratio = compute_signal_ratio(
                    self.data_path, start_idx, end_idx
                )
                
                entry_price = self.close[entry_idx]
                exit_price = self.close[exit_idx]
                
                # Determine trading strategy
                order = None
                
                # Trend following conditions
                if ((mean_noise_ratio - current_noise_ratio) / abs(mean_noise_ratio) > 0.25 and 
                    current_noise_ratio <= -0.5):
                    order = self.trend_strategy.generate_order(
                        start_idx + 25, start_idx + 72, end_idx
                    )
                    strategy_type = "trend"
                
                # Mean reversion conditions
                elif (current_noise_ratio >= 1.9 * mean_noise_ratio and 
                      mean_noise_ratio > 0):
                    order = self.mean_reversion_strategy.generate_order(
                        start_idx + 25, start_idx + 72, end_idx
                    )
                    strategy_type = "mean_reversion"
                
                # Execute trade if order generated
                if order is not None:
                    revenue = self._execute_trade(order, entry_price, exit_price)
                    self.returns.append(revenue)
                    
                    print(f"{self.timestamps[next_idx]} - {strategy_type} {order['type']}: "
                          f"Entry={entry_price:.2f}, Exit={exit_price:.2f}, "
                          f"MV={self.market_values[-1]:.4f}")
                else:
                    self.market_values.append(self.market_values[-1])
                    self.log_market_values.append(self.log_market_values[-1])
                    print(f"{self.timestamps[next_idx]} - No trade")
                
                self.trade_dates.append(self.df.iloc[start_idx, 0])
                self.noise_ratios.append(current_noise_ratio)
                self.mean_noise_ratios.append(mean_noise_ratio)
                
                # Track trend continuity
                continuity_flag = self._check_continuity(
                    entry_price, exit_price, 
                    self.close[start_idx + 15], self.close[end_idx]
                )
                self.continuity.append(continuity_flag)
            
            # Move to next trading day
            next_idx = self.find_next_trading_day(start_idx)
            if next_idx is not False:
                noise_window_end = next_idx - 1
                noise_window_start = noise_window_end - 5000
    
    def _execute_trade(self, order, entry_price, exit_price):
        """Execute trade and update market value."""
        num_contracts = order['num']
        
        if order['type'] == 'long':
            # Long position
            revenue = ((exit_price - entry_price) * self.contract_multiplier * 
                      num_contracts - 
                      self.fee_rate * (exit_price + entry_price) * 
                      self.contract_multiplier * num_contracts)
            
            mv_ratio = (exit_price * (1 - self.fee_rate) / 
                       (entry_price * (1 + self.fee_rate)))
            
        else:  # short
            # Short position
            revenue = ((entry_price - exit_price) * self.contract_multiplier * 
                      num_contracts - 
                      self.fee_rate * (exit_price + entry_price) * 
                      self.contract_multiplier * num_contracts)
            
            mv_ratio = 1 + ((entry_price * (1 - self.fee_rate) - 
                            exit_price * (1 + self.fee_rate)) / 
                           (entry_price * (1 + self.fee_rate)))
        
        new_mv = self.market_values[-1] * mv_ratio
        self.market_values.append(new_mv)
        self.log_market_values.append(math.log(new_mv))
        
        return revenue
    
    def _check_continuity(self, entry_price, exit_price, pre_close, end_close):
        """Check if intraday trend continues from pre-market trend."""
        intraday_up = exit_price > entry_price
        premarket_up = end_close > pre_close
        
        return 1 if (intraday_up and premarket_up) or (not intraday_up and not premarket_up) else 0
    
    def save_results(self, output_path="results"):
        """Save trading results to files."""
        os.makedirs(output_path, exist_ok=True)
        
        # Save continuity analysis
        df_continuity = pd.DataFrame({
            'noise_ratio': self.noise_ratios,
            'mean_noise_ratio': self.mean_noise_ratios,
            'continuity': self.continuity
        })
        df_continuity.to_csv(os.path.join(output_path, 'continuity_analysis.csv'), 
                            index=False)
        
        # Save market value curve
        df_mv = pd.DataFrame({
            'date': self.trade_dates,
            'market_value': self.market_values,
            'log_market_value': self.log_market_values
        })
        df_mv.to_csv(os.path.join(output_path, 'market_value.csv'), index=False)
        
        print(f"\nResults saved to {output_path}/")
    
    def plot_results(self):
        """Plot trading performance."""
        # Plot market value curve
        plt.figure(figsize=(12, 6))
        plt.plot(self.trade_dates[1:], self.market_values[1:])
        plt.xlabel('Date')
        plt.ylabel('Market Value (Normalized)')
        plt.title('Portfolio Market Value Over Time')
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('results/market_value.png', dpi=300)
        plt.show()
        
        # Plot noise ratios
        plt.figure(figsize=(12, 6))
        plt.plot(self.noise_ratios, label='Current Noise Ratio', alpha=0.7)
        plt.plot(self.mean_noise_ratios, label='Mean Noise Ratio', alpha=0.7)
        plt.xlabel('Trading Day')
        plt.ylabel('Noise Ratio')
        plt.title('EMD Noise Ratios')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('results/noise_ratios.png', dpi=300)
        plt.show()


def main():
    """Main execution function."""
    # Configuration
    data_path = "data/IF000.csv"
    
    # Initialize trading system
    system = EMDTradingSystem(data_path)
    
    print("Starting EMD Trading System Backtest...")
    print("=" * 60)
    
    # Run backtest
    system.run_backtest()
    
    # Save and plot results
    system.save_results()
    system.plot_results()
    
    print("=" * 60)
    print("Backtest completed!")


if __name__ == "__main__":
    main()


