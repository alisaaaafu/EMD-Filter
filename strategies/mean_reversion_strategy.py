# -*- coding: utf-8 -*-
"""
Mean reversion (anti-trend) strategy using technical indicators.

@author: Wanzhen Fu
"""
import numpy as np
import talib
import pandas as pd


class MeanReversionStrategy:
    """Mean reversion trading strategy."""
    
    def __init__(self, data_path):
        """
        Initialize mean reversion strategy with market data.
        
        Args:
            data_path: Path to CSV file containing OHLC data
        """
        df = pd.read_csv(data_path)
        self.open_price = df.iloc[:, 1]
        self.close_price = df.iloc[:, 4]
        self.high = df.iloc[:, 2]
        self.low = df.iloc[:, 3]
        
        # Calculate moving averages
        self.ma_close = talib.MA(np.array(self.close_price), timeperiod=5000)
        self.ma_low = talib.MA(np.array(self.low), timeperiod=5000)
        self.ma_high = talib.MA(np.array(self.high), timeperiod=5000)
    
    def detect_reversal(self, start_idx, end_idx):
        """
        Detect mean reversion signal.
        
        Args:
            start_idx: Start index for observation
            end_idx: End index for evaluation
        
        Returns:
            1 for long signal, -1 for short signal, 0 for no signal
        """
        high_max = max(self.high[start_idx:end_idx])
        low_min = min(self.low[start_idx:end_idx])
        close_current = self.close_price[end_idx]
        close_min = min(self.close_price[start_idx:end_idx])
        close_max = max(self.close_price[start_idx:end_idx])
        
        ma_high_mean = np.mean(self.ma_high[start_idx:end_idx])
        ma_low_mean = np.mean(self.ma_low[start_idx:end_idx])
        
        # Long signal: price rebounding from oversold
        if close_min < ma_high_mean and close_current > self.ma_high[end_idx]:
            return 1
        
        # Short signal: price falling from overbought
        elif close_max > ma_low_mean and close_current < self.ma_low[end_idx]:
            return -1
        
        # Short signal: price falling below mean after being above
        elif close_max > ma_high_mean and close_current < self.ma_close[end_idx]:
            return -1
        
        # Long signal: price rising above mean after being below
        elif close_min < ma_low_mean and close_current > self.ma_close[end_idx]:
            return 1
        
        else:
            return 0
    
    def generate_order(self, initial_idx, start_idx, end_idx):
        """
        Generate trading order based on mean reversion signals.
        
        Args:
            initial_idx: Initial observation index (unused but kept for consistency)
            start_idx: Start index for reversal detection
            end_idx: End index for evaluation
        
        Returns:
            Dictionary with order type and quantity, or None if no signal
        """
        signal = self.detect_reversal(start_idx, end_idx)
        
        if signal == 0:
            return None
        elif signal == 1:
            return {"type": 'long', "num": 10}
        elif signal == -1:
            return {"type": 'short', "num": 10}


def create_mean_reversion_order(data_path, initial_idx, start_idx, end_idx):
    """
    Convenience function to generate mean reversion order.
    
    Args:
        data_path: Path to CSV file
        initial_idx: Initial observation index
        start_idx: Start index for reversal detection
        end_idx: End index for evaluation
    
    Returns:
        Order dictionary or None
    """
    strategy = MeanReversionStrategy(data_path)
    return strategy.generate_order(initial_idx, start_idx, end_idx)


