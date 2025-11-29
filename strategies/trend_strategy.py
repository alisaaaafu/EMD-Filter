# -*- coding: utf-8 -*-
"""
Trend following strategy using technical indicators.

@author: Wanzhen Fu
"""
import pandas as pd
import numpy as np
import talib


class TrendStrategy:
    """Trend following trading strategy."""
    
    def __init__(self, data_path):
        """
        Initialize trend strategy with market data.
        
        Args:
            data_path: Path to CSV file containing OHLC data
        """
        df = pd.read_csv(data_path)
        self.open_price = df.iloc[:, 1]
        self.close_price = df.iloc[:, 4]
        self.high = df.iloc[:, 2]
        self.low = df.iloc[:, 3]
        
        # Calculate technical indicators
        self.momentum = talib.MOM(np.array(self.close_price), timeperiod=150)
        self.ma_momentum = talib.MA(np.array(self.momentum), timeperiod=900)
        self.ma_low = talib.MA(np.array(self.low), timeperiod=1000)
        self.ma_high = talib.MA(np.array(self.high), timeperiod=1000)
    
    def generate_order(self, initial_idx, start_idx, end_idx):
        """
        Generate trading order based on trend signals.
        
        Args:
            initial_idx: Initial observation index
            start_idx: Start index for trend detection
            end_idx: End index for evaluation
        
        Returns:
            Dictionary with order type and quantity, or None if no signal
        """
        highest = max(self.close_price[initial_idx:end_idx])
        lowest = min(self.close_price[initial_idx:end_idx])
        price_t = self.close_price[end_idx]
        price_initial = self.close_price[initial_idx]
        price_start = self.close_price[start_idx]
        
        # Find positions of highest and lowest prices
        temp_high_idx = 0
        temp_low_idx = 0
        for i, p in enumerate(self.close_price[initial_idx:end_idx]):
            if p == highest:
                temp_high_idx = i
            elif p == lowest:
                temp_low_idx = i
        
        # Long signal conditions
        if (price_start > price_initial and 
            (price_t - price_start) > 0 and 
            price_t > max(self.ma_high[initial_idx:end_idx]) and 
            self.momentum[end_idx] > self.ma_momentum[end_idx] and 
            temp_high_idx > temp_low_idx):
            return {"type": 'long', "num": 10}
        
        # Short signal conditions
        elif (price_start < price_initial and 
              (price_t - price_start) < 0 and 
              price_t < min(self.ma_low[initial_idx:end_idx]) and 
              self.momentum[end_idx] < self.ma_momentum[end_idx] and 
              temp_high_idx < temp_low_idx):
            return {"type": 'short', "num": 10}
        
        else:
            return None


def create_trend_order(data_path, initial_idx, start_idx, end_idx):
    """
    Convenience function to generate trend order.
    
    Args:
        data_path: Path to CSV file
        initial_idx: Initial observation index
        start_idx: Start index for trend detection
        end_idx: End index for evaluation
    
    Returns:
        Order dictionary or None
    """
    strategy = TrendStrategy(data_path)
    return strategy.generate_order(initial_idx, start_idx, end_idx)


