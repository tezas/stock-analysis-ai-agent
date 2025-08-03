"""
Tests for the StockAnalyzer class.
"""

import pytest
from unittest.mock import Mock, patch
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from stock_analysis_agent.stock_analyzer import StockAnalyzer


class TestStockAnalyzer:
    """Test cases for StockAnalyzer."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample stock data for testing."""
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
        data = {
            'Open': np.random.uniform(100, 200, len(dates)),
            'High': np.random.uniform(150, 250, len(dates)),
            'Low': np.random.uniform(50, 150, len(dates)),
            'Close': np.random.uniform(100, 200, len(dates)),
            'Volume': np.random.randint(1000000, 10000000, len(dates))
        }
        return pd.DataFrame(data, index=dates)
    
    @pytest.fixture
    def analyzer(self):
        """Create a StockAnalyzer instance."""
        return StockAnalyzer()
    
    def test_analyzer_initialization(self, analyzer):
        """Test analyzer initialization."""
        assert analyzer is not None
        assert hasattr(analyzer, 'calculate_rsi')
        assert hasattr(analyzer, 'calculate_macd')
        assert hasattr(analyzer, 'calculate_bollinger_bands')
    
    def test_calculate_rsi(self, analyzer, sample_data):
        """Test RSI calculation."""
        rsi = analyzer.calculate_rsi(sample_data['Close'], period=14)
        
        assert isinstance(rsi, pd.Series)
        assert len(rsi) == len(sample_data)
        assert all(0 <= val <= 100 for val in rsi.dropna())
    
    def test_calculate_macd(self, analyzer, sample_data):
        """Test MACD calculation."""
        macd_data = analyzer.calculate_macd(sample_data['Close'])
        
        assert isinstance(macd_data, dict)
        assert 'macd' in macd_data
        assert 'signal' in macd_data
        assert 'histogram' in macd_data
        assert len(macd_data['macd']) == len(sample_data)
    
    def test_calculate_bollinger_bands(self, analyzer, sample_data):
        """Test Bollinger Bands calculation."""
        bb_data = analyzer.calculate_bollinger_bands(sample_data['Close'])
        
        assert isinstance(bb_data, dict)
        assert 'upper' in bb_data
        assert 'middle' in bb_data
        assert 'lower' in bb_data
        assert len(bb_data['upper']) == len(sample_data)
        
        # Check that upper > middle > lower
        for i in range(len(sample_data)):
            if not (pd.isna(bb_data['upper'].iloc[i]) or 
                   pd.isna(bb_data['middle'].iloc[i]) or 
                   pd.isna(bb_data['lower'].iloc[i])):
                assert bb_data['upper'].iloc[i] >= bb_data['middle'].iloc[i] >= bb_data['lower'].iloc[i]
    
    def test_calculate_moving_averages(self, analyzer, sample_data):
        """Test moving averages calculation."""
        ma_data = analyzer.calculate_moving_averages(sample_data['Close'])
        
        assert isinstance(ma_data, dict)
        assert 'sma_20' in ma_data
        assert 'sma_50' in ma_data
        assert 'ema_12' in ma_data
        assert 'ema_26' in ma_data
        
        # Check that EMA responds faster than SMA
        if len(ma_data['ema_12'].dropna()) > 0 and len(ma_data['sma_20'].dropna()) > 0:
            ema_volatility = ma_data['ema_12'].dropna().std()
            sma_volatility = ma_data['sma_20'].dropna().std()
            assert ema_volatility >= sma_volatility
    
    def test_calculate_volume_indicators(self, analyzer, sample_data):
        """Test volume indicators calculation."""
        volume_data = analyzer.calculate_volume_indicators(sample_data)
        
        assert isinstance(volume_data, dict)
        assert 'volume_sma' in volume_data
        assert 'obv' in volume_data
        assert 'vwap' in volume_data
    
    def test_calculate_support_resistance(self, analyzer, sample_data):
        """Test support and resistance calculation."""
        levels = analyzer.calculate_support_resistance(sample_data)
        
        assert isinstance(levels, dict)
        assert 'support' in levels
        assert 'resistance' in levels
        assert isinstance(levels['support'], list)
        assert isinstance(levels['resistance'], list)
    
    def test_analyze_technical_indicators(self, analyzer, sample_data):
        """Test technical analysis."""
        analysis = analyzer.analyze_technical_indicators(sample_data)
        
        assert isinstance(analysis, dict)
        assert 'trend' in analysis
        assert 'strength' in analysis
        assert 'signals' in analysis
        assert 'summary' in analysis
    
    def test_analyze_fundamentals(self, analyzer):
        """Test fundamental analysis."""
        fundamental_data = {
            'market_cap': 2500000000000,
            'pe_ratio': 25.5,
            'dividend_yield': 0.5,
            'debt_to_equity': 0.3,
            'return_on_equity': 0.15,
            'revenue_growth': 0.08
        }
        
        analysis = analyzer.analyze_fundamentals(fundamental_data)
        
        assert isinstance(analysis, dict)
        assert 'valuation' in analysis
        assert 'financial_health' in analysis
        assert 'growth_potential' in analysis
        assert 'summary' in analysis
    
    def test_generate_trading_signals(self, analyzer, sample_data):
        """Test trading signal generation."""
        signals = analyzer.generate_trading_signals(sample_data)
        
        assert isinstance(signals, dict)
        assert 'buy_signals' in signals
        assert 'sell_signals' in signals
        assert 'hold_signals' in signals
        assert 'confidence' in signals
    
    def test_calculate_risk_metrics(self, analyzer, sample_data):
        """Test risk metrics calculation."""
        risk_metrics = analyzer.calculate_risk_metrics(sample_data)
        
        assert isinstance(risk_metrics, dict)
        assert 'volatility' in risk_metrics
        assert 'sharpe_ratio' in risk_metrics
        assert 'max_drawdown' in risk_metrics
        assert 'var_95' in risk_metrics
    
    def test_empty_data_handling(self, analyzer):
        """Test handling of empty data."""
        empty_df = pd.DataFrame()
        
        with pytest.raises(ValueError):
            analyzer.calculate_rsi(empty_df['Close'])
    
    def test_invalid_period(self, analyzer, sample_data):
        """Test handling of invalid periods."""
        with pytest.raises(ValueError):
            analyzer.calculate_rsi(sample_data['Close'], period=0)
    
    def test_data_with_nan_values(self, analyzer):
        """Test handling of data with NaN values."""
        data_with_nan = pd.Series([1, 2, np.nan, 4, 5])
        
        # Should handle NaN values gracefully
        rsi = analyzer.calculate_rsi(data_with_nan)
        assert isinstance(rsi, pd.Series)
    
    def test_performance_optimization(self, analyzer):
        """Test performance with large datasets."""
        # Create large dataset
        large_data = pd.DataFrame({
            'Close': np.random.uniform(100, 200, 10000),
            'Volume': np.random.randint(1000000, 10000000, 10000)
        })
        
        # Should complete within reasonable time
        import time
        start_time = time.time()
        
        rsi = analyzer.calculate_rsi(large_data['Close'])
        macd = analyzer.calculate_macd(large_data['Close'])
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Should complete within 5 seconds
        assert execution_time < 5.0
        assert len(rsi) == 10000
        assert len(macd['macd']) == 10000
    
    def test_indicator_consistency(self, analyzer, sample_data):
        """Test consistency of indicators."""
        # Calculate indicators multiple times
        rsi1 = analyzer.calculate_rsi(sample_data['Close'])
        rsi2 = analyzer.calculate_rsi(sample_data['Close'])
        
        # Results should be identical
        pd.testing.assert_series_equal(rsi1, rsi2)
    
    def test_edge_cases(self, analyzer):
        """Test edge cases."""
        # Single value
        single_value = pd.Series([100])
        rsi = analyzer.calculate_rsi(single_value)
        assert len(rsi) == 1
        
        # All same values
        same_values = pd.Series([100] * 50)
        rsi = analyzer.calculate_rsi(same_values)
        assert all(pd.isna(val) or val == 50 for val in rsi)
    
    def test_technical_analysis_integration(self, analyzer, sample_data):
        """Test integration of all technical indicators."""
        analysis = analyzer.analyze_technical_indicators(sample_data)
        
        # Should include all major indicators
        assert 'rsi' in analysis
        assert 'macd' in analysis
        assert 'bollinger_bands' in analysis
        assert 'moving_averages' in analysis
        assert 'volume' in analysis


if __name__ == '__main__':
    pytest.main([__file__]) 