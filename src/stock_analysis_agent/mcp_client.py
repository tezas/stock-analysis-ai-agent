"""
MCP Client for Stock Data

This module handles communication with MCP (Model Context Protocol) servers
to fetch real-time stock data, historical data, and fundamental information.
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Union
from datetime import datetime, timedelta

import aiohttp
import pandas as pd
import yfinance as yf
from pydantic import BaseModel


class StockData(BaseModel):
    """Stock data model."""
    
    symbol: str
    price: float
    volume: int
    market_cap: Optional[float] = None
    pe_ratio: Optional[float] = None
    dividend_yield: Optional[float] = None
    beta: Optional[float] = None
    fifty_two_week_high: Optional[float] = None
    fifty_two_week_low: Optional[float] = None
    timestamp: datetime


class MCPClient:
    """
    Client for communicating with MCP servers to fetch stock data.
    
    This client can connect to MCP servers that provide stock data APIs
    and also has fallback mechanisms using direct API calls.
    """
    
    def __init__(
        self,
        server_url: str = "http://localhost:8000",
        timeout: int = 30,
        max_retries: int = 3
    ):
        """
        Initialize MCP client.
        
        Args:
            server_url: URL of the MCP server
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
        """
        self.server_url = server_url.rstrip('/')
        self.timeout = timeout
        self.max_retries = max_retries
        self.session: Optional[aiohttp.ClientSession] = None
        self.logger = logging.getLogger(__name__)
        
        # Fallback data sources
        self.use_fallback = True
        self.fallback_sources = ['yfinance', 'alpha_vantage']
        
        self.logger.info(f"MCP Client initialized for server: {server_url}")
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
    
    async def connect(self):
        """Establish connection to MCP server."""
        if self.session is None:
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            self.session = aiohttp.ClientSession(timeout=timeout)
            self.logger.info("MCP client session created")
    
    async def close(self):
        """Close the client session."""
        if self.session:
            await self.session.close()
            self.session = None
            self.logger.info("MCP client session closed")
    
    async def get_stock_data(self, symbol: str) -> Dict:
        """
        Get current stock data for a symbol.
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL', 'TSLA')
            
        Returns:
            Dictionary containing stock data
        """
        try:
            # Try MCP server first
            data = await self._get_from_mcp_server(f"stock/{symbol}")
            if data:
                return data
            
            # Fallback to direct APIs
            if self.use_fallback:
                return await self._get_stock_data_fallback(symbol)
            
            raise Exception(f"Failed to get stock data for {symbol}")
            
        except Exception as e:
            self.logger.error(f"Error getting stock data for {symbol}: {str(e)}")
            raise
    
    async def get_historical_data(
        self,
        symbol: str,
        period: str = "1y",
        interval: str = "1d"
    ) -> pd.DataFrame:
        """
        Get historical stock data.
        
        Args:
            symbol: Stock symbol
            period: Time period ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max')
            interval: Data interval ('1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo')
            
        Returns:
            DataFrame with historical data
        """
        try:
            # Try MCP server first
            data = await self._get_from_mcp_server(
                f"historical/{symbol}",
                params={"period": period, "interval": interval}
            )
            if data:
                return pd.DataFrame(data)
            
            # Fallback to yfinance
            if self.use_fallback:
                return await self._get_historical_data_fallback(symbol, period, interval)
            
            raise Exception(f"Failed to get historical data for {symbol}")
            
        except Exception as e:
            self.logger.error(f"Error getting historical data for {symbol}: {str(e)}")
            raise
    
    async def get_fundamental_data(self, symbol: str) -> Dict:
        """
        Get fundamental data for a stock.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Dictionary containing fundamental data
        """
        try:
            # Try MCP server first
            data = await self._get_from_mcp_server(f"fundamentals/{symbol}")
            if data:
                return data
            
            # Fallback to yfinance
            if self.use_fallback:
                return await self._get_fundamental_data_fallback(symbol)
            
            raise Exception(f"Failed to get fundamental data for {symbol}")
            
        except Exception as e:
            self.logger.error(f"Error getting fundamental data for {symbol}: {str(e)}")
            raise
    
    async def get_technical_indicators(self, symbol: str) -> Dict:
        """
        Get technical indicators for a stock.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Dictionary containing technical indicators
        """
        try:
            # Try MCP server first
            data = await self._get_from_mcp_server(f"technical/{symbol}")
            if data:
                return data
            
            # Fallback: calculate from historical data
            if self.use_fallback:
                historical_data = await self.get_historical_data(symbol)
                return self._calculate_technical_indicators(historical_data)
            
            raise Exception(f"Failed to get technical indicators for {symbol}")
            
        except Exception as e:
            self.logger.error(f"Error getting technical indicators for {symbol}: {str(e)}")
            raise
    
    async def _get_from_mcp_server(self, endpoint: str, params: Optional[Dict] = None) -> Optional[Dict]:
        """
        Make request to MCP server.
        
        Args:
            endpoint: API endpoint
            params: Query parameters
            
        Returns:
            Response data or None if failed
        """
        if not self.session:
            await self.connect()
        
        url = f"{self.server_url}/api/{endpoint}"
        
        for attempt in range(self.max_retries):
            try:
                async with self.session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        self.logger.debug(f"MCP server response for {endpoint}: {data}")
                        return data
                    else:
                        self.logger.warning(f"MCP server returned {response.status} for {endpoint}")
                        
            except Exception as e:
                self.logger.warning(f"MCP server request failed (attempt {attempt + 1}): {str(e)}")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
        
        return None
    
    async def _get_stock_data_fallback(self, symbol: str) -> Dict:
        """Get stock data using fallback sources."""
        try:
            # Use yfinance as fallback
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            return {
                'symbol': symbol,
                'price': info.get('currentPrice', info.get('regularMarketPrice', 0)),
                'volume': info.get('volume', 0),
                'market_cap': info.get('marketCap'),
                'pe_ratio': info.get('trailingPE'),
                'dividend_yield': info.get('dividendYield'),
                'beta': info.get('beta'),
                '52_week_high': info.get('fiftyTwoWeekHigh'),
                '52_week_low': info.get('fiftyTwoWeekLow'),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Fallback stock data failed for {symbol}: {str(e)}")
            raise
    
    async def _get_historical_data_fallback(
        self,
        symbol: str,
        period: str = "1y",
        interval: str = "1d"
    ) -> pd.DataFrame:
        """Get historical data using fallback sources."""
        try:
            # Use yfinance as fallback
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period, interval=interval)
            
            # Reset index to make date a column
            data = data.reset_index()
            
            # Rename columns for consistency
            data = data.rename(columns={
                'Date': 'date',
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            })
            
            return data
            
        except Exception as e:
            self.logger.error(f"Fallback historical data failed for {symbol}: {str(e)}")
            raise
    
    async def _get_fundamental_data_fallback(self, symbol: str) -> Dict:
        """Get fundamental data using fallback sources."""
        try:
            # Use yfinance as fallback
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            return {
                'symbol': symbol,
                'market_cap': info.get('marketCap'),
                'enterprise_value': info.get('enterpriseValue'),
                'pe_ratio': info.get('trailingPE'),
                'forward_pe': info.get('forwardPE'),
                'peg_ratio': info.get('pegRatio'),
                'price_to_book': info.get('priceToBook'),
                'price_to_sales': info.get('priceToSalesTrailing12Months'),
                'debt_to_equity': info.get('debtToEquity'),
                'current_ratio': info.get('currentRatio'),
                'quick_ratio': info.get('quickRatio'),
                'return_on_equity': info.get('returnOnEquity'),
                'return_on_assets': info.get('returnOnAssets'),
                'profit_margins': info.get('profitMargins'),
                'revenue_growth': info.get('revenueGrowth'),
                'earnings_growth': info.get('earningsGrowth'),
                'dividend_yield': info.get('dividendYield'),
                'payout_ratio': info.get('payoutRatio'),
                'beta': info.get('beta'),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Fallback fundamental data failed for {symbol}: {str(e)}")
            raise
    
    def _calculate_technical_indicators(self, data: pd.DataFrame) -> Dict:
        """Calculate technical indicators from historical data."""
        try:
            import ta
            
            # Ensure we have required columns
            required_columns = ['close', 'high', 'low', 'volume']
            for col in required_columns:
                if col not in data.columns:
                    self.logger.warning(f"Missing column {col} for technical indicators")
                    return {}
            
            # Calculate indicators
            indicators = {}
            
            # Moving averages
            indicators['sma_20'] = ta.trend.sma_indicator(data['close'], window=20).iloc[-1]
            indicators['sma_50'] = ta.trend.sma_indicator(data['close'], window=50).iloc[-1]
            indicators['ema_12'] = ta.trend.ema_indicator(data['close'], window=12).iloc[-1]
            indicators['ema_26'] = ta.trend.ema_indicator(data['close'], window=26).iloc[-1]
            
            # MACD
            macd = ta.trend.MACD(data['close'])
            indicators['macd'] = macd.macd().iloc[-1]
            indicators['macd_signal'] = macd.macd_signal().iloc[-1]
            indicators['macd_histogram'] = macd.macd_diff().iloc[-1]
            
            # RSI
            indicators['rsi'] = ta.momentum.RSIIndicator(data['close']).rsi().iloc[-1]
            
            # Bollinger Bands
            bb = ta.volatility.BollingerBands(data['close'])
            indicators['bb_upper'] = bb.bollinger_hband().iloc[-1]
            indicators['bb_middle'] = bb.bollinger_mavg().iloc[-1]
            indicators['bb_lower'] = bb.bollinger_lband().iloc[-1]
            
            # Stochastic
            stoch = ta.momentum.StochasticOscillator(data['high'], data['low'], data['close'])
            indicators['stoch_k'] = stoch.stoch().iloc[-1]
            indicators['stoch_d'] = stoch.stoch_signal().iloc[-1]
            
            # Volume indicators
            indicators['volume_sma'] = ta.volume.volume_sma(data['close'], data['volume']).iloc[-1]
            
            # Remove NaN values
            indicators = {k: v for k, v in indicators.items() if pd.notna(v)}
            
            return indicators
            
        except Exception as e:
            self.logger.error(f"Error calculating technical indicators: {str(e)}")
            return {}
    
    async def health_check(self) -> bool:
        """
        Check if MCP server is healthy.
        
        Returns:
            True if server is healthy, False otherwise
        """
        try:
            data = await self._get_from_mcp_server("health")
            return data is not None and data.get('status') == 'healthy'
        except Exception:
            return False
    
    async def get_supported_symbols(self) -> List[str]:
        """
        Get list of supported stock symbols from MCP server.
        
        Returns:
            List of supported symbols
        """
        try:
            data = await self._get_from_mcp_server("symbols")
            return data.get('symbols', []) if data else []
        except Exception as e:
            self.logger.error(f"Error getting supported symbols: {str(e)}")
            return [] 