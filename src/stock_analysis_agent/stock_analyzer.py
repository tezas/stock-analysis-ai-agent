"""
Stock Analyzer

This module provides technical analysis, fundamental analysis,
and recommendation generation for stocks.
"""

import logging
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np


class StockAnalyzer:
    """
    Analyzes stock data and generates recommendations.
    
    This class provides technical analysis, fundamental analysis,
    risk assessment, and investment recommendations.
    """
    
    def __init__(self):
        """Initialize the stock analyzer."""
        self.logger = logging.getLogger(__name__)
        
        # Risk thresholds
        self.risk_thresholds = {
            'low': {'rsi_oversold': 30, 'rsi_overbought': 70, 'volatility': 0.15},
            'medium': {'rsi_oversold': 25, 'rsi_overbought': 75, 'volatility': 0.25},
            'high': {'rsi_oversold': 20, 'rsi_overbought': 80, 'volatility': 0.35}
        }
    
    def calculate_technical_indicators(self, data: pd.DataFrame) -> Dict:
        """
        Calculate technical indicators from historical data.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            Dictionary containing technical indicators
        """
        try:
            indicators = {}
            
            # Ensure we have required columns
            required_columns = ['close', 'high', 'low', 'volume']
            for col in required_columns:
                if col not in data.columns:
                    self.logger.warning(f"Missing column {col} for technical indicators")
                    return {}
            
            # Calculate basic indicators
            indicators.update(self._calculate_moving_averages(data))
            indicators.update(self._calculate_momentum_indicators(data))
            indicators.update(self._calculate_volatility_indicators(data))
            indicators.update(self._calculate_volume_indicators(data))
            indicators.update(self._calculate_trend_indicators(data))
            
            # Remove NaN values
            indicators = {k: v for k, v in indicators.items() if pd.notna(v)}
            
            self.logger.info(f"Calculated {len(indicators)} technical indicators")
            return indicators
            
        except Exception as e:
            self.logger.error(f"Error calculating technical indicators: {str(e)}")
            return {}
    
    def _calculate_moving_averages(self, data: pd.DataFrame) -> Dict:
        """Calculate moving averages."""
        indicators = {}
        
        try:
            # Simple Moving Averages
            for period in [5, 10, 20, 50, 200]:
                sma = data['close'].rolling(window=period).mean()
                indicators[f'sma_{period}'] = sma.iloc[-1]
            
            # Exponential Moving Averages
            for period in [12, 26, 50]:
                ema = data['close'].ewm(span=period).mean()
                indicators[f'ema_{period}'] = ema.iloc[-1]
            
            # Moving Average Crossovers
            if 'sma_20' in indicators and 'sma_50' in indicators:
                indicators['sma_crossover'] = indicators['sma_20'] > indicators['sma_50']
            
            if 'ema_12' in indicators and 'ema_26' in indicators:
                indicators['ema_crossover'] = indicators['ema_12'] > indicators['ema_26']
                
        except Exception as e:
            self.logger.warning(f"Error calculating moving averages: {str(e)}")
        
        return indicators
    
    def _calculate_momentum_indicators(self, data: pd.DataFrame) -> Dict:
        """Calculate momentum indicators."""
        indicators = {}
        
        try:
            # RSI
            delta = data['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            indicators['rsi'] = rsi.iloc[-1]
            
            # MACD
            ema_12 = data['close'].ewm(span=12).mean()
            ema_26 = data['close'].ewm(span=26).mean()
            macd_line = ema_12 - ema_26
            signal_line = macd_line.ewm(span=9).mean()
            histogram = macd_line - signal_line
            
            indicators['macd'] = macd_line.iloc[-1]
            indicators['macd_signal'] = signal_line.iloc[-1]
            indicators['macd_histogram'] = histogram.iloc[-1]
            
            # Stochastic
            low_min = data['low'].rolling(window=14).min()
            high_max = data['high'].rolling(window=14).max()
            k_percent = 100 * ((data['close'] - low_min) / (high_max - low_min))
            d_percent = k_percent.rolling(window=3).mean()
            
            indicators['stoch_k'] = k_percent.iloc[-1]
            indicators['stoch_d'] = d_percent.iloc[-1]
            
        except Exception as e:
            self.logger.warning(f"Error calculating momentum indicators: {str(e)}")
        
        return indicators
    
    def _calculate_volatility_indicators(self, data: pd.DataFrame) -> Dict:
        """Calculate volatility indicators."""
        indicators = {}
        
        try:
            # Bollinger Bands
            sma_20 = data['close'].rolling(window=20).mean()
            std_20 = data['close'].rolling(window=20).std()
            
            indicators['bb_upper'] = sma_20 + (std_20 * 2)
            indicators['bb_middle'] = sma_20
            indicators['bb_lower'] = sma_20 - (std_20 * 2)
            
            # Current price position in Bollinger Bands
            current_price = data['close'].iloc[-1]
            bb_range = indicators['bb_upper'] - indicators['bb_lower']
            if bb_range > 0:
                indicators['bb_position'] = (current_price - indicators['bb_lower']) / bb_range
            
            # Average True Range (ATR)
            high_low = data['high'] - data['low']
            high_close = np.abs(data['high'] - data['close'].shift())
            low_close = np.abs(data['low'] - data['close'].shift())
            true_range = np.maximum(high_low, np.maximum(high_close, low_close))
            atr = true_range.rolling(window=14).mean()
            indicators['atr'] = atr.iloc[-1]
            
        except Exception as e:
            self.logger.warning(f"Error calculating volatility indicators: {str(e)}")
        
        return indicators
    
    def _calculate_volume_indicators(self, data: pd.DataFrame) -> Dict:
        """Calculate volume indicators."""
        indicators = {}
        
        try:
            # Volume SMA
            indicators['volume_sma'] = data['volume'].rolling(window=20).mean().iloc[-1]
            
            # Volume ratio
            current_volume = data['volume'].iloc[-1]
            if indicators['volume_sma'] > 0:
                indicators['volume_ratio'] = current_volume / indicators['volume_sma']
            
            # On-Balance Volume (OBV)
            obv = (np.sign(data['close'].diff()) * data['volume']).fillna(0).cumsum()
            indicators['obv'] = obv.iloc[-1]
            
        except Exception as e:
            self.logger.warning(f"Error calculating volume indicators: {str(e)}")
        
        return indicators
    
    def _calculate_trend_indicators(self, data: pd.DataFrame) -> Dict:
        """Calculate trend indicators."""
        indicators = {}
        
        try:
            # ADX (Average Directional Index)
            high_diff = data['high'].diff()
            low_diff = data['low'].diff()
            
            plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
            minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)
            
            plus_di = 100 * (pd.Series(plus_dm).rolling(14).mean() / indicators.get('atr', 1))
            minus_di = 100 * (pd.Series(minus_dm).rolling(14).mean() / indicators.get('atr', 1))
            
            dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
            adx = dx.rolling(14).mean()
            indicators['adx'] = adx.iloc[-1]
            
            # Price position relative to moving averages
            current_price = data['close'].iloc[-1]
            if 'sma_20' in indicators:
                indicators['price_vs_sma20'] = (current_price / indicators['sma_20']) - 1
            if 'sma_50' in indicators:
                indicators['price_vs_sma50'] = (current_price / indicators['sma_50']) - 1
            
        except Exception as e:
            self.logger.warning(f"Error calculating trend indicators: {str(e)}")
        
        return indicators
    
    def generate_recommendations(
        self,
        technical_indicators: Dict,
        fundamental_data: Dict,
        ai_analysis: str
    ) -> List[str]:
        """
        Generate investment recommendations based on analysis.
        
        Args:
            technical_indicators: Technical analysis indicators
            fundamental_data: Fundamental data
            ai_analysis: AI-generated analysis
            
        Returns:
            List of recommendations
        """
        recommendations = []
        
        try:
            # Technical analysis recommendations
            tech_recs = self._generate_technical_recommendations(technical_indicators)
            recommendations.extend(tech_recs)
            
            # Fundamental analysis recommendations
            fund_recs = self._generate_fundamental_recommendations(fundamental_data)
            recommendations.extend(fund_recs)
            
            # Risk-based recommendations
            risk_recs = self._generate_risk_recommendations(technical_indicators, fundamental_data)
            recommendations.extend(risk_recs)
            
            # General recommendations
            general_recs = self._generate_general_recommendations(technical_indicators, fundamental_data)
            recommendations.extend(general_recs)
            
            # Limit to top recommendations
            recommendations = recommendations[:10]
            
            self.logger.info(f"Generated {len(recommendations)} recommendations")
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Error generating recommendations: {str(e)}")
            return ["Unable to generate specific recommendations at this time."]
    
    def _generate_technical_recommendations(self, indicators: Dict) -> List[str]:
        """Generate recommendations based on technical indicators."""
        recommendations = []
        
        try:
            # RSI-based recommendations
            if 'rsi' in indicators:
                rsi = indicators['rsi']
                if rsi < 30:
                    recommendations.append("RSI indicates oversold conditions - potential buying opportunity")
                elif rsi > 70:
                    recommendations.append("RSI indicates overbought conditions - consider taking profits")
            
            # Moving average recommendations
            if 'sma_crossover' in indicators:
                if indicators['sma_crossover']:
                    recommendations.append("Price above 20-day SMA - bullish short-term trend")
                else:
                    recommendations.append("Price below 20-day SMA - bearish short-term trend")
            
            # MACD recommendations
            if 'macd' in indicators and 'macd_signal' in indicators:
                if indicators['macd'] > indicators['macd_signal']:
                    recommendations.append("MACD above signal line - bullish momentum")
                else:
                    recommendations.append("MACD below signal line - bearish momentum")
            
            # Bollinger Bands recommendations
            if 'bb_position' in indicators:
                bb_pos = indicators['bb_position']
                if bb_pos < 0.2:
                    recommendations.append("Price near lower Bollinger Band - potential support level")
                elif bb_pos > 0.8:
                    recommendations.append("Price near upper Bollinger Band - potential resistance level")
            
        except Exception as e:
            self.logger.warning(f"Error generating technical recommendations: {str(e)}")
        
        return recommendations
    
    def _generate_fundamental_recommendations(self, data: Dict) -> List[str]:
        """Generate recommendations based on fundamental data."""
        recommendations = []
        
        try:
            # P/E ratio analysis
            if 'pe_ratio' in data and data['pe_ratio']:
                pe = data['pe_ratio']
                if pe < 15:
                    recommendations.append("Low P/E ratio suggests potential value opportunity")
                elif pe > 25:
                    recommendations.append("High P/E ratio indicates premium valuation")
            
            # Debt analysis
            if 'debt_to_equity' in data and data['debt_to_equity']:
                debt_equity = data['debt_to_equity']
                if debt_equity < 0.5:
                    recommendations.append("Low debt-to-equity ratio indicates strong financial position")
                elif debt_equity > 1.0:
                    recommendations.append("High debt-to-equity ratio suggests financial risk")
            
            # Profitability analysis
            if 'return_on_equity' in data and data['return_on_equity']:
                roe = data['return_on_equity']
                if roe > 0.15:
                    recommendations.append("Strong return on equity indicates efficient capital usage")
                elif roe < 0.05:
                    recommendations.append("Low return on equity suggests poor profitability")
            
            # Growth analysis
            if 'revenue_growth' in data and data['revenue_growth']:
                growth = data['revenue_growth']
                if growth > 0.1:
                    recommendations.append("Strong revenue growth indicates expanding business")
                elif growth < 0:
                    recommendations.append("Declining revenue growth suggests business challenges")
            
        except Exception as e:
            self.logger.warning(f"Error generating fundamental recommendations: {str(e)}")
        
        return recommendations
    
    def _generate_risk_recommendations(self, technical: Dict, fundamental: Dict) -> List[str]:
        """Generate risk-based recommendations."""
        recommendations = []
        
        try:
            # Volatility assessment
            if 'atr' in technical:
                atr = technical['atr']
                current_price = technical.get('sma_20', 100)  # Use SMA as proxy for current price
                if current_price > 0:
                    volatility = atr / current_price
                    if volatility > 0.03:
                        recommendations.append("High volatility detected - consider position sizing and stop losses")
                    elif volatility < 0.01:
                        recommendations.append("Low volatility environment - suitable for conservative strategies")
            
            # Liquidity assessment
            if 'volume_ratio' in technical:
                vol_ratio = technical['volume_ratio']
                if vol_ratio < 0.5:
                    recommendations.append("Low trading volume - consider liquidity risk")
                elif vol_ratio > 2.0:
                    recommendations.append("High trading volume - good liquidity for entry/exit")
            
        except Exception as e:
            self.logger.warning(f"Error generating risk recommendations: {str(e)}")
        
        return recommendations
    
    def _generate_general_recommendations(self, technical: Dict, fundamental: Dict) -> List[str]:
        """Generate general investment recommendations."""
        recommendations = []
        
        try:
            # Diversification
            recommendations.append("Consider diversifying across different sectors and asset classes")
            
            # Dollar-cost averaging
            recommendations.append("Consider dollar-cost averaging for long-term positions")
            
            # Stop losses
            recommendations.append("Implement stop-loss orders to manage downside risk")
            
            # Regular review
            recommendations.append("Review positions regularly and adjust based on changing market conditions")
            
        except Exception as e:
            self.logger.warning(f"Error generating general recommendations: {str(e)}")
        
        return recommendations
    
    def assess_risk(self, technical_indicators: Dict, fundamental_data: Dict) -> str:
        """
        Assess overall risk level of the stock.
        
        Args:
            technical_indicators: Technical analysis indicators
            fundamental_data: Fundamental data
            
        Returns:
            Risk assessment string
        """
        try:
            risk_score = 0
            risk_factors = []
            
            # Technical risk factors
            if 'rsi' in technical_indicators:
                rsi = technical_indicators['rsi']
                if rsi > 80 or rsi < 20:
                    risk_score += 2
                    risk_factors.append("Extreme RSI levels")
            
            if 'atr' in technical_indicators:
                atr = technical_indicators['atr']
                current_price = technical_indicators.get('sma_20', 100)
                if current_price > 0:
                    volatility = atr / current_price
                    if volatility > 0.05:
                        risk_score += 3
                        risk_factors.append("High volatility")
            
            # Fundamental risk factors
            if 'debt_to_equity' in fundamental_data and fundamental_data['debt_to_equity']:
                if fundamental_data['debt_to_equity'] > 1.0:
                    risk_score += 2
                    risk_factors.append("High debt levels")
            
            if 'pe_ratio' in fundamental_data and fundamental_data['pe_ratio']:
                if fundamental_data['pe_ratio'] > 30:
                    risk_score += 1
                    risk_factors.append("High valuation")
            
            # Determine risk level
            if risk_score >= 5:
                risk_level = "High"
            elif risk_score >= 3:
                risk_level = "Medium"
            else:
                risk_level = "Low"
            
            risk_assessment = f"Risk Level: {risk_level}"
            if risk_factors:
                risk_assessment += f" - Factors: {', '.join(risk_factors)}"
            
            return risk_assessment
            
        except Exception as e:
            self.logger.error(f"Error assessing risk: {str(e)}")
            return "Risk assessment unavailable"
    
    def calculate_confidence_score(self, technical_indicators: Dict, fundamental_data: Dict) -> float:
        """
        Calculate confidence score for the analysis.
        
        Args:
            technical_indicators: Technical analysis indicators
            fundamental_data: Fundamental data
            
        Returns:
            Confidence score between 0 and 1
        """
        try:
            score = 0.5  # Base score
            factors = 0
            
            # Technical indicators confidence
            if technical_indicators:
                tech_score = min(len(technical_indicators) / 10, 1.0)  # Normalize to 0-1
                score += tech_score * 0.3
                factors += 1
            
            # Fundamental data confidence
            if fundamental_data:
                fund_score = min(len(fundamental_data) / 15, 1.0)  # Normalize to 0-1
                score += fund_score * 0.2
                factors += 1
            
            # Data quality factors
            if 'rsi' in technical_indicators and pd.notna(technical_indicators['rsi']):
                score += 0.1
                factors += 1
            
            if 'pe_ratio' in fundamental_data and pd.notna(fundamental_data['pe_ratio']):
                score += 0.1
                factors += 1
            
            # Normalize by number of factors
            if factors > 0:
                score = score / factors
            
            # Ensure score is between 0 and 1
            score = max(0.0, min(1.0, score))
            
            return round(score, 3)
            
        except Exception as e:
            self.logger.error(f"Error calculating confidence score: {str(e)}")
            return 0.5 