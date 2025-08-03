"""
Tests for the StockAnalysisAgent class.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os
from pathlib import Path

from stock_analysis_agent.agent import StockAnalysisAgent
from stock_analysis_agent.agent import AnalysisResult
from stock_analysis_agent.mcp_client import StockData


class TestStockAnalysisAgent:
    """Test cases for StockAnalysisAgent."""
    
    @pytest.fixture
    def mock_mcp_client(self):
        """Create a mock MCP client."""
        mock_client = Mock()
        mock_client.get_stock_price.return_value = 150.0
        mock_client.get_historical_data.return_value = Mock()
        mock_client.get_fundamental_data.return_value = {
            'market_cap': 2500000000000,
            'pe_ratio': 25.5,
            'dividend_yield': 0.5
        }
        return mock_client
    
    @pytest.fixture
    def mock_ai_client(self):
        """Create a mock AI client."""
        mock_client = Mock()
        mock_client.analyze.return_value = {
            'summary': 'Test analysis summary',
            'recommendation': 'BUY',
            'confidence': 0.85,
            'risks': ['Market volatility'],
            'opportunities': ['Strong fundamentals']
        }
        return mock_client
    
    @pytest.fixture
    def agent(self, mock_mcp_client, mock_ai_client):
        """Create a StockAnalysisAgent instance with mocked dependencies."""
        with patch('stock_analysis_agent.agent.MCPClient', return_value=mock_mcp_client), \
             patch('stock_analysis_agent.agent.OpenAIClient', return_value=mock_ai_client):
            return StockAnalysisAgent(
                mcp_server_url="http://localhost:8000",
                ai_model="gpt-4",
                api_key="test_key"
            )
    
    def test_agent_initialization(self, agent):
        """Test agent initialization."""
        assert agent.mcp_server_url == "http://localhost:8000"
        assert agent.ai_model == "gpt-4"
        assert agent.api_key == "test_key"
    
    def test_analyze_from_text(self, agent):
        """Test analyzing from text prompt."""
        prompt = "Analyze AAPL stock"
        result = agent.analyze_from_text(prompt)
        
        assert isinstance(result, AnalysisResult)
        assert result.symbol == "AAPL"
        assert result.summary is not None
        assert result.recommendation is not None
    
    def test_analyze_from_file(self, agent):
        """Test analyzing from file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("Analyze TSLA stock for the next 30 days")
            temp_file = f.name
        
        try:
            result = agent.analyze_from_file(temp_file)
            assert isinstance(result, AnalysisResult)
            assert result.symbol == "TSLA"
        finally:
            os.unlink(temp_file)
    
    def test_analyze_from_document(self, agent):
        """Test analyzing from Word document."""
        # Mock document processing
        with patch('stock_analysis_agent.agent.extract_text_from_docx') as mock_extract:
            mock_extract.return_value = "Analyze MSFT stock fundamentals"
            
            with tempfile.NamedTemporaryFile(suffix='.docx', delete=False) as f:
                temp_file = f.name
            
            try:
                result = agent.analyze_from_document(temp_file)
                assert isinstance(result, AnalysisResult)
                assert result.symbol == "MSFT"
            finally:
                os.unlink(temp_file)
    
    def test_get_stock_data(self, agent, mock_mcp_client):
        """Test getting stock data."""
        stock_data = agent.get_stock_data("AAPL")
        
        assert isinstance(stock_data, StockData)
        assert stock_data.symbol == "AAPL"
        assert stock_data.current_price == 150.0
        mock_mcp_client.get_stock_price.assert_called_once_with("AAPL")
    
    def test_generate_report(self, agent):
        """Test report generation."""
        # Create a mock analysis result
        result = AnalysisResult(
            symbol="AAPL",
            summary="Test summary",
            recommendation="BUY",
            confidence=0.85,
            risks=["Risk 1"],
            opportunities=["Opportunity 1"],
            technical_analysis={},
            fundamental_analysis={}
        )
        
        report = agent.generate_report(result)
        assert isinstance(report, str)
        assert "AAPL" in report
        assert "BUY" in report
        assert "Test summary" in report
    
    def test_generate_report_html_format(self, agent):
        """Test HTML report generation."""
        result = AnalysisResult(
            symbol="AAPL",
            summary="Test summary",
            recommendation="BUY",
            confidence=0.85,
            risks=["Risk 1"],
            opportunities=["Opportunity 1"],
            technical_analysis={},
            fundamental_analysis={}
        )
        
        report = agent.generate_report(result, format="html")
        assert isinstance(report, str)
        assert "<html>" in report
        assert "AAPL" in report
    
    def test_extract_symbol_from_prompt(self, agent):
        """Test symbol extraction from prompts."""
        test_cases = [
            ("Analyze AAPL stock", "AAPL"),
            ("What about TSLA?", "TSLA"),
            ("MSFT analysis for next week", "MSFT"),
            ("GOOGL fundamentals", "GOOGL"),
            ("NVDA technical analysis", "NVDA"),
        ]
        
        for prompt, expected_symbol in test_cases:
            symbol = agent._extract_symbol_from_prompt(prompt)
            assert symbol == expected_symbol
    
    def test_invalid_prompt_no_symbol(self, agent):
        """Test handling of prompts without stock symbols."""
        prompt = "Analyze the market trends"
        
        with pytest.raises(ValueError, match="No stock symbol found"):
            agent.analyze_from_text(prompt)
    
    def test_mcp_connection_error(self):
        """Test handling of MCP connection errors."""
        with patch('stock_analysis_agent.agent.MCPClient') as mock_mcp_class:
            mock_mcp_class.side_effect = Exception("Connection failed")
            
            with pytest.raises(Exception, match="Connection failed"):
                StockAnalysisAgent(
                    mcp_server_url="http://invalid:8000",
                    ai_model="gpt-4",
                    api_key="test_key"
                )
    
    def test_ai_analysis_error(self, agent, mock_ai_client):
        """Test handling of AI analysis errors."""
        mock_ai_client.analyze.side_effect = Exception("AI analysis failed")
        
        with pytest.raises(Exception, match="AI analysis failed"):
            agent.analyze_from_text("Analyze AAPL stock")
    
    def test_cache_functionality(self, agent):
        """Test caching functionality."""
        # First call should cache the result
        result1 = agent.analyze_from_text("Analyze AAPL stock")
        
        # Second call should use cached result
        result2 = agent.analyze_from_text("Analyze AAPL stock")
        
        assert result1.symbol == result2.symbol
        assert result1.summary == result2.summary
    
    def test_clear_cache(self, agent):
        """Test cache clearing functionality."""
        # Add some data to cache
        agent.analyze_from_text("Analyze AAPL stock")
        
        # Clear cache
        agent.clear_cache()
        
        # Cache should be empty
        assert len(agent._cache) == 0


if __name__ == '__main__':
    pytest.main([__file__]) 