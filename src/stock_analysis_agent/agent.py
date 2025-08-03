"""
Main Stock Analysis AI Agent

This module contains the main agent class that orchestrates stock analysis
using MCP servers and AI models.
"""

import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd
from pydantic import BaseModel, Field, ConfigDict

from .mcp_client import MCPClient
from .stock_analyzer import StockAnalyzer
from .prompt_processor import PromptProcessor
from .report_generator import ReportGenerator
from .utils import setup_logging, load_config


class AnalysisResult(BaseModel):
    """Result of stock analysis."""
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    symbol: str
    analysis_type: str
    prompt: str
    stock_data: Dict
    technical_indicators: Dict
    fundamental_data: Dict
    ai_analysis: str
    recommendations: List[str]
    risk_assessment: str
    confidence_score: float = Field(ge=0.0, le=1.0)
    timestamp: pd.Timestamp = Field(default_factory=pd.Timestamp.now)


class StockAnalysisAgent:
    """
    Main AI agent for stock analysis using MCP servers.
    
    This agent coordinates between MCP servers for data retrieval,
    AI models for analysis, and report generation.
    """
    
    def __init__(
        self,
        mcp_server_url: str = "http://localhost:8000",
        ai_model: str = "gpt-4",
        config_path: Optional[str] = None,
        log_level: str = "INFO"
    ):
        """
        Initialize the Stock Analysis Agent.
        
        Args:
            mcp_server_url: URL of the MCP server for stock data
            ai_model: AI model to use for analysis (e.g., 'gpt-4', 'gpt-3.5-turbo')
            config_path: Path to configuration file
            log_level: Logging level
        """
        self.logger = setup_logging(__name__, log_level)
        self.config = load_config(config_path)
        
        # Initialize components
        self.mcp_client = MCPClient(mcp_server_url)
        self.stock_analyzer = StockAnalyzer()
        self.prompt_processor = PromptProcessor()
        self.report_generator = ReportGenerator()
        
        # AI model configuration
        self.ai_model = ai_model
        self.ai_client = self._setup_ai_client()
        
        self.logger.info(f"Stock Analysis Agent initialized with AI model: {ai_model}")
    
    def _setup_ai_client(self):
        """Setup AI client (OpenAI or other providers)."""
        try:
            import openai
            return openai.OpenAI()
        except ImportError:
            self.logger.warning("OpenAI not available, using mock AI client")
            return self._create_mock_ai_client()
    
    def _create_mock_ai_client(self):
        """Create a mock AI client for testing."""
        class MockAIClient:
            def chat(self, **kwargs):
                return type('Response', (), {
                    'choices': [type('Choice', (), {
                        'message': type('Message', (), {
                            'content': "This is a mock AI response for testing purposes."
                        })()
                    })()]
                })()
        return MockAIClient()
    
    async def analyze_from_file(self, prompt_file: Union[str, Path]) -> AnalysisResult:
        """
        Analyze stock based on prompt from a text file.
        
        Args:
            prompt_file: Path to the prompt file
            
        Returns:
            AnalysisResult containing the analysis
        """
        self.logger.info(f"Analyzing from file: {prompt_file}")
        
        # Read and process prompt
        prompt = self.prompt_processor.read_text_file(prompt_file)
        return await self._perform_analysis(prompt)
    
    async def analyze_from_document(self, document_file: Union[str, Path]) -> AnalysisResult:
        """
        Analyze stock based on prompt from a Word document.
        
        Args:
            document_file: Path to the Word document
            
        Returns:
            AnalysisResult containing the analysis
        """
        self.logger.info(f"Analyzing from document: {document_file}")
        
        # Read and process document
        prompt = self.prompt_processor.read_word_document(document_file)
        return await self._perform_analysis(prompt)
    
    async def analyze_from_text(self, prompt_text: str) -> AnalysisResult:
        """
        Analyze stock based on text prompt.
        
        Args:
            prompt_text: Text prompt for analysis
            
        Returns:
            AnalysisResult containing the analysis
        """
        self.logger.info("Analyzing from text prompt")
        
        # Process prompt
        prompt = self.prompt_processor.process_text(prompt_text)
        return await self._perform_analysis(prompt)
    
    async def _perform_analysis(self, prompt: str) -> AnalysisResult:
        """
        Perform the complete stock analysis workflow.
        
        Args:
            prompt: Processed prompt for analysis
            
        Returns:
            AnalysisResult containing the analysis
        """
        try:
            # Extract stock symbol and analysis requirements
            analysis_request = self.prompt_processor.extract_analysis_request(prompt)
            symbol = analysis_request.symbol
            
            self.logger.info(f"Starting analysis for {symbol}")
            
            # Get stock data from MCP server
            stock_data = await self.mcp_client.get_stock_data(symbol)
            
            # Get historical data
            historical_data = await self.mcp_client.get_historical_data(symbol)
            
            # Get fundamental data
            fundamental_data = await self.mcp_client.get_fundamental_data(symbol)
            
            # Calculate technical indicators
            technical_indicators = self.stock_analyzer.calculate_technical_indicators(
                historical_data
            )
            
            # Perform AI analysis
            ai_analysis = await self._get_ai_analysis(
                symbol, stock_data, technical_indicators, fundamental_data, prompt
            )
            
            # Generate recommendations
            recommendations = self.stock_analyzer.generate_recommendations(
                technical_indicators, fundamental_data, ai_analysis
            )
            
            # Assess risk
            risk_assessment = self.stock_analyzer.assess_risk(
                technical_indicators, fundamental_data
            )
            
            # Calculate confidence score
            confidence_score = self.stock_analyzer.calculate_confidence_score(
                technical_indicators, fundamental_data
            )
            
            # Create analysis result
            result = AnalysisResult(
                symbol=symbol,
                analysis_type=analysis_request.analysis_type,
                prompt=prompt,
                stock_data=stock_data,
                technical_indicators=technical_indicators,
                fundamental_data=fundamental_data,
                ai_analysis=ai_analysis,
                recommendations=recommendations,
                risk_assessment=risk_assessment,
                confidence_score=confidence_score
            )
            
            self.logger.info(f"Analysis completed for {symbol}")
            return result
            
        except Exception as e:
            self.logger.error(f"Error during analysis: {str(e)}")
            raise
    
    async def _get_ai_analysis(
        self,
        symbol: str,
        stock_data: Dict,
        technical_indicators: Dict,
        fundamental_data: Dict,
        prompt: str
    ) -> str:
        """
        Get AI analysis using the configured AI model.
        
        Args:
            symbol: Stock symbol
            stock_data: Current stock data
            technical_indicators: Technical analysis indicators
            fundamental_data: Fundamental data
            prompt: Original prompt
            
        Returns:
            AI analysis text
        """
        try:
            # Prepare context for AI
            context = self._prepare_ai_context(
                symbol, stock_data, technical_indicators, fundamental_data
            )
            
            # Create AI prompt
            ai_prompt = self._create_ai_prompt(prompt, context)
            
            # Get AI response
            response = self.ai_client.chat.completions.create(
                model=self.ai_model,
                messages=[
                    {"role": "system", "content": self._get_system_prompt()},
                    {"role": "user", "content": ai_prompt}
                ],
                temperature=0.7,
                max_tokens=2000
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            self.logger.error(f"Error getting AI analysis: {str(e)}")
            return f"AI analysis failed: {str(e)}"
    
    def _prepare_ai_context(
        self,
        symbol: str,
        stock_data: Dict,
        technical_indicators: Dict,
        fundamental_data: Dict
    ) -> str:
        """Prepare context for AI analysis."""
        context = f"""
Stock Symbol: {symbol}

Current Stock Data:
- Price: ${stock_data.get('price', 'N/A')}
- Volume: {stock_data.get('volume', 'N/A')}
- Market Cap: ${stock_data.get('market_cap', 'N/A')}
- 52-Week High: ${stock_data.get('52_week_high', 'N/A')}
- 52-Week Low: ${stock_data.get('52_week_low', 'N/A')}

Technical Indicators:
{self._format_technical_indicators(technical_indicators)}

Fundamental Data:
{self._format_fundamental_data(fundamental_data)}
"""
        return context
    
    def _format_technical_indicators(self, indicators: Dict) -> str:
        """Format technical indicators for AI context."""
        formatted = []
        for indicator, value in indicators.items():
            if isinstance(value, (int, float)):
                formatted.append(f"- {indicator}: {value:.4f}")
            else:
                formatted.append(f"- {indicator}: {value}")
        return "\n".join(formatted)
    
    def _format_fundamental_data(self, data: Dict) -> str:
        """Format fundamental data for AI context."""
        formatted = []
        for key, value in data.items():
            if isinstance(value, (int, float)):
                formatted.append(f"- {key}: {value:.2f}")
            else:
                formatted.append(f"- {key}: {value}")
        return "\n".join(formatted)
    
    def _create_ai_prompt(self, original_prompt: str, context: str) -> str:
        """Create AI prompt with context."""
        return f"""
{original_prompt}

Please analyze the following stock data and provide a comprehensive analysis:

{context}

Please provide:
1. Technical analysis summary
2. Fundamental analysis summary
3. Risk assessment
4. Investment recommendations
5. Key factors to watch
"""
    
    def _get_system_prompt(self) -> str:
        """Get system prompt for AI model."""
        return """You are an expert financial analyst specializing in stock analysis. 
You have access to real-time stock data, technical indicators, and fundamental data.
Provide clear, actionable analysis and recommendations based on the data provided.
Always consider both technical and fundamental factors in your analysis.
Be objective and mention risks and uncertainties."""
    
    def generate_report(self, result: AnalysisResult, format: str = "markdown") -> str:
        """
        Generate a formatted report from analysis result.
        
        Args:
            result: AnalysisResult object
            format: Report format ('markdown', 'html', 'json')
            
        Returns:
            Formatted report string
        """
        return self.report_generator.generate_report(result, format)
    
    def save_report(self, result: AnalysisResult, output_path: Union[str, Path], format: str = "markdown"):
        """
        Save analysis report to file.
        
        Args:
            result: AnalysisResult object
            output_path: Path to save the report
            format: Report format
        """
        report = self.generate_report(result, format)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        self.logger.info(f"Report saved to: {output_path}")
    
    def get_stock_data(self, symbol: str):
        """
        Get stock data for a given symbol.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            StockData object with current stock information
        """
        import asyncio
        try:
            # Run the async method in a sync context
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If we're already in an async context, create a new loop
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, self._get_stock_data_async(symbol))
                    return future.result()
            else:
                return asyncio.run(self._get_stock_data_async(symbol))
        except Exception as e:
            self.logger.error(f"Error getting stock data for {symbol}: {e}")
            raise
    
    async def _get_stock_data_async(self, symbol: str):
        """Async method to get stock data."""
        try:
            # Get current stock data
            stock_data = await self.mcp_client.get_stock_data(symbol)
            
            # Create a simple data structure that matches what the CLI expects
            class SimpleStockData:
                def __init__(self, data):
                    self.current_price = data.get('price', 0)
                    self.change = data.get('change', 0)
                    self.change_percent = data.get('change_percent', 0)
                    self.volume = data.get('volume', 0)
                    self.market_cap = data.get('market_cap', 0)
                    self.high_52_week = data.get('fifty_two_week_high', 0)
                    self.low_52_week = data.get('fifty_two_week_low', 0)
            
            return SimpleStockData(stock_data)
        except Exception as e:
            self.logger.error(f"Error in async stock data fetch: {e}")
            raise
    
    async def close(self):
        """Close connections and cleanup resources."""
        await self.mcp_client.close()
        self.logger.info("Stock Analysis Agent closed") 