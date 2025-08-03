"""
Stock Analysis AI Agent

An intelligent AI agent for stock analysis using MCP servers.
"""

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

from .agent import StockAnalysisAgent
from .mcp_client import MCPClient
from .stock_analyzer import StockAnalyzer
from .prompt_processor import PromptProcessor
from .report_generator import ReportGenerator
from .cli import main

__all__ = [
    "StockAnalysisAgent",
    "MCPClient", 
    "StockAnalyzer",
    "PromptProcessor",
    "ReportGenerator",
    "main",
] 