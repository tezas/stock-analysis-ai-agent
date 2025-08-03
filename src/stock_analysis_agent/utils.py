"""
Utility functions for the Stock Analysis AI Agent.

This module provides common utilities including logging setup,
configuration loading, and helper functions.
"""

import logging
import os
from pathlib import Path
from typing import Dict, Optional

import yaml
from dotenv import load_dotenv


def setup_logging(name: str, level: str = "INFO") -> logging.Logger:
    """
    Setup logging configuration.
    
    Args:
        name: Logger name
        level: Logging level
        
    Returns:
        Configured logger
    """
    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / "stock_analysis.log"),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(name)


def load_config(config_path: Optional[str] = None) -> Dict:
    """
    Load configuration from file or environment variables.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    config = {}
    
    # Load environment variables
    load_dotenv()
    
    # Default configuration
    config.update({
        'ai_model': os.getenv('AI_MODEL', 'gpt-4'),
        'ai_temperature': float(os.getenv('AI_TEMPERATURE', '0.7')),
        'mcp_server_url': os.getenv('MCP_SERVER_URL', 'http://localhost:8000'),
        'mcp_server_timeout': int(os.getenv('MCP_SERVER_TIMEOUT', '30')),
        'stock_data_cache_ttl': int(os.getenv('STOCK_DATA_CACHE_TTL', '300')),
        'report_format': os.getenv('REPORT_FORMAT', 'markdown'),
        'output_dir': os.getenv('OUTPUT_DIR', 'reports'),
        'log_level': os.getenv('LOG_LEVEL', 'INFO'),
        'log_file': os.getenv('LOG_FILE', 'logs/stock_analysis.log')
    })
    
    # Load from config file if provided
    if config_path and Path(config_path).exists():
        try:
            with open(config_path, 'r') as f:
                file_config = yaml.safe_load(f)
                config.update(file_config)
        except Exception as e:
            logging.warning(f"Could not load config file {config_path}: {e}")
    
    return config


def validate_stock_symbol(symbol: str) -> bool:
    """
    Validate stock symbol format.
    
    Args:
        symbol: Stock symbol to validate
        
    Returns:
        True if valid, False otherwise
    """
    import re
    
    # Basic validation: 1-5 letters, no numbers or special chars
    return bool(re.match(r'^[A-Z]{1,5}$', symbol.upper()))


def format_currency(value: float, currency: str = "USD") -> str:
    """
    Format currency value.
    
    Args:
        value: Numeric value
        currency: Currency code
        
    Returns:
        Formatted currency string
    """
    if currency == "USD":
        return f"${value:,.2f}"
    else:
        return f"{value:,.2f} {currency}"


def format_percentage(value: float) -> str:
    """
    Format percentage value.
    
    Args:
        value: Numeric value (0-1)
        
    Returns:
        Formatted percentage string
    """
    return f"{value:.2%}"


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Safely divide two numbers, returning default if denominator is zero.
    
    Args:
        numerator: Numerator
        denominator: Denominator
        default: Default value if division by zero
        
    Returns:
        Division result or default value
    """
    try:
        return numerator / denominator if denominator != 0 else default
    except (TypeError, ValueError):
        return default


def create_output_directory(output_dir: str = "reports") -> Path:
    """
    Create output directory if it doesn't exist.
    
    Args:
        output_dir: Output directory path
        
    Returns:
        Path object for output directory
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename for safe file system usage.
    
    Args:
        filename: Original filename
        
    Returns:
        Sanitized filename
    """
    import re
    
    # Remove or replace invalid characters
    sanitized = re.sub(r'[<>:"/\\|?*]', '_', filename)
    
    # Remove leading/trailing spaces and dots
    sanitized = sanitized.strip('. ')
    
    # Limit length
    if len(sanitized) > 255:
        sanitized = sanitized[:255]
    
    return sanitized


def get_file_extension(format: str) -> str:
    """
    Get file extension for report format.
    
    Args:
        format: Report format
        
    Returns:
        File extension
    """
    format_map = {
        'markdown': '.md',
        'html': '.html',
        'json': '.json',
        'txt': '.txt'
    }
    
    return format_map.get(format.lower(), '.md')


def validate_analysis_request(symbol: str, analysis_type: str) -> bool:
    """
    Validate analysis request parameters.
    
    Args:
        symbol: Stock symbol
        analysis_type: Type of analysis
        
    Returns:
        True if valid, False otherwise
    """
    # Validate symbol
    if not validate_stock_symbol(symbol):
        return False
    
    # Validate analysis type
    valid_types = [
        'technical', 'fundamental', 'comprehensive', 'short_term',
        'long_term', 'swing_trading', 'day_trading', 'value_investing'
    ]
    
    if analysis_type not in valid_types:
        return False
    
    return True


def calculate_timeframe_days(timeframe: str) -> int:
    """
    Calculate number of days for a given timeframe.
    
    Args:
        timeframe: Timeframe string
        
    Returns:
        Number of days
    """
    timeframe_map = {
        '1d': 1,
        '5d': 5,
        '1mo': 30,
        '3mo': 90,
        '6mo': 180,
        '1y': 365,
        '2y': 730,
        '5y': 1825,
        'max': 3650  # 10 years as max
    }
    
    return timeframe_map.get(timeframe, 365)  # Default to 1 year


def format_timestamp(timestamp) -> str:
    """
    Format timestamp for display.
    
    Args:
        timestamp: Timestamp object
        
    Returns:
        Formatted timestamp string
    """
    try:
        return timestamp.strftime("%Y-%m-%d %H:%M:%S")
    except AttributeError:
        return str(timestamp)


def get_confidence_class(confidence_score: float) -> str:
    """
    Get CSS class for confidence score styling.
    
    Args:
        confidence_score: Confidence score (0-1)
        
    Returns:
        CSS class name
    """
    if confidence_score >= 0.7:
        return "high"
    elif confidence_score >= 0.4:
        return "medium"
    else:
        return "low"


def truncate_text(text: str, max_length: int = 1000) -> str:
    """
    Truncate text to maximum length.
    
    Args:
        text: Text to truncate
        max_length: Maximum length
        
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    
    return text[:max_length-3] + "..." 