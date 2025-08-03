# API Reference

This document provides detailed API reference for the Stock Analysis AI Agent.

## Core Classes

### StockAnalysisAgent

The main class for performing stock analysis operations.

```python
from stock_analysis_agent import StockAnalysisAgent
```

#### Constructor

```python
StockAnalysisAgent(
    mcp_server_url: str = "http://localhost:8000",
    ai_model: str = "gpt-4",
    api_key: str = None,
    cache_ttl: int = 300,
    timeout: int = 30
)
```

**Parameters:**
- `mcp_server_url` (str): URL of the MCP server for stock data
- `ai_model` (str): AI model to use for analysis (default: "gpt-4")
- `api_key` (str): OpenAI API key (if None, reads from environment)
- `cache_ttl` (int): Cache time-to-live in seconds (default: 300)
- `timeout` (int): Request timeout in seconds (default: 30)

#### Methods

##### `analyze_from_text(prompt: str) -> AnalysisResult`

Analyze a stock based on a text prompt.

**Parameters:**
- `prompt` (str): Analysis prompt text

**Returns:**
- `AnalysisResult`: Analysis result object

**Example:**
```python
result = agent.analyze_from_text("Analyze AAPL stock for the next 30 days")
print(f"Symbol: {result.symbol}")
print(f"Recommendation: {result.recommendation}")
```

##### `analyze_from_file(file_path: str) -> AnalysisResult`

Analyze a stock based on a prompt file.

**Parameters:**
- `file_path` (str): Path to the prompt file

**Returns:**
- `AnalysisResult`: Analysis result object

**Example:**
```python
result = agent.analyze_from_file("prompts/analysis.txt")
```

##### `analyze_from_document(document_path: str) -> AnalysisResult`

Analyze a stock based on a Word document.

**Parameters:**
- `document_path` (str): Path to the Word document

**Returns:**
- `AnalysisResult`: Analysis result object

**Example:**
```python
result = agent.analyze_from_document("analysis.docx")
```

##### `generate_report(result: AnalysisResult, format: str = "markdown") -> str`

Generate a report from analysis results.

**Parameters:**
- `result` (AnalysisResult): Analysis result object
- `format` (str): Output format ("markdown", "html", "json")

**Returns:**
- `str`: Generated report

**Example:**
```python
report = agent.generate_report(result, format="html")
with open("report.html", "w") as f:
    f.write(report)
```

##### `get_stock_data(symbol: str) -> StockData`

Get current stock data for a symbol.

**Parameters:**
- `symbol` (str): Stock symbol

**Returns:**
- `StockData`: Stock data object

**Example:**
```python
stock_data = agent.get_stock_data("AAPL")
print(f"Current price: ${stock_data.current_price}")
print(f"Volume: {stock_data.volume}")
```

##### `clear_cache() -> None`

Clear the analysis cache.

**Example:**
```python
agent.clear_cache()
```

### MCPClient

Client for communicating with MCP servers.

```python
from stock_analysis_agent import MCPClient
```

#### Constructor

```python
MCPClient(
    server_url: str,
    timeout: int = 30
)
```

**Parameters:**
- `server_url` (str): MCP server URL
- `timeout` (int): Request timeout in seconds

#### Methods

##### `connect() -> None`

Connect to the MCP server.

**Example:**
```python
await client.connect()
```

##### `disconnect() -> None`

Disconnect from the MCP server.

**Example:**
```python
await client.disconnect()
```

##### `get_stock_price(symbol: str) -> float`

Get current stock price.

**Parameters:**
- `symbol` (str): Stock symbol

**Returns:**
- `float`: Current stock price

**Example:**
```python
price = await client.get_stock_price("AAPL")
```

##### `get_historical_data(symbol: str, period: str = "1y") -> List[Dict]`

Get historical stock data.

**Parameters:**
- `symbol` (str): Stock symbol
- `period` (str): Data period (e.g., "1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max")

**Returns:**
- `List[Dict]`: Historical data points

**Example:**
```python
data = await client.get_historical_data("AAPL", "6mo")
```

##### `get_fundamental_data(symbol: str) -> Dict`

Get fundamental data for a stock.

**Parameters:**
- `symbol` (str): Stock symbol

**Returns:**
- `Dict`: Fundamental data

**Example:**
```python
fundamentals = await client.get_fundamental_data("AAPL")
print(f"P/E Ratio: {fundamentals['pe_ratio']}")
```

##### `get_technical_indicators(symbol: str) -> Dict`

Get technical indicators for a stock.

**Parameters:**
- `symbol` (str): Stock symbol

**Returns:**
- `Dict`: Technical indicators

**Example:**
```python
indicators = await client.get_technical_indicators("AAPL")
print(f"RSI: {indicators['rsi']}")
```

### StockAnalyzer

Class for performing technical and fundamental analysis.

```python
from stock_analysis_agent import StockAnalyzer
```

#### Methods

##### `calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series`

Calculate Relative Strength Index.

**Parameters:**
- `prices` (pd.Series): Price series
- `period` (int): RSI period (default: 14)

**Returns:**
- `pd.Series`: RSI values

**Example:**
```python
rsi = analyzer.calculate_rsi(stock_data['Close'])
```

##### `calculate_macd(prices: pd.Series) -> Dict`

Calculate MACD (Moving Average Convergence Divergence).

**Parameters:**
- `prices` (pd.Series): Price series

**Returns:**
- `Dict`: MACD data with 'macd', 'signal', and 'histogram' keys

**Example:**
```python
macd_data = analyzer.calculate_macd(stock_data['Close'])
```

##### `calculate_bollinger_bands(prices: pd.Series, period: int = 20, std_dev: float = 2) -> Dict`

Calculate Bollinger Bands.

**Parameters:**
- `prices` (pd.Series): Price series
- `period` (int): Moving average period (default: 20)
- `std_dev` (float): Standard deviation multiplier (default: 2)

**Returns:**
- `Dict`: Bollinger Bands data with 'upper', 'middle', and 'lower' keys

**Example:**
```python
bb_data = analyzer.calculate_bollinger_bands(stock_data['Close'])
```

##### `analyze_technical_indicators(data: pd.DataFrame) -> Dict`

Perform comprehensive technical analysis.

**Parameters:**
- `data` (pd.DataFrame): Stock data with OHLCV columns

**Returns:**
- `Dict`: Technical analysis results

**Example:**
```python
analysis = analyzer.analyze_technical_indicators(stock_data)
```

##### `analyze_fundamentals(data: Dict) -> Dict`

Perform fundamental analysis.

**Parameters:**
- `data` (Dict): Fundamental data

**Returns:**
- `Dict`: Fundamental analysis results

**Example:**
```python
analysis = analyzer.analyze_fundamentals(fundamental_data)
```

### PromptProcessor

Class for processing and extracting information from prompts.

```python
from stock_analysis_agent import PromptProcessor
```

#### Methods

##### `extract_symbol(prompt: str) -> str`

Extract stock symbol from prompt.

**Parameters:**
- `prompt` (str): Analysis prompt

**Returns:**
- `str`: Extracted stock symbol

**Example:**
```python
symbol = processor.extract_symbol("Analyze AAPL stock")
# Returns: "AAPL"
```

##### `extract_timeframe(prompt: str) -> str`

Extract timeframe from prompt.

**Parameters:**
- `prompt` (str): Analysis prompt

**Returns:**
- `str`: Extracted timeframe

**Example:**
```python
timeframe = processor.extract_timeframe("Analyze AAPL for the next 30 days")
# Returns: "30 days"
```

##### `extract_indicators(prompt: str) -> List[str]`

Extract technical indicators from prompt.

**Parameters:**
- `prompt` (str): Analysis prompt

**Returns:**
- `List[str]`: List of technical indicators

**Example:**
```python
indicators = processor.extract_indicators("Analyze AAPL with RSI and MACD")
# Returns: ["RSI", "MACD"]
```

### ReportGenerator

Class for generating analysis reports.

```python
from stock_analysis_agent import ReportGenerator
```

#### Methods

##### `generate_markdown(result: AnalysisResult) -> str`

Generate markdown report.

**Parameters:**
- `result` (AnalysisResult): Analysis result

**Returns:**
- `str`: Markdown report

**Example:**
```python
report = generator.generate_markdown(result)
```

##### `generate_html(result: AnalysisResult) -> str`

Generate HTML report.

**Parameters:**
- `result` (AnalysisResult): Analysis result

**Returns:**
- `str`: HTML report

**Example:**
```python
report = generator.generate_html(result)
```

##### `generate_json(result: AnalysisResult) -> str`

Generate JSON report.

**Parameters:**
- `result` (AnalysisResult): Analysis result

**Returns:**
- `str`: JSON report

**Example:**
```python
report = generator.generate_json(result)
```

## Data Models

### AnalysisResult

Result of stock analysis.

```python
from stock_analysis_agent.utils import AnalysisResult
```

**Attributes:**
- `symbol` (str): Stock symbol
- `summary` (str): Analysis summary
- `recommendation` (str): Trading recommendation (BUY, SELL, HOLD)
- `confidence` (float): Confidence level (0.0 to 1.0)
- `risks` (List[str]): List of identified risks
- `opportunities` (List[str]): List of identified opportunities
- `technical_analysis` (Dict): Technical analysis results
- `fundamental_analysis` (Dict): Fundamental analysis results
- `timestamp` (datetime): Analysis timestamp

**Example:**
```python
result = AnalysisResult(
    symbol="AAPL",
    summary="Strong buy recommendation based on technical and fundamental analysis",
    recommendation="BUY",
    confidence=0.85,
    risks=["Market volatility", "Supply chain issues"],
    opportunities=["New product launches", "Market expansion"],
    technical_analysis={...},
    fundamental_analysis={...}
)
```

### StockData

Stock data information.

```python
from stock_analysis_agent.utils import StockData
```

**Attributes:**
- `symbol` (str): Stock symbol
- `current_price` (float): Current stock price
- `change` (float): Price change
- `change_percent` (float): Percentage change
- `volume` (int): Trading volume
- `market_cap` (float): Market capitalization
- `pe_ratio` (float): Price-to-earnings ratio
- `dividend_yield` (float): Dividend yield
- `timestamp` (datetime): Data timestamp

**Example:**
```python
stock_data = StockData(
    symbol="AAPL",
    current_price=150.25,
    change=2.50,
    change_percent=1.69,
    volume=50000000,
    market_cap=2500000000000,
    pe_ratio=25.5,
    dividend_yield=0.5
)
```

## Utility Functions

### Configuration

```python
from stock_analysis_agent.utils import load_config, setup_logging
```

#### `load_config(config_path: str = None) -> Dict`

Load configuration from file or environment.

**Parameters:**
- `config_path` (str): Path to configuration file (optional)

**Returns:**
- `Dict`: Configuration dictionary

**Example:**
```python
config = load_config("config/config.json")
```

#### `setup_logging(level: str = "INFO", log_file: str = None) -> None`

Setup logging configuration.

**Parameters:**
- `level` (str): Logging level
- `log_file` (str): Log file path (optional)

**Example:**
```python
setup_logging(level="DEBUG", log_file="logs/app.log")
```

### Document Processing

```python
from stock_analysis_agent.utils import extract_text_from_docx
```

#### `extract_text_from_docx(file_path: str) -> str`

Extract text from Word document.

**Parameters:**
- `file_path` (str): Path to Word document

**Returns:**
- `str`: Extracted text

**Example:**
```python
text = extract_text_from_docx("analysis.docx")
```

### Data Validation

```python
from stock_analysis_agent.utils import validate_symbol, validate_timeframe
```

#### `validate_symbol(symbol: str) -> bool`

Validate stock symbol format.

**Parameters:**
- `symbol` (str): Stock symbol

**Returns:**
- `bool`: True if valid, False otherwise

**Example:**
```python
if validate_symbol("AAPL"):
    print("Valid symbol")
```

#### `validate_timeframe(timeframe: str) -> bool`

Validate timeframe format.

**Parameters:**
- `timeframe` (str): Timeframe string

**Returns:**
- `bool`: True if valid, False otherwise

**Example:**
```python
if validate_timeframe("30 days"):
    print("Valid timeframe")
```

## Error Handling

### Custom Exceptions

```python
from stock_analysis_agent.utils import (
    MCPConnectionError,
    AnalysisError,
    ValidationError,
    ConfigurationError
)
```

#### `MCPConnectionError`

Raised when MCP server connection fails.

#### `AnalysisError`

Raised when analysis fails.

#### `ValidationError`

Raised when input validation fails.

#### `ConfigurationError`

Raised when configuration is invalid.

### Error Handling Example

```python
from stock_analysis_agent import StockAnalysisAgent
from stock_analysis_agent.utils import MCPConnectionError, AnalysisError

try:
    agent = StockAnalysisAgent()
    result = agent.analyze_from_text("Analyze AAPL stock")
except MCPConnectionError as e:
    print(f"MCP connection failed: {e}")
except AnalysisError as e:
    print(f"Analysis failed: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

## Async Support

All MCP client methods are async and should be awaited:

```python
import asyncio
from stock_analysis_agent import MCPClient

async def main():
    client = MCPClient("ws://localhost:8000")
    await client.connect()
    
    price = await client.get_stock_price("AAPL")
    data = await client.get_historical_data("AAPL", "1y")
    
    await client.disconnect()

# Run async function
asyncio.run(main())
```

## Performance Optimization

### Caching

The agent includes built-in caching to improve performance:

```python
# Configure cache TTL
agent = StockAnalysisAgent(cache_ttl=600)  # 10 minutes

# Clear cache when needed
agent.clear_cache()
```

### Batch Processing

For multiple analyses, use batch processing:

```python
symbols = ["AAPL", "TSLA", "MSFT"]
results = []

for symbol in symbols:
    result = agent.analyze_from_text(f"Analyze {symbol}")
    results.append(result)
```

## Integration Examples

### Flask Web Application

```python
from flask import Flask, request, jsonify
from stock_analysis_agent import StockAnalysisAgent

app = Flask(__name__)
agent = StockAnalysisAgent()

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.json
    prompt = data.get('prompt')
    
    try:
        result = agent.analyze_from_text(prompt)
        report = agent.generate_report(result)
        
        return jsonify({
            'success': True,
            'symbol': result.symbol,
            'recommendation': result.recommendation,
            'report': report
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
```

### Jupyter Notebook

```python
import pandas as pd
from stock_analysis_agent import StockAnalysisAgent

# Initialize agent
agent = StockAnalysisAgent()

# Analyze multiple stocks
symbols = ["AAPL", "TSLA", "MSFT", "GOOGL"]
results = []

for symbol in symbols:
    result = agent.analyze_from_text(f"Analyze {symbol}")
    results.append({
        'symbol': result.symbol,
        'recommendation': result.recommendation,
        'confidence': result.confidence
    })

# Create DataFrame
df = pd.DataFrame(results)
print(df)
```

### Scheduled Analysis

```python
import schedule
import time
from stock_analysis_agent import StockAnalysisAgent

def daily_analysis():
    agent = StockAnalysisAgent()
    result = agent.analyze_from_text("Analyze AAPL stock")
    report = agent.generate_report(result)
    
    # Save report with timestamp
    timestamp = time.strftime("%Y%m%d")
    with open(f"reports/aapl_{timestamp}.md", "w") as f:
        f.write(report)

# Schedule daily analysis at 9 AM
schedule.every().day.at("09:00").do(daily_analysis)

while True:
    schedule.run_pending()
    time.sleep(60)
``` 