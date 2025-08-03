# Usage Guide

This guide covers how to use the Stock Analysis AI Agent effectively.

## Quick Start

### Basic Analysis

The simplest way to analyze a stock:

```bash
# Analyze from text prompt
stock-analysis-agent analyze --prompt "Analyze AAPL stock" --output aapl_report.md

# Analyze from file
stock-analysis-agent analyze --file prompts/analysis_prompts.txt --output report.md

# Analyze from Word document
stock-analysis-agent analyze --document analysis.docx --output report.md
```

### Interactive Mode

For interactive analysis:

```bash
stock-analysis-agent interactive
```

This starts an interactive session where you can enter prompts and get immediate results.

## Command Line Interface

### Main Commands

#### `analyze` - Perform Stock Analysis

```bash
stock-analysis-agent analyze [OPTIONS]
```

**Options:**
- `--prompt, -p TEXT` - Analysis prompt text
- `--file, -f PATH` - Path to prompt file
- `--document, -d PATH` - Path to Word document
- `--output, -o PATH` - Output file path (default: reports/analysis_report.md)
- `--format, -fmt [markdown|html|json]` - Output format (default: markdown)

**Examples:**
```bash
# Basic analysis
stock-analysis-agent analyze --prompt "Analyze TSLA stock"

# From file with custom output
stock-analysis-agent analyze --file my_prompt.txt --output tesla_analysis.html --format html

# From document
stock-analysis-agent analyze --document analysis.docx --output report.md
```

#### `data` - Fetch Stock Data

```bash
stock-analysis-agent data --symbol SYMBOL [OPTIONS]
```

**Options:**
- `--symbol, -s TEXT` - Stock symbol (required)
- `--period, -p TEXT` - Data period (default: 1y)

**Examples:**
```bash
# Get current data for AAPL
stock-analysis-agent data --symbol AAPL

# Get 6-month data for TSLA
stock-analysis-agent data --symbol TSLA --period 6mo
```

#### `interactive` - Interactive Mode

```bash
stock-analysis-agent interactive
```

Starts an interactive session for real-time analysis.

#### `test-mcp` - Test MCP Connection

```bash
stock-analysis-agent test-mcp [OPTIONS]
```

**Options:**
- `--host, -h TEXT` - MCP server host (default: localhost)
- `--port, -p INTEGER` - MCP server port (default: 8000)

**Example:**
```bash
stock-analysis-agent test-mcp --host localhost --port 8000
```

### Global Options

- `--config, -c PATH` - Configuration file path
- `--verbose, -v` - Enable verbose logging
- `--version` - Show version and exit

## Python API Usage

### Basic Usage

```python
from stock_analysis_agent import StockAnalysisAgent

# Initialize the agent
agent = StockAnalysisAgent(
    mcp_server_url="http://localhost:8000",
    ai_model="gpt-4",
    api_key="your_openai_api_key"
)

# Analyze from text
result = agent.analyze_from_text("Analyze AAPL stock for the next 30 days")

# Generate report
report = agent.generate_report(result)
print(report)
```

### Advanced Usage

```python
from stock_analysis_agent import StockAnalysisAgent
from stock_analysis_agent.utils import AnalysisResult

# Initialize with custom configuration
agent = StockAnalysisAgent(
    mcp_server_url="http://localhost:8000",
    ai_model="gpt-4",
    api_key="your_openai_api_key",
    cache_ttl=600,  # 10 minutes cache
    timeout=60
)

# Analyze from file
result = agent.analyze_from_file("prompts/technical_analysis.txt")

# Analyze from Word document
result = agent.analyze_from_document("analysis.docx")

# Get stock data
stock_data = agent.get_stock_data("AAPL")
print(f"Current price: ${stock_data.current_price}")

# Generate different report formats
markdown_report = agent.generate_report(result, format="markdown")
html_report = agent.generate_report(result, format="html")
json_report = agent.generate_report(result, format="json")

# Clear cache
agent.clear_cache()
```

## Prompt Examples

### Technical Analysis

```bash
# Basic technical analysis
stock-analysis-agent analyze --prompt "Analyze AAPL using RSI, MACD, and Bollinger Bands"

# Support and resistance
stock-analysis-agent analyze --prompt "Find support and resistance levels for TSLA"

# Moving averages
stock-analysis-agent analyze --prompt "Analyze MSFT with 20, 50, and 200-day moving averages"
```

### Fundamental Analysis

```bash
# P/E ratio analysis
stock-analysis-agent analyze --prompt "Analyze NVDA's P/E ratio and valuation"

# Financial health
stock-analysis-agent analyze --prompt "Evaluate JPM's financial health and debt levels"

# Growth analysis
stock-analysis-agent analyze --prompt "Analyze AMZN's revenue growth and market position"
```

### Comprehensive Analysis

```bash
# Complete analysis
stock-analysis-agent analyze --prompt "Provide complete analysis of AAPL including technical and fundamental factors"

# Long-term investment
stock-analysis-agent analyze --prompt "Analyze TSLA for long-term investment potential (5+ years)"

# Trading strategy
stock-analysis-agent analyze --prompt "Generate swing trading strategy for MSFT"
```

### Sector Analysis

```bash
# Sector comparison
stock-analysis-agent analyze --prompt "Compare AAPL, MSFT, and GOOGL in the technology sector"

# Industry analysis
stock-analysis-agent analyze --prompt "Analyze the banking sector with focus on JPM, BAC, and WFC"
```

## Report Formats

### Markdown (Default)

```bash
stock-analysis-agent analyze --prompt "Analyze AAPL" --format markdown --output report.md
```

Produces a well-formatted markdown report with sections for:
- Executive Summary
- Technical Analysis
- Fundamental Analysis
- Risk Assessment
- Recommendations

### HTML

```bash
stock-analysis-agent analyze --prompt "Analyze AAPL" --format html --output report.html
```

Produces an HTML report with styling and charts.

### JSON

```bash
stock-analysis-agent analyze --prompt "Analyze AAPL" --format json --output report.json
```

Produces structured JSON data for programmatic use.

## Configuration

### Environment Variables

Set these in your `.env` file:

```env
# Required
OPENAI_API_KEY=your_api_key_here

# Optional
AI_MODEL=gpt-4
AI_TEMPERATURE=0.7
MCP_SERVER_URL=http://localhost:8000
REPORT_FORMAT=markdown
LOG_LEVEL=INFO
```

### Configuration Files

Create `config/config.json`:

```json
{
  "ai_model": "gpt-4",
  "mcp_server_url": "http://localhost:8000",
  "report_format": "markdown",
  "cache_ttl": 300,
  "timeout": 30
}
```

## Best Practices

### Writing Effective Prompts

1. **Be Specific**: Instead of "Analyze AAPL", use "Analyze AAPL stock for swing trading with technical indicators"

2. **Include Timeframe**: Specify your investment horizon (day trading, swing trading, long-term)

3. **Mention Indicators**: Specify which technical indicators you want analyzed

4. **Ask for Recommendations**: Request specific entry/exit points and stop-loss levels

### Example Effective Prompts

```bash
# Good: Specific and actionable
stock-analysis-agent analyze --prompt "Analyze AAPL for swing trading over the next 2 weeks using RSI, MACD, and support/resistance levels. Provide entry points, exit targets, and stop-loss recommendations."

# Good: Comprehensive analysis
stock-analysis-agent analyze --prompt "Provide a complete analysis of TSLA including technical indicators, fundamental metrics, risk factors, and long-term growth potential. Include specific price targets and confidence levels."

# Good: Sector comparison
stock-analysis-agent analyze --prompt "Compare NVDA, AMD, and INTC in the semiconductor sector. Analyze their competitive positions, growth prospects, and investment potential. Rank them by risk-adjusted returns."
```

### Managing Output

1. **Use Descriptive Filenames**: Include symbol and date in output files
2. **Organize Reports**: Create separate directories for different types of analysis
3. **Version Control**: Keep track of analysis versions

```bash
# Create organized output structure
mkdir -p reports/{technical,fundamental,comprehensive}
mkdir -p reports/$(date +%Y-%m-%d)

# Run analysis with organized output
stock-analysis-agent analyze \
  --prompt "Technical analysis of AAPL" \
  --output "reports/technical/aapl_$(date +%Y%m%d).md"
```

## Troubleshooting

### Common Issues

1. **MCP Server Not Running**
   ```bash
   # Start MCP server
   python scripts/setup_mcp_server.py --start-server
   ```

2. **API Key Issues**
   ```bash
   # Verify API key
   echo $OPENAI_API_KEY
   ```

3. **Import Errors**
   ```bash
   # Ensure virtual environment is active
   source venv/bin/activate
   ```

4. **Slow Performance**
   ```bash
   # Use caching
   # Increase timeout in configuration
   # Use more specific prompts
   ```

### Getting Help

```bash
# Show help
stock-analysis-agent --help
stock-analysis-agent analyze --help

# Verbose logging
stock-analysis-agent analyze --prompt "Analyze AAPL" --verbose

# Check logs
tail -f logs/stock_analysis.log
```

## Advanced Features

### Custom MCP Servers

If you have a custom MCP server:

```python
agent = StockAnalysisAgent(
    mcp_server_url="ws://your-server:9000",
    ai_model="gpt-4",
    api_key="your_key"
)
```

### Batch Processing

For multiple analyses:

```python
symbols = ["AAPL", "TSLA", "MSFT", "GOOGL"]
results = []

for symbol in symbols:
    result = agent.analyze_from_text(f"Analyze {symbol} stock")
    results.append(result)
    
    # Save individual reports
    report = agent.generate_report(result)
    with open(f"reports/{symbol}_analysis.md", "w") as f:
        f.write(report)
```

### Custom Report Templates

You can create custom report templates by modifying the report generator or using the JSON format for custom processing. 