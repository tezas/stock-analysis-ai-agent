# Stock Analysis AI Agent

An intelligent AI agent that analyzes stocks using MCP (Model Context Protocol) servers to fetch real-time stock data and provide comprehensive analysis based on user prompts.

## Features

- **MCP Integration**: Connects to MCP servers for real-time stock data
- **AI-Powered Analysis**: Uses advanced AI models for stock analysis
- **Prompt-Based Analysis**: Accepts analysis prompts from text files or Word documents
- **Comprehensive Reports**: Generates detailed stock analysis reports
- **Real-time Data**: Fetches current stock prices, historical data, and market indicators
- **Technical Analysis**: Performs technical analysis with various indicators
- **Fundamental Analysis**: Analyzes company fundamentals and financial ratios

## Project Structure

```
stock-analysis-ai-agent/
├── src/
│   └── stock_analysis_agent/
│       ├── __init__.py
│       ├── agent.py              # Main AI agent
│       ├── mcp_client.py         # MCP server client
│       ├── stock_analyzer.py     # Stock analysis logic
│       ├── prompt_processor.py   # Prompt processing
│       ├── report_generator.py   # Report generation
│       └── utils.py              # Utility functions
├── tests/
│   ├── __init__.py
│   ├── test_agent.py
│   ├── test_mcp_client.py
│   └── test_stock_analyzer.py
├── prompts/
│   ├── analysis_prompts.txt      # Sample analysis prompts
│   └── templates/                # Prompt templates
├── examples/
│   ├── sample_analysis.docx      # Sample Word document
│   └── sample_prompt.txt         # Sample text prompt
├── docs/
│   ├── setup.md
│   ├── usage.md
│   └── api.md
├── scripts/
│   ├── setup_mcp_server.py       # MCP server setup
│   └── run_analysis.py           # Main execution script
├── requirements.txt
├── requirements-dev.txt
└── setup.py
```

## Prerequisites

- Python 3.8+
- Git
- MCP server (for stock data)
- OpenAI API key (or other AI model provider)
- Stock data API access (via MCP server)

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd stock-analysis-ai-agent
   ```

2. **Create virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt
   ```

4. **Set up environment variables**:
   ```bash
   cp .env.example .env
   # Edit .env with your API keys and configuration
   ```

5. **Install the package**:
   ```bash
   pip install -e .
   ```

## Quick Start

1. **Set up MCP server** (see MCP Server Setup section)

2. **Create a prompt file**:
   ```bash
   echo "Analyze AAPL stock for the next 30 days with technical indicators" > prompts/my_analysis.txt
   ```

3. **Run analysis**:
   ```bash
   python scripts/run_analysis.py --prompt prompts/my_analysis.txt --output reports/aapl_analysis.md
   ```

## Usage

### Basic Usage

```python
from stock_analysis_agent import StockAnalysisAgent

# Initialize the agent
agent = StockAnalysisAgent(
    mcp_server_url="http://localhost:8000",
    ai_model="gpt-4"
)

# Analyze stock from prompt file
result = agent.analyze_from_file("prompts/analysis_prompt.txt")

# Analyze stock from Word document
result = agent.analyze_from_document("examples/sample_analysis.docx")

# Get analysis report
report = agent.generate_report(result)
print(report)
```

### Command Line Interface

```bash
# Analyze from text prompt
python -m stock_analysis_agent analyze --prompt "Analyze TSLA stock" --output report.md

# Analyze from file
python -m stock_analysis_agent analyze --file prompts/analysis.txt --output report.md

# Analyze from Word document
python -m stock_analysis_agent analyze --document sample.docx --output report.md

# Interactive mode
python -m stock_analysis_agent interactive
```

## MCP Server Setup

The agent requires an MCP server that provides stock data. You can use existing MCP servers or create your own.

### Using Existing MCP Server

1. **Install MCP server**:
   ```bash
   pip install mcp-server-stocks
   ```

2. **Start the server**:
   ```bash
   mcp-server-stocks --port 8000
   ```

### Creating Custom MCP Server

See `docs/mcp_server_setup.md` for detailed instructions on creating a custom MCP server for stock data.

## Configuration

Create a `.env` file with the following variables:

```env
# AI Model Configuration
OPENAI_API_KEY=your_openai_api_key
AI_MODEL=gpt-4
AI_TEMPERATURE=0.7

# MCP Server Configuration
MCP_SERVER_URL=http://localhost:8000
MCP_SERVER_TIMEOUT=30

# Stock Data Configuration
STOCK_DATA_CACHE_TTL=300
STOCK_DATA_SOURCES=yahoo,alpha_vantage

# Report Configuration
REPORT_FORMAT=markdown
REPORT_TEMPLATE=default
OUTPUT_DIR=reports/

# Logging
LOG_LEVEL=INFO
LOG_FILE=logs/stock_analysis.log
```

## API Reference

### StockAnalysisAgent

Main class for stock analysis operations.

#### Methods

- `analyze_from_file(prompt_file: str) -> AnalysisResult`
- `analyze_from_document(document_file: str) -> AnalysisResult`
- `analyze_from_text(prompt_text: str) -> AnalysisResult`
- `generate_report(result: AnalysisResult) -> str`
- `get_stock_data(symbol: str) -> StockData`

### MCPClient

Client for communicating with MCP servers.

#### Methods

- `get_stock_price(symbol: str) -> float`
- `get_historical_data(symbol: str, period: str) -> pd.DataFrame`
- `get_fundamental_data(symbol: str) -> dict`
- `get_technical_indicators(symbol: str) -> dict`

## Development

### Running Tests

```bash
pytest tests/ -v
```

### Code Formatting

```bash
black src/ tests/
flake8 src/ tests/
mypy src/
```

### Building Documentation

```bash
cd docs
make html
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Run the test suite
6. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Support

For support and questions:
- Create an issue on GitHub
- Check the documentation in `docs/`
- Review the examples in `examples/`

## Roadmap

- [ ] Support for multiple AI models
- [ ] Advanced technical indicators
- [ ] Portfolio analysis
- [ ] Real-time alerts
- [ ] Web interface
- [ ] Mobile app
- [ ] Integration with trading platforms 