# Setup Guide

This guide will help you set up the Stock Analysis AI Agent on your system.

## Prerequisites

Before you begin, ensure you have the following installed:

- **Python 3.8 or higher**
- **Git**
- **pip** (Python package installer)

### System Requirements

- **Operating System**: Windows, macOS, or Linux
- **Memory**: At least 4GB RAM (8GB recommended)
- **Storage**: 2GB free space
- **Network**: Internet connection for API access

## Installation Steps

### 1. Clone the Repository

```bash
git clone <repository-url>
cd stock-analysis-ai-agent
```

### 2. Create Virtual Environment

It's recommended to use a virtual environment to avoid conflicts with other Python packages.

**On macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**On Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies

Install the required packages:

```bash
pip install -r requirements.txt
```

For development, also install development dependencies:

```bash
pip install -r requirements-dev.txt
```

### 4. Install the Package

Install the package in development mode:

```bash
pip install -e .
```

### 5. Set Up Environment Variables

Create a `.env` file in the project root:

```bash
cp .env.example .env
```

Edit the `.env` file with your configuration:

```env
# AI Model Configuration
OPENAI_API_KEY=your_openai_api_key_here
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

### 6. Set Up MCP Server

The agent requires an MCP server for stock data. You have two options:

#### Option A: Use Existing MCP Server

```bash
# Install MCP server package
pip install mcp-server-stocks

# Start the server
mcp-server-stocks --port 8000
```

#### Option B: Set Up Custom MCP Server

Use the provided setup script:

```bash
python scripts/setup_mcp_server.py --server-type custom --port 8000 --install-deps
```

### 7. Verify Installation

Test the installation:

```bash
# Test the CLI
stock-analysis-agent --help

# Test MCP connection
stock-analysis-agent test-mcp --host localhost --port 8000

# Run a simple analysis
stock-analysis-agent analyze --prompt "Analyze AAPL stock" --output test_report.md
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | Your OpenAI API key | Required |
| `AI_MODEL` | AI model to use | `gpt-4` |
| `AI_TEMPERATURE` | AI response randomness | `0.7` |
| `MCP_SERVER_URL` | MCP server URL | `http://localhost:8000` |
| `MCP_SERVER_TIMEOUT` | MCP request timeout | `30` |
| `STOCK_DATA_CACHE_TTL` | Cache time-to-live | `300` |
| `REPORT_FORMAT` | Report output format | `markdown` |
| `LOG_LEVEL` | Logging level | `INFO` |

### Configuration Files

The agent supports configuration files in JSON or YAML format:

**config/config.json:**
```json
{
  "ai_model": "gpt-4",
  "mcp_server_url": "http://localhost:8000",
  "report_format": "markdown",
  "cache_ttl": 300
}
```

**config/config.yaml:**
```yaml
ai_model: gpt-4
mcp_server_url: http://localhost:8000
report_format: markdown
cache_ttl: 300
```

## API Keys Setup

### OpenAI API Key

1. Go to [OpenAI Platform](https://platform.openai.com/)
2. Sign up or log in to your account
3. Navigate to API Keys section
4. Create a new API key
5. Copy the key and add it to your `.env` file

### Alpha Vantage API Key (Optional)

For additional stock data sources:

1. Go to [Alpha Vantage](https://www.alphavantage.co/)
2. Sign up for a free API key
3. Add the key to your `.env` file:
   ```env
   ALPHA_VANTAGE_API_KEY=your_key_here
   ```

## Troubleshooting

### Common Issues

#### 1. Import Errors

If you encounter import errors, ensure you're in the virtual environment:

```bash
# Check if virtual environment is active
which python  # Should show path to venv

# Reactivate if needed
source venv/bin/activate  # macOS/Linux
# or
venv\Scripts\activate     # Windows
```

#### 2. MCP Server Connection Issues

If the agent can't connect to the MCP server:

```bash
# Check if server is running
curl http://localhost:8000/health

# Restart the server
python scripts/setup_mcp_server.py --start-server
```

#### 3. API Key Issues

If you get authentication errors:

```bash
# Verify API key is set
echo $OPENAI_API_KEY

# Test API key
python -c "import openai; openai.api_key='your_key'; print('Valid')"
```

#### 4. Permission Issues

On Linux/macOS, you might need to make scripts executable:

```bash
chmod +x scripts/*.py
```

### Getting Help

If you encounter issues:

1. Check the logs in `logs/stock_analysis.log`
2. Run with verbose logging: `--verbose` flag
3. Check the [GitHub Issues](https://github.com/yourusername/stock-analysis-ai-agent/issues)
4. Review the documentation in the `docs/` folder

## Next Steps

After successful setup:

1. Read the [Usage Guide](usage.md)
2. Check out the [API Reference](api.md)
3. Try the examples in the `examples/` folder
4. Explore the prompts in the `prompts/` folder

## Development Setup

For developers who want to contribute:

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install

# Run tests
pytest tests/ -v

# Format code
black src/ tests/
flake8 src/ tests/
mypy src/
``` 