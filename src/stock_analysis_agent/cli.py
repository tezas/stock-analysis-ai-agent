#!/usr/bin/env python3
"""
Command Line Interface for Stock Analysis AI Agent.
"""

import asyncio
import click
import os
import sys
from pathlib import Path
from typing import Optional

from .agent import StockAnalysisAgent
from .utils import setup_logging, load_config


@click.group()
@click.version_option()
@click.option('--config', '-c', type=click.Path(exists=True), help='Configuration file path')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
@click.pass_context
def cli(ctx, config: Optional[str], verbose: bool):
    """Stock Analysis AI Agent - Intelligent stock analysis using MCP servers."""
    ctx.ensure_object(dict)
    
    # Setup logging
    log_level = 'DEBUG' if verbose else 'INFO'
    setup_logging(__name__, level=log_level)
    
    # Load configuration
    if config:
        ctx.obj['config'] = load_config(config)
    else:
        ctx.obj['config'] = load_config()


@cli.command()
@click.option('--prompt', '-p', help='Analysis prompt text')
@click.option('--file', '-f', type=click.Path(exists=True), help='Prompt file path')
@click.option('--document', '-d', type=click.Path(exists=True), help='Word document path')
@click.option('--output', '-o', type=click.Path(), help='Output file path')
@click.option('--format', 'output_format', default='markdown', 
              type=click.Choice(['markdown', 'html', 'json']), help='Output format')
@click.pass_context
def analyze(ctx, prompt: Optional[str], file: Optional[str], 
           document: Optional[str], output: Optional[str], output_format: str):
    """Analyze stocks based on prompts or documents."""
    
    config = ctx.obj['config']
    
    # Initialize agent
    agent = StockAnalysisAgent(
        mcp_server_url=config.get('mcp_server_url', 'http://localhost:8000'),
        ai_model=config.get('ai_model', 'gpt-4'),
        api_key=config.get('openai_api_key')
    )
    
    try:
        # Determine input source
        if prompt:
            result = agent.analyze_from_text(prompt)
        elif file:
            result = agent.analyze_from_file(file)
        elif document:
            result = agent.analyze_from_document(document)
        else:
            click.echo("Error: Must provide either --prompt, --file, or --document", err=True)
            sys.exit(1)
        
        # Generate report
        report = agent.generate_report(result, format=output_format)
        
        # Output result
        if output:
            with open(output, 'w') as f:
                f.write(report)
            click.echo(f"Analysis report saved to: {output}")
        else:
            click.echo(report)
            
    except Exception as e:
        click.echo(f"Error during analysis: {str(e)}", err=True)
        sys.exit(1)


@cli.command()
@click.option('--symbol', '-s', required=True, help='Stock symbol')
@click.option('--period', '-p', default='1y', help='Data period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)')
@click.pass_context
def data(ctx, symbol: str, period: str):
    """Fetch stock data for a given symbol."""
    
    config = ctx.obj['config']
    
    # Initialize agent
    agent = StockAnalysisAgent(
        mcp_server_url=config.get('mcp_server_url', 'http://localhost:8000'),
        ai_model=config.get('ai_model', 'gpt-4')
    )
    
    try:
        # Get stock data
        stock_data = agent.get_stock_data(symbol.upper())
        
        # Display data
        click.echo(f"Stock Data for {symbol.upper()}:")
        click.echo(f"Current Price: ${stock_data.current_price:.2f}")
        click.echo(f"Change: {stock_data.change:.2f} ({stock_data.change_percent:.2f}%)")
        click.echo(f"Volume: {stock_data.volume:,}")
        click.echo(f"Market Cap: ${stock_data.market_cap:,.0f}")
        
    except Exception as e:
        click.echo(f"Error fetching data: {str(e)}", err=True)
        sys.exit(1)


@cli.command()
@click.pass_context
def interactive(ctx):
    """Start interactive analysis mode."""
    
    config = ctx.obj['config']
    
    # Initialize agent
    agent = StockAnalysisAgent(
        mcp_server_url=config.get('mcp_server_url', 'http://localhost:8000'),
        ai_model=config.get('ai_model', 'gpt-4')
    )
    
    click.echo("Stock Analysis AI Agent - Interactive Mode")
    click.echo("Type 'quit' to exit")
    click.echo("-" * 50)
    
    while True:
        try:
            prompt = click.prompt("Enter your analysis prompt")
            
            if prompt.lower() in ['quit', 'exit', 'q']:
                break
            
            if not prompt.strip():
                continue
            
            click.echo("Analyzing...")
            result = agent.analyze_from_text(prompt)
            report = agent.generate_report(result)
            
            click.echo("\n" + "="*50)
            click.echo(report)
            click.echo("="*50 + "\n")
            
        except KeyboardInterrupt:
            click.echo("\nGoodbye!")
            break
        except Exception as e:
            click.echo(f"Error: {str(e)}", err=True)


@cli.command()
@click.option('--port', '-p', default=8000, help='MCP server port')
@click.option('--host', '-h', default='localhost', help='MCP server host')
@click.pass_context
def test_mcp(ctx, port: int, host: str):
    """Test MCP server connection."""
    
    config = ctx.obj['config']
    
    # Initialize agent
    agent = StockAnalysisAgent(
        mcp_server_url=f"http://{host}:{port}",
        ai_model=config.get('ai_model', 'gpt-4')
    )
    
    try:
        # Test connection
        click.echo(f"Testing MCP server connection to {host}:{port}...")
        
        # Try to get some basic data
        test_symbol = "AAPL"
        stock_data = agent.get_stock_data(test_symbol)
        
        click.echo(f"✅ Connection successful!")
        click.echo(f"Test data for {test_symbol}: ${stock_data.current_price:.2f}")
        
    except Exception as e:
        click.echo(f"❌ Connection failed: {str(e)}", err=True)
        sys.exit(1)


def main():
    """Main entry point for the CLI."""
    cli()


if __name__ == '__main__':
    main() 