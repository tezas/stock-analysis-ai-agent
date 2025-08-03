#!/usr/bin/env python3
"""
Main execution script for Stock Analysis AI Agent.
"""

import argparse
import sys
import os
from pathlib import Path
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from stock_analysis_agent import StockAnalysisAgent
from stock_analysis_agent.utils import setup_logging, load_config


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Stock Analysis AI Agent - Intelligent stock analysis using MCP servers"
    )
    
    parser.add_argument(
        "--prompt", "-p",
        help="Analysis prompt text"
    )
    
    parser.add_argument(
        "--file", "-f",
        type=str,
        help="Path to prompt file"
    )
    
    parser.add_argument(
        "--document", "-d",
        type=str,
        help="Path to Word document"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="reports/analysis_report.md",
        help="Output file path (default: reports/analysis_report.md)"
    )
    
    parser.add_argument(
        "--format", "-fmt",
        choices=["markdown", "html", "json"],
        default="markdown",
        help="Output format (default: markdown)"
    )
    
    parser.add_argument(
        "--config", "-c",
        type=str,
        help="Configuration file path"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Run in interactive mode"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = "DEBUG" if args.verbose else "INFO"
    setup_logging(level=log_level)
    
    # Load configuration
    config = load_config(args.config) if args.config else load_config()
    
    # Create output directory if it doesn't exist
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        # Initialize agent
        agent = StockAnalysisAgent(
            mcp_server_url=config.get("mcp_server_url", "http://localhost:8000"),
            ai_model=config.get("ai_model", "gpt-4"),
            api_key=config.get("openai_api_key")
        )
        
        if args.interactive:
            run_interactive_mode(agent)
        else:
            run_analysis(agent, args, output_path)
            
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)


def run_analysis(agent, args, output_path):
    """Run the analysis based on provided arguments."""
    print("Stock Analysis AI Agent")
    print("=" * 50)
    
    # Determine input source
    if args.prompt:
        print(f"Analyzing from prompt: {args.prompt[:50]}...")
        result = agent.analyze_from_text(args.prompt)
    elif args.file:
        print(f"Analyzing from file: {args.file}")
        result = agent.analyze_from_file(args.file)
    elif args.document:
        print(f"Analyzing from document: {args.document}")
        result = agent.analyze_from_document(args.document)
    else:
        print("Error: Must provide either --prompt, --file, or --document")
        sys.exit(1)
    
    # Generate report
    print("Generating report...")
    report = agent.generate_report(result, format=args.format)
    
    # Save report
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"Analysis complete! Report saved to: {output_path}")
    print(f"Symbol analyzed: {result.symbol}")
    print(f"Recommendation: {result.recommendation}")
    print(f"Confidence: {result.confidence:.2%}")


def run_interactive_mode(agent):
    """Run the agent in interactive mode."""
    print("Stock Analysis AI Agent - Interactive Mode")
    print("Type 'quit' to exit")
    print("=" * 50)
    
    while True:
        try:
            prompt = input("\nEnter your analysis prompt: ").strip()
            
            if prompt.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if not prompt:
                continue
            
            print("Analyzing...")
            result = agent.analyze_from_text(prompt)
            report = agent.generate_report(result)
            
            print("\n" + "=" * 50)
            print(report)
            print("=" * 50)
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {str(e)}")


if __name__ == "__main__":
    main() 