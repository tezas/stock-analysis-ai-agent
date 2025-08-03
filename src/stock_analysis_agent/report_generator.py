"""
Report Generator

This module generates formatted reports from stock analysis results.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd


class ReportGenerator:
    """
    Generate formatted reports from stock analysis results.
    
    This class creates reports in various formats including
    Markdown, HTML, and JSON.
    """
    
    def __init__(self):
        """Initialize the report generator."""
        self.logger = logging.getLogger(__name__)
        
        # Load templates
        self.templates = self._load_templates()
    
    def _load_templates(self) -> Dict:
        """Load report templates."""
        return {
            'markdown': self._get_markdown_template(),
            'html': self._get_html_template(),
            'json': None  # JSON doesn't need a template
        }
    
    def generate_report(self, result, format: str = "markdown") -> str:
        """
        Generate a formatted report from analysis result.
        
        Args:
            result: AnalysisResult object
            format: Report format ('markdown', 'html', 'json')
            
        Returns:
            Formatted report string
        """
        try:
            if format.lower() == "json":
                return self._generate_json_report(result)
            elif format.lower() == "html":
                return self._generate_html_report(result)
            else:
                return self._generate_markdown_report(result)
                
        except Exception as e:
            self.logger.error(f"Error generating {format} report: {str(e)}")
            return f"Error generating report: {str(e)}"
    
    def _generate_markdown_report(self, result) -> str:
        """Generate Markdown report."""
        try:
            template = self.templates['markdown']
            
            # Format technical indicators
            tech_indicators = self._format_technical_indicators_md(result.technical_indicators)
            
            # Format fundamental data
            fundamental_data = self._format_fundamental_data_md(result.fundamental_data)
            
            # Format recommendations
            recommendations = self._format_recommendations_md(result.recommendations)
            
            # Fill template
            report = template.format(
                symbol=result.symbol,
                analysis_type=result.analysis_type,
                timestamp=result.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                confidence_score=f"{result.confidence_score:.1%}",
                risk_assessment=result.risk_assessment,
                current_price=f"${result.stock_data.get('price', 'N/A')}",
                volume=f"{result.stock_data.get('volume', 'N/A'):,}",
                market_cap=f"${result.stock_data.get('market_cap', 'N/A'):,}" if result.stock_data.get('market_cap') else 'N/A',
                technical_indicators=tech_indicators,
                fundamental_data=fundamental_data,
                ai_analysis=result.ai_analysis,
                recommendations=recommendations
            )
            
            return report
            
        except Exception as e:
            self.logger.error(f"Error generating Markdown report: {str(e)}")
            return f"# Stock Analysis Report\n\nError generating report: {str(e)}"
    
    def _generate_html_report(self, result) -> str:
        """Generate HTML report."""
        try:
            template = self.templates['html']
            
            # Format technical indicators
            tech_indicators = self._format_technical_indicators_html(result.technical_indicators)
            
            # Format fundamental data
            fundamental_data = self._format_fundamental_data_html(result.fundamental_data)
            
            # Format recommendations
            recommendations = self._format_recommendations_html(result.recommendations)
            
            # Fill template
            report = template.format(
                symbol=result.symbol,
                analysis_type=result.analysis_type,
                timestamp=result.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                confidence_score=f"{result.confidence_score:.1%}",
                risk_assessment=result.risk_assessment,
                current_price=f"${result.stock_data.get('price', 'N/A')}",
                volume=f"{result.stock_data.get('volume', 'N/A'):,}",
                market_cap=f"${result.stock_data.get('market_cap', 'N/A'):,}" if result.stock_data.get('market_cap') else 'N/A',
                technical_indicators=tech_indicators,
                fundamental_data=fundamental_data,
                ai_analysis=result.ai_analysis.replace('\n', '<br>'),
                recommendations=recommendations
            )
            
            return report
            
        except Exception as e:
            self.logger.error(f"Error generating HTML report: {str(e)}")
            return f"<html><body><h1>Stock Analysis Report</h1><p>Error generating report: {str(e)}</p></body></html>"
    
    def _generate_json_report(self, result) -> str:
        """Generate JSON report."""
        try:
            # Convert result to dictionary
            report_data = {
                'symbol': result.symbol,
                'analysis_type': result.analysis_type,
                'timestamp': result.timestamp.isoformat(),
                'confidence_score': result.confidence_score,
                'risk_assessment': result.risk_assessment,
                'stock_data': result.stock_data,
                'technical_indicators': result.technical_indicators,
                'fundamental_data': result.fundamental_data,
                'ai_analysis': result.ai_analysis,
                'recommendations': result.recommendations
            }
            
            return json.dumps(report_data, indent=2, default=str)
            
        except Exception as e:
            self.logger.error(f"Error generating JSON report: {str(e)}")
            return json.dumps({'error': str(e)})
    
    def _format_technical_indicators_md(self, indicators: Dict) -> str:
        """Format technical indicators for Markdown."""
        if not indicators:
            return "No technical indicators available."
        
        lines = []
        for indicator, value in indicators.items():
            if isinstance(value, (int, float)):
                if 'price' in indicator.lower() or 'sma' in indicator.lower() or 'ema' in indicator.lower():
                    lines.append(f"- **{indicator.replace('_', ' ').title()}**: ${value:.2f}")
                elif 'ratio' in indicator.lower() or 'percent' in indicator.lower():
                    lines.append(f"- **{indicator.replace('_', ' ').title()}**: {value:.2%}")
                else:
                    lines.append(f"- **{indicator.replace('_', ' ').title()}**: {value:.4f}")
            elif isinstance(value, bool):
                lines.append(f"- **{indicator.replace('_', ' ').title()}**: {'Yes' if value else 'No'}")
            else:
                lines.append(f"- **{indicator.replace('_', ' ').title()}**: {value}")
        
        return '\n'.join(lines)
    
    def _format_fundamental_data_md(self, data: Dict) -> str:
        """Format fundamental data for Markdown."""
        if not data:
            return "No fundamental data available."
        
        lines = []
        for key, value in data.items():
            if isinstance(value, (int, float)):
                if 'ratio' in key.lower() or 'yield' in key.lower() or 'growth' in key.lower():
                    lines.append(f"- **{key.replace('_', ' ').title()}**: {value:.2%}")
                elif 'price' in key.lower() or 'value' in key.lower() or 'cap' in key.lower():
                    lines.append(f"- **{key.replace('_', ' ').title()}**: ${value:,.0f}")
                else:
                    lines.append(f"- **{key.replace('_', ' ').title()}**: {value:.2f}")
            else:
                lines.append(f"- **{key.replace('_', ' ').title()}**: {value}")
        
        return '\n'.join(lines)
    
    def _format_recommendations_md(self, recommendations: List[str]) -> str:
        """Format recommendations for Markdown."""
        if not recommendations:
            return "No specific recommendations available."
        
        return '\n'.join([f"- {rec}" for rec in recommendations])
    
    def _format_technical_indicators_html(self, indicators: Dict) -> str:
        """Format technical indicators for HTML."""
        if not indicators:
            return "<p>No technical indicators available.</p>"
        
        lines = ['<ul>']
        for indicator, value in indicators.items():
            if isinstance(value, (int, float)):
                if 'price' in indicator.lower() or 'sma' in indicator.lower() or 'ema' in indicator.lower():
                    lines.append(f"<li><strong>{indicator.replace('_', ' ').title()}</strong>: ${value:.2f}</li>")
                elif 'ratio' in indicator.lower() or 'percent' in indicator.lower():
                    lines.append(f"<li><strong>{indicator.replace('_', ' ').title()}</strong>: {value:.2%}</li>")
                else:
                    lines.append(f"<li><strong>{indicator.replace('_', ' ').title()}</strong>: {value:.4f}</li>")
            elif isinstance(value, bool):
                lines.append(f"<li><strong>{indicator.replace('_', ' ').title()}</strong>: {'Yes' if value else 'No'}</li>")
            else:
                lines.append(f"<li><strong>{indicator.replace('_', ' ').title()}</strong>: {value}</li>")
        lines.append('</ul>')
        
        return '\n'.join(lines)
    
    def _format_fundamental_data_html(self, data: Dict) -> str:
        """Format fundamental data for HTML."""
        if not data:
            return "<p>No fundamental data available.</p>"
        
        lines = ['<ul>']
        for key, value in data.items():
            if isinstance(value, (int, float)):
                if 'ratio' in key.lower() or 'yield' in key.lower() or 'growth' in key.lower():
                    lines.append(f"<li><strong>{key.replace('_', ' ').title()}</strong>: {value:.2%}</li>")
                elif 'price' in key.lower() or 'value' in key.lower() or 'cap' in key.lower():
                    lines.append(f"<li><strong>{key.replace('_', ' ').title()}</strong>: ${value:,.0f}</li>")
                else:
                    lines.append(f"<li><strong>{key.replace('_', ' ').title()}</strong>: {value:.2f}</li>")
            else:
                lines.append(f"<li><strong>{key.replace('_', ' ').title()}</strong>: {value}</li>")
        lines.append('</ul>')
        
        return '\n'.join(lines)
    
    def _format_recommendations_html(self, recommendations: List[str]) -> str:
        """Format recommendations for HTML."""
        if not recommendations:
            return "<p>No specific recommendations available.</p>"
        
        lines = ['<ul>']
        for rec in recommendations:
            lines.append(f"<li>{rec}</li>")
        lines.append('</ul>')
        
        return '\n'.join(lines)
    
    def _get_markdown_template(self) -> str:
        """Get Markdown report template."""
        return """# Stock Analysis Report: {symbol}

**Analysis Type:** {analysis_type}  
**Generated:** {timestamp}  
**Confidence Score:** {confidence_score}  
**Risk Assessment:** {risk_assessment}

## Current Stock Data

- **Current Price:** {current_price}
- **Volume:** {volume}
- **Market Cap:** {market_cap}

## Technical Analysis

{technical_indicators}

## Fundamental Analysis

{fundamental_data}

## AI Analysis

{ai_analysis}

## Investment Recommendations

{recommendations}

---

*This report was generated by the Stock Analysis AI Agent. Please conduct your own research before making investment decisions.*
"""
    
    def _get_html_template(self) -> str:
        """Get HTML report template."""
        return """<!DOCTYPE html>
<html>
<head>
    <title>Stock Analysis Report: {symbol}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
        h1 {{ color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
        h2 {{ color: #34495e; margin-top: 30px; }}
        .header-info {{ background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin: 20px 0; }}
        .metric {{ margin: 5px 0; }}
        .confidence-high {{ color: #27ae60; }}
        .confidence-medium {{ color: #f39c12; }}
        .confidence-low {{ color: #e74c3c; }}
        ul {{ margin: 10px 0; }}
        li {{ margin: 5px 0; }}
        .ai-analysis {{ background-color: #f8f9fa; padding: 15px; border-left: 4px solid #3498db; margin: 20px 0; }}
        .recommendations {{ background-color: #e8f5e8; padding: 15px; border-radius: 5px; }}
        .footer {{ margin-top: 40px; padding-top: 20px; border-top: 1px solid #ddd; color: #666; font-size: 0.9em; }}
    </style>
</head>
<body>
    <h1>Stock Analysis Report: {symbol}</h1>
    
    <div class="header-info">
        <div class="metric"><strong>Analysis Type:</strong> {analysis_type}</div>
        <div class="metric"><strong>Generated:</strong> {timestamp}</div>
        <div class="metric"><strong>Confidence Score:</strong> <span class="confidence-{confidence_class}">{confidence_score}</span></div>
        <div class="metric"><strong>Risk Assessment:</strong> {risk_assessment}</div>
    </div>

    <h2>Current Stock Data</h2>
    <ul>
        <li><strong>Current Price:</strong> {current_price}</li>
        <li><strong>Volume:</strong> {volume}</li>
        <li><strong>Market Cap:</strong> {market_cap}</li>
    </ul>

    <h2>Technical Analysis</h2>
    {technical_indicators}

    <h2>Fundamental Analysis</h2>
    {fundamental_data}

    <h2>AI Analysis</h2>
    <div class="ai-analysis">
        {ai_analysis}
    </div>

    <h2>Investment Recommendations</h2>
    <div class="recommendations">
        {recommendations}
    </div>

    <div class="footer">
        <p><em>This report was generated by the Stock Analysis AI Agent. Please conduct your own research before making investment decisions.</em></p>
    </div>
</body>
</html>""" 