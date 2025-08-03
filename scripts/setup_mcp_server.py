#!/usr/bin/env python3
"""
Setup script for MCP server configuration.
"""

import argparse
import subprocess
import sys
import os
from pathlib import Path


def main():
    """Main setup function."""
    parser = argparse.ArgumentParser(
        description="Setup MCP server for Stock Analysis AI Agent"
    )
    
    parser.add_argument(
        "--server-type",
        choices=["stocks", "custom"],
        default="stocks",
        help="Type of MCP server to setup (default: stocks)"
    )
    
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port for MCP server (default: 8000)"
    )
    
    parser.add_argument(
        "--install-deps",
        action="store_true",
        help="Install required dependencies"
    )
    
    parser.add_argument(
        "--start-server",
        action="store_true",
        help="Start the MCP server after setup"
    )
    
    args = parser.parse_args()
    
    print("MCP Server Setup for Stock Analysis AI Agent")
    print("=" * 50)
    
    if args.install_deps:
        install_dependencies()
    
    if args.server_type == "stocks":
        setup_stocks_server(args.port)
    else:
        setup_custom_server(args.port)
    
    if args.start_server:
        start_server(args.server_type, args.port)


def install_dependencies():
    """Install required dependencies."""
    print("Installing dependencies...")
    
    dependencies = [
        "mcp-server-stocks",
        "websockets",
        "aiohttp",
        "yfinance",
        "pandas",
        "numpy"
    ]
    
    for dep in dependencies:
        try:
            print(f"Installing {dep}...")
            subprocess.run([sys.executable, "-m", "pip", "install", dep], check=True)
            print(f"✅ {dep} installed successfully")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install {dep}: {e}")
            return False
    
    print("✅ All dependencies installed successfully")
    return True


def setup_stocks_server(port):
    """Setup stocks MCP server."""
    print(f"Setting up stocks MCP server on port {port}...")
    
    # Create server configuration
    config_dir = Path("config")
    config_dir.mkdir(exist_ok=True)
    
    config_file = config_dir / "mcp_server_config.json"
    config_content = {
        "server_type": "stocks",
        "port": port,
        "host": "localhost",
        "data_sources": ["yahoo", "alpha_vantage"],
        "cache_ttl": 300,
        "rate_limit": 100
    }
    
    import json
    with open(config_file, 'w') as f:
        json.dump(config_content, f, indent=2)
    
    print(f"✅ Server configuration saved to {config_file}")
    
    # Create startup script
    startup_script = Path("scripts/start_mcp_server.py")
    startup_script.parent.mkdir(exist_ok=True)
    
    script_content = f'''#!/usr/bin/env python3
"""
Start MCP server for stock data.
"""

import asyncio
import json
from pathlib import Path

# Add your MCP server implementation here
# This is a placeholder for the actual server code

async def start_server():
    """Start the MCP server."""
    config_path = Path("config/mcp_server_config.json")
    
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
        
        print(f"Starting MCP server on {config['host']}:{config['port']}")
        print("Server started successfully!")
        
        # Keep server running
        while True:
            await asyncio.sleep(1)

if __name__ == "__main__":
    asyncio.run(start_server())
'''
    
    with open(startup_script, 'w') as f:
        f.write(script_content)
    
    # Make script executable
    os.chmod(startup_script, 0o755)
    
    print(f"✅ Startup script created: {startup_script}")
    print("✅ Stocks MCP server setup complete!")


def setup_custom_server(port):
    """Setup custom MCP server."""
    print(f"Setting up custom MCP server on port {port}...")
    
    # Create server template
    server_template = Path("src/mcp_server_template.py")
    server_template.parent.mkdir(exist_ok=True)
    
    template_content = '''#!/usr/bin/env python3
"""
Custom MCP Server Template for Stock Data.
"""

import asyncio
import json
import websockets
from typing import Dict, Any


class CustomMCPServer:
    """Custom MCP server implementation."""
    
    def __init__(self, host: str = "localhost", port: int = 8000):
        self.host = host
        self.port = port
        self.clients = set()
    
    async def handle_client(self, websocket, path):
        """Handle client connections."""
        self.clients.add(websocket)
        try:
            async for message in websocket:
                await self.process_message(websocket, message)
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            self.clients.remove(websocket)
    
    async def process_message(self, websocket, message: str):
        """Process incoming messages."""
        try:
            data = json.loads(message)
            method = data.get("method")
            params = data.get("params", {})
            request_id = data.get("id")
            
            if method == "get_stock_price":
                result = await self.get_stock_price(params.get("symbol"))
            elif method == "get_historical_data":
                result = await self.get_historical_data(params.get("symbol"), params.get("period"))
            elif method == "get_fundamental_data":
                result = await self.get_fundamental_data(params.get("symbol"))
            else:
                result = {"error": "Method not found"}
            
            response = {
                "id": request_id,
                "result": result
            }
            
            await websocket.send(json.dumps(response))
            
        except json.JSONDecodeError:
            await websocket.send(json.dumps({
                "error": "Invalid JSON"
            }))
    
    async def get_stock_price(self, symbol: str) -> Dict[str, Any]:
        """Get stock price for symbol."""
        # Implement your stock price fetching logic here
        return {"price": 150.0, "symbol": symbol}
    
    async def get_historical_data(self, symbol: str, period: str) -> Dict[str, Any]:
        """Get historical data for symbol."""
        # Implement your historical data fetching logic here
        return {"data": [], "symbol": symbol, "period": period}
    
    async def get_fundamental_data(self, symbol: str) -> Dict[str, Any]:
        """Get fundamental data for symbol."""
        # Implement your fundamental data fetching logic here
        return {"market_cap": 0, "pe_ratio": 0, "symbol": symbol}
    
    async def start(self):
        """Start the server."""
        server = await websockets.serve(
            self.handle_client,
            self.host,
            self.port
        )
        
        print(f"MCP Server started on ws://{self.host}:{self.port}")
        await server.wait_closed()


async def main():
    """Main function."""
    server = CustomMCPServer(port=8000)
    await server.start()


if __name__ == "__main__":
    asyncio.run(main())
'''
    
    with open(server_template, 'w') as f:
        f.write(template_content)
    
    print(f"✅ Custom server template created: {server_template}")
    print("✅ Custom MCP server setup complete!")
    print("📝 Edit the template file to implement your specific data sources")


def start_server(server_type, port):
    """Start the MCP server."""
    print(f"Starting {server_type} MCP server on port {port}...")
    
    try:
        if server_type == "stocks":
            # Start stocks server
            subprocess.run([
                sys.executable, "-m", "mcp_server_stocks",
                "--port", str(port)
            ], check=True)
        else:
            # Start custom server
            subprocess.run([
                sys.executable, "src/mcp_server_template.py"
            ], check=True)
        
        print(f"✅ {server_type} MCP server started successfully on port {port}")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to start server: {e}")
    except KeyboardInterrupt:
        print("\nServer stopped by user")


if __name__ == "__main__":
    main() 