"""
Tests for the MCP client functionality.
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock
import asyncio
import json

from stock_analysis_agent.mcp_client import MCPClient


class TestMCPClient:
    """Test cases for MCPClient."""
    
    @pytest.fixture
    def mock_websocket(self):
        """Create a mock websocket connection."""
        mock_ws = AsyncMock()
        mock_ws.send = AsyncMock()
        mock_ws.recv = AsyncMock()
        return mock_ws
    
    @pytest.fixture
    def mcp_client(self, mock_websocket):
        """Create an MCPClient instance with mocked websocket."""
        with patch('stock_analysis_agent.mcp_client.websockets.connect', return_value=mock_websocket):
            return MCPClient("ws://localhost:8000")
    
    @pytest.mark.asyncio
    async def test_client_initialization(self, mcp_client):
        """Test MCP client initialization."""
        assert mcp_client.server_url == "ws://localhost:8000"
        assert mcp_client.timeout == 30
    
    @pytest.mark.asyncio
    async def test_connect(self, mcp_client, mock_websocket):
        """Test connecting to MCP server."""
        await mcp_client.connect()
        
        assert mcp_client.websocket == mock_websocket
        mock_websocket.send.assert_called()
    
    @pytest.mark.asyncio
    async def test_disconnect(self, mcp_client, mock_websocket):
        """Test disconnecting from MCP server."""
        mcp_client.websocket = mock_websocket
        await mcp_client.disconnect()
        
        mock_websocket.close.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_send_request(self, mcp_client, mock_websocket):
        """Test sending requests to MCP server."""
        mcp_client.websocket = mock_websocket
        mock_websocket.recv.return_value = json.dumps({
            "id": "1",
            "result": {"data": "test_data"}
        })
        
        response = await mcp_client._send_request("test_method", {"param": "value"})
        
        assert response["result"]["data"] == "test_data"
        mock_websocket.send.assert_called_once()
        mock_websocket.recv.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_stock_price(self, mcp_client, mock_websocket):
        """Test getting stock price."""
        mcp_client.websocket = mock_websocket
        mock_websocket.recv.return_value = json.dumps({
            "id": "1",
            "result": {"price": 150.25}
        })
        
        price = await mcp_client.get_stock_price("AAPL")
        
        assert price == 150.25
        mock_websocket.send.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_historical_data(self, mcp_client, mock_websocket):
        """Test getting historical data."""
        mcp_client.websocket = mock_websocket
        mock_data = {
            "id": "1",
            "result": {
                "data": [
                    {"date": "2023-01-01", "close": 150.0},
                    {"date": "2023-01-02", "close": 151.0}
                ]
            }
        }
        mock_websocket.recv.return_value = json.dumps(mock_data)
        
        data = await mcp_client.get_historical_data("AAPL", "1y")
        
        assert len(data) == 2
        assert data[0]["close"] == 150.0
        mock_websocket.send.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_fundamental_data(self, mcp_client, mock_websocket):
        """Test getting fundamental data."""
        mcp_client.websocket = mock_websocket
        mock_data = {
            "id": "1",
            "result": {
                "market_cap": 2500000000000,
                "pe_ratio": 25.5,
                "dividend_yield": 0.5
            }
        }
        mock_websocket.recv.return_value = json.dumps(mock_data)
        
        data = await mcp_client.get_fundamental_data("AAPL")
        
        assert data["market_cap"] == 2500000000000
        assert data["pe_ratio"] == 25.5
        assert data["dividend_yield"] == 0.5
    
    @pytest.mark.asyncio
    async def test_get_technical_indicators(self, mcp_client, mock_websocket):
        """Test getting technical indicators."""
        mcp_client.websocket = mock_websocket
        mock_data = {
            "id": "1",
            "result": {
                "rsi": 65.5,
                "macd": {"macd": 1.2, "signal": 0.8},
                "bollinger_bands": {"upper": 155.0, "lower": 145.0}
            }
        }
        mock_websocket.recv.return_value = json.dumps(mock_data)
        
        indicators = await mcp_client.get_technical_indicators("AAPL")
        
        assert indicators["rsi"] == 65.5
        assert indicators["macd"]["macd"] == 1.2
        assert indicators["bollinger_bands"]["upper"] == 155.0
    
    @pytest.mark.asyncio
    async def test_connection_error(self):
        """Test handling of connection errors."""
        with patch('stock_analysis_agent.mcp_client.websockets.connect', side_effect=Exception("Connection failed")):
            client = MCPClient("ws://invalid:8000")
            
            with pytest.raises(Exception, match="Connection failed"):
                await client.connect()
    
    @pytest.mark.asyncio
    async def test_request_timeout(self, mcp_client, mock_websocket):
        """Test request timeout handling."""
        mcp_client.websocket = mock_websocket
        mock_websocket.recv.side_effect = asyncio.TimeoutError()
        
        with pytest.raises(asyncio.TimeoutError):
            await mcp_client._send_request("test_method", {})
    
    @pytest.mark.asyncio
    async def test_invalid_response(self, mcp_client, mock_websocket):
        """Test handling of invalid responses."""
        mcp_client.websocket = mock_websocket
        mock_websocket.recv.return_value = "invalid json"
        
        with pytest.raises(json.JSONDecodeError):
            await mcp_client._send_request("test_method", {})
    
    @pytest.mark.asyncio
    async def test_error_response(self, mcp_client, mock_websocket):
        """Test handling of error responses."""
        mcp_client.websocket = mock_websocket
        mock_websocket.recv.return_value = json.dumps({
            "id": "1",
            "error": {"code": 404, "message": "Not found"}
        })
        
        with pytest.raises(Exception, match="Not found"):
            await mcp_client._send_request("test_method", {})
    
    @pytest.mark.asyncio
    async def test_auto_reconnect(self, mcp_client, mock_websocket):
        """Test automatic reconnection."""
        # First connection
        mcp_client.websocket = mock_websocket
        mock_websocket.recv.return_value = json.dumps({
            "id": "1",
            "result": {"price": 150.0}
        })
        
        price1 = await mcp_client.get_stock_price("AAPL")
        assert price1 == 150.0
        
        # Simulate disconnection
        mock_websocket.recv.side_effect = Exception("Connection lost")
        
        # Should reconnect automatically
        with patch('stock_analysis_agent.mcp_client.websockets.connect', return_value=mock_websocket):
            mock_websocket.recv.side_effect = None
            mock_websocket.recv.return_value = json.dumps({
                "id": "2",
                "result": {"price": 151.0}
            })
            
            price2 = await mcp_client.get_stock_price("AAPL")
            assert price2 == 151.0


if __name__ == '__main__':
    pytest.main([__file__]) 