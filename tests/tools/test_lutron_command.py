import asyncio
import pytest
import pytest_asyncio
import unittest.mock as mock
from typing import Dict, List, Any, Optional

from nexusvoice.tools.lutron.types import (
    CommandType,
    CommandError,
    CommandTimeout,
    COMMAND_QUERY_PREFIX, 
    COMMAND_EXECUTE_PREFIX,
    COMMAND_RESPONSE_PREFIX,
)

from nexusvoice.tools.lutron.commands.base import (
    LutronCommand,
    CommandDefinition as Cmd,
    CommandSchema,
    CommandError,
    CommandTimeout,
)
from nexusvoice.tools.lutron.lutron import LutronHomeworksClient, SubscriptionToken, EventT

    
cmds = [
    Cmd("STATUS", lambda data: {"status": data}),
    Cmd("INFO", lambda data: {"info": data}),
]
schema = CommandSchema(format_template="TEST,{action},{param1},{param2}", commands=cmds)

# Test command implementation for testing
class TestCommand(LutronCommand[str], schema=schema):
    """A test implementation of LutronCommand for testing"""
    
    def __init__(self, action: str, param1: str, param2: str):
        super().__init__(action)
        self.param1 = param1
        self.param2 = param2

class MockSubscriptionToken(SubscriptionToken):
    def __init__(self, event: EventT, nonce: int):
        super().__init__(event, nonce)
        self.event_type = event
    
# Mock LutronHomeworksClient for testing
class MockLutronClient:
    def __init__(self, responses=None, errors=None):
        self.sent_commands = []
        self.subscriptions = {}
        self.responses = responses or []
        self.errors = errors or []
    
    def subscribe(self, event_type, callback):
        if event_type not in self.subscriptions:
            self.subscriptions[event_type] = {}
        token = MockSubscriptionToken(event_type, len(self.subscriptions[event_type]))
        self.subscriptions[event_type][token] = callback
        return token
        
    def unsubscribe(self, token):
        if token.event_type in self.subscriptions:
            if token in self.subscriptions[token.event_type]:
                del self.subscriptions[token.event_type][token]
    
    async def send_command(self, command):
        self.sent_commands.append(command)
        
        # Process any queued responses
        for response_type, response_data in self.responses:
            print(f"Processing response: {response_type} {response_data}")
            if response_type in self.subscriptions:
                callbacks = list(self.subscriptions[response_type].values())
                for callback in callbacks:
                    if callback:
                        print(f"Calling callback: {callback}")
                        callback(response_data)
                        
        # Process any queued errors
        for error_data in self.errors:
            print(f"Processing error: {error_data}")
            if "ERROR" in self.subscriptions:
                callbacks = list(self.subscriptions["ERROR"].values())
                for callback in callbacks:
                    if callback:
                        callback(error_data)
        
        return True

def test_command_initialization():
    """Test basic command initialization"""
    cmd = TestCommand("STATUS", "1", "2")
    assert cmd.action == "STATUS"
    assert cmd.param1 == "1"
    assert cmd.param2 == "2"
    assert cmd.command_type == CommandType.QUERY
    
def test_command_formatting():
    """Test command string formatting"""
    cmd = TestCommand("STATUS", "1", "2")
    assert cmd.formatted_command == f"{COMMAND_QUERY_PREFIX}TEST,STATUS,1,2"
    
    # Test execute command formatting
    cmd.set()
    assert cmd.command_type == CommandType.EXECUTE
    print(cmd.formatted_command)
    assert cmd.formatted_command == f"{COMMAND_EXECUTE_PREFIX}TEST,STATUS,1,2"
    
def test_matches_response():
    """Test response matching logic"""
    cmd = TestCommand("STATUS", "1", "2")
    
    # Exact match
    matches, unmatched = cmd._matches_response(["STATUS", "1", "2", "extra_data"])
    assert matches is True
    assert unmatched == ["extra_data"]
    
    # Mismatch on action
    matches, unmatched = cmd._matches_response(["INFO", "1", "2"])
    assert matches is False
    
    # Mismatch on param
    matches, unmatched = cmd._matches_response(["STATUS", "3", "2"])
    assert matches is False
    
def test_process_response():
    """Test response processing"""
    cmd = TestCommand("STATUS", "1", "2")
    result = cmd.process_response(["OK"])
    assert result == {"status": "OK"}
    
    cmd = TestCommand("INFO", "1", "2")
    result = cmd.process_response(["Version", "1.2.3"])
    assert result == {"info": ["Version", "1.2.3"]}
    
    # Test fallback to default processor
    try:
        cmd = TestCommand("UNKNOWN", "1", "2")
        assert False, "Expected exception"
    except Exception as e:
        assert True, f"Expected exception: {e}"

@pytest.mark.asyncio
async def test_command_execution_success():
    """Test successful command execution"""
    cmd = TestCommand("STATUS", "1", "2")
    
    # Setup mock with successful response
    mock_client = MockLutronClient(
        responses=[("TEST", ["STATUS", "1", "2", "OK"])]
    )
    
    print("Executing command...")
    result = await cmd.execute(mock_client) # type: ignore
    print(result)
    assert result == {"status": "OK"}
    assert mock_client.sent_commands == ["?TEST,STATUS,1,2"]
    
@pytest.mark.asyncio
async def test_command_execution_error():
    """Test command execution with error response"""
    cmd = TestCommand("STATUS", "1", "2")
    
    # Setup mock with error response
    mock_client = MockLutronClient(
        errors=[["ERROR", "3"]]  # Error code 3
    )
    
    with pytest.raises(CommandError) as exc_info:
        await cmd.execute(mock_client) # type: ignore
        
    assert exc_info.value.error_code == 3
    assert "?TEST,STATUS,1,2" in str(exc_info.value)
    
@pytest.mark.asyncio
async def test_command_timeout():
    """Test command execution timeout"""
    cmd = TestCommand("STATUS", "1", "2")
    
    # Setup mock with no responses
    mock_client = MockLutronClient()
    
    with pytest.raises(CommandTimeout):
        # Use small timeout for testing
        await cmd.execute(mock_client, timeout=0.1) # type: ignore
        
@pytest.mark.asyncio        
async def test_multiple_responses():
    """Test handling multiple responses"""
    cmd = TestCommand("STATUS", "1", "2")
    
    # Setup mock with multiple responses
    mock_client = MockLutronClient(
        responses=[
            ("TEST", ["OTHER", "1", "2", "IGNORED"]),  # Should be ignored (action mismatch)
            ("TEST", ["STATUS", "1", "2", "OK"]),      # Should be processed
        ]
    )
    
    result = await cmd.execute(mock_client) # type: ignore
    assert result == {"status": "OK"}
    
@pytest.mark.asyncio
async def test_connection_error():
    """Test handling connection error"""
    cmd = TestCommand("STATUS", "1", "2")
    
    # Setup mock that raises connection error
    mock_client = MockLutronClient()
    mock_client.send_command = mock.AsyncMock(side_effect=ConnectionError("Not connected"))
    
    with pytest.raises(ConnectionError):
        await cmd.execute(mock_client) # type: ignore
