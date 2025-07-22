import asyncio
import json
import logfire
from typing import Dict, Callable, Any
from pydantic import ValidationError
from pydantic.type_adapter import TypeAdapter

from .types import (
    CallRequest, CallResponse, CallResponseSuccess, CallResponseError, 
    BroadcastMessage, ServerInboundMessage
)

inbound_message_adapter = TypeAdapter(ServerInboundMessage)

class NexusServer:
    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self.server_socket = None
        self.clients: Dict[str, asyncio.StreamWriter] = {}
        self.running = False

        self.lock = asyncio.Lock()
        
        # Command handlers
        self.command_handlers: Dict[str, Callable] = {
            "ping": self._handle_ping,
            "queue_broadcast": self._handle_queue_broadcast
        }

    async def start(self):
        """Initialize the server socket and start listening for connections"""
        self.server = await asyncio.start_server(
            self._handle_client, self.host, self.port
        )
        logfire.info(f"Server started on {self.host}:{self.port}")
        self.running = True
        
        async with self.server:
            await self._main_loop()

    async def stop(self):
        """Stop the server and close all connections"""
        self.running = False
        
        async with self.lock:
            # Close all client connections
            for client_id, writer in self.clients.items():
                try:
                    writer.close()
                    await writer.wait_closed()
                except Exception as e:
                    # Don't allow inspection during shutdown, format message manually
                    error_msg = f"Error closing connection for {client_id}: {e}"
                    logfire.error(error_msg)
        
            # Close server
            if self.server:
                self.server.close()
                await self.server.wait_closed()
        
        logfire.info("Server stopped")

    async def _main_loop(self):
        """Main server loop"""
        with logfire.span("NexusServer Running"):
            try:
                # Keep the server running until stop is called
                while self.running:
                    await asyncio.sleep(1)
            except asyncio.CancelledError:
                logfire.info("Server loop cancelled")
                self.running = False
            except Exception as e:
                error_msg = f"Error in server main loop: {e}"
                logfire.error(error_msg)
                self.running = False
            finally:
                await self.stop()
    
    async def _handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        """Handle a new client connection"""
        # Generate a unique client ID   
        client_address = writer.get_extra_info('peername')
        client_id = f"{client_address[0]}:{client_address[1]}"
        
        # Store the client connection
        async with self.lock:
            self.clients[client_id] = writer
        logfire.info(f"New client connected: {client_id}")
        
        try:
            while self.running:
                # Read data from client
                data = await reader.readline()
                if not data:
                    break
                    
                # Process the received command
                await self._process_command(client_id, data, writer)
        
        except asyncio.CancelledError:
            logfire.info(f"Client handler for {client_id} cancelled")
        except Exception as e:
            logfire.error(f"Error handling client {client_id}: {e}")
        finally:
            # Clean up when client disconnects
            async with self.lock:
                writer.close()
                await writer.wait_closed()
                if client_id in self.clients:
                    del self.clients[client_id]
            logfire.info(f"Client disconnected: {client_id}")
    
    async def _process_command(self, client_id: str, data: bytes, writer: asyncio.StreamWriter):
        try:
            try:
                msg = inbound_message_adapter.validate_json(data)
            except ValidationError as e:
                logfire.error(f"Invalid message from client {client_id}: {data.decode(errors='replace')} :: {e}")
                error_response = CallResponseError(
                    request_id="ERROR",
                    error="Invalid JSON format",
                    details={"raw_data": data.decode(errors='replace')}
                )
                await self._send_response(writer, error_response)
                return

            if isinstance(msg, CallRequest):
                handler = self.command_handlers.get(msg.command)
                if handler:
                    try:
                        result = await handler(client_id, msg.payload)
                        response = CallResponseSuccess(
                            request_id=msg.request_id,
                            result=result
                        )
                    except Exception as e:
                        response = CallResponseError(
                            request_id=msg.request_id,
                            error=f"Command execution error",
                            details={"exception": str(e)}
                        )
                else:
                    response = CallResponseError(
                        request_id=msg.request_id,
                        error=f"Unknown command",
                        details={"command": msg.command}
                    )
                await self._send_response(writer, response)

        except Exception as e:
            logfire.error(f"Unexpected error in _process_command: {e}")
    
    async def _send_response(self, writer: asyncio.StreamWriter, response: CallResponse | BroadcastMessage):
        try:
            response_data = response.model_dump_json().encode() + b'\n'
            writer.write(response_data)
            await writer.drain()
        except Exception as e:
            logfire.error(f"Error sending response: {e}")

    
    async def _handle_ping(self, client_id: str, payload: dict) -> str:
        """Handle ping command from client"""
        logfire.info(f"Ping from {client_id}")
        return "pong"

    async def _handle_queue_broadcast(self, client_id: str, payload: dict) -> str:
        """Handle queue broadcast command from client"""
        logfire.info(f"Queue broadcast from {client_id}")
        # defer queue a broadcast message in 2 seconds
        async def do_broadcast():
            await asyncio.sleep(4)
            await self.broadcast(BroadcastMessage(message="Hello from NexusServer"))

        asyncio.create_task(do_broadcast())
        
        return "ok"
    
    async def broadcast(self, message: BroadcastMessage):
        """Broadcast a message to all connected clients"""
        async with self.lock:
            for client_id, writer in self.clients.items():
                asyncio.create_task(self._send_response(writer, message))