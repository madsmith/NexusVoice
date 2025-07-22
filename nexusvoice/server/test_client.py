#!/usr/bin/env python3
import argparse
import asyncio
import os
import signal
import readline
import logfire
import logging
from pathlib import Path
from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory

from nexusvoice.bootstrap import get_logfire

from .connection import NexusConnection
from .types import CallResponse

logger = logging.getLogger(__name__)

class REPLClient:
    """A REPL interface for the NexusServer connection."""
    
    def __init__(self, host: str, port: int):
        self.connection = NexusConnection(host, port)
        self.running = False
        self.shutdown_event = asyncio.Event()
        
        # Setup command history
        history_path = os.path.expanduser("~/.nexus_client_history")
        self.history = FileHistory(history_path)
        self.prompt = PromptSession(history=self.history)

        self.command_map = {
            "ping": self.connection.ping,
            "queue_broadcast": self.connection.queue_broadcast,
        }
    
    async def start(self):
        """Start the REPL interface"""
        self.running = True
        
        print("NexusClient REPL")
        print("Type 'help' for available commands or 'exit' to quit")
        
        # Auto-connect on first run
        print("Automatically connecting to server...")
        await self.connection.connect()
        
        while self.running:
            # Check if we're connected (for reconnection after a disconnect)
            if not self.connection.connected:
                try:
                    choice = await self.prompt.prompt_async("Not connected to server. Connect now? (y/n): ")
                    choice = choice.strip().lower()
                    if choice == 'y':
                        if not await self.connection.connect():
                            continue
                    else:
                        print("You can still use local commands. Type 'connect' to connect later.")
                except KeyboardInterrupt:
                    print("\nExiting...")
                    self.running = False
                    break
            
            # Get command input
            try:
                cmd = await self.prompt.prompt_async("nexus> ")
                cmd = cmd.strip()
            except KeyboardInterrupt:
                continue
            except EOFError:
                print("\nExiting...")
                self.running = False
                break
            
            if not cmd:
                continue
                
            # Process command
            cmd_parts = cmd.split()
            cmd_name = cmd_parts[0].lower()
            
            if cmd_name == 'exit' or cmd_name == 'quit':
                print("Exiting...")
                break
                
            elif cmd_name == 'help':
                self.print_help()
                
            elif cmd_name == 'connect':
                await self.connection.connect()
                
            elif cmd_name == 'disconnect':
                await self.connection.disconnect()
                
            elif cmd_name == 'status':
                print(f"Connected: {self.connection.connected}")
                print(f"Server address: {self.connection.host}:{self.connection.port}")
                
            elif cmd_name in self.command_map:
                response = await self.command_map[cmd_name]()
                if isinstance(response, CallResponse):
                    print(f"{response.result}")
                else:
                    print(f"{cmd_name} result: {response}")

            else:
                print(f"Unknown command: {cmd_name}")
                print("Type 'help' for available commands")
        
        # Clean up when exiting
        if self.connection.connected:
            await self.connection.disconnect()
        try:
            readline.write_history_file(self.history_file)
        except Exception:
            pass
    
    async def stop(self):
        self.running = False
        self.shutdown_event.set()  # Signal any pending input to exit

        # Shutdown REPL
        try:
            readline.write_history_file(self.history_file)
        except Exception:
            pass

        if self.connection.connected:
            await self.connection.disconnect()
    
    def print_help(self):
        """Print help information"""
        print("\nAvailable Commands:")
        print("  help           - Show this help message")
        print("  connect        - Connect to the server")
        print("  disconnect     - Disconnect from the server")
        print("  ping           - Send a ping to the server")
        print("  queue_broadcast - Queue a broadcast message on the server")
        print("  status         - Show connection status")
        print("  exit, quit     - Exit the client")
        print()
            
async def run_repl(args: argparse.Namespace):
    # Give some time for logfire to initialize
    await asyncio.sleep(.2)

    # Create and start client
    repl = REPLClient(args.host, args.port)
    try:
        # Start REPL - this will handle both user input and server messages concurrently
        await repl.start()
    finally:
        # Ensure we disconnect properly when exiting
        if repl.connection.connected:
            await repl.connection.disconnect()

# Synchronous entry point for command-line execution
def main():
    """Synchronous entry point that runs the async main function"""
    try:
        # Parse command line arguments
        parser = argparse.ArgumentParser(description="NexusVoice Test Client")
        parser.add_argument("--host", default="localhost", help="Server host (default: localhost)")
        parser.add_argument("--port", type=int, default=8008, help="Server port (default: 8008)")
        parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
        args = parser.parse_args()

        if args.verbose:
            logging.getLogger().setLevel(logging.TRACE) # type: ignore
            
            # TODO: redo bootstrap process so we don't have to dig deep into 
            # framework internals to reconfigure
            instance = get_logfire()
            assert instance is not None
            assert isinstance(instance.config.console, logfire.ConsoleOptions)
            instance.config.console.min_log_level = "trace"
            instance.config._initialized = False
            instance.config.initialize()

        
        asyncio.run(run_repl(args))
    except KeyboardInterrupt:
        print("\nClient terminated by user")
    return 0

if __name__ == "__main__":
    main()
