import asyncio
import logging
import logfire
import re
import traceback
from typing import TYPE_CHECKING, Awaitable, Callable, List, Any

from nexusvoice.utils.eventbus import EventBus, EventT, SubscriptionToken

from nexusvoice.tools.lutron.types import (
    COMMAND_QUERY_PREFIX,
    COMMAND_EXECUTE_PREFIX,
    COMMAND_RESPONSE_PREFIX,
    LINE_END,
    LutronSpecialEvents
)

RE_IS_INTEGER = re.compile(r"^\-?\d+$")
RE_IS_FLOAT = re.compile(r"^\-?\d+\.\d+$")

if TYPE_CHECKING:
    from nexusvoice.tools.lutron.commands.base import LutronCommand

class LutronHomeworksClient:
    def __init__(self, host, username=None, password=None, port=23, keepalive_interval=60):
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.keepalive_interval = keepalive_interval

        self._reader = None
        self._writer = None
        self.connected = False
        self.command_ready = False
        self._keepalive_task = None
        self._output_emitter_task = None
        self._reconnect_task = None

        self._eventbus = EventBus()
        self.logger = logging.getLogger(self.__class__.__name__)
        self._lock = asyncio.Lock()
        self._stop_event = asyncio.Event()

    @property
    def reader(self):
        assert self._reader is not None, "Connection not established. Call connect() first."
        return self._reader

    @property
    def writer(self):
        assert self._writer is not None, "Connection not established. Call connect() first."
        return self._writer
    
    @logfire.instrument("Connect")
    async def connect(self):
        async with self._lock:
            self.logger.info(f"Connecting to {self.host}:{self.port}")
            try:
                self._reader, self._writer = await asyncio.open_connection(self.host, self.port)
                self.connected = True
                await self._login()
                self._start_keepalive()
                self._start_output_emitter()
            except Exception as e:
                self.logger.error(f"Connection failed: {e}")
                self.connected = False
                self.command_ready = False
                self._reconnect_later()

    @logfire.instrument("Login")
    async def _login(self):
        try:
            if self.username is None or self.password is None:
                raise ValueError("Username and password must be provided.")

            self.logger.debug("Waiting for login prompt...")
            with logfire.span("Find Login Prompt"):
                data = await self._read_until(b"login: ", timeout=10)
                self.logger.debug(f"Login prompt received: {data}")
                logfire.debug("Sending Username")
                await self._write(self.username + LINE_END)

            with logfire.span("Find Password Prompt"):
                data = await self._read_until(b"password: ", timeout=10)
                self.logger.debug(f"Password prompt received: {data}")
                logfire.debug("Sending Password")
                await self._write(self.password + LINE_END)
            with logfire.span("Reading Command Ready Prompt"):
                await self._read_prompt(timeout=10)

            # Reset the command prompt once after logging in to discard
            # any residual data from the login process (like a \0 char
            # that is showing up attached to the first prompt)
            await self._write("\r\n")
            with logfire.span("Reading Command Ready Prompt 2"):
                await self._read_prompt(timeout=10)
            
            logfire.debug("Login complete.")
            self.logger.debug("Login complete.")
            self.command_ready = True
        except Exception as e:
            self.logger.error(f"Login failed: {e}")
            self.connected = False
            self.command_ready = False
            self._reconnect_later()

    async def _read_until(self, end_bytes: bytes, timeout: float=10):
        """Read until the given prompt or timeout."""

        prompt = b"QNET> "
        discard_prompt = self.command_ready and end_bytes != prompt

        buf = b""
        try:
            while not buf.endswith(end_bytes):
                chunk = await asyncio.wait_for(self.reader.read(1), timeout=timeout)
                if not chunk:
                    raise ConnectionError("Connection closed by server.")

                buf += chunk
                if discard_prompt:
                    if buf.endswith(prompt):
                        # Remove the prompt from end of buffer
                        self.logger.debug(f"Discarding prompt... [{prompt}]")
                        buf = buf[:-len(prompt)]
                        continue

            self.logger.debug(f"<< {buf.rstrip()}")

            return buf
        except asyncio.TimeoutError:
            raise TimeoutError(f"Timeout waiting for prompt: {end_bytes}")

    async def _read_line(self, timeout: float=10):
        return await self._read_until(LINE_END.encode('ascii'), timeout=timeout)

    async def _read_prompt(self, timeout: float=10):
        return await self._read_until(b"QNET> ", timeout=timeout)
    
    async def _write(self, data: str):
        self.logger.debug(f">> {data.rstrip()}")
        self.writer.write(data.encode('ascii'))
        await self.writer.drain()

    def _start_output_emitter(self):
        if self._output_emitter_task and not self._output_emitter_task.done():
            return
        self._output_emitter_task = asyncio.create_task(
            self._output_emitter_loop(),
            name="Lutron-OutputEmitter",
        )

    async def _output_emitter_loop(self):
        while not self._stop_event.is_set():
            try:
                output = await self._read_line(timeout=0.1)
                await asyncio.sleep(0.5)
                event, data = self._parse_output(output)
                if event is None:
                    await self._eventbus.emit(LutronSpecialEvents.NonResponseEvents.value, output)
                    await self._eventbus.emit(LutronSpecialEvents.AllEvents.value, output)
                    continue
                print(f"Event: {event}, Data: {data}")
                await self._eventbus.emit(event, data)
                await self._eventbus.emit(LutronSpecialEvents.AllEvents.value, data)
            except asyncio.TimeoutError:
                pass
            except Exception as e:
                self.logger.error(f"Error reading from server: {e}")
                self.connected = False
                self.command_ready = False
                self._reconnect_later()

    def _parse_output(self, output: bytes):
        line = output.decode('ascii').strip()
        if not line:
            return None, None

        if line.startswith('QNET'):
            self.logger.debug(f"Ignoring Command Prompt: {line}")
            return None, None

        if not line.startswith(COMMAND_RESPONSE_PREFIX):
            return None, None
        
        parts = line.split(',')
        event = parts[0][1:]
        data = self._infer_data(parts[1:])

        return event, data
    
    def _infer_data(self, parts: List[str]) -> List[Any]:
        result = []

        for part in parts:
            value: Any = part
            if RE_IS_INTEGER.match(part):
                value = int(part)
            elif RE_IS_FLOAT.match(part):
                value = float(part)
            result.append(value)

        return result
    
    def _start_keepalive(self):
        if self._keepalive_task and not self._keepalive_task.done():
            return
        self._keepalive_task = asyncio.create_task(
            self._keepalive_loop(),
            name="Lutron-Keepalive",
        )

    async def _keepalive_loop(self):
        while not self._stop_event.is_set():
            await asyncio.sleep(self.keepalive_interval)
            if not self.connected:
                continue
            try:
                await self.send_heartbeat()
            except Exception as e:
                self.logger.warning(f"Keepalive failed: {e}")
                self.connected = False
                self.command_ready = False
                self._reconnect_later()

    @logfire.instrument("Send Heartbeat")
    async def send_heartbeat(self):
        """Send a keep-alive/heartbeat command. Customize as needed."""
        self.logger.debug("Sending heartbeat...")
        # Example: await self.send_command('NOOP')
        pass

    @logfire.instrument("Send Command")
    async def send_command(self, command: str):
        async with self._lock:
            if not self.connected or self.writer is None:
                raise ConnectionError("Not connected to Lutron server.")
            await self._write(command + LINE_END)
            # Optionally, read response here

    @logfire.instrument("Execute Command")
    async def execute_command(self, command: 'LutronCommand', timeout: float = 5.0):
        """
        Execute a Lutron command and return the response.
        
        Args:
            command: The command to execute
            timeout: Command timeout in seconds

        Returns:
            The command response

        Raises:
            CommandError: If the command fails
            CommandTimeout: If the command times out
            ConnectionError: If not connected
        """
        assert self.connected, "Please connect client before invoking commands."
        assert self.command_ready, "Client wasn't ready to receive commands."
            
        return await command.execute(self, timeout=timeout)

    def subscribe(self, event_name: EventT, callback) -> SubscriptionToken:
        """
        Subscript to events announced by the Lutron Homeworks server.
        """
        return self._eventbus.subscribe(event_name, callback)

    def unsubscribe(self, token: SubscriptionToken):
        """
        Removed a previous subscription.
        """
        self._eventbus.unsubscribe(token)

    def _reconnect_later(self, delay: float = 5) -> None:
        if self._reconnect_task and not self._reconnect_task.done():
            return
        
        async def reconnect() -> None:
            await asyncio.sleep(delay)
            await self.connect()
        
        self._reconnect_task = asyncio.create_task(
            reconnect(),
            name="Lutron-Reconnect",
        )

    @logfire.instrument("Close")
    async def close(self):
        print("Closing Lutron client...")

        self._stop_event.set()

        if self.command_ready:
            await self.send_command("LOGOUT")
        
        # Cancel background tasks
        tasks_to_cancel = []
        
        if self._keepalive_task and not self._keepalive_task.done():
            self.logger.debug("Cancelling keepalive task")
            self._keepalive_task.cancel()
            tasks_to_cancel.append(self._keepalive_task)
        
        if self._output_emitter_task and not self._output_emitter_task.done():
            self.logger.debug("Cancelling output emitter task")
            self._output_emitter_task.cancel()
            tasks_to_cancel.append(self._output_emitter_task)
            
        if self._reconnect_task and not self._reconnect_task.done():
            self.logger.debug("Cancelling reconnect task")
            self._reconnect_task.cancel()
            tasks_to_cancel.append(self._reconnect_task)
        
        # Wait for all tasks to complete their cancellation
        if tasks_to_cancel:
            await asyncio.gather(*tasks_to_cancel, return_exceptions=True)
        
        async with self._lock:
            if self.writer:
                try:
                    self.writer.close()
                    await self.writer.wait_closed()
                except Exception as e:
                    self.logger.warning(f"Error closing connection: {e} {type(e)}")
                    stacktrace = traceback.format_exc()
                    self.logger.warning(stacktrace)
        
        self.connected = False
        self.command_ready = False

    def __del__(self):
        if not self._stop_event.is_set():
            asyncio.create_task(self.close())
