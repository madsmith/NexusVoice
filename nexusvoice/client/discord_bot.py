import discord
from discord.ext import commands
import logfire
import logging
from typing import TYPE_CHECKING

from nexusvoice.core.config import NexusConfig
from nexusvoice.server.connection import NexusConnection


logger = logging.getLogger(__name__)

class NexusDiscordBot():
    def __init__(self, host: str, port: int, client_id: str, config: NexusConfig):
        self.client_id = client_id
        self.config = config
        
        self._host = host
        self._port = port

        self._nexus_connection: NexusConnection = NexusConnection(self._host, self._port)

        intents = discord.Intents.default()
        intents.message_content = True
        self._bot: commands.Bot = commands.Bot(command_prefix="!", intents=intents)

    
    async def initialize(self):

        self._nexus_connection.subscribe("server_message", self._process_server_message)

        self._initialize_bot()


    async def start(self):
        discord_token = self.config.get("nexus.bot.discord_token")

        if not discord_token:
            raise ValueError("Missing config discord_token")
        
        with logfire.span("Connection Lifecycle"):
            await self._nexus_connection.connect()
            await self._bot.start(discord_token)

    async def stop(self):
        if self._nexus_connection.connected:
            await self._nexus_connection.disconnect()

    async def process_text(self, text: str):
        try:
            response = await self._nexus_connection.send_command(
                "prompt_agent", {"prompt": text})
            return response
        except Exception as e:
            # TODO: Remove This - api shouldn't throw exceptions
            logger.error(f"Error processing command: {e}")
            # Show the traceback
            import traceback
            logger.error(traceback.format_exc())
            return ""

    async def _process_server_message(self, msg: str):
        # TODO: send message to client?
        logger.info(msg)


    def _initialize_bot(self):
        self._initialize_commands()
        self._initialize_events()

    def _initialize_commands(self):
        @self._bot.command()
        async def ping(ctx: commands.Context):
            await ctx.send("Pong!")

    def _initialize_events(self):
        @self._bot.event
        async def on_ready():
            await self._on_ready()
        
        @self._bot.event
        async def on_message(message: discord.Message):
            await self._on_message(message)

    #===========================================================
    # Bot Events
    #===========================================================
    async def _on_ready(self):
        if self._bot.user is None:
            logger.warning("Bot user is None")
            return
        
        logger.info(f'Logged in as {self._bot.user} (ID: {self._bot.user.id})')

    async def _on_message(self, message: discord.Message):
        # Log every message the bot sees
        if message.guild:  # Message is from a server
            channel_name = message.channel.name if isinstance(message.channel, discord.TextChannel) else "DM"
            print(f'[{message.guild.name}] [#{channel_name}] {message.author.name}: {message.content}')
        else:  # Direct message
            print(f'[DM] {message.author.name}: {message.content}')
        
        # Respond if the message contains the word "test" (case insensitive)
        if "test" in message.content.lower() and self._bot.user and message.author.id != self._bot.user.id:
            await message.channel.send("I detected a test! 👀")
            return

        
        try:
            logger.info(f"Prompting agent: {message.content}")
            response = await self._nexus_connection.send_command(
                "prompt_agent", {"prompt": message.content})
            logger.info(f"Response: {response}")
            await message.channel.send(response)
        except BaseException as e:
            logger.error(f"Error sending message: {e}")
            # Show the traceback
            import traceback
            logger.error(traceback.format_exc())
        
        # Don't forget to process commands, or your commands won't work
        await self._bot.process_commands(message)