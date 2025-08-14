import discord
from discord.ext import commands
import logfire
import logging

from nexusvoice.core.config import NexusConfig
from nexusvoice.core.protocol.connection import NexusConnection


logger = logging.getLogger(__name__)

class NexusDiscordBot():
    def __init__(self, host: str, port: int, client_id: str, config: NexusConfig):
        self.client_id = client_id
        self.config = config
        
        self._host = host
        self._port = port

        self._nexus_connection: NexusConnection = NexusConnection(self._host, self._port)

        # Setup Discord Bot
        intents = discord.Intents.default()
        intents.message_content = True
        self._bot: commands.Bot = commands.Bot(command_prefix="!", intents=intents)

        self._notify_users: list[discord.User | discord.Member] = []


    
    async def initialize(self):

        self._nexus_connection.subscribe("server_message", self._process_server_message)

        self._initialize_bot()

    async def start(self):
        discord_token = self.config.get("nexus.bot.discord_token")

        if not discord_token:
            raise ValueError("Missing config discord_token")
        
        with logfire.span("Connection Lifecycle"):
            await self._nexus_connection.connect()
            print("Starting bot")

            await self._bot.login(discord_token)
            
            for user_id in self.config.get("nexus.bot.notify_users"):
                user = self._bot.get_user(user_id) or await self._bot.fetch_user(user_id)
                if user:
                    logger.info(f"Added notify user: {user}")
                    self._notify_users.append(user)
                else:
                    logger.warning(f"User not found: {user_id}")

            # TODO: run bot in a task
            await self._bot.connect()

    async def stop(self):
        if self._nexus_connection.connected:
            await self._nexus_connection.disconnect()

    async def task(self):
        return self._nexus_connection.get_task()

    async def _process_server_message(self, msg: str):
        # TODO: send message to client?
        logger.info(msg)
        for user in self._notify_users:
            await user.send(msg)

    def _initialize_bot(self):
        self._initialize_commands()
        self._initialize_events()

    def _initialize_commands(self):
        @self._bot.command()
        async def ping(ctx: commands.Context):
            await ctx.send("Pong!")

        @self._bot.command()
        async def subscribe(ctx: commands.Context):
            await ctx.send("Subscribed to server messages")
            self._notify_users.append(ctx.author)
            print(self._notify_users)

        @self._bot.command()
        async def unsubscribe(ctx: commands.Context):
            await ctx.send("Unsubscribed from server messages")
            self._notify_users.remove(ctx.author)
            print(self._notify_users)

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
        is_bot_message = self._bot.user and message.author.id == self._bot.user.id
        
        
        # Respond if the message contains the word "test" (case insensitive)
        if "test" in message.content.lower() and not is_bot_message:
            await message.channel.send("I detected a test! 👀")
            return

        if not is_bot_message:
            # Log every message the bot sees
            if message.guild:  # Message is from a server
                channel_name = message.channel.name if isinstance(message.channel, discord.TextChannel) else "DM"
                print(f'[{message.guild.name}] [#{channel_name}] {message.author.name}: {message.content}')
            else:  # Direct message
                print(f'[DM] {message.author.name}: {message.content}')
                if isinstance(message.author, discord.User):
                    print(message.author, message.author.name, message.author.id)
                elif isinstance(message.author, discord.Member):
                    print(message.author, message.author.name, message.author.nick, message.author.id)
                else:
                    print(message.author, message.author.name, message.author.id)
            
            ctx = await self._bot.get_context(message)
            if not ctx.valid and not ctx.command:
                await self._prompt_agent(message)
                return
            
        await self._bot.process_commands(message)

    async def _prompt_agent(self, message: discord.Message):
        try:
            if not self._nexus_connection.connected:
                logger.info("Connecting to server...")
                await self._nexus_connection.connect()
            
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
            