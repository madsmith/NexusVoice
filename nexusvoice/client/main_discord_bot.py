import argparse
import asyncio
import logfire
import logging
from socket import gethostname

from nexusvoice.bootstrap import get_logfire
from nexusvoice.utils.logging import get_logger
logger = get_logger(__name__)

from nexusvoice.utils.debug import reset_logging
from nexusvoice.client.discord_bot import NexusDiscordBot
from nexusvoice.core.config import load_config, NexusConfig

async def run_client(config: NexusConfig, args: argparse.Namespace):
    try:
        with logfire.span("NexusDiscordBot"):
            host = args.host
            port = args.port
            client_id = gethostname()

            discord_bot = NexusDiscordBot(host, port, client_id, config)
            await discord_bot.initialize()
            
            # In the future, NexusVoiceClient should support async start/stop
            await discord_bot.start()
    except Exception as e:
        logger.error(e)
        import traceback
        traceback.print_exc()

def main():
    try:
        parser = argparse.ArgumentParser(description="NexusDiscordBot")
        parser.add_argument("--host", "-H", default="localhost", help="Server host (default: localhost)")
        parser.add_argument("--port", "-p", type=int, default=8008, help="Server port (default: 8008)")
        parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')
        args = parser.parse_args()

        logger.debug("Loading config")
        config = load_config()

        reset_logging(config.get("logging.suppress"))

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

        asyncio.run(run_client(config, args))
    except KeyboardInterrupt:
        logger.debug(f"Exiting due to KeyboardInterrupt")
        pass
    except Exception as e:
        logger.error(f"UNHANDLED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()