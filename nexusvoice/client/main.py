import asyncio
import logfire
import logging
import argparse
from nexusvoice.utils.debug import reset_logging
from omegaconf import OmegaConf

from nexusvoice.utils.logging import get_logger
logger = get_logger(__name__)

from nexusvoice.client.NexusClient import NexusVoiceClient
from nexusvoice.client.commands import CommandProcessText
from nexusvoice.core.config import load_config

async def run_client():
    client = None
    try:
        with logfire.span("NexusVoiceClient"):
            parser = argparse.ArgumentParser(description="NexusVoice Online Client")
            parser.add_argument('-c', '--cmd', nargs=argparse.REMAINDER, help='Send a command prompt to the client')
            args = parser.parse_args()

            log_format = "[{levelname}]\t{threadName}\t{name}\t{message}"
            log_level = logging.DEBUG
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter(log_format, style="{"))
            logging.basicConfig(level=log_level, style="{", format=log_format, handlers=[handler])

            logger.debug("Loading config")
            config = load_config()

            reset_logging(config.get("logging.suppress"))
    
            logger.debug("Creating NexusVoiceClient")
            client = NexusVoiceClient("local", config)
            # In the future, NexusVoiceClient should support async start/stop
            run_coro = client.run()

            if args.cmd:
                prompt = " ".join(args.cmd).strip()
                if prompt:
                    client.add_command(CommandProcessText(prompt))

            await run_coro
    except KeyboardInterrupt:
        logger.warning(f"Exiting due to KeyboardInterrupt - run_client")
        if client:
            await client.stop()
    except Exception as e:
        logger.error(e)
        import traceback
        traceback.print_exc()

def main():
    try:
        asyncio.run(run_client())
    except KeyboardInterrupt:
        logger.warning(f"UNHANDLED: Exiting due to KeyboardInterrupt")
    except Exception as e:
        logger.error(f"UNHANDLED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()