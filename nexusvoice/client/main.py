import argparse
import asyncio
import logfire
import logging
from omegaconf import OmegaConf

from nexusvoice.bootstrap import get_logfire
from nexusvoice.utils.logging import get_logger
logger = get_logger(__name__)

from nexusvoice.utils.debug import reset_logging
from nexusvoice.client.NexusClient import NexusVoiceClient
from nexusvoice.client.commands import CommandProcessText
from nexusvoice.core.config import load_config

async def run_client():
    client: NexusVoiceClient | None = None
    try:
        with logfire.span("NexusVoiceClient"):
            parser = argparse.ArgumentParser(description="NexusVoice Online Client")
            parser.add_argument('-c', '--cmd', nargs=argparse.REMAINDER, help='Send a command prompt to the client')
            parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose logging')
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
        if client is not None:
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