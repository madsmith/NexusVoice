import argparse
import asyncio
import logfire
import logging
from omegaconf import OmegaConf
from socket import gethostname

from nexusvoice.bootstrap import get_logfire
from nexusvoice.utils.logging import get_logger
logger = get_logger(__name__)

from nexusvoice.utils.debug import reset_logging
from nexusvoice.client.voice_client import NexusVoiceClient
from nexusvoice.client.client_server import NexusClientServer
from nexusvoice.client.commands import CommandProcessText
from nexusvoice.core.config import load_config, NexusConfig

async def run_client(config: NexusConfig, args: argparse.Namespace):
    voice_client: NexusVoiceClient | None = None
    try:
        with logfire.span("NexusVoiceClient"):
            host = args.host
            port = args.port
            client_id = gethostname()

            client = NexusClientServer(host, port, client_id, config)
            voice_client = NexusVoiceClient(client_id, config, client)
            
            # In the future, NexusVoiceClient should support async start/stop
            run_coro = voice_client.run()

            if args.cmd:
                prompt = " ".join(args.cmd).strip()
                if prompt:
                    voice_client.add_command(CommandProcessText(prompt))

            await run_coro
    except KeyboardInterrupt:
        logger.warning(f"Exiting due to KeyboardInterrupt - run_client")
        if voice_client is not None:
            await voice_client.stop()
    except Exception as e:
        logger.error(e)
        import traceback
        traceback.print_exc()

def main():
    try:
        parser = argparse.ArgumentParser(description="NexusVoice Client")
        parser.add_argument("--host", "-H", default="localhost", help="Server host (default: localhost)")
        parser.add_argument("--port", "-p", type=int, default=8008, help="Server port (default: 8008)")
        parser.add_argument('--cmd', '-c', nargs=argparse.REMAINDER, help='Send a command prompt to the client')
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
        logger.warning(f"UNHANDLED: Exiting due to KeyboardInterrupt")
    except Exception as e:
        logger.error(f"UNHANDLED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()