
import asyncio
import argparse
import logging
import logfire

from nexusvoice.bootstrap import get_logfire
from nexusvoice.core.config import load_config

from nexusvoice.server.NexusServer import NexusServer

from nexusvoice.utils.logging import get_logger
logger = get_logger(__name__)

async def run_server(args: argparse.Namespace):
    server: NexusServer | None = None
    try:
        with logfire.span("NexusServer"):
            config = load_config()
            host = config.get("nexus.server.host", "localhost")
            port = config.get("nexus.server.port", 8000)
            
            logfire.info(f"Starting NexusServer on {host}:{port}")
            server = NexusServer(host, port)
            await server.start()
    except KeyboardInterrupt:
        logger.warning(f"Exiting due to KeyboardInterrupt")
    except OSError as e:
        logger.error(f"Error starting server: {e}")
    except Exception as e:
        logger.error(f"Error running server: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if server and server.running:
            await server.stop()

def main():
    parser = argparse.ArgumentParser(description="NexusVoice Server")
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose logging')
    args = parser.parse_args()
    try:
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

        asyncio.run(run_server(args))
    except KeyboardInterrupt:
        logger.warning(f"Exiting due to KeyboardInterrupt")
    except Exception as e:
        logger.error(e)
        import traceback
        traceback.print_exc()    
    
if __name__ == "__main__":
    main()