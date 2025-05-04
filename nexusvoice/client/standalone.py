import argparse
import logging
from omegaconf import OmegaConf

from nexusvoice.utils.logging import get_logger
logger = get_logger(__name__)

from nexusvoice.client import NexusVoiceStandalone, NexusVoiceClient
from nexusvoice.core.config import load_config

def main():
    client = None
    try:
        parser = argparse.ArgumentParser(description="NexusVoice Online Client")
        parser.add_argument('-c', '--cmd', nargs=argparse.REMAINDER, help='Send a command prompt to the client')
        args = parser.parse_args()

        log_format = "[{levelname}]\t{threadName}\t{message}"
        log_level = logging.DEBUG
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(log_format, style="{"))
        logging.basicConfig(level=log_level, style="{", format=log_format, handlers=[handler])

        logger.debug("Loading config")
        config = load_config()
 
        logger.debug("Creating NexusVoiceClient")
        client = NexusVoiceStandalone("test", config)
        client.start()

        if args.cmd:
            prompt = " ".join(args.cmd).strip()
            if prompt:
                client.add_command(NexusVoiceClient.CommandProcessText(prompt))
        else:
            client.add_command(NexusVoiceClient.CommandProcessText("Hey Nexus!"))

        client.join()
    except KeyboardInterrupt:
        if client:
            client.stop()
    except Exception as e:
        logger.error(e)


if __name__ == "__main__":
    main()