import logging
from omegaconf import OmegaConf

from nexusvoice.utils.logging import get_logger
logger = get_logger(__name__)

from nexusvoice.client import NexusVoiceStandalone

def main():
    try:
        log_format = "[{levelname}]\t{threadName}\t{message}"
        log_level = logging.DEBUG
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(log_format, style="{"))
        logging.basicConfig(level=log_level, style="{", format=log_format, handlers=[handler])

        logger.debug("Loading config")
        config = OmegaConf.load("config.yml")

        logger.debug("Creating NexusVoiceClient")
        client = NexusVoiceStandalone("test", config)
        client.start()

        client.join()
    except KeyboardInterrupt:
        client.stop()
    except Exception as e:
        logger.error(e)


if __name__ == "__main__":
    main()