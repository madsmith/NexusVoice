import logging
from omegaconf import OmegaConf

from nexusvoice.utils.logging import get_logger
logger = get_logger(__name__)
from nexusvoice.client import NexusVoiceClient

def main():
    try:
        from nexusvoice.utils.logging import StackTraceFormatter
        # logging.basicConfig(level=logging.DEBUG, format="%(message)s")
        log_format = "[%(asctime)s] [%(levelname)s] - %(name)s %(message)s"
        log_format = "[{levelname}]\t{threadName}\t{message}"
        log_level = logging.DEBUG
        # log_level = LOG_TRACE_LEVEL
        handler = logging.StreamHandler()
        # handler.setFormatter(StackTraceFormatter(log_format, style="{"))
        handler.setFormatter(logging.Formatter(log_format, style="{"))
        logging.basicConfig(level=log_level, style="{", format=log_format, handlers=[handler])
        # log_format = "[%(asctime)s] [%(levelname)s] - %(name)s (%(pathname)s:%(lineno)d): %(message)s"
        # logging.basicConfig(level=logging.DEBUG, format=log_format)

        logger.debug("Loading config")
        config = OmegaConf.load("config.yml")

        logger.debug("Creating NexusVoiceClient")
        client = NexusVoiceClient("test", config)
        client.start()

        client.join()
    except KeyboardInterrupt:
        client.stop()
    except Exception as e:
        logger.error(e)

if __name__ == "__main__":
    main()