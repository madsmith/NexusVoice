import logging
import os
from typing import Literal
import warnings

from nexusvoice.utils.logging import get_logger
logger = get_logger(__name__)

import openwakeword

import logfire

EnvironmentT = Literal['DEV', 'TEST', 'PROD']

logfire_instance = None
logfire_handler = None

def get_logfire():
    global logfire_instance
    return logfire_instance

def get_logfire_handler():
    global logfire_handler
    return logfire_handler

def configure_logfire(environment_mode: EnvironmentT, service_name: str):
    global logfire_instance, logfire_handler

    level = "debug" if environment_mode == 'DEV' else "info"
    console_options = logfire.ConsoleOptions(min_log_level=level)

    # Determine if running client/main or some other script
    logfire_instance = logfire.configure(service_name=service_name, environment=environment_mode, console=console_options)
    logfire_instance.instrument_pydantic_ai()
    logfire_instance.instrument_requests()
    logfire_instance.instrument_httpx(capture_all=True)
    logfire_instance.install_auto_tracing(modules=['nexusvoice.ai'], min_duration=0.001)

    handler = logfire.LogfireLoggingHandler()
    handler.setLevel(logging.DEBUG)
    logging.getLogger().addHandler(handler)
    logfire_handler = handler

def configure_logging(environment_mode: EnvironmentT, enable_console: bool = False):
    level = logging.DEBUG if environment_mode == 'DEV' else logging.INFO
    
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    if enable_console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)  # Set the level for console output
        formatter = logging.Formatter("[{levelname}]\t{name}\t{message}", style="{")
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

def bootstrap(environment_mode: EnvironmentT, service_name: str = "NexusVoice"): 
    setup_environment(environment_mode)
    suppress_warnings(environment_mode)
    configure_logfire(environment_mode, service_name)
    configure_logging(environment_mode)
    initialize_openwakeword()

def setup_environment(environment_mode: EnvironmentT):
    os.environ["LOGFIRE_SEND_TO_LOGFIRE"] = "true"
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

def suppress_warnings(environment_mode: EnvironmentT):
    warnings.filterwarnings(
        "ignore", message=".*will fall back to run on the CPU.*", category=UserWarning
    )
    warnings.filterwarnings(
        "ignore", message="dropout option adds dropout", category=UserWarning
    )
    warnings.filterwarnings(
        "ignore", message="`torch.nn.utils.weight_norm` is deprecated in favor", category=FutureWarning
    )
    warnings.filterwarnings(
        "ignore", message="`use_mps_device` is deprecated and will be removed", category=UserWarning
    )
    warnings.filterwarnings(
        "ignore", message="Torchaudio's I/O functions now support", category=UserWarning
    )
    
        

    if environment_mode == 'PROD':
        from transformers.utils import logging
        logging.set_verbosity_error()
        logging.disable_progress_bar()

def initialize_openwakeword():
    model_paths = openwakeword.get_pretrained_model_paths()

    if model_paths:
        all_paths_exist = all(os.path.exists(model_path) for model_path in model_paths)
        if not all_paths_exist:
            from openwakeword.utils import download_models
            logger.info("Downloading OpenWakeWord models...")
            download_models()
