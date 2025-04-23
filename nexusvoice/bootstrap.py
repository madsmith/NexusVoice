import os
from typing import Literal
import warnings

from nexusvoice.utils.logging import get_logger
logger = get_logger(__name__)

import openwakeword

def setup_environment(environment_mode: Literal['DEV', 'PROD']):
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    suppress_warnings(environment_mode)

def suppress_warnings(environment_mode: Literal['DEV', 'PROD']):
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
