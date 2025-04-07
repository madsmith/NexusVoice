import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import warnings
warnings.filterwarnings(
    "ignore", message=".*will fall back to run on the CPU.*", category=UserWarning
)
warnings.filterwarnings(
    "ignore", message="dropout option adds dropout", category=UserWarning
)
warnings.filterwarnings(
    "ignore", message="`torch.nn.utils.weight_norm` is deprecated in favor", category=FutureWarning
)

import openwakeword
import os

from utils.logging import get_logger
logger = get_logger(__name__)

def initialize_openwakeword():
    model_paths = openwakeword.get_pretrained_model_paths()

    all_paths_exist = all(os.path.exists(model_path) for model_path in model_paths)
    if not all_paths_exist:
        logger.info("Downloading OpenWakeWord models...")
        openwakeword.utils.download_models()
