import logging
import openwakeword
import os


logger = logging.getLogger(__name__)

def initialize_openwakeword():
    model_paths = openwakeword.get_pretrained_model_paths()

    all_paths_exist = all(os.path.exists(model_path) for model_path in model_paths)
    if not all_paths_exist:
        logger.info("Downloading OpenWakeWord models...")
        openwakeword.utils.download_models()