import numpy as np
from pathlib import Path
import threading
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration

from nexusvoice.utils.logging import get_logger
logger = get_logger(__name__)

from nexusvoice.ai.InferenceEngine import InferenceEngineBase

class AudioInferenceEngine(InferenceEngineBase):
    def __init__(self, model_id):
        super().__init__()
        self.model_id = model_id
        self.device = None
        self.processor = None
        self.model = None
        self.lock = threading.Lock()
        pass

    def initialize(self):
        self.initDevice()
        self.initModel()

    def initDevice(self):
        self.device = torch.device(
            "mps" if torch.mps.is_available()
              else "cuda" if torch.cuda.is_available() 
              else "cpu"
        )

    def initModel(self):
        # Attempt to load the model locally first
        for local_only in [True, False]:
            try:
                self.processor = WhisperProcessor.from_pretrained(
                    self.model_id,
                    return_attention_mask=True,
                    local_files_only=local_only)
                model = WhisperForConditionalGeneration.from_pretrained(
                    self.model_id,
                    local_files_only=local_only)
                break
            except Exception as e:
                logger.warning(f"Failed to load model locally: {e}")
                # Print statck trace of e
                logger.debug(e, exc_info=True)

        if self.processor is None or model is None:
            raise RuntimeError("Failed to load model or processor.")
        
        model.generation_config.forced_decoder_ids = None

        # Load model to device
        self.model = model.to(self.device)

        self._warmup()

    def _warmup(self):
        dummy_input = np.zeros(3000, dtype=np.int16)
        self.infer(dummy_input, 16000)

    def infer(self, audio, sampling_rate, **inference_params):
        with self.lock:
            # Tokenize the audio inputs
            inputs = self.processor(
                audio,
                sampling_rate=sampling_rate,
                return_tensors="pt").to(self.device)
            
            input_features = inputs.input_features
            attention_mask = inputs.attention_mask if "attention_mask" in inputs else None

            # Perform inference
            with torch.no_grad():
                outputs = self.model.generate(
                    input_features,
                    language="en",
                    task="transcribe",
                    attention_mask=attention_mask,
                    **inference_params)
        # End of lock
                
        # Decode the output
        transcription = self.processor.batch_decode(outputs, skip_special_tokens=True)

        return transcription[0]
