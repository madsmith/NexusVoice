from typing import Optional
import numpy as np
from pathlib import Path
import threading
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration

from nexusvoice.utils.logging import get_logger
logger = get_logger(__name__)

from nexusvoice.ai.InferenceEngine import InferenceEngineBase

class AudioInferenceEngine(InferenceEngineBase[np.ndarray, str]):
    def __init__(self, model_id):
        super().__init__()
        self.model_id = model_id
        self._device = None
        self._processor: Optional[WhisperProcessor]= None
        self._model = None
        self.lock = threading.Lock()
    
    @property
    def device(self):
        assert self._device is not None, "Device not initialized"
        return self._device
    
    @property
    def processor(self):
        assert self._processor is not None, "Processor not initialized"
        return self._processor
    
    @property
    def model(self):
        assert self._model is not None, "Model not initialized"
        return self._model

    def initialize(self):
        self.initDevice()
        self.initModel()

    def initDevice(self):
        self._device = torch.device(
            "mps" if torch.mps.is_available()
              else "cuda" if torch.cuda.is_available() 
              else "cpu"
        )

    def initModel(self):
        # Attempt to load the model locally first
        model = None
        for local_only in [True, False]:
            try:
                self._processor = WhisperProcessor.from_pretrained(
                    self.model_id,
                    return_attention_mask=True,
                    local_files_only=local_only) # type: ignore
                model = WhisperForConditionalGeneration.from_pretrained(
                    self.model_id,
                    local_files_only=local_only)
                break
            except Exception as e:
                logger.warning(f"Failed to load model locally: {e}")
                # Print statck trace of e
                logger.debug(e, exc_info=True)

        if self._processor is None or model is None:
            raise RuntimeError("Failed to load model or processor.")
        
        if model.generation_config:
            model.generation_config.forced_decoder_ids = None

        # Load model to device
        self._model = model.to(self.device) # type: ignore

        self._warmup()

    def _warmup(self):
        dummy_input = np.zeros(3000, dtype=np.int16)
        self.infer(dummy_input, sampling_rate=16000)

    def infer(self, inputs, **inference_params):
        if "sampling_rate" not in inference_params:
            raise ValueError("sampling_rate is required for audio inference")
        sampling_rate = inference_params["sampling_rate"]
        with self.lock:
            # Tokenize the audio inputs
            inputs = self.processor(
                inputs,
                sampling_rate=sampling_rate,
                return_tensors="pt").to(self._device)
            
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
