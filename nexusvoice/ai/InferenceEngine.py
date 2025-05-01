from abc import ABC, abstractmethod
from dataclasses import dataclass
import threading
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import TypeVar, Generic

InputType = TypeVar('InputType')
OutputType = TypeVar('OutputType')

class InferenceEngine:
    def __init__(self, model_id=None):
        self.model_id = model_id
        self._model = None
        self._tokenizer = None
        self._device = None
        self.lock = threading.Lock()
        pass

    @property
    def model(self):
        assert self._model is not None, "Model not initialized"
        return self._model
    
    @property
    def tokenizer(self):
        assert self._tokenizer is not None, "Tokenizer not initialized"
        return self._tokenizer
    
    @property
    def device(self):
        assert self._device is not None, "Device not initialized"
        return self._device
    
    def initialize(self):
        self.initDevice()
        self.initTokenizer()
        self.initModel()

    def initDevice(self):
        self._device = torch.device(
            "mps" if torch.mps.is_available()
              else "cuda" if torch.cuda.is_available() 
              else "cpu"
        )

    def initTokenizer(self):
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self._tokenizer.add_special_tokens({"pad_token": "<|pad_id|>"})

    def initModel(self):
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16
        ).to(self.device)

        self._model.resize_token_embeddings(len(self.tokenizer))

    def infer(self, inputs, **inference_params):
        """ Take the unencoded inputs, encode them and generate an output """
        with self.lock:
            inputs = self.tokenizer.encode(inputs, add_special_tokens=False, return_tensors="pt")
            outputs = self._generate(inputs, **inference_params)
            return self.tokenizer.decode(outputs[0], skip_special_tokens=False)
        
    def generate(self, inputs, **generation_params):
        with self.lock, torch.no_grad():
            return self._generate(inputs, **generation_params)

    def _generate(self, inputs, **generation_params):
        inputs = inputs.to(self.device)
        outputs = self.model.generate(inputs, **generation_params)
        return outputs
    
    def loadResource(self):
        self._model = self._model.to(self.device) # type: ignore

    def unloadResource(self):
        del self._model
        self._model = None
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        elif self.device.type == "mps":
            torch.mps.empty_cache()

class InferenceEngineBase(Generic[InputType, OutputType], ABC):
    def __init__(self):
        pass

    @abstractmethod
    def initialize(self):
        """ Initialize the inference engine """
        pass

    @abstractmethod
    def infer(self, inputs: InputType, **inference_params) -> OutputType:
        """ Perform inference """
        pass

        
class InferenceEnginePool:
    @dataclass
    class Entry:
        resource: InferenceEngine
        lock: threading.Lock

    def __init__(self):
        self.resources = {}
        self.lock = threading.Lock()

    def getResource(self, resource_id, **resource_params):
        with self.lock:
            if resource_id not in self.resources:
                resource = InferenceEngine(**resource_params)
                resource.initialize()
                entry = InferenceEnginePool.Entry(resource, threading.Lock())
                self.resources[resource_id] = entry

            entry = self.resources[resource_id]
            return entry.resource
        
    def getResourceLock(self, model_id):
        entry = self.resources[model_id]
        if entry is not None:
            return entry.lock
        return None