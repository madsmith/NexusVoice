from kokoro import KPipeline, KModel
from pathlib import Path
import sys
import threading
import torch

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
sys.path.append(str(ROOT_DIR))

from utils.logging import get_logger
logger = get_logger(__name__)

from ai.InferenceEngine import InferenceEngineBase

class TTSInferenceEngine(InferenceEngineBase):
    def __init__(self, model_id="hexgrad/Kokoro-82M", voices=None):
        super().__init__()
        self.model_id = model_id
        if type(voices) is str:
            voices = [voices]
        self.voices = voices
        self.device = None
        self.pipeline = None
        self.model = None
        self.lock = threading.Lock()

    def initialize(self):
        self.initDevice()
        self.initModel()

    def initDevice(self):
        # aten::angle isn't implemented for MPS devices and is used in the model
        # Fallback environ needs to be specified at runtime.
        self.device = torch.device(
            "mps" if torch.mps.is_available()
              else "cuda" if torch.cuda.is_available() 
              else "cpu"
        )

    def initModel(self):
        self.model = KModel(repo_id=self.model_id).to(self.device)

        self.pipeline = KPipeline(
            lang_code='en-us',
            repo_id=self.model_id,
            model=self.model,
            device=self.device)
        
        # Preload voices
        if self.voices:
            for voice in self.voices:
                # Pipeline caches the voice internally
                self.pipeline.load_voice(voice)

    def infer(self, inputs, **inference_params):
        with self.lock:
            generator = self.pipeline(inputs, **inference_params)

            for i, (gs, ps, audio) in enumerate(generator):
                return audio
                # print(i, gs, ps)
                # # display(Audio(data=audio, rate=24000, autoplay=i==0))
                # sf.write(f'{i}.wav', audio, 24000)
            # result = next(generator)
            # print(result)
            # print(result.output)
            # return result.audio


# pipeline = KPipeline(lang_code='a', repo_id="hexgrad/Kokoro-82M")

# generator = pipeline(text, voice='af_heart')
# for i, (gs, ps, audio) in enumerate(generator):
#     print(i, gs, ps)
#     display(Audio(data=audio, rate=24000, autoplay=i==0))
#     sf.write(f'{i}.wav', audio, 24000)