from kokoro import KPipeline, KModel
import re
import threading
import torch

from nexusvoice.utils.logging import get_logger
logger = get_logger(__name__)

from nexusvoice.ai.InferenceEngine import InferenceEngineBase

TEMPERATURE_FIX_PATTERN = re.compile(r"(\d+\.?\d*)\s*[°º]\s*([FfCc])")

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

        # TODO: make this function a generator
        with self.lock:
            processed_inputs = self.preprocess_for_tts(inputs)
            generator = self.pipeline(processed_inputs, **inference_params)

            result_audio = []
            for i, (gs, ps, audio) in enumerate(generator):
                logger.trace(f"Generated speech {i}: {gs} -> {ps}")
                result_audio.append(audio)
                # print(i, gs, ps)
                # # display(Audio(data=audio, rate=24000, autoplay=i==0))
                # sf.write(f'{i}.wav', audio, 24000)
            # result = next(generator)
            # print(result)
            # print(result.output)
            # return result.audio

            # Join the audio
            result_audio = torch.cat(result_audio, dim=0)
            return result_audio

    def preprocess_for_tts(self, text):
        # Convert temperature patterns like 55.9°F or 21°C
        def replace_temp(match):
            """ Insert a space between the number and the degree symbol """
            unit = match.group(2).upper()
            unit_word = "Fahrenheit" if unit == "F" else "Celsius"
            return f"{match.group(1)} °{unit_word}"

        # Adjust °F or °C after number
        text = TEMPERATURE_FIX_PATTERN.sub(replace_temp, text)

        return text

# pipeline = KPipeline(lang_code='a', repo_id="hexgrad/Kokoro-82M")

# generator = pipeline(text, voice='af_heart')
# for i, (gs, ps, audio) in enumerate(generator):
#     print(i, gs, ps)
#     display(Audio(data=audio, rate=24000, autoplay=i==0))
#     sf.write(f'{i}.wav', audio, 24000)