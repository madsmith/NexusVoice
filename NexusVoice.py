import json
import logging
import numpy as np
import io
import omegaconf
from openwakeword.model import Model
from openai import OpenAI
from pathlib import Path
from pydub import AudioSegment
from pydub.playback import play
import pyaudio
from silero import silero_stt
import silero_vad
import time
import torch
import webrtcvad
from transformers import WhisperProcessor, WhisperForConditionalGeneration

logging.basicConfig(level=logging.WARN, format="%(message)s")

from audio_utils import AudioBuffer, save_recording, save_recording_mp3
from init import initialize_openwakeword
from ai_agents import Agent, AgentManager, AudioInferenceEngine
from config import (
    AUDIO_FORMAT,
    NUMPY_AUDIO_FORMAT,
    AUDIO_SAMPLE_RATE,
    AUDIO_CHANNELS,
    WAKE_WORD_AUDIO_CHUNK,
    VAD_AUDIO_CHUNK,
    SILERO_VAD_AUDIO_CHUNK,
    AUDIO_CHUNK,
    VAD_SILENCE_DURATION,
    VAD_ACTIVATION_THRESHOLD,
    INFERENCE_FRAMEWORK,
    ACTIVATION_THRESHOLD
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class NexusVoice:
    def __init__(self, config):
        self.config = config
        self.audio_stream = None
        self.model = None
        self.silero_vad = None
        self.silero_stt_model = None
        self.silero_stt_decoder = None
        self.chat_manager = None
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.debug(f"Using device: {self.device}")

        self.openai = OpenAI(api_key=config.openai.api_key)

    def run(self):
        self.initalize_models()

        self.open_audio_stream()

        wake_word_buffer = AudioBuffer(AUDIO_FORMAT)
        vad_buffer = AudioBuffer(AUDIO_FORMAT)
        recording_buffer = AudioBuffer(AUDIO_FORMAT)
        silero_vad_buffer = AudioBuffer(AUDIO_FORMAT)

        is_recording = False
        is_listening = True
        silence_duration = 0.0
        bytes_per_frame = pyaudio.get_sample_size(AUDIO_FORMAT)

        def bytes_to_frames(byte_count):
            return byte_count // bytes_per_frame
        def bytes_to_seconds(byte_count):
            return byte_count / bytes_per_frame / AUDIO_SAMPLE_RATE
        def frames_to_seconds(frame_count):
            return frame_count / AUDIO_SAMPLE_RATE

        # buffer 80ms of audio
        logger.info("NexusVoice is ready!")
        while True:
            # How much audio is needed
            wake_word_needed = WAKE_WORD_AUDIO_CHUNK - wake_word_buffer.frame_count()
            vad_needed = VAD_AUDIO_CHUNK - vad_buffer.frame_count()
            silero_vad_needed = SILERO_VAD_AUDIO_CHUNK - silero_vad_buffer.frame_count()
            read_chunk_size = max(min(wake_word_needed, vad_needed, silero_vad_needed), AUDIO_CHUNK)

            # Read audio from the stream
            #logger.debug(f"Reading {read_chunk_size/16} ms of audio...")
            raw_audio = self.audio_stream.read(read_chunk_size, exception_on_overflow=False)
            #logger.debug(f"Read {len(raw_audio) // bytes_per_frame /16} ms of audio")
            #print(f"Read {len(raw_audio) // bytes_per_frame} bytes of requested {read_chunk_size} bytes of audio")

            # Add to relevant buffers
            if is_listening:
                wake_word_buffer.append(raw_audio)
            vad_buffer.append(raw_audio)
            silero_vad_buffer.append(raw_audio)
            if is_recording:
                recording_buffer.append(raw_audio)
                #print(f"Recording {frames_to_seconds(read_chunk_size)} seconds Total: {frames_to_seconds(recording_buffer.frame_count())} seconds")

            # Check if wake word detected
            if wake_word_buffer.frame_count() >= WAKE_WORD_AUDIO_CHUNK:
                audio = wake_word_buffer.get_bytes()
                wake_word_buffer.clear()
                
                np_audio = np.frombuffer(audio, dtype=NUMPY_AUDIO_FORMAT)
                detection = self.model.predict(np_audio)

                # Check wake words
                #print("Detection: ", detection)
                if any(value > ACTIVATION_THRESHOLD for value in detection.values()):
                    detected_wake_words = [key for key, value in detection.items() if value > ACTIVATION_THRESHOLD]
                    logger.debug(f"\nWake word detected! {detected_wake_words}")
                    self.reset_model()
                    is_listening = False
                    is_recording = True
                    logger.debug("Recording...")
                    silence_duration = 0.0

            # Check if Silero VAD silence detected
            if silero_vad_buffer.frame_count() >= SILERO_VAD_AUDIO_CHUNK:
                audio = silero_vad_buffer.get_bytes()
                silero_vad_buffer.clear()

                if len(audio) > SILERO_VAD_AUDIO_CHUNK:
                    # Split np_audio into a chunk of SILERO_VAD_AUDIO_CHUNK and the remaining bytes
                    split_index = SILERO_VAD_AUDIO_CHUNK * pyaudio.get_sample_size(AUDIO_FORMAT)
                    overflow = audio[split_index:]
                    kept_audio = audio[:split_index]

                    silero_vad_buffer.append(overflow)
                    audio = kept_audio

                np_audio = np.frombuffer(audio, dtype=NUMPY_AUDIO_FORMAT)
                np_audio = np_audio.astype(np.float32)
                np_audio = np_audio / np.iinfo(np.int16).max
                tensor_audio = torch.from_numpy(np_audio)
                vad_score = self.silero_vad(tensor_audio, AUDIO_SAMPLE_RATE).item()
                is_speech = vad_score > VAD_ACTIVATION_THRESHOLD

                if is_recording:
                    if is_speech:
                        silence_duration = 0
                    else:
                        # Silence detected
                        silence_duration += frames_to_seconds(SILERO_VAD_AUDIO_CHUNK)
                        if silence_duration >= VAD_SILENCE_DURATION:
                            logger.debug(f"Recording silence for {silence_duration} seconds")
                            is_recording = False
                            is_listening = True
                            silence_duration = 0.0
                            # Process recording
                            recording = recording_buffer.get_bytes()
                            recording_buffer.clear()

                            # filename_mp3 = self._generate_filename("mp3")
                            # filename_wav = self._generate_filename("wav")
                            # save_recording(recording, filename_wav)
                            # save_recording_mp3(recording, filename_mp3)

                            # time_stt = time.perf_counter()
                            # self.silero_stt(recording)
                            # print("STT time: ", time.perf_counter() - time_stt)

                            time_whisper_stt = time.perf_counter()
                            transcription = self.whisper_stt(recording)
                            logger.debug(f"Whisper STT time: {time.perf_counter() - time_whisper_stt}")

                            time_llama_agent = time.perf_counter()
                            future_result = self.llama_agent.process_request(transcription)
                            response = future_result.result()
                            logger.debug(f"LLAMA Agent time: {time.perf_counter() - time_llama_agent}")

                            time_llm_process = time.perf_counter()
                            self.process_agent_response(response)
                            logger.debug(f"LLM Process time: {time.perf_counter() - time_llm_process}")


                            # time_openai_stt = time.perf_counter()
                            # self.openai_stt(filename_mp3)
                            # print("OpenAI STT time: ", time.perf_counter() - time_openai_stt)

                            recording_duration = (len(recording) // pyaudio.get_sample_size(AUDIO_FORMAT)) / AUDIO_SAMPLE_RATE
                            logger.debug(f"Recorded {recording_duration} seconds of audio")
                            logger.debug("Listening...")
                # elif is_recording:
                #     print(f"Recording speech {frames_to_seconds(read_chunk_size)} seconds Total: {frames_to_seconds(recording_buffer.frame_count())} seconds")
                    


    def process_agent_response(self, response):
        # If response is json string, parse it
        try:
            response = json.loads(response)
            self.process_agent_json(response)
        except:
            self.process_agent_speach(response)

    def process_agent_json(self, response):
        logger.debug("TODO: Process agent JSON response")

        
    def process_agent_speach(self, response):
        logger.debug(response)
        self.speak_text(response)

    def speak_text(self, text):
        speech_file = Path("./response.mp3")

        response = self.openai.audio.speech.create(
            model="tts-1",
            voice="nova",
            input=text
        )

        response.write_to_file(speech_file)

        audio_bytes = response.content
        audio = AudioSegment.from_file(io.BytesIO(audio_bytes), format="mp3")
        play(audio)
        pass

    def initalize_models(self):
        logger.info("Initializing NexusVoice model...")
        models = [
            "hey jarvis",
            "./models/my_nexus-3.onnx"
            #"./models/my_nexus.onnx",
            #"./models/scarlette.onnx" # model seems broken
        ]

        self.model = Model(wakeword_models=models, vad_threshold=0.2, enable_speex_noise_suppression=True, inference_framework=INFERENCE_FRAMEWORK)

        logger.info("Initializing Voice Activity Detector...")
        self.voice_activity_detector = webrtcvad.Vad(3)

        self.silero_vad = silero_vad.load_silero_vad()

        logger.info("Initializing Silero STT...")
        self.silero_device = torch.device('cpu')
        model, decoder, utils = silero_stt(language='en', device=self.silero_device)
        # This is a fucking dumb interface, why are AI developers so bad at design?
        (read_batch, split_into_batches, read_audio, prepare_model_input) = utils
        self.silero_stt_model = model
        self.silero_stt_decoder = decoder
        self._silerto_stt_prepare_model_input = prepare_model_input

        logger.info("Initializing Whisper STT...")
        model = self.config.whisper.processor.get("model", "openai/whisper-large-v2")
        self.whisper_engine = AudioInferenceEngine(model)
        self.whisper_engine.initialize()


        # processor_model = self.config.whisper.processor.get("model", "openai/whisper-large-v2")
        # generator_model = self.config.whisper.processor.get("model", "openai/whisper-large-v2")

        # logger.debug(f"  Processor model: {processor_model}")
        # logger.debug(f"  Generator model: {generator_model}")
        # self.whisper_processor = WhisperProcessor.from_pretrained(processor_model, return_attention_mask=True)
        # model = WhisperForConditionalGeneration.from_pretrained(generator_model).to(self.device)
        # model.config.forced_decoder_ids = None
        # model.generation_config.forced_decoder_ids = None
        # #model.config.forced_decoder_ids = self.whisper_processor.get_decoder_prompt_ids(language="english", task="transcribe")
        # self.whisper_model = model

        # # Warm up model
        # dummy_audio = np.zeros(3000, dtype=np.int16)
        # inputs = self.whisper_processor(
        #     dummy_audio,
        #     sampling_rate=AUDIO_SAMPLE_RATE, 
        #     return_tensors="pt")
        # dummy_input = inputs.input_features.to(self.device)
        # dummy_attention_mask = (
        #         inputs.attention_mask if "attention_mask" in inputs else torch.ones((1, 80))
        #     ).to(self.device)
        # self.whisper_model.generate(dummy_input, attention_mask=dummy_attention_mask)

        logger.info("Initializing LLM...")
        self.agent_manager = AgentManager(self.config)
        self.llama_agent = self.agent_manager.get_agent("agent-llama")
        self.llama_agent.wait_until_ready()
        


    def open_audio_stream(self):
        self.audio_stream = pyaudio.PyAudio().open(
            format=AUDIO_FORMAT,
            channels=AUDIO_CHANNELS,
            rate=AUDIO_SAMPLE_RATE,
            input=True,
            frames_per_buffer=AUDIO_CHUNK
        )
        logger.info("NexusVoice is listening...")
        
    def _generate_filename(self, extension):
        time_format = time.strftime("%Y-%m-%d-%H-%M-%S")
        return Path(f"recordings/recording-{time_format}.{extension}")

    def silero_stt(self, audio, sample_rate=AUDIO_SAMPLE_RATE):

        np_audio = np.frombuffer(audio, dtype=NUMPY_AUDIO_FORMAT).copy()
        audio_tensor = torch.from_numpy(np_audio).float()
        audio_tensor = audio_tensor.unsqueeze(0)

        # It appears silero natively uses 16kHz audio so we don't need to resample
        # unless we change our audio sample rate.  Unfortunately it doesn't provide a
        # clear api to communicate its desired sample rate or associate the input with
        # a sample rate so this presumption may be incorrect.

        # This loads the audio input onto the appropriate torch tensor processing device
        input = self._silerto_stt_prepare_model_input(audio_tensor, device=self.silero_device)

        output = self.silero_stt_model(input)

        # I have no idea when there would be multiple values in output, seems the first output
        # is the input we piped in so we'll return that.  Unfortunately there's no documentation
        # on how this works.  So we'll assume a batch size of 1 and get the result.
        result = self.silero_stt_decoder(output[0].cpu())
        print(result)
        return result
    
    def whisper_stt(self, recording):
        # Convert audio bytes into numpy array
        np_audio = np.frombuffer(recording, dtype=NUMPY_AUDIO_FORMAT)

        transcription = self.whisper_engine.infer(np_audio, AUDIO_SAMPLE_RATE)
        return transcription
        # # Tokenize audio
        # inputs = self.whisper_processor(
        #     np_audio,
        #     sampling_rate=AUDIO_SAMPLE_RATE,
        #     return_tensors="pt").to(self.device)
        
        # input_features = inputs.input_features
        # attention_mask = inputs.attention_mask if "attention_mask" in inputs else None

        # # Inference
        # predicted_ids = self.whisper_model.generate(
        #     input_features,
        #     language="en",
        #     task="transcribe",
        #     attention_mask=attention_mask)
        
        # # Decoding
        # transcription = self.whisper_processor.batch_decode(predicted_ids, skip_special_tokens=True)

        # return transcription[0]

    def openai_stt(self, filename):
        response = self.openai.audio.transcriptions.create(
            model="whisper-1",
            file=filename
        )

        if hasattr(response, "text"):
            print(response.text) 
            return response.text
        
        logger.warning(f"OpenAI STT returned unexpected response: {response}")
        return None

    def reset_model(self):
        self.model.reset()

def main():
    initialize_openwakeword()

    config = omegaconf.OmegaConf.load("config.yml")

    app = NexusVoice(config)

    app.run()

if __name__ == '__main__':
    main()