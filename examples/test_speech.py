# Setup logging
from nexusvoice.utils.logging import get_logger

logger = get_logger(__name__)

import argparse
import pyaudio
import torch
import numpy as np

from nexusvoice.ai.TTSInferenceEngine import TTSInferenceEngine

def tensor_to_int16(audio_tensor: torch.Tensor) -> bytes:
    # Clamp values to avoid overflows
    audio_clamped = torch.clamp(audio_tensor, -1.0, 1.0)
    # Convert to int16 range

    max_int16 = np.iinfo(np.int16).max
    audio_int16 = (audio_clamped * max_int16).to(torch.int16)
    return audio_int16.numpy().tobytes()

def pronounce_voice(voice:str) -> str:
    voices = voice.split(",")

    result = ""
    for v in voices:
        parts = v.split("_")
        if len(parts) > 1:
            result += "".join([c + "." for c in parts[0].upper()]) + " " + parts[1]
        else:
            result += f"{parts[0]} "
    return result.strip()

def main():
    arg_parser = argparse.ArgumentParser(description="Test TTS Inference Engine")
    arg_parser.add_argument(
        "--voice",
        type=str,
        default="af_sky",
        help="Voice to use for TTS. Default: af_sky"
    )
    # Parse remaining arguments into speech var
    arg_parser.add_argument(
        "speech",
        nargs=argparse.REMAINDER,
        help="Text to convert to speech. Default: 'Hello, this is a test of the TTS Inference Engine.'"
    )

    args = arg_parser.parse_args()

    # Initialize TTS Inference Engine
    tts_engine = TTSInferenceEngine(model_id="hexgrad/Kokoro-82M", voices=[args.voice])
    tts_engine.initialize()

    # Generate speech
    if args.speech:
        text = " ".join(args.speech)
    else:
        # Default text if none provided
        text = "This is a test of the TTS Inference Engine."

    # Play the generated audio
    pa = pyaudio.PyAudio()

    stream = pa.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=24000,
        output=True
    )

    interactive_mode = not args.speech
    voice = args.voice

    while True:
        logger.info(f"Generating speech for text: {text}")
        tensor = tts_engine.infer(text, voice=voice)

        audio = tensor_to_int16(tensor)
        stream.write(audio)
        if not interactive_mode:
            break
        text = input("Enter text to convert to speech (or 'exit' to quit): ")
        if text.lower() == "exit":
            break
        if text.startswith(":"):
            voice = text[1:]
            text = f"Switching to voice: {pronounce_voice(voice)}"
            print(f"Switching to voice: {voice}")
            continue

    stream.stop_stream()
    stream.close()
    pa.terminate()

if __name__ == "__main__":
    main()