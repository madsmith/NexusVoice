import logging

# Setup logging
from nexusvoice.utils.logging import get_logger

logger = get_logger(__name__)
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)

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
    arg_parser.add_argument("--long", action="store_true", help="Use long text for TTS.")
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
    if args.long:
        text = ("""
            China manipulates its currency, the Renminbi (RMB), also known as the Chinese Yuan, through a combination of monetary and fiscal policies. This is often referred to as a managed currency regime.
To artificially weaken the value of the RMB, China:

1. Sells RMB on the foreign exchange market to increase its supply and reduce demand.
2. Lowers interest rates to attract foreign investors, who buy RMB and drive down its value.
3. Implements policies to encourage exports, such as subsidies or tax breaks, which can lead to an increase in exports and a decrease in imports, further weakening the RMB.
4. Uses its influence as a major economy to persuade other countries to hold more RMB in their foreign exchange reserves, increasing its supply and reducing demand.

By doing so, China aims to:

* Make its exports cheaper and more competitive in the global market
* Encourage foreign investment and trade
* Increase the value of its currency in the eyes of other countries, making its exports more attractive
* Reduce the cost of imports and make its economy more competitive

It's worth noting that China's currency manipulation is a subject of ongoing debate and criticism from other countries, including the United States, which has imposed tariffs on Chinese goods in response to what it sees as unfair trade practices. 
        """)
    elif args.speech:
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

        # Break audio into chunks of 1024 samples
        chunk_size = 1024
        for i in range(0, len(audio), chunk_size):
            stream.write(audio[i:i + chunk_size])
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