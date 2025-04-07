import time
import numpy as np

RATE = 16000
VOLUME = 0.5

class Note: 
    def play(self, device):
        raise NotImplementedError
    
    def note(self, name: str, semitone_offset: int = 0) -> float:
        """
        Returns the frequency of a note with optional semitone adjustment.
        Example: note("C4") → 261.63
                note("C4", 1) → C#4 ≈ 277.18
        """
        # Map note names to semitone offsets from C
        semitone_map = {
            'C': 0,  'C#': 1, 'Db': 1,
            'D': 2,  'D#': 3, 'Eb': 3,
            'E': 4,
            'F': 5,  'F#': 6, 'Gb': 6,
            'G': 7,  'G#': 8, 'Ab': 8,
            'A': 9,  'A#':10, 'Bb':10,
            'B':11
        }

        # Extract note name and octave
        name = name.strip()
        i = 1 if len(name) == 2 or name[1] not in "#b" else 2
        key = name[:i]
        
        # Default octave to 4 if not provided
        octave = int(name[i:]) if len(name) > i else 4

        # Calculate semitone distance from A4
        n = semitone_map[key] + (octave - 4) * 12 - 9 + semitone_offset

        # Frequency formula
        freq = 440.0 * (2 ** (n / 12))
        return round(freq, 2)
    
class Rest(Note):
    def __init__(self, duration):
        self.duration = duration

    def play(self, device):
        time.sleep(self.duration)

class Tone(Note):
    def __init__(self, freq, duration, fade=None):
        if isinstance(freq, str):
            freq = self.note(freq)
        self.freq = freq
        self.duration = duration
        self.fade = fade

    def play(self, device):
        # TODO: Get rate and volume from device
        device.play(self.render(rate=RATE, volume=VOLUME))
        time.sleep(self.duration)

    def render(self, rate=RATE, volume=VOLUME):
        if self.fade is None:
            self.fade = self.duration / 3
            self.fade = 0
        return self.generate_tone(self.freq, self.duration, self.fade, rate=RATE, volume=VOLUME)

    # Generate tone
    def generate_tone(self, freq, duration, end_fade=0.25, rate=RATE, volume=VOLUME):
        t = np.linspace(0, duration, int(rate * duration), False)
        tone = np.sin(freq * t * 2 * np.pi)

        # Apply linear fade-out over the last `end_fade` seconds
        if end_fade > 0 and end_fade < duration:
            fade_samples = int(rate * end_fade)
            fade = np.linspace(1.0, 0.0, fade_samples)
            tone[-fade_samples:] *= fade  # Apply fade to end of tone


        np_type = np.int16
        max_range = np.iinfo(np_type).max

        tone = (tone * (max_range * volume)).astype(np_type)
        return tone.tobytes()