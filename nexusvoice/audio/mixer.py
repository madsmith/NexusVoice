from typing import Tuple

import numpy as np


class ChannelMixer:
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ):
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        assert self.in_channels > 0 and self.out_channels > 0, "Channel counts must be positive"
        # Precompute mixing matrix when a specialized mapping is known
        self._mix_matrix: np.ndarray | None = None
        if self.out_channels == 1 and self.in_channels >= 1:
            self._mix_matrix = self._build_mono_downmix_matrix(self.in_channels)
        elif self.out_channels == 2 and self.in_channels >= 2:
            self._mix_matrix = self._build_stereo_downmix_matrix(self.in_channels)

    def _dtype_from_sample_size(self, sample_size: int) -> Tuple[np.dtype, bool]:
        """
        Map bytes-per-sample to numpy dtype.
        Returns (dtype, is_integer)
        Matches logic used in AudioData.get_type_info().
        """
        if sample_size == 1:
            return (np.uint8, True)
        elif sample_size == 2:
            return (np.int16, True)
        elif sample_size == 4:
            return (np.int32, True)
        elif sample_size == 8:
            return (np.float64, False)
        else:
            raise ValueError(f"Unsupported sample size: {sample_size}")

    def _apply_matrix(self, data_f: np.ndarray, W: np.ndarray) -> np.ndarray:
        """Apply mixing matrix W (out_ch x in_ch) to data_f (time x in_ch)."""
        return data_f @ W.T

    def _build_stereo_downmix_matrix(self, in_channels: int) -> np.ndarray:
        """
        Build a 2 x in_channels downmix matrix assuming SMPTE/WAV channel order:
        2.0: [L, R]
        3.0: [L, R, C]
        4.0: [L, R, Ls, Rs]
        5.1: [L, R, C, LFE, Ls, Rs]
        6.1: [L, R, C, LFE, Ls, Rs, Cs]
        7.1: [L, R, C, LFE, Ls, Rs, Lb, Rb]

        Coeffs based on common ITU/Dolby practice:
        - Center @ 0.707 to both L/R
        - Surrounds @ 0.707 to respective L/R
        - Back surrounds (Lb/Rb) @ 0.707 to respective L/R
        - LFE mixed lightly @ 0.5 to both (could be 0.0 if desired)
        Unknown extra channels are averaged equally to both.
        """
        W = np.zeros((2, in_channels), dtype=np.float32)

        def set_safe(row: int, col: int, val: float):
            if 0 <= col < in_channels:
                W[row, col] += val

        # base L/R always pass-through
        set_safe(0, 0, 1.0)  # L -> L
        set_safe(1, 1, 1.0)  # R -> R

        # Center -> both @ 0.707
        set_safe(0, 2, 0.707)
        set_safe(1, 2, 0.707)

        # LFE -> both @ 0.5
        set_safe(0, 3, 0.5)
        set_safe(1, 3, 0.5)

        # Ls, Rs -> respective @ 0.707
        set_safe(0, 4, 0.707)  # Ls -> L
        set_safe(1, 5, 0.707)  # Rs -> R

        # Cs (6.1) -> both @ 0.707 (index 6)
        set_safe(0, 6, 0.707)
        set_safe(1, 6, 0.707)

        # Lb, Rb (7.1) -> respective @ 0.707 (idx 6,7 or 7,8 depending on presence of Cs)
        # Already set Cs at 6; for 7.1 SMPTE, Lb=6, Rb=7? Many docs put Lb=6,Rb=7. We'll also set 7,8 just in case.
        set_safe(0, 6, 0.707)
        set_safe(1, 7, 0.707)
        set_safe(0, 7, 0.707)
        set_safe(1, 8, 0.707)

        # For any remaining channels beyond what we handled, distribute equally
        handled = np.any(W != 0, axis=0)
        for idx in range(in_channels):
            if not handled[idx]:
                set_safe(0, idx, 0.5)
                set_safe(1, idx, 0.5)

        return W

    def _build_mono_downmix_matrix(self, in_channels: int) -> np.ndarray:
        """
        Build a 1 x in_channels mono downmix matrix using surround-aware weights.
        Assumed channel order (SMPTE/WAV):
        [L, R, C, LFE, Ls, Rs, (Cs|Lb), (Rb), ...]

        Coeff choices (conservative):
        - L, R: 0.5 each
        - C: 0.707
        - LFE: 0.0 (or small value like 0.25 if desired)
        - Ls, Rs: 0.5 each
        - Cs / Lb / Rb: 0.5 each
        Remaining channels: 0.5 each
        """
        W = np.zeros((1, in_channels), dtype=np.float32)

        def set_safe(col: int, val: float):
            if 0 <= col < in_channels:
                W[0, col] += val

        set_safe(0, 0.5)   # L
        set_safe(1, 0.5)   # R
        set_safe(2, 0.707) # C
        set_safe(3, 0.0)   # LFE (muted in mono by default)
        set_safe(4, 0.5)   # Ls
        set_safe(5, 0.5)   # Rs
        set_safe(6, 0.5)   # Cs or Lb
        set_safe(7, 0.5)   # Rb

        # Any additional channels contribute moderately
        handled = (W[0] != 0)
        for idx in range(in_channels):
            if not handled[idx]:
                set_safe(idx, 0.5)

        return W

    def mix(self, frames: bytes, sample_size: int) -> bytes:
        """
        Mix interleaved audio frames from in_channels -> out_channels.

        - frames: interleaved PCM bytes
        - sample_size: bytes per sample (e.g., 2 for int16)

        Returns interleaved PCM bytes with out_channels.
        """
        if self.in_channels == self.out_channels:
            return frames

        if len(frames) == 0:
            return frames

        dtype, is_int = self._dtype_from_sample_size(sample_size)

        # bytes -> numpy, shape to (time, in_channels)
        data = np.frombuffer(frames, dtype=dtype)
        if data.size % self.in_channels != 0:
            raise ValueError(
                f"Frame data size {data.size} not divisible by in_channels {self.in_channels}"
            )
        data = data.reshape(-1, self.in_channels)

        # Convert to float32 for safe mixing
        data_f = data.astype(np.float32, copy=False)

        # Use precomputed mixing matrix when available
        if self._mix_matrix is not None:
            mixed = self._apply_matrix(data_f, self._mix_matrix)
        else:
            # Generic behaviors as fallback
            if self.out_channels < self.in_channels:
                # Downmix: average across input channels, then replicate to out_channels
                mono = np.mean(data_f, axis=1, keepdims=True)
                mixed = np.repeat(mono, self.out_channels, axis=1)
            elif self.out_channels > self.in_channels:
                # Upmix: keep existing channels, fill extras with mono mix
                mono = np.mean(data_f, axis=1, keepdims=True)
                frames_count = data_f.shape[0]
                mixed = np.empty((frames_count, self.out_channels), dtype=np.float32)
                mixed[:, : self.in_channels] = data_f
                mixed[:, self.in_channels :] = np.repeat(
                    mono, self.out_channels - self.in_channels, axis=1
                )
            else:
                mixed = data_f

        # Convert back to original dtype with clipping
        if is_int:
            info = np.iinfo(dtype)
            mixed = np.clip(mixed, info.min, info.max)
            mixed = np.rint(mixed).astype(dtype, copy=False)
        else:
            # float types (assume normalized range [-1, 1] is desired)
            mixed = mixed.astype(dtype, copy=False)

        # Interleave back and to bytes
        mixed_bytes = mixed.reshape(-1).tobytes()
        return mixed_bytes