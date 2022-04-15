from typing import NamedTuple


class Config(NamedTuple):
    # Training
    batch_size: int = 8
    # Preprocessing
    max_audio_secs: int = 10
    sample_rate: int = 32000
    n_mels: int = 80
    n_fft: int = 1024
    window_length: int = 512
    hop_length: int = 256
