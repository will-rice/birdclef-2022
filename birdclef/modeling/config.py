"""Base config"""
from dataclasses import asdict, dataclass


@dataclass
class Config:
    """Base config class."""

    # Training
    batch_size: int = 8
    max_steps: int = 100000
    steps_per_checkpoint = 1000
    # Preprocessing
    max_audio_secs: int = 11
    sample_rate: int = 32000
    n_mels: int = 80
    n_fft: int = 1024
    window_length: int = 512
    hop_length: int = 256

    def asdict(self):
        """Converts dataclass to dictionary."""
        return asdict(self)
