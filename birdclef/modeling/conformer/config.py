"""
Conformer Config.
+------------------+---------------+---------------+---------------+
| Model            | Conformer (S) | Conformer (M) | Conformer (L) |
+------------------+---------------+---------------+---------------+
| Num Params (M)   | 10.3          | 30.7          | 118.8         |
+------------------+---------------+---------------+---------------+
| Encoder Layers   | 16            | 16            | 17            |
+------------------+---------------+---------------+---------------+
| Encoder Dim      | 144           | 256           | 512           |
+------------------+---------------+---------------+---------------+
| Attention Heads  | 4             | 4             | 8             |
+------------------+---------------+---------------+---------------+
| Conv Kernel Size | 32            | 32            | 32            |
+------------------+---------------+---------------+---------------+
| Decoder Layers   | 1             | 1             | 1             |
+------------------+---------------+---------------+---------------+
| Decoder Dim      | 320           | 640           | 640           |
+------------------+---------------+---------------+---------------+
"""
from birdclef.modeling.config import Config


class ConformerConfig(Config):
    """Conformer Config."""

    # Training
    batch_size: int = 64
    steps_per_checkpoint = 1000

    # Encoder
    encoder_num_layers: int = 16
    encoder_units: int = 144
    encoder_dropout: float = 0.1
    num_attention_heads: int = 4
    feed_forward_expansion_factor: int = 4
    attention_dropout: float = 0.1
    depthwise_kernel_size: int = 32
    num_classes: int = 425

    # Preprocessing
    max_audio_secs: int = 10
    sample_rate: int = 32000
    num_mels: int = 80
    n_fft: int = 1024
    window_size: int = 512
    hop_size: int = 256
