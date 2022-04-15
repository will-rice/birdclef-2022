from typing import Any, Optional

import tensorflow as tf
from tensorflow import Tensor

from birdclef.modeling.conformer.config import ConformerConfig
from birdclef.modeling.conformer.layers import ConformerEncoder


class Conformer(tf.keras.Model):
    """Conformer Model."""

    def __init__(self, config: ConformerConfig, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.encoder = ConformerEncoder(
            units=config.encoder_units,
            num_layers=config.encoder_num_layers,
            num_attention_heads=config.num_attention_heads,
            feed_forward_expansion_factor=config.feed_forward_expansion_factor,
            encoder_dropout=config.encoder_dropout,
            attention_dropout=config.attention_dropout,
            depthwise_kernel_size=config.depthwise_kernel_size,
        )
        self.flatten = tf.keras.layers.Flatten()
        self.head = tf.keras.layers.Dense(units=config.num_classes, name="head")

    def call(
        self,
        inputs: Tensor,
        attention_mask: Optional[Tensor] = None,
        training: bool = False,
    ) -> Tensor:
        """Forward Pass."""
        out = self.encoder(inputs, attention_mask=attention_mask, training=training)
        out = self.flatten(out)
        out = self.head(out)
        return out
