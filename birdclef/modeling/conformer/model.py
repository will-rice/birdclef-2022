from typing import Any, Optional

import tensorflow as tf
from tensorflow import Tensor
from transformers import TFViTModel

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
        self.head = tf.keras.layers.Dense(units=config.num_classes)

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


class PretrainedModel(tf.keras.Model):
    def __init__(self, num_labels, **kwargs):
        super().__init__(**kwargs)
        self.base_model = TFViTModel.from_pretrained(
            "google/vit-base-patch16-224-in21k"
        )
        self.classifier = tf.keras.layers.Dense(num_labels)

    def call(self, inputs, training=False, interpolate_pos_encoding=True):
        inputs = tf.transpose(inputs, (0, 3, 1, 2))
        outputs = self.base_model(
            inputs, training=training, interpolate_pos_encoding=interpolate_pos_encoding
        )
        logits = self.classifier(outputs.pooler_output)
        return logits
