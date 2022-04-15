import tensorflow as tf
from transformers import TFViTModel


class PretrainedModel(tf.keras.Model):
    """Pretrained ViT model with a classification head."""

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
