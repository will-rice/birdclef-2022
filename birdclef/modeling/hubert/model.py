import tensorflow as tf
from transformers import TFHubertModel


class PretrainedModel(tf.keras.Model):
    def __init__(self, num_labels, **kwargs):
        super().__init__(**kwargs)
        self.base_model = TFHubertModel.from_pretrained("facebook/hubert-base-ls960")
        self.base_model.trainable = True
        self.classifier = tf.keras.layers.Dense(num_labels)

    def call(self, inputs, training=False):
        outputs = self.base_model(inputs, training=training)[0]
        outputs = tf.reduce_mean(outputs, 1)
        logits = self.classifier(outputs)
        return logits
