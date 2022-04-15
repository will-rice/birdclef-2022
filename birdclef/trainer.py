"""BirdCLEF Trainer."""
import tensorflow as tf

from birdclef.dataset import Dataset


class Trainer:
    """Simple model trainer."""

    def __init__(self, config, model: tf.keras.Model, dataset: Dataset):
        self.config = config
        self.model = model
        self.dataset = dataset
        self.loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.metric_fn = tf.keras.losses.Accuracy()
        self.train_loss = tf.keras.metrics.Mean()
        self.train_accuracy = tf.keras.metrics.Mean()
        self.val_loss = tf.keras.metrics.Mean()
        self.val_accuracy = tf.keras.metrics.Mean()

        self.optimizer = tf.keras.optimizers.Adam()

    def train(self):
        """Run model training."""
        for batch in self.dataset.train:
            with tf.GradientTape() as tape:
                logits = self.model(batch.audio, training=True)

                loss_value = self.loss_fn(batch.encoded_labels, logits)
                accuracy = self.metric_fn(batch.encoded_labels, logits)

            grads = tape.gradient(loss_value, self.model.trainable_weights)
            self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))

            self.train_loss.update_state(loss_value)
            self.train_accuracy.update_state(accuracy)

    def validate(self):
        """Run model validation."""
        for batch in self.dataset.validate:
            logits = self.model(batch.audio, training=False)

            val_loss = self.loss_fn(batch.encoded_labels, logits)
            val_accuracy = self.metric_fn(batch.encoded_labels, logits)

            self.val_loss.update_state(val_loss)
            self.val_accuracy.update_state(val_accuracy)

    def test(self):
        """Run model testing."""
        raise NotImplementedError
