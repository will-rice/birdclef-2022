"""BirdCLEF Trainer."""
import json
from pathlib import Path

import tensorflow as tf
import wandb
from sklearn import metrics

from birdclef.dataset import Dataset


class Trainer:
    """Simple model trainer."""

    def __init__(self, config, model: tf.keras.Model, dataset: Dataset, log_dir: Path):
        self.config = config
        self.model = model
        self.dataset = dataset
        self.log_dir = log_dir
        self.loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
        self.metric_fn = tf.keras.metrics.CategoricalAccuracy()
        self.train_loss = tf.keras.metrics.Mean()
        self.train_accuracy = tf.keras.metrics.Mean()
        self.train_f1_score = tf.keras.metrics.Mean()
        self.train_step = 0
        self.val_loss = tf.keras.metrics.Mean()
        self.val_accuracy = tf.keras.metrics.Mean()
        self.val_f1_score = tf.keras.metrics.Mean()
        self.val_step = 0

        self.optimizer = tf.keras.optimizers.Adam()

    def train(self):
        """Run model training."""
        for batch in self.dataset.train.take(self.config.steps_per_checkpoint):
            with tf.GradientTape() as tape:
                logits = self.model(batch.mel_spectrogram, training=True)

                loss_value = self.loss_fn(batch.labels, logits)
                accuracy = self.metric_fn(batch.labels, logits)

            grads = tape.gradient(loss_value, self.model.trainable_weights)
            self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))

            f1_score = metrics.f1_score(
                tf.cast(tf.argmax(batch.labels), tf.float32),
                tf.cast(tf.nn.softmax(logits), tf.float32),
                labels=None,
                average="macro",
                zero_division=1,
            )

            self.train_loss.update_state(loss_value)
            self.train_accuracy.update_state(accuracy)
            self.train_f1_score.update_state(f1_score)

            wandb.log(
                {
                    "train_loss": self.train_loss.result(),
                    "train_accuracy": self.train_accuracy.result(),
                    "train_f1_score": self.train_f1_score.result(),
                }
            )

            print(
                "Training loss (for one batch) at step %d: %.4f %.4f %.4f"
                % (
                    self.step,
                    float(self.train_loss.result()),
                    float(self.train_accuracy.result()),
                    float(self.train_f1_score.result()),
                )
            )
            self.train_step += 1

    def validate(self):
        """Run model validation."""
        for batch in self.dataset.validate:
            logits = self.model(batch.mel_spectrogram, training=False)

            val_loss = self.loss_fn(batch.labels, logits)
            val_accuracy = self.metric_fn(batch.labels, logits)

            f1_score = metrics.f1_score(
                tf.cast(tf.argmax(batch.labels), tf.float32),
                tf.cast(tf.nn.softmax(logits), tf.float32),
                labels=None,
                average="macro",
                zero_division=1,
            )

            self.val_loss.update_state(val_loss)
            self.val_accuracy.update_state(val_accuracy)
            self.val_f1_score.update_state(f1_score)

        wandb.log(
            {
                "val_loss": self.train_loss.result(),
                "val_accuracy": self.train_accuracy.result(),
                "val_f1_score": self.val_f1_score.result(),
            }
        )
        tf.saved_model.save(self.model, str(self.log_dir / "model"))

        with open(self.log_dir / "labels.json", "w") as file:
            json.dump(self.dataset.label_map, file)

    def test(self):
        """Run model testing."""
        raise NotImplementedError

    @property
    def step(self):
        """Current training step."""
        return self.train_step
