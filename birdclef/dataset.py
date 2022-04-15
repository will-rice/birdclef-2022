"""BirdCLEF Dataset"""
import ast
import json
import random
import typing
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import tensorflow as tf

from birdclef.modeling.config import Config


class Sample(typing.NamedTuple):
    """A single sample."""

    audio: typing.Union[np.ndarray, tf.TensorSpec, tf.TensorShape]
    mel_spectrogram: typing.Union[np.ndarray, tf.TensorSpec, tf.TensorShape]
    labels: typing.Union[np.ndarray, tf.TensorSpec, tf.TensorShape]


class Dataset:
    """Simple Birdclef Dataset.

    Note: There is no test dataset due to it being a notebook competition
    """

    def __init__(self, config: Config, data_path: Path):
        self.config = config
        self.data_path = data_path
        self.max_audio_length = self.config.sample_rate * self.config.max_audio_secs
        self.max_spec_length = 313

        self.train_meta = pd.read_csv(data_path / "train_metadata.csv")
        self.test_data = pd.read_csv(data_path / "test.csv")
        self.ebird_data = pd.read_csv(data_path / "eBird_Taxonomy_v2021.csv")
        self.sample_submission = pd.read_csv(data_path / "sample_submission.csv")

        self.labels = list(set(self.train_meta["primary_label"].values))
        for row in self.train_meta.index:
            self.labels.extend(
                ast.literal_eval(self.train_meta.loc[row, "secondary_labels"])
            )
        self.labels = ["nocall"] + list(set(self.labels))
        print(self.labels)

        self.label_map = {v: k for k, v in enumerate(self.labels)}

        self.category_encoder = tf.keras.layers.CategoryEncoding(
            len(self.labels), "multi_hot"
        )

        with open(data_path / "scored_birds.json") as f:
            self.scored_birds = json.load(f)

        samples = self.train_meta[
            ["primary_label", "secondary_labels", "filename"]
        ].values

        random.shuffle(samples)
        train_samples = samples[: int(0.6 * len(samples))]
        val_samples = samples[-int(0.4 * len(samples)) :]

        self._train = self.load(train_samples).repeat()
        self._validate = self.load(val_samples)

    def load(self, samples):
        """Load samples into tf.data.Dataset."""
        return (
            tf.data.Dataset.from_generator(
                lambda: self.generate(samples),
                output_signature=Sample(
                    audio=tf.TensorSpec((None,), tf.float32),
                    mel_spectrogram=tf.TensorSpec(
                        (None, self.config.n_mels), tf.float32
                    ),
                    labels=tf.TensorSpec((None,), tf.int32),
                ),
            )
            .padded_batch(
                self.config.batch_size,
                padded_shapes=Sample(
                    audio=tf.TensorShape((self.max_audio_length,)),
                    mel_spectrogram=tf.TensorShape((None, self.config.n_mels)),
                    labels=tf.TensorShape((None,)),
                ),
            )
            .prefetch(tf.data.AUTOTUNE)
        )

    def generate(self, samples):
        """Generate a single batch from a collection of samples."""
        for primary_label, secondary_labels, filename in samples:
            primary_label = self.label_map[primary_label]
            labels = [primary_label]

            if secondary_labels:
                secondary_labels = ast.literal_eval(secondary_labels)
                for label in secondary_labels:
                    secondary_label = self.label_map.get(label)
                    if secondary_label:
                        labels.append(secondary_label)

            labels_encoded = self.category_encoder(labels)

            audio, sr = librosa.load(
                self.data_path / "train_audio" / filename,
                sr=self.config.sample_rate,
            )

            # toss out longbois
            if len(audio) > self.max_audio_length:
                continue

            spectrogram = librosa.feature.melspectrogram(
                y=audio,
                sr=sr,
                n_mels=self.config.n_mels,
                n_fft=self.config.n_fft,
                hop_length=self.config.hop_length,
                win_length=self.config.window_length,
            )
            log_spectrogram = librosa.power_to_db(spectrogram, ref=np.max)

            yield Sample(
                audio=audio,
                mel_spectrogram=log_spectrogram.transpose(),
                labels=labels_encoded,
            )

    @property
    def train(self):
        """Training dataset."""
        return self._train

    @property
    def validate(self):
        """Validation dataset."""
        return self._validate

    @property
    def test(self):
        raise NotImplementedError
