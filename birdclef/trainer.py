"""BirdCLEF Trainer."""
import tensorflow as tf

from birdclef.dataset import Dataset


class Trainer:
    def __init__(self, config, model: tf.keras.Model, dataset: Dataset):
        self.config = config
        self.model = model
        self.dataset = dataset

    def train(self):
        pass

    def validate(self):
        pass

    def test(self):
        pass
