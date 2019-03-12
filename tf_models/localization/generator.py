import keras
import numpy as np
from keras.utils import to_categorical
from .keras_utils import minmax2center
"""
Reference : https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
"""


class DataGenerator(keras.utils.Sequence):
    'Generates Localization dataset for Keras'

    def __init__(self, dataset, batch_size=32, shuffle=True):
        'Initialization'
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_classes = self.dataset.labels.max() + 1
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return len(self.dataset) // self.batch_size

    def __getitem__(self, index):
        'Generate one batch of data'
        images, coords, labels = self.dataset[self.batch_size * index:
                                              self.batch_size * (index + 1)]

        # Add Channel axis (batch_size, 28, 28) -> (batch_size, 28, 28, 1)
        images = images[..., np.newaxis]

        points = minmax2center(coords)
        points= points.reshape(-1, 4)

        labels = to_categorical(labels, self.num_classes)
        labels = labels.reshape(-1, self.num_classes)
        return images, [points, labels]

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        if self.shuffle:
            self.dataset.shuffle()