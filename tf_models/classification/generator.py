import keras
import numpy as np
"""
Reference : https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
"""


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, dataset, batch_size=32, shuffle=True):
        'Initialization'
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return len(self.dataset) // self.batch_size

    def __getitem__(self, index):
        'Generate one batch of data'
        X, Y = self.dataset[self.batch_size * index:
                            self.batch_size * (index + 1)]

        # Add Channel axis (batch_size, 28, 28) -> (batch_size, 28, 28, 1)
        X = X[..., np.newaxis]
        return X, Y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        if self.shuffle:
            self.dataset.shuffle()
