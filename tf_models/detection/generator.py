import keras
import numpy as np
from .anchors import Anchor
"""
Reference : https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
"""


class RPNDataGenerator(keras.utils.Sequence):
    'Generates Region-Proposal-Network dataset for Keras'

    def __init__(self, dataset, anchor:Anchor, batch_size=16, shuffle=True):
        'Initialization'
        self.dataset = dataset
        self.anchor = anchor
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_classes = self.dataset.labels.max() + 1
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return len(self.dataset) // self.batch_size

    def __getitem__(self, index):
        'Generate one batch of data'
        images, labels, points = self.dataset[self.batch_size * index:
                                              self.batch_size * (index + 1)]

        # Add Channel axis (batch_size, 28, 28) -> (batch_size, 28, 28, 1)
        images = np.stack(images,axis=0)[..., np.newaxis]

        anchor_clfs, delta_regs = self.anchor.points_to_deltas(points, images)

        return images, [anchor_clfs, delta_regs]

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        if self.shuffle:
            self.dataset.shuffle()