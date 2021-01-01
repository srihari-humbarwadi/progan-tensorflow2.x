import numpy as np
import tensorflow as tf
from absl import logging


class BaseNetwork(tf.keras.Model):

    def __init__(self, max_resolution, use_equalized_layers, name, **kwargs):
        super(BaseNetwork, self).__init__(name=name, **kwargs)

        self.min_depth = 2
        self.current_depth = 2
        self.max_resolution = max_resolution
        self.max_depth = int(np.log2(max_resolution))
        self.use_equalized_layers = use_equalized_layers
        self._current_depth = tf.Variable(self.current_depth,
                                          trainable=False,
                                          name='current_depth',
                                          dtype=tf.uint8)

    @staticmethod
    def _nf(stage, fmap_base=8192, fmap_max=512, fmap_decay=1.0):
        return min(int(fmap_base / (2.0**(stage * fmap_decay))), fmap_max)

    def assign_depth(self, depth):
        depth = int(depth)
        if depth != self.current_depth:
            logging.info('Changing depth from {} to {}'.format(
                self.current_depth, depth))
            self.current_depth = depth
            self._current_depth.assign(depth)

    def increment_depth(self):
        assert self.current_depth != self.max_depth + 1, 'Max Depth Exceeded'
        logging.info('Incrementing depth from {} to {}'.format(
            self.current_depth, self.current_depth + 1))
        self.current_depth += 1
        self._current_depth.assign_add(1)

    def restore_current_depth(self):
        depth = self._current_depth.numpy()
        if depth != self.current_depth:
            logging.info('Changing depth from {} to {}'.format(
                self.current_depth, depth))
            self.current_depth = depth
