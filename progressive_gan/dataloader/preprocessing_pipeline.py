import functools

import numpy as np
import tensorflow as tf


class PreprocessingPipeline:

    def __init__(self, max_resolution, current_depth):
        self.max_resolution = max_resolution
        self.current_depth = current_depth

        max_depth = int(np.log2(self.max_resolution))

        downscale_factor = max_depth - current_depth
        pool_size = int(2 ** downscale_factor)

        self._downscale_a = functools.partial(
            tf.nn.avg_pool2d,
            ksize=pool_size,
            strides=pool_size,
            padding='VALID')

        self._downscale_b = functools.partial(
            tf.nn.avg_pool2d,
            ksize=pool_size * 2,
            strides=pool_size * 2,
            padding='VALID')

    def _upscale_2x(self, images):
        image_shape = tf.shape(images)
        return tf.image.resize(
            images,
            size=[2 * image_shape[1], 2 * image_shape[2]],
            method='nearest')

    @tf.function
    def __call__(self, sample, alpha):
        images = sample["image"]
        alpha = tf.cast(alpha, dtype=tf.float32)

        images_a = self._downscale_a(input=images)
        images_b = self._upscale_2x(self._downscale_b(input=images))

        images = alpha * images_a + (1 - alpha) * images_b
        return {
            'images': images,
        }
