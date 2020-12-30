import numpy as np
import tensorflow as tf

from progressive_gan.model import (
    DiscriminatorDownsampleBlock, DiscriminatorFinalBlock, GeneratorBaseBlock,
    GeneratorUpsampleBlock, ToRGBBlock)


class Generator(tf.keras.Model):

    def __init__(self, max_resolution, use_equalized_layers, **kwargs):
        super(Generator, self).__init__(**kwargs)

        self.min_depth = 2
        self.current_depth = 2
        self.max_resolution = max_resolution
        self.max_depth = int(np.log2(max_resolution))
        self.use_equalized_layers = use_equalized_layers

        self.blocks = {
            '2':
                GeneratorBaseBlock(filters=Generator._nf(stage=1),
                                   use_equalized_layers=use_equalized_layers,
                                   name='depth-2')
        }

        for stage in range(2, self.max_depth):
            self.blocks[str(stage + 1)] = GeneratorUpsampleBlock(
                filters=Generator._nf(stage=stage),
                use_equalized_layers=use_equalized_layers,
                name='depth-{}'.format(stage + 1))

        self.upscale_2x = tf.keras.layers.UpSampling2D(size=2,
                                                       interpolation='nearest')

        self.to_rgb_blocks = {
            str(i + 1): ToRGBBlock(use_equalized_layers=use_equalized_layers,
                                   name='depth-{}'.format(i + 1))
            for i in range(1, self.max_depth)
        }

    def call(self, x):
        noise, alpha = x
        y = noise

        if self.current_depth == self.min_depth:
            y = self.blocks[str(self.current_depth)](y)
            return self.to_rgb_blocks[str(self.current_depth)](y)

        for depth in range(self.min_depth, self.current_depth):
            y = self.blocks[str(depth)](y)

        residual = \
            self.upscale_2x(self.to_rgb_blocks[str(self.current_depth - 1)](y))

        straight = self.blocks[str(self.current_depth)](y)
        straight = self.to_rgb_blocks[str(self.current_depth)](straight)
        return (1 - alpha) * residual + alpha * straight

    @staticmethod
    def _nf(stage, fmap_base=8192, fmap_max=512, fmap_decay=1.0):
        return min(int(fmap_base / (2.0**(stage * fmap_decay))), fmap_max)
