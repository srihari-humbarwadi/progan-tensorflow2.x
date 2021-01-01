import tensorflow as tf

from progressive_gan.model import (BaseNetwork, DiscriminatorDownsampleBlock,
                                   DiscriminatorFinalBlock, FromRGBBlock,
                                   GeneratorBaseBlock, GeneratorUpsampleBlock,
                                   ToRGBBlock)


class Generator(BaseNetwork):

    def __init__(self, max_resolution, use_equalized_layers, **kwargs):
        super(Generator, self).__init__(
            max_resolution=max_resolution,
            use_equalized_layers=use_equalized_layers,
            name='Generator', **kwargs)

        self.blocks = {
            '2':
                GeneratorBaseBlock(filters=Generator._nf(stage=1),
                                   use_equalized_layers=use_equalized_layers,
                                   name='depth-2-conv-block')
        }

        for stage in range(2, self.max_depth):
            self.blocks[str(stage + 1)] = GeneratorUpsampleBlock(
                filters=Generator._nf(stage=stage),
                use_equalized_layers=use_equalized_layers,
                name='depth-{}-conv-block'.format(stage + 1))

        self.upscale_2x = tf.keras.layers.UpSampling2D(
            size=2, interpolation='nearest', name='nearest-2x-upsampling')

        self.to_rgb_blocks = {
            str(i + 1): ToRGBBlock(use_equalized_layers=use_equalized_layers,
                                   name='depth-{}-to-rgb'.format(i + 1))
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

        residual = self.to_rgb_blocks[str(self.current_depth - 1)](y)
        residual = self.upscale_2x(residual)

        straight = self.blocks[str(self.current_depth)](y)
        straight = self.to_rgb_blocks[str(self.current_depth)](straight)
        return (1 - alpha) * residual + alpha * straight


class Discriminator(BaseNetwork):
    def __init__(self, max_resolution, use_equalized_layers, **kwargs):

        super(Discriminator, self).__init__(
            max_resolution=max_resolution,
            use_equalized_layers=use_equalized_layers,
            name='Discriminator', **kwargs)

        self.blocks = {
            '2':
                DiscriminatorFinalBlock(
                    filters=Discriminator._nf(stage=1),
                    use_equalized_layers=use_equalized_layers,
                    name='depth-2-conv-block')
        }

        for stage in range(2, self.max_depth):
            self.blocks[str(stage + 1)] = DiscriminatorDownsampleBlock(
                filters=[
                    Discriminator._nf(stage=stage),
                    Discriminator._nf(stage=stage - 1)
                ],
                use_equalized_layers=use_equalized_layers,
                name='depth-{}-conv-block'.format(stage + 1))

        self.downscale_2x = tf.keras.layers.AvgPool2D(
            pool_size=2, name='avgpool2d-2x-downsampling')

        self.from_rgb_blocks = {
            str(i + 1): FromRGBBlock(filters=Discriminator._nf(stage=i),
                                     use_equalized_layers=use_equalized_layers,
                                     name='depth-{}-from-rgb'.format(i + 1))
            for i in range(1, self.max_depth)
        }

    def call(self, x):
        images, alpha = x
        y = images

        if self.current_depth == self.min_depth:
            y = self.from_rgb_blocks[str(self.current_depth)](y)
            return self.blocks[str(self.current_depth)](y)

        residual = self.downscale_2x(y)
        residual = self.from_rgb_blocks[str(self.current_depth - 1)](residual)

        straight = self.from_rgb_blocks[str(self.current_depth)](y)
        straight = self.blocks[str(self.current_depth)](straight)

        y = (1 - alpha) * residual + alpha * straight

        for depth in range(self.current_depth - 1, self.min_depth - 1, -1):
            y = self.blocks[str(depth)](y)

        return y
