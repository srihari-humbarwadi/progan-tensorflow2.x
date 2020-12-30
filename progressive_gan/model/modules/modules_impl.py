import tensorflow as tf

from progressive_gan.model.layers import (EqualizedConv2d, EqualizedDense,
                                          MiniBatchStandardDeviation,
                                          PixelwiseNorm)


class GeneratorBaseBlock(tf.keras.layers.Layer):

    def __init__(self, filters, use_equalized_layers=True, **kwargs):
        super(GeneratorBaseBlock, self).__init__(**kwargs)

        self.filters = filters
        self.use_equalized_layers = use_equalized_layers

        dense_layer = \
            EqualizedDense if use_equalized_layers else tf.keras.layers.Dense
        conv_layer = \
            EqualizedConv2d if use_equalized_layers else tf.keras.layers.Conv2D

        self.pixel_norm = PixelwiseNorm()
        self.leaky_relu = tf.keras.layers.LeakyReLU(alpha=0.2)
        self.dense = dense_layer(
            units=filters * 4 * 4,
            name='{}-latent-projection'.format(self.name))
        self.conv = conv_layer(
            filters=filters,
            kernel_size=3,
            padding='same',
            name='{}-conv-3x3'.format(self.name))

    def call(self, x):
        y = tf.expand_dims(tf.expand_dims(x, axis=1), axis=1)
        y = self.pixel_norm(y)
        y = self.leaky_relu(self.dense(y))
        y = tf.reshape(y, [-1, 4, 4, self.filters])
        y = self.leaky_relu(self.conv(y))
        y = self.pixel_norm(y)
        return y

    def get_config(self):
        config = {
            'filters': self.filters,
            'use_equalized_layers': self.use_equalized_layers
        }
        base_config = super(GeneratorBaseBlock, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class GeneratorUpsampleBlock(tf.keras.layers.Layer):

    def __init__(self, filters, use_equalized_layers=True, **kwargs):
        super(GeneratorUpsampleBlock, self).__init__(**kwargs)

        self.filters = filters
        self.use_equalized_layers = use_equalized_layers

        conv_layer = \
            EqualizedConv2d if use_equalized_layers else tf.keras.layers.Conv2D

        self.pixel_norm = PixelwiseNorm(name='{}-pixel-norm'.format(self.name))
        self.leaky_relu = tf.keras.layers.LeakyReLU(
            alpha=0.2, name='{}-leaky-relu'.format(self.name))
        self.conv_1 = conv_layer(
            filters=filters, kernel_size=3,
            padding='same',
            name='{}-conv-3x3-1'.format(self.name))
        self.conv_2 = conv_layer(
            filters=filters, kernel_size=3,
            padding='same',
            name='{}-conv-3x3-2'.format(self.name))
        self.upscale_2x = tf.keras.layers.UpSampling2D(
            size=2,
            interpolation='nearest',
            name='{}-nearest-2x-upsampling'.format(self.name))

    def call(self, x):
        y = self.upscale_2x(x)
        y = self.pixel_norm(self.leaky_relu(self.conv_1(y)))
        y = self.pixel_norm(self.leaky_relu(self.conv_2(y)))
        return y

    def get_config(self):
        config = {
            'filters': self.filters,
            'use_equalized_layers': self.use_equalized_layers
        }
        base_config = super(GeneratorUpsampleBlock, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class ToRGBBlock(tf.keras.layers.Layer):

    def __init__(self, use_equalized_layers=True, **kwargs):
        super(ToRGBBlock, self).__init__(**kwargs)

        self.use_equalized_layers = use_equalized_layers

        conv_layer = \
            EqualizedConv2d if use_equalized_layers else tf.keras.layers.Conv2D

        self.conv = conv_layer(
            filters=3,
            kernel_size=1,
            name='{}-conv-1x1'.format(self.name))

    def call(self, x):
        return self.conv(x)

    def get_config(self):
        config = {'use_equalized_layers': self.use_equalized_layers}
        base_config = super(ToRGBBlock, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class DiscriminatorDownsampleBlock(tf.keras.layers.Layer):

    def __init__(self, filters, use_equalized_layers=True, **kwargs):
        super(DiscriminatorDownsampleBlock, self).__init__(**kwargs)

        self.filters = filters
        self.use_equalized_layers = use_equalized_layers

        conv_layer = \
            EqualizedConv2d if use_equalized_layers else tf.keras.layers.Conv2D

        self.leaky_relu = tf.keras.layers.LeakyReLU(alpha=0.2)
        self.conv_1 = conv_layer(filters=filters, kernel_size=3, padding='same')
        self.conv_2 = conv_layer(filters=filters, kernel_size=3, padding='same')
        self.downsample_2x = tf.keras.layers.AveragePooling2D(pool_size=2)

    def call(self, x):
        y = self.leaky_relu(self.conv_1(x))
        y = self.leaky_relu(self.conv_1(y))
        y = self.downsample_2x(y)
        return y

    def get_config(self):
        config = {
            'filters': self.filters,
            'use_equalized_layers': self.use_equalized_layers
        }
        base_config = super(DiscriminatorDownsampleBlock, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class DiscriminatorFinalBlock(tf.keras.layers.Layer):

    def __init__(self,
                 filters,
                 use_equalized_layers=True,
                 group_size=4,
                 **kwargs):
        super(DiscriminatorFinalBlock, self).__init__(**kwargs)

        self.filters = filters
        self.group_size = group_size
        self.use_equalized_layers = use_equalized_layers

        conv_layer = \
            EqualizedConv2d if use_equalized_layers else tf.keras.layers.Conv2D

        self.leaky_relu = tf.keras.layers.LeakyReLU(alpha=0.2)
        self.mini_batch_stddev = MiniBatchStandardDeviation(
            group_size=group_size)
        self.conv_1 = conv_layer(filters=filters, kernel_size=3, padding='same')
        self.conv_2 = conv_layer(filters=filters,
                                 kernel_size=4,
                                 padding='valid')
        self.conv_3 = conv_layer(filters=1,
                                 kernel_size=1,
                                 padding='valid')

        self.flatten = tf.keras.layers.Flatten()

    def call(self, x):
        y = self.mini_batch_stddev(x)
        y = self.leaky_relu(self.conv_1(y))
        y = self.leaky_relu(self.conv_2(y))
        y = self.flatten(self.conv_3(y))
        return y

    def get_config(self):
        config = {
            'filters': self.filters,
            'group_size': self.group_size,
            'use_equalized_layers': self.use_equalized_layers
        }
        base_config = super(DiscriminatorFinalBlock, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
