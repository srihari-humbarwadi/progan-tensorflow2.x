import tensorflow as tf
from tensorflow.python.keras.layers.ops.core import dense
from tensorflow.python.ops.nn_ops import conv2d


class EqualizedConv2d(tf.keras.layers.Conv2D):

    def __init__(self, **kwargs):
        super(EqualizedConv2d,
              self).__init__(activation=None,
                             kernel_initializer=tf.initializers.RandomNormal(
                                 0, 1),
                             bias_initializer='zeros',
                             **kwargs)

    def build(self, input_shape):
        in_channels = self._get_input_channel(input_shape)
        self.scale = tf.sqrt(
            2 / (self.kernel_size[0] * self.kernel_size[1] * in_channels))
        super(EqualizedConv2d, self).build(input_shape)

    def call(self, x):
        x = conv2d(input=x,
                   filters=self.kernel * self.scale,
                   strides=self.strides,
                   padding=self.padding.upper(),
                   dilations=self.dilation_rate)

        if self.use_bias:
            x = tf.nn.bias_add(x, self.bias)
        return x

    def get_config(self):
        super(EqualizedConv2d, self).get_config()


class EqualizedDense(tf.keras.layers.Dense):

    def __init__(self, **kwargs):
        super(EqualizedDense,
              self).__init__(activation=None,
                             kernel_initializer=tf.initializers.RandomNormal(
                                 0, 1),
                             bias_initializer='zeros',
                             **kwargs)

    def build(self, input_shape):
        in_features = tf.TensorShape(input_shape).as_list()[-1]
        self.scale = tf.sqrt(2 / in_features)
        super(EqualizedDense, self).build(input_shape)

    def call(self, x):
        x = dense(inputs=x,
                  kernel=self.kernel * self.scale,
                  bias=self.bias,
                  activation=self.activation,
                  dtype=self._compute_dtype_object)

        if self.use_bias:
            x = tf.nn.bias_add(x, self.bias)
        return x

    def get_config(self):
        super(EqualizedDense, self).get_config()


class PixelwiseNorm(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super(PixelwiseNorm, self).__init__(**kwargs)

    def call(self, x):
        return x * tf.math.rsqrt(
            tf.reduce_mean(tf.square(x), axis=-1, keepdims=True) +
            tf.keras.backend.epsilon())

    def get_config(self):
        super(PixelwiseNorm, self).get_config()


class MiniBatchStandardDeviation(tf.keras.layers.Layer):

    def __init__(self, group_size=4, **kwargs):
        super(MiniBatchStandardDeviation, self).__init__(**kwargs)
        self.group_size = group_size

    def call(self, x):
        input_shape = tf.shape(x)
        N = input_shape[0]
        H = input_shape[1]
        W = input_shape[2]
        C = input_shape[3]
        group_size = tf.minimum(N, self.group_size)

        y = tf.reshape(x, shape=[group_size, N // group_size, H, W, C])

        y = tf.cast(y, dtype=tf.float32)
        y = y - tf.reduce_mean(y, axis=0)
        y = tf.sqrt(
            tf.reduce_mean(tf.square(y), axis=0) + tf.keras.backend.epsilon())
        y = tf.reduce_mean(y, axis=[1, 2, 3], keepdims=True)
        y = tf.cast(y, dtype=x.dtype)
        y = tf.tile(y, multiples=[group_size, H, W, 1])
        return tf.concat([x, y], axis=-1)

    def get_config(self):
        config = {'group_size': self.group_size}
        base_config = super(MiniBatchStandardDeviation, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
