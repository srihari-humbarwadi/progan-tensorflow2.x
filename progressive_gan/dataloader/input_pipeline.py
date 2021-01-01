import tensorflow as tf
from absl import logging

from progressive_gan.dataloader.tfrecord_parser import parse_example


class InputPipeline:

    def __init__(self, params):
        self.tfrecord_files = params.dataloader_params.tfrecords

    def __call__(self, input_context=None):
        options = tf.data.Options()
        options.experimental_deterministic = False
        autotune = tf.data.experimental.AUTOTUNE

        dataset = tf.data.Dataset.list_files(self.tfrecord_files)

        logging.info('Found {} tfrecords matching {}'.format(
            len(dataset), self.tfrecord_files))

        dataset = dataset.cache()
        dataset = dataset.repeat()

        dataset = dataset.interleave(
            map_func=tf.data.TFRecordDataset,
            cycle_length=32,
            num_parallel_calls=autotune)

        dataset = dataset.with_options(options)
        dataset = dataset.shuffle(1024)

        dataset = dataset.map(
            map_func=parse_example,
            num_parallel_calls=autotune)
        return dataset
