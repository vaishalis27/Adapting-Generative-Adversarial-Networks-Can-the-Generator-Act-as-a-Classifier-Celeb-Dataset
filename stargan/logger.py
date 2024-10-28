import tensorflow as tf
import os

class Logger(object):
    def __init__(self, log_dir):
        self.writer = tf.summary.create_file_writer(log_dir)

    def scalar_summary(self, tag, value, step):
        with self.writer.as_default():
            tf.summary.scalar(tag, value, step=step)
            self.writer.flush()

    def image_summary(self, tag, images, step):
        with self.writer.as_default():
            for i, img in enumerate(images):
                tf.summary.image(f"{tag}/{i}", img, step=step)
            self.writer.flush()

    def histo_summary(self, tag, values, step):
        with self.writer.as_default():
            tf.summary.histogram(tag, values, step=step)
            self.writer.flush()
