import tensorflow as tf
import numpy as np

import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
from tensorflow.python.client import device_lib

device_lib.list_local_devices()

with tf.device('/gpu:1'):
  a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
  b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
  c = tf.matmul(a, b)

sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

print sess.run(c)