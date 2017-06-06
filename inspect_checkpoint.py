import tensorflow as tf
slim = tf.contrib.slim
import numpy as np
import os
from collections import namedtuple
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
ck_pt = 'models_resnet/resnet_v2_101.ckpt'
print_tensors_in_checkpoint_file(ck_pt, '', True)

