from PIL import Image
import os
import numpy as np
import sys
import random
import cPickle
from tensorflow.python.lib.io.tf_record import TFRecordCompressionType
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))
  
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
sys.path.append('../coco/PythonAPI')
from pycocotools.coco import COCO
import tensorflow as tf
if len(sys.argv) < 2:
    print 'Usage: python gen_tfrecord.py [train/val]'
stage = sys.argv[1]
srcFile = 'cache/{}_annos.pkl'.format(stage)
annos = cPickle.load(open(srcFile,'rb'))
dstFile = 'cache/{}.tfrecords'.format(stage)
image_path = '/raid/ljyang/data/MS_COCO/%s2017/{:012d}.jpg' % (stage)
anno_path ='/raid/ljyang/data/MS_COCO/annotations/instances_{}2017.json'.format(stage)
data = COCO(anno_path)
writer = tf.python_io.TFRecordWriter(dstFile, options=tf.python_io.TFRecordOptions(
              compression_type=TFRecordCompressionType.GZIP))
mean_value = np.array((104, 117, 123))
new_size = (600, 600)
for i in range(len(annos)):
    anno = annos[i]
    if not i % 1000:
        print 'saving data: {}/{}'.format(i, len(annos))
        sys.stdout.flush()
    image_path = image_path.format(anno['image_id'])
    image = Image.open(image_path)
    label_data = data.annToMask(anno).astype(np.uint8)
    #label = Image.fromarray(label_data)
    bbox = anno['bbox']
    image_data = np.array(image, dtype=np.uint8)
    if len(image_data.shape) < 3:
        image_data = np.repeat(image_data[..., np.newaxis], 3, axis=2)
    #image_data = image_data[:,:,2::-1] - mean_value
    image_data = image_data[:,:,2::-1]
    feature = {'label':_bytes_feature(tf.compat.as_bytes(label_data.tostring())),
            'image':_bytes_feature(tf.compat.as_bytes(image_data.tostring())),
            'bbox': _int64_feature(list(bbox))}
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    writer.write(example.SerializeToString())
writer.close()
sys.stdout.flush()
