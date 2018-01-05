# coding=utf-8

'''
采用alexNet
公用函数
def getlabel(line) 目的：
从图片路径中获取，图片的label

def read_and_decode(tfrecord_filename, batch_size=4, random_crop=False, random_clip=False, shuffle_batch=True) 目的：
读取并且解析tfrecords

'''

import tensorflow as tf


def getlabel(line):
    arr = line.split("/")
    arr = arr[len(arr) - 1].split("_")
    # print(arr[1])
    arr = arr[1].split("-")
    label = arr[0]
    # print(label)
    return label


def read_and_decode(tfrecord_filename, batch_size=4, random_crop=False, random_clip=False, shuffle_batch=True):
    filename_queue = tf.train.string_input_producer([tfrecord_filename], num_epochs=4)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        features={
            "labelName": tf.FixedLenFeature([], tf.string),
            'img_raw': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.int64)
        })

    image = tf.decode_raw(features['img_raw'], tf.uint8)
    image = tf.reshape(image, [120, 120, 3])
    label = tf.cast(features['label'], tf.int32)

    label_name = features['labelName']

    if shuffle_batch:
        images, labels, label_names = tf.train.shuffle_batch(
            [image, label, label_name],
            batch_size=batch_size,
            capacity=8000,
            num_threads=batch_size,
            min_after_dequeue=2000)
    else:
        images, labels, label_names = tf.train.batch([image, label, label_name],
                                                     batch_size=batch_size,
                                                     capacity=8000,
                                                     num_threads=batch_size)
    print(images, labels, label_names)
    return images, labels, label_names
