# coding=utf-8
import os
import tensorflow as tf
from PIL import Image
import console

'''
制作图片压缩文件，用来当作训练集和测试集
'''

if __name__ == '__main__':
    writerTrain = tf.python_io.TFRecordWriter("result/testAll.tfrecords")
    # writerTest = tf.python_io.TFRecordWriter("test.tfrecords")

    trainMovieUrl = "/data_b/bd-recommend/songfeng/video/lib/cnn/conf/trainAll/test"

    fileTrainMovie = open(trainMovieUrl, "r")
    trainMovieList = fileTrainMovie.read().strip().split("\n")

    print("trainMovieList ...")
    print("...")
    for imgPath in trainMovieList:
        img = Image.open(imgPath)
        img_raw = img.tobytes()
        label_name = console.getlabel(imgPath)
        label = 0
        # print("label_name\t",label_name)
        if label_name == "游戏":
            label = 1
        else:
            label = 0
        labelName = bytes(console.getlabel(imgPath), encoding="UTF-8")

        example = tf.train.Example(features=tf.train.Features(
            feature={
                "labelName": tf.train.Feature(bytes_list=tf.train.BytesList(value=[labelName])),
                "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
                "img_raw": tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
            }
        ))
        writerTrain.write(example.SerializeToString())
    writerTrain.close()
