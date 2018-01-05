# coding=utf-8
import tensorflow as tf
import console

# 罗列tfrecord

'''
    for serialized_example in tf.python_io.tf_record_iterator(
            "E:\workspace\offline\\video\myCnnNetWork\\train.tfrecords"):
        example = tf.train.Example()
        example.ParseFromString(serialized_example)

        image = example.features.feature['img_raw'].bytes_list.value
        label = example.features.feature['label'].int64_list.value
        # 可以做一些预处理之类的
        print(label)
'''

if __name__ == '__main__':
    for serialized_example in tf.python_io.tf_record_iterator(
            "/data_b/bd-recommend/songfeng/video/lib/cnn/result/test20000.tfrecords"):
        example = tf.train.Example()
        example.ParseFromString(serialized_example)

        image = example.features.feature['img_raw'].bytes_list.value
        labelName = example.features.feature['labelName'].bytes_list.value
        # labelName = str(labelName,encoding="UTF-8")
        label = example.features.feature['label'].int64_list.value
        # 可以做一些预处理之类的
        print(str(labelName[0], encoding="UTF-8"), "\t", label)
