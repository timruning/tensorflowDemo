# coding=utf-8

import console
import myCnnNet
import tensorflow as tf
import os
import cv2
import matplotlib.pyplot as plt


def transferNpArray2(nparr2, batch_size):
    lis = []
    for i in range(len(nparr2)):
        lis.append(nparr2[i])
    return lis


def transferNpArray(nparr, batch_size):
    lis = []
    for i in range(len(nparr)):
        lis.append(nparr[i][0])
    return lis


def getAccuracy(lis1, lis2, pro=0.5):
    len1 = len(lis1)
    len2 = len(lis2)
    if len1 != len2:
        print("error label and predict!")
        return -1
    count = 0
    for i in range(len1):
        temp = 0
        if lis2[i] >= pro:
            temp = 0
        else:
            temp = 1
        if lis1[i] == temp:
            count += 1
    return 1.0 * count / len1


def train():
    batch_image, batch_label, batch_lab_name = console.read_and_decode(
        "/data_b/bd-recommend/songfeng/video/lib/cnn/result/trainAll.tfrecords",batch_size=100)
    net = myCnnNet.network()

    inf = net.inference(batch_image)
    predict = tf.nn.sigmoid(inf)
    # loss = net.sorfmax_loss(inf, batch_label)
    loss = net.sigmoid_cross_entropy(inf, batch_label)

    opti = net.optimer(loss)

    # 验证集所用 
    # init = tf.initialize_all_variables()
    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())
    # 这儿会影响后面读取的错误
    saver = tf.train.Saver()
    with tf.Session() as session:
        with tf.device("/gpu:1"):
            session.run(init_op)
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
            max_iter = 40000
            iter = 0
            while iter < max_iter:
                loss_np, _, image_np, label_np, inf_np, predict_np = session.run(
                    [loss, opti, batch_image, batch_label, inf, predict])
                if iter % 200 == 0:
                    label_list = transferNpArray2(label_np, batch_size=100)
                    predict_list = transferNpArray(predict_np, batch_size=100)
                    print(label_list)
                    print(predict_list)
                    accuracy = getAccuracy(label_list, predict_list, pro=0.5)

                    # print(iter, '\tlabel:', transferNpArray2(label_np, batch_size=50))
                    # print(iter, '\tinf:', transferNpArray(predict_np, batch_size=50))
                    print(loss_np,"\t",accuracy)
                iter += 1
            saver.save(session,
                    "/data_b/bd-recommend/songfeng/video/lib/cnn/model/file_name.ckpt")
            coord.request_stop()  # queue需要关闭，否则报错 
            coord.join(threads)


if __name__ == '__main__':
    # 主函数训练
    train()
