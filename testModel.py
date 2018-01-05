import tensorflow as tf
import console
import myCnnNet

if __name__ == '__main__':
    img, lab, lab_name = console.read_and_decode("/data_b/bd-recommend/songfeng/video/lib/cnn/result/train20000.tfrecords",shuffle_batch=False ,batch_size=1)
    net = myCnnNet.network()
    inf = net.inference(img)
    predict = tf.nn.sigmoid(inf, name="prob")
    init = tf.group(tf.local_variables_initializer(), tf.global_variables_initializer())
#    saver = tf.train.import_meta_graph("/data_b/bd-recommend/songfeng/video/lib/cnn/model/model3/file_name.ckpt.meta")
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.4
    saver = tf.train.Saver()
    with tf.Session(config=config) as sess:
        sess.run(init)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        saver.restore(sess, "/data_b/bd-recommend/songfeng/video/lib/cnn/model/model2/file_name.ckpt")
        graph = tf.get_default_graph()
        # prob_op = graph.get_operation_by_name('prob')
        # prediction = graph.get_tensor_by_name("prob:0")
        iter = 0
        while True:
            try:
                img_i, lab_i, inf_i, predict_i, lab_name_i = sess.run([img, lab, inf, predict, lab_name])
                print("prediction:", inf_i[0][0], "\tResult:", predict_i[0][0], "\tlabel:", lab_i[0], "\tlab_name:",str(lab_name_i[0],encoding='utf-8'))
                iter+=1
            except Exception as err:
                print("max : ",iter)
                break
        print("end")
        coord.request_stop()
        coord.join(threads)
