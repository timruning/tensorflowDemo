import tensorflow as tf

print(tf.__version__)
if __name__ == '__main__':
    with tf.compat.v1.Session() as sess:
        with tf.compat.v1.gfile.FastGFile(
                "../model/2.4/model.pb", 'rb') as f:
            g = tf.compat.v1.GraphDef()
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name="")

            train_writer = tf.compat.v1.summary.FileWriter("../logs/2.4/sub", sess.graph)

            print("hello")
