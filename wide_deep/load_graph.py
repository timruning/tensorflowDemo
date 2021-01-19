import tensorflow as tf

print(tf.__version__)
if __name__ == '__main__':
    with tf.Session() as sess:
        with tf.gfile.FastGFile(
                "/Users/songfeng/workspace/github/tensorflowDemo/model/model3.pb", 'rb') as f:
            g = tf.GraphDef()
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name="")

            train_writer = tf.compat.v1.summary.FileWriter("logs/wide_deep3", sess.graph)

            add = sess.graph.get_tensor_by_name(name='add:0')
            Age_ids = sess.graph.get_tensor_by_name(name='Age_1:0')
            Sex_ids = sess.graph.get_tensor_by_name(name='Sex_1:0')
            Chol_ids = sess.graph.get_tensor_by_name(name='Chol_1:0')
            Fbs_ids = sess.graph.get_tensor_by_name(name='Fbs_1:0')
            Oldpeak_ids = sess.graph.get_tensor_by_name(name='Oldpeak_1:0')
            Slope_ids = sess.graph.get_tensor_by_name(name='Slope_1:0')
            Ca_ids = sess.graph.get_tensor_by_name(name='Ca_1:0')
            Thal_ids = sess.graph.get_tensor_by_name(name='Thal_1:0')

            add_1 = sess.run(add, feed_dict={Age_ids: [50],
                                             Sex_ids: [2],
                                             Chol_ids: [10],
                                             Fbs_ids: [3],
                                             Oldpeak_ids: [20],
                                             Slope_ids: [3],
                                             Ca_ids: [100],
                                             Thal_ids: ['fixed']
                                             })
            print(add_1)
