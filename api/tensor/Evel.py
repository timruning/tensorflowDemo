import tensorflow as tf

if __name__ == '__main__':
    print(tf.__version__)
    tensor = tf.constant([1, 2, 3])
    with tf.compat.v1.Session() as sess:
        print(sess.run(tensor))
