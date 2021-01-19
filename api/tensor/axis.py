import tensorflow as tf

a = tf.constant([[1., 2., 3.], [1., 2., 3.]],name="a",dtype=tf.float32)
b = tf.constant([[5., 6.], [8., 9.]],name="b",dtype=tf.float32)
c = tf.concat([a, b], axis=1)

with tf.Session() as sess:
    a_, b_,c_ = sess.run([a, b,c])
    print(a_)
    print("______________")
    print(b_)
    print("______________")
    print(c_)
