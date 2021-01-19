import tensorflow as tf

a = tf.constant([[1, 2, 3, 4], [3, 2, 3, 4], [1, 4, 1, 5]])

b = tf.constant([1, 0, 0])

c = tf.nn.embedding_lookup(a, b)
with tf.Session() as sess:
    a_, c_ = sess.run([a, c])
    print(a_)
    print(c_)
