import tensorflow as tf

a=tf.constant([1,2,3,4])
b=tf.constant([3,4,5,6])
c=tf.multiply(a,b)

with tf.Session() as sess:
    a_,b_,c_=sess.run([a,b,c])
    print(a_)
    print("--------------------")
    print(b_)
    print("--------------------")
    print(c_)