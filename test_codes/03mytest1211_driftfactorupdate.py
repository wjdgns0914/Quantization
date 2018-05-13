import tensorflow as tf
import numpy as np
n=np.array([0,0,0,0,0])
t=tf.constant([0,0,0,0,0])
t2=tf.Variable([0,0,0,0,0])
t3=tf.Variable([0,0,0,0,0])

random=tf.cast(tf.random_normal(shape=[5,],mean=0.,stddev=1.)>0,dtype=tf.int32)
t4=tf.constant([0,0,0,0,0])+random
def forfor(n,t,t2,t3):
    n=n+random
    t=t+random
    t2=tf.assign(t2,t2+random)
    with tf.control_dependencies([tf.assign(t3,t3+random)]):
        t4=tf.identity(t3)
    return n,t,t2,t4
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    n, t, t2,t4 = forfor(n, t, t2,t3)
    for i in range(10):
        x,y,z,h=sess.run([n,t,t2,t4])
        print(sess.run(t4))
        print("Numpy array: n=",x)
        print("Tensorflow : t=",y)
        print("Tensorflow :t2=",z)
        print("Tensorflow :t4=",h)











