import tensorflow as tf
import numpy as np
def scalebybits(bits):
    return 2.0 ** (bits - 1)

target_level=11.
bits=np.ceil(np.log(target_level) / np.log(2.0))
x=np.array([-5,-4,-3,-2,-1,0,1,2,3,4,5.])/8
level_index = tf.to_int32(tf.round(x*2**(bits-1)+tf.ceil(target_level/2)))
rr= (tf.to_float(level_index)-tf.ceil(target_level/2))/scalebybits(bits)

target_level=14
bits=np.ceil(np.log(target_level) / np.log(2.0))
x=np.array([-13,-11,-9,-7,-5,-3,-1,1,3,5,7,9,11,13.])/16
level_index = tf.to_int32(tf.ceil((x*2**(bits)+target_level)/2))
rr =(2*tf.to_float(level_index)-1-target_level)/scalebybits(bits+1)
with tf.Session() as sess:
    print(x)
    print(sess.run(level_index))
    print(sess.run(rr))