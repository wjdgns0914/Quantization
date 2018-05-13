import tensorflow as tf
import numpy as np
a=[[1,2,3],[4,5,6],[7,8,9]]
print(dir(a))
for i,j,k in zip([1,2,3],[4,5,6],[7,8,9]):
    print(i,j,k)

print("333") if a!=None else print("22")
a=tf.Variable(tf.zeros(shape=[3,4,5]))
print(dir(a.get_shape()))
print(a.get_shape().num_elements())
b=tf.Variable([[3,4,5,2,1,2,3.],[77,0,0,0,0,0,0]])
c=tf.reduce_sum(tf.cast(b>2,dtype=tf.float32))
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(c))
    print(sess.run(b/100))
print("######################")
for i,j,k in zip(*[[1,2,3],[4,5,6],[[],[],[]]]):
    print(i,j,k)


#dvalue 마이너스 상황보려고 어쩌다 마이너스까지 나오냐
drift_Meanmean, drift_Meanstd = 0.09, 0.001
drift_Stdmean, drift_Stdstd = 0, 0.03
filter_shape=[10,10]
drift_Meanvalue = tf.Variable(tf.random_normal(shape=filter_shape,
                                               mean=drift_Meanmean, stddev=drift_Meanstd, dtype=tf.float32,
                                               name="Mean_Value_drift"), trainable=False, name='Im5')
drift_Stdvalue = tf.cast(tf.Variable(((drift_Meanvalue - drift_Meanmean) / drift_Meanstd) * drift_Stdstd +
                                     drift_Stdmean, trainable=False, name='Im6'), dtype=tf.float32)
new_value = tf.reshape(tf.distributions.Normal(loc=drift_Meanvalue, scale=drift_Stdvalue).sample(1),shape=filter_shape)
assert a==a, "a is not 4"
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    temp=[]
    temp1=[]
    temp2=[]
    for i in range(5):
        temp.append(sess.run(new_value))
        temp1.append(sess.run(drift_Meanvalue))
        temp2.append(sess.run(drift_Stdvalue))
    temp=np.array(temp)
    temp1 = np.array(temp1)
    temp2 = np.array(temp2)
    print(temp1)
    print(temp.min(),temp.max())
    print(temp1.min(), temp1.max())
    print(temp2.min(), temp2.max())
"""
일단 temp1,temp2는 for 1,2,3,4,5에 대해 불변하는 결과를 출력,맞음
결과:
-0.0183226 0.188605
0.087562 0.0918404
-0.0731407 0.055213
stdvalue가 0보다 작으면 어떻게 하니 너네?
"""

#dvalue가 마이너스 나오고
#1->-1될때도 바뀌고
a=[]
print(a is None)
if a!=[]:
    print("Error")

a=tf.convert_to_tensor([])
print(a.get_shape().as_list()[0])