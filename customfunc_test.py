from customfunc import *
"""
target_level=input('input the target_level')
target_level=float(target_level)
assert (target_level > 1) and (int(target_level) == target_level), \
    'target_level is not valid, please input the integer bigger than 1 '
print('Success:' ,int(target_level))
"""
from scipy.optimize import minimize
import numpy as np
import tensorflow as tf
"""
def rosen(x):
    #The Rosenbrock function
    return 100.0*(x[1:]-x[:-1]**2.0)**2.0 + (1-x[:-1])**2.0
x0 = np.array([1.3, 0.7])
res = minimize(rosen, x0, method='nelder-mead',options={'xtol': 1e-8, 'disp': True})

print(type(rosen(x0)))

def cost_func(x):
    step_size=x[-2]
    xmin=x[-1]
    xmax = step_size + xmin
    x_bin = tf.round((tf.clip_by_value(x, xmin, xmax) - xmin) / step_size)
    results = step_size * x_bin + xmin
    return results-x
def my_func(x=None):
    if x is None:
        x=np.random.standard_normal([2,])
    result=minimize(cost_func, x, method='nelder-mead', options={'xtol': 1e-8, 'disp': True})
    result=np.float32(result.x)
    return result

print(np.sinh(x0))
def my_func2(x):
    return np.sinh(x)
x1=tf.Variable(x0)
y=tf.py_func(my_func,[x1],tf.float32)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(3):
        x1array = sess.run(x1)
        result=my_func()
        new_x=np.random.standard_normal(size=[2,])
"""
a=np.random.normal(loc=0.0,scale=0.5,size=[3,3])
b=((a - 0) / 0.5)* 0.22 + 0.15
fluc_Reset = tf.reshape(tf.distributions.Normal(loc=a, scale=b).sample(1),[3,3])
with tf.Session() as sess:
  for i in range(10):
    print("a===================\n",a)
    c=sess.run(fluc_Reset)
    print("c===================\n",c)
