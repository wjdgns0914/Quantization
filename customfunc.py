from datetime import datetime
from scipy.optimize import minimize
import numpy as np
import tensorflow as tf
def beautifultime():
    timestr = [str(x) for x in list(datetime.now().timetuple())[:6]]
    timestr[1] = timestr[1].rjust(2, '0')
    day = '-'.join(timestr[:3])
    time = '-'.join(timestr[3:])
    #time = '[' + ':'.join(timestr[3:]) + ']'

    return day,time

def my_func(cost_func,x=None):
    if x is None:
        x=np.random.standard_normal([2,])
    result=minimize(cost_func, x, method='nelder-mead', options={'xtol': 1e-8, 'disp': True})
    result=np.float32(result.x)
    return result