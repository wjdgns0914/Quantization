import tensorflow as tf
import math

from tensorflow.python.training import moving_averages
from tensorflow.contrib.framework import get_name_scope
import numpy as np
FLAGS = tf.app.flags.FLAGS
# 20180119 이 부분이 FLAGS 파싱하기 전에 적용이 되서 variation이 계속 안먹혔었다. 일단 급하니까 그냥 먹임
# if FLAGS.Variation==True:
#     Reset_Meanmean, Reset_Meanstd = 0., 0.1707
#     Reset_Stdmean, Reset_Stdstd = 0.0942, 0.01884
#     Set_Meanmean, Set_Meanstd = 0., 0.1538
#     Set_Stdmean, Set_Stdstd = 0.1311, 0.06894
# else:
#     Reset_Meanmean, Reset_Meanstd = 0., 0.0000000001
#     Reset_Stdmean, Reset_Stdstd = 0., 0.
#     Set_Meanmean, Set_Meanstd = 0., 0.0000000001
#     Set_Stdmean, Set_Stdstd = 0., 0.
# Reset_Meanmean, Reset_Meanstd = 0., 0.1707
# Reset_Stdmean, Reset_Stdstd = 0.0942, 0.01884
# Set_Meanmean, Set_Meanstd = 0., 0.1538
# Set_Stdmean, Set_Stdstd = 0.1311, 0.06894
drift_Meanmean, drift_Meanstd = 0.09, 0.001
drift_Stdmean, drift_Stdstd = 0, 0.0003


def binarize(x,target_level=7):
    assert (target_level>1) and (int(target_level)==target_level), \
        'target_level is not valid, please input the integer bigger than 1 '
    """
    Clip and binarize tensor using the straight through estimator (STE) for the gradient.
    """

    g = tf.get_default_graph()
    if target_level==2:
        with tf.name_scope("Binarized") as name:
            with g.gradient_override_map({"Sign": "Identity"}):
                x=tf.clip_by_value(x,-1,1)
                results=tf.sign(x)
                if FLAGS.bit_deterministic:
                    results=results+tf.cast(x>(tf.reduce_max(x)/2),dtype=tf.float32)*0.4
    else:

        with g.gradient_override_map({"Round":"Identity"}):
            x = tf.tanh(x)
            x = x / tf.reduce_max(tf.abs(x)) * 0.5 + 0.5
            x=tf.round(x*(target_level-1))/(target_level-1)
        results=2*x-1
    return results

@tf.RegisterGradient("fluc_grad")
def fluc_grad(op,grad):
    shape=op.inputs[1]._shape_as_list()
    return grad,tf.zeros(shape=shape)

def fluctuate(x,scale=1,Drift=False):
    filter_shape = x.get_shape().as_list()
    pre_Wbin = tf.Variable(initial_value=tf.zeros(shape=filter_shape),name='pre_Wbin',trainable=False)
    pre_Wbin_val_place=tf.placeholder(dtype=tf.float32,shape=filter_shape)
    pre_Wbin_update_op=pre_Wbin.assign(pre_Wbin_val_place)
    pre_Wfluc = tf.Variable(initial_value=tf.zeros(shape=filter_shape),name='pre_Wfluc',trainable=False)
    pre_Wfluc_val_place = tf.placeholder(dtype=tf.float32, shape=filter_shape)
    pre_Wfluc_update_op = pre_Wfluc.assign(pre_Wfluc_val_place)
    tf.add_to_collection('pre_Wbin',pre_Wbin_val_place)
    tf.add_to_collection('pre_Wbin_update_op', pre_Wbin_update_op)
    tf.add_to_collection('pre_Wfluc', pre_Wfluc_val_place)
    tf.add_to_collection('pre_Wfluc_update_op', pre_Wfluc_update_op)
    if FLAGS.Variation==True:
        Reset_Meanmean, Reset_Meanstd = 0., 0.1707
        Reset_Stdmean, Reset_Stdstd = 0.0942, 0.01884
        Set_Meanmean, Set_Meanstd = 0., 0.1538
        Set_Stdmean, Set_Stdstd = 0.1311, 0.06894
    else:
        Reset_Meanmean, Reset_Meanstd = 0., 0.0000000001
        Reset_Stdmean, Reset_Stdstd = 0., 0.
        Set_Meanmean, Set_Meanstd = 0., 0.0000000001
        Set_Stdmean, Set_Stdstd = 0., 0.
    with tf.name_scope("Fluctuated") as name:
        Reset_Meanvalue = tf.Variable(tf.random_normal(shape=filter_shape,
                                            mean=Reset_Meanmean,
                                            stddev=Reset_Meanstd, dtype=tf.float32),
                                            name="Reset_MeanValue",trainable=False)
        Reset_Stdvalue = tf.Variable(((Reset_Meanvalue - Reset_Meanmean) / Reset_Meanstd)
                                     * Reset_Stdstd + Reset_Stdmean,
                                     name='Reset_Stdvale',
                                     trainable=False)
        Set_Meanvalue = tf.Variable(tf.random_normal(shape=filter_shape,
                                        mean=Set_Meanmean,
                                        stddev=Set_Meanstd,
                                        dtype=tf.float32),
                                        name="Set_MeanValue",trainable=False)
        Set_Stdvalue = tf.Variable(((Set_Meanvalue - Set_Meanmean) / Set_Meanstd)
                                   * Set_Stdstd + Set_Stdmean,
                                   name='Set_Stdvalue',
                                   trainable=False)
        #20180226 : 여기에 variable을 주구장창 추가하면 variable 이 너무 많아지지 않나
        fluc_Reset = tf.reshape(tf.distributions.Normal(loc=Reset_Meanvalue, scale=Reset_Stdvalue).sample(1),filter_shape)
        fluc_Set = tf.reshape(tf.distributions.Normal(loc=Set_Meanvalue, scale=Set_Stdvalue).sample(1), filter_shape)
        # 드리프트에 var 넣기2##################################################################
        drift_Meanvalue = tf.Variable(tf.random_normal(shape=filter_shape,
                                                       mean=drift_Meanmean, stddev=drift_Meanstd, dtype=tf.float32,
                                                       name="Mean_Value_drift"), trainable=False, name='drift_Meanvalue')
        drift_Stdvalue = tf.cast(tf.Variable(((drift_Meanvalue - drift_Meanmean) / drift_Meanstd) * drift_Stdstd +
                                             drift_Stdmean, trainable=False, name='drift_Stdvalue'), dtype=tf.float32)

        g = tf.get_default_graph()
            # assign 1 to elements which have same state with pre-state
        keep_element = tf.cast(tf.equal(x,pre_Wbin), tf.float32)
        # assign 1 to elements which have different state with pre-state
        update_element = tf.cast(tf.not_equal(x,pre_Wbin), tf.float32)
        # 이 부분에서 쓰지도않는 랜덤 값이 많이 발생하는데 일단 두고 나중에 고치든가 하자
        Wfluc_Reset = update_element * fluc_Reset * tf.cast(x > 0, tf.float32)
        Wfluc_Set = update_element * fluc_Set * tf.cast(x <= 0, tf.float32)
        # fluctuation이 적용 된 최종 weight 값,Reset,set에 drift를 따로 적용하기 위해 pre_Wbin이 0보다 큰 부분,작은 부분, 두 부분으로 나누었다.
        step_col = tf.get_collection("Step")

        drift_value = tf.Variable(tf.ones(shape=filter_shape) * 0.09, trainable=False)
        step = tf.Variable(tf.zeros(shape=filter_shape, dtype=tf.float32), trainable=False)
        if Drift and step_col!=[]:

            new_drift = tf.cast((pre_Wbin <= 0), dtype=tf.float32)* tf.cast((x > 0), dtype=tf.float32)
            # 전사이클에서는 -1이었고 이번 사이클에 1로 업데이트 된 element 부분을 1로
            new_value = tf.reshape(tf.distributions.Normal(loc=drift_Meanvalue, scale=drift_Stdvalue).sample(1)
                                   , shape=filter_shape)
            # # 전체의 random 값을 generate.
            assign_drift = drift_value.assign(new_value * new_drift + drift_value * (1 - new_drift))
            tf.add_to_collection("Drift_value",drift_value)
            with tf.control_dependencies([assign_drift]):
                #step = tf.Variable(tf.zeros(shape=filter_shape, dtype=tf.float32), trainable=False)
                step = tf.assign(step, step * keep_element + keep_element,name="Drift_step")
                tf.add_to_collection("Drift_step",step)
                #step은 각 웨이트가 드리프트를 지금 몇스텝째 하고있는지를 나타낸다.
                drift_factor = (step+1.)/(tf.cast(tf.equal(step,0.),dtype=tf.float32)+step)
                drift_scale =  (tf.log(drift_factor) / tf.log(10.))*drift_value
                #drift_value는 dvalue이다,즉 power law에서 지수부분
                tf.add_to_collection("testt", drift_scale)
        else:
            tf.add_to_collection("Drift_value", tf.convert_to_tensor([]))
            tf.add_to_collection("Drift_step", tf.convert_to_tensor([]))
            drift_scale = tf.constant(0.)
        with g.gradient_override_map({"Mul": "fluc_grad", "Cast": "Identity",
                                      "Equal": "fluc_grad", "Greater": "fluc_grad",
                                      "LessEqual": "fluc_grad", "NotEqual": "fluc_grad",
                                      "Add": "fluc_grad"}):
            with tf.control_dependencies([drift_scale]):
                Wfluc = tf.multiply(x, update_element) +tf.cast(tf.greater(pre_Wbin,0), tf.float32) * keep_element * (pre_Wfluc + drift_scale)+ \
                        tf.cast(tf.less_equal(pre_Wbin,0), tf.float32) * keep_element * pre_Wfluc * 1. \
                        + Wfluc_Reset + Wfluc_Set
        return Wfluc
"""
주의:binarize(x)가 activation까지 바이너리화 하는 중이다.
아래의 세개는 각각 Binary Conv layer,Binary Conv layer for weight, Vanilla Conv layer
"""
def BinarizedSpatialConvolution(nOutputPlane, kW, kH, dW=1, dH=1,
        padding='VALID', bias=True, reuse=None, name='BinarizedSpatialConvolution',bin=True,fluc=True,Drift=False):
    def b_conv2d(x, is_training=True):
        nInputPlane = x.get_shape().as_list()[3]
        with tf.variable_scope(values=[x], name_or_scope=None, default_name=name,reuse=reuse):
            w = tf.get_variable('weight', [kH, kW, nInputPlane, nOutputPlane],
                            initializer=tf.contrib.layers.xavier_initializer_conv2d())
            bin_x = binarize(x) if bin else x
            tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, bin_x)
            bin_w = binarize(w)
            fluc_w = fluctuate(bin_w,Drift=Drift) if fluc else bin_w
            tf.add_to_collection('Original_Weight',w)
            tf.add_to_collection('Binarized_Weight', bin_w)
            tf.add_to_collection('Fluctuated_Weight', fluc_w)
            '''
            Note that we use binarized version of the input and the weights. Since the binarized function uses STE
            we calculate the gradients using bin_x and bin_w but we update w (the full precition version).
            '''
            out = tf.nn.conv2d(bin_x, fluc_w, strides=[1, dH, dW, 1], padding=padding)
            if bias:
                b = tf.get_variable('bias', [nOutputPlane],initializer=tf.zeros_initializer)
                out = tf.nn.bias_add(out, b)
            return out
    return b_conv2d

def BinarizedWeightOnlySpatialConvolution(nOutputPlane, kW, kH, dW=1, dH=1,
        padding='VALID', bias=True, reuse=None, name='BinarizedWeightOnlySpatialConvolution',fluc=True,Drift=False):
    '''
    This function is used only at the first layer of the model as we dont want to binarized the RGB images
    '''
    def bc_conv2d(x, is_training=True):
        nInputPlane = x.get_shape().as_list()[3]
        with tf.variable_scope(values=[x], name_or_scope=None, default_name=name, reuse=reuse):
            w = tf.get_variable('weight', [kH, kW, nInputPlane, nOutputPlane],
                            initializer=tf.contrib.layers.xavier_initializer_conv2d())
            bin_w = binarize(w)
            fluc_w = fluctuate(bin_w,Drift=Drift) if fluc else bin_w
            tf.add_to_collection('Original_Weight', w)
            tf.add_to_collection('Binarized_Weight', bin_w)
            tf.add_to_collection('Fluctuated_Weight', fluc_w)

            out = tf.nn.conv2d(x, fluc_w, strides=[1, dH, dW, 1], padding=padding)
            if bias:
                b = tf.get_variable('bias', [nOutputPlane],initializer=tf.zeros_initializer)
                out = tf.nn.bias_add(out, b)
            return out
    return bc_conv2d

def SpatialConvolution(nOutputPlane, kW, kH, dW=1, dH=1,
        padding='VALID', bias=True, reuse=None, name='SpatialConvolution'):
    def conv2d(x, is_training=True):
        nInputPlane = x.get_shape().as_list()[3]
        with tf.variable_scope(values=[x], name_or_scope=None, default_name=name, reuse=reuse):
            w = tf.get_variable('weight', [kH, kW, nInputPlane, nOutputPlane],
                            initializer=tf.contrib.layers.xavier_initializer_conv2d())
            tf.add_to_collection('Original_Weight', w)
            out = tf.nn.conv2d(x, w, strides=[1, dH, dW, 1], padding=padding)
            if bias:
                b = tf.get_variable('bias', [nOutputPlane],initializer=tf.zeros_initializer)
                out = tf.nn.bias_add(out, b)
            return out
    return conv2d
"""
아래의 세개는 각각 Vanilla Affine layer,Binary Affine layer,Binary Affine layer for weight, 
"""
#Fully connected layer
def Affine(nOutputPlane, bias=True, name=None, reuse=None):
    def affineLayer(x, is_training=True):
        with tf.variable_scope(values=[x], name_or_scope=name, default_name='Affine', reuse=reuse):
            temp=x.get_shape().as_list()
            reshaped = tf.reshape(x, [-1,np.array(temp[1:]).prod()])
            nInputPlane = reshaped.get_shape().as_list()[1]
            w = tf.get_variable('weight', [nInputPlane, nOutputPlane], initializer=tf.contrib.layers.xavier_initializer())
            tf.add_to_collection('Original_Weight', w)
            output = tf.matmul(reshaped, w)
            if bias:
                b = tf.get_variable('bias', [nOutputPlane],initializer=tf.zeros_initializer)
                output = tf.nn.bias_add(output, b)
        return output
    return affineLayer

def BinarizedAffine(nOutputPlane, bias=True, name=None, reuse=None,bin=True,fluc=True,Drift=False):
    def b_affineLayer(x, is_training=True):
        with tf.variable_scope(values=[x], name_or_scope=name, default_name='Affine', reuse=reuse):
            '''
            Note that we use binarized version of the input (bin_x) and the weights (bin_w). Since the binarized function uses STE
            we calculate the gradients using bin_x and bin_w but we update w (the full precition version).
            '''
            bin_x = binarize(x) if bin else x
            tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, bin_x)
            reshaped = tf.reshape(bin_x, [x.get_shape().as_list()[0], -1])
            nInputPlane = reshaped.get_shape().as_list()[1]
            w = tf.get_variable('weight', [nInputPlane, nOutputPlane], initializer=tf.contrib.layers.xavier_initializer())
            bin_w = binarize(w)
            fluc_w = fluctuate(bin_w,Drift=Drift) if fluc else bin_w
            tf.add_to_collection('Original_Weight', w)
            tf.add_to_collection('Binarized_Weight', bin_w)
            tf.add_to_collection('Fluctuated_Weight', fluc_w)

            output = tf.matmul(reshaped, fluc_w)
            if bias:
                b = tf.get_variable('bias', [nOutputPlane],initializer=tf.zeros_initializer)
                output = tf.nn.bias_add(output, b)
        return output
    return b_affineLayer

def BinarizedWeightOnlyAffine(nOutputPlane, bias=True, name=None, reuse=None, Drift=False):
    def bwo_affineLayer(x, is_training=True):
        with tf.variable_scope(values=[x], name_or_scope=name, default_name='Affine', reuse=reuse):

            '''
            Note that we use binarized version of the input (bin_x) and the weights (bin_w). Since the binarized function uses STE
            we calculate the gradients using bin_x and bin_w but we update w (the full precition version).
            '''
            reshaped = tf.reshape(x, [x.get_shape().as_list()[0], -1])
            nInputPlane = reshaped.get_shape().as_list()[1]
            w = tf.get_variable('weight', [nInputPlane, nOutputPlane], initializer=tf.contrib.layers.xavier_initializer())
            bin_w = binarize(w)
            fluc_w = fluctuate(bin_w,Drift=Drift)
            tf.add_to_collection('Original_Weight', w)
            tf.add_to_collection('Binarized_Weight', bin_w)
            tf.add_to_collection('Fluctuated_Weight', fluc_w)
            output = tf.matmul(reshaped, fluc_w)
            if bias:
                b = tf.get_variable('bias', [nOutputPlane],initializer=tf.zeros_initializer)
                output = tf.nn.bias_add(output, b)
        return output
    return bwo_affineLayer

#bias 더 해주는 레이어
def Linear(nInputPlane, nOutputPlane):
    return Affine(nInputPlane, nOutputPlane, add_bias=False)

#여기서는 Sequential을 구성하기 위해서 일반레이어함수를 Sequential용으로 바꿔준다.
#*args,**kwargs 이 둘은 지금은 확정되지않았지만, 추후에 들어올 수 있는 인풋을 담당한다.
def wrapNN(f,*args,**kwargs):
    def layer(x, is_training=True):
        return f(x,*args,**kwargs)
    return layer

#이것도 사실 위의 함수와 기능은 똑같으나, dropout이기 때문에 training일 때와 test일 때를 다르게 해줘야해서, 따로 함수를 정의
def Dropout(p, name='Dropout'):
    def dropout_layer(x, is_training=True):
        with tf.variable_scope(values=[x], name_or_scope=None, default_name=name):
            # def drop(): return tf.nn.dropout(x,p)
            # def no_drop(): return x
            # return tf.cond(is_training, drop, no_drop)
            if is_training:
                return tf.nn.dropout(x,p)
            else:
                return x
    return dropout_layer

"""
아래의 두 함수는 activation이다, activation layer를 만드는 셈
다시 한번 말하지만 아래처럼 함수로 반환하는 이유는 인풋의 자유를 남겨놓기 위해서다.
HardTanh은 Tanh랑 다르다.
"""
def ReLU(name='ReLU'):
    def layer(x, is_training=True):
        with tf.variable_scope(values=[x], name_or_scope=None, default_name=name):
            return tf.nn.relu(x)
    return layer

def HardTanh(name='HardTanh'):
    def layer(x, is_training=True):
        with tf.variable_scope(values=[x], name_or_scope=None, default_name=name):
            return tf.clip_by_value(x,-1,1)
    return layer


def View(shape, name='View'):
    return wrapNN(tf.reshape,shape=shape)
"""
    kH = kH or kW    의 뜻은 전자가 없으면 후자로 쓴다는 뜻
->최소 kW는 인풋으로 넣어줘야 멀쩡한 함수 실행이 가능하다,
kW는 window size, kH없으면 윈도우는 kH=kW인 정사각형이 된다 
만약 strides도 안넣어준다면 stride는 윈도우의 가로세로와 같다.
Tip:파이썬에서는 and와 or이 본래 논리연산자의 기능을 하면서 인풋으로 들어간 값들을 이용 할 수 있도록하여
효율을 극대화 하였다.
"""
def SpatialMaxPooling(kW, kH=None, dW=None, dH=None, padding='VALID',
            name='SpatialMaxPooling'):
    kH = kH or kW
    dW = dW or kW
    dH = dH or kH
    def max_pool(x,is_training=True):
        with tf.variable_scope(values=[x], name_or_scope=None, default_name=name):
              return tf.nn.max_pool(x, ksize=[1, kW, kH, 1], strides=[1, dW, dH, 1], padding=padding)
    return max_pool

def SpatialAveragePooling(kW, kH=None, dW=None, dH=None, padding='VALID',
        name='SpatialAveragePooling'):
    kH = kH or kW
    dW = dW or kW
    dH = dH or kH
    def avg_pool(x,is_training=True):
        with tf.variable_scope(values=[x], name_or_scope=None, default_name=name):
              return tf.nn.avg_pool(x, ksize=[1, kW, kH, 1], strides=[1, dW, dH, 1], padding=padding)
    return avg_pool

def BatchNormalization(name='BatchNormalization',*kargs, **kwargs):
    with tf.variable_scope(name_or_scope=name):
        output=wrapNN(tf.contrib.layers.batch_norm, *kargs, **kwargs)
    return output

def Sequential(moduleList):
    def model(x, is_training=True):
    # Create model
        output = x
        #with tf.variable_op_scope([x], None, name):
        for i,m in enumerate(moduleList):
            output = m(output, is_training=is_training)
            tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, output)
        return output
    return model
"""
아래 두 함수는 현재 사용하지 않음, 다만 나중에 네트워크 바꿀 때 사용할 수도 있어서 냅두었음
"""
#병렬연결모드
def Concat(moduleList, dim=3):
    def model(x, is_training=True):
    # Create model
        outputs = []
        for i,m in enumerate(moduleList):
            name = 'layer_'+str(i)
            with tf.variable_scope(values=[x], name_or_scope=name, default_name='Layer'):
                outputs[i] = m(x, is_training=is_training)
            output = tf.concat(dim, outputs)
        return output
    return model

#말그대로 residual
def Residual(moduleList, name='Residual'):
    m = Sequential(moduleList)
    def model(x, is_training=True):
    # Create model
        with tf.variable_scope(values=[x], name_or_scope=None, default_name=name):
            output = tf.add(m(x, is_training=is_training), x)
            return output
    return model
