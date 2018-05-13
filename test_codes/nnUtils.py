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
            bin_x = quantize(x) if bin else x
            tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, bin_x)
            bin_w = quantize(w)
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
            bin_w = quantize(w)
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