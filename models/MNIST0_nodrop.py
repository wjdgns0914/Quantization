from nnUtils import *
FLAGS = tf.app.flags.FLAGS
print("Model")
print(FLAGS.Drift1,FLAGS.Drift2,FLAGS.Variation)
Dri1=FLAGS.Drift1
Dri2=FLAGS.Drift2
model = Sequential([
    SpatialConvolution(32,3,3, padding='SAME', bias=False,name='L1_Convolution'),

    BatchNormalization(name='L2_Batch'),
    # ReLU(name='L3_ReLU'),
    HardTanh(name='L3_HardTanh'),
    SpatialConvolution(64, 3, 3, padding='SAME', bias=False, name='L4_Convolution'),
    SpatialMaxPooling(2,2,2,2,name='L5_MaxPooling',padding='SAME'),

    # ReLU(name='L5_ReLU'),

    BatchNormalization(name='L6_Batch'),
    HardTanh(name='L7_HardTanh'),
    # ReLU(name='L7_ReLU'),
    SpatialConvolution(128, 3, 3, padding='SAME', bias=False, name='L8_Convolution'),
    SpatialMaxPooling(2,2,2,2,name='L9_MaxPooling',padding='SAME'),
    # ReLU(name='L8_ReLU'),

    BatchNormalization(name='L10_Batch'),
    HardTanh(name='L11_HardTanh'),
    # ReLU(name='L11_ReLU'),
    SpatialConvolution(256, 3, 3, padding='SAME', bias=False, name='L12_Convolution'),
    SpatialMaxPooling(2,2,2,2,name='L13_MaxPooling',padding='SAME'),
    # ReLU(name='L11_ReLU'),

    BatchNormalization(name='L14_Batch'),
    HardTanh(name='L15_HardTanh'),
    # ReLU(name='L15_ReLU'),
    Affine(625, bias=False,name='L16_FullyConnected'),
    # ReLU(name='L14_ReLU'),

    BatchNormalization(name='L17_Batch'),
    HardTanh(name='L18_HardTanh'),
    # ReLU(name='L18_ReLU'),
    Affine(10,bias=False,name='L19_FullyConnected'),
])
