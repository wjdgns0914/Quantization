from nnUtils import *
FLAGS = tf.app.flags.FLAGS
print("Model")
print(FLAGS.Drift1,FLAGS.Drift2,FLAGS.Variation)
Dri1=FLAGS.Drift1
Dri2=FLAGS.Drift2

model = Sequential([
    BinarizedWeightOnlySpatialConvolution(32,3,3, padding='SAME', bias=False,name='L1_Convolution',Drift=Dri1),
    BatchNormalization(name='L2_Batch'),
    ReLU(name='L3_ReLU'),
    BinarizedSpatialConvolution(32, 3, 3, padding='SAME', bias=False, name='L4_Convolution',Drift=Dri1),
    BatchNormalization(name='L5_Batch'),
    ReLU(name='L6_ReLU'),
    SpatialMaxPooling(2, 2, 2, 2, name='L7_MaxPooling', padding='SAME'),

    BinarizedSpatialConvolution(64, 3, 3, padding='SAME', bias=False, name='L8_Convolution',Drift=Dri1),
    BatchNormalization(name='L9_Batch'),
    ReLU(name='L10_ReLU'),
    BinarizedSpatialConvolution(64, 3, 3, padding='SAME', bias=False, name='L11_Convolution',Drift=Dri1),
    BatchNormalization(name='L12_Batch'),
    ReLU(name='L13_ReLU'),
    SpatialMaxPooling(2, 2, 2, 2, name='L14_MaxPooling', padding='SAME'),

    BinarizedAffine(512, bias=False,name='L15_FullyConnected',Drift=Dri2),
    BatchNormalization(name='L16_Batch'),
    ReLU(name='L17_ReLU'),

    BinarizedAffine(10,bias=False,name='L18_FullyConnected',Drift=Dri2),
])