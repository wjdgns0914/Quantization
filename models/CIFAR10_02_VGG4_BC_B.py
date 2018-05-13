from test_codes.nnUtils import *
FLAGS = tf.app.flags.FLAGS
print("Model")
print(FLAGS.Drift1,FLAGS.Drift2,FLAGS.Inter_variation_options)
Dri1=FLAGS.Drift1
Dri2=FLAGS.Drift2

model = Sequential([
    BinarizedWeightOnlySpatialConvolution(32,3,3, padding='SAME', bias=False,name='L1_Convolution',Drift=Dri1),
    BatchNormalization(name='L2_Batch'),
    ReLU(name='L3_ReLU'),
    BinarizedWeightOnlySpatialConvolution(32, 3, 3, padding='SAME', bias=False, name='L4_Convolution',Drift=Dri1),
    SpatialMaxPooling(2, 2, 2, 2, name='L5_MaxPooling', padding='SAME'),

    BatchNormalization(name='L6_Batch'),
    ReLU(name='L7_ReLU'),
    BinarizedWeightOnlySpatialConvolution(64, 3, 3, padding='SAME', bias=False, name='L8_Convolution',Drift=Dri1),
    BatchNormalization(name='L9_Batch'),
    ReLU(name='L10_ReLU'),
    BinarizedWeightOnlySpatialConvolution(64, 3, 3, padding='SAME', bias=False, name='L11_Convolution',Drift=Dri1),
    SpatialMaxPooling(2, 2, 2, 2, name='L12_MaxPooling', padding='SAME'),

    BatchNormalization(name='L13_Batch'),
    ReLU(name='L14_ReLU'),
    BinarizedWeightOnlyAffine(512, bias=False,name='L15_FullyConnected',Drift=Dri2),

    BatchNormalization(name='L16_Batch'),
    ReLU(name='L17_ReLU'),
    BinarizedWeightOnlyAffine(10,bias=False,name='L18_FullyConnected',Drift=Dri2),
])