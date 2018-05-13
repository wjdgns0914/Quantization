from test_codes.nnUtils import *
FLAGS = tf.app.flags.FLAGS
print("Model")
print(FLAGS.Drift1,FLAGS.Drift2,FLAGS.Inter_variation_options)
# Dri1=FLAGS.Drift1
# Dri2=FLAGS.Drift2
Dri1=False
Dri2=False
model = Sequential([
    BinarizedWeightOnlySpatialConvolution(32,5,5, padding='SAME', bias=False,name='L1_Convolution'),
    BatchNormalization(name='L2_Batch'),
    ReLU(name='L3_ReLU'),
    SpatialMaxPooling(2, 2, 2, 2, name='L4_MaxPooling', padding='SAME'),

    BinarizedWeightOnlySpatialConvolution(64, 5, 5, padding='SAME', bias=False, name='L5_Convolution'),
    BatchNormalization(name='L6_Batch'),
    ReLU(name='L7_ReLU'),
    SpatialMaxPooling(2, 2, 2, 2, name='L8_MaxPooling', padding='SAME'),

    BinarizedWeightOnlyAffine(512, bias=False,name='L9_FullyConnected'),
    BatchNormalization(name='L10_Batch'),
    ReLU(name='L11_ReLU'),

    BinarizedWeightOnlyAffine(10,bias=False,name='L12_FullyConnected'),
])