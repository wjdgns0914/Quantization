#Small version of 'CIFAR10_04_acc90.py'
from test_codes.nnUtils import *
FLAGS = tf.app.flags.FLAGS
print("Model")
print(FLAGS.Drift1,FLAGS.Drift2,FLAGS.Inter_variation_options)
Dri1=FLAGS.Drift1
Dri2=FLAGS.Drift2

model = Sequential([
    SpatialConvolution(128,3,3, padding='SAME', bias=False),
    BatchNormalization(name='L2_Batch'),
    ReLU(name='L3_ReLU'),
    SpatialConvolution(128, 3, 3, padding='SAME', bias=False),
    SpatialMaxPooling(2, 2, 2, 2, name='L5_MaxPooling', padding='SAME'),
    BatchNormalization(name='L6_Batch'),
    ReLU(name='L7_ReLU'),

    SpatialConvolution(192, 3, 3, padding='SAME', bias=False),
    BatchNormalization(name='L9_Batch'),
    ReLU(name='L10_ReLU'),
    SpatialConvolution(192, 3, 3, padding='SAME', bias=False),
    SpatialMaxPooling(2, 2, 2, 2, name='L12_MaxPooling', padding='SAME'),
    BatchNormalization(name='L13_Batch'),
    ReLU(name='L14_ReLU'),

    SpatialConvolution(288, 3, 3, padding='SAME', bias=False),
    BatchNormalization(name='L16_Batch'),
    ReLU(name='L17_ReLU'),
    SpatialConvolution(288, 3, 3, padding='SAME', bias=False),
    SpatialMaxPooling(2, 2, 2, 2, name='L19_MaxPooling', padding='SAME'),
    BatchNormalization(name='L20_Batch'),
    ReLU(name='L21_ReLU'),

    Affine(512, bias=False),
    BatchNormalization(name='L23_Batch'),
    ReLU(name='L24_ReLU'),

    Affine(512, bias=False),
    BatchNormalization(name='L26_Batch'),
    ReLU(name='L27_ReLU'),

    Affine(10, bias=False),
    BatchNormalization(name='L29_Batch'),
])