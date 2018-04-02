from nnUtils import *
FLAGS = tf.app.flags.FLAGS
print("Model")
print(FLAGS.Drift1,FLAGS.Drift2,FLAGS.Variation)
model = Sequential([
    BinarizedWeightOnlyAffine(512, bias=False,name='L1_FullyConnected'),
    BatchNormalization(),
    ReLU(name='L2_ReLU'),
    BinarizedWeightOnlyAffine(512, bias=False, name='L3_FullyConnected'),
    BatchNormalization(),
    ReLU(name='L4_ReLU'),
    BinarizedWeightOnlyAffine(512, bias=False, name='L5_FullyConnected'),
    BatchNormalization(),
    ReLU(name='L6_ReLU'),
    BinarizedWeightOnlyAffine(512, bias=False, name='L7_FullyConnected'),
    BatchNormalization(),
    ReLU(name='L8_ReLU'),
    BinarizedWeightOnlyAffine(10,bias=False,name='L9_FullyConnected'),
])
