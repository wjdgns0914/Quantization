from funcLayer import *
FLAGS = tf.app.flags.FLAGS
print("Model")
print(FLAGS.Drift1,FLAGS.Drift2)
model = Sequential([
    BinarizedAffine(256, bias=False,name='L1_FullyConnected'),
    Sigmoid(name='L2_ReLU'),
    BinarizedAffine(10,bias=False,name='L3_FullyConnected'),
])
