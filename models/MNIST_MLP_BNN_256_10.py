from nnUtils import *
FLAGS = tf.app.flags.FLAGS
print("Model")
print(FLAGS.Drift1,FLAGS.Drift2,FLAGS.Variation)
Dri1=FLAGS.Drift1
Dri2=FLAGS.Drift2
model = Sequential([
    BinarizedWeightOnlyAffine(256, bias=False,name='L1_FullyConnected',Drift=Dri2),
    ReLU(name='L2_ReLU'),
    BinarizedWeightOnlyAffine(10,bias=False,name='L3_FullyConnected',Drift=Dri2),
])
