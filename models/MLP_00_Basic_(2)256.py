FLAGS = tf.app.flags.FLAGS
print("Model")
print(FLAGS.Drift1,FLAGS.Drift2,FLAGS.Inter_variation_options)
model = Sequential([
    BinarizedWeightOnlyAffine(1024, bias=False,name='L1_FullyConnected'),
    BatchNormalization(),
    ReLU(name='L2_ReLU'),
    BinarizedWeightOnlyAffine(256, bias=False, name='L3_FullyConnected'),
    BatchNormalization(),
    ReLU(name='L5_ReLU'),
    BinarizedWeightOnlyAffine(1024, bias=False, name='L4_FullyConnected'),
    BatchNormalization(),
    ReLU(name='L6_ReLU'),
    BinarizedWeightOnlyAffine(10,bias=False,name='L7_FullyConnected'),
])
