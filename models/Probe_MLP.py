FLAGS = tf.app.flags.FLAGS
print("Model")
print(FLAGS.Drift1,FLAGS.Drift2,FLAGS.Inter_variation_options)
model = Sequential([
    BinarizedWeightOnlyAffine(256, bias=False,name='L1_FullyConnected',Drift=False),
    # Affine(256, bias=False,name='L1_FullyConnected'),
    # BatchNormalization(),
    ReLU(name='L2_ReLU'),
    # Affine(10, bias=False,name='L3_FullyConnected')
    BinarizedWeightOnlyAffine(10,bias=False,name='L3_FullyConnected',Drift=False),
])
