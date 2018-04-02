from nnUtils import *
model = Sequential([
    SpatialConvolution(32,5,5, padding='SAME', bias=False,name='L1_Convolution'),
    BatchNormalization(name='L2_Batch'),
    ReLU(name='L3_ReLU'),
    SpatialMaxPooling(2, 2, 2, 2, name='L4_MaxPooling', padding='SAME'),
    SpatialConvolution(64, 5, 5, padding='SAME', bias=False, name='L5_Convolution'),
    BatchNormalization(name='L6_Batch'),
    ReLU(name='L7_ReLU'),
    SpatialMaxPooling(2, 2, 2, 2, name='L8_MaxPooling', padding='SAME'),

    Affine(512, bias=False,name='L9_FullyConnected'),
    BatchNormalization(name='L10_Batch'),
    ReLU(name='L11_ReLU'),

    Affine(10,bias=False,name='L12_FullyConnected'),
])