from test_codes.nnUtils import *

model = Sequential([
    SpatialConvolution(64,3,3, padding='VALID', bias=False),
    SpatialMaxPooling(2,2,2,2),
    BatchNormalization(),
    ReLU(),
    SpatialConvolution(64,3,3, padding='SAME', bias=False),
    SpatialMaxPooling(2,2,2,2),
    BatchNormalization(),
    ReLU(),
    SpatialConvolution(128,3,3, padding='SAME', bias=False),
    SpatialMaxPooling(2, 2, 2, 2),
    BatchNormalization(),
    ReLU(),
    Affine(1024, bias=False),
    BatchNormalization(),
    ReLU(),
    Dropout(0.5),
    Affine(1024, bias=False),
    BatchNormalization(),
    ReLU(),
    Dropout(0.5),
    Affine(10)
])
