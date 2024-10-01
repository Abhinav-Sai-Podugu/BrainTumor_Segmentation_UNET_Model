from uu import decode

import tensorflow as tf
from keras.src.layers import concatenate
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Conv2DTranspose, Concatenate, Input
from tensorflow.keras.models import Model

def conv_block(inputs, num_filters):
    X = Conv2D(num_filters, 3, padding = 'same')(inputs)
    X = BatchNormalization()(X)
    X = Activation("relu")(X)

    X = Conv2D(num_filters, 3, padding='same')(X)
    X = BatchNormalization()(X)
    X = Activation("relu")(X)

    return X

def encoder_block(inputs, num_filters):
    X = conv_block(inputs, num_filters)
    p = MaxPool2D((2, 2))(X)
    return X, p

def decoder_block(inputs, skip_features,  num_filters):
    X = Conv2DTranspose(num_filters, 2, strides = 2, padding = "same")(inputs)
    X = Concatenate()([X, skip_features])
    X = conv_block(X, num_filters)
    return X

def unet(input_shape):
    inputs = Input(input_shape)

    s1, p1 = encoder_block(inputs, 64)
    s2, p2 = encoder_block(p1, 128)
    s3, p3 = encoder_block(p2, 256)
    s4, p4 = encoder_block(p3, 512)

    b1 = conv_block(p4, 1024)

    d1 = decoder_block(b1, s4, 512)
    d2 = decoder_block(d1, s3, 256)
    d3 = decoder_block(d2, s2, 128)
    d4 = decoder_block(d3, s1, 64)

    outputs = Conv2D(1, 1, padding = "same")(d4)
    outputs = Activation("sigmoid")(outputs)

    model = Model(inputs, outputs, name = "UNET")
    return model


if __name__ == "__main__":
    input_shape = (256, 256, 3)
    model = unet(input_shape)
    model.summary()


