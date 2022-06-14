import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Add, MaxPooling2D, AveragePooling2D, Activation, Dense, PReLU, Layer
from tensorflow.keras.layers import Input, BatchNormalization, GlobalAveragePooling2D, Concatenate, Cropping2D, Multiply, Lambda, Flatten, Reshape
from tensorflow.keras.activations import relu, softmax, sigmoid, tanh
from tensorflow.keras import initializers
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2

import tensorflow.keras.backend as K

class Patches(Layer):
    def __init__(self, patch_size, **kwargs):
        super(Patches, self).__init__(**kwargs)
        self.patch_size = patch_size

    def call(self, inputs, **kwargs):
        patch = tf.concat((tf.split(inputs,num_or_size_splits=7,axis=1)), axis=-1)
        patch = tf.concat((tf.split(patch, num_or_size_splits=7, axis=2)), axis=-1)
        return patch

    def get_config(self):
        config = {'patch_size':self.patch_size}
        base_config = super(Patches, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

def Global_Net(input, eps):
    x_g_1 = Conv2D(filters=64, kernel_size=(7, 7), strides=(1, 1), padding='same', activation=relu)(input)
    x_g_1 = BatchNormalization(axis=-1, epsilon=eps)(x_g_1)
    x_g_1 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(x_g_1)
    x_g_1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=relu)(x_g_1)
    x_g_1 = BatchNormalization(axis=-1, epsilon=eps)(x_g_1)
    x_g_1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x_g_1)

    x_g_2 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=relu)(x_g_1)
    x_g_2 = BatchNormalization(axis=-1, epsilon=eps)(x_g_2)
    x_g_2 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x_g_2)

    x_g_3 = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=relu)(x_g_2)
    x_g_3 = BatchNormalization(axis=-1, epsilon=eps)(x_g_3)
    x_g_3 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x_g_3)

    x_g_4 = Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=relu)(x_g_3)
    x_g_4 = BatchNormalization(axis=-1, epsilon=eps)(x_g_4)
    x_g_4 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x_g_4)

    return x_g_4

def Local_Net(input, eps):
    x_l_1 = Conv2D(filters=128, kernel_size=(7, 7), strides=(1, 1), padding='same', activation=relu)(input)
    x_l_1 = BatchNormalization(axis=-1, epsilon=eps)(x_l_1)
    x_l_1 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(x_l_1)
    x_l_1 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=relu)(x_l_1)
    x_l_1 = BatchNormalization(axis=-1, epsilon=eps)(x_l_1)
    x_l_1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x_l_1)

    x_l_2 = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=relu)(x_l_1)
    x_l_2 = BatchNormalization(axis=-1, epsilon=eps)(x_l_2)
    x_l_2 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x_l_2)

    x_l_3 = Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=relu)(x_l_2)
    x_l_3 = BatchNormalization(axis=-1, epsilon=eps)(x_l_3)
    x_l_3 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x_l_3)

    x_l_4 = Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=relu)(x_l_3)
    x_l_4 = BatchNormalization(axis=-1, epsilon=eps)(x_l_4)
    x_l_4 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x_l_4)

    return x_l_4

def Enhance_Net(eps = 1.1e-5):

    input = Input(shape=(224, 224, 3))

    x_l_0 = Patches(patch_size=32)(input)
    x_g_0 = input

    x_l = Local_Net(x_l_0, eps)
    x_g = Global_Net(x_g_0, eps)

    x_l = GlobalAveragePooling2D()(x_l)
    x_g = GlobalAveragePooling2D()(x_g)

    x_c = Concatenate()([x_l, x_g])

    x_l = Dense(units=512, activation=relu)(x_l)
    x_g = Dense(units=512, activation=relu)(x_g)
    x_c = Dense(units=1024, activation=relu)(x_c)

    x_l = Dense(units=1024, activation=relu)(x_l)
    x_g = Dense(units=1024, activation=relu)(x_g)
    x_c = Dense(units=2048, activation=relu)(x_c)

    out_l = Dense(units=7, activation=softmax)(x_l)
    out_g = Dense(units=7, activation=softmax)(x_g)
    out_c = Dense(units=7, activation=softmax)(x_c)

    model = Model(input, [out_l, out_g, out_c])

    return model

# model = Enhance_Net()
# model.summary()