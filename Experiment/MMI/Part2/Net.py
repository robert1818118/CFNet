import tensorflow as tf
import tensorflow.keras.backend as K

from tensorflow.keras.layers import Input, Dense, Lambda, Multiply, Add, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.activations import relu, softmax

def Cosin_similarity(input):
    dot1 = K.batch_dot(input[0], input[1], axes=1)
    dot2 = K.batch_dot(input[0], input[0], axes=1)
    dot3 = K.batch_dot(input[1], input[1], axes=1)
    max = K.maximum(K.sqrt(dot2 * dot3), K.epsilon())
    value = dot1 / max
    return K.tanh(value)

def Bund(input):
    alpha_1 = input[0]
    alpha_2 = input[1]

    alpha_l = alpha_1/(alpha_1+alpha_2)
    alpha_g = alpha_2/(alpha_1+alpha_2)

    return alpha_l, alpha_g

def Fusion_Net():
    input_1 = Input(shape=(1024, ))
    input_2 = Input(shape=(1024, ))
    input_3 = Input(shape=(2048, ))

    x_l = Dense(units=2048, activation=relu)(input_1)
    x_g = Dense(units=2048, activation=relu)(input_2)
    x_c = Dense(units=2048, activation=relu)(input_3)

    alpha_1 = Lambda(Cosin_similarity)([x_l, x_c])
    alpha_2 = Lambda(Cosin_similarity)([x_g, x_c])
    alpha_l, alpha_g = Lambda(Bund)([alpha_1, alpha_2])
    out_l = Multiply()([alpha_l, x_l])
    out_g = Multiply()([alpha_g, x_g])

    out = Concatenate()([out_l, out_g])

    out = Dense(units=6, activation=softmax)(out)

    model = Model([input_1, input_2, input_3], out)

    return model