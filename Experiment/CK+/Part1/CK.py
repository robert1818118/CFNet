import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
import os
import cv2
import h5py
import skimage

from sklearn.metrics import accuracy_score
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.models import load_model, Model

from Net import Enhance_Net, Patches

os.environ["CUDA_VISIBLE_DEVICES"] = '3'
session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
session_conf.gpu_options.allow_growth = True
session_conf.gpu_options.per_process_gpu_memory_fraction = 0.99
sess = tf.Session(config=session_conf)
K.set_session(sess)

def Load_data(path):
    label = []
    sample = []
    filename = os.listdir(path)
    for i in range(len(filename)):
        if filename[i][5:7] == 'an':
            label.append(0)
            img = cv2.imread(path + filename[i])
            sample.append(img.astype('float32') / 255.0)
        elif filename[i][5:7] == 'co':
            label.append(1)
            img = cv2.imread(path + filename[i])
            sample.append(img.astype('float32') / 255.0)
        elif filename[i][5:7] == 'di':
            label.append(2)
            img = cv2.imread(path + filename[i])
            sample.append(img.astype('float32') / 255.0)
        elif filename[i][5:7] == 'fe':
            label.append(3)
            img = cv2.imread(path + filename[i])
            sample.append(img.astype('float32') / 255.0)
        elif filename[i][5:7] == 'ha':
            label.append(4)
            img = cv2.imread(path + filename[i])
            sample.append(img.astype('float32') / 255.0)
        elif filename[i][5:7] == 'sa':
            label.append(5)
            img = cv2.imread(path + filename[i])
            sample.append(img.astype('float32') / 255.0)
        else:
            label.append(6)
            img = cv2.imread(path + filename[i])
            sample.append(img.astype('float32') / 255.0)

    return np.array(sample), np.array(label)

def Add_Blur(sample, k_size):
    sample_blur = []
    for i in range(len(sample)):
        sample_blur.append(cv2.GaussianBlur(sample[i], ksize=(k_size, k_size), sigmaX=0))

    return np.array(sample_blur)

def Add_Noise(sample):
    sample_noise = []
    for i in range(len(sample)):
        sample_noise.append(skimage.util.random_noise(sample[i], mode='gaussian', seed=None, clip=True))

    return np.array(sample_noise)

if __name__ == '__main__':

    path = './CK_Database/'
    x, y = Load_data(path=path)
    print(x.shape, y.shape)
    x_te_1, x_tr_1, y_te_1, y_tr_1 = x[:65], x[65:], y[:65], y[65:]
    x_te_2, x_tr_2, y_te_2, y_tr_2 = x[65:130], np.concatenate((x[:65], x[130:]), axis=0), y[65:130], np.concatenate((y[:65], y[130:]), axis=0)
    x_te_3, x_tr_3, y_te_3, y_tr_3 = x[130:195], np.concatenate((x[:130], x[195:]), axis=0), y[130:195], np.concatenate((y[:130], y[195:]), axis=0)
    x_te_4, x_tr_4, y_te_4, y_tr_4 = x[195:260], np.concatenate((x[:195], x[260:]), axis=0), y[195:260], np.concatenate((y[:195], y[260:]), axis=0)
    x_te_5, x_tr_5, y_te_5, y_tr_5 = x[260:], x[:260], y[260:], y[:260]

    model = Enhance_Net()
    # opt = keras.optimizers.Nadam(lr=0.0001)
    # model.compile(optimizer=opt,
    #               loss=['categorical_crossentropy', 'categorical_crossentropy', 'categorical_crossentropy'],
    #               metrics=['acc'])

    for i in range(5):
        if i == 0:
            # x_te, x_tr = x_te_1, x_tr_1
            # x_te, x_tr = Add_Blur(sample=x_te_1, k_size=15), Add_Blur(sample=x_tr_1, k_size=15)
            x_te, x_tr = Add_Noise(sample=x_te_1), Add_Noise(sample=x_tr_1)
            y_te, y_tr = to_categorical(y_te_1, 7), to_categorical(y_tr_1, 7)
        elif i == 1:
            # x_te, x_tr = x_te_2, x_tr_2
            # x_te, x_tr = Add_Blur(sample=x_te_2, k_size=15), Add_Blur(sample=x_tr_2, k_size=15)
            x_te, x_tr = Add_Noise(sample=x_te_2), Add_Noise(sample=x_tr_2)
            y_te, y_tr = to_categorical(y_te_2, 7), to_categorical(y_tr_2, 7)
        elif i == 2:
            # x_te, x_tr = x_te_3, x_tr_3
            # x_te, x_tr = Add_Blur(sample=x_te_3, k_size=15), Add_Blur(sample=x_tr_3, k_size=15)
            x_te, x_tr = Add_Noise(sample=x_te_3), Add_Noise(sample=x_tr_3)
            y_te, y_tr = to_categorical(y_te_3, 7), to_categorical(y_tr_3, 7)
        elif i == 3:
            # x_te, x_tr = x_te_4, x_tr_4
            # x_te, x_tr = Add_Blur(sample=x_te_4, k_size=15), Add_Blur(sample=x_tr_4, k_size=15)
            x_te, x_tr = Add_Noise(sample=x_te_4), Add_Noise(sample=x_tr_4)
            y_te, y_tr = to_categorical(y_te_4, 7), to_categorical(y_tr_4, 7)
        else:
            # x_te, x_tr = x_te_5, x_tr_5
            # x_te, x_tr = Add_Blur(sample=x_te_5, k_size=15), Add_Blur(sample=x_tr_5, k_size=15)
            x_te, x_tr = Add_Noise(sample=x_te_5), Add_Noise(sample=x_tr_5)
            y_te, y_tr = to_categorical(y_te_5, 7), to_categorical(y_tr_5, 7)

        print(x_te.shape, x_tr.shape, y_te.shape, y_tr.shape)

        print('############# The {} Fold #############'.format(i+1))

        file_path = './weights/best_weights_ck_noise_{}.h5'.format(i + 1)

        reduce_lr = ReduceLROnPlateau(monitor='val_dense_8_acc',
                                      factor=0.1,
                                      patience=200,
                                      verbose=1,
                                      mode='auto',
                                      min_delta=0.0001,
                                      min_lr=0)
        checkpoint = ModelCheckpoint(filepath=file_path,
                                     monitor='val_dense_8_acc',
                                     mode='auto',
                                     save_best_only=True,
                                     verbose=1,
                                     save_weights_only=True
                                     )

        model.fit(x=x_tr, y=[y_tr, y_tr, y_tr],
                  batch_size=32,
                  epochs=2000,
                  verbose=2,
                  callbacks=[reduce_lr, checkpoint],
                  validation_split=0.2)

        model.load_weights('./weights/best_weights_ck_noise_{}.h5'.format(i + 1))
        model_pred = Model(inputs=model.input, outputs=[model.get_layer('dense_3').output, model.get_layer('dense_4').output, model.get_layer('dense_5').output])
        x_l_tr_ck, x_g_tr_ck, x_c_tr_ck = model_pred.predict(x=x_tr, batch_size=32, verbose=1)
        model_pred = Model(inputs=model.input, outputs=[model.get_layer('dense_3').output, model.get_layer('dense_4').output, model.get_layer('dense_5').output])
        x_l_te_ck, x_g_te_ck, x_c_te_ck = model_pred.predict(x=x_te, batch_size=32, verbose=1)

        with h5py.File('./outputs/ck_sample_label_noise_{}.h5'.format(i + 1), 'w') as f:
            f.create_dataset(name='x_l_tr_ck', data=x_l_tr_ck)
            f.create_dataset(name='x_g_tr_ck', data=x_g_tr_ck)
            f.create_dataset(name='x_c_tr_ck', data=x_c_tr_ck)
            f.create_dataset(name='x_l_te_ck', data=x_l_te_ck)
            f.create_dataset(name='x_g_te_ck', data=x_g_te_ck)
            f.create_dataset(name='x_c_te_ck', data=x_c_te_ck)
            f.create_dataset(name='label_tr', data=y_tr)
            f.create_dataset(name='label_te', data=y_te)