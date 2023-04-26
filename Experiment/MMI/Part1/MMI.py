import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
import os
import cv2
import h5py

from sklearn.metrics import accuracy_score
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.models import load_model, Model

from Net import Enhance_Net, Patches

os.environ["CUDA_VISIBLE_DEVICES"] = '2'
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
        if filename[i][5] == '1':
            label.append(0)
            img = cv2.imread(path + filename[i])
            sample.append(img.astype('float32') / 255.0)
        elif filename[i][5] == '2':
            label.append(1)
            img = cv2.imread(path + filename[i])
            sample.append(img.astype('float32') / 255.0)
        elif filename[i][5] == '3':
            label.append(2)
            img = cv2.imread(path + filename[i])
            sample.append(img.astype('float32') / 255.0)
        elif filename[i][5] == '4':
            label.append(3)
            img = cv2.imread(path + filename[i])
            sample.append(img.astype('float32') / 255.0)
        elif filename[i][5] == '5':
            label.append(4)
            img = cv2.imread(path + filename[i])
            sample.append(img.astype('float32') / 255.0)
        else:
            label.append(5)
            img = cv2.imread(path + filename[i])
            sample.append(img.astype('float32') / 255.0)

    return np.array(sample), np.array(label)

if __name__ == '__main__':

    path = './MMI_Database/'
    x, y = Load_data(path)
    print(x.shape, y.shape)

    x_te_1, x_tr_1, y_te_1, y_tr_1 = x[:41], x[41:], y[:41], y[41:]
    x_te_2, x_tr_2, y_te_2, y_tr_2 = x[41:82], np.concatenate((x[:41], x[82:]), axis=0), y[41:82], np.concatenate((y[:41], y[82:]), axis=0)
    x_te_3, x_tr_3, y_te_3, y_tr_3 = x[82:123], np.concatenate((x[:82], x[123:]), axis=0), y[82:123], np.concatenate((y[:82], y[123:]), axis=0)
    x_te_4, x_tr_4, y_te_4, y_tr_4 = x[123:164], np.concatenate((x[:123], x[164:]), axis=0), y[123:164], np.concatenate((y[:123], y[164:]), axis=0)
    x_te_5, x_tr_5, y_te_5, y_tr_5 = x[164:], x[:164], y[164:], y[:164]

    model = Enhance_Net()
    # model.compile(optimizer=keras.optimizers.Nadam(lr=0.0001),
    #               loss=['categorical_crossentropy', 'categorical_crossentropy', 'categorical_crossentropy'],
    #               metrics=['acc'])

    for i in range(1):
        if i == 0:
            x_te, x_tr = x_te_1, x_tr_1
            y_te, y_tr = to_categorical(y_te_1, 6), to_categorical(y_tr_1, 6)
        elif i == 1:
            x_te, x_tr = x_te_2, x_tr_2
            y_te, y_tr = to_categorical(y_te_2, 6), to_categorical(y_tr_2, 6)
        elif i == 2:
            x_te, x_tr = x_te_3, x_tr_3
            y_te, y_tr = to_categorical(y_te_3, 6), to_categorical(y_tr_3, 6)
        elif i == 3:
            x_te, x_tr = x_te_4, x_tr_4
            y_te, y_tr = to_categorical(y_te_4, 6), to_categorical(y_tr_4, 6)
        else:
            x_te, x_tr = x_te_5, x_tr_5
            y_te, y_tr = to_categorical(y_te_5, 6), to_categorical(y_tr_5, 6)

        print(x_te.shape, x_tr.shape, y_te.shape, y_tr.shape)

        print('############# The {} Fold #############'.format(i + 1))

        file_path = './weights/best_weights_mmi_{}.h5'.format(i + 1)
        # file_path = './best_weights_mmi_{}.h5'.format(i + 1)
        reduce_lr = ReduceLROnPlateau(monitor='val_dense_8_acc',
                                      factor=0.1,
                                      patience=200,
                                      verbose=1,
                                      mode='auto',
                                      min_delta=0.0001,
                                      min_lr=0)
        checkpoint = ModelCheckpoint(filepath=file_path,
                                     monitor='val_dense_8_acc',
                                     verbose=1,
                                     save_best_only=True,
                                     save_weights_only=True,
                                     mode='auto')
        
        model.fit(x=x_tr, y=[y_tr, y_tr, y_tr],
                  batch_size=32,
                  epochs=2000,
                  verbose=1,
                  callbacks=[reduce_lr, checkpoint],
                  validation_split=0.2)

        model.load_weights('./weights/best_weights_mmi_{}.h5'.format(i + 1))
        model_pred = Model(inputs=model.input, outputs=[model.get_layer('dense_3').output, model.get_layer('dense_4').output, model.get_layer('dense_5').output])
        x_l_tr_mmi, x_g_tr_mmi, x_c_tr_mmi = model_pred.predict(x=x_tr, batch_size=32, verbose=1)
        model_pred = Model(inputs=model.input, outputs=[model.get_layer('dense_3').output, model.get_layer('dense_4').output, model.get_layer('dense_5').output])
        x_l_te_mmi, x_g_te_mmi, x_c_te_mmi = model_pred.predict(x=x_te, batch_size=32, verbose=1)

        with h5py.File('./outputs/mmi_sample_label_{}.h5'.format(i + 1), 'w') as f:
            f.create_dataset(name='x_l_tr_mmi', data=x_l_tr_mmi)
            f.create_dataset(name='x_g_tr_mmi', data=x_g_tr_mmi)
            f.create_dataset(name='x_c_tr_mmi', data=x_c_tr_mmi)
            f.create_dataset(name='x_l_te_mmi', data=x_l_te_mmi)
            f.create_dataset(name='x_g_te_mmi', data=x_g_te_mmi)
            f.create_dataset(name='x_c_te_mmi', data=x_c_te_mmi)
            f.create_dataset(name='label_tr', data=y_tr)
            f.create_dataset(name='label_te', data=y_te)
