import h5py
import tensorflow as tf
import numpy as np
import os

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model
from tensorflow import keras
from sklearn.metrics import balanced_accuracy_score, accuracy_score
from Net import Fusion_Net

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
session_conf.gpu_options.allow_growth = True
session_conf.gpu_options.per_process_gpu_memory_fraction = 0.99
sess = tf.Session(config=session_conf)
keras.backend.set_session(sess)

if __name__ == '__main__':

    ACC = []

    model = Fusion_Net()
    model.compile(optimizer=keras.optimizers.Nadam(lr=0.0001), loss='categorical_crossentropy', metrics=['acc'])

    for i in range(5):
        with h5py.File('./input_noise/ck_sample_label_noise_{}.h5'.format(i+1), 'r') as f:
            x_l_tr_ck = np.array(f['x_l_tr_ck'])
            x_g_tr_ck = np.array(f['x_g_tr_ck'])
            x_c_tr_ck = np.array(f['x_c_tr_ck'])
            x_l_te_ck = np.array(f['x_l_te_ck'])
            x_g_te_ck = np.array(f['x_g_te_ck'])
            x_c_te_ck = np.array(f['x_c_te_ck'])
            label_tr = np.array(f['label_tr'])
            label_te = np.array(f['label_te'])

        # label_tr, label_te = to_categorical(label_tr), to_categorical(label_te)
        print(x_l_tr_ck.shape, x_g_tr_ck.shape, x_c_tr_ck.shape, x_l_te_ck.shape, x_g_te_ck.shape, x_c_te_ck.shape, label_tr.shape, label_te.shape)

        file_path = './models/CK_best_model_noise_{}.h5'.format(i + 1)

        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_acc',
                                                      verbose=1,
                                                      mode='auto',
                                                      factor=0.1,
                                                      min_delta=0.001,
                                                      patience=200,
                                                      cooldown=0,
                                                      min_lr=0.00000001)

        checkpoint = keras.callbacks.ModelCheckpoint(filepath=file_path,
                                                     monitor='val_acc',
                                                     mode='auto',
                                                     save_best_only=True,
                                                     verbose=1,
                                                     )

        model.fit(x=[x_l_tr_ck, x_g_tr_ck, x_c_tr_ck], y=label_tr,
                  batch_size=32,
                  epochs=3000,
                  verbose=2,
                  callbacks=[checkpoint, reduce_lr],
                  validation_split=0.2)

        model = load_model(filepath=file_path)

        y_pred = model.predict([x_l_te_ck, x_g_te_ck, x_c_te_ck], batch_size=32, verbose=1)
        y_pred = np.argmax(y_pred, axis=1)
        y_true = np.argmax(label_te, axis=1)

        ACC.append(accuracy_score(y_true=y_true, y_pred=y_pred))

    print(ACC)
    print(np.mean(ACC))
