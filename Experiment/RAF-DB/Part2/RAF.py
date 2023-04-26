import h5py
import tensorflow as tf
import numpy as np
import os

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model
from tensorflow import keras
from sklearn.metrics import balanced_accuracy_score
from Net import Fusion_Net

os.environ["CUDA_VISIBLE_DEVICES"] = '3'
session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
session_conf.gpu_options.allow_growth = True
session_conf.gpu_options.per_process_gpu_memory_fraction = 0.99
sess = tf.Session(config=session_conf)
keras.backend.set_session(sess)

if __name__ == '__main__':
    with h5py.File('./raf_sample_label.h5', 'r') as f:
        x_l_tr_raf = f['x_l_tr_raf']
        x_g_tr_raf = f['x_g_tr_raf']
        x_c_tr_raf = f['x_c_tr_raf']
        x_l_te_raf = f['x_l_te_raf']
        x_g_te_raf = f['x_g_te_raf']
        x_c_te_raf = f['x_c_te_raf']
        label_tr = f['label_tr']
        label_te = f['label_te']

    label_tr, label_te = to_categorical(label_tr), to_categorical(label_te)

    model = Fusion_Net()
    model.compile(optimizer=keras.optimizers.Nadam(lr=0.0001), loss='categorical_crossentropy', metrics=['acc'])

    file_path = './RAF_best_model.h5'

    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss',
                                                  verbose=1,
                                                  mode='auto',
                                                  factor=0.1,
                                                  min_delta=0.001,
                                                  patience=50,
                                                  cooldown=0,
                                                  min_lr=0.00000001)

    checkpoint = keras.callbacks.ModelCheckpoint(filepath=file_path,
                                                 monitor='val_acc',
                                                 mode='auto',
                                                 save_best_only=True,
                                                 verbose=1,
                                                 save_weights_only=True
                                                 )

    model.fit(x=[x_l_tr_raf, x_g_tr_raf, x_c_tr_raf], y=label_tr,
              batch_size=32,
              epochs=2000,
              verbose=1,
              callbacks=[checkpoint, reduce_lr],
              validation_split=0.2)

    model = load_model(filepath=file_path, compile=False)

    y_pred = model.predict([x_l_tr_raf, x_g_tr_raf, x_c_tr_raf], batch_size=32, verbose=1)
    y_pred = np.argmax(y_pred, axis=1)
    y_true = np.argmax(label_te, axis=1)

    print(balanced_accuracy_score(y_true=y_true, y_pred=y_pred))
