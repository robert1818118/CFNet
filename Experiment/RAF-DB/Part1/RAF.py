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

os.environ["CUDA_VISIBLE_DEVICES"] = '3'
session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
session_conf.gpu_options.allow_growth = True
session_conf.gpu_options.per_process_gpu_memory_fraction = 0.99
sess = tf.Session(config=session_conf)
K.set_session(sess)

train_path = '/home/user001/XJH/RAF-DB/train/'
test_path = '/home/user001/XJH/RAF-DB/test/'

def Load_data(path):

    sample = []
    lable = []
    filename = os.listdir(path=path)

    for i in range(len(filename)):
        if filename[i][0] == '1':
            lable.append(0)
            img = cv2.imread(path+filename[i])
            sample.append(img.astype('float32')/255.0)
        elif filename[i][0] == '2':
            lable.append(1)
            img = cv2.imread(path+filename[i])
            sample.append(img.astype('float32')/255.0)
        elif filename[i][0] == '3':
            lable.append(2)
            img = cv2.imread(path+filename[i])
            sample.append(img.astype('float32')/255.0)
        elif filename[i][0] == '4':
            lable.append(3)
            img = cv2.imread(path+filename[i])
            sample.append(img.astype('float32')/255.0)
        elif filename[i][0] == '5':
            lable.append(4)
            img = cv2.imread(path+filename[i])
            sample.append(img.astype('float32')/255.0)
        elif filename[i][0] == '6':
            lable.append(5)
            img = cv2.imread(path+filename[i])
            sample.append(img.astype('float32')/255.0)
        else:
            lable.append(6)
            img = cv2.imread(path+filename[i])
            sample.append(img.astype('float32')/255.0)

    return np.array(sample), np.array(lable)

x_train, y_train = Load_data(path=train_path)
x_test, y_test = Load_data(path=test_path)

y_train, y_test = to_categorical(y_train, 7), to_categorical(y_test, 7)

model = Enhance_Net()
opt = keras.optimizers.Nadam(lr=0.0001)
model.compile(optimizer=opt, loss=['categorical_crossentropy', 'categorical_crossentropy', 'categorical_crossentropy'], metrics=['acc'])
file_path = './The_best_weights_raf_new.h5'

reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                              verbose=1,
                              mode='auto',
                              factor=0.1,
                              min_delta=0.001,
                              patience=50,
                              cooldown=0,
                              min_lr=0.00000001)

checkpoint = ModelCheckpoint(filepath=file_path,
                             monitor='val_loss',
                             mode='auto',
                             save_best_only=True,
                             verbose=1,
                             save_weights_only=True
                             )

print('########## fit processing ##########')

model.fit(x=x_train, y=[y_train, y_train, y_train],
          batch_size=32,
          epochs=2000,
          verbose=1,
          shuffle=True,
          callbacks=[checkpoint, reduce_lr],
          validation_split=0.2)

print('########## predict pocessing ##########')


model = load_model(filepath=file_path, custom_objects={"Patches":Patches}, compile=False)

y_pred = model.predict(x_test, batch_size=32, verbose=1)
y_pred = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

print(accuracy_score(y_true=y_true, y_pred=y_pred))


model.load_weights(filepath=file_path)
model_pred = Model(inputs=model.input, outputs=[model.get_layer('dense_3').output, model.get_layer('dense_4').output, model.get_layer('dense_5').output])
x_l_tr_raf, x_g_tr_raf, x_c_tr_raf = model_pred.predict(x=x_train, batch_size=32, verbose=1)
model_pred = Model(inputs=model.input, outputs=[model.get_layer('dense_3').output, model.get_layer('dense_4').output, model.get_layer('dense_5').output])
x_l_te_raf, x_g_te_raf, x_c_te_raf = model_pred.predict(x=x_test, batch_size=32, verbose=1)

with h5py.File('./raf_sample_label.h5', 'w') as f:
    f.create_dataset(name='x_l_tr_raf', data=x_l_tr_raf)
    f.create_dataset(name='x_g_tr_raf', data=x_g_tr_raf)
    f.create_dataset(name='x_c_tr_raf', data=x_c_tr_raf)
    f.create_dataset(name='x_l_te_raf', data=x_l_te_raf)
    f.create_dataset(name='x_g_te_raf', data=x_g_te_raf)
    f.create_dataset(name='x_c_te_raf', data=x_c_te_raf)
    f.create_dataset(name='label_tr', data=y_train)
    f.create_dataset(name='label_te', data=y_test)
