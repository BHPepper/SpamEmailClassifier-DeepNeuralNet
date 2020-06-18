# -*- coding: utf-8 -*-
"""
Benjamin H Pepper
B.H.Pepper@gmail.com
https://www.linkedin.com/in/benjamin-pepper-62936714b/
"""

import pandas as pd
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.utils import to_categorical
#from keras.constraints import maxnorm
#from tensorflow.keras import regularizers
#from tensorflow.keras import optimizers
from sklearn.metrics import accuracy_score
from tensorflow.keras.callbacks import TensorBoard
import tensorflow as tf

def get_spam_dat():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data"
    dat = pd.read_csv(url, sep=',', header=None) 
    return(dat)

def nn_mod(X, y, seed = 1, params = None):
    tf.random.set_seed(seed)
    y = to_categorical(y)
    mod = Sequential()
    mod.add(Dense(57, input_dim=np.shape(X)[1], activation = 'sigmoid'))
    mod.add(Dropout(.5))
    mod.add(Dense(12, activation = 'sigmoid'))
    mod.add(Dropout(.5))
    mod.add(Dense(2, activation = 'sigmoid'))
    mod.compile(loss = 'binary_crossentropy', optimizer = 'adam', 
                 metrics=['accuracy'])
    log_dir = 'logs'
    callback = [
        tf.keras.callbacks.EarlyStopping(monitor='loss', patience=1000),
        tf.keras.callbacks.TensorBoard(log_dir = log_dir, histogram_freq = 3),
    ]
    # on command line type "tensorboard --logdir logs" if you are in the
    # parent dir of logs
    class_weight = {0: 1., 1: 1.}
    mod.fit(X, y, epochs = 10100, batch_size = 99999, class_weight = class_weight,
            callbacks = callback)
    return(mod)

def nn_outer_perf(mod, X, y):
    preds = mod.predict(X)
    preds = [1 if x[1] > x[0] else 0 for x in preds]
    return(accuracy_score(y, preds))

def nn_score(mod, X, y):
    preds = mod.predict(X)
    return([x[1] for x in preds])
