import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense, Dropout, Activation, Input, Concatenate, Lambda
from keras import backend as K
import numpy as np
import pandas as pd

def nn_binary_clasification(dim, lay, activation, wup = None, dropout = False, dr = 0.25, regularization = False, l2 = 1, fine_tune = False):
    """_summary_

    Args:
        dim (_type_): _description_
        lay (_type_): _description_
        activation (_type_): _description_
        dropout (bool, optional): _description_. Defaults to False.
        dr (float, optional): _description_. Defaults to 0.1.

    Returns:
        _type_: _description_
    """
    
    kernel_init = tf.keras.initializers.RandomNormal(mean=0., stddev=0.01)
    bias_init = tf.keras.initializers.RandomNormal(mean=0., stddev=0.01)
    model = keras.Sequential()
    model.add(Input(shape=(dim,)))
    if lay:        
        for units in lay:
            if regularization:
                model.add(Dense(units, activation=activation, kernel_initializer = kernel_init, bias_initializer = bias_init, kernel_regularizer=keras.regularizers.l2(l2)))
            else:
                #model.add(Dense(units, activation=activation, kernel_initializer = kernel_init, bias_initializer = bias_init))
                model.add(Dense(units, activation=activation))

            if dropout:
                model.add(Dropout(dr))   
    if regularization:
        model.add(Dense(1, activation='sigmoid', kernel_initializer = kernel_init, bias_initializer = bias_init, kernel_regularizer=keras.regularizers.l2(l2)))
    else:
        #model.add(Dense(1, activation='sigmoid', kernel_initializer = kernel_init, bias_initializer = bias_init))
        model.add(Dense(1, activation='sigmoid'))
    if fine_tune:
        for i in range(len(lay)+1):
            model.layers[i].set_weights([wup[2*i], wup[2*i+1]])
            model.layers[i].trainable = True


    return model




def fit_model(model, X_train, y_train, epo, bs, vs, es, pat):
    if es:
        history = model.fit(X_train, y_train, epochs=epo, batch_size=bs, verbose = 0, validation_split = vs,
                        callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience = pat)])
    else:
        history = model.fit(X_train, y_train, epochs=epo, batch_size=bs, verbose = 0, validation_split = vs)
    return 

