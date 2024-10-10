from sklearn.model_selection import StratifiedKFold, TimeSeriesSplit, KFold
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense, Dropout, Activation, Input, Concatenate, Lambda
from keras import backend as K
import math



#TODO: rename the parameters to be more informative
def skfold(X, y, n, r = 0, name = 'stratified'):
    """Returns the partitions for a K-fold cross validation

    Args:
        X (_type_): Input features
        y (_type_): Output feature
        n (_type_): number of splits
        r (int, optional): random seed. Defaults to 0.
        name (str, optional): _description_. Defaults to 'stratified'.

    Returns:
        _type_: _description_
    """
    if name == 'stratified':
        skf = StratifiedKFold(n_splits = n, shuffle = True, random_state = r)
    if name == 'time':
        skf = TimeSeriesSplit(n_splits = n)
    if name == 'kfold':
        skf = KFold(n_splits = n,  shuffle = True, random_state = r)
    
    d= {}
    j = 0
    for train_index, test_index in skf.split(X, y): 
            d['X_train' + str(j)] = X.loc[train_index]
            d['X_test' + str(j)] = X.loc[test_index]
            d['y_train' + str(j)] = y.loc[train_index]
            d['y_test' + str(j)] = y.loc[test_index]

            d['X_train' + str(j)].reset_index(drop = True, inplace = True)
            d['X_test' + str(j)].reset_index(drop = True, inplace = True)
            d['y_train' + str(j)].reset_index(drop = True, inplace = True)
            d['y_test' + str(j)].reset_index(drop = True, inplace = True)
            j+=1
    return d



def feat_correlation(X, y):
    """Selects as the privileged variable the most correlated one (w.r.t the class)

    Args:
        X (_type_): _description_
        y (_type_): _description_

    Returns:
        _type_: _description_
    """
    df = pd.concat([X, y], axis = 1)
    val1 = sorted(list(np.abs(df.corr(method = 'spearman')).iloc[-1]), reverse = True)[1]
    indice1 = list(np.abs(df.corr(method = 'spearman')).iloc[-1]).index(val1)
    pi_features = [X.columns[indice1]][0]
    return pi_features



#TODO: rename the parameters
def train_test_fold(dr, h):
    """Given a dictionary with k-fold partitions, selects the h-th one

    Args:
        dr (_type_): _description_
        h (_type_): _description_

    Returns:
        _type_: _description_
    """
    X_train = dr['X_train' + str(h)]
    y_train = dr['y_train' + str(h)]
    X_test = dr['X_test' + str(h)]
    y_test = dr['y_test' + str(h)]
    
    X_train = X_train.reset_index(drop = True)
    y_train = y_train.reset_index(drop = True)
    X_test = X_test.reset_index(drop = True)
    y_test = y_test.reset_index(drop = True)
    return X_train, y_train, X_test, y_test



def LUPI_gain(ub, lb, x):
    """Gain of the privileged model (x) with respect to the upper- and lower-bound models (ub and lb)

    Args:
        ub (_type_): _description_
        lb (_type_): _description_
        x (_type_): _description_

    Returns:
        _type_: _description_
    """
    return ((x - lb) / (ub - lb) )*100




#Loss for Generalized distillation algorithm

def loss_GD(T, l):
    def loss(y_true, y_pred):
        y_tr = y_true[:, 0]
        y_prob = y_true[:, 1]
        #Aquí estaba el problema con los nan añadiendo estas sumas se corrige
        ft = (-tf.math.log(1/(y_prob+1e-6) - 1 + 1e-6)) / T
        y_pr = 1 / (1 + tf.exp(-ft))
        #tf.print(y_pr)
        d1 = tf.keras.losses.BinaryCrossentropy()(y_tr, y_pred)
        d2 = tf.keras.losses.BinaryCrossentropy()(y_pr +1e-6, y_pred + 1e-6)
        #tf.print(d2)
        #No puedo Categorical por el single layer del output y_pred
        return (1-l)*d1 + l*d2
    return loss


def loss_TPD(T, beta):
    def loss(y_true, y_pred):
        y_tr = y_true[:, 0]
        y_prob = y_true[:, 1]
        d = y_true[:, 2]
        
        ft = (-tf.math.log(1/(y_prob+1e-6) - 1 + 1e-6)) / T
        y_pr = 1 / (1 + tf.exp(-ft))

        #BCE instance by instance
        bce = tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
        bce_inst = bce(y_pred, y_pr )
        return tf.reduce_mean(tf.math.multiply(d,bce_inst) - tf.math.multiply(1-d, bce_inst)) 
    return loss


