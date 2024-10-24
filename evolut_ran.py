# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: privileged_pyenv
#     language: python
#     name: python3
# ---

# %%
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, mean_absolute_error
import datasets as dat
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import utils as ut
import models as mo
import scipy.optimize as so
import argparse


# %%

#python NN_train_parser.py -dataset obesity  -no-drp  -es -epo 100 -bs 32 -samp 100 -uncert 0.3 -iter 1
#python NN_train_parser.py -dataset phishing obesity diabetes wm  -l_reg 10 20 -l_clas 10 20 -drp  -es -epo 200 -bs 32 -samp 100 -uncert 0.3 -iter 20

#python NN_train_synt.py  -no-drp -epo 10 -bs 50 -pat 20 -no-regu -l2regu 1 -iter 10
#python evolut.py -train 200 -test 200 -epo 1 -bs 40 -pat 20 -iter 1


parser = argparse.ArgumentParser()
# Define arguments
parser.add_argument("-l_clas", dest = "l_clas", nargs = '+', default = [], type = int)
parser.add_argument("-train", dest = "train", type = int)
parser.add_argument("-test", dest = "test", type = int)
parser.add_argument("-np", dest = "np", type = float)
#parser.add_argument("-es", dest = "es", action='store_true')
#parser.add_argument("-no-es", dest = "es", action='store_false')
parser.add_argument("-epo", dest = "epo", type = int)
parser.add_argument("-bs", dest = "bs", type = int)



parser.add_argument("-pat", dest = "pat", type = int)


parser.add_argument("-iter", dest = "iter", type = int)
args = parser.parse_args()



# %%
# NN architectures of the regressor and the classifier
lay_clas = args.l_clas      #Layers for the classifier

drp = False    #Dropout False
epo = args.epo     #Epochs 100
bs = args.bs       #Batch Size 32
vs = 0.33          #Validation Split
#es = args.es      #Early Stopping True
pat = args.pat           #Patience

temperature = 1    #Temperature privileged distillation algorithms
imitation = 0.5    #Imitation parameters
beta = 1           #Weight od misclassifications TPD

#Regularization
regu = False
l2regu = 0

# Determines the number of iterations of the k-fold CV
n_iter  = args.iter
n_tr = args.train
n_test = args.test

noise_priv = args.np
'''
n_iter = 5
epo = 10
bs = 200
pat = 20
lay_clas = []
temperature = 1    #Temperature privileged distillation algorithms
imitation = 0.5    #Imitation parameters
beta = 1           #Weight od misclassifications TPD
vs = 0.33          #Validation Split
drp = False
n_tr = 200
n_test = 2000
'''


dff = pd.DataFrame()  #Dataframe to store the results of each dataset



def synthetic(n, l, a, noise_priv):
    Xr = np.random.randn(n, len(a))
    Xpr = np.random.randn(n)
    Xp = np.random.randn(n)
    y = pd.Series(((l*Xp + (1-l)*Xpr + noise_priv*np.random.randn(n)) > 0)*1)
    Xr = pd.DataFrame(Xr, columns =['Xr1', 'Xr2'])
    Xp = pd.DataFrame({'Xpr': Xpr, 'Xp': Xp})
    X = pd.concat([Xr, Xp], axis = 1)

    pi_features = ['Xp']
    return X, y, pi_features



# %%

#Process each dataset
for l in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]:
    #n_tr = 200
    #n_test = 2000

    #Create a list to save the results 
    err_up, err_b = [[] for i in range(2)]
    err_up_priv, err_gd, err_pfd, err_tpd = [[] for i in range(4)]
    #For each fold (k is the random seed of each fold)
    for k in range(n_iter):
        #Create a dictionary with all the fold partitions
        a = np.random.randn(2)
        X_train, y_train, pi_features = synthetic(n_tr, l, a, noise_priv)
        X_test, y_test, pi_features = synthetic(n_test, l, a, noise_priv)

        SS = StandardScaler()
        X_train = pd.DataFrame(SS.fit_transform(X_train), columns = X_train.columns)
        X_test = pd.DataFrame(SS.transform(X_test), columns = X_train.columns)
        

        # Get the privileged feature
        pri = X_train[pi_features]
        pri_test = X_test[pi_features]

        #Drop the privileged feature from the train set
        X_trainr = X_train.drop(pi_features, axis = 1)
        X_testr = X_test.drop(pi_features, axis = 1)
        
        #TRAIN THE THREE MAIN MODELS: UPPER, LOWER AND PRIVILEGED
        ###########################################################

        #UPPER (PRIV)
        #----------------------------------------------------------
        #Create the model 
        model =  mo.nn_binary_clasification(pri.shape[1], lay_clas, 'relu', dropout = drp, regularization = False, l2 = 1)   
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        
        #Fit the model
        #mo.fit_model(model, pri, y_train, epo, bs, vs, es, pat)
        model.fit(pri, y_train, epochs=epo, batch_size=bs, verbose = 0, validation_split = vs,
                    callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience = pat)])
        
        #Measure test error
        y_pre = np.ravel([np.round(i) for i in model.predict(pri_test)])
        y_proba_tr_p = model.predict(pri)
        err_up_priv.append(1-accuracy_score(y_test, y_pre))
        

        #UPPER (REGULAR + PRIV)
        #----------------------------------------------------------
        #Create the model 
        model =  mo.nn_binary_clasification( X_train.shape[1], lay_clas, 'relu', dropout = drp, regularization = False, l2 = 1)     
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        
        #Fit the model
        #mo.fit_model(model, X_train, y_train, epo, bs, vs, es, pat)
        model.fit(X_train, y_train, epochs=epo, batch_size=bs, verbose = 0, validation_split = vs,
                    callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience = pat)])
        
        #Measure test error
        y_pre_up = np.ravel([np.round(i) for i in model.predict(X_test)])
        y_proba_tr = model.predict(X_train)
        err_up.append(1-accuracy_score(y_test, y_pre_up))
        

        
        #LOWER
        #----------------------------------------------------------
        
        #Create the model 
        
        
        model =  mo.nn_binary_clasification( X_trainr.shape[1], lay_clas, 'relu', dropout = drp, regularization = False, l2 = 1)  
        model.compile(loss='binary_crossentropy', optimizer="adam", metrics=['accuracy'])
        #mo.fit_model(model, X_trainr, y_train, epo, bs, vs, es, pat)
        model.fit(X_trainr, y_train, epochs=epo, batch_size=bs, verbose = 0, validation_split = vs,
                    callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience = pat)])


        #Measure test error
        y_pre_b = np.ravel([np.round(i) for i in model.predict(X_testr)])

        err_b.append(1-accuracy_score(y_test, y_pre_b))
        
        #STATE OF THE ART PRIVILEGED 
        #---------------------------------------------------------- 
        #### GD
        #### ---------------------------------------------------------- 
        yy_GD = np.column_stack([np.ravel(y_train), 
                            np.ravel(y_proba_tr_p)])
        
        model =  mo.nn_binary_clasification( X_trainr.shape[1], lay_clas, 'relu', dropout = drp, regularization = False, l2 = 1)     
        model.compile(loss= ut.loss_GD(temperature, imitation), optimizer= 'adam', metrics=['accuracy'])
        
        #Fit the model
        #mo.fit_model(model, X_trainr, yy_GD, epo, bs, vs, es, pat)
        model.fit(X_trainr, yy_GD, epochs=epo, batch_size=bs, verbose = 0, validation_split = vs,
                    callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience = pat)])
        
        #Measure test error
        y_pre = np.ravel([np.round(i) for i in model.predict(X_testr)])
        err_gd.append(1-accuracy_score(y_test, y_pre))
        
        

        #### ---------------------------------------------------------- 
        #### PFD
        #### ---------------------------------------------------------- 
        yy_PFD = np.column_stack([np.ravel(y_train), 
                            np.ravel(y_proba_tr)])
        
        model =  mo.nn_binary_clasification( X_trainr.shape[1], lay_clas, 'relu', dropout = drp, regularization = False, l2 = 1)     
        model.compile(loss= ut.loss_GD(temperature, imitation), optimizer='adam', metrics=['accuracy'])
        
        #Fit the model
        #mo.fit_model(model, X_trainr, yy_PFD, epo, bs, vs, es, pat)
        model.fit(X_trainr, yy_PFD, epochs=epo, batch_size=bs, verbose = 0, validation_split = vs,
                    callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience = pat)])
        
        #Measure test error
        y_pre = np.ravel([np.round(i) for i in model.predict(X_testr)])
        err_pfd.append(1-accuracy_score(y_test, y_pre))          

        
        #### TPD
        #### ---------------------------------------------------------- 
        delta_i = np.array((y_train == np.round(np.ravel(y_proba_tr)))*1)
        yy_TPD = np.column_stack([np.ravel(y_train), np.ravel(y_proba_tr), delta_i])
        
        model =  mo.nn_binary_clasification( X_trainr.shape[1], lay_clas, 'relu', dropout = drp, regularization = False, l2 = 1)     
        model.compile(loss= ut.loss_TPD(temperature, beta), optimizer='adam', metrics=['accuracy'])

        #Fit the model
        #mo.fit_model(model, X_trainr, yy_TPD, epo, bs, vs, es, pat)
        model.fit(X_trainr, yy_TPD, epochs=epo, batch_size=bs, verbose = 0, validation_split = vs,
                    callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience = pat)])
        
        #Measure test error
        y_pre = np.ravel([np.round(i) for i in model.predict(X_testr)])
        err_tpd.append(1-accuracy_score(y_test, y_pre))
        
        
        tf.keras.backend.clear_session()
    
    #Save the results
    off = {'l' : l,
        'tp':np.round(np.mean(err_up_priv), 3),
        'tpr':np.round(np.mean(err_up), 3),
        'base':  np.round(np.mean(err_b), 3),
        'GD': np.round(np.mean(err_gd), 3),
        'PFD': np.round(np.mean(err_pfd), 3),
        'TPD': np.round(np.mean(err_tpd), 3),
        'LUPIGD %': np.round(ut.LUPI_gain(np.round(np.mean(err_up_priv), 3),  np.round(np.mean(err_b), 3), np.round(np.mean(err_gd), 3)),1),
        'LUPIPFD %': np.round(ut.LUPI_gain(np.round(np.mean(err_up), 3),  np.round(np.mean(err_b), 3), np.round(np.mean(err_pfd), 3)),1),
        'LUPITPD %': np.round(ut.LUPI_gain(np.round(np.mean(err_up), 3),  np.round(np.mean(err_b), 3), np.round(np.mean(err_tpd), 3)),1),
        'std_tp':np.round(np.std(err_up_priv), 3),
        'std_tpr':np.round(np.std(err_up), 3),
        'std_b':  np.round(np.std(err_b), 3),
        'std_GD': np.round(np.std(err_gd), 3),
        'std_PFD': np.round(np.std(err_pfd), 3),
        'std_TPD': np.round(np.std(err_tpd), 3)
        }   
    
    df1 = pd.DataFrame(off, index = [0])
        
    dff  = pd.concat([dff, df1]).reset_index(drop = True)

#layers, Iteraciones, dropout, earlystopping, uncertainty threshold y sampling
str_clas = [str(i) for i in args.l_clas]
lc = '-'.join(str_clas)

dff.to_csv('synth_ran' + '_'  + lc + '_' + str(n_tr) + '_' + str(n_test) + '_' + str(noise_priv) + '_' + str(epo) + '_' + str(bs)  + '_' +  str(pat)  + '_' + str(n_iter)+ '.csv')
    
        



# %%
