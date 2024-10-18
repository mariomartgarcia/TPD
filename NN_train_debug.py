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
text    = ['phishing'] 
dataset = [ dat.phishing(from_csv = True), dat.obesity(from_csv = True), dat.diabetes(), dat.wm() , dat.phoneme(), dat.magictelescope(),  dat.mozilla4()]
datasets_dict = dict(zip(text, dataset))


# NN architectures of the regressor and the classifier
lay_reg  = []      #Layers for the regressor
lay_clas = []      #Layers for the classifier

drp = False     #Dropout False
epo = 1000     #Epochs 100
bs = 32       #Batch Size 32
vs = 0.33          #Validation Split
es = True          #Early Stopping True
pat = 20           #Patience

temperature = 1    #Temperature privileged distillation algorithms
imitation = 1      #Imitation parameters
beta = 1           #Weight od misclassifications TPD



# Determines the number of iterations of the k-fold CV
n_iter  = 1
ran = np.random.randint(1000, size = n_iter)

dff = pd.DataFrame()  #Dataframe to store the results of each dataset



# %%

#Process each dataset
for ind in text:
    t = ind #text of the current dataset

    #Retrieve the current dataset and extract the privileged feature
    X, y = datasets_dict[ind]
    pi_features = ut.feat_correlation(X,y)

    #Number of folds
    cv = 5
    
    #Create a list to save the results 
    err_up, err_b, err1, per1, err2, per2, mae1, mae2= [[] for i in range(8)]
    err_up_priv, err_gd, err_pfd, err_tpd = [[] for i in range(4)]
    #For each fold (k is the random seed of each fold)
    for k in ran:
        #Create a dictionary with all the fold partitions
        dr = ut.skfold(X, pd.Series(list(y)), cv, r = k, name = 'stratified')
    
        #Process each fold individually
        for h in range(cv):
            #Get the current partition
            X_train, y_train, X_test, y_test  = ut.train_test_fold(dr, h)
            
            #Preprocess the data
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

            model =  mo.nn_binary_clasification( 1, lay_clas, 'relu', dropout = drp)   
            model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
            
            #Fit the model
            mo.fit_model(model, pri, y_train, epo, bs, vs, es, pat)
            
            #Measure test error
            y_pre = np.ravel([np.round(i) for i in model.predict(pri_test)])
            y_proba_tr_p = model.predict(pri)
            err_up_priv.append(1-accuracy_score(y_test, y_pre))
            
  
            #UPPER (REGULAR + PRIV)
            #----------------------------------------------------------

            
            #Create the model 

            model =  mo.nn_binary_clasification( X_train.shape[1], lay_clas, 'relu', dropout = drp)   
            model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
            
            #Fit the model
            mo.fit_model(model, X_train, y_train, epo, bs, vs, es, pat)
            
            #Measure test error
            y_pre_up = np.ravel([np.round(i) for i in model.predict(X_test)])
            y_proba_tr = model.predict(X_train)
        

            err_up.append(1-accuracy_score(y_test, y_pre_up))
            
            if ft:
                weights_up = model.get_weights()



            #LOWER
            #----------------------------------------------------------
            
            #Create the model 
            
            
            model =  mo.nn_binary_clasification( X_trainr.shape[1], lay_clas, 'relu', dropout = drp)
            model.compile(loss='binary_crossentropy', optimizer="adam", metrics=['accuracy'])
            mo.fit_model(model, X_trainr, y_train, 1000, bs, vs, False, pat)
            #Measure test error
            y_pre_b = np.ravel([np.round(i) for i in model.predict(X_testr)])

            err_b.append(1-accuracy_score(y_test, y_pre_b))
            print('B', 1-accuracy_score(y_test, y_pre_b))

        
            


            #STATE OF THE ART PRIVILEGED 
            #---------------------------------------------------------- 
            #### GD
            #### ---------------------------------------------------------- 
            
            yy_GD = np.column_stack([np.ravel(y_train), 
                               np.ravel(y_proba_tr_p)])
            
            model =  mo.nn_binary_clasification( X_trainr.shape[1], lay_clas, 'relu', dropout = False)   
            model.compile(loss= loss_GD(temperature, imitation), optimizer= 'adam', metrics=['accuracy'])
            
            #Fit the model
            mo.fit_model(model, X_trainr, yy_GD, 1, 32, vs, es, pat)
            
            #Measure test error
            y_pre = np.ravel([np.round(i) for i in model.predict(X_testr)])
            err_gd.append(1-accuracy_score(y_test, y_pre))
            print(1-accuracy_score(y_test, y_pre))

            
            #### ---------------------------------------------------------- 
            #### PFD
            #### ---------------------------------------------------------- 
            yy_PFD = np.column_stack([np.ravel(y_train), 
                               np.ravel(y_proba_tr)])
            
            model =  mo.nn_binary_clasification( X_trainr.shape[1], lay_clas, 'relu', dropout = drp)   
            model.compile(loss= ut.loss_GD(temperature, imitation), optimizer='adam', metrics=['accuracy'])
            
            #Fit the model
            mo.fit_model(model, X_trainr, yy_PFD, epo, bs, vs, es, pat)
            
            #Measure test error
            V = model.predict(X_testr)
            y_pre = np.ravel([np.round(i) for i in model.predict(X_testr)])
            err_pfd.append(1-accuracy_score(y_test, y_pre))          

            
            #### TPD
            #### ---------------------------------------------------------- 

            delta_i = np.array((y_train == np.round(np.ravel(y_proba_tr)))*1)
            yy_TPD = np.column_stack([np.ravel(y_train), np.ravel(y_proba_tr), delta_i])
            
            model =  mo.nn_binary_clasification( X_trainr.shape[1], lay_clas, 'relu', dropout = drp)   
            model.compile(loss= loss_TPD(temperature, beta), optimizer='adam', metrics=['accuracy'])
   
            #Fit the model
            history = model.fit(X_trainr, yy_TPD, epochs=400, batch_size=128, verbose = 0, validation_split = vs)
            
            #Measure test error
            y_pre = np.ravel([np.round(i) for i in model.predict(X_testr)])
            err_tpd.append(1-accuracy_score(y_test, y_pre))
            

            tf.keras.backend.clear_session()
    
    #Save the results
    off = {'name' : t,
           'err_up':np.round(np.mean(err_up), 4),
           'err_b':  np.round(np.mean(err_b), 4),
           'err_GD': np.round(np.mean(err_gd), 4),
           'err_PFD': np.round(np.mean(err_pfd), 4),
           'err_TPD': np.round(np.mean(err_tpd), 4),
           'LUPIGD %': ut.LUPI_gain(np.round(np.mean(err_up), 4),  np.round(np.mean(err_b), 4), np.round(np.mean(err_gd), 4)),
           'LUPIPFD %': ut.LUPI_gain(np.round(np.mean(err_up), 4),  np.round(np.mean(err_b), 4), np.round(np.mean(err_pfd), 4)),
           'LUPITPD %': ut.LUPI_gain(np.round(np.mean(err_up), 4),  np.round(np.mean(err_b), 4), np.round(np.mean(err_tpd), 4))
           }   
    
    df1 = pd.DataFrame(off, index = [0])
        
    dff  = pd.concat([dff, df1]).reset_index(drop = True)

#layers, Iteraciones, dropout, earlystopping, uncertainty threshold y sampling
#v = '_'.join()
#str_reg = [str(i) for i in args.l_reg]
#lr = '-'.join(str_reg)
#str_clas = [str(i) for i in args.l_clas]
#lc = '-'.join(str_clas)

#dff.to_csv(v + '_' + lr + '_' + lc + '_' + str(drp) + '_' + str(es) + '_' + str(epo) + '_' + str(bs) + '_' + str(n_samples) + '_' + str(alpha) + '_' + str(n_iter)+ '.csv')
    
        



# %%
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
        
        d3 = tf.math.reduce_sum(-y_pr*tf.math.log(y_pred) -(1 - y_pr)*tf.math.log(1- y_pred))
        print('XXXXXXX')
        #tf.print(c)
        #print('-------')
        #tf.print(d2)
        
        return d2
        #tf.print(d2)
        
        #No puedo Categorical por el single layer del output y_pred

    return loss




def loss_TPD(T, beta):
    def loss(y_true, y_pred):
        y_tr = y_true[:, 0]
        y_prob = y_true[:, 1]
        d = y_true[:, 2]
        
        ft = (-tf.math.log(1/y_prob - 1 + 1e-6)) / T
        y_pr = 1 / (1 + tf.exp(-ft))
        #BCE instance by instance
        bce = tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
        bce_inst = bce(y_prob, y_pred )
    
        #tf.print(tf.math.multiply(1-d,bce_inst))
        #print(tf.math.multiply(d,bce_inst).numpy() - tf.math.multiply(1-d, bce_inst).numpy())
        #print(np.mean(np.array(tf.math.multiply(d,bce_inst))- np.array(tf.math.multiply(1-d,bce_inst))))
        #tf.print(tf.reduce_mean(tf.math.multiply(d,bce_inst) - tf.math.multiply(1-d, bce_inst)))
        return tf.reduce_mean(tf.math.multiply(d,bce_inst) - tf.math.multiply(1-d, bce_inst)) 
    return loss