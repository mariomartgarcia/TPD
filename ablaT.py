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

#python NN_train.py -dataset obesity  -no-drp -epo 100 -bs 32 -pat 20 -no-regu -l2regu 1 -iter 10
#python NN_train.py -dataset phishing obesity diabetes wm phoneme  -no-drp  -es -epo 1000 -bs 32 -no-regu -l2regu 1 -iter 10


parser = argparse.ArgumentParser()
# Define arguments
parser.add_argument("-dataset", dest = "dataset", nargs = '+' )
parser.add_argument("-l_clas", dest = "l_clas", nargs = '+', default = [], type = int)
parser.add_argument("-drp", dest = "drp", action='store_true')
parser.add_argument("-no-drp", dest = "drp", action='store_false')
#parser.add_argument("-es", dest = "es", action='store_true')
#parser.add_argument("-no-es", dest = "es", action='store_false')
parser.add_argument("-epo", dest = "epo", type = int)
parser.add_argument("-bs", dest = "bs", type = int)

parser.add_argument("-pat", dest = "pat", type = int)
parser.add_argument("-regu", dest = "regu", action='store_true')
parser.add_argument("-no-regu", dest = "regu", action='store_false')
parser.add_argument("-l2regu", dest = "l2regu", default = 0, type = float)

parser.add_argument("-iter", dest = "iter", type = int)
args = parser.parse_args()



# %%
text    = ['phishing', 'obesity', 'diabetes', 'wm', 'phoneme', 'magic_telescope', 'mozilla'] 
dataset = [ dat.phishing(from_csv = True), dat.obesity(from_csv = True), dat.diabetes(), dat.wm() , dat.phoneme(), dat.magictelescope(),  dat.mozilla4()]
datasets_dict = dict(zip(text, dataset))


# NN architectures of the regressor and the classifier
lay_clas = args.l_clas      #Layers for the classifier

drp = args.drp     #Dropout False
epo = args.epo     #Epochs 100
bs = args.bs       #Batch Size 32
vs = 0.33          #Validation Split
#es = args.es      #Early Stopping True
pat = args.pat           #Patience

temperature = 1    #Temperature privileged distillation algorithms
imitation = 0.5    #Imitation parameters
beta = 1           #Weight od misclassifications TPD

#Regularization
regu = args.regu
l2regu = args.l2regu


# Determines the number of iterations of the k-fold CV
n_iter  = args.iter
ran = np.random.randint(1000, size = n_iter)

dff = pd.DataFrame()  #Dataframe to store the results of each dataset
beta_val = [0.5, 1, 2, 5, 10, 20, 50]




# NN architectures of the regressor and the classifier
lay_clas = []
drp = False
epo = 2
bs = 500
vs = 0.33
pat = 20
temperature = 1    #Temperature privileged distillation algorithms
imitation = 0.5    #Imitation parameters
beta = 1           #Weight od misclassifications TPD
regu = False
l2regu = 0
n_iter  = 1
ran = np.random.randint(1000, size = n_iter)


beta_val = [0.5, 1]
temp_val = [1,2]


# %%

##---------------------------------------------------------------------
##---------------------------------------------------------------------
##---------------------------------------------------------------------
##---------------------------------------------------------------------
##OJOOOOOOOOOOOOOO ars.dataset
##---------------------------------------------------------------------
##---------------------------------------------------------------------
##---------------------------------------------------------------------
##---------------------------------------------------------------------
#Process each dataset
for ind in ['obesity']:
    t = ind #text of the current dataset

    #Retrieve the current dataset and extract the privileged feature
    X, y = datasets_dict[ind]
    pi_features = ut.feat_correlation(X,y)

    X = X.sample(frac = 1)
    ind = X.index
    X = X.reset_index(drop = True)
    y = y[ind].reset_index(drop = True)

    #Number of folds
    cv = 5
    
    #Create a list to save the results 
    err_up, err_b = [[] for i in range(2)]
    err_up_priv, err_gd, err_pfd, err_tpd, err_bci = [[] for i in range(5)]

    df_TB = pd.DataFrame()
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
            model =  mo.nn_binary_clasification( 1, lay_clas, 'relu', dropout = drp, regularization = False, l2 = 1)   
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
            err_abla1, err_abla2, err_abla3 = [[] for i in range(3)]


            for q in temp_val:

                yy_GD = np.column_stack([np.ravel(y_train), 
                                np.ravel(y_proba_tr_p)])
                
                model =  mo.nn_binary_clasification( X_trainr.shape[1], lay_clas, 'relu', dropout = drp, regularization = regu, l2 = l2regu)     
                model.compile(loss= ut.loss_GD(q, imitation), optimizer= 'adam', metrics=['accuracy'])
                
                #Fit the model
                #mo.fit_model(model, X_trainr, yy_GD, epo, bs, vs, es, pat)
                model.fit(X_trainr, yy_GD, epochs=epo, batch_size=bs, verbose = 0, validation_split = vs,
                            callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience = pat)])
                
                #Measure test error
                y_pre = np.ravel([np.round(i) for i in model.predict(X_testr)])
                err_abla1.append(1-accuracy_score(y_test, y_pre))
                
            

                #### ---------------------------------------------------------- 
                #### PFD
                #### ---------------------------------------------------------- 
                yy_PFD = np.column_stack([np.ravel(y_train), 
                                np.ravel(y_proba_tr)])
                
                model =  mo.nn_binary_clasification( X_trainr.shape[1], lay_clas, 'relu', dropout = drp, regularization = regu, l2 = l2regu)     
                model.compile(loss= ut.loss_GD(temperature, imitation), optimizer='adam', metrics=['accuracy'])
                
                #Fit the model
                #mo.fit_model(model, X_trainr, yy_PFD, epo, bs, vs, es, pat)
                model.fit(X_trainr, yy_PFD, epochs=epo, batch_size=bs, verbose = 0, validation_split = vs,
                            callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience = pat)])
                
                #Measure test error
                y_pre = np.ravel([np.round(i) for i in model.predict(X_testr)])
                err_abla2.append(1-accuracy_score(y_test, y_pre))          

            
                #### TPD
                #### ---------------------------------------------------------- 
                delta_i = np.array((y_train == np.round(np.ravel(y_proba_tr)))*1)
                yy_TPD = np.column_stack([np.ravel(y_train), np.ravel(y_proba_tr), delta_i])
                
                model =  mo.nn_binary_clasification( X_trainr.shape[1], lay_clas, 'relu', dropout = drp, regularization = regu, l2 = l2regu)     
                
                
                for b in beta_val:
                    model.compile(loss= ut.loss_TPD(temperature, b), optimizer='adam', metrics=['accuracy'])
                    #Fit the model
                    model.fit(X_trainr, yy_TPD, epochs=epo, batch_size=bs, verbose = 0, validation_split = vs,
                            callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience = pat)])
                    #Measure test error
                    y_pre = np.ravel([np.round(i) for i in model.predict(X_testr)])
                    err_abla3.append(1-accuracy_score(y_test, y_pre))
            
                '''
                #BCI
                ###-----------------------------------------------------------------

                model =  mo.nn_binary_clasification( X_trainr.shape[1], lay_clas, 'relu', dropout = drp)   

                model.compile(loss= ut.bce_inv, optimizer='adam', metrics=['accuracy'])
                #Fit the model
                model.fit(X_trainr, np.ravel(y_proba_tr), epochs=epo, batch_size=bs, verbose = 0, validation_split = vs,
                        callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience = pat)])
                #Measure test error
                y_pre = np.ravel([np.round(i) for i in model.predict(X_testr)])
                err_bci.append(1-accuracy_score(y_test, y_pre))
                '''
            if len(err_tpd) == 0:
                err_gd = err_abla1
                err_pfd = err_abla2
                err_tpd = err_abla3

            else:
                err_pfd = np.vstack([err_pfd, err_abla1])
                err_gd = np.vstack([err_gd, err_abla2])
                err_tpd = np.vstack([err_tpd, err_abla3])





                
            
            tf.keras.backend.clear_session()
    
    #Save the results
    off = {'name' : t,
           'tp':np.round(np.mean(err_up_priv), 3),
           'tpr':np.round(np.mean(err_up), 3),
           'base':  np.round(np.mean(err_b), 3),
           'std_tp':np.round(np.std(err_up_priv), 3),
           'std_tpr':np.round(np.std(err_up), 3),
           'std_b':  np.round(np.std(err_b), 3),    
            }   
    
    for q, k in enumerate(temp_val):
        off['PFD'+str(k)] =  np.mean(err_pfd, axis = 0)[q]
        off['std_PFD'+str(k)] =  np.std(err_pfd, axis = 0)[q]
        off['GD'+str(k)] =  np.mean(err_gd, axis = 0)[q]
        off['std_GD'+str(k)] =  np.std(err_gd, axis = 0)[q]


    df1 = pd.DataFrame(off, index = [0])
        
    dff  = pd.concat([dff, df1]).reset_index(drop = True)


    df_TB = pd.DataFrame(list(itertools.product(temp_val, beta_val)), columns = ['T', 'b'])
    df_TB['name'] = t
    df_TB['err_TPD'] = np.mean(err_tpd, axis = 0)
    df_TB['std_TPD'] = np.std(err_tpd, axis = 0)

#layers, Iteraciones, dropout, earlystopping, uncertainty threshold y sampling
v = '_'.join(args.dataset)
str_clas = [str(i) for i in args.l_clas]
lc = '-'.join(str_clas)

dff.to_csv('abla_' + v +  '_' + lc + '_' + str(drp) + '_' + str(epo) + '_' + str(bs)  + '_' +  str(pat)  + '_' + str(regu) + '_' + str(l2regu) + '_' + str(n_iter)+ '.csv')
    
        

# %%