import numpy as np
import pandas as pd
import seaborn as sns
import math
import lrplusTPD as priv
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import StratifiedKFold, TimeSeriesSplit, ShuffleSplit
from mrmr import mrmr_classif, mrmr_regression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split  
from sklearn.metrics import accuracy_score, roc_auc_score, mean_absolute_error
from sklearn.linear_model import SGDClassifier
import datasets as dat
import RVFL_plus as rvfl 
import KRVFL_plus as krvfl 
import svmplus as svmp
import lrplus as lrp
from sklearn import svm
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV

def skfold(X, y, n, r = 0, name = 'stratified', ts = None):
    if name == 'stratified':
        skf = StratifiedKFold(n_splits = n, shuffle = True, random_state = r)
    if name == 'time':
        skf = TimeSeriesSplit(n_splits = n)
    if name == 'train_tune':
        skf = ShuffleSplit(n_splits=n, test_size=ts, random_state=r)
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






# %%

#------------------------------------------------
#NÚMERO FIJO DE VARIABLES PRIVILEGIADAS
#------------------------------------------------

#X, y, pi_features = dat.ionosphere()
#--------------------
#OFFLINE
#--------------------


ran= np.random.randint(1000, size = 5)

name, model, lower, upper, opti, gainer = [], [], [], [], [], []
name2, model2, lower2, upper2, opti2, gainer2 = [], [], [], [], [], []
c_media = []


text  =  ['mu_284', 'phishing', 'obesity', 'diabetes', 'wm', 'magictelescope', 'phoneme', 'mozilla4']
dataset = [ dat.mu_284(), dat.phishing(), dat.obesity(), dat.diabetes(), dat.wm() , dat.magictelescope(), dat.phoneme(), dat.mozilla4()]

#text  =  ['phoneme', 'mozilla4']
#dataset = [  dat.phoneme(), dat.mozilla4()]



#text  = [ 'mnist_r', 'fruit', 'mnist_g' ] #, 'mozilla4']
#dataset = [   dat.mnist_r(), dat.fruit(), dat.mnist_g()] 

#text  = [  'mu_284']
#dataset = [   dat.mu_284()]

lambda_val = [1]
temperature = [1]

fin = pd.DataFrame([])

for index, datt in enumerate(dataset):
    gn = []
    t = text[index]
    if t in ['ionosphere', 'kc2', 'parkinsons', 'mnist_g', 'mnist_r', 'cifar10', 'fruit', 'fashion', 'hds']:
        X, y, pi_features = datt
        if t in ['mnist_g', 'mnist_r']:
            indexx = np.random.choice(y.index, size = 3000)
            y = y[indexx].reset_index(drop = True)
            X = X.iloc[indexx].reset_index(drop = True)
        
    else:
        X, y = datt
        df = pd.concat([X, y], axis = 1)
        val1 = sorted(list(np.abs(df.corr(method = 'spearman')).iloc[-1]), reverse = True)[1]
        indice1 = list(np.abs(df.corr(method = 'spearman')).iloc[-1]).index(val1)
        pi_features = [X.columns[indice1]]

    print(t)

    ite = 0

    tpr = []
    tp = []
    base = []

    e1, e2, e3, e4 = [], [], [], []
    e5, e6, e7, e8 = [], [], [], []

    for k in ran:   
        cv = 5
        dr = skfold(X, pd.Series(list(y)), cv, r = k, name = 'stratified')
        
        acc_lb, acc_tp, acc_tpr = [], [], []
        acc_tpd, acc_pfd, acc_pfds, acc_gd, acc_gds =  [], [], [], [], []
        
        ite+=1
        print(ite)
        
        for h in range(cv):
            
            if t in ['mnist_g', 'mnist_r', 'fruit']:
                X_train = dr['X_train' + str(h)]/255
                y_train = dr['y_train' + str(h)]
                X_test = dr['X_test' + str(h)]/255
                y_test = dr['y_test' + str(h)]
                
                
                X_train = X_train.reset_index(drop = True)
                y_train = y_train.reset_index(drop = True)
            
            else:
                X_train = dr['X_train' + str(h)]
                y_train = dr['y_train' + str(h)]
                X_test = dr['X_test' + str(h)]
                y_test = dr['y_test' + str(h)]
                
                
                X_train = X_train.reset_index(drop = True)
                y_train = y_train.reset_index(drop = True)
    
                SS = StandardScaler()
                X_train = pd.DataFrame(SS.fit_transform(X_train), columns = X_train.columns)
                X_test = pd.DataFrame(SS.transform(X_test), columns = X_train.columns)
            
            
            
            
            #PFD
            
            
            X_train_reg = X_train.drop(pi_features, axis = 1)
            X_test_reg = X_test.drop(pi_features, axis = 1)
            
            #LR_UPPER

            lr = LogisticRegression()#C = grid_search.best_params_['C'])
            lr.fit(X_train, y_train)
            
            y_pre = lr.predict_proba(X_test)[:,1]
            pre = lr.predict(X_test)
            acc_tpr.append(1-accuracy_score(y_test, pre))
            pre_train = lr.predict(X_train)
            c = np.array((y_train == pre_train)*1)
    
    
            #PRI
            omega = lr.coef_[0]
            beta = lr.intercept_
            
          


            #l_pfd, xxx, t_pfd, xxx, t_tpd = regularization(X_train, y_train, pi_features, lambda_val, temperature, teacher_full = True, sw = False, TPD = True)
            #print(l_pfd, t_pfd, t_tpd)
            l_pfd, t_pfd, t_tpd = 1, 1, 1
            
            #----------------------------------------------------
            
            #TPD
            tpd = priv.TPDGD(l = 1, T = t_tpd)
            tpd.fit(X_train, X_train_reg, np.array(y_train), omega, beta, c)
            pre_p = tpd.predict(pd.DataFrame(X_test_reg))
            acc_tpd.append(1-accuracy_score(y_test, pre_p))
            
    
            #PFD
            PFD = priv.GD(l = l_pfd, T = t_pfd)
            PFD.fit(X_train, X_train_reg, np.array(y_train), omega, beta)
            pre_p = PFD.predict(pd.DataFrame(X_test_reg))
            acc_pfd.append(1-accuracy_score(y_test, pre_p))
            
            #PFD INV
            #PFDsw = priv.GDsw(l = l_tpdi, T = 1)
            #PFDsw.fit(X_train, X_train_reg, np.array(y_train), omega, beta)
            #pre_p = PFDsw.predict(pd.DataFrame(X_test_reg))
            #acc_pfds.append(1-accuracy_score(y_test, pre_p))
            
            
            
            #----------------------------------------------------
            X_train = X_train[pi_features]
            X_test = X_test[pi_features]
            
            lr = LogisticRegression()#C = grid_search.best_params_['C'])
            lr.fit(X_train, y_train)
            
            y_pre = lr.predict_proba(X_test)[:,1]
            pre = lr.predict(X_test)
            acc_tp.append(1-accuracy_score(y_test, pre))
            pre_train = lr.predict(X_train)
            c = np.array((y_train == pre_train)*1)
    
    
            #PRI
            omega = lr.coef_[0]
            beta = lr.intercept_
            #----------------------------------------------------
            
            #l_gd, xxx, t_gd, xxx = regularization(X_train, y_train, pi_features, lambda_val, temperature, teacher_full = False, sw = False, TPD = False)
            l_gd, t_gd = 1, 1
            #print(l_gd, t_gd)
        
            
            #GD
            GenD = priv.GD(l = l_gd, T = t_gd)
            GenD.fit(X_train, X_train_reg, np.array(y_train), omega, beta)
            pre_p = GenD.predict(pd.DataFrame(X_test_reg))
            acc_gd.append(1-accuracy_score(y_test, pre_p))
            
            #GD INV
            #GenDsw = priv.GDsw(l = l_gdi, T = 1)
            #GenDsw.fit(X_train, X_train_reg, np.array(y_train), omega, beta)
            #pre_p = GenDsw.predict(pd.DataFrame(X_test_reg))
            #acc_gds.append(1-accuracy_score(y_test, pre_p))
            
            
        
        
            #LR_BASE
            
            lrb = LogisticRegression()#C = grid_search.best_params_['C'])
            lrb.fit(X_train_reg, y_train)
            pre = lrb.predict_proba(X_test_reg)[:,1]
            preb = lrb.predict(X_test_reg)
            acc_lb.append(1-accuracy_score(y_test, preb))
            
        
    
        tpr.append(np.round(np.mean(acc_tpr),3))
        tp.append(np.round(np.mean(acc_tp),3))
        base.append(np.round(np.mean(acc_lb),3))
        e1.append( np.round(np.mean(acc_tpd),3))
        e2.append( np.round(np.mean(acc_pfd),3))
        e3.append( np.round(np.mean(acc_pfds),3))
        e4.append( np.round(np.mean(acc_gd),3))
        e5.append( np.round(np.mean(acc_gds),3))

    
    off_lr = {'tpr': np.round(np.mean(tpr),3),
              'tp': np.round(np.mean(tp),3),
              'base': np.round(np.mean(base),3),
              'stdtpr': np.round(np.std(tpr),3),
              'stdtp': np.round(np.std(tp),3),
              'stdbase': np.round(np.std(base),3),

               'TPD': np.round(np.mean(e1),3),
               'PFD': np.round(np.mean(e2),3),
               'PFDsw': np.round(np.mean(e3),3),
               'GD': np.round(np.mean(e4),3),
               'GDsw': np.round(np.mean(e5),3),
               'stdTPD': np.round(np.std(e1),3),
               'stdPFD': np.round(np.std(e2),3),
               'stdPFDsw': np.round(np.std(e3),3),
               'stdGD': np.round(np.std(e4),3),
               'stdGDsw': np.round(np.std(e5),3)
               } 

    
    d1= pd.DataFrame(off_lr, index = [0])
    #d2= pd.DataFrame(off_svm, index = [0])
    
    #h = d1.drop(['tpr', 'tp', 'base', 'stdTPD', 'stdPFD', 'stdPFDsw', 'stdGD', 'stdGDsw', 'stdtpr', 'stdtp', 'stdbase'], axis = 1)
    #h = d1.drop(['err_lrup', 'err_lrb'], axis = 1)
    
    def LUPI_gain(ub, lb, x):
        return ((x - lb) / (ub - lb) )*100
    

    h_tpr = d1.drop(['tpr', 'tp', 'base', 'stdTPD', 'stdPFD', 'stdPFDsw',  'stdtpr', 'stdtp', 'stdbase', 'stdGD', 'stdGDsw', 'GD', 'GDsw'], axis = 1)
    gain_tpr = LUPI_gain(d1['tpr'].iloc[0], d1['base'].iloc[0], h_tpr)
    h_tp = d1.drop(['tpr', 'tp', 'base', 'stdTPD', 'stdPFD', 'stdPFDsw',  'stdtpr', 'stdtp', 'stdbase', 'stdGD', 'stdGDsw', 'TPD', 'PFD', 'PFDsw'], axis = 1)
    gain_tp = LUPI_gain(d1['tp'].iloc[0], d1['base'].iloc[0], h_tp)
    
    gain = pd.concat([gain_tp, gain_tpr], axis = 1)
    df_lr = pd.concat([d1, gain]).reset_index(drop = True).T
    print(df_lr)
    df_lr.to_csv('results/22_05_2024/segcat_regu/' + t +'.csv')
    
    #name.append(t)
    #model.append(h.stack().idxmin()[1])
    #upper.append(d1['err_lrup'].item())
    #lower.append(d1['err_lrb'].item())
    #opti.append(optimal)
    #gainer.append(gain_op)
    
    #h2 = d2.drop(['err_svmup', 'err_svmb' ], axis = 1)
    #optimal = h2.values.min()
    #gain = LUPI_gain(d2['err_svmup'].iloc[0], d2['err_svmb'].iloc[0], h2)
    #gain_op = LUPI_gain(d2['err_svmup'].iloc[0], d2['err_svmb'].iloc[0], optimal)
    

    #df_svm = pd.concat([d2, gain]).reset_index(drop = True).T
    #df_svm.to_csv('results/15_04_2024/' + t +'svm_priv.csv')
    
    #name2.append(t)
    #model2.append(h2.stack().idxmin()[1])
    #upper2.append(d2['err_svmup'].item())
    #lower2.append(d2['err_svmb'].item())
    #opti2.append(optimal)
    #gainer2.append(gain_op)

    
     
# %%
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
def regularization(X, y, pi_features, l_val, t_val, teacher_full = True, sw = False, TPD = False):
 
    cv = 5
    dr = skfold(X, y, cv, r = 0)
    t_val_h, l_val_h = [], []
    t_val_kdist, l_val_kdist = [], []
    t_val_tpd = []
    R_kdi, R_kd, R_tpd = [],[], []
    for j  in l_val:
        for t in t_val:
            racc_kdi,  racc_kd, racc_tpd =  [], [], []
            for h in range(cv):
                X_train = dr['X_train' + str(h)]
                y_train = dr['y_train' + str(h)]
                X_test = dr['X_test' + str(h)]
                y_test = dr['y_test' + str(h)]
                
                X_train = X_train.reset_index(drop = True)
                y_train = y_train.reset_index(drop = True)
                
                lr = LogisticRegression()
                lr.fit(X_train, y_train)
                pre_train = lr.predict(X_train)
                c = np.array((y_train == pre_train)*1)
                
                omega = lr.coef_[0]
                beta = lr.intercept_
                
                X_train_reg = X_train.drop(pi_features, axis = 1)
                X_test_reg = X_test.drop(pi_features, axis = 1)
                
                if teacher_full == False:
                    X_train = X_train[pi_features]
                    X_test = X_test[pi_features]
    
                #Modelos privilegiados
                #GENERALIZED DISTILLATION
                lrplusKD = priv.GD(l = j, T = t)
                lrplusKD.fit(X_train, X_train_reg, np.array(y_train), omega, beta)
                pre_p = lrplusKD.predict(pd.DataFrame(X_test_reg))
                racc_kd.append(1-accuracy_score(y_test, pre_p))
                
                if sw:
                    
                    #GENERALIZED DISTILLATION INV
                    lrplusKD = priv.GDsw(l = j, T = t)
                    lrplusKD.fit(X_train, X_train_reg, np.array(y_train), omega, beta)
                    pre_p = lrplusKD.predict(pd.DataFrame(X_test_reg))
                    racc_kdi.append(1-accuracy_score(y_test, pre_p))
                
                #TPD
                if teacher_full:
                    if TPD:
                        if j == 1:
                            lrplus = priv.TPD(l = 1, T = t)
                            lrplus.fit(X_train, X_train_reg, np.array(y_train), omega, beta, c)
                            pre_p = lrplus.predict(pd.DataFrame(X_test_reg))
                            racc_tpd.append(1-accuracy_score(y_test, pre_p))
                
            l_val_h.append(j)
            t_val_h.append(t)
            
            
            l_val_kdist.append(j)
            t_val_kdist.append(t)
            
            t_val_tpd.append(t)
            
            R_kd.append(np.mean(racc_kd))
            R_kdi.append(np.mean(racc_kdi)) 
            R_tpd.append(np.mean(racc_tpd)) 

        #-------------------------------------------

    cv_kdi  = pd.DataFrame({'l': l_val_h, 'T': t_val_h, 'ACC': R_kdi})
    cv_kd   = pd.DataFrame({'l': l_val_kdist, 'T': t_val_kdist, 'ACC': R_kd})
    cv_tpd   = pd.DataFrame({'T': t_val_tpd, 'ACC': R_tpd})

     

    l_kdi = cv_kdi.sort_values('ACC', ascending = True).reset_index(drop = True)['l'][0]  
    t_kdi= cv_kdi.sort_values('ACC', ascending = True).reset_index(drop = True)['T'][0]  
    l_kd  = cv_kd.sort_values('ACC', ascending = True).reset_index(drop = True)['l'][0]
    t_kd  = cv_kd.sort_values('ACC', ascending = True).reset_index(drop = True)['T'][0]
    
    t_tpd = cv_tpd.sort_values('ACC', ascending = True).reset_index(drop = True)['T'][0]
    
    
    if TPD: 
        return l_kd, l_kdi, t_kd, t_kdi, t_tpd
    else:
        return l_kd, l_kdi, t_kd, t_kdi        




# %%

# experiment 1: noiseless labels as privileged info
def synthetic_01(a,n):
    x  = np.random.randn(n,a.size)
    e  = (np.random.randn(n))[:,np.newaxis]
    xs = np.dot(x,a)[:,np.newaxis]
    y  = ((xs + e) > 0).ravel()
    return (xs,x,y)


# experiment 3: relevant inputs as privileged info
def synthetic_03(a,n):
    x  = np.random.randn(n,a.size)
    xs = np.copy(x)
    xs = xs[:,0:3]
    a  = a[0:3]
    y  = (np.dot(xs,a) > 0).ravel()
    return (xs,x,y)


def data(expe, a, n):
    (x_pri, x_r, yf) = expe(a, n)
    X = pd.concat([pd.DataFrame(x_pri), pd.DataFrame(x_r)], axis = 1).reset_index(drop = True)
    X.columns = range(X.shape[1])
    y = pd.Series(yf*1)
    pi_features = list(np.arange(0, x_pri.shape[1] ))
    return X, y, pi_features



# %%
#Replicación del experimento sintético paper unifying distillation
n_tr = 200
n_test = 10000

lambda_val = [0.3, 0.6, 1]
lambda_val = 1
temperature = [1]
n_reps = 20


acc_lb, acc_tp, acc_tpr = [], [], []
acc_tpd, acc_pfd, acc_pfds, acc_gd, acc_gds =  [], [], [], [], []

    
for rep in range(n_reps):
    if rep % 10 == 0:
        print(rep)
    a   = np.random.randn(50)
    X_train, y_train, pi_features = data(synthetic_03, a, n_tr)
    X_test, y_test, pi_features = data(synthetic_03, a, n_test)

    
                
    
    #PFD
    
    
    X_train_reg = X_train.drop(pi_features, axis = 1)
    X_test_reg = X_test.drop(pi_features, axis = 1)
    
    #LR_UPPER

    lr = LogisticRegression()#C = grid_search.best_params_['C'])
    lr.fit(X_train, y_train)
    
    y_pre = lr.predict_proba(X_test)[:,1]
    pre = lr.predict(X_test)
    acc_tpr.append(1-accuracy_score(y_test, pre))
    pre_train = lr.predict(X_train)
    c = np.array((y_train == pre_train)*1)


    #PRI
    omega = lr.coef_[0]
    beta = lr.intercept_
    
  


    #l_tpd, l_tpdi, t_kd, t_kdi = regularization(X_train, y_train, pi_features, lambda_val, temperature, teacher_full = True)
    l_tpd = 1
    
    #----------------------------------------------------
    
    #TPD
    tpd = priv.TPDGD(l = 1, T = 1)
    tpd.fit(X_train, X_train_reg, np.array(y_train), omega, beta, c)
    pre_p = tpd.predict(pd.DataFrame(X_test_reg))
    acc_tpd.append(1-accuracy_score(y_test, pre_p))
    

    #PFD
    PFD = priv.GD(l = l_tpd, T = 1)
    PFD.fit(X_train, X_train_reg, np.array(y_train), omega, beta)
    pre_p = PFD.predict(pd.DataFrame(X_test_reg))
    acc_pfd.append(1-accuracy_score(y_test, pre_p))
    
    #PFD INV
    #PFDsw = priv.GDsw(l = l_tpdi, T = 1)
    #PFDsw.fit(X_train, X_train_reg, np.array(y_train), omega, beta)
    #pre_p = PFDsw.predict(pd.DataFrame(X_test_reg))
    #acc_pfds.append(1-accuracy_score(y_test, pre_p))
    
    
    
    #----------------------------------------------------
    X_train = X_train[pi_features]
    X_test = X_test[pi_features]
    
    lr = LogisticRegression()#C = grid_search.best_params_['C'])
    lr.fit(X_train, y_train)
    
    y_pre = lr.predict_proba(X_test)[:,1]
    pre = lr.predict(X_test)
    acc_tp.append(1-accuracy_score(y_test, pre))
    pre_train = lr.predict(X_train)
    c = np.array((y_train == pre_train)*1)


    #PRI
    omega = lr.coef_[0]
    beta = lr.intercept_
    #----------------------------------------------------
    
    #l_gd, l_gdi, t_kd, t_kdi = regularization(X_train, y_train, pi_features, lambda_val, temperature, teacher_full = False)
    l_gd = 1

    
    #GD
    GenD = priv.GD(l = l_gd, T = 1)
    GenD.fit(X_train, X_train_reg, np.array(y_train), omega, beta)
    pre_p = GenD.predict(pd.DataFrame(X_test_reg))
    acc_gd.append(1-accuracy_score(y_test, pre_p))
    
    #GD INV
    #GenDsw = priv.GDsw(l = l_gdi, T = 1)
    #GenDsw.fit(X_train, X_train_reg, np.array(y_train), omega, beta)
    #pre_p = GenDsw.predict(pd.DataFrame(X_test_reg))
    #acc_gds.append(1-accuracy_score(y_test, pre_p))
    
    


    #LR_BASE
    
    lrb = LogisticRegression()#C = grid_search.best_params_['C'])
    lrb.fit(X_train_reg, y_train)
    pre = lrb.predict_proba(X_test_reg)[:,1]
    preb = lrb.predict(X_test_reg)
    acc_lb.append(1-accuracy_score(y_test, preb))
    



off_lr = {'tpr': np.round(np.mean(acc_tpr),3),
          'tp': np.round(np.mean(acc_tp),3),
          'base': np.round(np.mean(acc_lb),3),
            'stdtpr': np.round(np.std(acc_tpr),3),
            'stdtp': np.round(np.std(acc_tp),3),
            'stdbase': np.round(np.std(acc_lb),3),

           'TPD': np.round(np.mean(acc_tpd),3),
           'PFD': np.round(np.mean(acc_pfd),3),
           'PFDsw': np.round(np.mean(acc_pfds),3),
           'GD': np.round(np.mean(acc_gd),3),
           'GDsw': np.round(np.mean(acc_gds),3),
           'stdTPD': np.round(np.std(acc_tpd),3),
           'stdPFD': np.round(np.std(acc_pfd),3),
           'stdPFDsw': np.round(np.std(acc_pfds),3),
           'stdGD': np.round(np.std(acc_gd),3),
           'stdGDsw': np.round(np.std(acc_gds),3)
           } 


d1= pd.DataFrame(off_lr, index = [0])
#d2= pd.DataFrame(off_svm, index = [0])



def LUPI_gain(ub, lb, x):
    return ((x - lb) / (ub - lb) )*100


h_tpr = d1.drop(['tpr', 'tp', 'base', 'stdTPD', 'stdPFD', 'stdPFDsw',  'stdtpr', 'stdtp', 'stdbase', 'stdGD', 'stdGDsw', 'GD', 'GDsw'], axis = 1)
gain_tpr = LUPI_gain(d1['tpr'].iloc[0], d1['base'].iloc[0], h_tpr)
h_tp = d1.drop(['tpr', 'tp', 'base', 'stdTPD', 'stdPFD', 'stdPFDsw',  'stdtpr', 'stdtp', 'stdbase', 'stdGD', 'stdGDsw', 'TPD', 'PFD', 'PFDsw'], axis = 1)
gain_tp = LUPI_gain(d1['tp'].iloc[0], d1['base'].iloc[0], h_tp)

gain = pd.concat([gain_tp, gain_tpr], axis = 1)
df_lr = pd.concat([d1, gain]).reset_index(drop = True).T
print(df_lr)
df_lr.to_csv('results/22_05_2024/pricat_regu/s03.csv')




