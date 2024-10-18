# %%

import numpy as np
import pandas as pd
import lrplusTPD as priv
#from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import StratifiedKFold, TimeSeriesSplit, ShuffleSplit
#from mrmr import mrmr_classif, mrmr_regression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split  
from sklearn.metrics import accuracy_score, roc_auc_score, mean_absolute_error
#from sklearn.linear_model import SGDClassifier
import datasets as dat
#import RVFL_plus as rvfl 
#import KRVFL_plus as krvfl 
#import svmplus as svmp

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

#------------------------------------------------
#NÚMERO FIJO DE VARIABLES PRIVILEGIADAS
#------------------------------------------------

#X, y, pi_features = dat.ionosphere()
#--------------------
#OFFLINE
#--------------------


ran= np.random.randint(1000, size = 30)

name, model, lower, upper, opti, gainer = [], [], [], [], [], []
name2, model2, lower2, upper2, opti2, gainer2 = [], [], [], [], [], []
c_media = []


text  =  ['mu_284', 'phishing', 'obesity', 'diabetes', 'wm', 'magictelescope', 'phoneme', 'mozilla4' ]
dataset = [ dat.mu_284(), dat.phishing(from_csv = True), dat.obesity(from_csv = True), dat.diabetes(), dat.wm() , dat.magictelescope(), dat.phoneme(), dat.mozilla4()]

#text  =  ['phoneme', 'mozilla4']
#dataset = [  dat.phoneme(), dat.mozilla4()]



#text  = [ 'mnist_r', 'fruit', 'mnist_g' ] #, 'mozilla4']
#dataset = [   dat.mnist_r(), dat.fruit(), dat.mnist_g()] 

#text  = [  'mu_284']
#dataset = [   dat.mu_284()]

lambda_val = [0.5]
temperature = [1]

df_lr = pd.DataFrame([])
df_meta = pd.DataFrame([])


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

    #tpr = []
    #tp = []
    #base = []

    #e1, e2, e3, e4 = [], [], [], []
    #e5, e6, e7, e8 = [], [], [], []
    acc_lb, acc_tp, acc_tpr = [], [], []
    acc_tpd, acc_pfd, acc_pfds, acc_gd, acc_gds =  [], [], [], [], []
        
    m_tpd, m_gd, m_pfd = [], [], []
    lenTEST = []

    for k in ran:   
        cv = 5
        dr = skfold(X, pd.Series(list(y)), cv, r = k, name = 'stratified')
        

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
            

            #l_pfd, xxx, t_pfd, xxx = regularization(X_train, y_train, pi_features, lambda_val, temperature, teacher_full = True, sw = False, TPD = False)
            #print(l_pfd, t_pfd)
            #l_pfd, t_pfd, t_tpd = 1, 1, 1
            
            #----------------------------------------------------
            
            #TPD
            tpd = priv.TPD(l = 1, T = 1)
            tpd.fit(X_train, X_train_reg, np.array(y_train), omega, beta, c)
            pre_tpd = tpd.predict(pd.DataFrame(X_test_reg))
            acc_tpd.append(1-accuracy_score(y_test, pre_tpd))

        
            
    
            #PFD
            PFD = priv.GD(l = 0.5, T = 1)
            PFD.fit(X_train, X_train_reg, np.array(y_train), omega, beta)
            pre_pfd = PFD.predict(pd.DataFrame(X_test_reg))
            acc_pfd.append(1-accuracy_score(y_test, pre_pfd))
            
            
            
            #----------------------------------------------------
            X_train = X_train[pi_features]
            X_test = X_test[pi_features]
            
            lr = LogisticRegression()#C = grid_search.best_params_['C'])
            lr.fit(X_train, y_train)
            
            y_pre = lr.predict_proba(X_test)[:,1]
            pre = lr.predict(X_test)
            acc_tp.append(1-accuracy_score(y_test, pre))
            pre_train = lr.predict(X_train)
    
    
            #PRI
            omega = lr.coef_[0]
            beta = lr.intercept_
            #----------------------------------------------------
            
            #l_gd, xxx, t_gd, xxx = regularization(X_train, y_train, pi_features, lambda_val, temperature, teacher_full = False, sw = False, TPD = False)
            #print(l_gd, t_gd)
        
            
            #GD
            GenD = priv.GD(l = 0.5, T = 1)
            GenD.fit(X_train, X_train_reg, np.array(y_train), omega, beta)
            pre_gd = GenD.predict(pd.DataFrame(X_test_reg))
            acc_gd.append(1-accuracy_score(y_test, pre_gd))
            
        
            #LR_BASE
            
            lrb = LogisticRegression()#C = grid_search.best_params_['C'])
            lrb.fit(X_train_reg, y_train)
            pre = lrb.predict_proba(X_test_reg)[:,1]
            preb = lrb.predict(X_test_reg)
            acc_lb.append(1-accuracy_score(y_test, preb))
            
            '''
            c_tpd = (pre_tpd == y_test)*1
            c_pfd = (pre_pfd == y_test)*1
            c_gd = (pre_gd == y_test)*1
            cb = (preb == y_test)*1


            
            def meta(c1, c2):
                dd = pd.DataFrame({'base': c1, 'pri': c2})

                ft = np.round(len(dd[(dd['base'] == 0)  &  (dd['pri'] == 1) ])/len(dd)*100,1)
                tf = np.round(len(dd[(dd['base'] == 1)  &  (dd['pri'] == 0) ])/len(dd)*100,1)
                tt = np.round(len(dd[(dd['base'] == 1)  &  (dd['pri'] == 1) ])/len(dd)*100,1)
                ff = np.round(len(dd[(dd['base'] == 0)  &  (dd['pri'] == 0) ])/len(dd)*100,1)
                return np.array([ft, tf, tt, ff])

            def appen(MET, new):
                if len(MET) == 0:
                    MET = new
                else:
                    MET = np.vstack([MET, new])
                return MET

            m_tpd = appen(m_tpd, meta(cb, c_tpd))
            m_gd = appen(m_gd, meta(cb, c_gd))
            m_pfd = appen(m_pfd, meta(cb, c_pfd))

            lenTEST.append(len(cb))
            '''
            
        
    
    def LUPI_gain(ub, lb, x):
        return ((x - lb) / (ub - lb) )*100
    lg_pfd = np.round(LUPI_gain(np.mean(acc_tpr), np.mean(acc_lb), np.mean(acc_pfd)), 3)
    lg_gd = np.round(LUPI_gain(np.mean(acc_tp), np.mean(acc_lb), np.mean(acc_gd)), 3)
    lg_tpd = np.round(LUPI_gain(np.mean(acc_tpr), np.mean(acc_lb), np.mean(acc_tpd)), 3)

    off_lr = { 'Dataset': t,
              'tpr': np.round(np.mean(acc_tpr),3),
              'tp': np.round(np.mean(acc_tp),3),
              'base': np.round(np.mean(acc_lb),3),
              'stdtpr': np.round(np.std(acc_tpr),3),
              'stdtp': np.round(np.std(acc_tp),3),
              'stdbase': np.round(np.std(acc_lb),3),
               'TPD': np.round(np.mean(acc_tpd),3),
               'PFD': np.round(np.mean(acc_pfd),3),
               'GD': np.round(np.mean(acc_gd),3),
               'stdTPD': np.round(np.std(acc_tpd),3),
               'stdPFD': np.round(np.std(acc_pfd),3),
               'stdGD': np.round(np.std(acc_gd),3),
               'LG_tpd': lg_tpd,
               'LG_pfd': lg_pfd,
               'LG_gd': lg_gd,
               } 

    
    d1= pd.DataFrame(off_lr, index = [0])
    df_lr = pd.concat([df_lr, d1])




    '''
    off_meta = { 'Dataset': t,
              'tpd_01': np.mean(m_tpd, axis = 0)[0],
              'tpd_10': np.mean(m_tpd, axis = 0)[1],
              'tpd_11': np.mean(m_tpd, axis = 0)[2],
              'tpd_00': np.mean(m_tpd, axis = 0)[3],
              'gd_01': np.mean(m_gd, axis = 0)[0],
              'gd_10': np.mean(m_gd, axis = 0)[1],
              'gd_11': np.mean(m_gd, axis = 0)[2],
              'gd_00': np.mean(m_gd, axis = 0)[3],
              'pfd_01': np.mean(m_pfd, axis = 0)[0],
              'pfd_10': np.mean(m_pfd, axis = 0)[1],
              'pfd_11': np.mean(m_pfd, axis = 0)[2],
              'pfd_00': np.mean(m_pfd, axis = 0)[3],
              'lenTest': np.round(np.mean(lenTEST), 3)
               } 

    
    dm= pd.DataFrame(off_meta, index = [0])
    df_meta = pd.concat([df_meta, dm])
    '''

df_lr.to_csv('0.5_30_iterations.csv')
#df_meta.to_csv('feasibility_meta.csv')
    
     

# %%
'''
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
#df_lr.to_csv('results/22_05_2024/pricat_regu/s03.csv')


'''


