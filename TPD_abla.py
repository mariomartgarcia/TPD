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
import itertools
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

#------------------------------------------------
#NÚMERO FIJO DE VARIABLES PRIVILEGIADAS
#------------------------------------------------

#X, y, pi_features = dat.ionosphere()
#--------------------
#OFFLINE
#--------------------


ran= np.random.randint(1000, size = 30)


text  =  ['mu_284', 'phishing', 'obesity', 'diabetes', 'wm', 'magictelescope', 'phoneme', 'mozilla4' ]
dataset = [ dat.mu_284(), dat.phishing(from_csv = True), dat.obesity(from_csv = True), dat.diabetes(), dat.wm() , dat.magictelescope(), dat.phoneme(), dat.mozilla4()]

#text  =  ['phoneme', 'mozilla4']
#dataset = [  dat.phoneme(), dat.mozilla4()]



#text  = [ 'mnist_r', 'fruit', 'mnist_g' ] #, 'mozilla4']
#dataset = [   dat.mnist_r(), dat.fruit(), dat.mnist_g()] 

#text  = [ 'phishing']
#dataset = [dat.phishing(from_csv = True)]



df_lr = pd.DataFrame([])
df_tpd = pd.DataFrame([])

temp_val = [0.5, 1, 5, 10, 20]
beta_val = [0, 1, 2, 5, 10, 20]


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

    err_b, err_up, err_up_priv = [], [], []
    err_tpd, err_pfd, err_gd, err_bce, err_pfd1 =  [], [], [], [], []
    failures = []
        

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
            
            #LR_UPPER (PRI)         
            #----------------------------------------------------
            X_train_pri = X_train[pi_features]
            X_test_pri = X_test[pi_features]
            
            lr = LogisticRegression()
            lr.fit(X_train_pri, y_train)
            
            pre = lr.predict(X_test_pri)
            err_up_priv.append(1-accuracy_score(y_test, pre))
            pre_train = lr.predict(X_train_pri)
    
    
            #PRI
            omega_pri = lr.coef_[0]
            beta_pri = lr.intercept_
            #----------------------------------------------------

            #LR_UPPER (PRI + REG)
            #----------------------------------------------------
            lr = LogisticRegression()
            lr.fit(X_train, y_train)
            
            pre = lr.predict(X_test)
            err_up.append(1-accuracy_score(y_test, pre))
            pre_train = lr.predict(X_train)
            c = np.array((y_train == pre_train)*1)
    
    
            #PRI
            omega = lr.coef_[0]
            beta = lr.intercept_
            #----------------------------------------------------

            #LR_BASE
            #----------------------------------------------------
            lrb = LogisticRegression()#C = grid_search.best_params_['C'])
            lrb.fit(X_train_reg, y_train)
            pre = lrb.predict_proba(X_test_reg)[:,1]
            preb = lrb.predict(X_test_reg)
            err_b.append(1-accuracy_score(y_test, preb))
            #----------------------------------------------------
            #----------------------------------------------------
            #----------------------------------------------------

            err_abla1, err_abla2, err_abla3 = [[] for i in range(3)]

            for q in temp_val:
                #GD
                GenD = priv.GD(l = 0.5, T = q)
                GenD.fit(X_train_pri, X_train_reg, np.array(y_train), omega_pri, beta_pri)
                pre_gd = GenD.predict(pd.DataFrame(X_test_reg))
                err_abla1.append(1-accuracy_score(y_test, pre_gd))

                #PFD
                PFD = priv.GD(l = 0.5, T = q)
                PFD.fit(X_train, X_train_reg, np.array(y_train), omega, beta)
                pre_pfd = PFD.predict(pd.DataFrame(X_test_reg))
                err_abla2.append(1-accuracy_score(y_test, pre_pfd))
            

                for b in beta_val:
                    #TPD
                    tpd = priv.TPD(l = b, T = q)
                    tpd.fit(X_train, X_train_reg, np.array(y_train), omega, beta, c)
                    pre_tpd = tpd.predict(pd.DataFrame(X_test_reg))
                    err_abla3.append(1-accuracy_score(y_test, pre_tpd))


            if len(err_tpd) == 0:
                err_gd = err_abla1
                err_pfd = err_abla2
                err_tpd = err_abla3

            else:
                err_pfd = np.vstack([err_pfd, err_abla1])
                err_gd = np.vstack([err_gd, err_abla2])
                err_tpd = np.vstack([err_tpd, err_abla3])
            

            #BCEi
            tpd = priv.TPD_inv(T = 1)
            tpd.fit(X_train, X_train_reg, np.array(y_train), omega, beta, c)
            pre_tpd = tpd.predict(pd.DataFrame(X_test_reg))
            err_bce.append(1-accuracy_score(y_test, pre_tpd))
            failures.append((np.sum(1-c)/len(X_train))*100)


            #PFD
            PFD = priv.GD(l = 1, T = 1)
            PFD.fit(X_train, X_train_reg, np.array(y_train), omega, beta)
            pre_pfd = PFD.predict(pd.DataFrame(X_test_reg))
            err_pfd1.append(1-accuracy_score(y_test, pre_pfd))



            
    
    
    def LUPI_gain(ub, lb, x):
        return ((x - lb) / (ub - lb) )*100


    off = {'name' : t,
           'tp':np.round(np.mean(err_up_priv), 3),
           'tpr':np.round(np.mean(err_up), 3),
           'base':  np.round(np.mean(err_b), 3),
           'std_tp':np.round(np.std(err_up_priv), 3),
           'std_tpr':np.round(np.std(err_up), 3),
           'std_b':  np.round(np.std(err_b), 3),  
           'bceinv':  np.round(np.mean(err_bce), 3),
           'std_bceinv':  np.round(np.std(err_bce), 3),
           'failures':  np.round(np.mean(failures), 2),
           'pf1':  np.round(np.mean(err_pfd1), 3),
           'std_pf1':  np.round(np.std(err_pfd1), 3),

            }   
    
    '''
    'LUPIGD %': np.round(LUPI_gain(np.round(np.mean(err_up_priv), 3),  np.round(np.mean(err_b), 3), np.round(np.mean(err_gd), 3)),1),
    'LUPIPFD %': np.round(LUPI_gain(np.round(np.mean(err_up), 3),  np.round(np.mean(err_b), 3), np.round(np.mean(err_pfd), 3)),1),
    'LUPITPD %': np.round(LUPI_gain(np.round(np.mean(err_up), 3),  np.round(np.mean(err_b), 3), np.round(np.mean(err_tpd), 3)),1),
    '''

    for q, k in enumerate(temp_val):
        off['PFD'+str(k)] =  np.mean(err_pfd, axis = 0)[q]
        off['std_PFD'+str(k)] =  np.std(err_pfd, axis = 0)[q]
        off['GD'+str(k)] =  np.mean(err_gd, axis = 0)[q]
        off['std_GD'+str(k)] =  np.std(err_gd, axis = 0)[q]


    df1 = pd.DataFrame(off, index = [0])
        
    df_lr  = pd.concat([df_lr, df1]).reset_index(drop = True)


    df_TB = pd.DataFrame(list(itertools.product(temp_val, beta_val)), columns = ['T', 'b'])
    df_TB['name'] = t
    df_TB['err_TPD'] = np.mean(err_tpd, axis = 0)
    df_TB['std_TPD'] = np.std(err_tpd, axis = 0)


    df_tpd = pd.concat([df_tpd, df_TB])


df_lr.to_csv('30_temp_lr.csv')
df_tpd.to_csv('30_tpd_beta_temp_lr.csv')
#df_meta.to_csv('feasibility_meta.csv')
#---------------------------------------------------------------------------------------------
# %%

df = pd.read_csv('10_temp_lr.csv', index_col = [0])
df_tpd = pd.read_csv('10_tpd_beta_temp_lr.csv', index_col = [0])



import matplotlib.pyplot as plt



for i in df.name:
    c_T = len(df_tpd['T'].unique())
    beta = df_tpd['b'].unique()
    
    val_b = np.repeat(df['base'][df['name'] == i], len(beta))
    std_b = np.repeat(df['std_b'][df['name'] == i], len(beta))
    val_up = np.repeat(df['tpr'][df['name'] == i], len(beta))
    std_up = np.repeat(df['std_tpr'][df['name'] == i], len(beta))





    # Creando la gráfica con un tamaño de figura más grande
    plt.figure(figsize=(10, 6)) 
    for j in [1]: #df_tpd['T'].unique():
        val_tpd = np.array(df_tpd[(df_tpd['T'] == j) & (df_tpd['name'] == i)]['err_TPD'])
        std_tpd = np.array(df_tpd[(df_tpd['T'] == j) & (df_tpd['name'] == i)]['std_TPD'])
        plt.plot(beta, val_tpd, linestyle='--', marker='o', label='TPD_' + str(j))
        plt.fill_between(beta, val_tpd - std_tpd, val_tpd + std_tpd, alpha=0.2)


    
    #plt.plot(beta, val_gd, linestyle='--', marker='o', label='GD')
    #plt.plot(beta, val_pfd, linestyle='--', marker='o', label='PFD')
    #plt.plot(beta, val_bci, linestyle='--', marker='o', label='BCI')
    plt.plot(beta, val_b, linestyle='--', marker='o', label='B')
    plt.plot(beta, val_up, linestyle='--', marker='o', label='UP')

   
    plt.fill_between(beta, val_b - std_b, val_b + std_b, alpha=0.2)
    plt.fill_between(beta, val_up - std_up, val_up + std_up, alpha=0.2)

    # Añadiendo etiquetas y título con mayor tamaño de fuente
    plt.xlabel('Beta', fontsize=14)
    plt.ylabel('Error Rate', fontsize=14)
    plt.title('Comparison of Methods against Beta | ' + str(i), fontsize=16)

    # Añadiendo la cuadrícula
    plt.grid()

    # Cambiando la ubicación de la leyenda a la parte superior derecha y ajustando el tamaño de fuente
    plt.legend(loc='upper right', fontsize=12)

    # Mostrando la gráfica
    plt.show()




# %%
#Evolution of the temperature

val_t = df_tpd['T'].unique()
df_temp = pd.DataFrame()
df_temp['name'] =df['name']
for i in val_t:
    i = int(i) if i >= 1 else i
    df_temp['GD_'+str(i)] = np.round(df['GD'+str(i)],3)
    df_temp['PFD_'+str(i)] = np.round(df['PFD'+str(i)],3)
    df_temp['TPD_'+str(i)] = np.round(list(df_tpd[(df_tpd['T'] == i) & (df_tpd['b'] == 1)]['err_TPD']),3)
    df_temp['std_GD_'+str(i)] = np.round(df['std_GD'+str(i)],3)
    df_temp['std_PFD_'+str(i)] = np.round(df['std_PFD'+str(i)],3)
    df_temp['std_TPD_'+str(i)] = np.round(list(df_tpd[(df_tpd['T'] == i) & (df_tpd['b'] == 1)]['std_TPD']),3)



for j in df.name:
    gd, pfd, tpd = [], [], []
    sgd, spfd, stpd = [], [], []
    val_b = np.repeat(df['base'][df['name'] == j], len(val_t))

    plt.figure(figsize=(10, 6)) 
    for i in val_t:
        i = int(i) if i >= 1 else i
        tpd.append(df_temp[df_temp['name'] == j]['TPD_'+str(i)].item())
        gd.append(df_temp[df_temp['name'] == j]['GD_'+str(i)].item())
        pfd.append(df_temp[df_temp['name'] == j]['PFD_'+str(i)].item())

        stpd.append(df_temp[df_temp['name'] == j]['std_TPD_'+str(i)].item())
        sgd.append(df_temp[df_temp['name'] == j]['std_GD_'+str(i)].item())
        spfd.append(df_temp[df_temp['name'] == j]['std_PFD_'+str(i)].item())

    tpd = np.array(tpd)
    gd = np.array(gd)
    pfd = np.array(pfd)

    stpd = np.array(stpd)
    sgd = np.array(sgd)
    spfd = np.array(spfd)



    plt.fill_between(val_t, tpd - stpd, tpd+ stpd, alpha=0.2)
    plt.fill_between(val_t, gd - sgd, gd + sgd, alpha=0.2)
    plt.fill_between(val_t, pfd - spfd, pfd + spfd, alpha=0.2)

    plt.plot(val_t, tpd, linestyle='--', marker='o', label='TPD')
    plt.plot(val_t, gd, linestyle='--', marker='o', label='GD')
    plt.plot(val_t, pfd, linestyle='--', marker='o', label='PFD')

    plt.plot(val_t, val_b, linestyle='--', marker='o', label='B')

    # Añadiendo etiquetas y título con mayor tamaño de fuente
    plt.xlabel('Temperature', fontsize=14)
    plt.ylabel('Error Rate', fontsize=14)
    plt.title('Temperature evolution | ' + str(j), fontsize=16)

    # Añadiendo la cuadrícula
    plt.grid()

    # Cambiando la ubicación de la leyenda a la parte superior derecha y ajustando el tamaño de fuente
    plt.legend(loc='upper right', fontsize=12)

    # Mostrando la gráfica
    plt.show()


# %%

tpd, pfd, bcei, failure, names = [[] for i in range(5)]

for i in df['name']:
    tpd.append(df_tpd[(df_tpd['T'] == 1) & (df_tpd['b'] == 1) & (df_tpd['name'] == i)]['err_TPD'].item())
    pfd.append(df[df['name'] == i]['PFD1'].item())
    bcei.append(df[df['name'] == i]['bceinv'].item())
    failure.append(df[df['name'] == i]['failures'].item())
    names.append(i)


off ={'name': names,
      'failures': failure,
      'bcei': bcei,
      'pfd': pfd,
      'tpd': tpd}

df_bcei = pd.DataFrame(off)


# %%
'''