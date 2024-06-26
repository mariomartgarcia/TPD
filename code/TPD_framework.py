import scipy.optimize as so
import numpy as np
from sklearn.metrics import log_loss
import pandas as pd

    


class TPD():
    #l número de instancias
    #n número de variables involucradas
    def __init__(self, l = 1, T = 1):
        if l < 0:
            raise Exception("l hyperparameter must be 0 or a real positive number")
        self.l = l
        self.T = T
    
    def fit(self, Xc, Xr, y, omega, beta, c ):
        
        zp = np.matmul(Xc, omega.T) + beta
        self.y = np.ravel(y.reshape((1,-1)))
        self.zp = np.array(zp).reshape((1,-1))
        self.Xr = Xr
        self.c = c
        

        ini = np.random.randn(self.Xr.shape[1] + 1 )*0.01
        result = so.minimize(self.log_loss_inv, ini, method='L-BFGS-B')


        self.w = result['x'][0:-1].reshape(1,-1)
        self.b = result['x'][-1]
    
   
   
    def log_loss_inv(self, w):       
       self.z = np.array(list(np.matmul(w[0:-1], self.Xr.transpose()) + w[-1])).reshape((1,-1))
       y_pred = np.array([[1-i,i] for i in np.ravel(self.sigmoid(self.z))])
       y_upper = np.array([[1-i,i] for i in np.ravel(self.sigmoid(self.zp/self.T)) ])

       d1 = np.mean(self.c * (- np.sum(y_pred*np.log(y_upper + 1e-15), axis = 1)))
       d2 = np.mean((1 - self.c) * (- np.sum(y_pred*np.log(y_upper + 1e-15), axis = 1)))
       
       #print(d1 - self.l * d2)
       return d1 - self.l * d2
   
    
    def sigmoid(self, x):
        z = np.exp(-x)
        return 1 / (1 + z)
    
    def predict(self, x):
        x_dot_weights = np.matmul(self.w, x.transpose()) + self.b
        probabilities = self.sigmoid(x_dot_weights.iloc[0])
        pre = [1 if p > 0.5 else 0 for p in probabilities]
        return  pre
    
    def predict_proba(self, x):
        x_dot_weights = np.matmul(self.w, x.transpose()) + self.b
        probabilities = self.sigmoid(x_dot_weights.iloc[0])
        return  probabilities
    
    def coef_(self):
        return self.w[0]
        
    def intercept_(self):
        return self.b 