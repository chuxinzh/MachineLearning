# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 22:37:43 2020

@author: 17917
"""

import numpy as np
dt = np.genfromtxt('classification.txt',delimiter = ',')
lin_dt = np.genfromtxt('linear-regression.txt',delimiter = ',')

####PERCEPTRON#####

def compare(X,W,y):
    scores = np.dot(X,W)
    y_pred = np.ones((scores.shape[0],1))
    loc_neg = np.where(scores<0)[0]
    y_pred[loc_neg] = -1
    loc_wrong =np.where(y_pred!=y)[0]
    return loc_wrong

def update(X,W,y):
    loc_wrong = compare(X,W,y)
    W += y[loc_wrong][0]*X[loc_wrong,:][0].reshape(4,1)*0.01
    return W

def perception(X,W,y):
    iternum = 0
    while len(compare(X,W,y))>0:
        iternum+=1
        W=update(X,W,y)
        accuracy = 1 - len(compare(X,W,y))/len(X)
        #print("Accuracy: {}".format(len(compare(X,W,y))))
    print(iternum)
    return W,accuracy

X = dt[:,:3]
y = dt[:,3]
X = np.c_[X,np.ones((X.shape[0],1))]
y = y.reshape(2000,1)
W = np.random.rand(4)
W[3] = 0
print(W)
W=W.reshape(4,1)
W,accuracy = perception(X,W,y)
W=W.reshape(1,4)
print('Perceptron Weights: ', W)
print('Perceptron Accuracy: ', accuracy)


###POCKET#####

def pocket(X,W,y):
    for _ in range(7000):
        W=update(X,W,y)
        accuracy = 1 - len(compare(X,W,y))/len(X)
        #print("Accuracy: {}".format(len(compare(X,W,y))))
    return W,accuracy

X_po = dt[:,:3]
y_po = dt[:,4]
X_po = np.c_[X_po,np.ones((X_po.shape[0],1))]
y_po = y_po.reshape(2000,1)
W_po = np.random.rand(4)
W_po[3] = 0
W_po=W_po.reshape(4,1)
W_po,accuracy_po = pocket(X_po,W_po,y_po)
W_po=W_po.reshape(1,4)
print('Pocket Weights: ', W_po)
print('Pocket Accuracy: ', accuracy_po)

######LOGISTIC REGRESSION####
def gradient(X,W,y):
    W=W.reshape(4,1)
    N,dim = np.shape(X)
    score = np.dot(X,W)
    sumAll = np.zeros(dim)
    for i in range(len(X)):
        sumfrac = 1/(1 + np.exp(score[i]*y[i]))
        sumN = y[i]*X_log[i]*sumfrac
        sumAll += sumN
    grad = sumAll * (-1/N)
    return grad

def logisitc(X,W,y):
    for i in range(7000):
        grad = gradient(X,W,y)
        W_new = W - 0.01*grad
        W = W_new
    return W

def predict(X,W):
    sig = np.dot(X,W)
    prob = 1+np.exp(sig)
    prob = np.exp(sig)/prob
    predict = np.zeros(np.shape(prob))
    for i in range(len(prob)):
        if prob[i] < 0.5:
            predict[i] = -1
        else:
            predict[i] = 1
    return prob,predict  

X_log = dt[:,:3]
y_log = dt[:,4]
X_log = np.c_[X_log,np.ones((X_log.shape[0],1))]
y_log = y_log.reshape(2000,1)
W_log = np.random.rand(4)
W_log[3] = 0

W_log = logisitc(X_log,W_log,y_log)

W_log = W_log.reshape(4,1)
prob,y_pred = predict(X_log,W_log)
loc_wrong =np.where(y_pred!=y_log)[0]
accuracy_log = 1 - len(loc_wrong)/len(X_log)
print('LOGISTIC Weights: ', W_log.reshape(1,4))
print('LOGISTIC Accuracy: ', accuracy_log)


####LINEAR REGRESSION####
X_lin = lin_dt[:,:2]
y_lin = lin_dt[:,2]
X_lin = np.c_[X_lin,np.ones((X_lin.shape[0],1))]
X_lin = X_lin.T
y_lin = y_lin.reshape(3000,1)

Ddot = np.dot(X_lin,X_lin.T)
Ddiag = np.linalg.inv(Ddot)
W_lin = np.dot(np.dot(Ddiag,X_lin),y_lin)
print('LINEAR REGRESSION Weights: ', W_lin.reshape(1,3))




from sklearn.linear_model import Perceptron
###Perceptron###
X = dt[:,:3]
y = dt[:,3]
clf = Perceptron(tol=None,eta0=0.01)#,max_iter=7000#,class_weight = 'balanced'
clf.fit(X,y)
print(clf.score(X,y))
print(clf.coef_)
print(clf.intercept_)
#clf.n_iter_

###Pocket###
X_po = dt[:,:3]
y_po = dt[:,4]
clf = Perceptron(tol=None,eta0=0.01,max_iter=7000)
clf.fit(X_po,y_po)
print(clf.score(X_po,y_po))
print(clf.coef_)
print(clf.intercept_)
#clf.n_iter_

####LOGISTIC###
from sklearn.linear_model import LogisticRegression
X_log = dt[:,:3]
y_log = dt[:,4]
clf = LogisticRegression(max_iter=7000,C=0.01)
clf.fit(X_log, y_log)
print(clf.score(X_log,y_log))
print(clf.coef_)
print(clf.intercept_)

###LINEAR###
from sklearn.linear_model import LinearRegression
X_lin = lin_dt[:,:2]
y_lin = lin_dt[:,2]
reg = LinearRegression().fit(X_lin, y_lin)
print(reg.coef_)
print(reg.intercept_)
