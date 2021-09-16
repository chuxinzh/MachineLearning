# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 00:20:42 2020
@title: DSCI 552 HW6
@type: Group Project
@group member: Chuxin Zhang, Wei Huang
"""
import numpy as np
import cvxopt

"""
linsep Part
"""
dt = np.genfromtxt('linsep.txt',delimiter = ',')

X = dt[:,:2]
y = dt[:,2]

n_samples,n_features = X.shape
K = np.dot(X,X.T)
P = cvxopt.matrix(np.outer(y,y)*K)
q = cvxopt.matrix(np.ones(n_samples)*-1)
A = cvxopt.matrix(y,(1,n_samples))
b = cvxopt.matrix(0.0)
G = cvxopt.matrix(np.diag(np.ones(n_samples)*-1))
h = cvxopt.matrix(np.zeros(n_samples))

solution = cvxopt.solvers.qp(P,q,G,h,A,b)
a = np.ravel(solution['x'])
sv = a > 1e-5
a = a[sv]
X_sv = X[sv]
y_sv = y[sv]

w = np.zeros(n_features)
for i in range(len(a)):
    w += a[i]*y_sv[i]*X_sv[i]
b_lin = y_sv - np.dot(X_sv, w)
b_lin = b_lin[0]

print("========linsep========")
print('w-value:',w)
print('b-value:',b_lin[0])
print('Support Vector:',X_sv)

"""
non-linsep Part
"""
def _rbf(x, y, gamma):
    return np.exp(-gamma * np.sum((x[..., None, :] - y) ** 2, axis=2))

dt_non = np.genfromtxt('nonlinsep.txt',delimiter = ',')
X_non = dt_non[:,:2]
y_non = dt_non[:,2]


C = 10
non_samples,non_features = X_non.shape
K_non = _rbf(X_non,X_non,1)
P = cvxopt.matrix(np.outer(y_non,y_non)*K_non)
q = cvxopt.matrix(np.ones(non_samples)*-1)
A = cvxopt.matrix(y_non,(1,non_samples))  
b = cvxopt.matrix(0.0)

tmp1 = np.diag(np.ones(non_samples)*-1)
tmp2 = np.identity(non_samples)
G = cvxopt.matrix(np.vstack((tmp1,tmp2)))
tmp1 = np.zeros(non_samples)
tmp2 = np.ones(non_samples) * C
h = cvxopt.matrix(np.hstack((tmp1,tmp2)))
solution_non = cvxopt.solvers.qp(P,q,G,h,A,b)
a_non = np.ravel(solution_non['x'])

sv_non = a_non > 1e-5
a_non = a_non[sv_non]
X_sv_non = X_non[sv_non]
y_sv_non = y_non[sv_non]
b_non = y_sv_non - np.sum(_rbf(X_sv_non, X_sv_non, 1)*a_non*y_sv_non,axis=0)
b_non = b_non[0]

print("========non-linsep========")
print("kernel function: Gaussian Kernel(rbf)")
#print('w-value:',w_non)
print('b-value:',b_non[0])
print('Support Vector:',X_sv_non)

"""
linsep sklearn part
"""
from sklearn.svm import SVC
clf = SVC(kernel='linear',tol=1e-5,C=100)
clf.fit(X,y)
print("sklearn w-value:",clf.coef_)
print("sklearn b-value:",clf.intercept_)
print("sklearn-Support Vector:",clf.support_vectors_)

"""
non-linsep sklearn part
"""
svm = SVC(kernel='rbf', random_state=0, gamma=1, C=10)
svm.fit(X_non, y_non)
print("sklearn b-value:",svm.intercept_)
print("sklearn-Support Vector:",svm.support_vectors_)

"""
linsep Visualization
"""
import matplotlib.pyplot as plt
import pylab as pl
#See data point distribution on the graph
def plotFeature(dataMat, labelMat):
    dataArr = np.array(dataMat)
    n = np.shape(dataArr)[0]
    xcord1 = []; ycord1 = []
    xcord2 = []; ycord2 = []
    for i in range(n):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i, 0])
            ycord1.append(dataArr[i, 1])
        else:
            xcord2.append(dataArr[i, 0])
            ycord2.append(dataArr[i, 1])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')

#Add line
def plot_margin(X,y, w,b):
        def f(x, w, b, c=0):
            return (-w[0] * x - b + c) / w[1]

        plotFeature(X,y)
        # w.x + b = 0
        a0 = -0.2; a1 = f(a0, w, b)
        b0 = 0.6; b1 = f(b0, w, b)
        pl.plot([a0,b0], [a1,b1], "k")
        
        # w.x + b = 1
        a0 = -0.2; a1 = f(a0, w, b, 1)
        b0 = 0.6; b1 = f(b0, w, b, 1)
        pl.plot([a0,b0], [a1,b1], "k--")

        # w.x + b = -1
        a0 = -0.2; a1 = f(a0, w, b, -1)
        b0 = 0.6; b1 = f(b0, w, b, -1)
        pl.plot([a0,b0], [a1,b1], "k--")

        pl.axis("tight")
        pl.show()

plot_margin(X,y,w,b_lin)

"""
non-linsep prediction
"""
from numpy import linalg
def gaussian_kernel(x, y, gamma):
    return np.exp(-linalg.norm(x-y)**2 * gamma)

def predict(X,a_non,y_sv_non,X_sv_non,b):
    y_predict = np.zeros(len(X))
    for i in range(len(X)):
        s = 0
        for a, sv_y, sv in zip(a_non, y_sv_non, X_sv_non):
            s += a * sv_y * gaussian_kernel(X[i], sv,1)
        y_predict[i] = s
    return np.sign(y_predict + b)

y_predict = predict(X_non,a_non,y_sv_non,X_sv_non,b_non)

