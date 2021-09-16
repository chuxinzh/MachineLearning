# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 11:19:10 2020

@author: 17917
"""

import numpy as np
import cvxopt
import matplotlib.pyplot as plt

dt = np.genfromtxt('linsep.txt',delimiter = ',')

X = dt[:,:2]
y = dt[:,2]

#np.dot(X,X.T)


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
b = 0
for n in range(len(a)):
    b += y_sv[i] - np.dot(w.T,X_sv[i])

from sklearn.svm import SVC
clf = SVC(kernel='linear',tol=1e-5,C=100)
clf.fit(X,y)
clf.support_vectors_
clf.dual_coef_
clf.intercept_

dt_non = np.genfromtxt('nonlinsep.txt',delimiter = ',')
X_non = dt_non[:,:2]
y_non = dt_non[:,2]

def kernel(X):
    X_norm = np.sum(X ** 2, axis = -1)
    K = np.exp(-1 * (X_norm[:,None] + X_norm[None,:] - 2 * np.dot(X, X.T)))
    return K

from sklearn.metrics.pairwise import rbf_kernel
K_skl = rbf_kernel(X_non,X_non,gamma=1)

K_non = kernel(X_non,X_non)

X_norm = np.sum(X_non ** 2, axis = -1)
K_non = np.exp(-1 * (X_norm[:,None] + X_norm[None,:] - 2 * np.dot(X_non, X_non.T)))

#np.exp(-(X_non-X_non.T))

def gaussianKernel(Xi, X, sigma):
    K = np.zeros((Xi.shape[0], X.shape[0]))
    for i, x1 in enumerate(Xi):
        for j, x2 in enumerate(X):
            x1 = x1.ravel()
            x2 = x2.ravel()
            K[i, j] = np.exp(-np.sum(np.square(x1 - x2)) * sigma)
    return K

test = gaussianKernel(X_non,X_non,1)

C = 10
non_samples,non_features = X_non.shape
#K_non = (np.dot(X_non,X_non.T)*1/non_features+1)**2
P = cvxopt.matrix(np.outer(y_non,y_non)*K_non)
q = cvxopt.matrix(np.ones(non_samples)*-1)
A = cvxopt.matrix(y_non,(1,non_samples))  
b = cvxopt.matrix(0.0)

tmp1 = np.diag(np.ones(non_samples)*-1)
tmp2 = np.identity(non_samples)
G = cvxopt.matrix(np.vstack((tmp1,tmp2)))
#np.vstack((np.eye(non_samples)*-1,np.eye(non_samples)))
tmp1 = np.zeros(non_samples)
tmp2 = np.ones(non_samples) * C
h = cvxopt.matrix(np.hstack((tmp1,tmp2)))
#np.hstack((np.zeros(non_samples),np.ones(non_samples)*C))
solution_non = cvxopt.solvers.qp(P,q,G,h,A,b)
a_non = np.ravel(solution_non['x'])

sv_non = a_non > 1e-5
a_non = a_non[sv_non]
X_sv_non = X_non[sv_non]
y_sv_non = y_non[sv_non]

w_non = np.zeros(non_features)
for i in range(len(a_non)):
    w_non += a_non[i]*y_sv_non[i]*X_sv_non[i]
b_non = 0
for n in range(len(a_non)):
    b_non += y_sv_non[n] - np.dot(w_non.T,X_sv_non[n])   

def project(X,a,sv_y,sv,b):
    y_predict = np.zeros(len(X))
    for i in range(len(X)):
        s = 0
        for a, sv_y, sv in zip(a, sv_y, sv):
            s += a * sv_y * gaussianKernel(X[i],sv,1)
        y_predict[i] = s
    return y_predict + b

y_predict = project(X_test,a_non,y_sv_non,X_sv_non,b_non)

def predict(self, X):
    return np.sign(self.project(X))


test = np.dot(X_non,w_non.T) + b_non

kernel(test)

def predict_sgd(Xi,X,sigma,w,b):
    K=gaussianKernel(Xi,X,sigma)
    return [ 1 if np.dot(w, K[i])+b >= 1 else 0
                       for i in range(len(K))]
    
y_predict = predict_sgd(X_test,X_non,1,w_non,b_non)

X_test = X_non[0:5]

np.shape(gaussianKernel(X_test,X_non,1))

np.dot(w_non,K_non[0])
test = kernel(X_non)

y_predict = np.sign(y_predict)

from sklearn.svm import SVC
clf = SVC(C=1,kernel='poly',degree=2,gamma='auto',coef0=1)
clf.fit(X_non,y_non)
clf.support_vectors_
clf.intercept_
np.abs(clf.dual_coef_)
y_predict = clf.predict(X_non)

from matplotlib.colors import ListedColormap

def versiontuple(v):
    return tuple(map(int, (v.split("."))))


def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):

    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                    alpha=0.8, c=cmap(idx),
                    marker=markers[idx], label=cl)

    # highlight test samples
    if test_idx:
        # plot all samples
        if not versiontuple(np.__version__) >= versiontuple('1.9.0'):
            X_test, y_test = X[list(test_idx), :], y[list(test_idx)]
            #warnings.warn('Please update to NumPy 1.9.0 or newer')
        else:
            X_test, y_test = X[test_idx, :], y[test_idx]

        plt.scatter(X_test[:, 0],
                    X_test[:, 1],
                    c='',
                    alpha=1.0,
                    linewidths=1,
                    marker='o',
                    s=55, label='test set')

# Create a SVC classifier using an RBF kernel
svm = SVC(kernel='rbf', random_state=0, gamma=10, C=100)
# Train the classifier
svm.fit(X_non, y_non)
#svm.dual_coef_

# Visualize the decision boundaries
plot_decision_regions(X_non, y_non, classifier=svm)
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()


right_predict = 0
for b in range(len(y_non)):
    if y_non[b] == y_predict[b]:
        right_predict += 1
        

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
    #x = np.arange(-0.1, 0.1, 0.01)
    #y = (-b * x) - 10 / np.linalg.norm(weights)
    #ax.plot(x, y)
    #plt.xlabel('X1'); plt.ylabel('X2')
    #plt.show()



plotFeature(X_non,y_non)


import pylab as pl
def plot_margin(X,y, w,b):
        def f(x, w, b, c=0):
            # given x, return y such that [x,y] in on the line
            # w.x + b = c
            return (-w[0] * x - b + c) / w[1]

        #pl.plot(X1_train[:,0], X1_train[:,1], "ro")
        #pl.plot(X2_train[:,0], X2_train[:,1], "bo")
        #pl.scatter(clf.sv[:,0], clf.sv[:,1], s=100, c="g")

        plotFeature(X,y)
        # w.x + b = 0
        a0 = -0.2; a1 = f(a0, w, b)
        b0 = 0.6; b1 = f(b0, w, b)
        pl.plot([a0,b0], [a1,b1], "k")

        # w.x + b = 1
#        a0 = -0.2; a1 = f(a0, w, b, 1)
#        b0 = 0.6; b1 = f(b0, w, b, 1)
#        pl.plot([a0,b0], [a1,b1], "k--")

        # w.x + b = -1
#        a0 = -0.2; a1 = f(a0, w, b, -1)
#        b0 = 0.6; b1 = f(b0, w, b, -1)
#        pl.plot([a0,b0], [a1,b1], "k--")

        pl.axis("tight")
        pl.show()


plot_margin(X_non,y_non,w_non,b_non)




