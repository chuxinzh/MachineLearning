# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 21:35:28 2020

@author: 17917
"""

import numpy as np
import random as rd

dt = np.genfromtxt('clusters.txt',delimiter = ',')

#####K-mean#####

def rancentroid(dt,k):
    return rd.sample(list(dt),k)

def minDis(dt,centroid):
    k = len(centroid)
    clusterDict = {}
    for item in dt:
        vec1 = item
        flag = 1
        minDis = float('inf')
        for i in range(k):
            vec2 = centroid[i]
            dist = np.sqrt(np.sum(np.square(vec1 - vec2)))
            if dist < minDis:
                minDis = dist
                flag = i
        if flag not in clusterDict.keys():
            clusterDict[flag] = []
            clusterDict[flag].append(item)
        else:
            clusterDict[flag].append(item)
    return clusterDict

def getReCentroid(clusterDict):
    recentroidlist = []
    for key in clusterDict.keys():
        recentroid = np.mean(clusterDict[key],axis=0)
        recentroidlist.append(recentroid)
    return recentroidlist

def whileCen(oldvec,newvec):
    k = len(newvec)
    Truenum = 0
    for i in range(k):
        if (oldvec[i][0] == newvec[i][0] and oldvec[i][1] == newvec[i][1]):
            Truenum += 1
    return Truenum
        
def k_means(dt,k):
    centroid = oldcen = rancentroid(dt,k)
    clusterDict = minDis(dt,centroid)
    newcen = getReCentroid(clusterDict)
    time = 1
    Truenum = whileCen(oldcen,newcen)
    while Truenum != k:
        oldcen = newcen
        clusterDict = minDis(dt,newcen)
        newcen = getReCentroid(clusterDict)
        time += 1
        Truenum = whileCen(oldcen,newcen)
    return newcen,clusterDict

print(k_means(dt,3))
 
####Generate kmean for 100 times###

kmeanop = []
for op in range(100):
    kmeanop.append(k_means(dt,3))

kemanop_num = {}
for op in kmeanop:
    opstr = str(op)
    if opstr not in kemanop_num.keys():
        kemanop_num[opstr] = 1
    else:
        kemanop_num[opstr] += 1

#########GMM###############

def getGaussian(x,mean,cov):
    dim = np.shape(cov)[0]
    covdet = np.linalg.det(cov)
    covinv = np.linalg.inv(cov)
    xdiff = x - mean
    expin = -0.5 * np.dot(np.dot(xdiff,covinv),xdiff)
#    N = (1/np.power(2*np.pi,dim/2)) * np.power(abs(covdet),-0.5) * np.exp(expin)
    N = (1/np.power(np.power(2*np.pi,dim)*abs(covdet),0.5)) * np.exp(expin)
    return N

###M-step###

def getMstep(dt,ric):
    k = np.shape(ric)[1]
    mean = [0] * k
    amp = [0] * k
    cov = [0] * k
    length,dim = np.shape(dt)
    for i in range(k):
        amp[i] = np.sum([ric[n][i] for n in range(length)])
        mean[i] = (1.0/amp[i]) * np.sum([ric[n][i] * dt[n] for n in range(length)], axis=0)
#        mean[i] = np.average(dt,axis=0,weights=ric[:,i])
        xdiffs = dt - mean[i]
        cov[i] = (1.0/amp[i])*np.sum([ric[n][i]*xdiffs[n].reshape((dim,1)).dot(xdiffs[n].reshape((1,dim))) for n in range(length)],axis=0)
    return mean,cov,amp

###E-step###

def getEstep(dt,mean,cov,amp):
    k = len(mean)
    length = np.shape(dt)[0]
    ric = np.zeros((length,k))
    for x in range(length):
        numer = [amp[i] * getGaussian(dt[x],mean[i],cov[i]) for i in range(k)]
        denom = np.sum(numer)
        for i in range(k):
            ric[x][i] = numer[i] / denom
    return ric

###main-fucntion###

def covergence(old,new):
    return(np.sum(abs(new - old)))

def GMM(dt,k):
    length,dim = np.shape(dt)
    oldric = np.ones((length,k))
    sk_kmean = KMeans(n_clusters=3, random_state=0).fit(dt)
    mean = sk_kmean.cluster_centers_
    cov = []
    for i in range(k):
        cov.append(np.array([[1,0],
                             [0,1]]))
    oldric = np.ones((length,k))/k
    amp = oldric.sum(axis=0)
#    for a in range(length):
#        oldric[a] = np.random.dirichlet(np.ones(k),size=1)
    newric = getEstep(dt,mean,cov,amp)
    mean,cov,amp = getMstep(dt,newric)
    time = 1
    diff = covergence(oldric,newric)
    while diff > 0.0001:
        oldric = newric
        newric = getEstep(dt,mean,cov,amp)
        mean,cov,amp = getMstep(dt,newric)
        time += 1
        diff = covergence(oldric,newric)
    print('No.',time,'difference: ',diff)
    return mean,cov,amp,newric

for i in amp:
    print(i/150)

mean,cov,amp,ric = GMM(dt,3)
print('mean:',mean)
print('cov:',cov)
print('amp:',amp)

####Generate kmean for 100 times###

means = []
amps = []
covs = []

for op in range(100):
    print(op,'OUTPUT')
    mean,cov,amp,ric = GMM(dt,3)
    means.append(mean)
    amps.append(amp)
    covs.append(cov)

def countfreq(result):
    op_num = {}
    for op in result:
        opstr = str(op)
        if opstr not in op_num.keys():
            op_num[opstr] = 1
        else:
            op_num[opstr] += 1
    return op_num

means_num = countfreq(means)
amps_num = countfreq(amps)
covs_num = countfreq(covs)



############sklearn-kmean########
from sklearn.cluster import KMeans

sk_kmean = KMeans(n_clusters=3, random_state=0).fit(dt)

print(sk_kmean.cluster_centers_)

#########sklearn-GMM#######
from sklearn.mixture import GaussianMixture

sk_gmm = GaussianMixture(n_components=3)
sk_gmm.fit(dt)

print(sk_gmm.means_)
mean
print(sk_gmm.covariances_)
cov
print((sk_gmm.weights_)*len(dt))
amp
gmm = sk_gmm.predict(dt)

for i in range(len(cov)):
    cov[i] = cov[i].tolist()

cov[0]
cov[0].tolist()
mean[0]
#####Gauissian model####
from scipy.stats import multivariate_normal
multivariate_normal.pdf(dt[0], mean=mean[0], cov=cov[0])
getGaussian(dt[0],mean[0],cov[0])

