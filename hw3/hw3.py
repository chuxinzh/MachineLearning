# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 22:10:55 2020

@author: Chuxin Zhang
"""

import numpy as np
import matplotlib.pyplot as plt
import random as rd
import pandas as pd

####PCA####

def minusMean(dt):
    mean = np.mean(dt,axis=0)
    newdt = dt-mean
    return newdt,mean

def main_pca(dt,n):
    newdt,mean = minusMean(dt)
    cov = np.cov(newdt,rowvar=0)
    eigval,eigvect = np.linalg.eig(cov)
    eigvalsort = np.argsort(-eigval)
    selectvec = np.matrix(eigvect.T[eigvalsort[:n]]).T
    fin = newdt * selectvec
    return selectvec,fin

pca_dt = np.genfromtxt('pca-data.txt',delimiter = '	')
pca_vec,pca_fin = main_pca(pca_dt,2)
print('#######PCA OUTPUT#######')
print('PCA direction: ')
print(pca_vec)
print('PCA 2D dataset sample: ')
print(pca_fin[0:2])

###FASTMAP###

def findDist(a,b,count):
    if a == b:
        d = 0
    else:
        if count == 1:
            pointlst = [a,b]
            pointlst.sort()
            d = fm_dt[(fm_dt['a'] == pointlst[0]) & (fm_dt['b'] == pointlst[1])].iloc[0,2]
        else:
            d = (pow(findDist(a,b,count-1),2)-(pow(X[a][count-1]-X[b][count-1],2))) ** 0.5
    return d

def findlocalFar(count):
    k = len(fm_wd)
    a = rd.randint(1,k)
    globalmax = -1
    far_a = 0
    far_b = 0
    localmax = 0
    while True:
        localmax = 0
        for i in range(1,k+1):
            d = findDist(a,i,count)
            if d > localmax:
                localmax = d
                far_a = a
                far_b = i
        if localmax > globalmax:
            globalmax = localmax
            a = far_b
        elif localmax <= globalmax:
            break
    pointlst = [far_a,far_b]
    pointlst.sort()
    far_a = pointlst[0]
    far_b = pointlst[1]
    return far_a,far_b

def FastMap(k):
    global count
    global X
    if (k<=0):
        return
    else:
        count+=1

    a,b = findlocalFar(count)
    length = len(fm_wd)
    for i in range(1,length+1):
        if i == a:
            X[i][count] = 0
        elif i ==b:
            X[i][count] = findDist(a,b,count)
        else:
            X[i][count]  = (pow(findDist(a,i,count),2) + pow(findDist(a,b,count),2) - pow(findDist(b,i,count),2))/(2 * findDist(a,b,count))
    
    FastMap(k-1)
            
####FAST MAP EXECUTION###
fm_dt = pd.read_table("fastmap-data.txt",sep='\t',names =['a','b','dis'])

with open('fastmap-wordlist.txt') as f:
    fm_wd = [line.rstrip() for line in f]

X = np.zeros([len(fm_wd)+1,2+1],dtype = float)
count = 0
FastMap(2)

###PLOT###
X = pd.DataFrame(X)
X.columns = ['word','x','y']
X = X.drop([0]).reset_index(drop=True)
X['word'] = pd.Series(fm_wd)

print('########FAST MAP#########')
print(X)
ax = X.plot.scatter(x='x', y='y', alpha=0.5)
for i, txt in enumerate(X.word):
    ax.annotate(txt, (X.x.iat[i],X.y.iat[i]))
plt.show()

