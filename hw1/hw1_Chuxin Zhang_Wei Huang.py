# -*- coding: utf-8 -*-
"""
Created on Thu Sep 10 22:08:04 2020
@title: DSCI 552 HW1 Decision Tree
@type: Group Project
@group member: Chuxin Zhang, Wei Huang
"""

from math import log
import json
import matplotlib.pyplot as plt

dataset = [['High','Expensive','Loud','Talpiot','No','No','No'],
           ['High','Expensive','Loud','City-Center','Yes','No','Yes'],
           ['Moderate','Normal','Quiet','City-Center','No','Yes','Yes'],
           ['Moderate','Expensive','Quiet','German-Colony','No','No','No'],
           ['Moderate','Expensive','Quiet','German-Colony','Yes','Yes','Yes'],
           ['Moderate','Normal','Quiet','Ein-Karem','No','No','Yes'],
           ['Low','Normal','Quiet','Ein-Karem','No','No','No'],
           ['Moderate','Cheap','Loud','Mahane-Yehuda','No','No','Yes'],
           ['High','Expensive','Loud','City-Center','Yes','Yes','Yes'],
           ['Low','Cheap','Quiet','City-Center','No','No','No'],
           ['Moderate','Cheap','Loud','Talpiot','No','Yes','No'],
           ['Low','Cheap','Quiet','Talpiot','Yes','Yes','No'],
           ['Moderate','Expensive','Quiet','Mahane-Yehuda','No','Yes','Yes'],
           ['High','Normal','Loud','Mahane-Yehuda','Yes','Yes','Yes'],
           ['Moderate','Normal','Loud','Ein-Karem','No','Yes','Yes'],
           ['High','Normal','Quiet','German-Colony','No','No','No'],
           ['High','Cheap','Loud','City-Center','No','Yes','Yes'],
           ['Low','Normal','Quiet','City-Center','No','No','No'],
           ['Low','Expensive','Loud','Mahane-Yehuda','No','No','No'],
           ['Moderate','Normal','Quiet','Talpiot','No','No','Yes'],
           ['Low','Normal','Quiet','City-Center','No','No','Yes'],
           ['Low','Cheap','Loud','Ein-Karem','Yes','Yes','Yes']]
    
labels = ['Occupied', 'Price', 'Music', 'Location', 'VIP', 'Favorite Beer', 'Enjoy']

def calcEntropy(dataset):
    numdata = len(dataset)
    labelCounts = {}
    for row in dataset:
        currentLabel = row[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    entropy = 0
    for i in labelCounts:
        pro = labelCounts[i]/numdata
        entropy += pro * log(1/pro,2)
    return entropy

index = 1
value = 'Normal'

def subDataset(dataset,index,value):
    subData = []
    for subrow in dataset:
        if subrow[index] == value:
            reduceSub = subrow[:index]
            reduceSub.extend(subrow[index+1:])
            subData.append(reduceSub)
    return subData

def chooseBest(dataset):
    numFeature = len(dataset[0])-1
    baseEnt = calcEntropy(dataset)
    bestGain = 0
    bestFeature = -1
    for i in range(numFeature):
        featurelist = []
        for val in dataset:
            featurelist.append(val[i])
        uniqueVals = set(featurelist) 
        newEnt = 0
        for val in uniqueVals:
            subData = subDataset(dataset,i,val)
            prob = len(subData)/len(dataset)
            newEnt += prob*calcEntropy(subData)
        infoGain = baseEnt - newEnt
        if (infoGain > bestGain):
            bestGain = infoGain
            bestFeature = i
    return bestFeature

def createTree(dataset,labels):
    classlist = [example[-1] for example in dataset]
    if classlist.count(classlist[0]) == len(classlist):
        return classlist[0]
    if len(dataset[0]) == 1:
        return max(classlist,key=classlist.count)
    bestFeat = chooseBest(dataset)
    if bestFeat == -1:
        return max(classlist,key=classlist.count)
    bestFeatLabel = labels[bestFeat]
    
    myTree = {bestFeatLabel:{}}
    featval = [i[bestFeat] for i in dataset]
    uniqueVals = set(featval)
    
    for val in uniqueVals:
        sublabel = labels[:bestFeat]
        sublabel.extend(labels[bestFeat+1:])
        myTree[bestFeatLabel][val] = createTree(subDataset(dataset,bestFeat,val),sublabel)
    return myTree

decisionNode = dict(boxstyle="sawtooth",fc="0.8")
leafNode = dict(boxstyle="round4")
arrow = dict(arrowstyle="<-")

def getLeafs(tree):
    leafs = 0
    firstStr = list(tree.keys())[0]
    secondDict = tree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':
            leafs += getLeafs(secondDict[key])
        else:
            leafs += 1
    return leafs

def getDepth(tree):
    depth = 0
    firstStr = list(tree.keys())[0]
    secondDict = tree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':
            currentDepth = 1 + getDepth(secondDict[key])
        else:
            currentDepth = 1
        if currentDepth > depth:
            depth = currentDepth
    return depth

def createPlot(tree):
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)
    plotTree.totalW = getLeafs(tree)
    plotTree.totalD = getDepth(tree)
    plotTree.xOff = -0.5 / plotTree.totalW
    plotTree.yOff = 1.0
    plotTree(tree, (0.5, 1.0), '')
    plt.show()

def plotMidText(cntrPt, parentPt, txtString):
    xMid = (parentPt[0] - cntrPt[0]) / 2.0 + cntrPt[0]
    yMid = (parentPt[1] - cntrPt[1]) / 2.0 + cntrPt[1]
    createPlot.ax1.text(xMid, yMid, txtString, va="center", ha="center", rotation=30)

def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    createPlot.ax1.annotate(nodeTxt, xy=parentPt, xycoords='axes fraction',
                            xytext=centerPt, textcoords='axes fraction',
                            va="center", ha="center", bbox=nodeType, arrowprops=arrow)

def plotTree(tree, parentPt, nodeTxt): 
    numLeafs = getLeafs(tree)
    depth = getDepth(tree)
    firstStr = list(tree.keys())[0]
    cntrPt = (plotTree.xOff + (1.0 + float(numLeafs)) / 2.0 / plotTree.totalW, plotTree.yOff)
    plotMidText(cntrPt, parentPt, nodeTxt)
    plotNode(firstStr, cntrPt, parentPt, decisionNode)
    secondDict = tree[firstStr]
    plotTree.yOff = plotTree.yOff - 1.0 / plotTree.totalD
    for key in secondDict.keys():
        if type(secondDict[
                    key]).__name__ == 'dict':
            plotTree(secondDict[key], cntrPt, str(key))
        else:
            plotTree.xOff = plotTree.xOff + 1.0 / plotTree.totalW
            plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), cntrPt, leafNode)
            plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))
    plotTree.yOff = plotTree.yOff + 1.0 / plotTree.totalD


myTree = createTree(dataset,labels)
print(json.dumps(myTree,indent=4, ensure_ascii=False))
createPlot(myTree)


def predict(myTree,labels,testdata):
    root = list(myTree.keys())[0]
    subtree = myTree[root]
    featindex = labels.index(root)
    for key in subtree.keys():
        if testdata[featindex] == key:
            if type(subtree[key]).__name__ == 'dict':
                outcome = predict(subtree[key],labels,testdata)
            else:
                outcome = subtree[key]
    return outcome

testdata = ['Moderate','Cheap','Loud','City-Center','No','No']

print(predict(myTree,labels,testdata))




        