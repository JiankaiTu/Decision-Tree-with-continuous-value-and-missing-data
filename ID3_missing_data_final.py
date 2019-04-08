from math import *
import operator
import treePlotter
import random
import numpy as np
# import pandas as pd
import os


def remove_sample(inputData):   #删除含missing data的样本
    Data = inputData[:]
    for sample in Data:
        for x in sample:
            if x == -1:
                Data.remove(sample)
                break
    return Data

def remove_feature(inputData, labels,inputtest):       #删除含missing data的样属性
    rownum = len(inputData)
    colnum = len(inputData[0])
    #print(rownum)
    #print(colnum)
    i = 0
    j = 0
    list = []
    label_list = []
    for i in range(rownum):
        for j in range(colnum):
            if inputData[i][j] == -1:
                list.append(j)
                if labels[j] not in label_list:
                    label_list.append(labels[j])
                break
    inputData = np.delete(inputData, list, axis=1)
    inputData = inputData.tolist()
    inputtest = np.delete(inputtest, list, axis=1)
    inputtest = inputtest.tolist()
    for i in label_list:
        labels.remove(i)
    return inputData,labels,inputtest


"""
函数说明:创建测试数据集
"""


def createDataSet(train_missing_ratio):
    alldata = [[int(x) for x in y.split(',')] for y in open('num_mushrooms.csv').read().rstrip().split('\n')[1:]]
    allnum = len(alldata)
    train_ratio = 0.3  # 训练集数目
    test_ratio = 0.5
    trainnum = int(train_ratio * allnum)
    testnum = int(test_ratio * allnum)
    dataSet = [0 for i in range(trainnum)]
    testSet = [0 for i in range(testnum)]
    for i in range(len(dataSet)):  # 随机选取训练集
        dataSet[i] = alldata[random.randint(0, allnum - 1)]
    for i in range(len(testSet)):
        testSet[i] = alldata[random.randint(0, allnum - 1)]
    labels = ['cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor', 'gill-attachment', 'gill-spacing',
              'gill-size',
              'gill-color', 'stalk-shape', 'stalk-root', 'stalk-surface-above-ring', 'stalk-surface-below-ring',
              'stalk-color-above-ring', 'stalk-color-below-ring', 'veil-type', 'veil-color', 'ring-number', 'ring-type',
              'spore-print-color', 'population', 'habitat']  # 分类属性
    dataSet = random_missing(dataSet,train_missing_ratio)
    num = len(dataSet)
    weights = [1 for i in range(1, num + 1)]
    return dataSet,labels, weights, testSet  # 返回数据集和分类属性和权重


"""
函数说明:计算给定数据集的经验熵(香农熵)
Parameters:
    dataSet - 数据集
Returns:
    shannonEnt - 经验熵(香农熵)
"""


def calcShannonEnt(dataSet, weights):
    numEntires = len(dataSet)  # 返回数据集的行数
    overallWeight = sum(weights)
    labelCounts = {}  # 保存每个标签(Label)出现次数的字典
    flag = 0;
    for featVec in dataSet:  # 对每组特征向量进行统计
        currentLabel = featVec[-1]  # 提取标签(Label)信息
        if currentLabel not in labelCounts.keys():  # 如果标签(Label)没有放入统计次数的字典,添加进去
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += weights[flag]  # Label计权值
        flag = flag + 1

    shannonEnt = 0.0  # 经验熵(香农熵)
    for key in labelCounts:  # 计算香农熵
        prob = float(labelCounts[key]) / overallWeight  # 选择该标签(Label)的概率
        shannonEnt -= prob * log(prob, 2)  # 利用公式计算
    return shannonEnt  # 返回经验熵(香农熵)


"""
函数说明:按照给定特征划分数据集
Parameters:
    dataSet - 待划分的数据集
    axis - 划分数据集的特征
    value - 需要返回的特征的值
"""


def splitDataSet(dataSet, axis, value, weights, r=[], index=0):
    retDataSet = []  # 创建返回的数据集列表
    retweights = []
    flag = 0
    for featVec in dataSet:  # 遍历数据集
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]  # 去掉axis特征
            reducedFeatVec.extend(featVec[axis + 1:])  # 将符合条件的添加到返回的数据集
            retDataSet.append(reducedFeatVec)
            retweights.append(weights[flag])
        if featVec[axis] == -1:
            reducedFeatVec = featVec[:axis]  # 去掉axis特征
            reducedFeatVec.extend(featVec[axis + 1:])  # 将符合条件的添加到返回的数据集
            retDataSet.append(reducedFeatVec)
            retweights.append(weights[flag] * r[index])
        flag += 1

    return retDataSet, retweights  # 返回划分后的数据集


"""
函数说明:选择最优特征
Parameters:
    dataSet - 数据集
Returns:
    bestFeature - 信息增益最大的(最优)特征的索引值
"""


def chooseBestFeatureToSplit(dataSet, weights):
    numFeatures = len(dataSet[0]) - 1  # 特征数量
    bestInfoGain = -100  # 信息增益
    bestFeature = -1  # 最优特征的索引值
    for i in range(numFeatures):  # 遍历所有特征
        # 获取dataSet的第i个所有特征
        nmdataSet, nmweights = selectNoMissingData(dataSet, i, weights)
        rho = len(nmdataSet) * 1.0 / len(dataSet)
        baseEntropy = calcShannonEnt(nmdataSet, nmweights)  # 计算数据集的香农熵

        featList = [example[i] for example in nmdataSet]
        uniqueVals = set(featList)  # 创建set集合{},元素不可重复
        newEntropy = 0.0  # 经验条件熵
        flag = 0
        num = len(uniqueVals)
        r = [1 for i in range(1, num + 1)]
        for value in uniqueVals:  # 计算信息增益
            subDataSet, subweights = splitDataSet(nmdataSet, i, value, nmweights)  # subDataSet划分后的子集
            r[flag] = sum(subweights) / float(sum(nmweights))  # 计算子集的概率
            newEntropy += r[flag] * calcShannonEnt(subDataSet, subweights)  # 根据公式计算经验条件熵
            flag += 1
        infoGain = rho * (baseEntropy - newEntropy)  # 信息增益
        #print("第%d个特征的增益为%.3f" % (i, infoGain))  # 打印每个特征的信息增益
        if (infoGain > bestInfoGain):  # 计算信息增益
            bestInfoGain = infoGain  # 更新信息增益，找到最大的信息增益
            bestFeature = i  # 记录信息增益最大的特征的索引值
            bestr = r
    return bestFeature, bestr  # 返回信息增益最大的特征的索引值


def selectNoMissingData(dataSet, feature, weights):
    noMissingdataSet = dataSet[:]
    nmweights = weights[:]
    flag = 0
    for example in dataSet:
        if example[feature] == -1:
            noMissingdataSet.remove(example)
            nmweights[flag] = 0
        flag += 1

    length = len(nmweights)
    x = 0
    while x < length:
        if nmweights[x] == 0:
            del nmweights[x]
            x -= 1
            length -= 1
        x += 1

    return noMissingdataSet, nmweights


"""
函数说明:统计classList中出现此处最多的元素(类标签)
Parameters:
    classList - 类标签列表
Returns:
    sortedClassCount[0][0] - 出现此处最多的元素(类标签)
"""


def majorityCnt(classList):
    classCount = {}
    for vote in classList:  # 统计classList中每个元素出现的次数
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)  # 根据字典的值降序排序
    return sortedClassCount[0][0]  # 返回classList中出现次数最多的元素


"""
函数说明:递归构建决策树
Parameters:
    dataSet - 训练数据集
    labels - 分类属性标签
    featLabels - 存储选择的最优特征标签
Returns:
    myTree - 决策树
"""


def createTree(dataSet, labels, featLabels, weights):
    classList = [example[-1] for example in dataSet]  # 取分类标签
    if classList.count(classList[0]) == len(classList):  # 如果类别完全相同则停止继续划分
        return classList[0]
    if len(dataSet[0]) == 1:  # 遍历完所有特征时返回出现次数最多的类标签
        return majorityCnt(classList)
    bestFeat, bestr = chooseBestFeatureToSplit(dataSet, weights)  # 选择最优特征
    bestFeatLabel = labels[bestFeat]  # 最优特征的标签
    featLabels.append(bestFeatLabel)
    myTree = {bestFeatLabel: {}}  # 根据最优特征的标签生成树
    del (labels[bestFeat])  # 删除已经使用特征标签
    featValues = [example[bestFeat] for example in dataSet]  # 得到训练集中所有最优特征的属性值
    ## 有缺失的情况下会出现-1，想办法解决
    uniqueVals = set(featValues)  # 去掉重复的属性值
    if -1 in uniqueVals:
        uniqueVals.remove(-1)

    Vals_array = []
    for value in uniqueVals:
        Vals_array.append(value)
    for value in uniqueVals:
        subLabels = labels[:]
        # 递归调用函数createTree(),遍历特征，创建决策树。
        subdataSet, subweights = splitDataSet(dataSet, bestFeat, value, weights, bestr, Vals_array.index(value))
        myTree[bestFeatLabel][value] = createTree(subdataSet, subLabels, featLabels, subweights)
    return myTree


"""
函数说明:使用决策树执行分类
Parameters:
    inputTree - 已经生成的决策树
    featLabels - 存储选择的最优特征标签
    testVec - 测试数据列表，顺序对应最优特征标签
Returns:
    classLabel - 分类结果
"""


def classify(inputTree, featLabels, testVec, Labels):
    firstStr = list(inputTree.keys())[0]  # 树的根节点名称
    secondDict = inputTree[firstStr]  # 根节点的所有子节点
    featIndex = Labels.index(firstStr)  # 根节点特征对应的下标
    classLabel = 'Cannot verify'
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__ == 'dict':
                classLabel = classify(secondDict[key], featLabels, testVec, Labels)
            else:
                classLabel = secondDict[key]
    return classLabel


##

def random_missing(inputData, missing_ratio):
    # DataSet = inputData
    rownum = len(inputData)
    colnum = len(inputData[0])
    missingnum = int(rownum * (colnum - 1) * missing_ratio)
    for i in range(missingnum):
        inputData[random.randint(0, rownum - 1)][random.randint(0, colnum - 2)] = -1
    return inputData

def cal_acc(myTree,featLabels,labels_ori,testSet):
    testfeature = np.delete(testSet, -1, axis=1)  # 测试集
    testlabel = [x[-1] for x in testSet]
    correct_cnt = 0
    for i in range(len(testSet)):
        result_of_class = classify(myTree, featLabels, testfeature[i], labels_ori)
        # print(result_of_class)
        if (result_of_class == testlabel[i]):
            correct_cnt += 1
    accuracy = correct_cnt / len(testSet)
    return accuracy

"""
dataSet：以train_missing_ratio 缺失的原始数据集
labels：带missing数据集标签
weights：带missing数据集weights
testSet: 测试集

dataSet_s：删去含缺失的样本
weights_s 

dataSet_f：删去含缺失的属性
labels_f：删后的属性集
testSet 变

注意：
法三只适用于缺失较少的情况，否则属性集删空，print"Warning: empty feature"
"""
if __name__ == '__main__':
    train_missing_ratio = 0.001        #随机缺失
    dataSet, labels, weights,  testSet = createDataSet(train_missing_ratio)
    labels_ori = labels[:]
    labels_ori_s = labels_ori[:]
    labels_ori_f = labels_ori[:]
    labels_s = labels[:]

    featLabels = []
    myTree = createTree(dataSet, labels, featLabels, weights)
    # treePlotter.createPlot(myTree)
    # print(myTree)
    accuracy1 = cal_acc(myTree, featLabels, labels_ori, testSet)
    print("Accuacy of probability split")
    print(accuracy1)

    dataSet_s = remove_sample(dataSet)  #remove samples
    featLabels_s=[]
    weights_s = [1 for i in range(1, len(dataSet_s) + 1)]
    myTree_s = createTree(dataSet_s, labels_s, featLabels_s, weights_s)
    #print(myTree_s)
    accuracy2 = cal_acc(myTree_s, featLabels_s, labels_ori_s, testSet)
    print("Accuracy of Complete case method")
    print(accuracy2)

    dataSet_f, labels_f,testSet = remove_feature(dataSet, labels_ori_f,testSet)  # 删除含missing data的属性
    if not labels_f:
        print("Warning: empty feature")
    else:
        testfeature = np.delete(testSet, -1, axis=1)  # 新测试集
        testlabel = [x[-1] for x in testSet]
        testnum = len(testSet)
        weights_f = [1 for i in range(1, len(dataSet_f) + 1)]
        labels_ori_ff = labels_f[:]
        featLabels_f = []
        myTree_f = createTree(dataSet_f, labels_f, featLabels_f, weights_f)
        #print(myTree_f)
        accuracy3 = cal_acc(myTree_f, featLabels_f, labels_ori_ff, testSet)
        print("Accuracy of Complete variable method")
        print(accuracy3)




