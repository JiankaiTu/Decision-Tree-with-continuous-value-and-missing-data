import math
import random

class C45:
    """Creates a decision tree with C4.5 algorithm"""

    def __init__(self, pathToData, pathToNames):
        self.filePathToData = pathToData
        self.filePathToNames = pathToNames
        self.data = []
        self.repeat = True
        self.traindata = []
        self.classes = []
        self.numAttributes = -1
        self.attrValues = {}
        self.attributes = []
        self.tree = None
        self.prunningacc = 0
        self.testdata = []
        self.pruning_threshold = 20
        self.segment = False
        self.K = 0
        self.child_label = None

    def setprunning(self,threshold):
        self.pruning_threshold = threshold

    def bool_segment(self, option, parameter_K):
        self.segment = option
        self.K = parameter_K

    def fetchData(self):
        with open(self.filePathToNames, "r") as file:
            # add classes
            classes = file.readline()
            self.classes = [x.strip() for x in classes.split(",")]
            # add attributes
            for line in file:
                [attribute, values] = [x.strip() for x in line.split(":")]
                values = [x.strip() for x in values.split(",")]
                self.attrValues[attribute] = values
        self.numAttributes = len(self.attrValues.keys())
        self.attributes = list(self.attrValues.keys())
        with open(self.filePathToData, "r") as file:
            for line in file:
                row = [x.strip() for x in line.split(",")]
                if row != [] or row != [""]:
                    self.data.append(row)

    def preprocessData(self):
        for index, row in enumerate(self.data):
            for attr_index in range(self.numAttributes):
                # if continuouse
                if (not self.isAttrDiscrete(self.attributes[attr_index])):
                    self.data[index][attr_index] = float(self.data[index][attr_index])

        for example in self.data:
            example.append(1)

        random.shuffle(self.data)
        lendata = len(self.data)
        self.traindata = self.data[0: math.floor(0.9*lendata)]
        self.testdata = self.data[math.floor(0.9*lendata):lendata]
    def convert_dic(self):
        tree_dict = {}
        self.tree_to_dict(self.tree, tree_dict)
        return tree_dict

    def tree_to_dict(self, tree_root, tree_dict):
        tree_dict[tree_root.label] = {}
        for index, child in enumerate(tree_root.children):
            if tree_root.threshold is None:
                tree_dict[tree_root.label][index + 1] = {}
                if not child.isLeaf:
                    self.tree_to_dict(child, tree_dict[tree_root.label][index + 1])
                else:
                    tree_dict[tree_root.label][index + 1] = {child.label}
            else:
                if index:
                    label = ">" + str(round(tree_root.threshold, 2))
                else:
                    label = "<=" + str(round(tree_root.threshold, 2))
                tree_dict[tree_root.label][label] = {}
                if not child.isLeaf:
                    self.tree_to_dict(child, tree_dict[tree_root.label][label])
                else:
                    tree_dict[tree_root.label][label] = {child.label}

    def printTree(self):
        self.printNode(self.tree)


    def printNode(self, node, indent=""):
        if not node.isLeaf:
            if node.threshold is None:
                # discrete
                for index, child in enumerate(node.children):
                    if child.isLeaf:
                        print(indent + node.label + " = " + self.attrValues[node.label][index] + " : " + child.label)
                    else:
                        print(indent + node.label + " = " + self.attrValues[node.label][index] + " : ")
                        self.printNode(child, indent + "    ")
            else:
                # numerical
                leftChild = node.children[0]
                rightChild = node.children[1]
                if leftChild.isLeaf:
                    print(indent + node.label + " <= " + str(node.threshold) + " : " + leftChild.label)
                else:
                    print(indent + node.label + " <= " + str(node.threshold) + " : ")
                    self.printNode(leftChild, indent + "    ")

                if rightChild.isLeaf:
                    print(indent + node.label + " > " + str(node.threshold) + " : " + rightChild.label)
                else:
                    print(indent + node.label + " > " + str(node.threshold) + " : ")
                    self.printNode(rightChild, indent + "   ")

    def generateTree(self, repeat):
        self.repeat = repeat
        self.tree = self.recursiveGenerateTree(self.traindata, self.attributes,self.pruning_threshold, self.repeat)
     #   self.treenow = self.Retran(self.tree)

   # def Retran(self,node):



    def recursiveGenerateTree(self, curData, curAttributes,pruning_threshold, repeat):
        if not repeat:
        # if all the data in the same class
            if len(curData) < pruning_threshold:
                # Fail
                majClass = self.getMajClass(curData)
                #return Node(True, "Fail", None)
                return Node(True, majClass, None)
            else:
                allSame = self.allSameClass(curData)
                if allSame is not False:  # curdata from the same class
                    # return a node with that class
                    return Node(True, allSame, None)
                elif len(curAttributes) == 0:
                    # return a node with the majority class
                    majClass = self.getMajClass(curData)
                    return Node(True, majClass, None)
                else:
                    (best, best_threshold, splitted,best_attribute_index) = self.splitAttribute(curData, curAttributes)
                    remainingAttributes = curAttributes[:]
                    remainingAttributes.remove(best)
                    node = Node(False, best, best_threshold)
                    node.children = [self.recursiveGenerateTree(subset, remainingAttributes,pruning_threshold, repeat) for subset in splitted]
        else:
            if len(curData) < pruning_threshold:
                majClass = self.getMajClass(curData)
                # return Node(True, "Fail", None)
                return Node(True, majClass, None)
            else:
                allSame = self.allSameClass(curData)
                if allSame is not False:  # curdata from the same class
                # return a node with that class
                   return Node(True, allSame, None)
                else:
                    (best, best_threshold, splitted,best_attribute_index) = self.splitAttribute(curData, self.attributes)
                    node = Node(False, best, best_threshold)
                    node.children = [self.recursiveGenerateTree(subset, self.attributes,pruning_threshold, repeat) for subset in splitted]
        return node

    def getMajClass(self, curData):
        freq = [0] * len(self.classes)
        for row in curData:
            index = self.classes.index(row[-2])
            freq[index] += 1
        maxInd = freq.index(max(freq))
        return self.classes[maxInd]

    def allSameClass(self, data):
        for row in data:
            if row[-2] != data[0][-2]:
                return False
        #print(data)
        return data[0][-2]

    def isAttrDiscrete(self, attribute):
        if attribute not in self.attributes:
            raise ValueError("Attribute not listed")
        elif len(self.attrValues[attribute]) == 1 and self.attrValues[attribute][0] == "continuous":
            return False
        else:
            return True

#######################################################
    def selectNoMissingData(self,dataSet,feature):
        noMissingdataSet = dataSet[:]
        for example in dataSet:
            if example[feature] == -1:
                noMissingdataSet.remove(example)
        return noMissingdataSet
##########################################################
    def addMissingData(self,subsets,allData,feature):
        all_weight = self.cal_weight(allData)
        for example in allData:
            if example[feature] == -1:
                for subset in subsets:
                    weight = self.cal_weight(subset) / all_weight
                    subset.append(example)
                    subset[-1][-1] = weight
        return subsets

    ##########################################################

    def splitAttribute(self, curData, curAttributes):
        splitted = []
        maxEnt = -1 * float("inf")
        best_attribute = -1
        best_attribute_index = -1
        # None for discrete attributes, threshold value for continuous attributes
        best_threshold = None
        continuous_attribute = []
        for attribute in curAttributes:
            #print(attribute)
            indexOfAttribute = self.attributes.index(attribute)
            ###########################################################
            nmData = self.selectNoMissingData(curData, indexOfAttribute)
            rho = self.cal_weight(nmData) / self.cal_weight(curData)
            ###########################################################
            if self.isAttrDiscrete(attribute):
                # split nmData into n-subsets, where n is the number of
                # different values of attribute i. Choose the attribute with
                # the max gain
                valuesForAttribute = self.attrValues[attribute]
                subsets = [[] for a in valuesForAttribute]
                all_the_same = True
                for j in range(0,len(nmData)):
                    if nmData[j][indexOfAttribute] != nmData[0][indexOfAttribute]:
                        all_the_same = False
                if not all_the_same:
                    for row in nmData:
                        for index in range(len(valuesForAttribute)):
                            if row[indexOfAttribute] == valuesForAttribute[index]:
                                subsets[index].append(row)
                                break
                    e = rho * self.gain(nmData, subsets)
                    #print('The attributes now is: %s, the gainratio is: %.3f'%(attribute, e))
                    if e > maxEnt:
                        maxEnt = e
                        splitted = subsets
                        best_attribute = attribute
                        best_threshold = None
                        best_attribute_index = indexOfAttribute
            else:
                # sort the data according to the column.Then try all
                # possible adjacent pairs. Choose the one that
                # yields maximum gain
                nmData.sort(key=lambda x: x[indexOfAttribute])
                sub_Ent = -1 * float("inf")
                sub_attribute = -1
                sub_threshold = None
                all_the_same = True
                for j in range(0,len(nmData)):
                    if nmData[j][indexOfAttribute] != nmData[0][indexOfAttribute]:
                        all_the_same = False
                if not all_the_same:
                    for j in range(0, len(nmData) -1):
                        if nmData[j][indexOfAttribute] != nmData[j + 1][indexOfAttribute]:
                            threshold = (nmData[j][indexOfAttribute] + nmData[j + 1][indexOfAttribute]) / 2
                            less = []
                            greater = []
                            for row in nmData:
                                if (row[indexOfAttribute] > threshold):
                                    greater.append(row)
                                else:
                                    less.append(row)
                            e = rho * self.gain(nmData, [less, greater])
                            if e >= sub_Ent:
                                sub_splitted = [less, greater]
                                sub_Ent = e
                                sub_attribute = attribute
                                sub_threshold = threshold
                            if e >= maxEnt:
                                splitted = [less, greater]
                                maxEnt = e
                                best_attribute = attribute
                                best_threshold = threshold
                                best_attribute_index = indexOfAttribute
                    continuous_current = [sub_attribute, sub_Ent, sub_splitted, sub_threshold]
                    continuous_attribute.append(continuous_current)
                #print('The attributes now is: %s, the best threshold is: %lf, the gainratio is: %.3f' % (attribute,sub_threshold, e))
        #print("Then we choose the attibute:%s, the gainratio is %.3lf" %(best_attribute, maxEnt))
        if self.isAttrDiscrete(best_attribute) or (not self.segment):
            return best_attribute, best_threshold, splitted, best_attribute_index
        else:
            return self.select_segment(continuous_attribute)

    def select_segment(self, continuous_attribute):
        if len(continuous_attribute) == 1:
            return continuous_attribute[0][0], continuous_attribute[0][3], continuous_attribute[0][2], 1
        else:
            continuous_attribute.sort(key=lambda x: x[1])
            segment_value = []
            for i in range(min(self.K, len(continuous_attribute))):
                segment_value.append(self.cal_segment(continuous_attribute[i]))
            index = segment_value.index(min(segment_value))
            return continuous_attribute[index][0], continuous_attribute[index][3], continuous_attribute[index][2], 1

    def cal_segment(self, attrArray):
        attrIndex = self.attributes.index(attrArray[0])
        data1 = attrArray[2][0]
        data2 = attrArray[2][1]
        data1.sort(key=lambda x: x[attrIndex])
        data2.sort(key=lambda x: x[attrIndex])
        data1_value = 0
        data2_value = 0
        for i in range(len(data1) - 1):
            if data1[i][-2] != data1[i + 1][-2]:
                data1_value = data1_value + 1
        for i in range(len(data2) - 1):
            if data2[i][-2] != data2[i + 1][-2]:
                data2_value = data2_value + 1
        return (data1_value * len(data1) + data2_value * len(data2)) / (len(data1) + len(data2))

    def gain(self, unionSet, subsets):
        # input : data and disjoint subsets of it
        # output : information gain
        S = self.cal_weight(unionSet)
        # calculate entropy before split
        entropyBeforeSplit = self.entropy(unionSet)
        # calculate entropy after split
        weights = [self.cal_weight(subset) / S for subset in subsets]
        entropyAfterSplit = 0
        SplitInformation = 0
        for i in range(len(subsets)):
            entropyAfterSplit += weights[i] * self.entropy(subsets[i])
        # calculate total gain
        totalGain = entropyBeforeSplit - entropyAfterSplit
        for i in range(len(subsets)):
            SplitInformation -= weights[i] * self.log(weights[i])
        GainRatio = totalGain / SplitInformation
        return GainRatio

#####################################################
    def cal_weight(self,dataSet):
        L = len(dataSet)
        S = 0
        for x in range(L):
            S += dataSet[x][-1]
        return S
######################################################

    def entropy(self, dataSet):
        S = self.cal_weight(dataSet)
        if S == 0:
            return 0
        num_classes = [0 for i in self.classes]
        for row in dataSet:
            classIndex = list(self.classes).index(row[-2])
            num_classes[classIndex] += 1
        num_classes = [x / S for x in num_classes]
        ent = 0
        for num in num_classes:
            ent += num * self.log(num)
        return ent * -1

    def log(self, x):
        if x == 0:
            return 0
        else:
            return math.log(x, 2)

    def test_tree(self):
        result = []
        right_num=0
        for example in self.testdata:
            self.test_node(self.tree, example)
            curlabel = self.child_label
            result.append(curlabel == example[-2])
        #print(result)
        for index in range(len(result)):
            if result[index]==True:
                right_num += 1
        acc = right_num/len(result)
        self.prunningacc = acc
        if self.segment:
            print("strategy with segment:", "Accuracy: " + ' ', acc)
        elif self.pruning_threshold>1:
            print("Accuracy with prunning: " + ' ', acc,'threshold = ', self.pruning_threshold)
        else:
            print("Accuracy without prunning: " , acc)
        return  self.prunningacc

    def printself(self):
        print("data", self.data, '\n')
        print("classes", self.classes, '\n')
        print("numAttributes", self.numAttributes, '\n')
        print("attrValues", self.attrValues, '\n')
        print("attributes", self.attributes, '\n')

    def test_node(self, node, example):
        if not node.isLeaf:
            attrindex = self.attributes.index(node.label)
            if node.threshold is None:
                childindex = self.attrValues[node.label].index(example[attrindex])
                child = node.children[childindex]
                if child.isLeaf:
                    self.child_label = child.label
                else:
                    self.test_node(child, example)
            else:
                leftchild = node.children[0]
                rightchild = node.children[1]
                if example[attrindex] < node.threshold:
                    if leftchild.isLeaf:
                        self.child_label = leftchild.label
                    else:
                        self.test_node(leftchild, example)
                else:
                    if rightchild.isLeaf:
                        self.child_label = rightchild.label
                    else:
                        self.test_node(rightchild, example)



class Node:
    def __init__(self, isLeaf, label, threshold):
        self.label = label
        self.threshold = threshold
        self.isLeaf = isLeaf
        self.children = []

