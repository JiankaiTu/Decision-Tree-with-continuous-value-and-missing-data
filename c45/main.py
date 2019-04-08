#!/usr/bin/env python
import pdb
from c45 import C45
import treePlotter
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

c1 = C45("../data/iris/iris.data", "../data/iris/iris.names")
c1.fetchData()
c1.preprocessData()

'''
#if want to test pruning #########
pruningacc = []
for threshold in range(2,20):
    c1.setprunning(threshold)
    c1.generateTree(True)
    acc= c1.test_tree()
    pruningacc.append(acc)
print(pruningacc)
x = range(2,20)
plt.figure()
plt.plot(x,pruningacc)
plt.xlabel('pruning threshold')
plt.ylabel('accuracy')
plt.title('pruning')
plt.show()
'''
'''
# if don't want to use the segment
c1.bool_segment(False, 2)
threshold = 1
#if cl.generateTree(True), the attribute can be used once, False many times
c1.setprunning(threshold)
c1.generateTree(False)
c1.printTree()
acc = c1.test_tree()
#c1.printself()
tree = c1.convert_dic()
treePlotter.createPlot(tree)
'''

#if want to use the segment
c1.bool_segment(False, 2)
threshold = 1
#if cl.generateTree(True), the attribute can be used once, False many times
c1.setprunning(threshold)
c1.generateTree(False)

c1.printTree()
#c1.printself()
acc = c1.test_tree()
tree = c1.convert_dic()
treePlotter.createPlot(tree)
