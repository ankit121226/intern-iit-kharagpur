# coding: utf-8

# In[63]:

'''
Decision Tree Source Code for Steel Quality Prediction
@author: Ankit Kumar
'''

from math import log
import operator
import csv


def createDataSet(filename):
    with open(filename,"r") as f:
        lines=csv.reader(f)
        dataset=list(lines)
        dataset=preprocess(dataset)
        f.close()
        labels=["family","product-type","steel","carbon","hardness","temper_rolling","condition","formability","strength","non-ageing","surface-finish","surface-quality","enamelability","bc","bf","bt","bw/me","bl","m","chrom","phos","cbond","marvi","exptl","ferro","corr","blue/bright/varn/clean","lustre","jurofm","s","p","shape","thick","width","len","oil","bore","packing"]
    return dataset, labels



def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet:  # the the number of unique elements and their occurance
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys(): labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntries
        shannonEnt -= prob * log(prob, 2)  # log base 2
    return shannonEnt


def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]  # chop out axis used for splitting
            reducedFeatVec.extend(featVec[axis + 1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet


def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1  # the last column is used for the labels
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0;
    bestFeature = -1
    for i in range(numFeatures):  # iterate over all the features
        featList = [example[i] for example in dataSet]  # create a list of all the examples of this feature
        uniqueVals = set(featList)  # get a set of unique values
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet) / float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)


        infoGain = baseEntropy - newEntropy  # calculate the info gain; ie reduction in entropy
        """
        print("feature : " + str(i))
        print("baseEntropy : "+str(baseEntropy))
        print("newEntropy : " + str(newEntropy))
        print("infoGain : " + str(infoGain))
        """
        if (infoGain > bestInfoGain):  # compare this to the best gain so far
            bestInfoGain = infoGain  # if better than current best, set to best
            bestFeature = i
    return bestFeature  # returns an integer


def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys(): classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


def createTree(dataSet, labels):
    # extracting data
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList):
        return classList[0]  # stop splitting when all of the classes are equal
    if len(dataSet[0]) == 1:  # stop splitting when there are no more features in dataSet
        return majorityCnt(classList)
    # use Information Gain
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]

    #build a tree recursively
    myTree = {bestFeatLabel: {}}
    #print("myTree : "+labels[bestFeat])
    del (labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    #print("featValues: "+str(featValues))
    uniqueVals = set(featValues)
    #print("uniqueVals: " + str(uniqueVals))
    for value in uniqueVals:
        subLabels = labels[:]  # copy all of labels, so trees don't mess up existing labels
        #print("subLabels"+str(subLabels))
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)
        #print("myTree : " + str(myTree))
    return myTree


def classify(inputTree, featLabels, testVec):
    firstStr = inputTree.keys()[0]
    #print("fistStr : "+firstStr)
    secondDict = inputTree[firstStr]
    #print("secondDict : " + str(secondDict))
    featIndex = featLabels.index(firstStr)
    #print("featIndex : " + str(featIndex))
    key = testVec[featIndex]
    #print("key : " + str(key))
    if key not in secondDict.keys():
        return -1
    
    valueOfFeat = secondDict[key]
    #print("valueOfFeat : " + str(valueOfFeat))
    if isinstance(valueOfFeat, dict):
        #print("is instance: "+str(valueOfFeat))
        classLabel = classify(valueOfFeat, featLabels, testVec)
    else:
        #print("is Not instance: " + valueOfFeat)
        classLabel = valueOfFeat
    return classLabel

def drawTree(myTree):
    import pydot

    menu=myTree
    
    def draw(parent_name, child_name):
        edge = pydot.Edge(parent_name, child_name)
        graph.add_edge(edge)

    def visit(node, parent=None):
        for k,v in node.iteritems():
            if isinstance(v, dict):
            # We start with the root node whose parent is None
            # we don't want to graph the None node
                if parent:
                    draw(parent, k)
                visit(v, k)
            else:
                draw(parent, k)
            # drawing the label using a distinct name
                draw(k, str(k)+'_'+str(v))

    graph = pydot.Dot(graph_type='graph')
    visit(menu)
    graph.write_png('DecisionTree.png')
    
def preprocess(dataset):
    
    #Change all attributes's classes to class of integers except atrributes with continuous values

    
    continuous=[3,4,8,32,33,34]
    total_attributes=len(dataset[0])-1
    for attribute in range(total_attributes):
        if attribute not in continuous:
            #print (attribute)
            classes_in_attributes=list(set([row[attribute] for row in dataset]))
            classDict={}
            for index in range(len(classes_in_attributes)):
                classDict[classes_in_attributes[index]]=index
            for row in dataset:
                row[attribute]=int(classDict[row[attribute]])
    
    for attribute in continuous:
        for row in dataset:
            row[attribute]=float(row[attribute])
                
    for attribute in continuous:
        bins=80
        binSize=0
        minVal=(dataset[0][attribute])
        maxVal=(dataset[0][attribute])
        for i in range(1,len(dataset)):
            if dataset[i][attribute]>maxVal:
                maxVal=dataset[i][attribute]
            elif dataset[i][attribute]<minVal:
                minVal=dataset[i][attribute]
        binSize=(maxVal-minVal)/bins
        #print (attribute,minVal,maxVal)
        for i in range(bins):
            for row in dataset:
                if row[attribute]>=minVal+(i*binSize) and row[attribute]<=minVal+((i+1)*binSize):
                    row[attribute]=i
        
        #column=[row[attribute] for row in dataset]
        #print (column)
    return dataset


# collect data
myDat, labels = createDataSet("train.csv")

#build a tree
mytree = createTree(myDat, labels)
print(mytree)

drawTree(mytree)

testDat,labels=createDataSet("test.csv")

correct=0

for row in testDat:
    prediction=classify(mytree,labels,row[0:len(row)-1])
    actual=row[-1]
    print ("actual: "+str(actual)+"\tPredicted: "+str(prediction))
    if(prediction==actual):
        correct+=1
print ("Accuracy Percentage: "+str(correct/float(len(testDat))*100))
    

