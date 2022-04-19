#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: Sami Idreesi

Student ID: 201348278
"""
import numpy as np
import random

#method to parse training and test data to array of tuples (classlabel,features)
def getInstances(fname):
    data = []
    with open(fname) as file:
        for line in file:
            currentObject = line.split(",")
            label = currentObject.pop().strip()
            features = np.asarray(currentObject, dtype=np.float64, order='C')
            data.append((label,features))
    return data

#adds the class label +1 to all class1 objects and -1 to call class2 objects in the dataset and only uses class1 and class2 objects
# giving an array of (+1/-1, original classlabel, features)
#for one vs rest multi-class classification, 2nd argument has to be passed as "" so sets the label of the class1 to +1 and the
#rest of the classes to -1
def changeClassLabel(data,class1,class2):
    perceptronData = []
    for i in data:
        #class label
        y = list(i)[0]
        #features
        X = list(i)[1]
        if(y == class1):
            perceptronData.append((1,y,X))
        elif(y == class2):
            perceptronData.append((-1,y,X))
        #one vs rest multi-class classification
        elif(class2 == ""):
            perceptronData.append((-1,y,X))
    return perceptronData 

#get training and test instances from files
training_data = getInstances('train.data')
test_data = getInstances('test.data')

#uncomment/comment to randomize/unrandomize datasets
#random.shuffle(training_data)
#random.shuffle(test_data)


#Perceptron algorithm
def perceptronTrain(trainingData,maxIter,class1,class2):
    #change class labels to +1 and -1 and only use class1 and class2 objects
    trainingData = changeClassLabel(trainingData, class1, class2)
    #get the number of features from an abritrary training object, the first in this case
    numFeatures = len(trainingData[0][2])
    #initialize weights and bias
    weightVector = np.zeros(numFeatures)
    bias = 0
    #number of iterations to run the perceptron on training dataset
    for i in range(maxIter):
        for (classLabel, classl, features) in trainingData:
            activationScore = np.inner(weightVector,features) + bias
            #print(classLabel, classl, activationScore)
            if(classLabel*activationScore <=0):
                #print("Misclassification:",classLabel,activationScore)
                weightVector = weightVector + classLabel*features
                bias += classLabel
    return (bias,weightVector)   


def perceptronAccuracy(data,bias,weightVector,class1,class2):
    #change class labels to +1 and -1 and only use class1 and class2 objects
    data = changeClassLabel(data, class1, class2)
    #correctPredictions = 0
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    #totalInstances = len(data)
    for (classLabel, classl, features) in data:
        activationScore = np.inner(weightVector,features) + bias
        #print("Activation score:,", activationScore)
        #print("Class label:", classLabel)
        #correct prediction
        if(np.sign(activationScore) == classLabel):
            if(np.sign(activationScore) == 1):
                tp+=1
            elif(np.sign(activationScore) == -1):
                tn+=1
            #correctPredictions +=1
            #print("Correct prediction")
        #incorrect prediction 
        else:
            #print("Incorrect prediction")
            if(np.sign(activationScore) == 1):
                fp+=1
            elif(np.sign(activationScore) == -1):
                fn+=1
    accuracy = (tp + tn)/(tp+tn+fp+fn)*100
    return accuracy


"""
Q3. Perceptron accuracies when testing with training data
"""
print("Q3. Binary Perceptron accuracies: Unrandomised\n")

(bias,weightVector) = perceptronTrain(training_data, 20, "class-1", "class-2")
class1_class2_TrainingAccuracy = perceptronAccuracy(training_data, bias, weightVector, "class-1", "class-2")
print("Training data accuracy: class 1 and class 2: ", class1_class2_TrainingAccuracy,"%")

(bias,weightVector) = perceptronTrain(training_data, 20, "class-2", "class-3")
class2_class3_TrainingAccuracy = perceptronAccuracy(test_data, bias, weightVector, "class-2", "class-3")
print("Training data accuracy: class 2 and class 3: ", class2_class3_TrainingAccuracy,"%")

(bias,weightVector) = perceptronTrain(training_data, 20, "class-1", "class-3")
class1_class3_TrainingAccuracy = perceptronAccuracy(training_data, bias, weightVector, "class-1", "class-3")
print("Training data accuracy: class 1 and class 3: ", class1_class3_TrainingAccuracy,"%\n")

"""
Perceptron accuracies when testing with test data
"""
(bias,weightVector) = perceptronTrain(training_data, 20, "class-1", "class-2")
class1_class2_TestAccuracy = perceptronAccuracy(training_data, bias, weightVector, "class-1", "class-2")
print("Test data accuracy: class 1 and class 2: ", class1_class2_TestAccuracy,"%")

(bias,weightVector) = perceptronTrain(training_data, 20, "class-2", "class-3")
class2_class3_TestAccuracy = perceptronAccuracy(training_data, bias, weightVector, "class-2", "class-3")
print("Test data accuracy: class 2 and class 3: ", class2_class3_TestAccuracy,"%")

(bias,weightVector) = perceptronTrain(training_data, 20, "class-1", "class-3")
class1_class3_TestAccuracy = perceptronAccuracy(training_data, bias, weightVector, "class-1", "class-3")
print("Test data accuracy: class 1 and class 3: ", class1_class3_TestAccuracy,"%\n")

#calculate prediction models for each class using one vs rest approach    
def calculatePredictionValues(data,maxIter):
    #dictionary to store the prediction model of each class: {key: class, value:(bias,weightVector)}
    classPredictionModel = {}
    numFeatures = len(data[0][1])
    predictBias = 0
    predictWeightVector = np.zeros(numFeatures)
    #initialize prediction models dictionary
    for (classLabel,features) in data:
        if classLabel not in classPredictionModel:
            classPredictionModel[classLabel] = (predictBias,predictWeightVector)
    #apply one vs rest approach for each class for maxIter iterations        
    for i in range(maxIter):
        for classLabel in classPredictionModel:
            classPredictionModel[classLabel] = perceptronTrain(data, maxIter, classLabel, "")
    return classPredictionModel
        

#apply the one vs rest approach using the prediction models
def multiClassPerceptron(data,predictionModels):
    correctPredictions = 0
    totalInstances = len(data)
    for(classLabel,features) in data:
        predictedLabel = ""
        maxScore = -9999999
        for predictModelLabel in predictionModels:
            (bias,weightVector) = predictionModels[predictModelLabel]
            activationScore = np.inner(weightVector,features) + bias
            #print("Predict label:",predictModelLabel," Activation score:",activationScore)
            if(activationScore >= maxScore):
                maxScore = activationScore
                predictedLabel = predictModelLabel       
        if(predictedLabel==classLabel):
            correctPredictions+=1
        #print("Predicted:",predictedLabel,"Actual:",classLabel)
    accuracy = (correctPredictions/totalInstances)*100
    #print("Accuracy =",accuracy,"%")
    return accuracy

"""
Q4. Perform multi-class classification using the 1-vs-rest approach and calculate accuracies
"""

print("Q4. Multi-class Perceptron accuracies: Unrandomised\n")
    
predictionModels = calculatePredictionValues(training_data, 20)    
        
multiClassTestAccuracy = multiClassPerceptron(test_data, predictionModels)

print("Test data accuracy:",multiClassTestAccuracy,"%\n")

multiClassTrainAccuracy = multiClassPerceptron(training_data, predictionModels)

print("Training data accuracy:",multiClassTrainAccuracy,"%\n")


#Perceptron algorithm with update rule using an l2 regression term
def l2perceptronTrain(trainingData,maxIter,l2term,class1,class2):
    #change class labels to +1 and -1 and only use class1 and class2 objects
    trainingData = changeClassLabel(trainingData, class1, class2)
    numFeatures = len(trainingData[0][2])
    #initialize weights and bias
    weightVector = np.zeros(numFeatures)
    bias = 0
    #number of iterations to run the perceptron on training dataset
    for i in range(maxIter):
        for (classLabel, classl, features) in trainingData:
            activationScore = np.inner(weightVector,features) + bias
            #print(classLabel, classl, activationScore)
            if(classLabel*activationScore <=0):
                #print("Misclassification:",classLabel,activationScore)
                weightVector = (1-2*l2term)*weightVector + classLabel*features
                bias += classLabel
    return (bias,weightVector)  

#claculate prediction models for each class using one vs rest approach with an l2 regularisation term  
def calculatel2predictionValues(data,maxIter,l2term):
    classPredictionModel = {}
    numFeatures = len(data[0][1])
    bias = 0
    weightVector = np.zeros(numFeatures)
    #initialize prediction models dictionary
    for (classLabel,features) in data:
        if classLabel not in classPredictionModel:
            classPredictionModel[classLabel] = (bias,weightVector)   
    for i in range(maxIter):
        for classLabel in classPredictionModel:
            classPredictionModel[classLabel] = l2perceptronTrain(data, maxIter, l2term, classLabel, "")
    return classPredictionModel

#calculate accuracies using each l2 term and output best l2term
def l2multiClassPerceptron(trainingData,testData,maxIter,l2terms):
    bestAccuracy = 0
    bestl2term = 0
    #apply one vs rest multi class classification using for each l2 value
    for l2term in l2terms:
        l2multiClassPredictions = calculatel2predictionValues(trainingData, maxIter, l2term)
        accuracy = multiClassPerceptron(testData, l2multiClassPredictions)
        print("l2 value:",l2term, "Accuracy:", accuracy,"%")
        if(accuracy >= bestAccuracy):
            bestAccuracy = accuracy
            bestl2term = l2term
            bestl2multiClassPredictions = l2multiClassPredictions
    print("best l2 value:",bestl2term, "Accuracy:", bestAccuracy,"%")       
    return(bestl2term,bestl2multiClassPredictions)        
    
"""
Q5. Perform multi-class classification using the 1-vs-rest approach with l2 regularisation values and calculate accuracies
"""    
    
print("Q5. Multi-class Perceptron accuracies using l2 values: Unrandomised\n")

l2values = [0.01, 0.1, 1.0, 10.0, 100.0]

print("Training data accuracies: ")
l2multiClassPerceptron(training_data, training_data, 20, l2values)

print()

print("Test data accuracies: ")
l2multiClassPerceptron(training_data, test_data, 20, l2values)


