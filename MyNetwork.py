# -*- coding: utf-8 -*-
import numpy as np
from numpy.random import seed
import time
import sys
from matplotlib import pyplot as plt

GraphDataX = []
GraphDataY = []
'''
    This function will be used to read data from train and test files.
    This function is used for both train and test files because the format is
    the same for both the files.
    
    This function will be returning a list which has all the image data
    from the text file - the name is passed as an argument.
'''
def ReadFile(FileName):
    OpenedFile = open(FileName, 'r')
    
    DataList = []
    Line = 'starting'
    
    # Iterating through the file - line after line - till an empty string is not read
    while Line != '':
        List = []
        
        Line = OpenedFile.readline()
        
        while True:
            Length = len(Line)
            length = 0
            Result = ''
            check = True
            
            for x in range(Length):
                if Line[x] == ']':
                    Result = Line[:length]
                    check = False
                else:
                    length += 1
                    
            EndOrNotResult = check
            
            if EndOrNotResult == False:
                break
            
            LineSplit = Line.split()
            NumberOfElementsInLine = len(LineSplit)
            
            if NumberOfElementsInLine <= 0:
                break
            elif LineSplit[0] == '[':
                for x in range(1, NumberOfElementsInLine):
                    Number = LineSplit[x]
                    Number = int(Number)
                    # Normalisation
                    Number = Number/float(255)
                    
                    List.append(Number)
                Line = OpenedFile.readline()
            else:
                for x in range(NumberOfElementsInLine):
                    Number = LineSplit[x]
                    Number = int(Number)
                    # Normalisation
                    Number = Number/float(255)
                    
                    List.append(Number)
                
                Line = OpenedFile.readline()
                
                Length = len(Line)
                length = 0
                Result = ''
                check = True
                
                for x in range(Length):
                    if Line[x] == ']':
                        Result = Line[:length]
                        check = False
                    else:
                        length += 1
                        
                EndOrNotResult = check
                
                
                if EndOrNotResult == False:
                    
                    LineSplit = Result.split()
                    NumberOfElementsInLine = len(LineSplit)
                    
                    for x in range(NumberOfElementsInLine):
                        Number = LineSplit[x]
                        Number = int(Number)
                        Number = Number/float(255)
                        
                        List.append(Number)
                    break
        DataList.append(List)
    return DataList

'''
    This function will be used to read data from train-labels and
    test-labels files.
    This function is used for both train-labels and test-labels files
    because the format is the same for both the files.
    
    This function will be returning a list which has all the labels for
    each of the images corresponding to each row of the train.txt and
    test.txt from the text file - the name is passed as an argument.
'''
def ReadLabels(FileName):
    OpenedFile = open(FileName, 'r')
    
    Labels = []
    
    label = OpenedFile.readline()
    while label != '':
        label = int(label)
        Labels.append(label)
        label = OpenedFile.readline()
    return Labels

'''
    This function implements the sigmoid function using numpy
    Sigmoid function is 1/(1+e^(-x))
'''
def sigmoid(x):
    exp = np.exp(-x)
    denominator = 1.0 + exp
    Final = 1.0/denominator
    
    return Final

'''
    This function returns your label into one hot encoding array
    label is a digit
    Ex-> If label is 1 then one hot encoding should be [0,1,0,0,0,0,0,0,0,0]
    Ex-> If label is 9 then one hot encoding shoudl be [0,0,0,0,0,0,0,0,0,1]
'''
def generate_label(label):
    List = np.zeros(10)
    List[label] = 1
    List = np.array(List)
    return List

'''
    This function will be training the model.
'''
def TrainModel(TrainingData, TrainingLabels, LearningRate):
    # initialising weights
    HiddenLayerWeightRange = (784, 30)
    seed(1)
    HiddenLayerWeight = 2 * np.random.random(HiddenLayerWeightRange) - 1
    seed(1)
    OutputLayerWeightRange = (30, 10)
    OutputLayerWeight = 2 * np.random.random(OutputLayerWeightRange) - 1
    
    # training now
    Epoches = 2
    for x in range(Epoches):
        print('Epoch Number: ', x + 1)
        LengthOfTrainingLabels = len(TrainingLabels)
        
        for y in range(LengthOfTrainingLabels):
            HiddenLayer = np.dot(TrainingData[y], HiddenLayerWeight)
            HiddenLayer = sigmoid(HiddenLayer)
            HiddenLayer = np.array(HiddenLayer, dtype = float)
            
            
            OutputLayer = np.dot(HiddenLayer, OutputLayerWeight)
            Activation = sigmoid(OutputLayer)
            Activation = np.array(Activation, dtype = float)
            
            TargetLabelArray = generate_label(TrainingLabels[y])
            Difference = Activation - TargetLabelArray
            
            x = 1 - TargetLabelArray
            
            LogValue = np.log(Activation)
            Dotproduct = x.dot(LogValue)
            
            Result = LogValue + Dotproduct
            
            Error = TargetLabelArray.dot(Result)
            
            Size = len(TrainingLabels)
            Error = Error/Size
            Error = -1 * Error

            if Error < 0.0000002:
                break

            HiddenLayerArray = np.array([HiddenLayer])
            HiddenLayerArrayTranspose = HiddenLayerArray.transpose()
            
            TargetLabelDifferenceArray = np.array([Difference])
            
            HiddenLayerError = np.matmul(HiddenLayerArrayTranspose, TargetLabelDifferenceArray)
            
            
            DeltaOutputLayer = LearningRate*HiddenLayerError            
            OutputLayerWeight =  OutputLayerWeight - DeltaOutputLayer
            
            TargetLabelDifferenceArray = np.array([Difference])
            TargetLabelDifferenceArrayTranspose = TargetLabelDifferenceArray.transpose()
            
            E1 = DerivativeFunction(HiddenLayer)
            E1 = np.array([E1])
            
            E2 = np.dot(OutputLayerWeight, TargetLabelDifferenceArrayTranspose)
            E2Transpose = E2.transpose()
            
            E3 = np.multiply(E1, E2Transpose)
            
            TrainingDataArray = np.array([TrainingData[y]])
            TrainingDataArrayTranspose = TrainingDataArray.transpose()
            
            InputToHiddenErrorValue = np.dot(TrainingDataArrayTranspose, E3)
            
            
            DeltaHiddenLayer = LearningRate*InputToHiddenErrorValue
            HiddenLayerWeight = HiddenLayerWeight - DeltaHiddenLayer
        
    print('Writing to netWights.txt\n')
    # writing weights to file 'netWeights.txt'
    WeightsFile = open('netWeights.txt', 'w')
    
    WeightsFile.write('Hidden Layer\n')
    np.savetxt(WeightsFile, HiddenLayerWeight)
    
    WeightsFile.write('Output Layer\n')
    np.savetxt(WeightsFile, OutputLayerWeight)
    
    WeightsFile.write('End\n')
    WeightsFile.close()

def DerivativeFunction(x):
    Result = x * (1 - x)
    return Result


'''
    This function reads file which contains the netWeights - the ones which
    formed during the training of the model
'''
def ReadNetWeights(FileName):
    OpenedFile = open(FileName, 'r')
    
    HiddenLayer = []
    
    Line = OpenedFile.readline()
    
    Line = OpenedFile.readline()
    while Line != 'Output Layer\n':
        OldLine = Line.split()
        Line = map(float, OldLine)
        Line = list(Line)
                
        if OldLine != []:
            HiddenLayer.append(Line)
        Line = OpenedFile.readline()
    HiddenLayer = np.array(HiddenLayer)
    
    
    OutputLayer = []
    
    Line = OpenedFile.readline()
    while Line != 'End\n':
        OldLine = Line.split()
        Line = map(float, OldLine)
        Line = list(Line)
        
        if OldLine != []:
            OutputLayer.append(Line)
        Line = OpenedFile.readline()
    OutputLayer = np.array(OutputLayer)
    
    OpenedFile.close()
    
    return (HiddenLayer, OutputLayer)
    
'''
    This function will be testing the model.
'''
def TestModel(TestingData, TestingLabels, HiddenLayerWeight, OutputLayerWeight):
    print('Testing now...\n')
    
    LengthOfTestingLabels = len(TestingLabels)
    Epoches = 1
    for num in range(Epoches):
        Accurate = 0
        for x in range(LengthOfTestingLabels):
            GraphDataX.append(time.time())
            GraphDataY.append(Accurate*100/len(TestingLabels))
            HiddenLayer = np.dot(TestingData[x], HiddenLayerWeight)
            
            HiddenLayer = sigmoid(HiddenLayer)
            HiddenLayer = np.array(HiddenLayer, dtype = float)
            
            OutputLayer = np.dot(HiddenLayer, OutputLayerWeight)
            OutputLayer = sigmoid(OutputLayer)
            OutputLayer = np.array(OutputLayer, dtype = float)
            
            MaxNumber = OutputLayer[0]
            MaxNumberIndex = 0
            LengthOfOutputLayer = len(OutputLayer)
            for y in range( LengthOfOutputLayer):
                if MaxNumber < OutputLayer[y]:
                
                    MaxNumber = OutputLayer[y]
                    MaxNumberIndex = y
            
            Label = MaxNumberIndex
            if(TestingLabels[x] == Label):
                Accurate += 1
        
        print('Epoch Number ', num+1, ' -------> ',  Accurate, '/', len(TestingLabels), 'images correctly classified.\n')
        Percentage = Accurate*100/len(TestingLabels)
        Error = 100 - Percentage
        print('Accuracy ', Percentage, ' %  ------------------- Error ', Error, ' %\n')
    print('Testing ended...\n')


def makeGraph():
    plt.plot(GraphDataX, GraphDataY)
    plt.title('Accuracy % vs Time')
    plt.xlabel('Time(s)')
    plt.ylabel('Accuracy %')
    plt.show()

def main(Argument1, Argument2, Argument3, Argument4):
    if Argument1 == 'train':
        print('Loading training data... \n')
        TrainingData = ReadFile(Argument2)
        TrainingData = np.array(TrainingData)

        print('Loading training labels... \n')
        TrainingLabels = ReadLabels(Argument3)
        TrainingLabels = np.array(TrainingLabels)

        LearningRate = Argument4
        LearningRate = float(LearningRate)

        print('Training model now....\n')

        start_time = time.time()
        TrainModel(TrainingData, TrainingLabels, LearningRate)
        print('Training ended...\n')
        end_time = time.time()
        Duration = end_time - start_time
        print('Time taken to train is ', Duration/float(60), ' minutes.')
        
    elif Argument1 == 'test':
        print('Loading testing data... \n')
        TestingData = ReadFile(Argument2)
        TestingData = np.array(TestingData)
        
        print('Loading testing labels... \n')
        TestingLabels = ReadLabels(Argument3)
        TestingLabels = np.array(TestingLabels)
        
        print('Loading netWeights.txt \n')
        HiddenLayerWeight, OutputLayerWeight = ReadNetWeights(Argument4)
        
        TestModel(TestingData, TestingLabels, HiddenLayerWeight, OutputLayerWeight)
        makeGraph()
        
        
        
main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])

