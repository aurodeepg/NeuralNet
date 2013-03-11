from pylab import *
import numpy as np
import matplotlib.pyplot as plt
import csv
import random
import math

######################################
# Class to represent a neuron
######################################
class Neuron:
    # Variables
    input = None
    bias = None
    output = None

    # Constructor
    def __init__(self):
        self.input = []
        
######################################
# Class to represent the hidden layer
######################################
class HiddenLayer:
    hiddenList = None

    # Constructor
    def __init__(self):
        hiddenList = []
        
######################################
# Class to represent the bias neurons
######################################
class OutputLayer:
    outputList = None

    # Constructor
    def __init__(self):
        outputList = []
   
######################################
# Function to print updated weights
# after given epoch
######################################   
def print_weights(outFile,hiddenNeurons,outputNeurons):
    writer = csv.writer(outFile,dialect = 'excel')
    for neuron in hiddenNeurons:
        writer.writerow([neuron.bias]+neuron.input)
    for neuron in outputNeurons:
        writer.writerow([neuron.bias]+neuron.input)
        
######################################
# Function to read values from file
######################################
def read_values(filename):
    values = []
    print("Loading data from : " + filename)
    
    # read values from csv into list
    values = list(csv.reader(open(filename)))
    
    return values    
        
#########################
# Main program
#########################   
def main():    
    if len(sys.argv) != 3:
        print('Usage: python2.7 execute.py TrainingFilename CSVFilename')
        return
        
    else:
        # Declaring Variables
        maxOutput = 0
        recogRate = 0.0
    
        # Read training weight values
        trainingVals = []
        trainingVals = read_values(sys.argv[1])
        
        # Read input values
        inputVals = []
        inputVals = read_values(sys.argv[2])
        
        # List containing hidden and output neurons
        hiddenNeurons = []
        outputNeurons = []
        
        # Profit function
        boltProfit = [0.20,-0.07,-0.07,-0.07]
        nutProfit = [-0.07,0.15,-0.07,-0.07]
        ringProfit = [-0.07,-0.07,0.05,-0.07]
        scrapProfit = [-0.03,-0.03,-0.03,-0.03]
        prftFunction = []
        prftFunction.append(boltProfit)
        prftFunction.append(nutProfit)
        prftFunction.append(ringProfit)
        prftFunction.append(scrapProfit)
        
        #  Create Confusion Matrix
        confMat = []        
        for i in range (0,4):
            confList = []
            for j in range (0,4):
                confList.append(0.0)
            confMat.append(confList)                
        
        # Creating Hidden and Output Neurons with trained weights
        for i in range (0,9):
            if i <5:
                tempHiddenNeuron = Neuron()
                tempinput = []
                for j in range (0,2):
                    tempinput.append(float(trainingVals[i][j+1]))
                tempHiddenNeuron.input = tempinput
                tempHiddenNeuron.bias = float(trainingVals[i][0])
                hiddenNeurons.append(tempHiddenNeuron)
            else:
                tempOutputNeuron = Neuron()
                tempinput = []
                for j in range (0,5):
                    tempinput.append(float(trainingVals[i][j+1]))
                tempOutputNeuron.input = tempinput
                tempOutputNeuron.bias = float(trainingVals[i][0])
                outputNeurons.append(tempOutputNeuron)
           
        #'''        
        for val in inputVals:                
            # Declare local variables
            rowNum = 0
            columnNum = 0
            tempOutputList = []
            maxPos = 0
            valZero = float(val[0])
            valOne = float(val[1])
            valTwo = int(val[2])
            subVal = 0
            
            # running Forward Propagation
            # In Hidden Layer
            for neuron in hiddenNeurons:
                neuron.output = 1/(1+math.exp((-1)*((valZero*neuron.input[0])+(valOne*neuron.input[1])+neuron.bias)))            
                
            # In Output Layer
            for neuron in outputNeurons:
                sumOfWeights = 0
                for i in range(0,5):
                    sumOfWeights = sumOfWeights+(hiddenNeurons[i].output*neuron.input[i])
                sumOfWeights = sumOfWeights+neuron.bias
                neuron.output = 1/(1+math.exp((-1)*sumOfWeights))
                tempOutputList.append(neuron.output)
                #print(neuron.output)
                
            # Getting max output position
            maxPos = tempOutputList.index(max(tempOutputList))
            
            # Assign Confusion Matrix values
            confMat[valTwo-1][maxPos] = confMat[valTwo-1][maxPos] + 1
            
        #'''
        
        # Print Confusion Matrix
        print "\n"
        print "Confusion Matrix : "
        
        for i in range(0,4):
            print confMat[i]
            print "\n"
        
        # Calculate classification errors and recognition rate
        numCorrectClass = 0
        for i in range(0,4):
            for j in range(0,4):
                if i == j:
                    numCorrectClass = numCorrectClass + int(confMat[i][j])
                        
            
        # Print profit matrix
        print "\nProfit Matrix : "        
        prftMat = []
        for i in range(0,4):
            tempList = []
            for j in range(0,4):
                tempList.append(0)
            prftMat.append(tempList)            
        for i in range(0,4):
            for j in range(0,4):
                prftMat[i][j] = round(confMat[i][j]*prftFunction[i][j],2)                
        for i in range(0,4):
            print prftMat[i]
            print "\n"
            
        # Print Profit/Loss Amount
        profitLoss = 0
        for i in range (0,4):
            for j in range (0,4):
                profitLoss = profitLoss + prftMat[i][j]
                
        
        print "Classification errors : "+str(len(inputVals)-numCorrectClass)
        totalVals = len(inputVals)
        print "Recognition Rate : "+str((float(numCorrectClass)/totalVals)*100)+"%"
        print "Total Profit/Loss : " + str(profitLoss) + "\n"
        
        # Histogram Plot
        norm_conf = []
        for i in confMat:
            a = 0
            tmp_arr = []
            a = sum(i, 0)
            for j in i:
                tmp_arr.append(float(j)/float(a))
            norm_conf.append(tmp_arr)

        fig = plt.figure()
        plt.clf()
        ax = fig.add_subplot(111)
        ax.set_aspect(1)
        res = ax.imshow(np.array(norm_conf), cmap=plt.cm.jet, 
                        interpolation='nearest')

        width = len(confMat)
        height = len(confMat[0])

        for x in xrange(width):
            for y in xrange(height):
                ax.annotate(str(int(confMat[x][y])), xy=(y, x), 
                            horizontalalignment='center',
                            verticalalignment='center')

        cb = fig.colorbar(res)
        alphabetX = '0123456789'
        alphabetY = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        plt.xticks(range(width), alphabetX[:width])
        plt.yticks(range(height), alphabetY[:height])
        plt.savefig('confusion_matrix.png', format='png')
       
# Execute the main program. 
main()