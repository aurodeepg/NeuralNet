from pylab import *
from matplotlib.pyplot import plot, draw, show
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
    if len(sys.argv) != 2:
        print('Usage: python2.7 train.py CSVFilename')
        return
        
    else:
        # Declaring Variables  
        # for calculating SSE
        maxOutput = 0
        sumSqErrors = 0
        
        # Errors for output neurons
        delOut = []
        
        # Errors for hidden layer neurons
        delHidden = []
        
        # Sum of Squared errors list
        sseList = []
        
        # List containing hidden and output neurons
        hiddenNeurons = []
        outputNeurons = []
        
        # File handling variables
        outZero = open("training_run_zero.csv", 'wb')        
        outTen = open("training_run_ten.csv", 'wb')
        outHundred = open("training_run_hundred.csv", 'wb')
        outThousand = open("training_run_thousand.csv", 'wb')
        outTenThousand = open("training_run_ten_thousand.csv", 'wb')
    
        # Read input values
        inputVals = []
        inputVals = read_values(sys.argv[1])
        
        # Creating Hidden Neurons
        for i in range (0,5):
            tempHiddenNeuron = Neuron()
            tempinput = []
            for j in range (0,2):
                randomWeight = random.uniform(-1.0, 1.0)
                tempinput.append(randomWeight)
            tempHiddenNeuron.input = tempinput
            tempHiddenNeuron.bias = random.uniform(-1.0, 1.0)
            hiddenNeurons.append(tempHiddenNeuron)
        
        # Creating Output Neurons
        for i in range (0,4):
            tempOutputNeuron = Neuron()
            tempinput = []
            for j in range (0,5):
                randomWeight = random.uniform(-1.0, 1.0)
                tempinput.append(randomWeight)
            tempOutputNeuron.input = tempinput
            tempOutputNeuron.bias = random.uniform(-1.0, 1.0)
            outputNeurons.append(tempOutputNeuron)
               
        # Print weights at Zero-th Epoch 
        print_weights(outZero,hiddenNeurons,outputNeurons)
                
        # train the Neural Network and give weights as output
        # at each specified epoch
        # '''
        for k in range(0,10000):
            sumSqErrors = 0
            
            # train the network
            for sampleNum in range (0,len(inputVals)):                
                # Declare local variables
                tempOutputList = []
                valZero = float(inputVals[sampleNum][0])
                valOne = float(inputVals[sampleNum][1])
                valTwo = int(inputVals[sampleNum][2])
                subVal = 0
                
                # running Forward Propagation
                # In Hidden Layer
                for neuron in hiddenNeurons:
                    valCalc = (valZero*neuron.input[0])+(valOne*neuron.input[1])+neuron.bias
                    neuron.output = 1/(1+math.exp(-valCalc))                
                    
                # In Output Layer
                for neuron in outputNeurons:
                    sumOfWeights = 0
                    for i in range(0,5):
                        sumOfWeights = sumOfWeights+(hiddenNeurons[i].output*neuron.input[i])
                    sumOfWeights = sumOfWeights+neuron.bias
                    neuron.output = 1/(1+math.exp(-sumOfWeights))
                    tempOutputList.append(neuron.output)                
                
                # Calculate the sum of square errors for each sample
                maxOutput = max(tempOutputList)
                maxIndex = tempOutputList.index(max(tempOutputList))                
                for i in range(0,4):
                    if i == maxIndex:
                        sumSqErrors = sumSqErrors+(math.pow((tempOutputList[i] - 1),2))
                    else:
                        sumSqErrors = sumSqErrors+(math.pow((tempOutputList[i] - 0),2))
                
                # Calculating errors at output neurons using sigmoid derivation
                delOut = []
                for i in range(0,4):
                    subVal = 0
                    #print i
                    if(valTwo == (i+1)):
                        #print(valTwo)
                        subVal = 1
                    output = outputNeurons[i].output
                    error = (output - subVal)*output*(1.0 - output)
                    delOut.append(error)
                    
                # Calculating errors at hidden neurons using sigmoid derivation
                delHidden = []
                for i in range(0,5):
                    output = hiddenNeurons[i].output
                    sumOfErrors = 0
                    for j in range(0,4):
                        sumOfErrors = sumOfErrors+(outputNeurons[j].input[i]*delOut[j])
                    error = sumOfErrors*output*(1.0 - output)
                    delHidden.append(error)
                    
                # Update weights
                eta = 0.1
                
                # for output layer
                for i in range(0,4):
                    for j in range(0,5):
                        outputNeurons[i].input[j] = outputNeurons[i].input[j] - (eta*delOut[i]*hiddenNeurons[j].output)
                    outputNeurons[i].bias = outputNeurons[i].bias - (eta*delOut[i]*1)
                    
                # for hidden layer
                for i in range(0,5):
                    for j in range(0,2):
                        hiddenNeurons[i].input[j] = hiddenNeurons[i].input[j] - (eta*delHidden[i]*float(inputVals[sampleNum][j]))
                    hiddenNeurons[i].bias = hiddenNeurons[i].bias - (eta*delHidden[i]*1)
               
            # Append Sum of Squared errors to list for plotting
            sseList.append(sumSqErrors)
            
            if k == 9:
                print_weights(outTen,hiddenNeurons,outputNeurons)
            if k == 99:
                print_weights(outHundred,hiddenNeurons,outputNeurons)
            if k == 999:
                print_weights(outThousand,hiddenNeurons,outputNeurons)
            if k == 9999:
                print_weights(outTenThousand,hiddenNeurons,outputNeurons)
        # '''
        
        # Plot Sum of Squared Errors
        plot(sseList)
        draw()
        show()
        
        
# Execute the main program. 
main()
