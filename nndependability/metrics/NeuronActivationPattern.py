import matplotlib.pyplot as plt
import numpy as np 

class Neuron_Activation_Pattern_Metric():
    """Metric for summarizing neuron on-off activation patterns, where on (>0) is set to 1 and off (<= 0) is set to 0

    Attributes:
        numberOfClasses  The number of classes to be classified.
        numberOfNeurons  The number of neurons to be monitored.
        coveredPatterns  Map for mapping each class with its associated monitor
        
    """    
    def __init__(self, numberOfClasses, numberOfNeurons):
    
        self.numberOfClasses = numberOfClasses
        self.numberOfNeurons = numberOfNeurons
        
        self.coveredPatterns = {}
        
        # For each classified class, create a BDD to be monitored
        for i in range(numberOfClasses):
            self.coveredPatterns[i] = {}
            for j in range(numberOfNeurons):
                self.coveredPatterns[i][j] = 0
    
    
    def addAllNeuronPatternsToClass(self, neuronValuesNp, predictedNp, labelsNp):
        """ Process neuron values, and add the processed pattern to the set of visited patterns 
             of a given classification.
             
            Keyword arguments:
            hammingDistance -- distance for added patterns and the original pattern created from the training
        """
        
        if (not (type(neuronValuesNp) == np.ndarray)) or (not (type(predictedNp) == np.ndarray)) or (not (type(labelsNp) == np.ndarray)):
            raise TypeError('Input should be numpy array')
        
        mat = np.zeros(neuronValuesNp.shape)
        abs = np.greater(neuronValuesNp, mat)
        for exampleIndex in range(neuronValuesNp.shape[0]): 
            if(predictedNp[exampleIndex] == labelsNp[exampleIndex]):
                self.addInputs(abs[exampleIndex,:], predictedNp[exampleIndex])
        return True

 
    def addInputs(self, neuronOnOffPattern, classIndex):
        """ Add a single neuron activation pattern to the set of visited patterns of a given classification.
   
        """
        
        # Basic data sanity check
        if not (len(neuronOnOffPattern) == self.numberOfNeurons):            
            raise ValueError('Neuron pattern size is not the same as the number of neurons being monitored')
        

        numberOfOn = 0
        for i in range(self.numberOfNeurons) :
            if neuronOnOffPattern[i]:
                numberOfOn = numberOfOn +1

        
        # Add into 
        self.coveredPatterns[classIndex][numberOfOn] = self.coveredPatterns[classIndex][numberOfOn] +1 
      
        return True  
        

    def printMetricQuantity(self, classIndex):
        plt.bar(list(self.coveredPatterns[classIndex].keys()), self.coveredPatterns[classIndex].values(), color='g')
        plt.xlabel('Number of neurons activated')
        plt.ylabel('Frequency')
        plt.title('Neuron Activation Pattern Histogram for class '+str(classIndex) )
        plt.show()