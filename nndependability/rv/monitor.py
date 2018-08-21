from dd.autoref import BDD
import numpy as np 

class BddMonitor():
    """Monitor implementing neuron interval pattern, with restrictions of using only two bits per neuron.

    Attributes:
        numberOfClasses  The number of classes to be classified.
        numberOfNeurons  The number of neurons to be monitored.
        coveredPatterns  Map for mapping each class with its associated monitor
        omittedNeuronIndices  Map for mapping each class with corresponding neuron indices to be omitted in the construction. If the indice i is omitted, variable x_i is always set to TRUE.
        bdd            Internal data structure for BDD access  
    """
    
    
    def __init__(self, numberOfClasses, numberOfNeurons, omittedNeuronIndices = {}):
    
        self.bdd = BDD()
        
        self.numberOfClasses = numberOfClasses
        self.numberOfNeurons = numberOfNeurons
        self.omittedNeuronIndices = omittedNeuronIndices 
        self.coveredPatterns = {}
        
        # For each classified class, create a BDD to be monitored
        for i in range(numberOfClasses):
            self.coveredPatterns[i] = self.bdd.false

            
            
      
    def isPatternContained(self, neuronValuesNp, classIndex):
        """ Check if the vector of neuronValues is within the monitor[classIndex]
             
            Keyword arguments:
            neuronValuesNp -- numpy 1D vector of neuron values, with shape being this.numberOfNeurons
            classIndex -- the class to be examined
        """       
        raise NotImplementedError("Please Implement this method")
    

    def addAllNeuronPatternsToClass(self, neuronValuesNp, predictedNp, labelsNp, classToKeep):
        """ Process neuron values, and add the processed pattern to the set of visited patterns of a given classification.
             
            Keyword arguments:
            neuronValuesNp -- numpy tensor of neuron values, with shape being (size of input or batch, this.numberOfNeurons)
            predictedNp  -- numpy result of prediction for the input or the batch
            labelsNp -- numpy ground truth of for the input or the batch
            classToKeep -- class index where one only wants to perform analysis; use -1 if one wants to process all classes
            classToKeep -- class index where one only wants to perform analysis; use -1 if one wants to process all classes
        """
        raise NotImplementedError("Please Implement this method")
    
    