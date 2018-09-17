from dd.autoref import BDD
import numpy as np 
from . import monitor

class NAP_Monitor(monitor.BddMonitor):
    """Monitor implementing neuron on-off activation pattern, where on (>0) is set to 1 and off (<= 0) is set to 0

    Attributes:
        numberOfClasses  The number of classes to be classified.
        numberOfNeurons  The number of neurons to be monitored.
        coveredPatterns  Map for mapping each class with its associated monitor
        
    """    
    def __init__(self, numberOfClasses, numberOfNeurons, omittedNeuronIndices = {}):
    
        monitor.BddMonitor.__init__(self, numberOfClasses, numberOfNeurons, omittedNeuronIndices)
       
        
        # Create BDD variables
        for i in range(numberOfNeurons):
            self.bdd.add_var("x"+str(i))
    
        
    def isPatternContained(self, neuronValuesNp, classIndex):
        """ Check if the vector of neuronValues is within the monitor[classIndex]
             
            Keyword arguments:
            neuronValues -- 1D vector of numerical neuron values, with shape being this.numberOfNeurons
            classIndex -- the class to be examined
        """     
        
        if not (type(neuronValuesNp) == np.ndarray):
            raise TypeError('Input should be numpy array')
        
        zero = np.zeros(neuronValuesNp.shape)
        neuronOnOffPattern = np.greater(neuronValuesNp, zero)
        return self.isOnOffPatternContained(neuronOnOffPattern, classIndex)
        
    
    def addAllNeuronPatternsToClass(self, neuronValuesNp, predictedNp, labelsNp, classToKeep):
        """ Process neuron values, and add the processed pattern to the set of visited patterns 
             of a given classification.
             
            Keyword arguments:
            hammingDistance -- distance for added patterns and the original pattern created from the training
            classToKeep -- class index where one only wants to perform analysis; use -1 if one wants to process all classes
        """
        
        if (not (type(neuronValuesNp) == np.ndarray)) or (not (type(predictedNp) == np.ndarray)) or (not (type(labelsNp) == np.ndarray)):
            raise TypeError('Input should be numpy array')
        
        mat = np.zeros(neuronValuesNp.shape)
        abs = np.greater(neuronValuesNp, mat)
        for exampleIndex in range(neuronValuesNp.shape[0]): 
            if classToKeep == -1:
                if(predictedNp[exampleIndex] == labelsNp[exampleIndex]):
                    self.addOnOffPatternToClass(abs[exampleIndex,:], predictedNp[exampleIndex])
            else:
                if(predictedNp[exampleIndex] == labelsNp[exampleIndex] and labelsNp[exampleIndex] == classToKeep):
                    self.addOnOffPatternToClass(abs[exampleIndex,:], predictedNp[exampleIndex])
        return True

        
    
    def isOnOffPatternContained(self, neuronOnOffPattern, classIndex):
        """ Check if the provided neuron activation pattern has been visited.
             
            Keyword arguments:
            neuronOnOffPattern -- 1D vector of binary neuron values, with shape being this.numberOfNeurons
            classIndex -- the class to be examined
        """
        
        # Basic data sanity check
        if not (len(neuronOnOffPattern) == self.numberOfNeurons):            
            raise IndexError('Neuron pattern size is not the same as the number of neurons being monitored')
        
        # Prepare BDD constraint
        constraint = ''
        for i in range(self.numberOfNeurons) :
            if neuronOnOffPattern[i] or ((classIndex in self.omittedNeuronIndices) and (i in self.omittedNeuronIndices[classIndex])) :
                if i == 0: 
                    constraint = constraint + " x"+str(i)
                else: 
                    constraint = constraint + " & x"+str(i)
            else:
                if i == 0: 
                    constraint = constraint + " !x"+str(i)
                else: 
                    constraint = constraint + " & !x"+str(i)
        if (self.coveredPatterns[classIndex] & self.bdd.add_expr(constraint)) == self.bdd.false :            
            return False
        else:
            return True

 
    def addOnOffPatternToClass(self, neuronOnOffPattern, classIndex):
        """ Add a single neuron activation pattern to the set of visited patterns of a given classification.
   
        """
        
        # Basic data sanity check
        if not (len(neuronOnOffPattern) == self.numberOfNeurons):            
            raise IndexError('Neuron pattern size is not the same as the number of neurons being monitored')
        
        # Prepare BDD constraint
        constraint = ''
        for i in range(self.numberOfNeurons) :
            if neuronOnOffPattern[i] or ((classIndex in self.omittedNeuronIndices) and (i in self.omittedNeuronIndices[classIndex])):
                if i == 0: 
                    constraint = constraint + " x"+str(i)
                else: 
                    constraint = constraint + " & x"+str(i)
            else:
                if i == 0: 
                    constraint = constraint + " !x"+str(i)
                else: 
                    constraint = constraint + " & !x"+str(i)
        
        # Add into BDD
        self.coveredPatterns[classIndex] = self.coveredPatterns[classIndex] | self.bdd.add_expr(constraint)
      
        return True  
        
        
    def enlargeSetByOneBitFluctuation(self, classIndex = -1):
        """ Enlarge the monitor by adding all elements whose Hamming distance is 1 to the element, which can be done by bdd.exists

        """   
        
        if classIndex == -1:
            # Apply on all classifier
            enlargedPatterns = {}
            for i in range(self.numberOfClasses):
                enlargedPatterns[i] = self.coveredPatterns[i]
                # For each variable
                for j in range(self.numberOfNeurons):
                    enlargedPatterns[i] = enlargedPatterns[i] | self.bdd.exist(["x"+str(j)], self.coveredPatterns[i])
                self.coveredPatterns[i] = enlargedPatterns[i]        
        else :
            # Apply on specific classifier
            enlargedPatterns = self.coveredPatterns[classIndex]
            # For each variable
            for i in range(self.numberOfNeurons):
                if ((classIndex not in self.omittedNeuronIndices) or ((classIndex in self.omittedNeuronIndices) and (i not in self.omittedNeuronIndices[classIndex]))):
                    enlargedPatterns = enlargedPatterns | self.bdd.exist(["x"+str(i)], self.coveredPatterns[classIndex])
            self.coveredPatterns[classIndex] = enlargedPatterns   


    def saveToFile(self, fileName):
        """ Store the monitor based on the specified file name.
        """   
        if not (type(fileName) is string) : 
            raise TypeError("Please provide a fileName in string")
        
        self.bdd.dump(fileName) 