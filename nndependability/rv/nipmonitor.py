from dd.autoref import BDD
import numpy as np 
from . import monitor


class NIP_Monitor(monitor.BddMonitor):
    """Monitor implementing neuron interval pattern, with restrictions of using only two bits per neuron.

    Attributes:
        numberOfClasses  The number of classes to be classified.
        numberOfNeurons  The number of neurons to be monitored.
        coveredPatterns  Map for mapping each class with its associated monitor
        __threshold1     First threshold - values between (0, threahold1] is translated to 01
        __threshold2     Second threshold - values between (threahold1, threahold2] is translated to 10, and above is translated to 11
    """
    
    def __init__(self, numberOfClasses, numberOfNeurons, threshold1, threshold2, omittedNeuronIndices = {}):
    
        monitor.BddMonitor.__init__(self, numberOfClasses, numberOfNeurons, omittedNeuronIndices)
        
        # Create BDD variables
        for i in range(numberOfNeurons):
            self.bdd.add_var("x"+str(i)+"_0")
            self.bdd.add_var("x"+str(i)+"_1")
    
        # 
        self.__threshold1 = threshold1
        self.__threshold2 = threshold2
           
    def isPatternContained(self, neuronValuesNp, classIndex):
        """ Check if the vector of neuronValues is within the monitor[classIndex]
             
            Keyword arguments:
            neuronValues -- 1D vector of neuron values, with shape being this.numberOfNeurons
            classIndex -- the class to be examined
        """       
        if not (type(neuronValuesNp) == np.ndarray):
            raise TypeError('Input should be numpy array')
        
        thr0 = np.zeros(neuronValuesNp.shape)
        greaterThanTh2 = np.greater(neuronValuesNp, self.__threshold2)
        greaterThanTh1 = np.greater(neuronValuesNp, self.__threshold1)
        greaterThanTh0 = np.greater(neuronValuesNp, thr0)
        result = greaterThanTh2.astype(int) + greaterThanTh1.astype(int) + greaterThanTh0.astype(int)
        # Change the dimension from [1, numberOfNeurons] to [numberOfNeurons]
        result = result.squeeze(0)
        if (self.coveredPatterns[classIndex] & self.bdd.add_expr(self.prepareConstraint(result, classIndex))) == self.bdd.false :            
            return False
        else:
            return True
        
        
    def addAllNeuronPatternsToClass(self, neuronValuesNp, predictedNp, labelsNp, classToKeep):
        """ Process neuron values, and add the processed pattern to the set of visited patterns of a given classification.
             
            Keyword arguments:
            neuronValuesNp -- numpy tensor of neuron values, with shape being (size of input or batch, this.numberOfNeurons)
            predictedNp  -- numpy result of prediction for the input or the batch
            labelsNp -- numpy ground truth of for the input or the batch
            classToKeep -- class index where one only wants to perform analysis; use -1 if one wants to process all classes
        """
        if (not (type(neuronValuesNp) == np.ndarray)) or (not (type(predictedNp) == np.ndarray)) or (not (type(labelsNp) == np.ndarray)):
            raise TypeError('Input should be numpy array')                
        
        thr0 = np.zeros(neuronValuesNp.shape)
        # Create 2D array from existing 1D array
        thr2 = np.tile(self.__threshold2, (neuronValuesNp.shape[0],1))
        thr1 = np.tile(self.__threshold1, (neuronValuesNp.shape[0],1))
        
        greaterThanTh2 = np.greater(neuronValuesNp, thr2)
        greaterThanTh1 = np.greater(neuronValuesNp, thr1)
        greaterThanTh0 = np.greater(neuronValuesNp, thr0)
        
        result = greaterThanTh2.astype(int) + greaterThanTh1.astype(int) + greaterThanTh0.astype(int)
        for exampleIndex in range(neuronValuesNp.shape[0]): 
            if classToKeep == -1:
                if(predictedNp[exampleIndex] == labelsNp[exampleIndex]):
                    self.addDiscretePatternToClass(result[exampleIndex,:], predictedNp[exampleIndex])
            else:
                if(predictedNp[exampleIndex] == labelsNp[exampleIndex] and labelsNp[exampleIndex] == classToKeep):
                    self.addDiscretePatternToClass(result[exampleIndex,:], predictedNp[exampleIndex])
                    
        return True

    def prepareConstraint(self, neuronDiscretePattern, classIndex):
        # Prepare BDD constraint
        constraint = ''
        for i in range(self.numberOfNeurons) :
            if neuronDiscretePattern[i] == 3  or ((classIndex in self.omittedNeuronIndices) and (i in self.omittedNeuronIndices[classIndex])) :
                if i == 0: 
                    constraint = constraint + " x"+str(i)+"_0" + " & x"+str(i)+"_1"
                else: 
                    constraint = constraint + " & x"+str(i)+"_0" + " & x"+str(i)+"_1"                    
            elif neuronDiscretePattern[i] == 2:
                if i == 0: 
                    constraint = constraint + " !x"+str(i)+"_0" + " & x"+str(i)+"_1"
                else: 
                    constraint = constraint + " & !x"+str(i)+"_0" + " & x"+str(i)+"_1"   
            elif neuronDiscretePattern[i] == 1:
                if i == 0: 
                    constraint = constraint + " x"+str(i)+"_0" + " & !x"+str(i)+"_1"
                else: 
                    constraint = constraint + " & x"+str(i)+"_0" + " & !x"+str(i)+"_1"   
            elif neuronDiscretePattern[i] == 0:
                if i == 0: 
                    constraint = constraint + " !x"+str(i)+"_0" + " & !x"+str(i)+"_1"
                else: 
                    constraint = constraint + " & !x"+str(i)+"_0" + " & !x"+str(i)+"_1"   
            else:
                raise Error('Impossible to have pattern with values different from 0,1,2,3')
            
        return constraint
        

    def addDiscretePatternToClass(self, neuronDiscretePattern, classIndex):
        """ Add a single neuron activation pattern (value 0, 1, 2, 3) to the set of visited patterns of a given classification.
   
        """
        
        # Basic data sanity check
        if not (len(neuronDiscretePattern) == self.numberOfNeurons):            
            raise IndexError('Discrete pattern size is not the same as the number of neurons being monitored')
        
                   
        # Add into BDD
        self.coveredPatterns[classIndex] = self.coveredPatterns[classIndex] | self.bdd.add_expr(self.prepareConstraint(neuronDiscretePattern, classIndex))
      
        return True  
 
    def enlargeSetByOneBitFluctuation(self, classIndex):
        """ Enlarge the monitor by adding all elements whose Hamming distance is 1 to the element, which can be done by bdd.exists

            Notice that here it is only changing the encoding, so if 
                one originally contains 01, now it has 01, 00, 11
                one originally contains 00, now it has 00, 10, 01 --> Thus the expansion is not uniform
        
        """   
        
        if classIndex == -1:
            # Apply on all classifier
            enlargedPatterns = {}
            for i in range(self.numberOfClasses):
                enlargedPatterns[i] = self.coveredPatterns[i]
                # For each variable
                for j in range(self.numberOfNeurons):
                    enlargedPatterns[i] = enlargedPatterns[i] | self.bdd.exist(["x"+str(j)+"_0"], self.coveredPatterns[i])
                    enlargedPatterns[i] = enlargedPatterns[i] | self.bdd.exist(["x"+str(j)+"_1"], self.coveredPatterns[i])
                self.coveredPatterns[i] = enlargedPatterns[i]        
        else :
            # Apply on specific classifier
            enlargedPatterns = self.coveredPatterns[classIndex]
            # For each variable
            for i in range(self.numberOfNeurons):
                if ((classIndex not in self.omittedNeuronIndices) or ((classIndex in self.omittedNeuronIndices) and (i not in self.omittedNeuronIndices[classIndex]))):
                    enlargedPatterns = enlargedPatterns | self.bdd.exist(["x"+str(i)+"_0"], self.coveredPatterns[classIndex])
                    enlargedPatterns = enlargedPatterns | self.bdd.exist(["x"+str(i)+"_1"], self.coveredPatterns[classIndex])
            self.coveredPatterns[classIndex] = enlargedPatterns   

 
