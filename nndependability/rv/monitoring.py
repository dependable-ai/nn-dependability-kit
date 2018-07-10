from dd.autoref import BDD
import numpy as np 



class NAP_Monitor():
    
    def __init__(self, numberOfClasses, numberOfNeurons):
    
        self.__bdd = BDD()
        
        self.numberOfClasses = numberOfClasses
        self.numberOfNeurons = numberOfNeurons
        
        self.coveredPatterns = {}
        
        # For each classified class, create a BDD to be monitored
        for i in range(numberOfClasses):
            self.coveredPatterns[i] = self.__bdd.false
        
        # Create BDD variables
        for i in range(numberOfNeurons):
            self.__bdd.add_var("x"+str(i))
    
    
    def isOnOffPatternContained(self, neuronPattern, classIndex):
        """ Check if the provided neuron activation pattern has been visited.
        """
        
        # Basic data sanity check
        if not (len(neuronPattern) == self.numberOfNeurons):            
            raise IndexError('Neuron pattern size is not the same as the number of neurons being monitored')
        
        # Prepare BDD constraint
        constraint = ''
        for i in range(self.numberOfNeurons) :
            if neuronPattern[i] :
                if i == 0: 
                    constraint = constraint + " x"+str(i)
                else: 
                    constraint = constraint + " & x"+str(i)
            else:
                if i == 0: 
                    constraint = constraint + " !x"+str(i)
                else: 
                    constraint = constraint + " & !x"+str(i)
        if (self.coveredPatterns[classIndex] & self.__bdd.add_expr(constraint)) == self.__bdd.false :            
            return False
        else:
            return True

        


    def addOnOffPatternNeighborToClass(self, neuronPattern, classIndex,  hammingDistance):
        """ Add the neuron activation pattern (on-off), together with all patterns less or equal to hammingDistance, to the set of visited patterns of a given classification.
        """
        
        # Basic data sanity check
        if not (len(neuronPattern) == self.numberOfNeurons):            
            raise IndexError('Neuron pattern size is not the same as the number of neurons being monitored')
        if hammingDistance == 0:
            self.addOnOffPatternToClass(neuronPattern, classIndex)
        elif hammingDistance == 1:
            for i in range(self.numberOfNeurons) :
                # Perform deep copy
                patternChange1Bit = np.empty_like (neuronPattern)
                patternChange1Bit[:] = neuronPattern
                # Flip the bit
                patternChange1Bit[i] = not neuronPattern[i]
                self.addOnOffPatternToClass(patternChange1Bit, classIndex)
        elif hammingDistance == 2:
            # First create those with distance 1
            self.addOnOffPatternNeighborToClass(neuronPattern, classIndex,  1)
            # Then create those with distance 2
            for i in range(self.numberOfNeurons) :
                for j in range(i+1, self.numberOfNeurons) :
                    # Perform deep copy
                    patternChange1Bit = np.empty_like (neuronPattern)
                    patternChange1Bit[:] = neuronPattern
                    # Flip the bit
                    patternChange1Bit[i] = not neuronPattern[i]
                    patternChange1Bit[j] = not neuronPattern[j]
                    self.addOnOffPatternToClass(patternChange1Bit, classIndex)
        else:
            raise Error('Currently Hamming distance > 2 is not supported')
        
        return True  
 
    def addOnOffPatternToClass(self, neuronPattern, classIndex):
        """ Add a single neuron activation pattern to the set of visited patterns of a given classification.
   
        """
        
        # Basic data sanity check
        if not (len(neuronPattern) == self.numberOfNeurons):            
            raise IndexError('Neuron pattern size is not the same as the number of neurons being monitored')
        
        # Prepare BDD constraint
        constraint = ''
        for i in range(self.numberOfNeurons) :
            if neuronPattern[i] :
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
        self.coveredPatterns[classIndex] = self.coveredPatterns[classIndex] | self.__bdd.add_expr(constraint)
      
        return True  
 
    def addAllNeuronPatternsToClass(self, neuronValues, predicted, labels, hammingDistance, classToKeep):
        """ Process neuron values, and add the processed pattern to the set of visited patterns 
             of a given classification.
             
            Keyword arguments:
            hammingDistance -- distance for added patterns and the original pattern created from the training
            classToKeep -- class index where one only wants to perform analysis; use -1 if one wants to process all classes
        """
        
        predictedNp = predicted.numpy()
        iv = neuronValues.numpy()
        mat = np.zeros(neuronValues.shape)
        ivabs = np.greater(iv, mat)
        for exampleIndex in range(iv.shape[0]): 
            if classToKeep == -1:
                if(predicted[exampleIndex] == labels[exampleIndex]):
                    self.addOnOffPatternNeighborToClass(ivabs[exampleIndex,:], predicted.numpy()[exampleIndex], hammingDistance)
            else:
                if(predicted[exampleIndex] == labels[exampleIndex] and labels[exampleIndex] == classToKeep):
                    self.addOnOffPatternNeighborToClass(ivabs[exampleIndex,:], predicted.numpy()[exampleIndex], hammingDistance)
                    
        return True

        

        
        
    def saveToFile(self, fileName):
        """ Store the monitor based on the specified file name.
        """   
        if not (type(fileName) is string) : 
            raise TypeError("Please provide a fileName in string")
        
        self.__bdd.dump(fileName)