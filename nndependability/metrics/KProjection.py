import numpy as np 
import scipy
import math


class Neuron_OnOff_KProjection_Metric():
    
    def __init__(self, kValue, numberOfNeuronsToTrack):
            
        if not (kValue == 2):
            raise Exception('for k-projection coverage where k != 2, it is not supported')
            
        self.kValue = kValue
        self.numberOfNeuronsToTrack = numberOfNeuronsToTrack        
        self.k_Activation_record = {}
        
        for neuronIndexI in range(numberOfNeuronsToTrack):
            for neuronIndexJ in range(neuronIndexI+1, numberOfNeuronsToTrack):
                self.k_Activation_record["N" + str(neuronIndexI) + "N"+ str(neuronIndexJ)] = set()
    

    
    def printMetricQuantity(self):
        denominator = scipy.misc.comb(self.numberOfNeuronsToTrack, self.kValue)* math.pow(2,  self.kValue)
        numerator = 0
        for key in self.k_Activation_record:
            numerator = numerator + len(self.k_Activation_record[key])
        print(str(self.kValue)+"-projection neuron on-off activation coverage:"+str(numerator) + "/"+str(int(denominator))+"="+str(numerator/denominator))
    
    
    def addInputs(self, neuronValues):
        """ Process neuron values, and store the number of visited k-activation patterns.
             
            Keyword arguments:
            
        """
        
        mat = np.zeros(self.numberOfNeuronsToTrack)
        ivabs = np.greater(neuronValues, mat)

        for exampleIndex in range(neuronValues.shape[0]):                
            constraint = ''
            for neuronIndexI in range(self.numberOfNeuronsToTrack) :
                for neuronIndexJ in range(neuronIndexI+1, self.numberOfNeuronsToTrack):
                    if (ivabs[exampleIndex,neuronIndexI] >0 and ivabs[exampleIndex,neuronIndexJ]  >0) :                        
                        # self.k_Activation_record["N" + str(neuronIndexI) + "N"+ str(neuronIndexJ)].add("TT")
                        self.k_Activation_record["N" + str(neuronIndexI) + "N"+ str(neuronIndexJ)].add(3)
                    elif (ivabs[exampleIndex,neuronIndexI] >0):
                        # self.k_Activation_record["N" + str(neuronIndexI) + "N"+ str(neuronIndexJ)].add("TF")
                        self.k_Activation_record["N" + str(neuronIndexI) + "N"+ str(neuronIndexJ)].add(2)
                    elif (ivabs[exampleIndex,neuronIndexJ] >0):
                        # self.k_Activation_record["N" + str(neuronIndexI) + "N"+ str(neuronIndexJ)].add("FT")
                        self.k_Activation_record["N" + str(neuronIndexI) + "N"+ str(neuronIndexJ)].add(1)
                    else:
                        # self.k_Activation_record["N" + str(neuronIndexI) + "N"+ str(neuronIndexJ)].add("FF")
                        self.k_Activation_record["N" + str(neuronIndexI) + "N"+ str(neuronIndexJ)].add(0)

        

        
       