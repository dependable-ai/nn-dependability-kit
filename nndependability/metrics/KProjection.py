import numpy as np 
import scipy
import math
import scipy.misc

class Neuron_OnOff_KProjection_Metric():
    
    def __init__(self, kValue, numberOfNeuronsToTrack):
            
        if not (kValue == 1 or kValue == 2):
            raise Exception('for k-projection coverage where k > 2, it is not supported')
            
        self.kValue = kValue
        self.numberOfNeuronsToTrack = numberOfNeuronsToTrack        
        self.k_Activation_record = {}
        
        if kValue == 1:
            for neuronIndexI in range(numberOfNeuronsToTrack):
                self.k_Activation_record["N" + str(neuronIndexI)] = set()
        elif kValue == 2:
            for neuronIndexI in range(numberOfNeuronsToTrack):
                for neuronIndexJ in range(neuronIndexI+1, numberOfNeuronsToTrack):
                    self.k_Activation_record["N" + str(neuronIndexI) +"_"+ "N"+ str(neuronIndexJ)] = set()
        else:
            print("Currently not supported")

    

    
    def printMetricQuantity(self):
        denominator = scipy.misc.comb(self.numberOfNeuronsToTrack, self.kValue)* math.pow(2,  self.kValue)
        numerator = 0
        for key in self.k_Activation_record:
            numerator = numerator + len(self.k_Activation_record[key])
        print(str(self.kValue)+"-projection neuron on-off activation coverage:"+str(numerator) + "/"+str(int(denominator))+"="+str(numerator/denominator))
    
    
    def addInputs(self, neuronValuesNp):
        """ Process neuron values, and store the number of visited k-activation patterns.
             
            Keyword arguments:
            
        """
        
        if (not (type(neuronValuesNp) == np.ndarray)):
            raise TypeError('Input should be numpy array')
        
        mat = np.zeros(self.numberOfNeuronsToTrack)
        ivabs = np.greater(neuronValuesNp, mat)

        if self.kValue == 1:
            for exampleIndex in range(neuronValuesNp.shape[0]):                
                constraint = ''
                for neuronIndexI in range(self.numberOfNeuronsToTrack) :               
                        if (ivabs[exampleIndex,neuronIndexI] >0 ) :   
                            self.k_Activation_record["N" + str(neuronIndexI) ].add("1")                     
                        else:
                            self.k_Activation_record["N" + str(neuronIndexI) ].add("0") 
        elif self.kValue == 2: 
            for exampleIndex in range(neuronValuesNp.shape[0]):                
                constraint = ''
                for neuronIndexI in range(self.numberOfNeuronsToTrack) :
                    if (ivabs[exampleIndex,neuronIndexI] >0):
                        for neuronIndexJ in range(neuronIndexI+1, self.numberOfNeuronsToTrack):
                            if(ivabs[exampleIndex,neuronIndexJ]  >0) : 
                                self.k_Activation_record["N" + str(neuronIndexI) +"_"+ "N"+ str(neuronIndexJ)].add("11")
                            else:
                                self.k_Activation_record["N" + str(neuronIndexI) +"_"+ "N"+ str(neuronIndexJ)].add("10")
                    else : 
                        for neuronIndexJ in range(neuronIndexI+1, self.numberOfNeuronsToTrack):
                            if(ivabs[exampleIndex,neuronIndexJ]  >0) : 
                                self.k_Activation_record["N" + str(neuronIndexI) +"_"+ "N"+ str(neuronIndexJ)].add("01")
                            else:    
                                self.k_Activation_record["N" + str(neuronIndexI) +"_"+ "N"+ str(neuronIndexJ)].add("00")           
        else:
            print("Currently not supported")


    def dumpMetricState():
        return self.k_Activation_record, self.kValue, self.numberOfNeuronsToTrack

        
       