import numpy as np 
import scipy
import math
import scipy.misc

class Neuron_OnOff_KProjection_Metric():
    """Computing neuron on-off k-projection metric.

    Details on how this metric is defined is in Sec II.B of the paper 
    "Towards Dependability Metrics for Neural Networks" (https://arxiv.org/abs/1806.02338) 
    It is essentially applying the idea of combinatorial testing and covering arrays.

    Attributes:
        kValue: The constant k value for creating the coverage table
        numberOfNeuronsToTrack: number of neurons being monitored
        k_Activation_record: actual record 
    """


    def __init__(self, kValue, numberOfNeuronsToTrack):
            
        if not ((kValue == 1 or kValue == 2) or kValue == 3):
            raise NotImplementedError('for k-projection coverage where k > 3, it is not supported')
            
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
        elif kValue == 3:
            for neuronIndexI in range(numberOfNeuronsToTrack):
                for neuronIndexJ in range(neuronIndexI+1, numberOfNeuronsToTrack):
                    for neuronIndexK in range(neuronIndexJ+1, numberOfNeuronsToTrack):
                        self.k_Activation_record["N" + str(neuronIndexI) +"_"+ "N"+ str(neuronIndexJ)+"_"+ "N"+ str(neuronIndexK)] = set()
                 
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
        elif self.kValue == 3: 
            for exampleIndex in range(neuronValuesNp.shape[0]):                
                constraint = ''
                for neuronIndexI in range(self.numberOfNeuronsToTrack) :
                    if (ivabs[exampleIndex,neuronIndexI] >0):
                        for neuronIndexJ in range(neuronIndexI+1, self.numberOfNeuronsToTrack):
                            if(ivabs[exampleIndex,neuronIndexJ]  >0) : 
                                for neuronIndexK in range(neuronIndexJ+1, self.numberOfNeuronsToTrack):
                                    if(ivabs[exampleIndex,neuronIndexK]  >0) :
                                        self.k_Activation_record["N" + str(neuronIndexI) +"_"+ "N"+ str(neuronIndexJ)+"_"+ "N"+ str(neuronIndexK)].add("111")                                
                                    else:
                                        self.k_Activation_record["N" + str(neuronIndexI) +"_"+ "N"+ str(neuronIndexJ)+"_"+ "N"+ str(neuronIndexK)].add("110")
                            else:
                                for neuronIndexK in range(neuronIndexJ+1, self.numberOfNeuronsToTrack):
                                    if(ivabs[exampleIndex,neuronIndexK]  >0) :
                                        self.k_Activation_record["N" + str(neuronIndexI) +"_"+ "N"+ str(neuronIndexJ)+"_"+ "N"+ str(neuronIndexK)].add("101")                                
                                    else:
                                        self.k_Activation_record["N" + str(neuronIndexI) +"_"+ "N"+ str(neuronIndexJ)+"_"+ "N"+ str(neuronIndexK)].add("100")
                    else : 
                        for neuronIndexJ in range(neuronIndexI+1, self.numberOfNeuronsToTrack):
                            if(ivabs[exampleIndex,neuronIndexJ]  >0) : 
                                for neuronIndexK in range(neuronIndexJ+1, self.numberOfNeuronsToTrack):
                                    if(ivabs[exampleIndex,neuronIndexK]  >0) :
                                        self.k_Activation_record["N" + str(neuronIndexI) +"_"+ "N"+ str(neuronIndexJ)+"_"+ "N"+ str(neuronIndexK)].add("011")                                
                                    else:
                                        self.k_Activation_record["N" + str(neuronIndexI) +"_"+ "N"+ str(neuronIndexJ)+"_"+ "N"+ str(neuronIndexK)].add("010")
                            else: 
                                for neuronIndexK in range(neuronIndexJ+1, self.numberOfNeuronsToTrack):
                                    if(ivabs[exampleIndex,neuronIndexK]  >0) :
                                        self.k_Activation_record["N" + str(neuronIndexI) +"_"+ "N"+ str(neuronIndexJ)+"_"+ "N"+ str(neuronIndexK)].add("001")                                
                                    else:
                                        self.k_Activation_record["N" + str(neuronIndexI) +"_"+ "N"+ str(neuronIndexJ)+"_"+ "N"+ str(neuronIndexK)].add("000")                            
        else:
            print("Currently not supported")


    def dumpMetricState():
        return self.k_Activation_record, self.kValue, self.numberOfNeuronsToTrack

        
       