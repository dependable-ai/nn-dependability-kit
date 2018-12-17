import numpy as np
import math


class NeuralNetwork():
    """Basic data structure for storing the neural network 1D for formal analysis.

    Attributes:
        numberOfInputs  The number of inputs to the network.

    """
    
    def __init__(self):
    
        self.layers = []        
      
    def addReLULayer(self, weights, bias):
        """ Add ReLU layer
             
            Keyword arguments:
            weights -- 
        """       
        layer = {}
        layer["type"] = "relu"
        layer["weights"] = weights
        layer["bias"] = bias
        self.layers.append(layer)
        return
        
    def addLinearLayer(self, weights, bias):       
        layer = {}
        layer["type"] = "linear"
        layer["weights"] = weights
        layer["bias"] = bias
        self.layers.append(layer)
        return

    def addBatchNormLayer(self, moving_mean, moving_variance, gamma, beta, epsilon = 0.001):       
        layer = {}
        layer["type"] = "BN"
        layer["moving_mean"] = moving_mean
        layer["moving_variance"] = moving_variance
        layer["gamma"] = gamma  
        layer["beta"] = beta 
        layer["epsilon"] = epsilon
        self.layers.append(layer)
        return
     
    def addELULayer(self, weights, bias, alpha=1):       
        layer = {}
        layer["type"] = "elu"
        layer["weights"] = weights
        layer["bias"] = bias
        layer["alpha"] = alpha
        self.layers.append(layer)    
        return

    def checkConsistency(self):
        """ Check if the dimension between two layers are correct
             
            Keyword arguments:
        """
        
        for layerIndex in range(len(self.layers)):

            if self.layers[layerIndex]["type"] == "BN":
                if not ( self.layers[layerIndex]["gamma"].shape[0] == self.layers[layerIndex]["beta"].shape[0]): 
                    raise Exception("Inconsistent dimension for parameters in BN layer "+str(layerIndex))
                    
                if(self.layers[layerIndex]["gamma"].shape[0] != self.layers[layerIndex-1]["weights"].shape[0] ):
                    print(self.layers[layerIndex]["gamma"].shape[0])
                    print(self.layers[layerIndex-1]["weights"].shape[0])
                    raise Exception("Inconsistent dimension between BN layer "+str(layerIndex)+" and its previous layer")
            else: 
                if layerIndex > 0:
                    # Check against previous layer
                    if(self.layers[layerIndex-1]["type"] == "BN"):
                        if self.layers[layerIndex]["weights"].shape[1] != self.layers[layerIndex-1]["beta"].shape[0]:
                            print(self.layers[layerIndex]["weights"].shape)
                            print(self.layers[layerIndex-1]["beta"].shape)
                            raise Exception("Inconsistent dimension between layer "+str(layerIndex)+" and its previous layer")
                    else: 
                        if self.layers[layerIndex]["weights"].shape[1] != self.layers[layerIndex-1]["weights"].shape[0]:
                            print(self.layers[layerIndex]["weights"].shape)
                            print(self.layers[layerIndex-1]["weights"].shape)
                            raise Exception("Inconsistent dimension between layer "+str(layerIndex)+" and its previous layer")
            
        print("Finish checking - dimension OK!")
        return 
                    
                    
