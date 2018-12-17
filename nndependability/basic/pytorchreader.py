import torch
import torch.nn as nn
import numpy as np

from . import neuralnet

def loadMlpFromPytorch(net, layersNamesToBeExtracted = [], layerTypes = []):
    """ Extract a pytorch model its MLP part to form a network suitable for formal analysis. 

        Usage: loadMlpFromPytorch(net, ["fc1", "fc2", "fc3", "fc4"], ["relu", "relu", "relu", "linear"])
         
        Keyword arguments:
        net -- network for analysis
        layersNamesToBeExtracted -- sequence of layers where the engine should follow and extract. Default value implies that the network is directly a MLP. 
        layerTypes -- for the layers, what are their type. Currently it can be relu, linear, BN, elu
        
    """
        
    # TODO(CH): Add a mechanism to detect if the structure of the network is MLP, with hidden layers using ReLU and last using linear.

    nNet = neuralnet.NeuralNetwork()
    
    if len(layersNamesToBeExtracted) == 0 :
        # Extract all layers
        
        lastLayerName = None
        # Derive the name of the last layer
        for name, param in net.named_parameters():
            if name.endswith(".bias"):
                lastLayerName = name[:-5]
        
        #print(lastLayerName)
        
        for name, param in net.named_parameters():
            if name.endswith(".bias"):
                print("Processing layer "+name.replace(".bias", ""))
                weights = None
                bias = None
                
                # Find the associated weights
                for name2, param2 in net.named_parameters():
                    if(name2 == name.replace(".bias", ".weight")):
                        weights = param2.detach().numpy()

                
                bias = param.detach().numpy()
                numberOfOutputs = weights.shape[0]
                numberOfInputs = weights.shape[1]

                if name[:-5] == lastLayerName:
                    # Should be linear
                    nNet.addLinearLayer(weights, bias)
                else:
                    # Should be ReLU
                    nNet.addReLULayer(weights, bias)
                      

        # Finally, do a sanity check if the dimension is right
        nNet.checkConsistency()
        return nNet
        
    else:
        for layerIndex in range(len(layersNamesToBeExtracted)):
            print("Processing layer "+layersNamesToBeExtracted[layerIndex])
            if layerTypes[layerIndex] != "BN":
            
                weights = None
                bias = None
                for name, param in net.named_parameters():
                    if name == layersNamesToBeExtracted[layerIndex]+".weight":
                        weights = param.detach().numpy()
                        
                for name, param in net.named_parameters():
                    if name == layersNamesToBeExtracted[layerIndex]+".bias":
                        bias = param.detach().numpy()
                        
                if layerTypes[layerIndex] == "linear":
                    nNet.addLinearLayer(weights, bias)
                elif  layerTypes[layerIndex] == "relu":  
                    nNet.addReLULayer(weights, bias)
                elif  layerTypes[layerIndex] == "elu":  
                    nNet.addELULayer(weights, bias)
                else:
                    raise Exception("Unknown layer type")
                    
            else:
                weights = None
                bias = None
                for name, param in net.named_parameters():
                    if name == layersNamesToBeExtracted[layerIndex]+".weight":
                        weights = param.detach().numpy()
                        
                for name, param in net.named_parameters():
                    if name == layersNamesToBeExtracted[layerIndex]+".bias":
                        bias = param.detach().numpy()
                
                # Moving mean and moving variance can't be accessed from named_parameters.
                # FIXME: Change the below code by accessing the moving mean and moving variance of the network.
                movingMean = 0
                movingVar = 1
                epsilon = 0.00001
                nNet.addBatchNormLayer(movingMean, movingVar, weights, bias)
                #raise NotImplementedError("Encountered BN whose implementation is still WiP")
                
                
        nNet.checkConsistency()
        return nNet

def loadMlpFromTensorFlow(net, layersNamesToBeExtracted, layerTypes):
    raise NotImplementedError
        
    