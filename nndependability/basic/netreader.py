import torch
import torch.nn as nn

from . import neuralnet

def loadMlpFromPytorch(net, isUsingCompleteNet = True, layersNamesToBeExtracted = [], layerTypes = []):
        
    # TODO(CH): Add a mechanism to detect if the structure of the network is MLP, with hidden layers using ReLU and last using linear.

    nNet = neuralnet.NeuralNetwork()
    
    if isUsingCompleteNet == True:
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
        raise NotImplementedError("Currently the net should be MLP with ReLU + the last layer being linear")


def loadMlpFromTensorFlow(net, isUsingCompleteNet = True, layersNamesToBeExtracted = [], layerTypes = []):
    raise NotImplementedError
        
    