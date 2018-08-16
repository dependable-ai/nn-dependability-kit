from . import dataflow
from . import octagon


import numpy as np

def verify(inputMinBound, inputMaxBound, net, isUsingBox = True, inputConstraints = [], riskProperty = [], bigM = 1000000.0 ):
    
    lastLayerName = None
    # Derive the name of the last layer
    for name, param in net.named_parameters():
        if name.endswith(".bias"):
            lastLayerName = name[:-5]
    
    #print(lastLayerName)
    
    minBound = dict()
    maxBound = dict()

    minBound[0] = inputMinBound
    maxBound[0] = inputMaxBound

    # Select an M value that is just large - internally the solver does mixed-integer-linear-programming to derive a smaller big-M value

    layerIndex = 1
    firstLayer = True

    for name, param in net.named_parameters():
        if name.endswith(".weight"):
            weights = param.detach().numpy()
            print("[Boxed abstraction] Processing layer "+name.replace(".weight", ""))
        elif name.endswith(".bias"):
            weights = None
            bias = None
            
            # Find the associated weights
            for name2, param2 in net.named_parameters():
                if(name2 == name.replace(".bias", ".weight")):
                    weights = param2.detach().numpy()

            
            bias = param.detach().numpy()
            numberOfOutputs = weights.shape[0]
            numberOfInputs = weights.shape[1]

            minBound[layerIndex] = np.zeros(numberOfOutputs)
            maxBound[layerIndex] = np.zeros(numberOfOutputs)

            if firstLayer:
                # Input layer
                firstLayer = False
                # Add additional input constraints to derive the bound
                for i in range(numberOfOutputs):
                    minBound[layerIndex][i] = dataflow.deriveReLuOutputBound(False, layerIndex, weights[i], bias[i], numberOfInputs, bigM, i, minBound[layerIndex -1], maxBound[layerIndex -1], inputConstraints)
                    maxBound[layerIndex][i] = dataflow.deriveReLuOutputBound(True, layerIndex, weights[i], bias[i], numberOfInputs, bigM, i, minBound[layerIndex -1], maxBound[layerIndex -1], inputConstraints)        
                    
            elif(name.startswith(lastLayerName)):  
                # Output layer
                if len(riskProperty) == 0:
                    # Compute output bounds
                    for i in range(numberOfOutputs):
                        minBound[layerIndex][i] = dataflow.deriveLinearOutputBound(False, layerIndex, weights[i], bias[i], numberOfInputs, bigM, i, minBound[layerIndex -1], maxBound[layerIndex -1])
                        maxBound[layerIndex][i] = dataflow.deriveLinearOutputBound(True, layerIndex, weights[i], bias[i], numberOfInputs, bigM, i, minBound[layerIndex -1], maxBound[layerIndex -1])        
                else:
                    # Check if the property can be violated
                    isRiskReachable = dataflow.isRiskPropertyReachable(True, layerIndex, weights, bias, numberOfInputs, numberOfOutputs, bigM, minBound[layerIndex -1], maxBound[layerIndex -1], [], riskProperty)        
                    if(isRiskReachable == False):
                        print("Risk property is not reachable (using boxed abstraction)")
                        return [], []
                    # raise Error("Currently property are not supported")
            else:  
                # Intermediate layer                
                for i in range(numberOfOutputs):
                    minBound[layerIndex][i] = dataflow.deriveReLuOutputBound(False, layerIndex, weights[i], bias[i], numberOfInputs, bigM, i, minBound[layerIndex -1], maxBound[layerIndex -1])
                    maxBound[layerIndex][i] = dataflow.deriveReLuOutputBound(True, layerIndex, weights[i], bias[i], numberOfInputs, bigM, i, minBound[layerIndex -1], maxBound[layerIndex -1])        

            layerIndex = layerIndex + 1                        

    if(isUsingBox):
        print("Completed\n\n")
        return minBound[layerIndex-1], maxBound[layerIndex-1]
    
    
    # Compute a smaller bigM value, in order to be fed into the Octagon analysis 
    bigM = 0
    for i in range(len(maxBound)):
        if bigM < np.max(maxBound[i]):
            bigM =  np.max(maxBound[i])
    
    # The user requested to perform analysis over the network with Octagon abstraction
    octagonBound = dict()
    octagonBound[0] = []
    layerIndex = 1
    firstLayer = True
    
    for name, param in net.named_parameters():
        if name.endswith(".weight"):
            print("[Octagon abstraction] Processing layer "+name.replace(".weight", ""))
        elif name.endswith(".bias"):
            weights = None
            bias = None
            
            # Find the associated weights
            for name2, param2 in net.named_parameters():
                if(name2 == name.replace(".bias", ".weight")):
                    weights = param2.detach().numpy()
                    
            bias = param.detach().numpy()
            numberOfOutputs = weights.shape[0]
            numberOfInputs = weights.shape[1]

            octagonBound[layerIndex] = []
 
            if(name.startswith(lastLayerName) == False):  
                # Process first & intermediate layer
                inputConstraintForThisLayer = []
                if firstLayer:
                    firstLayer = False
                    # For the first layer, the input constraint is taken from parameters
                    inputConstraintForThisLayer = inputConstraints
                    
                
                for i in range(numberOfOutputs):
                    print("\t["+str(i)+"]: ", end='')
                    for j in range(numberOfOutputs):
                        if (j>i):
                            if(j%5==0):
                                print(str(j)+"," , end='')
                            octagonconstraint = []

                            minimumValue = octagon.deriveReLuOutputDifferenceBound(False, layerIndex, weights, bias, numberOfInputs, i, j, bigM, minBound[layerIndex -1], maxBound[layerIndex -1], octagonBound[layerIndex -1], inputConstraintForThisLayer)
                            maximumValue = octagon.deriveReLuOutputDifferenceBound(True, layerIndex, weights, bias, numberOfInputs, i, j, bigM,  minBound[layerIndex -1], maxBound[layerIndex -1], octagonBound[layerIndex -1], inputConstraintForThisLayer)        
                            vari = "v_"+str(layerIndex)+"_"+str(i)
                            varj = "v_"+str(layerIndex)+"_"+str(j)
                            octagonconstraint = [minimumValue, vari, varj, maximumValue]
                            octagonBound[layerIndex].append(octagonconstraint)
                    print()
            print("Layer "+str(layerIndex) +": completed")
            print()
            layerIndex = layerIndex + 1        

            
    layerIndex = 1

    for name, param in net.named_parameters():
        if name.endswith(".bias"):
            weights = None
            bias = None        
            # Find the associated weights
            for name2, param2 in net.named_parameters():
                if(name2 == name.replace(".bias", ".weight")):
                    weights = param2.detach().numpy()
            bias = param.detach().numpy()
            numberOfOutputs = weights.shape[0]
            numberOfInputs = weights.shape[1]

            minBoundOctagon = np.zeros(numberOfOutputs)
            maxBoundOctagon = np.zeros(numberOfOutputs)
            
            if(name.startswith(lastLayerName)):  
            
                if len(riskProperty) == 0:
                    # Perform bound analysis over the final layer. 
                    for i in range(numberOfOutputs):
                        print (str(layerIndex) + "  "+str(i))
                        minBoundOctagon[i] = dataflow.deriveLinearOutputBound(False, layerIndex, weights[i], bias[i], numberOfInputs, bigM, i, minBound[layerIndex -1], maxBound[layerIndex -1], octagonBound[layerIndex -1])
                        maxBoundOctagon[i] = dataflow.deriveLinearOutputBound(True, layerIndex, weights[i], bias[i], numberOfInputs, bigM, i, minBound[layerIndex -1], maxBound[layerIndex -1], octagonBound[layerIndex -1])        
                    return minBoundOctagon, maxBoundOctagon
                else:
                    # Check if the property can be violated
                    isRiskReachable = dataflow.isRiskPropertyReachable(True, layerIndex, weights, bias, numberOfInputs, numberOfOutputs, bigM, minBound[layerIndex -1], maxBound[layerIndex -1], octagonBound[layerIndex -1], riskProperty)        
                    if(isRiskReachable == False):
                        print("Risk property is not reachable (using octagon abstraction)")
                        return [], []    
                    else:
                        print("Risk property may be reachable (using octagon abstraction)")
                        return [], []                          
            layerIndex = layerIndex + 1
                    
    return 
    