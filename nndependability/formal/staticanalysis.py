from . import dataflow
from . import octagon

import numpy as np
import math

def verify(inputMinBound, inputMaxBound, net, isUsingBox = True, inputConstraints = [], riskProperty = [], isAvoidQuadraticConstraints = False, bigM = 1000000.0):
    """Check if the risk property is not reachable for the neural network, or (if no risk property is provided) derive the min and max output bound for a network.

    This function takes a network and applies dataflow analysis (boxed domain, octagon domain) to consecutively compute the bound of each neuron.
    Finally, if risk property is provided, it checks if the computed bound implies the risk property. The computation is done by formulating 
    the bound computation as a MILP problem. 
    
    Currently, it only works for multi-layer perceptron network where 
    (1) all layers are fully connected,
    (2) all-but-output layers are with ReLU, and 
    (3) output layer has identity activation function (i.e., it is linear)
    
    Args:
        inputMinBound: array of input lower bounds
        inputMaxBound: array of input upper bounds
        net: neural network description 
        isUsingBox: true if apply only boxed abstraction; false if also apply octagon abstraction
        isAvoidQuadraticConstraints -- for octagon abstraction, whether to avoid generating full octagon constraints (quadratic to the number of neurons)
    
    Returns:    
    Raises:
    """

    # TODO(CH): Add a mechanism to detect if the structure of the network is MLP with FC.
    # TODO(CH): Refactor the function
    
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

    # Select an M value that is just large enough - internally the solver does mixed-integer-linear-programming to derive a smaller big-M value

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
                    minBound[layerIndex][i] = dataflow.deriveReLuOutputBound(False, layerIndex, weights[i], bias[i], numberOfInputs, i, bigM, minBound[layerIndex -1], maxBound[layerIndex -1], inputConstraints)
                    maxBound[layerIndex][i] = dataflow.deriveReLuOutputBound(True, layerIndex, weights[i], bias[i], numberOfInputs, i, bigM, minBound[layerIndex -1], maxBound[layerIndex -1], inputConstraints)        
                    
            elif(name.startswith(lastLayerName)):  
                # Output layer
                if len(riskProperty) == 0:
                    # Compute output bounds
                    for i in range(numberOfOutputs):
                        minBound[layerIndex][i] = dataflow.deriveLinearOutputBound(False, layerIndex, weights[i], bias[i], numberOfInputs, i, minBound[layerIndex -1], maxBound[layerIndex -1])
                        maxBound[layerIndex][i] = dataflow.deriveLinearOutputBound(True, layerIndex, weights[i], bias[i], numberOfInputs, i, minBound[layerIndex -1], maxBound[layerIndex -1])        
                else:
                    # Check if the property can be violated
                    isRiskReachable = dataflow.isRiskPropertyReachable(layerIndex, weights, bias, numberOfInputs, numberOfOutputs, minBound[layerIndex -1], maxBound[layerIndex -1], [], riskProperty)        
                    if(isRiskReachable == False):
                        print("Risk property is not reachable (using boxed abstraction)")
                        return [], []
                    else:  
                        print("Risk property may be reachable (using boxed abstraction)")
                    # raise Exception("Currently property are not supported")
            else:  
                # Intermediate layer                
                for i in range(numberOfOutputs):
                    minBound[layerIndex][i] = dataflow.deriveReLuOutputBound(False, layerIndex, weights[i], bias[i], numberOfInputs, i, bigM, minBound[layerIndex -1], maxBound[layerIndex -1])
                    maxBound[layerIndex][i] = dataflow.deriveReLuOutputBound(True, layerIndex, weights[i], bias[i], numberOfInputs, i, bigM, minBound[layerIndex -1], maxBound[layerIndex -1])        

            layerIndex = layerIndex + 1                        

    if(isUsingBox):
        print("Completed\n\n")
        return minBound[layerIndex-1], maxBound[layerIndex-1]
    

    # Also compute separately to derive the lower bound of the linear sum, in order to get a small bigM value
    minLinearBound = dict()
    minLinearBound[0] = inputMinBound

    layerIndex = 1
    for name, param in net.named_parameters():
        if name.endswith(".weight"):
            weights = param.detach().numpy()
            print("[Boxed abstraction] Processing layer "+name.replace(".weight", "") + "; derive linear sum lower bound")
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

            minLinearBound[layerIndex] = np.zeros(numberOfOutputs)

            for i in range(numberOfOutputs):
                minLinearBound[layerIndex][i] = dataflow.deriveLinearOutputBound(False, layerIndex, weights[i], bias[i], numberOfInputs, i, minBound[layerIndex -1], maxBound[layerIndex -1])

            layerIndex = layerIndex + 1     

    
    # Compute a smaller bigM value, in order to be fed into the Octagon analysis 
    bigM = 0
    for i in range(len(maxBound)):
        if bigM < np.max(np.absolute(maxBound[i])):
            bigM =  np.max(np.absolute(maxBound[i]))
    print("bigM under abs(max bound): "+ str(bigM))
    
    for i in range(len(minLinearBound)):
        if bigM < np.max(np.absolute(minLinearBound[i])):
            bigM =  np.max(np.absolute(minLinearBound[i]))
    print("bigM under abs(min bound): "+ str(bigM))
            
    if bigM == math.inf:
        # Just replace it by some big number (warning - this is a temporarily solution as there are bugs inside cbc LP solver)
        bigM = 30000000
            
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
                    
                if isAvoidQuadraticConstraints == False:
                    for i in range(numberOfOutputs):
                        print("\t["+str(i)+"]: ", end='')
                        for j in range(numberOfOutputs):
                            if (j>i):
                                if(j%5==0):
                                    print(str(j)+"," , end='')
                                octagonconstraint = []
                                minimumValue = octagon.deriveReLuOutputOctagonBound(False, layerIndex, weights, bias, numberOfInputs, i, j, bigM, minBound[layerIndex -1], maxBound[layerIndex -1], octagonBound[layerIndex -1], True, inputConstraintForThisLayer)
                                maximumValue = octagon.deriveReLuOutputOctagonBound(True, layerIndex, weights, bias, numberOfInputs, i, j, bigM,  minBound[layerIndex -1], maxBound[layerIndex -1], octagonBound[layerIndex -1], True, inputConstraintForThisLayer)        
                                vari = "v_"+str(layerIndex)+"_"+str(i)
                                varj = "v_"+str(layerIndex)+"_"+str(j)
                                octagonconstraint = [minimumValue, vari, -1, varj, maximumValue]
                                octagonBound[layerIndex].append(octagonconstraint)
                                
                                octagonconstraint = []
                                minimumValue = octagon.deriveReLuOutputOctagonBound(False, layerIndex, weights, bias, numberOfInputs, i, j, bigM, minBound[layerIndex -1], maxBound[layerIndex -1], octagonBound[layerIndex -1], False, inputConstraintForThisLayer)
                                maximumValue = octagon.deriveReLuOutputOctagonBound(True, layerIndex, weights, bias, numberOfInputs, i, j, bigM,  minBound[layerIndex -1], maxBound[layerIndex -1], octagonBound[layerIndex -1], False, inputConstraintForThisLayer)        
                                vari = "v_"+str(layerIndex)+"_"+str(i)
                                varj = "v_"+str(layerIndex)+"_"+str(j)
                                octagonconstraint = [minimumValue, vari, 1, varj, maximumValue]
                                octagonBound[layerIndex].append(octagonconstraint)
                                                                
                                
                        print()                
                else:
                    # Only create octagon difference constraints linear to the size of the neuron
                    # (i, i+1)
                    print("  (constraint shape x_{i} - x_{i+1}: ", end='')
                    for i in (range(numberOfOutputs  -1)):  
                        if(i%5==0):
                            print(str(i)+"," , end='')
                        octagonconstraint = []
                        minimumValue = octagon.deriveReLuOutputOctagonBound(False, layerIndex, weights, bias, numberOfInputs, i, i+1, bigM, minBound[layerIndex -1], maxBound[layerIndex -1], octagonBound[layerIndex -1], True, inputConstraintForThisLayer)
                        maximumValue = octagon.deriveReLuOutputOctagonBound(True, layerIndex, weights, bias, numberOfInputs, i, i+1, bigM,  minBound[layerIndex -1], maxBound[layerIndex -1], octagonBound[layerIndex -1], True, inputConstraintForThisLayer)        
                        vari = "v_"+str(layerIndex)+"_"+str(i)
                        varj = "v_"+str(layerIndex)+"_"+str(i+1)
                        octagonconstraint = [minimumValue, vari, -1, varj, maximumValue]
                        octagonBound[layerIndex].append(octagonconstraint)
                                            
                    print() 
                    print("  (constraint shape x_{i} + x_{i+1}: ", end='')
                    for i in (range(numberOfOutputs  -1)):  
                        if(i%5==0):
                            print(str(i)+"," , end='')
                        octagonconstraint = []
                        minimumValue = octagon.deriveReLuOutputOctagonBound(False, layerIndex, weights, bias, numberOfInputs, i, i+1, bigM, minBound[layerIndex -1], maxBound[layerIndex -1], octagonBound[layerIndex -1], False, inputConstraintForThisLayer)
                        maximumValue = octagon.deriveReLuOutputOctagonBound(True, layerIndex, weights, bias, numberOfInputs, i, i+1, bigM,  minBound[layerIndex -1], maxBound[layerIndex -1], octagonBound[layerIndex -1], False, inputConstraintForThisLayer)        
                        vari = "v_"+str(layerIndex)+"_"+str(i)
                        varj = "v_"+str(layerIndex)+"_"+str(i+1)
                        octagonconstraint = [minimumValue, vari, 1, varj, maximumValue]
                        octagonBound[layerIndex].append(octagonconstraint)    
                        
                    print()                      
                    # (i, i+ N/2)  
                    print("  (constraint shape x_{i} - x_{i+(N/2)}: ", end='')                    
                    half = (int(math.ceil(numberOfOutputs)/2))
                    for i in range(half):   
                        if(i%5==0):
                            print(str(i)+"," , end='')                     
                        octagonconstraint = []
                        minimumValue = octagon.deriveReLuOutputOctagonBound(False, layerIndex, weights, bias, numberOfInputs, i, i+half, bigM, minBound[layerIndex -1], maxBound[layerIndex -1], octagonBound[layerIndex -1], True, inputConstraintForThisLayer)
                        maximumValue = octagon.deriveReLuOutputOctagonBound(True, layerIndex, weights, bias, numberOfInputs, i, i+half, bigM,  minBound[layerIndex -1], maxBound[layerIndex -1], octagonBound[layerIndex -1], True, inputConstraintForThisLayer)        
                        vari = "v_"+str(layerIndex)+"_"+str(i)
                        varj = "v_"+str(layerIndex)+"_"+str(i+half)
                        octagonconstraint = [minimumValue, vari, -1, varj, maximumValue]
                        octagonBound[layerIndex].append(octagonconstraint)
                                            
                    print()  
                    print("  (constraint shape x_{i} + x_{i+(N/2)}: ", end='')                    
                    half = (int(math.ceil(numberOfOutputs)/2))
                    for i in range(half):   
                        if(i%5==0):
                            print(str(i)+"," , end='')                     
                        octagonconstraint = []
                        minimumValue = octagon.deriveReLuOutputOctagonBound(False, layerIndex, weights, bias, numberOfInputs, i, i+half, bigM, minBound[layerIndex -1], maxBound[layerIndex -1], octagonBound[layerIndex -1], False, inputConstraintForThisLayer)
                        maximumValue = octagon.deriveReLuOutputOctagonBound(True, layerIndex, weights, bias, numberOfInputs, i, i+half, bigM,  minBound[layerIndex -1], maxBound[layerIndex -1], octagonBound[layerIndex -1], False, inputConstraintForThisLayer)        
                        vari = "v_"+str(layerIndex)+"_"+str(i)
                        varj = "v_"+str(layerIndex)+"_"+str(i+half)
                        octagonconstraint = [minimumValue, vari, 1, varj, maximumValue]
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
                        minBoundOctagon[i] = dataflow.deriveLinearOutputBound(False, layerIndex, weights[i], bias[i], numberOfInputs, i, minBound[layerIndex -1], maxBound[layerIndex -1], octagonBound[layerIndex -1])
                        maxBoundOctagon[i] = dataflow.deriveLinearOutputBound(True, layerIndex, weights[i], bias[i], numberOfInputs, i, minBound[layerIndex -1], maxBound[layerIndex -1], octagonBound[layerIndex -1])        
                    return minBoundOctagon, maxBoundOctagon
                else:
                    # Check if the property can be violated
                    isRiskReachable = dataflow.isRiskPropertyReachable(layerIndex, weights, bias, numberOfInputs, numberOfOutputs, minBound[layerIndex -1], maxBound[layerIndex -1], octagonBound[layerIndex -1], riskProperty)        
                    if(isRiskReachable == False):
                        print("Risk property is not reachable (using octagon abstraction)")
                        return [], []    
                    else:
                        print("Risk property may be reachable (using octagon abstraction)")
                        return [], []                          
            layerIndex = layerIndex + 1
                    
    return 
    