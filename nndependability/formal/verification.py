from . import dataflow
from . import octagon

import numpy as np
import math

def verify(inputMinBound, inputMaxBound, net, inputConstraints = [], riskProperty = [], bigM = 1000000.0):
    """Check if the risk property is not reachable for the neural network, or (if no risk property is provided) derive the min and max output bound for a network, 
       via viewing the network as a large MILP program.

    
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
    
  
                    
    return 
    