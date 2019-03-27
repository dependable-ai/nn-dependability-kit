from . import dataflow
from . import octagon
from ..basic import pytorchreader
from ..basic import neuralnet

from pulp import *
import numpy as np
import math
import torch
import torch.nn as nn

def verify(inputMinBound, inputMaxBound, net, inputConstraints = [], riskProperty = [], isMaxBound = True, indexOutputNeuron = 0, bigM = 1000000.0, isUsingCplex = False):
    """Check if the risk property is not reachable for the neural network, or (if no risk property is provided) derive the min and max output bound for a network, 
       via viewing the network as a large MILP program.

    
    Currently, it only works for multi-layer perceptron network where 

    (1) all-but-output layers are with ReLU or BN,  
    (2) all ReLU and linear layers are fully connected, and 
    (3) output layer has identity activation function (i.e., it is linear)
    
    Warning: here a default big M of 1000000 is used. One can apply dataflow analysis to get a fine grained big M value, and it will result in some speed up.
    
    Args:
        inputMinBound: array of input lower bounds
        inputMaxBound: array of input upper bounds
        net: neural network description 
    
    Returns:    
    Raises:
    """

    if isinstance(net, nn.Module):
        # Translate from Pytorch to internal format
        net = pytorchreader.loadMlpFromPytorch(net)
    
    # Set up a new LP problem
    prob = None
    if isMaxBound: 
        prob = LpProblem("test1", LpMaximize)    
    else:
        prob = LpProblem("test1", LpMinimize)                

    
    # Prepare variables for the neural network verification problem
    variableDict = dict() 

    for layerIndex in range(1, len(net.layers)+1):
        numberOfOutputs = 0
        numberOfInputs = 0
        
        # in neuralnet.py for storing weights, the index starts with 0, so we need to do "layerIndex - 1"
        if ((net.layers[layerIndex-1]["type"] == "relu" or net.layers[layerIndex-1]["type"] == "elu") or net.layers[layerIndex-1]["type"] == "linear"): 
                numberOfOutputs = net.layers[layerIndex-1]["weights"].shape[0]
                numberOfInputs = net.layers[layerIndex-1]["weights"].shape[1]
        elif net.layers[layerIndex-1]["type"] == "BN":
            # Take the previous layer output (assume that it is ReLu or Elu) as its input dimension
            numberOfInputs = net.layers[layerIndex-2]["weights"].shape[0]
            numberOfOutputs = net.layers[layerIndex-2]["weights"].shape[0]
            
        # Add variables used for computation of neurons 
        for nout in range(numberOfOutputs):
            # Here im and v are useless, if the layer is BN
            variableDict["v_"+str(layerIndex)+"_"+str(nout)] =  LpVariable("v_"+str(layerIndex)+"_"+str(nout), lowBound=None, upBound=None , cat='Continuous')  
            variableDict["b_"+str(layerIndex)+"_"+str(nout)] = LpVariable("b_"+str(layerIndex)+"_"+str(nout), lowBound=0, upBound=1 , cat='Integer')
            variableDict["im_"+str(layerIndex)+"_"+str(nout)] =  LpVariable("im_"+str(layerIndex)+"_"+str(nout), lowBound=None, upBound=None , cat='Continuous')
  
    
    # Create input variables with specified bounds from minBound and maxBound
    numberOfInputs = net.layers[0]["weights"].shape[1]
    for nin in range(numberOfInputs):
        variableDict["v_"+str(0)+"_"+str(nin)] = LpVariable("v_"+str(0)+"_"+str(nin), lowBound=inputMinBound[nin], upBound=inputMaxBound[nin] , cat='Continuous') 
    

    # Add additional input constraints provided by the user    
    for inputConstr in inputConstraints:  
        try:
            boundConstraint = []
            for i in range(int(len(inputConstr)/2) - 1):
                boundConstraint.append((variableDict[inputConstr[2*(i+1)+1].replace("in", "v_0_")], inputConstr[2*(i+1)]))
            c = LpAffineExpression(boundConstraint)
            # Here be careful - if the sign is "<=" then one shall place ">="
            if inputConstr[1] == "<=":
                prob += c >= inputConstr[0]
            elif inputConstr[1] == "==":
                prob += c == inputConstr[0]
            elif inputConstr[1] == ">=":
                prob += c <= inputConstr[0]
            else:
                raise ValueError('Operators shall be <=, == or >=')                
            #print(c)
        except:
            print("Problem is processing input constraint: "+ str(inputConstr))
            #print("", end='')    
    


    # Create constraints via a layer wise fashion
    for layerIndex in range(1, len(net.layers)+1):
        print("Processing layer "+str(layerIndex))
        
        numberOfOutputs = 0
        numberOfInputs = 0
        
        # in neuralnet.py for storing weights, the index starts with 0, so we need to do "layerIndex - 1"
        if ((net.layers[layerIndex-1]["type"] == "relu" or net.layers[layerIndex-1]["type"] == "elu") or net.layers[layerIndex-1]["type"] == "linear"): 
            numberOfOutputs = net.layers[layerIndex-1]["weights"].shape[0]
            numberOfInputs = net.layers[layerIndex-1]["weights"].shape[1]
        elif net.layers[layerIndex-1]["type"] == "BN":
            # Take the previous layer output (assume that it is ReLu or Elu) as its input dimension
            numberOfInputs = net.layers[layerIndex-2]["weights"].shape[0]
            numberOfOutputs = net.layers[layerIndex-2]["weights"].shape[0]
        else:
            raise NotImplementedError("Currently layers beyond relu, elu, linear, BN are not supported")
        
        
        # Based on the structure of the layer, create corresponding constraints
        if net.layers[layerIndex-1]["type"] == "relu": 
            # in neuralnet.py for storing weights, the index starts with 0
            weights = net.layers[layerIndex-1]["weights"]
            bias = net.layers[layerIndex-1]["bias"]
            for nout in range(numberOfOutputs):
                # equality constraint: im = w1 in1 + .... + bias <==> -bias = -im + w1 in1 + ....   
                equalityConstraint = []
                equalityConstraint.append((variableDict["im_"+str(layerIndex)+"_"+str(nout)], -1))
                for nin in range(numberOfInputs):
                    equalityConstraint.append((variableDict["v_"+str(layerIndex-1)+"_"+str(nin)], weights[nout][nin].item()))
                c1 = LpAffineExpression(equalityConstraint)
                prob += c1 == -1*bias[nout], "relueq_"+str(layerIndex)+"_"+str(nout)
                
                # c11: v >= im  <==> im - v  <= 0   
                prob += variableDict["im_"+str(layerIndex)+"_"+str(nout)] - variableDict["v_"+str(layerIndex)+"_"+str(nout)] <= 0, "c_"+str(layerIndex)+"_"+str(nout)+"_11"
                # c12: v <= im + (1-b)M <==> v - im + M(b) <= M 
                prob += variableDict["v_"+str(layerIndex)+"_"+str(nout)] -variableDict["im_"+str(layerIndex)+"_"+str(nout)] + bigM*variableDict["b_"+str(layerIndex)+"_"+str(nout)] <= bigM, "c_"+str(layerIndex)+"_"+str(nout)+"_12"
                # c13: v <= bM <=>  M(b) - V >= 0
                prob += bigM*variableDict["b_"+str(layerIndex)+"_"+str(nout)] - variableDict["v_"+str(layerIndex)+"_"+str(nout)] >= 0, "c_"+str(layerIndex)+"_"+str(nout)+"_13"
                # c14: im + (1-b) M >= 0 <=> im - M(b) >= -M
                prob += variableDict["im_"+str(layerIndex)+"_"+str(nout)] - bigM*variableDict["b_"+str(layerIndex)+"_"+str(nout)] >= -1*bigM, "c_"+str(layerIndex)+"_"+str(nout)+"_14"   
                # c15: im  - b M <= 0 
                prob += variableDict["im_"+str(layerIndex)+"_"+str(nout)] - bigM*variableDict["b_"+str(layerIndex)+"_"+str(nout)] <= 0, "c_"+str(layerIndex)+"_"+str(nout)+"_15" 
                
        elif net.layers[layerIndex-1]["type"] == "BN": 

            moving_mean = net.layers[layerIndex-1]["moving_mean"]
            moving_variance = net.layers[layerIndex-1]["moving_variance"]
            epsilon = net.layers[layerIndex-1]["epsilon"]                    
            gamma = net.layers[layerIndex-1]["gamma"]
            beta = net.layers[layerIndex-1]["beta"]     
            
            for nout in range(numberOfOutputs):
                # z_norm = (z -  moving_mean) / (np.sqrt(moving_variance + epsilon))
                # z_tilda = (gamma * z_norm) + beta
                # In one formula:
                # z_tilda = [gamma / (np.sqrt(moving_variance + epsilon)] z + [gamma * (-1) * moving_mean / np.sqrt(moving_variance + epsilon) + beta]
                # Change z_tilda to v_out, z to v_in, and place the left hand side with constant
                # [gamma  * moving_mean / (np.sqrt(moving_variance + epsilon) + beta] = -1 * v_out + [gamma / (np.sqrt(moving_variance + epsilon)] * v_in
                equalityConstraint = []
                equalityConstraint.append((variableDict["v_"+str(layerIndex)+"_"+str(nout)], -1))
                equalityConstraint.append((variableDict["v_"+str(layerIndex-1)+"_"+str(nout)], gamma[nout] / (np.sqrt(moving_variance[nout] + epsilon))))
                c1 = LpAffineExpression(equalityConstraint)
                prob += c1 == (((gamma[nout] * moving_mean[nout]) / np.sqrt(moving_variance[nout] + epsilon)) + beta[nout]), "bneq_"+str(layerIndex)+"_"+str(nout)
                
        elif net.layers[layerIndex-1]["type"] == "linear":
            weights = net.layers[layerIndex-1]["weights"]
            bias = net.layers[layerIndex-1]["bias"]
            for nout in range(numberOfOutputs):        
                # equality constraint: vout = w1 in1 + .... + bias <==> -bias = -im + w1 in1 + ....   
                equalityConstraint = []
                equalityConstraint.append((variableDict["v_"+str(layerIndex)+"_"+str(nout)], -1))
                for nin in range(numberOfInputs):
                    equalityConstraint.append((variableDict["v_"+str(layerIndex-1)+"_"+str(nin)], weights[nout][nin].item()))
                c1 = LpAffineExpression(equalityConstraint)
                prob += c1 == -1*bias[nout], "lieq_"+str(layerIndex)+"_"+str(nout)
            
        else:
            raise NotImplementedError("Currently intermediate layers beyond ReLU, BN, and linear are not supported")    
    
    
    outputLayerIndex = len(net.layers)
    for riskConstraint in riskProperty:  
        try:
            boundConstraint = []
            for i in range(int(len(riskConstraint)/2) - 1):
                boundConstraint.append((variableDict[riskConstraint[2*(i+1)+1].replace("out", "v_"+str(outputLayerIndex)+"_")], riskConstraint[2*(i+1)]))
            c = LpAffineExpression(boundConstraint)
            # Here be careful - if the sign is "<=" then one shall place ">="
            if riskConstraint[1] == "<=":
                prob += c >= riskConstraint[0]
            elif riskConstraint[1] == "==":
                prob += c == riskConstraint[0]
            elif riskConstraint[1] == ">=":
                prob += c <= riskConstraint[0]
            else:
                raise ValueError('Operators shall be <=, == or >=')
                
            #print(c)
        except:
            print("Problem is processing risk Constraint: "+ str(riskConstraint))

        
    # Objective
    if len(riskProperty) != 0 :
        prob += 1, "obj"
    else:
        prob += variableDict["v_"+str(len(net.layers))+"_"+str(indexOutputNeuron)], "obj"
    

    prob.writeLP("fv_"+str(layerIndex)+"_"+str(indexOutputNeuron)+".lp")


    if(isUsingCplex == False):
        # Solve the problem using the default solver (CBC)
        try:
            prob.solve()
        except: 
            prob.solve(GLPK("/usr/local/bin/glpsol", options=["--cbg"]))
    else:
        prob.solve(CPLEX())
        
    if(len(riskProperty) == 0):
        if prob.status == 1:
            return value(prob.objective)
        else:
            if isMaxBound:            
                print("Warning in computing maxbound of "+"v_"+str(layerIndex)+"_"+str(indexOutputNeuron)+": solver Status:", LpStatus[prob.status])        
                if(prob.status == -1):
                    print("Error in LP/MILP solver: Impossible for simple linear constraint to have no bound")
                prob.writeLP("fv_"+str(layerIndex)+"_"+str(indexOutputNeuron)+".lp") 
                #printSolverStatus()
                return math.inf
            else: 
                print("Warning in computing minbound of "+"v_"+str(layerIndex)+"_"+str(indexOutputNeuron)+": solver Status:", LpStatus[prob.status])
                if(prob.status == -1):
                    print("Error in LP/MILP solver: Impossible for simple linear constraint to have no bound")            
                prob.writeLP("fv_"+str(layerIndex)+"_"+str(indexOutputNeuron)+".lp")              
                #printSolverStatus()
                return -math.inf       
    else:
        if prob.status == -1:
            print("Risk property is not reachable")
        else:
            print("Risk property may be reachable")

    
    
    return
   
