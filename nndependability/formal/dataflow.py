# Import PuLP modeler functions
from pulp import *
import numpy as np
import math



def deriveLinearOutputBound(isMaxBound, layerIndex, weights, bias, numberOfInputs, nout, minBound, maxBound, octagonBound = [], inputConstraints = [], isUsingCplex = False):
    """Derive the min or max output of a neuron using boxed domain and MILP, where the neuron is a linear function.

    This is based on a partial re-implementation of the ATVA'17 paper https://arxiv.org/pdf/1705.01040.pdf.
    See Proposition 1 for the MILP encoding, and using MILP to derive bounds is essentially the heuristic 1 in the paper.     
    
    Args:
        isMaxBound: derive max bound
        layerIndex: indexing of the layer in the overall network (for debugging purposes)
        minBound: array of input lower bounds
        maxBound: array of input upper bounds
        octagonBound: array of input constraints, with each specified with a shape of L <= in_i - in_j <= U, stored as [L, i, j, U] 

    Returns:    
    Raises:
    """

    if layerIndex < 1:
        raise ValueError("layer index shall be smaller than X")

    # A new LP problem
    prob = None
    if isMaxBound: 
        prob = LpProblem("test1", LpMaximize)    
    else:
        prob = LpProblem("test1", LpMinimize)                
        
    variableDict = dict()  
    variableDict["v_"+str(layerIndex)+"_"+str(nout)] =  LpVariable("v_"+str(layerIndex)+"_"+str(nout), lowBound=None, upBound=None , cat='Continuous')             
    for nin in range(numberOfInputs):
        variableDict["v_"+str(layerIndex-1)+"_"+str(nin)] = LpVariable("v_"+str(layerIndex -1)+"_"+str(nin), lowBound=minBound[nin], upBound=maxBound[nin] , cat='Continuous') 
      
    # equality constraint: im = w1 in1 + .... + bias <==> -bias = -im + w1 in1 + ....   
    equalityConstraint = []
    equalityConstraint.append((variableDict["v_"+str(layerIndex)+"_"+str(nout)], -1))
    for nin in range(numberOfInputs):
        equalityConstraint.append((variableDict["v_"+str(layerIndex-1)+"_"+str(nin)], weights[nin].item()))
    c = LpAffineExpression(equalityConstraint)
    prob += c == -1*bias, "eq"

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
    
    
    for constraint in octagonBound:    
        try:
            boundConstraint = []
            boundConstraint.append((variableDict[constraint[1]], 1))
            boundConstraint.append((variableDict[constraint[3]], constraint[2]))
            c = LpAffineExpression(boundConstraint)
            prob += c >= constraint[0]
            prob += c <= constraint[4]
        except:
            # Print warning, then move on
            print("Problem in processing octagonBound: "+ str(constraint))
            #print("", end='')
        
    # Objective
    prob += variableDict["v_"+str(layerIndex)+"_"+str(nout)], "obj"
    
    if nout == 0:
        if isMaxBound:
            prob.writeLP("bound_nout_max_"+str(layerIndex)+"_"+str(nout)+".lp")
        else:
            prob.writeLP("bound_nout_min_"+str(layerIndex)+"_"+str(nout)+".lp")

    if(isUsingCplex == False):
        # Solve the problem using the default solver (CBC)
        try:
            prob.solve()
        except: 
            prob.solve(GLPK("/usr/local/bin/glpsol", options=["--cbg"]))
    else:
        prob.solve(CPLEX())

    if prob.status == 1:
        return value(prob.objective)
    else:
        if isMaxBound:            
            print("Warning in computing maxbound of "+"v_"+str(layerIndex)+"_"+str(nout)+": solver Status:", LpStatus[prob.status])        
            if(prob.status == -1):
                print("Error in LP/MILP solver: Impossible for simple linear constraint to have no bound")
            prob.writeLP("bound_nout_max_"+str(layerIndex)+"_"+str(nout)+".lp")  
            #printSolverStatus()
            return math.inf
        else: 
            print("Warning in computing minbound of "+"v_"+str(layerIndex)+"_"+str(nout)+": solver Status:", LpStatus[prob.status])
            if(prob.status == -1):
                print("Error in LP/MILP solver: Impossible for simple linear constraint to have no bound")            
            prob.writeLP("bound_nout_min_"+str(layerIndex)+"_"+str(nout)+".lp")              
            #printSolverStatus()
            return -math.inf   
    

def printSolverStatus():
    print("#LpStatusOptimal	“Optimal”	1")
    print("#LpStatusNotSolved	“Not Solved”	0")
    print("#LpStatusInfeasible	“Infeasible”	-1")
    print("#LpStatusUnbounded	“Unbounded”	-2")
    print("#LpStatusUndefined	“Undefined”	-3")

    
def isRiskPropertyReachable(layerIndex, weights, bias, numberOfInputs, numberOfOutputs, minBound, maxBound, octagonBound = [], riskProperty = [], isUsingCplex = False):
    """Compute if a certain risk property associated with a certain neuron layer is reachable.

    This is based on a partial re-implementation of the ATVA'17 paper https://arxiv.org/pdf/1705.01040.pdf.
    See Proposition 1 for the MILP encoding, and using MILP to derive bounds is essentially the heuristic 1 in the paper.     
    
    Args:
        layerIndex: indexing of the layer in the overall network (for debugging purposes)
        minBound: array of input lower bounds
        maxBound: array of input upper bounds
        octagonBound: array of input constraints, with each specified with a shape of L <= in_i - in_j <= U, stored as [L, i, j, U]
        riskProperty: array of linear constraints related to output value of the layer
    Returns:    
    Raises:
    """

    if layerIndex < 1:
        raise ValueError("layer index shall be smaller than X")
    if len(riskProperty) == 0:
        print("Safety property trivially hold")
    
        
    # A new LP problem
    prob = LpProblem("test1", LpMaximize)    
              
        
    variableDict = dict()  
    for nout in range(numberOfOutputs):
        variableDict["v_"+str(layerIndex)+"_"+str(nout)] =  LpVariable("v_"+str(layerIndex)+"_"+str(nout), lowBound=None, upBound=None , cat='Continuous')             
    for nin in range(numberOfInputs):
        variableDict["v_"+str(layerIndex-1)+"_"+str(nin)] = LpVariable("v_"+str(layerIndex -1)+"_"+str(nin), lowBound=minBound[nin], upBound=maxBound[nin] , cat='Continuous') 
      
    # equality constraint: im = w1 in1 + .... + bias <==> -bias = -im + w1 in1 + ....   
    for nout in range(numberOfOutputs):
        equalityConstraint = []
        equalityConstraint.append((variableDict["v_"+str(layerIndex)+"_"+str(nout)], -1))
        for nin in range(numberOfInputs):
            equalityConstraint.append((variableDict["v_"+str(layerIndex-1)+"_"+str(nin)], weights[nout][nin].item()))
        c = LpAffineExpression(equalityConstraint)
        prob += c == -1*bias[nout], "eq"+str(nout)
    
    for constraint in octagonBound:    
        try:
            boundConstraint = []
            boundConstraint.append((variableDict[constraint[1]], 1))
            boundConstraint.append((variableDict[constraint[3]], constraint[2]))
            c = LpAffineExpression(boundConstraint)
            prob += c >= constraint[0]
            prob += c <= constraint[4]
        except:
            print("", end='')
 
    for riskConstraint in riskProperty:  
        try:
            boundConstraint = []
            for i in range(int(len(riskConstraint)/2) - 1):
                boundConstraint.append((variableDict[riskConstraint[2*(i+1)+1].replace("out", "v_"+str(layerIndex)+"_")], riskConstraint[2*(i+1)]))
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

     
    # Objective (here no objective is needed)
    # prob += variableDict["v_"+str(layerIndex)+"_"+str(0)], "obj"
    
    # Debug purpose
    if len(octagonBound) == 0:
        prob.writeLP("propertyBoxedAbstraction.lp")
    else:
        prob.writeLP("propertyOctagonAbstraction.lp")
        
    # Solve the problem using the default solver (CBC)
    # Solve the problem using the default solver (CBC)
    if(isUsingCplex == False):
        # Solve the problem using the default solver (CBC)
        try:
            prob.solve()
        except: 
            prob.solve(GLPK("/usr/local/bin/glpsol", options=["--cbg"]))
    else:
        prob.solve(CPLEX())
    
    # Print the status of the solved LP
    print("==============================")
    #print("#LpStatusOptimal	“Optimal”	1")
    #print("#LpStatusNotSolved	“Not Solved”	0")
    #print("#LpStatusInfeasible	“Infeasible”	-1")
    #print("#LpStatusUnbounded	“Unbounded”	-2")
    #print("#LpStatusUndefined	“Undefined”	-3")
    print("Solver Status:", LpStatus[prob.status])

    if prob.status == -1:
        return False
    else:
        return True

def deriveReLuOutputBound(isMaxBound, layerIndex, weights, bias, numberOfInputs, nout, bigM, minBound, maxBound, inputConstraints = [], octagonBound = [], isUsingCplex = False):
    value = deriveLinearOutputBound(isMaxBound, layerIndex, weights, bias, numberOfInputs, nout, minBound, maxBound, octagonBound, inputConstraints)
    if value < 0:
        return 0
    else:    
        return value
        
def deriveELuOutputBound(isMaxBound, layerIndex, weights, bias, numberOfInputs, nout, bigM, minBound, maxBound, inputConstraints = [], alpha = 1.0, isUsingCplex = False):
    value = deriveLinearOutputBound(isMaxBound, layerIndex, weights, bias, numberOfInputs, nout, minBound, maxBound, [], inputConstraints)
    if value < 0:
        return alpha*(np.exp(value) - 1)
    else:    
        return value
        
def deriveBNOutputBound(isMaxBound, nout, minBound, maxBound, moving_mean, moving_variance, gamma, beta, epsilon):
    # BN in operating time is just a linear transformation. 
    # Therefore, the min and max value will be either from the original min or original max.
    
    #value1 = computeBN(minBound[nout], moving_mean[nout], moving_variance[nout], gamma[nout], beta[nout], epsilon)
    #value2 = computeBN(maxBound[nout], moving_mean[nout], moving_variance[nout], gamma[nout], beta[nout], epsilon)
    
    value1 = computeBN(minBound[nout], moving_mean[nout], moving_variance[nout], gamma[nout], beta[nout], epsilon)
    value2 = computeBN(maxBound[nout], moving_mean[nout], moving_variance[nout], gamma[nout], beta[nout], epsilon)
    
    if(isMaxBound):
        return max(value1, value2)
    else:
        return min(value1, value2)
        

def computeBN(z, moving_mean, moving_variance, gamma, beta, epsilon):
    #print(z.shape)
    #print(moving_mean)
    #print(moving_variance)
    #print(gamma.shape)
    #print(beta.shape)
    z_norm = (z -  moving_mean) / (np.sqrt(moving_variance + epsilon))
    z_tilda = (gamma * z_norm) + beta
    return z_tilda