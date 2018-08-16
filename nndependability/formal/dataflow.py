# Import PuLP modeler functions
from pulp import *
import numpy as np



def deriveLinearOutputBound(isMaxBound, layerIndex, weights, bias, numberOfInputs, nout, minBound, maxBound, octagonBound = []):
    if layerIndex < 1:
        raise Error("layer index shall be smaller than X")

    # A new LP problem
    prob = None
    if isMaxBound: 
        prob = LpProblem("test1", LpMaximize)    
    else:
        prob = LpProblem("test1", LpMinimize)                
        
    variableDict = dict()  
    variableDict["v_"+str(layerIndex)+"_"+str(nout)] =  LpVariable("v_"+str(layerIndex)+"_"+str(nout), lowBound=0, upBound=None , cat='Continuous')             
    for nin in range(numberOfInputs):
        variableDict["v_"+str(layerIndex-1)+"_"+str(nin)] = LpVariable("v_"+str(layerIndex -1)+"_"+str(nin), lowBound=minBound[nin], upBound=maxBound[nin] , cat='Continuous') 
      
    # equality constraint: im = w1 in1 + .... + bias <==> -bias = -im + w1 in1 + ....   
    equalityConstraint = []
    equalityConstraint.append((variableDict["v_"+str(layerIndex)+"_"+str(nout)], -1))
    for nin in range(numberOfInputs):
        equalityConstraint.append((variableDict["v_"+str(layerIndex-1)+"_"+str(nin)], weights[nin].item()))
    c = LpAffineExpression(equalityConstraint)
    prob += c == -1*bias, "eq"
    
    for constraint in octagonBound:    
        try:
            boundConstraint = []
            boundConstraint.append((variableDict[constraint[1]], 1))
            boundConstraint.append((variableDict[constraint[2]], -1))
            c = LpAffineExpression(boundConstraint)
            prob += c >= constraint[0]
            prob += c <= constraint[3]
        except:
            # print(octagonBound)
            print("Problem is processing octagonBound: "+ str(constraint))
            #print("", end='')
        
    # Objective
    prob += variableDict["v_"+str(layerIndex)+"_"+str(nout)], "obj"
    
    if nout == 0:
        prob.writeLP("bound_nout_"+str(layerIndex)+"_"+str(nout)+".lp")    

    # Solve the problem using the default solver (CBC)
    prob.solve()

    return value(prob.objective)

def isRiskPropertyReachable(isMaxBound, layerIndex, weights, bias, numberOfInputs, numberOfOutputs, minBound, maxBound, octagonBound = [], riskProperty = []):
    if layerIndex < 1:
        raise Error("layer index shall be smaller than X")
    if len(riskProperty) == 0:
        print("Safety property trivially hold")
    
        
    # A new LP problem
    prob = None
    if isMaxBound: 
        prob = LpProblem("test1", LpMaximize)    
    else:
        prob = LpProblem("test1", LpMinimize)                
        
    variableDict = dict()  
    for nout in range(numberOfOutputs):
        variableDict["v_"+str(layerIndex)+"_"+str(nout)] =  LpVariable("v_"+str(layerIndex)+"_"+str(nout), lowBound=0, upBound=None , cat='Continuous')             
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
            boundConstraint.append((variableDict[constraint[2]], -1))
            c = LpAffineExpression(boundConstraint)
            prob += c >= constraint[0]
            prob += c <= constraint[3]
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
    prob.solve()

    
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

    
def deriveReLuOutputBound(isMaxBound, layerIndex, weights, bias, numberOfInputs, nout, bigM, minBound, maxBound, inputConstraints = []):
    '''
    Apply dataflow analysis (i.e., abstract interpretation with boxed domain) to derive the bound of an output neuron, where MILP is called internally.  
    
    '''
    
    if layerIndex < 1:
        raise Error("layer index shall be smaller than X")

    # A new LP problem
    prob = None
    if isMaxBound: 
        prob = LpProblem("test1", LpMaximize)    
    else:
        prob = LpProblem("test1", LpMinimize)                
        
    variableDict = dict()
    variableDict["b_"+str(layerIndex)+"_"+str(nout)] = LpVariable("b_"+str(layerIndex)+"_"+str(nout), lowBound=0, upBound=1 , cat='Integer')
    variableDict["im_"+str(layerIndex)+"_"+str(nout)] =  LpVariable("im_"+str(layerIndex)+"_"+str(nout), lowBound=None, upBound=None , cat='Continuous')
    variableDict["v_"+str(layerIndex)+"_"+str(nout)] =  LpVariable("v_"+str(layerIndex)+"_"+str(nout), lowBound=0, upBound=None , cat='Continuous')     
    for nin in range(numberOfInputs):
        variableDict["v_"+str(layerIndex-1)+"_"+str(nin)] = LpVariable("v_"+str(layerIndex -1)+"_"+str(nin), lowBound=minBound[nin], upBound=maxBound[nin] , cat='Continuous') 
      
    # equality constraint: im = w1 in1 + .... + bias <==> -bias = -im + w1 in1 + ....   
    equalityConstraint = []
    equalityConstraint.append((variableDict["im_"+str(layerIndex)+"_"+str(nout)], -1))
    for nin in range(numberOfInputs):
        equalityConstraint.append((variableDict["v_"+str(layerIndex-1)+"_"+str(nin)], weights[nin].item()))
    c = LpAffineExpression(equalityConstraint)
    prob += c == -1*bias, "eq"
    
    # v >= 0 is omitted 
    # c1: v >= im  <==> im - v  <= 0   
    prob += variableDict["im_"+str(layerIndex)+"_"+str(nout)] - variableDict["v_"+str(layerIndex)+"_"+str(nout)] <= 0, "c1"
    # c2: v <= im + (1-b)M <==> v - im + M(b) <= M 
    prob += variableDict["v_"+str(layerIndex)+"_"+str(nout)] -variableDict["im_"+str(layerIndex)+"_"+str(nout)] + bigM*variableDict["b_"+str(layerIndex)+"_"+str(nout)] <= bigM, "c2"
    # c3: v <= bM <=>  M(b) - V >= 0
    prob += bigM*variableDict["b_"+str(layerIndex)+"_"+str(nout)] - variableDict["v_"+str(layerIndex)+"_"+str(nout)] >= 0, "c3"

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
    
    
    # Objective
    prob += variableDict["v_"+str(layerIndex)+"_"+str(nout)], "obj"

    if nout == 0:
        prob.writeLP("bound_nout_"+str(layerIndex)+"_"+str(nout)+".lp")       
    
    # Solve the problem using the default solver (CBC)
    prob.solve()

    # Print the status of the solved LP
    # print("Status:", LpStatus[prob.status])

    # Print the value of the variables at the optimum
    #for v in prob.variables():
        #print(v.name, "=", v.varValue)

    # Print the value of the objective
    # print("objective=", value(prob.objective))

    return value(prob.objective)
