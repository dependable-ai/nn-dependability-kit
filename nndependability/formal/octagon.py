# Import PuLP modeler functions
from pulp import *

def deriveReLuOutputDifferenceBound(isMaxBound, layerIndex, weights, bias, numberOfInputs, nout1, nout2, bigM, minBound, maxBound, octagonBound, inputConstraints = [], seconds = 5):
    '''

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
    variableDict["b_"+str(layerIndex)+"_"+str(nout1)] = LpVariable("b_"+str(layerIndex)+"_"+str(nout1), lowBound=0, upBound=1 , cat='Integer')
    variableDict["im_"+str(layerIndex)+"_"+str(nout1)] =  LpVariable("im_"+str(layerIndex)+"_"+str(nout1), lowBound=None, upBound=None , cat='Continuous')
    variableDict["v_"+str(layerIndex)+"_"+str(nout1)] =  LpVariable("v_"+str(layerIndex)+"_"+str(nout1), lowBound=0, upBound=None , cat='Continuous')     
    
    variableDict["b_"+str(layerIndex)+"_"+str(nout2)] = LpVariable("b_"+str(layerIndex)+"_"+str(nout2), lowBound=0, upBound=1 , cat='Integer')
    variableDict["im_"+str(layerIndex)+"_"+str(nout2)] =  LpVariable("im_"+str(layerIndex)+"_"+str(nout2), lowBound=None, upBound=None , cat='Continuous')
    variableDict["v_"+str(layerIndex)+"_"+str(nout2)] =  LpVariable("v_"+str(layerIndex)+"_"+str(nout2), lowBound=0, upBound=None , cat='Continuous')     
        
    
    for nin in range(numberOfInputs):
        variableDict["v_"+str(layerIndex-1)+"_"+str(nin)] = LpVariable("v_"+str(layerIndex -1)+"_"+str(nin), lowBound=minBound[nin], upBound=maxBound[nin] , cat='Continuous') 
      
    # equality constraint: im = w1 in1 + .... + bias <==> -bias = -im + w1 in1 + ....   
    equalityConstraint1 = []
    equalityConstraint1.append((variableDict["im_"+str(layerIndex)+"_"+str(nout1)], -1))
    for nin in range(numberOfInputs):
        equalityConstraint1.append((variableDict["v_"+str(layerIndex-1)+"_"+str(nin)], weights[nout1][nin].item()))
    c1 = LpAffineExpression(equalityConstraint1)
    prob += c1 == -1*bias[nout1], "eq1"

    equalityConstraint2 = []
    equalityConstraint2.append((variableDict["im_"+str(layerIndex)+"_"+str(nout2)], -1))
    for nin in range(numberOfInputs):
        equalityConstraint2.append((variableDict["v_"+str(layerIndex-1)+"_"+str(nin)], weights[nout2][nin].item()))
    c2 = LpAffineExpression(equalityConstraint2)
    prob += c2 == -1*bias[nout2], "eq2"    
    
    # for neuron nout1
    # c1: v >= im  <==> im - v  <= 0   
    prob += variableDict["im_"+str(layerIndex)+"_"+str(nout1)] - variableDict["v_"+str(layerIndex)+"_"+str(nout1)] <= 0, "c1"
    # c2: v <= im + (1-b)M <==> v - im + M(b) <= M 
    prob += variableDict["v_"+str(layerIndex)+"_"+str(nout1)] -variableDict["im_"+str(layerIndex)+"_"+str(nout1)] + bigM*variableDict["b_"+str(layerIndex)+"_"+str(nout1)] <= bigM, "c2"
    # c3: v <= bM <=>  M(b) - V >= 0
    prob += bigM*variableDict["b_"+str(layerIndex)+"_"+str(nout1)] - variableDict["v_"+str(layerIndex)+"_"+str(nout1)] >= 0, "c3"

    # for neuron nout2
    # c4: v >= im  <==> im - v  <= 0   
    prob += variableDict["im_"+str(layerIndex)+"_"+str(nout2)] - variableDict["v_"+str(layerIndex)+"_"+str(nout2)] <= 0, "c4"
    # c5: v <= im + (1-b)M <==> v - im + M(b) <= M 
    prob += variableDict["v_"+str(layerIndex)+"_"+str(nout2)] -variableDict["im_"+str(layerIndex)+"_"+str(nout2)] + bigM*variableDict["b_"+str(layerIndex)+"_"+str(nout2)] <= bigM, "c5"
    # c6: v <= bM <=>  M(b) - V >= 0
    prob += bigM*variableDict["b_"+str(layerIndex)+"_"+str(nout2)] - variableDict["v_"+str(layerIndex)+"_"+str(nout2)] >= 0, "c6"
    
    for constraint in octagonBound:
        boundConstraint = []
        boundConstraint.append((variableDict[constraint[1]], 1))
        boundConstraint.append((variableDict[constraint[2]], -1))
        c = LpAffineExpression(boundConstraint)
        prob += c >= constraint[0]
        prob += c <= constraint[3]

    for inputConstr in inputConstraints:  
        try:
            boundConstraint = []
            for i in range(int(len(inputConstr)/2) - 1):
                boundConstraint.append((variableDict[inputConstr[2*(i+1)+1].replace("in", "v_0_")], inputConstr[2*(i+1)]))
            c = LpAffineExpression(boundConstraint)
            # Here be careful - if the sign is "<=" then one shall place ">="
            if inputConstr[1] == "<=":
                prob += c <= inputConstr[0]
            elif inputConstr[1] == "==":
                prob += c == inputConstr[0]
            elif inputConstr[1] == ">=":
                prob += c >= inputConstr[0]
            else:
                raise ValueError('Operators shall be <=, == or >=')
                
            #print(c)
        except:
            print("Problem is processing input constraint: "+ str(inputConstr))
            #print("", end='')     
    
    # Objective
    prob += variableDict["v_"+str(layerIndex)+"_"+str(nout1)] - variableDict["v_"+str(layerIndex)+"_"+str(nout2)], "obj"
    
    # prob.writeLP("test.lp")
    # Solve the problem using the default solver (CBC)
    prob.solve()
    # Here we would like to allow timeout in the solver, such that the solver only performs a fixed amount for it can found at best
    # prob.solve(PULP_CBC_CMD(maxSeconds=2))

    return value(prob.objective)