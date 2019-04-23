# Import PuLP modeler functions
from pulp import *
import math

def deriveReLuOutputOctagonBound(isMaxBound, layerIndex, weights, bias, numberOfInputs, nout1, nout2, bigM, minBound, maxBound, octagonBound, isDifference, inputConstraints = [], seconds = 5, isUsingCplex = False):
    """Derive the output difference bound for two neurons nout1 and nout2.

    This is based on a partial re-implementation of the ATVA'17 paper https://arxiv.org/pdf/1705.01040.pdf.
    See Proposition 1 for the MILP encoding, and using MILP to derive bounds is essentially the heuristic 1 in the paper. 
    
    Args:
        isMaxBound: derive max bound
        layerIndex: indexing of the layer in the overall network (for debugging purposes)
        minBound: array of input lower bounds
        maxBound: array of input upper bounds
        octagonBound: array of input constraints, with each specified with a shape of L <= in_i - in_j <= U, stored as [L, i, j, U] 
        isDifference: compute v1-v2 if True, v1+v2 if False
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
    # c11: v >= im  <==> im - v  <= 0   
    prob += variableDict["im_"+str(layerIndex)+"_"+str(nout1)] - variableDict["v_"+str(layerIndex)+"_"+str(nout1)] <= 0, "c11"
    # c12: v <= im + (1-b)M <==> v - im + M(b) <= M 
    prob += variableDict["v_"+str(layerIndex)+"_"+str(nout1)] -variableDict["im_"+str(layerIndex)+"_"+str(nout1)] + bigM*variableDict["b_"+str(layerIndex)+"_"+str(nout1)] <= bigM, "c12"
    # c13: v <= bM <=>  M(b) - V >= 0
    prob += bigM*variableDict["b_"+str(layerIndex)+"_"+str(nout1)] - variableDict["v_"+str(layerIndex)+"_"+str(nout1)] >= 0, "c13"
    # c14: im + (1-b) M >= 0 <=> im - M(b) >= -M
    prob += variableDict["im_"+str(layerIndex)+"_"+str(nout1)] - bigM*variableDict["b_"+str(layerIndex)+"_"+str(nout1)] >= -1*bigM, "c14"    
    # c15: im  - b M <= 0 
    prob += variableDict["im_"+str(layerIndex)+"_"+str(nout1)] - bigM*variableDict["b_"+str(layerIndex)+"_"+str(nout1)] <= 0, "c15" 
    
    
    # for neuron nout2
    # c21: v >= im  <==> im - v  <= 0   
    prob += variableDict["im_"+str(layerIndex)+"_"+str(nout2)] - variableDict["v_"+str(layerIndex)+"_"+str(nout2)] <= 0, "c21"
    # c22: v <= im + (1-b)M <==> v - im + M(b) <= M 
    prob += variableDict["v_"+str(layerIndex)+"_"+str(nout2)] -variableDict["im_"+str(layerIndex)+"_"+str(nout2)] + bigM*variableDict["b_"+str(layerIndex)+"_"+str(nout2)] <= bigM, "c22"
    # c23: v <= bM <=>  M(b) - V >= 0
    prob += bigM*variableDict["b_"+str(layerIndex)+"_"+str(nout2)] - variableDict["v_"+str(layerIndex)+"_"+str(nout2)] >= 0, "c23"
    # c24: im + (1-b) M >= 0 <=> im - M(b) >= -M
    prob += variableDict["im_"+str(layerIndex)+"_"+str(nout2)] - bigM*variableDict["b_"+str(layerIndex)+"_"+str(nout2)] >= -1*bigM, "c24"    
    # c25: im  - b M <= 0 
    prob += variableDict["im_"+str(layerIndex)+"_"+str(nout2)] - bigM*variableDict["b_"+str(layerIndex)+"_"+str(nout2)] <= 0, "c25" 
    
    for constraint in octagonBound:
        boundConstraint = []
        boundConstraint.append((variableDict[constraint[1]], 1))
        boundConstraint.append((variableDict[constraint[3]], constraint[2]))
        c = LpAffineExpression(boundConstraint)
        prob += c >= constraint[0]
        prob += c <= constraint[4]

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
    if isDifference == True:
        prob += variableDict["v_"+str(layerIndex)+"_"+str(nout1)] - variableDict["v_"+str(layerIndex)+"_"+str(nout2)], "obj"    
    else:
        prob += variableDict["v_"+str(layerIndex)+"_"+str(nout1)] + variableDict["v_"+str(layerIndex)+"_"+str(nout2)], "obj"

    
    # prob.writeLP("test.lp")
    if(isUsingCplex == False):
        # Solve the problem using the default solver (CBC)
        try:
            prob.solve()
        except: 
            prob.solve(GLPK(options=["--cbg"]))
			#prob.solve(GLPK("/usr/local/bin/glpsol", options=["--cbg"]))
    else:
        prob.solve(CPLEX())
    
    if prob.status == 1:
        return value(prob.objective)
    else:
        if isMaxBound:
            if isDifference:
                print("Warning in computing maxbound of "+"v_"+str(layerIndex)+"_"+str(nout1)+ "-"+"v_"+str(layerIndex)+"_"+str(nout2)+": solver Status:", LpStatus[prob.status])  
            else:
                print("Warning in computing maxbound of "+"v_"+str(layerIndex)+"_"+str(nout1)+ "+"+"v_"+str(layerIndex)+"_"+str(nout2)+": solver Status:", LpStatus[prob.status])
            return math.inf
        else: 
            if isDifference:
                print("Warning in computing minbound of "+"v_"+str(layerIndex)+"_"+str(nout1)+ "-"+"v_"+str(layerIndex)+"_"+str(nout2)+": solver Status:", LpStatus[prob.status])  
            else:
                print("Warning in computing minbound of "+"v_"+str(layerIndex)+"_"+str(nout1)+ "+"+"v_"+str(layerIndex)+"_"+str(nout2)+": solver Status:", LpStatus[prob.status])
            return -math.inf   
    
    # Here we would like to allow timeout in the solver, such that the solver only performs a fixed amount for it can found at best
    # prob.solve(PULP_CBC_CMD(maxSeconds=2))

    #return value(prob.objective)
