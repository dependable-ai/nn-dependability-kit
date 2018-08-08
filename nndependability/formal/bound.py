# Import PuLP modeler functions
from pulp import *

def deriveReLuOutputBound(isMaxBound, layerIndex, weights, bias, numberOfInputs, nout, bigM, minBound, maxBound):
                                              
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

    # c1: v >= im  <==> 0 >= im - v >= -INF
    
    prob += variableDict["im_"+str(layerIndex)+"_"+str(nout)] - variableDict["v_"+str(layerIndex)+"_"+str(nout)] <= 0, "c1"
    prob += variableDict["v_"+str(layerIndex)+"_"+str(nout)] -variableDict["im_"+str(layerIndex)+"_"+str(nout)] + bigM*variableDict["b_"+str(layerIndex)+"_"+str(nout)] <= bigM, "c2"
    prob += bigM*variableDict["b_"+str(layerIndex)+"_"+str(nout)] - variableDict["v_"+str(layerIndex)+"_"+str(nout)] >= 0, "c3"

    # Objective
    prob += variableDict["v_"+str(layerIndex)+"_"+str(nout)], "obj"
    
    # prob.writeLP("test.lp")
    # Solve the problem using the default solver (CBC)
    prob.solve()
    # Use prob.solve(GLPK()) instead to choose GLPK as the solver
    # Use GLPK(msg = 0) to suppress GLPK messages
    # If GLPK is not in your path and you lack the pulpGLPK module,
    # replace GLPK() with GLPK("/path/")
    # Where /path/ is the path to glpsol (excluding glpsol itself).
    # If you want to use CPLEX, use CPLEX() instead of GLPK().
    # If you want to use XPRESS, use XPRESS() instead of GLPK().
    # If you want to use COIN, use COIN() instead of GLPK(). In this last case,
    # two paths may be provided (one to clp, one to cbc).

    # Print the status of the solved LP
    # print("Status:", LpStatus[prob.status])

    # Print the value of the variables at the optimum
    #for v in prob.variables():
        #print(v.name, "=", v.varValue)

    # Print the value of the objective
    # print("objective=", value(prob.objective))

    return value(prob.objective)
