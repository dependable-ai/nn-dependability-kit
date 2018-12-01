import numpy as np 

from ...metrics import NeuronKProjection
from .. import milp

def proposeNAPcandidate(metric):
    """ Based on the current metric, propose a new neuron activation pattern to maximally increase k-projection coverage.
    """
    lpConstraint = prepareLPconstraint(metric)
    solveCP(lpConstraint, metric)

def solveCP(lpConstraint, metric):
    """ Solve the MILP constraint program.
    
    """
    
    totalnum, maxImprove, variableAssignment = milp.solveCP(lpConstraint)
    # The objective value of the solution.
    print('Maximum possibility for improvement =', totalnum)
    print('Optimal objective value computed from IP = %d' % maxImprove)
    print() 

    for i in range(metric.numberOfNeuronsToTrack):
        for j in range(2): 
            var = "C" + str(i) + "_" + str(j)
            if(variableAssignment[var] == 1):
                print("for neuron "+str(i)+", set it to "+str(j))
    
    
def prepareLPconstraint(metric):
    """ Prepare the MILP constraint for maximally improving neuron-k-activation-pattern coverage.
    """

    # kValue, numberOfNeuronsToTrack, k_Activation_record
    
    if not (type(metric) == NeuronKProjection.Neuron_OnOff_KProjection_Metric):
        raise TypeError("The method only takes input with type Neuron_OnOff_KProjection_Metric")
    
    if metric.kValue != 2:
        raise NotImplementedError("The method does not support cases where k != 2")
        
    lpConstraint = milp.TestCaseGenConstraint()
    
    for i in range(metric.numberOfNeuronsToTrack):
        exp = milp.Expression()
        # Sum to be 1, therefore set upper and lower bound to be 1.
        exp.lowerbound = 1
        exp.upperbound = 1
        
        # The possible assignment can only be 0 or 1, so set j < 2
        for j in range(2) :
            exp.coefficients.append(1.0)
            exp.variables.append(milp.createValueAssignmentVariable(i, j))
            lpConstraint.vars.append(milp.createValueAssignmentVariable(i, j))

        lpConstraint.constraints.append(exp)
        
    
        
    occupyVariables = set()

    allPossibleAssignments = set(["00", "01", "10", "11"])
    for key in metric.k_Activation_record.keys():        
        for value in allPossibleAssignments - metric.k_Activation_record[key]:
            occupyVariable = "occupy_" + str(key) + "_be_" + str(value)
            occupyVariables.add(occupyVariable)
        
            variables = key.replace("N", "").split('_')
            
            exp = milp.Expression()
            exp.lowerbound = 0
            exp.upperbound = metric.kValue - 1
            for i in range(len(variables)):
                # exp.variables.append("C" + variables[i] + "_" + value[i: i + 1])
                exp.variables.append(milp.createValueAssignmentVariable(variables[i], value[i: i + 1]))
                exp.coefficients.append(1.0)
            
            exp.variables.append(occupyVariable)
            exp.coefficients.append(-1.0 * metric.kValue)

            lpConstraint.constraints.append(exp)
            
    lpConstraint.vars.extend(occupyVariables)
    lpConstraint.occupyVars.extend(occupyVariables)

    # lpConstraint.printConstraint()
    return lpConstraint
    
