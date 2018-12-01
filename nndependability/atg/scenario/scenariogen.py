import numpy as np 
from ortools.linear_solver import pywraplp

from ...metrics import ScenarioKProjection
from .. import milp

def proposeScenariocandidate(metric):
    """ Based on the current metric, propose a new scenario to maximally increase k-projection coverage.
    """
    lpConstraint = prepareLPconstraint(metric)
    variableAssignment = solveCP(lpConstraint, metric)

    for criterion in metric.sd.operatingConditionCriteria:
        for assignment in metric.sd.operatingConditionItems[criterion]: 
            var = milp.createValueAssignmentVariable(metric.sd.operatingConditionCriteria.index(criterion), metric.sd.operatingConditionItems[criterion].index(assignment))
            if(variableAssignment[var] == 1):
                print("for criterion "+str(criterion)+", set it to "+str(assignment))
    

    return variableAssignment
    
    
def solveCP(lpConstraint, metric):
    """ Solve the MILP constraint program.
    
    """
    
    totalnum, maxImprove, variableAssignment = milp.solveCP(lpConstraint)
    # The objective value of the solution.
    print('Maximum possibility for improvement =', totalnum)
    print('Optimal objective value computed from IP = %d' % maxImprove)
    print() 

    return variableAssignment

def prepareLPconstraint(metric):
    """ Prepare the MILP constraint for maximally improving neuron-k-activation-pattern coverage.
    """

    # kValue, numberOfNeuronsToTrack, k_Activation_record
    
    if not (type(metric) == ScenarioKProjection.Scenario_KProjection_Metric):
        raise TypeError("The method only takes input with type Scenario_KProjection_Metric")
    
       
    lpConstraint = milp.TestCaseGenConstraint()
   
    
    for i in range(metric.cm.numberOfCategorizations):
        exp = milp.Expression()
        # Sum to be 1, therefore set upper and lower bound to be 1.
        exp.lowerbound = 1
        exp.upperbound = 1
        
        # The possible assignment can only be 0 or 1, so set j < 2
        for j in range(metric.cm.maxType[i]):
            exp.coefficients.append(1.0)
            exp.variables.append(milp.createValueAssignmentVariable(i, j))
            lpConstraint.vars.append(milp.createValueAssignmentVariable(i, j))
            
        lpConstraint.constraints.append(exp)
        

    occupyVariables = set()
    
    for projectedCategorization, record in metric.cm.projectionRecords.items():
        for assignment, quantity in record.currentlyOccupiedEntities.items():
            
            if (quantity < record.maxOccupiedEntities.get(assignment)):
                #  This item can be improved

                occupyVariable = "occupy_" + str(projectedCategorization) + "_be_" + str(assignment)
                occupyVariables.add(occupyVariable)

                variables = projectedCategorization.split('_')
                
                exp = milp.Expression()
                exp.lowerbound = 0
                exp.upperbound = metric.cm.kValue - 1
                for i in range(len(variables)):
                    # print("C" + variables[i] + "_" + assignment[2*i: 2*i + 1])
                    exp.variables.append("C" + variables[i] + "_" + assignment[2*i: 2*i + 1])
                    exp.coefficients.append(1.0)

                exp.variables.append(occupyVariable)
                exp.coefficients.append(-1.0 * metric.cm.kValue)

                lpConstraint.constraints.append(exp)


    # Add all domain restrictions
    lpConstraint.constraints.extend(metric.dr.domainRestrictions)
    
    lpConstraint.vars.extend(occupyVariables)
    lpConstraint.occupyVars.extend(occupyVariables)

    # lpConstraint.printConstraint()
    return lpConstraint
    
     
        
