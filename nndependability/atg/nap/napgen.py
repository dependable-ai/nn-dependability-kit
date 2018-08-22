import numpy as np 
from ortools.linear_solver import pywraplp

from ...metrics import KProjection

def proposeNAPcandidate(metric):
    """ Based on the current metric, propose an on-off neuron activation neuron that maximally increase k-projection coverage (non-quantitative setup).
    """
    lpConstraint = prepareLPconstraint(metric)
    solveCP(lpConstraint, metric)

def solveCP(lpConstraint, metric):
    """ Solve the MILP constraint program.
    
    """

    # Instantiate a mixed-integer solver, naming it SolveIntegerProblem.
    solver = pywraplp.Solver('SolveIntegerProblem', pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING)

    variableDict = dict()
    for var in lpConstraint.vars:
        variableDict[var] = solver.IntVar(0, 1, var)

    for exp in lpConstraint.constraints:
        constraint = None
        if(exp.lowerbound == float('-inf') and exp.upperbound == float('inf')): 
            constraint = solver.Constraint(-solver.infinity(), solver.infinity())
        elif (exp.lowerbound == float('-inf')):
            constraint = solver.Constraint(-solver.infinity(), exp.upperbound)
        elif (exp.upperbound == float('inf')):
            constraint = solver.Constraint(exp.lowerbound, solver.infinity())            
        else:
            constraint = solver.Constraint(exp.lowerbound, exp.upperbound)   
        
        for i in range(len(exp.variables)):
            constraint.SetCoefficient(variableDict[exp.variables[i]], exp.coefficients[i])

    objective = solver.Objective()
    for var in lpConstraint.occupyVars:
        objective.SetCoefficient(variableDict[var], 1)
    objective.SetMaximization()        
            
    """Solve the problem and print the solution."""
    result_status = solver.Solve()
    # The problem has an optimal solution.
    assert result_status == pywraplp.Solver.OPTIMAL

    # The solution looks legit (when using solvers other than
    assert solver.VerifySolution(1e-7, True)

    #print('Number of variables =', solver.NumVariables())
    #print('Number of constraints =', solver.NumConstraints())

    # The objective value of the solution.
    print('Maximum possibility for improvement =', len(lpConstraint.occupyVars))
    print('Optimal objective value computed from IP = %d' % solver.Objective().Value())
    print()

    for i in range(metric.numberOfNeuronsToTrack):
        for j in range(2): 
            var = "C" + str(i) + "_" + str(j)
            if(variableDict[var].solution_value() == 1):
                print("for neuron "+str(i)+", set it to "+str(j))
    
    
    
def prepareLPconstraint(metric):
    """ Prepare the MILP constraint for maximally improving neuron-k-activation-pattern coverage.
    """

    # kValue, numberOfNeuronsToTrack, k_Activation_record
    
    if not (type(metric) == KProjection.Neuron_OnOff_KProjection_Metric):
        raise Error("The method only takes input with type Neuron_OnOff_KProjection_Metric")
    
    if metric.kValue != 2:
        raise NotImplementedError("The method does not support cases where k != 2")
        
    lpConstraint = TestCaseGenConstraint()
    
    for i in range(metric.numberOfNeuronsToTrack):
        exp = Expression()
        # Sum to be 1, therefore set upper and lower bound to be 1.
        exp.lowerbound = 1
        exp.upperbound = 1
        
        # The possible assignment can only be 0 or 1, so set j < 2
        for j in range(2) :
            exp.coefficients.append(1.0)
            exp.variables.append(createValueAssignmentVariable(i, j))
            lpConstraint.vars.append(createValueAssignmentVariable(i, j))

        lpConstraint.constraints.append(exp)
        
    
        
    occupyVariables = set()

    allPossibleAssignments = set(["00", "01", "10", "11"])
    for key in metric.k_Activation_record.keys():        
        for value in allPossibleAssignments - metric.k_Activation_record[key]:
            occupyVariable = "occupy_" + str(key) + "_be_" + str(value)
            occupyVariables.add(occupyVariable)
        
            variables = key.replace("N", "").split('_')
            
            exp = Expression()
            exp.lowerbound = 0
            exp.upperbound = metric.kValue - 1
            for i in range(len(variables)):
                exp.variables.append("C" + variables[i] + "_" + value[i: i + 1])
                exp.coefficients.append(1.0)
            
            exp.variables.append(occupyVariable)
            exp.coefficients.append(-1.0 * metric.kValue)

            lpConstraint.constraints.append(exp)
            
    lpConstraint.vars.extend(occupyVariables)
    lpConstraint.occupyVars.extend(occupyVariables)

    # lpConstraint.printConstraint()
    return lpConstraint
    
    
        
def createValueAssignmentVariable(criteria, value):
    return "C" + str(criteria) + "_" + str(value)
        
        
        
class TestCaseGenConstraint():

    def __init__(self):
        self.constraints = [];
        self.vars = [];
        self.occupyVars = [];   
    
    def printConstraint(self):
        print("All variables used in optimization objective\n\n")
        print(self.occupyVars)
        print("\n\nAll variables in the constraint system\n\n")
        print(self.vars)
        print("\n\nAll constraints\n\n\n")
        for exp in self.constraints:
            exp.printExpression()
        
    
    

class Expression():
    
    def __init__(self):
        self.name = "";
        self.lowerbound = float('-inf');
        self.upperbound = float("inf");    
        self.variables = [];
        self.coefficients= [];

    def printExpression(self):
        print(str(self.lowerbound) + " <= ", end='')
        for i in range(len(self.variables)):
            if self.coefficients[i] >= 0:
                print(" + "+ str(self.coefficients[i]) +" " +str(self.variables[i]), end='')
            else:
                print(" "+ str(self.coefficients[i]) + " "+ str(self.variables[i]), end='')
        print(" <= " + str(self.upperbound))
                