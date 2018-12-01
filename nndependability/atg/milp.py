import numpy as np 
from ortools.linear_solver import pywraplp


def solveCP(lpConstraint):
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
    
    solver.SetTimeLimit(10000)
    result_status = solver.Solve()
    # The problem has an optimal solution.
    if result_status == pywraplp.Solver.OPTIMAL:
        print("Optimal solution found")
    elif result_status == pywraplp.Solver.FEASIBLE:
        print("Timeout but feasible solution found in 10 seconds")
    else: 
        print(result_status)
        raise Exception("The solver can not find optimal or feasible solution within time bound in 10 seconds") 
    
    # The solution looks legit (when using solvers other than
    assert solver.VerifySolution(1e-7, True)

    #print('Number of variables =', solver.NumVariables())
    #print('Number of constraints =', solver.NumConstraints())

    variableAssignment = dict()
    for key, var in variableDict.items():
        variableAssignment[key] = var.solution_value()
    
    
    return len(lpConstraint.occupyVars), solver.Objective().Value(), variableAssignment 

        
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
 
        
def createValueAssignmentVariable(criteria, value):
    return "C" + str(criteria) + "_" + str(value)
         