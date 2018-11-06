import numpy as np 
import scipy
import math
import scipy.misc
import xml.etree.ElementTree as ET


"""
ATVA'18 re-implementation from Java to Python 3.x
"""
class Scenario_KProjection_Metric():
    """For the set of inputs with semantic information contained, compute the quantitative k-projection coverage.
    """

    def __init__(self, scenarioDescrptionFileName):
        self.sd = ScenarioDescription(scenarioDescrptionFileName)
        # Change this line if you want to set to different k-projection value
        self.kValue = 2

        maxType = [0] * len(self.sd.operatingConditionCriteria)

        for i in range(len(maxType)):
            maxType[i] = len(self.sd.operatingConditionItems[self.sd.operatingConditionCriteria[i]])

            
        self.cm = CoverageManagement(self.kValue, len(maxType), maxType);            


    def readScenariosFromFile(self, fileName):

        allScenarios = []

        # Read and process input file
        tree = ET.parse(fileName)
        root = tree.getroot()

        for child in root:
               
            if child.tag != "scenario":
                raise Error("Error in parsing the XML")
                
            scenario = [-1] * len(self.sd.operatingConditionCriteria)
            for subchild in child:
                operatingCondition = subchild.tag
                value = subchild.text.strip()
                scenario[self.sd.operatingConditionCriteria.index(operatingCondition)] = self.sd.operatingConditionItems[operatingCondition].index(value)            
            allScenarios.append(scenario)
        
        for scenario in allScenarios:
            self.cm.insertTestCase(scenario)
            
        self.cm.printMetricQuantity()
        
        
class CoverageManagement:
    """For the set of inputs with semantic information contained, compute the quantitative k-projection coverage.

    Attributes:
        kValue: The constant k value for creating the coverage table
        numberOfCategorizations:
        maxType: 
        projectionRecords:
    """
    def __init__(self, kValue, numberOfCategorizations, maxType):
        

        if not ((kValue == 1 or kValue == 2) or kValue == 3):
            raise Exception('for k-projection coverage where k > 3, it is not supported')
            
        self.kValue = kValue
        self.numberOfCategorizations = numberOfCategorizations
        self.maxType = maxType
        self.projectionRecords = {}

        if (kValue == 1): 
            for i in range(numberOfCategorizations):  
                projectionType = [0] * 1
                projectionType[0] = maxType[i]
                self.projectionRecords[str(i)] = ProjectionRecord(kValue, projectionType)
            

        elif (kValue == 2): 
            for i in range(numberOfCategorizations):  
                for j in range(i+1, numberOfCategorizations):  
                    projectionType = [0] * 2
                    projectionType[0] = maxType[i]
                    projectionType[1] = maxType[j]
                    self.projectionRecords[str(i)+"_"+str(j)] = ProjectionRecord(kValue, projectionType)

            
        elif (kValue == 3): 
            for i in range(numberOfCategorizations):  
                for j in range(i+1, numberOfCategorizations):  
                    for k in range(j+1, numberOfCategorizations):  
                        projectionType = [0] * 3
                        projectionType[0] = maxType[i]
                        projectionType[1] = maxType[j]
                        projectionType[2] = maxType[k]
                        self.projectionRecords[str(i)+"_"+str(j)+"_"+str(k)] = ProjectionRecord(kValue, projectionType)
                    
    
    def printMetricQuantity(self):
    
        
        improvedItem = 0
        totalItems = 0
        for projectedCategorization, projRec in self.projectionRecords.items():
            #print(projectedCategorization)
            #print("\t"+str(projRec.currentlyOccupiedEntities))
            #print("\t"+str(projRec.maxOccupiedEntities))
            for value, currentQuantity in projRec.currentlyOccupiedEntities.items():
                totalItems = totalItems + projRec.maxOccupiedEntities[value]
                if (currentQuantity < projRec.maxOccupiedEntities[value]) :
                    # This item can be improved
                    improvedItem = improvedItem + (projRec.maxOccupiedEntities[value] - currentQuantity)
        
        
        print(str(self.kValue) + "-projection coverage (without considering domain restrictions): " + str(totalItems - improvedItem) + "/" + str(totalItems))


    def addDomainRestrictionConstraints(self):
        print("dummy")
    
    def insertTestCase(self, inputVector):
        
        #print(inputVector)
        
        if self.numberOfCategorizations != len(inputVector):
            return False
            
        if (self.kValue == 1) :
            for i in range(self.numberOfCategorizations):  
                self.projectionRecords[str(i)].currentlyOccupiedEntities[str(inputVector[i])] =  self.projectionRecords[str(i)].currentlyOccupiedEntities[str(inputVector[i])] + 1 
            return True
        elif (self.kValue == 2): 
            for i in range(self.numberOfCategorizations):  
                for j in range(i+1, self.numberOfCategorizations):  
                    self.projectionRecords[str(i)+ "_" + str(j)].currentlyOccupiedEntities[str(inputVector[i])+"_"+str(inputVector[j])] =  (
                        self.projectionRecords[str(i)+ "_" + str(j)].currentlyOccupiedEntities[str(inputVector[i])+"_"+str(inputVector[j])]  + 1)
            return True
        elif (self.kValue == 3): 
            for i in range(self.numberOfCategorizations):  
                for j in range(i+1, self.numberOfCategorizations):  
                    for k in range(j+1, self.numberOfCategorizations):  
                        self.projectionRecords[str(i)+ "_" + str(j)+"_" + str(k)].currentlyOccupiedEntities[str(inputVector[i])+"_"+str(inputVector[j])+"_"+str(inputVector[k])] = (
                            self.projectionRecords[str(i)+ "_" + str(j)+"_" + str(k)].currentlyOccupiedEntities[str(inputVector[i])+"_"+str(inputVector[j])+"_"+str(inputVector[k])]  + 1 )
            return True
        else: 
            return False
        

    def dumpMetricState(self):
        print(self.projectionRecords)

    
        
class ProjectionRecord():
    """Single table belonging to a particular projection.

    Attributes:
        kValue: The constant k value for creating the coverage table

    """

    def __init__(self, kValue, maxType):
            
        if not ((kValue == 1 or kValue == 2) or kValue == 3):
            raise Exception('for k-projection coverage where k > 3, it is not supported')
            
        self.kValue = kValue    
        self.currentlyOccupiedEntities = {}
        self.maxOccupiedEntities = {}
        
        if self.kValue == 1: 
            for i in range(maxType[0]):
                self.currentlyOccupiedEntities[str(i)] = 0
                # FIXME: Change below value from 1 to others when quantitative projection is required.
                self.maxOccupiedEntities[str(i)] = 1

            

        elif self.kValue == 2: 
            for i in range(maxType[0]):
                for j in range(maxType[1]):
                    self.currentlyOccupiedEntities[str(i)+"_"+str(j)] = 0
                    # FIXME: Change below value from 1 to others when quantitative projection is required.
                    self.maxOccupiedEntities[str(i)+"_"+str(j)] = 1
                
            
        elif self.kValue == 3:
            for i in range(maxType[0]):
                for j in range(maxType[1]):
                    for k in range(maxType[2]):
                        self.currentlyOccupiedEntities[str(i)+"_"+str(j)+"_"+str(k)] = 0
                        # FIXME: Change below value from 1 to others when quantitative projection is required.
                        self.maxOccupiedEntities[str(i)+"_"+str(j)+"_"+str(k)] = 1
        else:
            raise Error("Currently k>3 is not supported")

            
class ScenarioDescription():       

    def __init__(self, fileName):
        self.operatingConditionCriteria = [];
        self.operatingConditionItems = {};

        # Read and process input file
        tree = ET.parse(fileName)
        root = tree.getroot()

        for child in root:
            nodeName = child.tag
            options = [x.strip() for x in child.text.split(',')]
            self.operatingConditionItems[nodeName] = options
            self.operatingConditionCriteria.append(nodeName);
       
    def printScenarioDescription(self):
        print(str(self.operatingConditionCriteria))
        print(str(self.operatingConditionItems))
    
    
    
