from nndependability.metrics import ScenarioKProjection

metric = ScenarioKProjection.Scenario_KProjection_Metric("data/kitti/description.xml")
metric.addScenariosFromFile("data/kitti/scenarios.xml")


from nndependability.atg.scenario import scenariogen
variableAssignment = scenariogen.proposeScenariocandidate(metric)
metric.writeScenarioToFile(variableAssignment, "tmp.xml")
