import pandas as pd
import os

data = pd.read_csv("final_data.csv")

from pgmpy.estimators import BicScore,BdeuScore,ExhaustiveSearch,HillClimbSearch

est = HillClimbSearch(data, scoring_method=BdeuScore(data, equivalent_sample_size=10))

best_model = est.estimate()
print("Structure found!")
print("Best Model")
print(best_model.edges())

'''
Parameters Estimation and CPD tables
'''
from pgmpy.models import BayesianModel
from pgmpy.estimators import BayesianEstimator

model = BayesianModel(best_model.edges())
data = pd.read_csv("final_training_set.csv")

estimator = BayesianEstimator(model, data)
parameters = estimator.get_parameters(prior_type='BDeu', equivalent_sample_size=10)

for cpd in parameters:
    model.add_cpds(cpd)

'''
Inference and Validation
'''
from pgmpy.inference import VariableElimination
import csv 

f = open("validation_data.csv")
reader = csv.reader(f)

inference = VariableElimination(model)
valid=0
invalid=0
for row in reader:
    br=row[3]
    ig=row[1]
    #map_quey returns Dictionary!!!
    if int(row[0])==inference.map_query(["SC"], evidence={"BR": int(br)})["SC"]:
        valid+=1
    else:
        invalid+=1

total=valid+invalid
print(valid)
print(invalid)
accuracy = float(valid/total)
print("Accuracy: ", accuracy)     
f.close()

os.system('spd-say  -i -10 -p 50 -t female3  "Training Finished!"')




 
 
 
 
 
 
 
 
    
    







