import pandas as pd

df = pd.read_csv("data/food_inspection_proof.csv")
indexNames = df[(df.Results == 'Pass') & (df.Risk == 'Risk 1 (High)')].index
print(len(indexNames))
indexNames = df[(df.Results == 'Fail') & (df.Risk == 'Risk 3 (Low)')].index
print(len(indexNames))




