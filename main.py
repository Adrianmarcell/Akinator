import pandas as pd
import torch
from random_forest import RandomForest
from synthesis import Feature_Vector

data = pd.read_csv('metal_data.csv')
X = data.drop(columns=['#', 'Name']).values
y = data['#'].values 
X, y = torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.int64)
metal_names = data['Name'].values

model = RandomForest()
model.fit(X, y)

vector = Feature_Vector()
test_vector = vector.predict_vector()

predictions = model.predict(test_vector.unsqueeze(dim=0))
predicted_index = predictions.item() - 1
print(f'logam yang anda pikirkan adalah: {metal_names[predicted_index]}')
