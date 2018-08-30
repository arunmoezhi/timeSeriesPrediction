import pandas as pd
from sklearn import preprocessing

data = {'score': [234,24,14,27,-74,46,73,-18,59,160]}
df = pd.DataFrame(data)
print(df)

min_max_scaler = preprocessing.MinMaxScaler()
np_scaled = min_max_scaler.fit_transform(df)
df_normalized = pd.DataFrame(np_scaled,columns=["x"])
print(df_normalized)
print(list(df_normalized))
