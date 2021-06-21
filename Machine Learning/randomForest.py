import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
#ensemble->farklı ya da aynı lagoritmayı tekrar tekrar gerçekleştirilerek yapılan modeldir.

data=pd.read_csv("positions.csv")
level=data.iloc[:,1].values.reshape(-1,1)
salary=data.iloc[:,2].values

regression=RandomForestRegressor(n_estimators=10,random_state=0)#n_estimators->kaç tane decision tree oluşturacağımızı belirler
#random_state->ortaya çıkacak sonucu her seferinde farklı bir sayıyla ifade etmemesini sağlıyor
regression.fit(level,salary)

print(regression.predict(np.array([[8.3]])))
