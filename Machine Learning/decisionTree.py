import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor

data=pd.read_csv("positions.csv")
level=data.iloc[:,1].values.reshape(-1,1)#x
salary=data.iloc[:,2].values.reshape(-1,1)#y

regression=DecisionTreeRegressor()#instance
regression.fit(level,salary)
print(regression.predict(np.array([[8.3]])))

plt.scatter(level,salary,color="red")
x=np.arange(min(level),max(level),0.01).reshape(-1,1)#0.01 aralıklarıyla değerleri gösterir
plt.plot(x,regression.predict(x),color="orange")#bütün x değerlerimiz için çizim yaptırdık
plt.xlabel("Level")
plt.ylabel("Salary")
plt.title("Decision Tree Model")
plt.show()

