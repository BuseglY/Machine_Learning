# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

data=pd.read_csv("insurance.csv")
print(data.columns)
print(data.describe())#kolonlar hakkında bilgi

#y eksenini oluşturduk
expenses=data.expenses.values.reshape(-1,1)
#x eksenini oluşturduk
ageBmis=data.iloc[:,[0,2]].values#x  ekseni için age ve Bmi kolonlarını aldık.
#bütün satırların age ve bmi kolon bilgilerinin alınmasını sağladık.

regression=LinearRegression()#linearRegression instance ını oluşturduk
regression.fit(ageBmis,expenses)#x ve y değerlerimizi regresyonumuza aktardık
print(regression.predict(np.array([[20,20]])))#20 yaşında ve 20 bmi ye sahip bir kişinin ortalama harcamasını verir


















