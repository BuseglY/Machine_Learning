# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures#polinom özelliklerini import eder

data=pd.read_csv("positions.csv")
print(data.columns)

level=data.iloc[:,1].values.reshape(-1,1)#x koordinatı
salary=data.iloc[:,2].values.reshape(-1,1)#y koordinatı

regression=LinearRegression()#instance
regression.fit(level,salary)

tahmin=regression.predict(np.array([[8.3]]))#8.3 seviyesinde olan birisinin aldığı maaşı tahmin eder
print(tahmin)

#Bu data için LinearRegression tercih edilmemeli.
#Sebebi->datanın çok fazla olması ve liear(doğrusal) regression da tahminlerin sapmasıdır.
#Tahmin sapması olmaması için doğrusal olmayan regresyona yani polinom regresyonuna ihitiyacımız olur. 

plt.scatter(level,salary,color="red")#grafik çizdirdik
plt.plot(level,regression.predict(level),color="blue")#her level için tahminleme yaparak çizgi oluşturur
plt.show()


regressionPoly= PolynomialFeatures(degree=4)#degree->detay seviyesidir(grafik için)
levelPoly=regressionPoly.fit_transform(level)#level değerlerini polinom görüntü haline getirir.(x in yapısını değiştirdik)
regression2=LinearRegression()#yeni x değişkenimiz için LinearRegression yapıcaz
regression2.fit(levelPoly,salary)#x ve y koordinatlarımız
tahmin2=regression2.predict(regressionPoly.fit_transform(np.array([[8.3]])))
#8.3 değeri için tahmin algoritmasını gerçekleştirdik
plt.plot(level,regression2.predict(levelPoly))#levelPoly için grafik çizdik
