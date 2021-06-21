import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score#R Square için

data=pd.read_csv("hw_25000.csv")

boy=data.Height.values.reshape(-1,1)#reshape i LinearRegresyon bu boyutta istediği için yaptık
kilo=data.Weight.values.reshape(-1,1) 

regression=LinearRegression()#Lineer Regresyon sınıfından örnek oluşturduk
regression.fit(boy,kilo)#boy ve kilo dataları için eğim hesaplamaları yaparak bizim istediğimiz değeri tahmin etmeye yarar
print(regression.predict(np.array([[71]])))#•71 boyundaki bir kişinin kilo tahminini yapar

print(data.columns)

plt.scatter(data.Height,data.Weight)#grafik oluşturmak için x ve y değişkenlerini verdik
x=np.arange(min(data.Height),max(data.Height)).reshape(-1,1)#
plt.plot(x,regression.predict(x),color="red")#her x değeri için y nin tahminini yapar
plt.xlabel("Boy")#x koordinatını isimlendirdik
plt.ylabel("Kilo")#y koordinatını isimlendirdik
plt.title("Simple Linear Regression Model")#grafiğimizi isimlendirdik
plt.show()

#R Square yöntemiyle algoritmanın başarı oranı test edilir.
print(r2_score(kilo,regression.predict(boy)))#doğruluk payını verir


























