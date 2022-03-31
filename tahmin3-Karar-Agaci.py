# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 19:51:09 2022

@author: Alperen
"""


#kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#veriseti yükleme
veriler = pd.read_csv('train.csv')
testVerisi = pd.read_csv('testFeatures.csv')

#encoder: Kategorik -> numerik
from sklearn import preprocessing
le = preprocessing.LabelEncoder()

# üretim yerini sayısal hale getirme
uretimYeriSayisal = le.fit_transform(veriler.iloc[:,-3:-2])
uretimYeriSayisal_test = le.fit_transform(testVerisi.iloc[:,-3:-2])

#sehirleri sayısal hale getirme
ohe = preprocessing.OneHotEncoder()
sehirSayisal = ohe.fit_transform(veriler["şehir"].values.reshape(-1,1)).toarray()
sehirSayisal_test = ohe.fit_transform(testVerisi["şehir"].values.reshape(-1,1)).toarray()

#marketleri sayısal hale getirme
ohe = preprocessing.OneHotEncoder()
marketSayisal = ohe.fit_transform(veriler["market"].values.reshape(-1,1)).toarray()
marketSayisal_test = ohe.fit_transform(testVerisi["market"].values.reshape(-1,1)).toarray()

#ürün kategorisini sayısal hale getirme
ohe = preprocessing.OneHotEncoder()
urunKategoriSayisal = ohe.fit_transform(veriler["ürün kategorisi"].values.reshape(-1,1)).toarray()
urunKategoriSayisal_test = ohe.fit_transform(testVerisi["ürün kategorisi"].values.reshape(-1,1)).toarray()

#tarihi sayısal hale getirme
veriler['tarih'] = pd.to_datetime(veriler['tarih'])
veriler["Yıl"] = veriler["tarih"].dt.year
veriler["Ay"] = veriler["tarih"].dt.month
veriler["Gün"] = veriler["tarih"].dt.day

# Test verisi tarihi sayısal hale getirme
testVerisi['tarih'] = pd.to_datetime(testVerisi['tarih'])
testVerisi["Yıl"] = testVerisi["tarih"].dt.year
testVerisi["Ay"] = testVerisi["tarih"].dt.month
testVerisi["Gün"] = testVerisi["tarih"].dt.day


#numpy dizileri dataframe donusumu
uretimYeriSayisal = pd.DataFrame(data = uretimYeriSayisal, index = range(227520), columns=["üYeri"])
sehirSayisal = pd.DataFrame(data = sehirSayisal, index = range(227520), columns=["sehirA","sehirB","sehirC","sehirD","sehirE","sehirF","sehirG","sehirH"])
marketSayisal = pd.DataFrame(data = marketSayisal, index = range(227520), columns=["marketB","marketC","marketM"])
urunKategoriSayisal = pd.DataFrame(data = urunKategoriSayisal, index = range(227520), columns=["et","kuruyemiş","meyve","sebze ve bakliyat", "süt ürünlei ve kahvaltılık", "tahıl ve ürünleri"])


#Test verisi numpy dizileri dataframe donusumu
uretimYeriSayisal_test = pd.DataFrame(data = uretimYeriSayisal_test, index = range(45504), columns=["üYeri"])
sehirSayisal_test = pd.DataFrame(data = sehirSayisal_test, index = range(45504), columns=["sehirA","sehirB","sehirC","sehirD","sehirE","sehirF","sehirG","sehirH"])
marketSayisal_test = pd.DataFrame(data = marketSayisal_test, index = range(45504), columns=["marketB","marketC","marketM"])
urunKategoriSayisal_test = pd.DataFrame(data = urunKategoriSayisal_test, index = range(45504), columns=["et","kuruyemiş","meyve","sebze ve bakliyat", "süt ürünlei ve kahvaltılık", "tahıl ve ürünleri"])


#dataframe birlestirme islemi
b1= pd.concat([sehirSayisal, veriler.iloc[:,-3:]], axis=1)
b2= pd.concat([marketSayisal, b1], axis=1)
b3= pd.concat([uretimYeriSayisal, b2], axis=1)
b4 = pd.concat([veriler.iloc[:,2], b3], axis=1)
numerikVeri= pd.concat([urunKategoriSayisal, b4], axis=1)


#Testverisi dataframe birlestirme islemi
b1_test= pd.concat([sehirSayisal_test, testVerisi.iloc[:,-3:]], axis=1)
b2_test= pd.concat([marketSayisal_test, b1_test], axis=1)
b3_test= pd.concat([uretimYeriSayisal_test, b2_test], axis=1)
b4_test= pd.concat([testVerisi.iloc[:,3], b3_test], axis=1)
numerikVeri_test= pd.concat([urunKategoriSayisal_test, b4_test], axis=1)




# train.csv'deki verilerin eğitim ve test için bölünmesi
orj_sonuclar=veriler.iloc[:,4]
sol = numerikVeri.iloc[:,:4]
sag = numerikVeri.iloc[:,5:]
sonVeri = pd.concat([sol,sag],axis = 1)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(sonVeri, orj_sonuclar, test_size=0.33, random_state=0)



# Decision Tree (Karar Agaci) regresyon kullanarak eğitme
from sklearn.tree import DecisionTreeRegressor
dt_reg = DecisionTreeRegressor(random_state=0)
dt_reg.fit(x_train, y_train)
dt_tahmin = dt_reg.predict(x_test)

# Algoritma Degerlendirme skor hesaplama
from sklearn.metrics import r2_score
print('\nR2 Degeri')
print(r2_score(y_test, dt_tahmin))


#backward elemination
import statsmodels.api as sm
X_l = sonVeri.iloc[:,[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]].values
X_l = np.array(X_l, dtype=float)
model = sm.OLS(orj_sonuclar, X_l).fit()
print(model.summary())


#pi degeri yuksek olani eleme
# train.csv'deki verilerin eğitim ve test için bölünmesi
orj_sonuclar=veriler.iloc[:,4]
sonVeri=sonVeri.drop(["üYeri"], axis=1)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(sonVeri, orj_sonuclar, test_size=0.33, random_state=0)


# Decision Tree (Karar Agaci) regresyon kullanarak eğitme
from sklearn.tree import DecisionTreeRegressor
dt_reg = DecisionTreeRegressor(random_state=0)
dt_reg.fit(x_train, y_train)
dt_tahmin2 = dt_reg.predict(x_test)

# Algoritma Degerlendirme skor hesaplama
from sklearn.metrics import r2_score
print('\nÜretim yeri çıkarılmış R2 Degeri')
print(r2_score(y_test, dt_tahmin2))






# DATATHON 2022 Predict3_backward_elemination Skor 17.5
"""

# veri setindeki istenen fiyat listesi
orj_sonuclar=veriler.iloc[:,4]


# Decision Tree (Karar Agaci) regresyon kullanarak eğitme
from sklearn.tree import DecisionTreeRegressor
dt_reg = DecisionTreeRegressor(random_state=0)
dt_reg.fit(numerikVeri, orj_sonuclar)
dt_tahmin = dt_reg.predict(numerikVeri_test)


#backward elemination
import statsmodels.api as sm
X_l = numerikVeri.iloc[:,[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]].values
X_l = np.array(X_l, dtype=float)
model = sm.OLS(orj_sonuclar, X_l).fit()
print(model.summary())


#pi degeri yuksek olani eleme
numerikVeri=numerikVeri.drop(["üYeri"], axis=1)
numerikVeri_test = numerikVeri_test.drop(["üYeri"], axis=1)

# Decision Tree (Karar Agaci) regresyon kullanarak eğitme
from sklearn.tree import DecisionTreeRegressor
dt_reg = DecisionTreeRegressor(random_state=0)
dt_reg.fit(numerikVeri, orj_sonuclar)
dt_tahmin2 = dt_reg.predict(numerikVeri_test)


#tahminleri yazdirma
tahmin = pd.DataFrame(data = dt_tahmin2, index = range(45504), columns=["ürün fiyatı"])
import sys
tahmin.to_csv(sys.stdout,columns=["ürün fiyatı"])
tahmin.to_csv("Predict3_backward_elemination.csv")
"""








# DATATHON 2022 Predict3 Score:17.4
"""
# veri setindeki istenen fiyat listesi
orj_sonuclar=veriler.iloc[:,4]


# Decision Tree (Karar Agaci) regresyon kullanarak eğitme
from sklearn.tree import DecisionTreeRegressor
dt_reg = DecisionTreeRegressor(random_state=0)
dt_reg.fit(numerikVeri, orj_sonuclar)
dt_tahmin = dt_reg.predict(numerikVeri_test)



#tahminleri yazdirma
tahmin = pd.DataFrame(data = dt_tahmin, index = range(45504), columns=["ürün fiyatı"])
import sys
tahmin.to_csv(sys.stdout,columns=["ürün fiyatı"])
tahmin.to_csv("predict3.csv")
"""




print("Tahmin Başarıyla Gerçekleşti")
