# -*- coding: utf-8 -*-
"""
Created on Sun Mar 27 18:20:05 2022

@author: Alperen
"""

#kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#veriseti yükleme
veriler = pd.read_csv('train.csv')
testVerisi = pd.read_csv('testFeatures.csv')

#test
print(veriler.head(50))

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





# eğitim işlemi
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(numerikVeri, veriler.iloc[:,4])


# tahmin işlemi
y_pred = regressor.predict(numerikVeri_test)



#tahminleri yazdirma
tahmin = pd.DataFrame(data = y_pred, index = range(45504), columns=["ürün fiyatı"])
import sys
tahmin.to_csv(sys.stdout,columns=["id","ürün fiyatı"])

tahmin.to_csv("predict1.csv")

    




"""
skor= np.mean((y_test - y_pred)**2)
print(skor)
"""