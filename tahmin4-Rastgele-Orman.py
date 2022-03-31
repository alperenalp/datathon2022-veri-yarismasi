# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 16:19:52 2022

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

# ürün sayısal hale getirme
urunSayisal = le.fit_transform(veriler.iloc[:,1])
urunSayisal_test = le.fit_transform(testVerisi.iloc[:, 2])


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
urunSayisal = pd.DataFrame(data = urunSayisal, index = range(227520), columns=["ürün"])
uretimYeriSayisal = pd.DataFrame(data = uretimYeriSayisal, index = range(227520), columns=["üYeri"])
sehirSayisal = pd.DataFrame(data = sehirSayisal, index = range(227520), columns=["sehirA","sehirB","sehirC","sehirD","sehirE","sehirF","sehirG","sehirH"])
marketSayisal = pd.DataFrame(data = marketSayisal, index = range(227520), columns=["marketB","marketC","marketM"])
urunKategoriSayisal = pd.DataFrame(data = urunKategoriSayisal, index = range(227520), columns=["et","kuruyemiş","meyve","sebze ve bakliyat", "süt ürünlei ve kahvaltılık", "tahıl ve ürünleri"])


#Test verisi numpy dizileri dataframe donusumu
urunSayisal_test = pd.DataFrame(data = urunSayisal_test, index = range(45504), columns=["ürün"])
uretimYeriSayisal_test = pd.DataFrame(data = uretimYeriSayisal_test, index = range(45504), columns=["üYeri"])
sehirSayisal_test = pd.DataFrame(data = sehirSayisal_test, index = range(45504), columns=["sehirA","sehirB","sehirC","sehirD","sehirE","sehirF","sehirG","sehirH"])
marketSayisal_test = pd.DataFrame(data = marketSayisal_test, index = range(45504), columns=["marketB","marketC","marketM"])
urunKategoriSayisal_test = pd.DataFrame(data = urunKategoriSayisal_test, index = range(45504), columns=["et","kuruyemiş","meyve","sebze ve bakliyat", "süt ürünlei ve kahvaltılık", "tahıl ve ürünleri"])


#dataframe birlestirme islemi
b1= pd.concat([sehirSayisal, veriler.iloc[:,-3:]], axis=1)
b2= pd.concat([marketSayisal, b1], axis=1)
b3= pd.concat([uretimYeriSayisal, b2], axis=1)
b4 = pd.concat([veriler.iloc[:,2], b3], axis=1)
b5= pd.concat([urunSayisal, b4], axis=1)
numerikVeri= pd.concat([urunKategoriSayisal, b5], axis=1)
numerikVeri = numerikVeri.drop(["Gün"], axis=1)


#Testverisi dataframe birlestirme islemi
b1_test= pd.concat([sehirSayisal_test, testVerisi.iloc[:,-3:]], axis=1)
b2_test= pd.concat([marketSayisal_test, b1_test], axis=1)
b3_test= pd.concat([uretimYeriSayisal_test, b2_test], axis=1)
b4_test= pd.concat([testVerisi.iloc[:,3], b3_test], axis=1)
b5_test= pd.concat([urunSayisal_test, b4_test], axis=1)
numerikVeri_test= pd.concat([urunKategoriSayisal_test, b5_test], axis=1)
numerikVeri_test = numerikVeri_test.drop(["Gün"], axis=1)

# veri setindeki istenen fiyat listesi
orj_sonuclar=veriler.iloc[:,4]




""""
#Random Forest(Rastgele Orman) Regresyon algoritması kullanarak eğitme
from sklearn.ensemble import RandomForestRegressor
rf_reg = RandomForestRegressor(n_estimators=100, random_state=0)
rf_reg.fit(numerikVeri, orj_sonuclar.ravel())
rf_tahmin = rf_reg.predict(numerikVeri_test)


#tahminleri yazdirma
tahmin = pd.DataFrame(data = rf_tahmin, index = range(45504), columns=["ürün fiyatı"])
import sys
tahmin.to_csv(sys.stdout,columns=["ürün fiyatı"])
tahmin.to_csv("predict5.csv")



"""








#Train verisi tahmin sonuçları

# train.csv'deki verilerin eğitim ve test için bölünmesi
orj_sonuclar=veriler.iloc[:,4]
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(numerikVeri, orj_sonuclar, test_size=0.33, random_state=0)



#Random Forest(Rastgele Orman) Regresyon algoritması kullanarak eğitme
from sklearn.ensemble import RandomForestRegressor
rf_reg = RandomForestRegressor(n_estimators=10, random_state=0)
rf_reg.fit(x_train, y_train.ravel())
tahmin1=  rf_reg.predict(x_test)


# Algoritma Degerlendirme skor hesaplama
from sklearn.metrics import r2_score
print('\nR2 Skor: ')
print(r2_score(y_test, rf_reg.predict(x_test)))


test_dogrulugu = rf_reg.score(x_test, y_test)
print(f"\nAccuary: {test_dogrulugu}")

from sklearn.metrics import mean_absolute_error, mean_squared_error

absolute_error = mean_absolute_error(y_test, tahmin1)
print(f"\nmean_absolute_error: {absolute_error} ")

squared_error = mean_squared_error(y_test, tahmin1)
print(f"\nmean_squared_error: {squared_error} ")



testTahminleri = pd.Series(tahmin1.reshape(75082,))
tahminDF = pd.concat([y_test, testTahminleri], axis=1)
tahminDF.columns = ["Gerçek Y", "Tahmin Y"]
import seaborn as sbn
sbn.scatterplot(x="Gerçek Y", y="Tahmin Y", data=tahminDF)




"""
#backward elemination
import statsmodels.api as sm
X_l = numerikVeri.iloc[:,[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]].values
X_l = np.array(X_l, dtype=float)
model = sm.OLS(orj_sonuclar, X_l).fit()
print(model.summary())

#pi degeri yuksek olani eleme
# train.csv'deki verilerin eğitim ve test için bölünmesi
orj_sonuclar=veriler.iloc[:,4]
numerikVeri=numerikVeri.drop(["üYeri"], axis=1)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(numerikVeri, orj_sonuclar, test_size=0.33, random_state=0)


#Random Forest(Rastgele Orman) Regresyon algoritması kullanarak eğitme
from sklearn.ensemble import RandomForestRegressor
rf_reg = RandomForestRegressor(n_estimators=100, random_state=0)
rf_reg.fit(x_train, y_train.ravel())
tahmin2=  rf_reg.predict(x_test)

# Algoritma Degerlendirme skor hesaplama
from sklearn.metrics import r2_score
print('\nDeğişken çıkarılmış R2 Skor: ')
print(r2_score(y_test, rf_reg.predict(x_test)))
"""








