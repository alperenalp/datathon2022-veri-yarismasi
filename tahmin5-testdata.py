#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sbn


# In[2]:


train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("testFeatures.csv")
train_data.head()


# In[3]:


#encoder: Kategorik -> numerik
from sklearn import preprocessing
le = preprocessing.LabelEncoder()


# In[4]:


# ürün sayısal hale getirme
urunSayisal = le.fit_transform(train_data.iloc[:,1])
urunSayisal_test = le.fit_transform(test_data.iloc[:, 2])

# üretim yerini sayısal hale getirme
uretimYeriSayisal = le.fit_transform(train_data.iloc[:,-3:-2])
uretimYeriSayisal_test = le.fit_transform(test_data.iloc[:,-3:-2])

#sehirleri sayısal hale getirme
ohe = preprocessing.OneHotEncoder()
sehirSayisal = ohe.fit_transform(train_data["şehir"].values.reshape(-1,1)).toarray()
sehirSayisal_test = ohe.fit_transform(test_data["şehir"].values.reshape(-1,1)).toarray()

#marketleri sayısal hale getirme
ohe = preprocessing.OneHotEncoder()
marketSayisal = ohe.fit_transform(train_data["market"].values.reshape(-1,1)).toarray()
marketSayisal_test = ohe.fit_transform(test_data["market"].values.reshape(-1,1)).toarray()

#ürün kategorisini sayısal hale getirme
ohe = preprocessing.OneHotEncoder()
urunKategoriSayisal = ohe.fit_transform(train_data["ürün kategorisi"].values.reshape(-1,1)).toarray()
urunKategoriSayisal_test = ohe.fit_transform(test_data["ürün kategorisi"].values.reshape(-1,1)).toarray()

#tarihi sayısal hale getirme
train_data['tarih'] = pd.to_datetime(train_data['tarih'])
train_data["Yıl"] = train_data["tarih"].dt.year
train_data["Ay"] = train_data["tarih"].dt.month
train_data["Gün"] = train_data["tarih"].dt.day

# Test verisi tarihi sayısal hale getirme
test_data['tarih'] = pd.to_datetime(test_data['tarih'])
test_data["Yıl"] = test_data["tarih"].dt.year
test_data["Ay"] = test_data["tarih"].dt.month
test_data["Gün"] = test_data["tarih"].dt.day


# In[5]:


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


# In[6]:


#dataframe birlestirme islemi
b1= pd.concat([sehirSayisal, train_data.iloc[:,-3:]], axis=1)
b2= pd.concat([marketSayisal, b1], axis=1)
b3= pd.concat([uretimYeriSayisal, b2], axis=1)
b4 = pd.concat([train_data.iloc[:,2], b3], axis=1)
b5= pd.concat([urunSayisal, b4], axis=1)
numerikVeri= pd.concat([urunKategoriSayisal, b5], axis=1)
numerikVeri = numerikVeri.drop(["Gün"], axis=1)


# In[7]:


#Testverisi dataframe birlestirme islemi
b1_test= pd.concat([sehirSayisal_test, test_data.iloc[:,-3:]], axis=1)
b2_test= pd.concat([marketSayisal_test, b1_test], axis=1)
b3_test= pd.concat([uretimYeriSayisal_test, b2_test], axis=1)
b4_test= pd.concat([test_data.iloc[:,3], b3_test], axis=1)
b5_test= pd.concat([urunSayisal_test, b4_test], axis=1)
numerikVeri_test= pd.concat([urunKategoriSayisal_test, b5_test], axis=1)
numerikVeri_test = numerikVeri_test.drop(["Gün"], axis=1)


# In[8]:


# veri setindeki istenen fiyat listesi
y=train_data.iloc[:,4]


# In[9]:


#scaling
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()


# In[10]:


"""scaler.fit(numerikVeri)
x_train = scaler.transform(numerikVeri)
x_test = scaler.transform(numerikVeri_test)
"""

x_train = scaler.fit_transform(numerikVeri)
x_test = scaler.transform(numerikVeri_test)


# In[11]:


import tensorflow as tf


# In[12]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


# In[13]:


model = Sequential()

model.add(Dense(512, activation="relu"))
model.add(Dense(512, activation="relu"))
model.add(Dense(256, activation="relu"))
model.add(Dense(64))
model.add(Dense(1))

model.compile(optimizer="adamax", loss="mse")

model.fit(x_train, y, batch_size=500,  epochs=100)


# In[14]:


import seaborn as sbn
loss= model.history.history["loss"]
sbn.lineplot(x=range(len(loss)), y=loss)


# In[15]:


kayipVerisi = pd.DataFrame(model.history.history)


# In[16]:


kayipVerisi.head()


# In[17]:


kayipVerisi.plot()


# In[18]:


trainLoss= model.evaluate(x_train, y, verbose=0)
trainLoss


# In[19]:


testTahminleri = model.predict(x_test)
testTahminleri


# In[20]:


#tahminleri yazdirma
tahmin = pd.DataFrame(data = testTahminleri, index = range(45504), columns=["ürün fiyatı"])
import sys
tahmin.to_csv(sys.stdout,columns=["ürün fiyatı"])
tahmin.to_csv("predict15.csv")


# In[ ]:




