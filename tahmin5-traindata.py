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
#train_data.head()


# In[3]:


#sbn.displot(train_data["ürün fiyatı"])


# In[4]:


##train_data = train_data.sort_values("ürün fiyatı", ascending=False).iloc[2000:]
#sbn.displot(train_data["ürün fiyatı"])


# In[ ]:





# In[5]:


train_data.tail(20)


# In[6]:


#encoder: Kategorik -> numerik
from sklearn import preprocessing
le = preprocessing.LabelEncoder()


# In[7]:


# ürün sayısal hale getirme
urunSayisal = le.fit_transform(train_data.iloc[:,1])

# üretim yerini sayısal hale getirme
uretimYeriSayisal = le.fit_transform(train_data.iloc[:,-3:-2])

#sehirleri sayısal hale getirme
ohe = preprocessing.OneHotEncoder()
sehirSayisal = ohe.fit_transform(train_data["şehir"].values.reshape(-1,1)).toarray()

#marketleri sayısal hale getirme
ohe = preprocessing.OneHotEncoder()
marketSayisal = ohe.fit_transform(train_data["market"].values.reshape(-1,1)).toarray()

#ürün kategorisini sayısal hale getirme
ohe = preprocessing.OneHotEncoder()
urunKategoriSayisal = ohe.fit_transform(train_data["ürün kategorisi"].values.reshape(-1,1)).toarray()


#tarihi sayısal hale getirme
train_data['tarih'] = pd.to_datetime(train_data['tarih'])
train_data["Yıl"] = train_data["tarih"].dt.year
train_data["Ay"] = train_data["tarih"].dt.month
train_data["Gün"] = train_data["tarih"].dt.day


# In[8]:


#numpy dizileri dataframe donusumu
urunSayisal = pd.DataFrame(data = urunSayisal, index = range(227520), columns=["ürün"])
uretimYeriSayisal = pd.DataFrame(data = uretimYeriSayisal, index = range(227520), columns=["üYeri"])
sehirSayisal = pd.DataFrame(data = sehirSayisal, index = range(227520), columns=["sehirA","sehirB","sehirC","sehirD","sehirE","sehirF","sehirG","sehirH"])
marketSayisal = pd.DataFrame(data = marketSayisal, index = range(227520), columns=["marketB","marketC","marketM"])
urunKategoriSayisal = pd.DataFrame(data = urunKategoriSayisal, index = range(227520), columns=["et","kuruyemiş","meyve","sebze ve bakliyat", "süt ürünlei ve kahvaltılık", "tahıl ve ürünleri"])


# In[9]:


#dataframe birlestirme islemi
b1= pd.concat([sehirSayisal, train_data.iloc[:,-3:]], axis=1)
b2= pd.concat([marketSayisal, b1], axis=1)
b3= pd.concat([uretimYeriSayisal, b2], axis=1)
b4 = pd.concat([train_data.iloc[:,2], b3], axis=1)
b5= pd.concat([urunSayisal, b4], axis=1)
numerikVeri= pd.concat([urunKategoriSayisal, b5], axis=1)
numerikVeri = numerikVeri.drop(["Gün"], axis=1)


# In[10]:


# veri setindeki istenen fiyat listesi
y=train_data.iloc[:,4]


# In[11]:


from sklearn.model_selection import train_test_split


# In[12]:


x = numerikVeri
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=0)


# In[13]:


type(y_test)


# In[14]:


#scaling
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()


# In[15]:


"""
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
"""

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


# In[16]:


import tensorflow as tf


# In[17]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


# In[18]:


model = Sequential()

model.add(Dense(512, activation="relu"))
model.add(Dense(512, activation="relu"))
model.add(Dense(256, activation="relu"))
model.add(Dense(64, activation="relu"))
model.add(Dense(32, activation="relu"))
model.add(Dense(1))

model.compile(optimizer="adamax", loss="mse")

model.fit(x_train, y_train, validation_data=(x_test,y_test), batch_size=1000,  epochs=100)


# In[19]:


import seaborn as sbn
loss= model.history.history["loss"]
sbn.lineplot(x=range(len(loss)), y=loss)


# In[20]:


kayipVerisi = pd.DataFrame(model.history.history)


# In[21]:


kayipVerisi.head()


# In[22]:


kayipVerisi.plot()


# In[23]:


trainLoss= model.evaluate(x_train, y_train, verbose=0)
trainLoss


# In[24]:


testLoss = model.evaluate(x_test, y_test, verbose=0)
testLoss


# In[25]:


testTahminleri = model.predict(x_test)
testTahminleri


# In[26]:


tahminDF = pd.DataFrame(y_test.values, columns=["Gerçek Y"])


# In[27]:


tahminDF


# In[28]:


testTahminleri = pd.Series(testTahminleri.reshape(75082,))


# In[29]:


testTahminleri


# In[30]:


tahminDF = pd.concat([tahminDF, testTahminleri], axis=1)
tahminDF.columns = ["Gerçek Y", "Tahmin Y"]


# In[31]:


tahminDF.head(50)


# In[32]:


sbn.scatterplot(x="Gerçek Y", y="Tahmin Y", data=tahminDF)


# In[33]:


from sklearn.metrics import mean_absolute_error, mean_squared_error


# In[34]:


train_data.describe()


# In[35]:


mean_absolute_error(tahminDF["Gerçek Y"], tahminDF["Tahmin Y"])


# In[36]:


mean_squared_error(tahminDF["Gerçek Y"], tahminDF["Tahmin Y"])


# In[37]:


numerikVeri


# In[38]:


yeni_urun=[[1, 0, 0, 0, 0, 0, 65, 120, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 2021, 1]]
yeni_urun = scaler.transform(yeni_urun)
model.predict(yeni_urun)


# In[ ]:




