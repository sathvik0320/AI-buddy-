#data preprocessing and data preparation 
#use sed s/"//g input_file > output_file" removes "
import pandas as pd
import numpy as np 
import time as t 
#df = pd.read_csv("data.csv").applymap(lambda x : x.strip('"') if isinstance(x,str) else x)
df = pd.read_csv("data.csv")
print(df.head())
#df.shape
if  df.isna().any().any():
       df.isna().sum()
       print("removing the nan values rows.....")
       t.sleep(3)
       df=df.dropna()

df["binary"]=[
    1 if x <= len(df["Time (s)"]) else 1 for x in range(0,len(df["Time (s)"]))]

x=df.drop("binary",axis=1)
y= df["binary"]

x= np.array(x)
y= np.array(y)
print(x[0:3])

from sklearn.model_selection import train_test_split
x_temp,x_val,y_temp,y_val = train_test_split(x,y,test_size=0.1,random_state=False,shuffle=False)
x_train,x_test,y_train,y_test=train_test_split(x_temp,y_temp,test_size=0.2,random_state=False,shuffle=False)

def create(x,t):
  d=[]
  for i in range(x.shape[0]-t-1):
    a= x[i:i+t]
    d.append(a)
  return np.array(d)

x_train = create(x_train,7)
y_train = create(y_train,7)
x_val = create(x_val,7)
y_val = create(y_val,7)
x_test = create(x_test,7)
y_test = create(y_test,7)

y_train = np.mean(y_train, axis=1).reshape(-1, 1)
y_val = np.mean(y_val, axis=1).reshape(-1, 1)
y_test = np.mean(y_test, axis=1).reshape(-1, 1)

#model architecture 
import tensorflow as tf
import keras 
from keras import layers 
from keras import models 


model = keras.Sequential()
model.add(layers.LSTM(128,return_sequences=True,input_shape=(7,7)))
model.add(layers.Dropout(0.2))
model.add(layers.LSTM(64,return_sequences=False))
model.add(layers.Dense(1,activation="sigmoid"))
model.summary()
model.compile(loss="binary_crossentropy",optimizer="adam",metrics=["accuracy","Precision","Recall"])
model.fit(x_train,y_train,epochs=5,validation_data=(x_val,y_val))
#classify the output
model.predict(x_test)

