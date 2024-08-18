import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from wordcloud import WordCloud,STOPWORDS,ImageColorGenerator

df = pd.read_csv("sms+spam+collection/SMSSpamCollection",sep="\t",names=["labels","messages"])
df.head(5)


df.describe()
#creating two labels
dh=df[df.labels =='ham']
ds=df[df.labels =='spam']

dht="".join(dh.messages.to_numpy().tolist())
dhs="".join(ds.messages.to_numpy().tolist())
#use wodcloud to plot these text

#plotting number of spam and ham
plt.figure(figsize=(8,8))
sn.countplot(df.labels)

dhs=dh.sample(n=len(ds),random_state=44)

dm=pd.concat([ds,dhs]).reset_index(drop=True)
dm.head(5)

plt.figure(figsize=(7,7))
sn.countplot(dm.label

dm["length"]=df["messages"].apply(len)
average=dm.groupby("labels").mean(numeric_only=True)
dm["label"]=dm["labels"].map({"ham":0,"spam":1})
dm.head(5)
label=dm["label"].values


#importing necessary libraries for test and train data
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(dm["messages"],label,test_size=0.2,random_state=43)

#text preprocessing tokenization,word padding,sequensing


