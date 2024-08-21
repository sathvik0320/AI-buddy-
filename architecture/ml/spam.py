import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from wordcloud import WordCloud,STOPWORDS,ImageColorGenerator

df = pd.read_csv("/home/sathvik/Documents/sms+spam+collection/SMSSpamCollection",sep="\t",names=["labels","messages"])
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
sn.countplot(dm.labels)

dm["length"]=df["messages"].apply(len)
average=dm.groupby("labels").mean(numeric_only=True)
dm["label"]=dm["labels"].map({"ham":0,"spam":1})
dm.head(5)
label=dm["label"].values


#importing necessary libraries for test and train data
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(dm["messages"],label,test_size=0.2,random_state=43)

#text preprocessing tokenization,word padding,sequensing
import torchtext
import torchtext.data

tokenizer = torchtext.data.get_tokenizer("spacy",language="en_core_web_sm")
traintokenized = x_train.apply(tokenizer)
testtokenized = x_test.apply(tokenizer)
#use df.reset_index(drop=True) to reset indexes

#now vocab building
#we use torchtext.vocab for this

from torchtext.vocab import build_vocab_from_iterator

vocab = build_vocab_from_iterator(traintokenized,specials=["<unk>"],max_tokens=500)
vocab.set_default_index(vocab["<unk>"])
#check the word(string) to index values
print(len(vocab.get_stoi()))

#applying vocab on data frame so strings converted into intergers so computer can process important in nlp
tranvocab= traintokenized.apply(vocab)
testvocab = testtokenized.apply(vocab)
print(tranvocab.head(5))
print(testvocab.head(5))

print(tranvocab.dtype)

#now we need to make sure that length of training set is having equal length with padding

#necessary libraries for padding
import torch
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as f


#convert to pandas data frame for making sure it is pandasa frame
#tranvocab = pd.DataFrame(tranvocab)
#convert to tensors pytorch processes everything with tensors
tranvocab=tranvocab.apply(lambda x : torch.tensor(x,dtype=torch.float64))
testvocab=testvocab.apply(lambda x : torch.tensor(x,dtype=torch.float64))
#tranvocab["messages"]=tranvocab["messages"].apply(lambda x : torch.tensor) also gives same but default dtype

#length of the sequence for trainning and testing aslo matters so we need make sure every row have same length
#for messages more than specified max_length we delete the extra part and if it is less we padd 0's to it 
#same for testing also

l=int(input("max length (try different lengths and check the results if want)"))
tranpad = tranvocab.apply(lambda x : x=x[:l] if len(x) > l else (f.pad(x,pad=(0,l-len(x)),mode="constant",value=0) if len(x) < l else x) )
testpad = testvocab.apply(lambda x : x=[:l] if len(x)> l else(f.pad(x,pad=(0,l-len(x),mode="constant",value=0)) if len(x) < l else x))
#or just use the converted tensors to
#tranpad = tranvocab.apply(lambda x : x=x[:l] if len(x)>l else x)
#tranpad = pad_sequence(tranpad,batch_first=True,padd_value=0)
print(tranpad.shape)
print(testpad.shape)

#creating the model
def spam(torch.nn.Module):

       def __init__(self,embedding_dim,hidden_dim,mak_tokens,target_size):

         super(spam,self).__init__()
         self.hidden_dim = hidden_dim
         self.embedding = torch.nn.Embedding(mak_tokens,embedding_dim)

         #lstm takes embeddings as inputs and  ouputs hidden output
         self.lstm = torch.nn.LSTM(embedding_dim,hidden_dim,num_layers=2,dropout=0.2,bidirectional=False)
         self.dropout = torch.nn.Dropout(0.2)
         self.linear = torch.nn.Linear(hidden_dim,target_size)
         self.sigmoid = torch.nn.Sigmoid()




model = spam()
loss_function = torch.nn.MSELoss()
optimzer = torch.optim.Adam(model.parameters(),)










