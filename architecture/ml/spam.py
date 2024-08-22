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

y_train =torch.tensor(y_train,dtype=torch.float32)
y_test =torch.tensor(y_test,dtype=torch.float32)

y_train =y_train.view(-1,1)
print(y_train.size())
y_test = y_test.view(-1,1)
print(y_test.size())


#creating the model
def spam(torch.nn.Module):

       def __init__(self,embedding_dim,hidden_dim,mak_tokens,target_size):

         super(spam,self).__init__()
         self.hidden_dim = hidden_dim
         self.embedding = torch.nn.Embedding(mak_tokens,embedding_dim)

         #lstm takes embeddings as inputs and  ouputs hidden output
         self.lstm = torch.nn.LSTM(embedding_dim,hidden_dim,num_layers=2,batch_first=false,dropout=0.2,bidirectional=False)
         self.dropout = torch.nn.Dropout(0.2)
         self.linear = torch.nn.Linear(hidden_dim,target_size)
         self.sigmoid = torch.nn.Sigmoid()

       def forward(self,x)

        embedings = self.embedding(x)
        output,(ho,co)=self.lstm(embedings)
        finalout = self.linear(output_last
        finaloutput = self.linear(output)
        probability = self.Sigmoid(output)
        #print(embedings.size())
        #output,(ho,co)=self.lstm()
        #input to lstm is len(sentence),batch,input_size(embedding_dimesnion)
        #return_sequences=False, you can use h_n[-1] or lstm_out[:, -1, :]
        #print(embedings.size())
        #embedings.view(1,len(x),-1)
        #print(output.size())
        #input to linear is numbner of features (numberof elements,input_features 

        return probability


embedding_dim = 16
hidden_dim=20
max_tokens=500
target_size=1

model = spam(embedding_dim,hidden_dim,max_tokens,target_size)
lossfunction = torch.nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(),lr = 0.9,betas=(0.9,0.99),eps=10**-6,weight_decay=0.002)

from torcheval.metrics import BinaryAccuracy
metric = BinaryAccuracy()
#metric.update(input,target)
#metric.compute()
#metric.reset()
from torch.utils.data import Dataset,DataLoader

#training the model
epochs = int(input("number of epochs for training"))


best =500
for epoch in range(epochs):
  #model.train() will tell model to act as training model

  model.train()
  trainavgLoss=0
  for msg,label in zip(traintensor,y_train):

         msg=msg.view(1,-1)
         #print(msg.size())
         model.zero_grad()
         outputt = model(msg)
         outputt =outputt.view(-1)
         #label=label.view(1,-1)

         losst = lossfunction(outputt,label)
         losst.backward()
         optimizer.step()
         trainavgLoss += losst.item()

  model.eval()
  valavgLoss=0
  with torch.no_grad():
     for msg,label in zip(testtensor,y_test):

           msg=msg.view(1,-1)
           outputv = model(msg)
           outputv =outputv.view(-1)
           lossv = lossfunction(outputv,label)
           #lossv.backward()
           #optimizer.step()
           valavgLoss += lossv.item()

  trainavgLoss = trainavgLoss/len(y_train)
  valavgLoss = valavgLoss/len(y_test)

  trainavgLoss =torch.tensor(trainavgLoss,dtype=torch.float32)
  valavgLoss=torch.tensor(valavgLoss,dtype=torch.float32)

  a=0
  a=torch.tensor(a,dtype=torch.float32)

  #early stopping
  if best > valavgLoss:
      best = valavgLoss
      bestepoch = epoch


  #calculate accuracy
  a=(100-valavgLoss)%100


print(bestepoch)






