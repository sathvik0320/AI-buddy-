#google colab------
#from google.colab import drive
#drive.mount("/content/gdrive")
#!ln -s /content/gdrive/MY\Drive/ /mydrive
#%cd  /content/gdrive/MyDrive

#pip install torch torcheval
import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from wordcloud import WordCloud,STOPWORDS,ImageColorGenerator

df = pd.read_csv("sms+spam+collection/SMSSpamCollection",sep="\t",names=["labels","messages"])
#this was my path to file
print(df.head(5))
df.describe()
df.duplicated()[:5]

#creating two labels
dh=df[df.labels =='ham']
ds=df[df.labels =='spam']

dht="".join(dh.messages.to_numpy().tolist())
dhs="".join(ds.messages.to_numpy().tolist())
#use wodcloud to plot these text

#dht,dhs have full text of ham and spam not need except plotting with wordcloud

#plotting number of spam and ham
plt.figure(figsize=(8,8))
sn.countplot(df.labels)

#imbalance data set is an issue for binary classification and other classifications so we can use upsampling and downsampling
#down sampling of ham or upsamping of spam should be done because of imblance of dataset
dhs=dh.sample(n=len(ds),random_state=44)

dm=pd.concat([ds,dhs]).reset_index(drop=True)
print(dm.head(5))

#plotting new  lbels after sampling
plt.figure(figsize=(7,7))
sn.countplot(dm.labels)

#only map function is important here
dm["length"]=df["messages"].apply(len)
average=dm.groupby("labels").mean(numeric_only=True)
dm["label"]=dm["labels"].map({"ham":0,"spam":1})
dm.head(5)
label=dm["label"].values

#split data
#importing necessary libraries for test and train data
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(dm["messages"],label,test_size=0.2,random_state=43)


#text preprocessing tokenization
import torchtext
from torchtext.data import get_tokenizer
tokenizer=get_tokenizer("spacy",language="en_core_web_sm")
xtraintokenizer=x_train.apply(tokenizer)
xtesttokenizer = x_test.apply(tokenizer)


#buiding vocabulary and apply that to messgaes data
from torchtext.vocab import build_vocab_from_iterator
#l=df["messages"].tolist ()
vocab = build_vocab_from_iterator(xtraintokenizer,specials=["<unk>"],max_tokens=500)
vocab.set_default_index(vocab["<unk>"])
#print(vocab.get_stoi())
#word to number mapping get_stoi()
#number of unique tokens
print(len(vocab.get_stoi()))


xtraintokenizer=xtraintokenizer.apply(vocab)
xtesttokenizer =xtesttokenizer.apply(vocab)

import torch
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as f


#print( torch.tensor(xtraintokenizer.to_numpy(), dtype=torch.int64))


#xtrainpad = pad_sequence(torch.tensor((xtraintokenizer.values)),batch_first=True)
#ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
#torchvision 0.17.0 requires torch==2.2.0, but you have torch 2.4.0 which is incompatible.
#Successfully installed nvidia-cudnn-cu12-9.1.0.70 nvidia-nccl-cu12-2.20.5 torch-2.4.0 torchtext-0.18.0 triton-3.0.0



#xtraintensor=xtraintokenizer.apply(torch.tensor)
xtraintensor=xtraintokenizer.apply(lambda x : torch.tensor(x,dtype=torch.int64))
xtesttensor =xtesttokenizer.apply(lambda x : torch.tensor(x,dtype=torch.int64))


l=50
xtraintensor = xtraintensor.apply(
    lambda x: x[:l] if len(x) > l else x )
xtesttensor = xtesttensor.apply(lambda x : x[:l] if len(x) > l else x)

traintensor=pad_sequence(xtraintensor, batch_first=True, padding_value=0)
testtensor = pad_sequence(xtesttensor,batch_first=True,padding_value=0)



y_train =torch.tensor(y_train,dtype=torch.float32)
y_test =torch.tensor(y_test,dtype=torch.float32)

print(y_train.shape)
print(y_test.shape)




#model architecture
class spam(torch.nn.Module):

  def __init__(self,embedding_dim,hidden_dim,mak_tokens,target_size):
     super(spam,self).__init__()
     self.hidden_dim = hidden_dim
     self.embedding = torch.nn.Embedding(mak_tokens,embedding_dim)

     #lstm takes embeddings as inputs and  ouputs hidden output
     self.lstm1= torch.nn.LSTM(embedding_dim,hidden_dim,batch_first=True,dropout=0.2,bidirectional=False)
     self.lstm2 =torch.nn.LSTM(hidden_dim,hidden_dim,batch_first=True,dropout=0,bidirectional=False)
     self.dropout = torch.nn.Dropout(0.2)
     self.linear = torch.nn.Linear(hidden_dim,target_size)
     #changing target size chages will help build Dense(n_neurons)
     self.sigmoid = torch.nn.Sigmoid()

  def forward(self,x):

     embedings = self.embedding(x)
     #print(embedings.size())
     #output,(ho,co)=self.lstm()
     #input to lstm is len(sentence),batch,input_size(embedding_dimesnion)
     #return_sequences=False, you can use h_n[-1] or lstm_out[:, -1, :]
     #print(embedings.size())
     #embedings.view(1,len(x),-1)
     output,_=self.lstm1(embedings)
     #print(output.size())
     #input to linear is numbner of features (numberof elements,input_features)
     output2,_ = self.lstm2(output)
     #print(output2.size())
     output_last = output2[:, -1, :]
     #print(output_last.size())
     finalout = self.linear(output_last)
     probability = self.sigmoid(finalout)
     #print(probability.size())
     #print("size of output from model")

     return probability



class spamlinear(torch.nn.Module):

       def __init__(self,embedding_dim,mak_tokens,target_size,target_in):

           super(spamlinear,self).__init__()
           self.embedding = torch.nn.Embedding(mak_tokens,embedding_dim)
           self.pooling = torch.nn.AvgPool1d(1)
           self.linear1=torch.nn.Linear(embedding_dim,target_in)
           self.drop = torch.nn.Dropout(0.2)
           self.linear2=torch.nn.Linear(24,target_size)
           self.relu = torch.nn.ReLU()
           self.sigmoid = torch.nn.Sigmoid()

       def forward(self,x):

          embeddings = self.embedding(x)
          #print(embeddings.size())
          emb = embeddings.permute(0,2,1)
          #print(emb.size())
          emb=torch.mean(emb,dim=2)
          #print(emb.size())
          emb=emb.squeeze(-1)
          #print(emb.size())
          l1 = self.linear1(emb)
          l1 = self.relu(l1)
          l1 = self.drop(l1)
          l2 = self.linear2(l1)
          l2 = self.sigmoid(l2)

          return  l2

cla = input("which mode linears or lstm")

if cla == "linears":

 embedding_dim=16
 mak_tokens=500
 target_in=24
 target_size=1

 model=spamlinear(embedding_dim,mak_tokens,target_size,target_in)
 optimizer = torch.optim.Adam(model.parameters(),lr=0.001)
 lossfunction=torch.nn.BCELoss()

 from torcheval.metrics import BinaryAccuracy
 metricv=BinaryAccuracy()
 metrict=BinaryAccuracy()

 from torch.utils.data import Dataset,DataLoader


if cla=="lstm" :

 embedding_dim = 16
 hidden_dim=20
 max_tokens=500
 target_size=1

 model = spam(embedding_dim,hidden_dim,max_tokens,target_size)
 lossfunction = torch.nn.BCELoss()
 #,lr = 0.00001,betas=(0.9,0.99),eps=10**-8,
 optimizer = torch.optim.Adam(model.parameters(),lr=0.0001)

 from torcheval.metrics import BinaryAccuracy
 metricv = BinaryAccuracy()
 metrict =BinaryAccuracy()
 #metric.update(input,target)
 #metric.compute()
 #metric.reset()
 from torch.utils.data import Dataset,DataLoader


y_train =y_train.view(-1,1)
print(y_train.size())
y_test = y_test.view(-1,1)
print(y_test.size())
#training the model
epochs = int(input("number of epochs for training"))

best =500
t=list()
v=list()
ep=list()
for epoch in range(epochs):
  print(str(epoch)+"epoch")
  #model.train() will tell model to act as training model

  metricv.reset()
  metrict.reset()

  model.train()
  trainavgLoss=0
  for msg,label in zip(traintensor,y_train):

         #metricv.reset()
         #metrict.reset()

         msg=msg.view(1,-1)
         optimizer.zero_grad()
         outputt = model(msg)
         outputt =outputt.view(-1)
         #print(outputt.size())
         #label=label.view(1,-1)
         #print(label)
         metrict.update(outputt,label)
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
           metricv.update(outputv,label)

  if best > valavgLoss:
     best =valavgLoss
     patience=10
  else :
    patience=patience - 1
    if patience ==0:
       break

  #we can use x=copy.deepcopy(model.state_dict()) to load  models best parameters

  trainavgLoss = trainavgLoss/len(y_train)
  valavgLoss = valavgLoss/len(y_test)

  trainavgLoss =torch.tensor(trainavgLoss,dtype=torch.float32)
  valavgLoss=torch.tensor(valavgLoss,dtype=torch.float32)

  a=0
  a=torch.tensor(a,dtype=torch.float32)

  t.append(trainavgLoss)
  v.append(valavgLoss)
  ep.append(epoch)

  #calculate accuracy
  print("trainigLoss"+str(trainavgLoss),end=" ")
  print("validationLoss"+str(valavgLoss),end=" ")
  print(metricv.compute())
  print(end=" ")
  print(metrict.compute())


#graph between validation loss and training loss vs number of epochs
#ep = list(range(epochs))
plt.figure(figsize=(7,7))
plt.plot(ep,t,"blue",label="training loss")
plt.plot(ep,v,"red",label="validation loss")
plt.title("spam with lstm")
plt.show()


