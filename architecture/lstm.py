import numpy as np
import random
import math
import time
#data preprocessing
import numpy as np
import pandas as pd
import  matplotlib.pyplot as plt

def data():
  ds  = pd.read_csv("/home/sathvik/YESBANK.NS.csv")
  print(ds.head())
  print(ds.tail())
  ds_s = ds.loc[:,["High","Low","Volume"]]
  ds_s = ds_s.reset_index(drop = True )
  print("after reseting the to high low volume\n")
  print(ds_s.head())
  print(ds_s.tail())

  #use this if want to plot

  #plt.plot(ds_s["High"],label = "High")
  #plt.plot(ds_s["Low"],label = "Low")
  #plt.plot(ds_s["Volume"],label ="Volume")
  #plt.legend()
  #plt.show()

  #normalzation of data
  min = ds_s.min()
  max = ds_s.max()
  dn = (ds_s - min)/(max - min)
  print("after normalization of data\n")
  print(dn.head())
  print(dn.tail())
  print("converting pandas to numpy array\n")
  dn = np.array(dn)
  return dn

def weights(hd_l,xl):
    #weights should in dimesniosns of length of hidden state X (hiddenstate + input)
    wf = np.random.rand(hd_l,hd_l+xl)
    wi = np.random.rand(hd_l,hd_l+xl)
    wc = np.random.rand(hd_l,hd_l+xl)
    wo = np.random.rand(hd_l,hd_l+xl)

    wim = np.random.rand(hd_l,hd_l+xl)
    wom = np.random.rand(hd_l,hd_l+xl)
    wfm = np.random.rand(hd_l,hd_l+xl)
    wcm = np.random.rand(hd_l,hd_l+xl)

    wiv = np.random.rand(hd_l,hd_l+xl)
    wov = np.random.rand(hd_l,hd_l+xl)
    wfv = np.random.rand(hd_l,hd_l+xl)
    wcv = np.random.rand(hd_l,hd_l+xl)
    #biases ,w1m,w2m,wim,wom,wfm,wcm,w1v,w2v,wiv,wov,wfm,wcv
    # biases dimensions are equal to hidden state dimensions because after matrix multiplication resultant matrix will be n dimensions of hidden state (where we add biases to it so ...)
    bf = np.random.rand(hd_l)
    bi = np.random.rand(hd_l)
    bc = np.random.rand(hd_l)
    bo = np.random.rand(hd_l)
    hp = np.random.rand(hd_l)
    cp = np.random.rand(hd_l)
    w1= np.random.rand(d1,hd_l)
    w2= np.random.rand(d2,d1)

    w1m=np.random.rand(d1,hd_l)
    w2m=np.random.rand(d2,d1)
    w1v=np.random.rand(d1,hd_l)
    w2v=np.random.rand(d2,d1)

    b1=np.random.rand(d1)
    b2=np.random.rand(d2)

    return wf,wi,wc,wo,bf,bi,bc,bo,hp,cp,w1,b1,w2,b2,w1m,w2m,wim,wom,wfm,wcm,w1v,w2v,wiv,wov,wfm,wcv

def sigmoid(x):
    return 1 / (1+np.exp(-x))


def tanh(x):
  # we use tanh activaton function in calculating of a C^ or g state (candidate state)
  # we use tanh in  calculating the ht hidden state ot*tanh(ct) (cell state)
  return np.tanh(x)


def tanh_d(x):
    return 1 - np.tanh(x)**2


def sigmoid_d(x):
  # sigmoid derivative is used in calculating the delta of layer or neuron where we use (y(1-y))
  return x*(1 - x)


def forgetgate(wf,co,bf):
   wn = np.dot(wf,co)
   return sigmoid(wn + bf)

def c_(wc,co,bc):
  wn = np.dot(wc,co)
  return tanh(wn + bc)

def inpu(wi,co,bi):
  wn = np.dot(wi,co)
  return sigmoid(wn + bi)

def output(wo,co,bo):
   wn = np.dot(wo,co)
   return sigmoid(wn + bo)

def cc(hp,xt):

   xt = np.array(xt)
   return  np.concatenate((hp,xt),axis=0)

def cs(wf,wi,wc,bf,bi,bc,co,cp):
   #cd for cell state current cell state
   fo = forgetgate(wf,co,bf)
   fo = np.array(fo)
   c = c_(wc,co,bc)
   i  = inpu(wi,co,bi)
   i = np.array(i)
   return fo*cp + i*c,i,fo,c

def h(wo,co,bo,ct):
  ot = output(wo,co,bo)
  ot = np.array(ot)
  return ot*tanh(ct),ot

def forward(wf,wi,wc,wo,bf,bi,bc,bo,co,cp):

  ct,it,fo,c = cs(wf,wi,wc,bf,bi,bc,co,cp)
  ht,ot = h(wo,co,bo,ct)
  return ct,ht,fo,it,c,ot


def dense103(ht,d1,d2,hd_l,w1,b1):
   #dense layer with 103 neurons
   #xavier random weights
   w1 = w1*np.sqrt(1/hd_l + d1)
   mu = np.dot(w1,ht)
   out = mu + b1
   return sigmoid(out)

def dense3(d1o,d1,d2,w2,b2):
   #output layer
   w2 = w2*np.sqrt(1/d1+d2)
   mu = np.dot(w2,d1o)
   out = mu + b2
   return sigmoid(out)

def loss():

  #mean square error because we have used min max scaler 0-1
  e = np.mean((y_true - y_predict)**2,axis = 0)
  e = np.mean(e)

def backpropogation(w1,w2,b1,b2,d1o,d2o,alpha,dn,b,ht,ct,cp,fo,it,c_,ot,co,wf,wi,wc,wo,bf,bi,bc,bo,w1m,w2m,wim,wom,wfm,wcm,w1v,w2v,wiv,wov,wfv,wcv,t):
     #epsilon for non zero division
     epsilon = 1e-8
     beta1 = 0.7
     beta2 = 0.777
     #input into sigmoids
     zf= (np.dot(wf,co) + bf)
     zi= (np.dot(wi,co) + bi)
     zc= (np.dot(wc,co) + bc)
     zo= (np.dot(wo,co) + bo)
     #backpopogiton or calculation updating the weights using gradient descent
     #dense layer updation
     #output delta error
     #d2o is output predicted
     #r term is 0.1
     xt_1 = dn[b]
     od = (d2o - xt_1)*sigmoid_d(d2o)
     he = np.dot(od,w2)
     hd = he*sigmoid_d(d1o)
     #calculating the delta of ht ot dE/dht
     hte = np.dot(hd,w1)
     htd = hte*sigmoid_d(ht)

     #in back propogation through lstms we use dervatives of states so this derivative of state will  be used in calculating the dervative of loss wth respective to weights and biasses
     #updating of wf,wi,wo,wc,bf,bo,bi,bc
     #dE/wo = dE/dot.dot/dow similarly to every weights
     hx =co.reshape(-1,1).T
     gi =(htd*ot*tanh_d(ct)*c_*sigmoid(zi)*(1-sigmoid(zi))).reshape(-1,1)*hx
     go =(htd*tanh(ct)*sigmoid(zo)*(1-sigmoid(zo))).reshape(-1,1)*hx
     gf =(htd*ot*tanh_d(ct)*cp*sigmoid(zf)*(1-sigmoid(zf))).reshape(-1,1)*hx
     gc =(htd*ot*tanh_d(ct)*it*tanh_d(zc)).reshape(-1,1)*hx

     bgi =htd*ot*tanh_d(ct)*c_*sigmoid(zi)*(1-sigmoid(zi))
     bgo =htd*tanh(ct)*sigmoid(zo)*(1-sigmoid(zo))
     bgf =htd*ot*tanh_d(ct)*cp*sigmoid(zf)*(1-sigmoid(zf))
     bgc =htd*ot*tanh_d(ct)*it*(tanh_d(zc))

     #updating
     #updating weights of dense layers
     od=od.reshape(-1,1)
     d1o = d1o.reshape(-1,1).T
     g2 = np.dot(od,d1o)

     hd = hd.reshape(-1,1)
     ht=ht.reshape(-1,1)
     g1 = np.dot(hd,ht.T)

     #based updation of dense layers
     b1 += np.sum(od)*alpha
     b2 += np.sum(hd)*alpha

     #calculatng average momentums for w1,w2,wi,wo,wf,wc,(gradients,and squared gradients)
     w1m = w1m*beta1 + (1-beta1)*g1
     w2m = w2m*beta1 + (1-beta1)*g2
     wim = wim*beta1 + (1-beta1)*gi
     wom = wom*beta1 + (1-beta1)*go
     wfm = wfm*beta1 + (1-beta1)*gf
     wcm = wcm*beta1 + (1-beta1)*gc

     w1v = w1v*beta1 + (1-beta1)*g1**2
     w2v = w2v*beta1 + (1-beta1)*g2**2
     wiv = wiv*beta1 + (1-beta1)*gi**2
     wov = wov*beta1 + (1-beta1)*go**2
     wfv = wfv*beta1 + (1-beta1)*gf**2
     wcv = wcv*beta1 + (1-beta1)*gc**2

     #calculating the biase corrected weights as every momenteum average should be bias corrected
     w1cm = w1m/(1-beta1**t)
     w2cm = w2m/(1-beta1**t)
     wicm = wim/(1-beta1**t)
     wocm = wom/(1-beta1**t)
     wfcm = wfm/(1-beta1**t)
     wccm = wcm/(1-beta1**t)

     w1cv = w1v/(1-beta2**t)
     w2cv = w2v/(1-beta2**t)
     wicv = wiv/(1-beta2**t)
     wocv = wov/(1-beta2**t)
     wfcv = wfv/(1-beta2**t)
     wccv = wcv/(1-beta2**t)

     #updating the parameters
     w1 = w1 - (w1cm/(np.sqrt(w1cv)+epsilon))*alpha
     w2 = w2 - (w2cm/(np.sqrt(w2cv)+epsilon))*alpha
     wi = wi - (wicm/(np.sqrt(wicv)+epsilon))*alpha
     wo = wo - (wocm/(np.sqrt(wocv)+epsilon))*alpha
     wf = wf - (wfcm/(np.sqrt(wfcv)+epsilon))*alpha
     wc = wc - (wccm/(np.sqrt(wccv)+epsilon))*alpha

     bi -= bgi*alpha
     bo -= bgo*alpha
     bf -= bgf*alpha
     bc -= bgc*alpha

     return  w1,w2,b1,b2,wi,wf,wc,wo,bi,bf,bo,bc


class lstm:
    def __init__(self,hd_l,xl,ts,dn,d1,d2,mode,alpha,p,da):
     # taking dimensiosn of hidden state be 64
     # consider ts is timestep
     #dnl = len(dn)
     print("length of dataset" + " "+ str(dnl))
     print("calculating random values for weights and biases ...")
     wf,wi,wc,wo,bf,bi,bc,bo,hp,cp,w1,b1,w2,b2,w1m,w2m,wim,wom,wfm,wcm,w1v,w2v,wiv,wov,wfm,wcv = weights(hd_l,xl)

     i = 0
     while  i < p:
       print("*********************************")
       j = i
       print("i value " + str(i))
       b =i+ts
       while j < b :
          print("j value " + str(j))
          co = cc(hp,dn[j])
          #print("concatinated " + str(co))
          ct,ht,fo,it,c,ot = forward(wf,wi,wc,wo,bf,bi,bc,bo,co,cp)
          cpp = cp
          hp = ht
          cp = ct
          j =j+1
       #print("hdden output of 5 timesteps" + " " + str(ht))
       print("inputting into dense layers done with lstm layers...")
       d1o = dense103(ht,d1,d2,hd_l,w1,b1)
       d2o = dense3(d1o,d1,d2,w2,b2)
       print("done with dense layers...")
       if mode == "back" or mode =="pb":
        w1,w2,b1,b2,wi,wf,wc,wo,bi,bf,bo,bc= backpropogation(w1,w2,b1,b2,d1o,d2o,alpha,dn,b,ht,ct,cpp,fo,it,c,ot,co,wf,wi,wc,wo,bf,bi,bc,bo,w1m,w2m,wim,wom,wfm,wcm,w1v,w2v,wiv,wov,wfm,wcv,i)
        print("back propogation done")
       i += 1
     if mode=="pb":
       hp=ht
       cp=ct
       i=0
       while  i < p:
         print("*********************************")
         j = i
         print("i value " + str(i))
         b =i+ts
         while j < b :
           print("j value " + str(j))
           co = cc(hp,dn[j])
           #print("concatinated " + str(co))
           ct,ht,fo,it,c,ot = forward(wf,wi,wc,wo,bf,bi,bc,bo,co,cp)
           cpp = cp
           hp = ht
           cp = ct
           j =j+1
         #print("hdden output of 5 timesteps" + " " + str(ht))
         print("inputting into dense layers done with lstm layers...")
         d1o = dense103(ht,d1,d2,hd_l,w1,b1)
         d2o = dense3(d1o,d1,d2,w2,b2)
         i += 1
     if i==p:
         hp =ht
         cp =ct
         predict=[]
         predict = np.array(predict)
         while  i < (p + da) :
            print("*********************************")
            j = i
            print("i value " + str(i))
            b =i+ts
            while j < b :
              print("j value " + str(j))
              co = cc(hp,d2o)
              #print("concatinated " + str(co))
              ct,ht,fo,it,c,ot = forward(wf,wi,wc,wo,bf,bi,bc,bo,co,cp)
              cpp = cp
              hp = ht
              cp = ct
              j =j+1
            #print("hdden output of 5 timesteps" + " " + str(ht))
            print("inputting into dense layers done with lstm layers...")
            d1o = dense103(ht,d1,d2,hd_l,w1,b1)
            d2o = dense3(d1o,d1,d2,w2,b2)
            dn = np.append(dn,d2o)
            predict = np.append(predict,d2o)
            i += 1
     print("printing the predicted values")
     time.sleep(5)
     print(predict)


print("steps\n data\n train ")
step = input("step give data as input for preprocess data :")
if step == "data":
  dn = data()
else :
   print("error should have data")
   exit()
hl = input("dimension or length of hidden state should be default/new(64/128) :")
if hl == "default":
    hL = 128
if hl == "new" :
   hL = int(input("give new dimesions for hidden state :"))
ts = int(input("time step to be considered recommends 5 :"))
xl = int(input("dimensions of input x or number of features in x(3) :"))
d1 = int(input("dense function1 no of neurons(103) :"))
d2 = int(input("dense function2 no of neurons(3) :"))
mode = input("give mode of program back for backropogation/forward and predict pb :")
if mode == "pb":
      da = int(input("give number of days to predict :"))
alpha = float(input("give the value for learning rate :"))
dnl = len(dn)
p=dnl - ts
lstm(hL,xl,ts,dn,d1,d2,mode,alpha,p,da)
#hL == hd_l

