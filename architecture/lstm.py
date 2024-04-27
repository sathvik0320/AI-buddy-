import numpy as np
import random
import math
import time
#data preprocessing
import numpy as np
import pandas as pd
import  matplotlib.pyplot as plt

def data():
  ds  = pd.read_csv("YESBANK.NS.csv")
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
    #biases
    # biases dimensions are equal to hidden state dimensions because after matrix multiplication resultant matrix will be n dimensions of hidden state (where we add biases to it so ...)
    bf = np.random.rand(hd_l)
    bi = np.random.rand(hd_l)
    bc = np.random.rand(hd_l)
    bo = np.random.rand(hd_l)
    hp = np.random.rand(hd_l)
    cp = np.random.rand(hd_l)

    return wf,wi,wc,wo,bf,bi,bc,bo,hp,cp

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

  ct,i,fo,c = cs(wf,wi,wc,bf,bi,bc,co,cp)
  ht,ot = h(wo,co,bo,ct)
  return ct,ht,fo,i,c,ot


def dense103(ht,d1,d2,hd_l):
   #dense layer with 103 neurons
   #xavier random weights
   w1 = np.random.rand(d1,hd_l)*np.sqrt(1/hd_l + d1)
   b1 = np.random.rand(d1)
   mu = np.dot(w1,ht)
   out = mu + b1
   return sigmoid(out),w1,b1

def dense3(d1o,d1,d2):
   #output layer
   w2 = np.random.rand(d2,d1)*np.sqrt(1/d1+d2)
   b2 = np.random.rand(d2)
   mu = np.dot(w2,d1o)
   out = mu + b2
   return sigmoid(out),w2,b2

def loss():

  #mean square error because we have used min max scaler 0-1
  e = np.mean((y_true - y_predict)**2,axis = 0)
  e = np.mean(e)

def backpropogation(w1,w2,b1,b2,d1o,d2o,alpha,dn,b,ht,ct,cp,fo,it,c_,ot,co,wf,wi,wc,wo,bf,bi,bc,bo):

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
     lr =0.1
     xt_1 = dn[b + 1]
     od = (d2o - xt_1)*sigmoid_d(d2o)
     he = np.dot(od,w2)
     hd = he*sigmoid_d(d1o)
     #calculating the delta of ht ot dE/dht
     hte = np.dot(hd,w1)
     htd = hte*sigmoid_d(ht)

     #in back propogation through lstms we use dervatives of states so this derivative of state will  be used in calculating the dervative of loss wth respective to weights and biasses
     #updating of wf,wi,wo,wc,bf,bo,bi,bc
     #dE/wo = dE/dot.dot/dow similarly to every weights
     gi =htd*ot*tanh_d(ct)*c_*sigmoid(zi)*(1-sigmoid(zi))*co
     go =htd*tanh(ct)*sigmoid(zo)*(1-sigmoid(zo))*co
     gf =htd*ot*tanh_d(ct)*cp*sigmoid(zf)*(1-sigmoid(zf))*co
     gc =htd*ot*tanh_d(ct)*it*(tanh_d(zc))*co


     bgi =htd*ot*tanh_d(ct)*c_*sigmoid(zi)*(1-sigmoid(zi))
     bgo =htd*tanh(ct)*sigmoid(zo)*(1-sigmoid(zo))
     bgf =htd*ot*tanh_d(ct)*cp*sigmoid(zf)*(1-sigmoid(zf))
     bgc =htd*ot*tanh_d(ct)*it*(tanh_d(zc))

     #updating
     #updating weights of dense layers
     re = lr*w2
     g2 = np.dot(d10.T,od)
     g2 = g2 + re
     w2 = w2 + g2*alpha

     re = lr*w1
     g1 = np.dot(ht.T,hd)
     g1 = g1 + re
     w1 = w1 + g1*alpha

     #based updation of dense layers
     b1 += np.sum(od)*alpha
     b2 += np.sum(hd)*alpha

     re = lr*wi
     gi = gi + re
     wi -= gi*alpha

     re = lr*wo
     go = go + re
     wo -= go*alpha

     re = lr*wf
     gf = gf + re
     wf -= gf*alpha

     re = lr*wc
     gc = gc + re
     wc -= gc*alpha

     bi -= bgi*alpha
     bi -= bgo*alpha
     bi -= bgf*alpha
     bi -= bgc*alpha


     return


class lstm:
    def __init__(self,hd_l,xl,ts,dn,d1,d2,mode,alpha):
     # taking dimensiosn of hidden state be 64
     # consider ts is timestep
     dnl = len(dn)
     print("length of dataset" + " "+ str(dnl))
     print("calculating random values for weights and biases ...")
     time.sleep(5)
     wf,wi,wc,wo,bf,bi,bc,bo,hp,cp = weights(hd_l,xl)

     i = 0
     while i <= dnl - ts:
       print("*********************************")
       j = i
       print("i value " + str(i))
       time.sleep(5)
       b =i+ts
       while j < b :
          print("j value " + str(j))
          co = cc(hp,dn[j])
          print("concatinated " + str(co))
          ct,ht,fo,i,c,ot = forward(wf,wi,wc,wo,bf,bi,bc,bo,co,cp)
          cpp = cp
          hp = ht
          cp = ct
          j =j+1

       print("hdden output of 5 timesteps" + " " + str(ht))
       print("inputting into dense layers done with lstm layers...")
       time.sleep(5)
       d1o,w1,b1 = dense103(ht,d1,d2,hd_l)
       d2o,w2,b2 = dense3(d1o,d1,d2)
       print("done with dense layers...")
       time.sleep(2)
       if mode == "back":
         backpropogation(w1,w2,b1,b2,d1o,d2o,alpha,dn,b,ht,ct,cpp,fo,i,c,ot,co,wf,wi,wc,wo,bf,bi,bc,bo)
       i = i + 1


print("steps\n data\n train ")
step = input("step give data as input for preprocess data :")
if step == "data":
  dn = data()
else :
   print("error should have data forward or backward")
   exit()
hl = input("dimension or length of hidden state should be default/new(64/128) :")
if hl == "default":
    hL = 64
if hl == "new" :
   hL = int(input("give new dimesions for hidden state :"))
ts = int(input("time step to be considered recommends 5 :"))
xl = int(input("dimensions of input x or number of features in x(3) :"))
d1 = int(input("dense function1 no of neurons(103) :"))
d2 = int(input("dense function2 no of neurons(3) :"))
mode = input("give mode of program back for backropogation/forward forward propogation :")
alpha = float(input("give the value for learning rate :"))
lstm(hL,xl,ts,dn,d1,d2,mode,alpha)
#hL == hd_l
