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
  ds_s = ds.loc[:,["High","Low"]]
  ds_s = ds_s.reset_index(drop = True )
  print("after reseting the to high low volume\n")
  print(ds_s.head())
  print(ds_s.tail())
  print(ds_s.shape)
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
  print(dn.head())
  print(dn.tail())
  dn = np.array(dn)
  return dn,min,max

def weights(hd_l,xl):
    #weights should in dimesniosns of length of hidden state X (hiddenstate + input)
    #xavier initialization
    scale = np.sqrt(2 / (hd_l + xl))
    wf = np.random.rand(hd_l,hd_l+xl)*scale
    wi = np.random.rand(hd_l,hd_l+xl)*scale
    wc = np.random.rand(hd_l,hd_l+xl)*scale
    wo = np.random.rand(hd_l,hd_l+xl)*scale

    wim = np.zeros((hd_l,hd_l+xl))
    wom = np.zeros((hd_l,hd_l+xl))
    wfm = np.zeros((hd_l,hd_l+xl))
    wcm = np.zeros((hd_l,hd_l+xl))

    wiv = np.zeros((hd_l,hd_l+xl))
    wov = np.zeros((hd_l,hd_l+xl))
    wfv = np.zeros((hd_l,hd_l+xl))
    wcv = np.zeros((hd_l,hd_l+xl))
    #biases ,w1m,w2m,wim,wom,wfm,wcm,w1v,w2v,wiv,wov,wfm,wcv
    # biases dimensions are equal to hidden state dimensions because after matrix multiplication resultant matrix will be n dimensions of hidden state (where we add biases to it so ...)
    bf = np.random.rand(hd_l)
    bi = np.random.rand(hd_l)
    bc = np.random.rand(hd_l)
    bo = np.random.rand(hd_l)
    hp = np.random.rand(hd_l)
    cp = np.random.rand(hd_l)

    scale1 = np.sqrt(2/(d1+hd_l))
    w1 = np.random.rand(d1,hd_l)*scale1
    scale2 = np.sqrt(2/(d2 + d1))
    w2 = np.random.rand(d2,d1)*scale2

    w1m=np.zeros((d1,hd_l))
    w2m=np.zeros((d2,d1))
    w1v=np.zeros((d1,hd_l))
    w2v=np.zeros((d2,d1))

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

   #print("wf" + str(wf))
   #print("co" + str(co))
   wn = np.dot(wf,co)
   #print("at forgate" + str(wn))
   si=sigmoid(wn + bf)
   #print("sigmoid" + str(si))
   #time.sleep(3)
   return si

def c_(wc,co,bc):
  #print("wc" + str(wc))
  #print("co" + str(co))
  wn = np.dot(wc,co)
  #print("wn" + str(wn))
  ta = tanh(wn + bc)
  #print("tanh" + str(ta))
  #time.sleep(3)
  return ta

def inpu(wi,co,bi):
  #print("wi" + str(wi))
  #print("co" + str(co))
  wn = np.dot(wi,co)
  #print("wn" + str(wn))
  si = sigmoid(wn + bi)
  #print("sigmoid" + str(si))
  #time.sleep(3)
  return si

def output(wo,co,bo):
   #print("wo" + str(wo))
   #print("co" + str(co))
   wn = np.dot(wo,co)
   #print("wn" + str(wn))
   si = sigmoid(wn + bo)
   #print("digmoid" + str(si))
   #time.sleep(3)
   return si

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
   sc = fo*cp + i*c
   #print("sc cell state" + str(sc))
   #time.sleep(3)
   return sc,i,fo,c

def h(wo,co,bo,ct):
  ot = output(wo,co,bo)
  ot = np.array(ot)
  sh = ot*tanh(ct)
  #print("sh hidden state" + str(sh))
  #time.sleep(3)
  return sh,ot

def forward(wf,wi,wc,wo,bf,bi,bc,bo,co,cp):

  ct,it,fo,c = cs(wf,wi,wc,bf,bi,bc,co,cp)
  ht,ot = h(wo,co,bo,ct)
  return ct,ht,fo,it,c,ot


def dense103(ht,d1,d2,hd_l,w1,b1):
   #dense layer with 103 neurons
   #xavier random weights
   #print("ht" + str(ht))
   mu = np.dot(w1,ht)
   #print("mu output of multiplcation" + str(mu))
   out = mu + b1
   si = sigmoid(out)
   #print("sigmod" + str(si))
   return si

def dense3(d1o,d1,d2,w2,b2):
   #output layer
   mu = np.dot(w2,d1o)
   out = mu + b2
   return sigmoid(out)

   #mean square error because we have used min max scaler 0-1

def backpropogation(w1,w2,b1,b2,d1o,d2o,alpha,dn,b,ht,ct,cp,fo,it,c_,ot,co,wf,wi,wc,wo,bf,bi,bc,bo,w1m,w2m,wim,wom,wfm,wcm,w1v,w2v,wiv,wov,wfv,wcv,t):
     #epsilon for non zero division
     epsilon = 0.01
     beta1 = 0.5
     beta2 = 0.555
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
     xt_1 = dn[b+1]
     loss = (d2o - xt_1)
     #time.sleep(2)
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

     #print(str(g1)+"g1")
     #print(str(g2)+"g2")
     #print(str(gi)+"gi")
     #print(str(go)+"go")
     #print(str(gf)+"gf")
     #print(str(gc)+"gc")


     #print(str(w1m)+"w1m")
     #print(str(w2m)+"w2m")
     #print(str(wim)+"wim")
     #print(str(wom)+"wom")
     #print(str(wfm)+"wfm")
     #print(str(wcm)+"wcm")

     #calculatng average momentums for w1,w2,wi,wo,wf,wc,(gradients,and squared gradients)
     w1m = w1m*beta1 + (1-beta1)*g1
     w2m = w2m*beta1 + (1-beta1)*g2
     wim = wim*beta1 + (1-beta1)*gi
     wom = wom*beta1 + (1-beta1)*go
     wfm = wfm*beta1 + (1-beta1)*gf
     wcm = wcm*beta1 + (1-beta1)*gc

     #print(str(w1m)+"w1m")
     #print(str(w2m)+"w2m")
     #print(str(wim)+"wim")
     #print(str(wom)+"wom")
     #print(str(wfm)+"wfm")
     #print(str(wcm)+"wcm")
     #time.sleep(3)

     w1v = w1v*beta1 + (1-beta1)*(g1**2)
     w2v = w2v*beta1 + (1-beta1)*(g2**2)
     wiv = wiv*beta1 + (1-beta1)*(gi**2)
     wov = wov*beta1 + (1-beta1)*(go**2)
     wfv = wfv*beta1 + (1-beta1)*(gf**2)
     wcv = wcv*beta1 + (1-beta1)*(gc**2)


     if t >= 1:
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
       w1 = w1 - (w1cm/(np.sqrt(w1cv+epsilon)))*alpha
       w2 = w2 - (w2cm/(np.sqrt(w2cv+epsilon)))*alpha
       wi = wi - (wicm/(np.sqrt(wicv+epsilon)))*alpha
       wo = wo - (wocm/(np.sqrt(wocv+epsilon)))*alpha
       wf = wf - (wfcm/(np.sqrt(wfcv+epsilon)))*alpha
       wc = wc - (wccm/(np.sqrt(wccv+epsilon)))*alpha

     if t == 0:
       #updating parameters without biase correction as i initilzed the momentums to 0 in initial
       w1 = w1 - (w1m/(np.sqrt(w1v+epsilon)))*alpha
       w2 = w2 - (w2m/(np.sqrt(w2v+epsilon)))*alpha
       wi = wi - (wim/(np.sqrt(wiv+epsilon)))*alpha
       wo = wo - (wom/(np.sqrt(wov+epsilon)))*alpha
       wf = wf - (wfm/(np.sqrt(wfv+epsilon)))*alpha
       wc = wc - (wcm/(np.sqrt(wcv+epsilon)))*alpha

     bi -= bgi*alpha
     bo -= bgo*alpha
     bf -= bgf*alpha
     bc -= bgc*alpha
     return  w1,w2,b1,b2,wi,wf,wc,wo,bi,bf,bo,bc,w1m,w2m,wim,wom,wfm,wcm,w1v,w2v,wiv,wov,wfm,wcv


class lstm:
    def __init__(self,hd_l,xl,ts,dn,d1,d2,mode,alpha,p,da,min,max):
     # taking dimensiosn of hidden state be 64
     # consider ts is timestep
     #dnl = len(dn)
     print("length of dataset" + " "+ str(dnl))
     print("calculating random values for weights and biases ...")
     wf,wi,wc,wo,bf,bi,bc,bo,hp,cp,w1,b1,w2,b2,w1m,w2m,wim,wom,wfm,wcm,w1v,w2v,wiv,wov,wfm,wcv = weights(hd_l,xl)

     i = 0
     while  i < p-1:
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
       #print("inputting into dense layers done with lstm layers...")
       d1o = dense103(ht,d1,d2,hd_l,w1,b1)
       d2o = dense3(d1o,d1,d2,w2,b2)
       print("done with dense layers...")
       wff,wii,wcc,woo,w11,w22,w1mm,wimm,w1vv=wf,wi,wc,wo,w1,w2,w1m,wim,w1v
       #if i > 4:
       #  print("*")
       #  print(wff)
       #  print(wii)
       #  print(woo)
       #  print(w11)
       #  print(w22)
       #  print(w1mm)

       if mode == "back" or mode =="pb":
         w1,w2,b1,b2,wi,wf,wc,wo,bi,bf,bo,bc,w1m,w2m,wim,wom,wfm,wcm,w1v,w2v,wiv,wov,wfm,wcv= backpropogation(w1,w2,b1,b2,d1o,d2o,alpha,dn,b,ht,ct,cpp,fo,it,c,ot,co,wf,wi,wc,wo,bf,bi,bc,bo,w1m,w2m,wim,wom,wfm,wcm,w1v,w2v,wiv,wov,wfm,wcv,i)
         print("back propogation done")
       i += 1

       #if i > 4:
       #  print(wf)
       #  print(wi)
       #  print(wo)
       #  print(w1)
       #  print(w2)
       #  print(w1m)
       #  print("*")
       #  time.sleep(3)
       #  if i>6:
       #    exit()

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
         #print("inputting into dense layers done with lstm layers...")
         #d1o = dense103(ht,d1,d2,hd_l,w1,b1)
         #d2o = dense3(d1o,d1,d2,w2,b2)
         i += 1
     #print("wi" + str(wi))
     #print("wo" + str(wo))
     #print("wf" + str(wf))
     #print("wc" + str(wc))
     #print("w1" + str(w1))
     #print("w2" + str(w2))
     #print("ht" + str(ht))
     #print("ct" + str(ct))
     if i==p:
         hp =ht
         cp =ct
         predict = np.empty((1,3))
         print("started predicting...")
         while  i < (p + da) :
            print("*********************************")
            j = i
            print("i value " + str(i))
            b =i+ts
            while j < b :
              print("j value " + str(j))
              #print("input or xt for predicting" + str(d2o))
              print(dn[j])
              co = cc(hp,dn[j])
              #print("concatinated " + str(co))
              ct,ht,fo,it,c,ot = forward(wf,wi,wc,wo,bf,bi,bc,bo,co,cp)
              hp = ht
              cp = ct
              j =j+1
            #print("hdden output of 5 timesteps" + " " + str(ht))
            print("inputting into dense layers done with lstm layers...")
            d1o = dense103(ht,d1,d2,hd_l,w1,b1)
            d2o = dense3(d1o,d1,d2,w2,b2)
            d2oo = d2o.reshape(-1,1).T
            dn = np.concatenate((dn, d2oo), axis=0)
            predict = np.concatenate((predict,d2oo))
            i += 1
     print("printing the predicted values")
     print(predict)
     mm = input("if you wan to undo the minmaax scaler y/n:")
     if mm == "y":
        d = max-min
        print(d)
        columns_n = ["High","Low","Volume"]
        predict = pd.DataFrame({"High":predict[:,0],"Low":predict[:,1],"Volume":predict[:,2]})
        predict = (predict*(d) + min )
        print("new predict" + str(predict))
     if mm == "n":
        exit()
     if mm != "n" or mm !="y":
        exit()

print("steps\n data\n train ")
step = input("step give data as input for preprocess data :")
if step == "data":
  dn,min,max = data()
else :
   print("error should have data")
   exit()
hl = input("dimension or length of hidden state should be default/new :")
if hl == "default":
    hL = 64
if hl == "new" :
   hL = int(input("give new dimesions for hidden state :"))
ts = int(input("time step to be considered recommends 5 :"))
xl = int(input("dimensions of input x or number of features in x(2) :"))
d1 = int(input("dense function1 no of neurons(103) :"))
d2 = int(input("dense function2 no of neurons(2) :"))
mode = input("give mode of program back for backropogation/forward and predict pb :")
if mode == "pb":
      da = int(input("give number of days to predict :"))
alpha = float(input("give the value for learning rate :"))
dnl = len(dn)
p=dnl - ts
#p is the last iteration with out predicting
lstm(hL,xl,ts,dn,d1,d2,mode,alpha,p,da,min,max)
#hL == hd_l
#10 line for data
#39 line  for weights
#141 line for dense
#149 line for dense3
#158 back propogation
#266 lstm class
