import numpy as np
import random

#x = input
#ws = weights
#h1 = hidden layer output 1
#h2 = hidden layer output 2
#y = output of total network
#y1 = outputs of hidden layer two neurons
#y2 output variable having outputs from hid2

def weightsb():
      #one input two weight
      #weights
      ws = [0,0,0,0,0,0,0,0,0,0,0]
      # no of neurons in total architecture
      bi = [0,0,0,0,0,0]
      print("default weights ")
      print(ws)
      print("default biases")
      print(bi)
      return ws,bi


def hid1(ws,x,bi):

     #hidden layer 1 outputs after getting into activaton function
     # two neurons
     # x.w1 ,x.w2
     # bias 1

     y1=[0,0]
     y1[0] = sigmoid(x*ws[0] + bi[0])
     y1[1] = sigmoid(x*ws[1] + bi[1])
     return y1


def hid2(ws,y1,bi):
     #hidden layer 2 output 3 neurons output afte getting into activation function
     # 6 weights used w[2] to w[7]
     # y1.weights

     # and adding bias b2
     #y21 multiplication of weights and y1[0]
     y21 = [0,0,0]
     y21[0] = y1[0]*ws[2]
     y21[1] = y1[0]*ws[3]
     y21[2] = y1[0]*ws[4]

     #y22 multiplication of weights and y1[1]
     y22 = [0,0,0]
     y22[0] = y1[1]*ws[5]
     y22[1] = y1[1]*ws[6]
     y22[2] = y1[1]*ws[7]

     #y2 outuput from all neurons in hidden layer neuron 1,2,3 after getting into activation function
     y2=[0,0,0]
     y2[0] = sigmoid(y21[0] + y22[0] + bi[2])
     y2[1] = sigmoid(y22[1] + y21[1] + bi[3])
     y2[2] = sigmoid(y21[2] + y22[2] + bi[4])
     return y2

def out(ws,h2,bi):

     #output from output layer
     # 1 neuron

     y3 = sigmoid(h2[0]*ws[8] + h2[1]*ws[9] + h2[2]*ws[10] + bi[5])
     return y3


def sigmoid(x):
     return 1 / (1 + np.exp(-x))

def sigmoid_derivative( x):
     return x * (1 - x)

def back(x,h1,h2,y,ws,r,bi,target):
    #back propogation for all values
    #e0 = ((y-target)*(y*(1-y))*ws[8]*(h2[0]*(1-h2[0]))*ws[2]*(h1[0]*(1-h1[0]))*x) + ((y-target)*(y*(1-y))*ws[9]*(h2[1]*(1-h2[1]))*ws[3]*(h1[0]*(1-h1[0]))*x) +  ((y-target)*(y*(1-y))*ws[10]*(h2[2]*(1-h2[2]))*ws[4]*(h1[0]*(1-h1[0]))*x)
    #calculated with informtion from geeks for geeks 
    loss = (target - y)
    delta5 = (loss)*(sigmoid_derivative(y))
    delta4 = h2[2]*(1-h2[2])*(ws[10]*delta5)
    delta3 = h2[1]*(1-h2[1])*(ws[9]*delta5)
    delta2 = h2[0]*(1-h2[0])*(ws[8]*delta5)
    delta0 = h1[0]*(1-h1[0])*(ws[2]*delta3 + ws[3]*delta4 + ws[4]*delta5)
    delta1 = h1[1]*(1-h1[1])*(ws[5]*delta3 + ws[6]*delta4 + ws[7]*delta5)
    print("delta5" + " " + str(delta5))
    print("delta4" + " " + str(delta4))
    print("delta3" + " " + str(delta3))
    print("delta2" + " " + str(delta2))
    print("delta1" + " " + str(delta1))
    print("delta0" + " " + str(delta0))

    e10 = r*h2[2]*delta5
    print("e10" + " " + str(e10))
    e9 = r*h2[1]*delta5
    print("e9" + " " + str(e9))
    e8 = r*h2[0]*delta5
    print("e8"+ " " + str(e8))
    e7 = r*h1[1]*delta4
    print("e7" + " " + str(e7))
    e6 = r*h1[1]*delta3
    print("e6" + " " +str(e6))
    e5 = r*h1[1]*delta2
    print("e5"+ " " + str(e5))
    e4 = r*h1[0]*delta4
    print("e4"+ " " + str(e4))
    e3 = r*h1[0]*delta3
    print("e3"+ " " + str(e3))
    e2 = r*h1[0]*delta2
    print("e2"+ " " + str(e2))
    e1 = r*x*delta1
    print("e1"+ " " + str(e1))
    e0 = r*x*delta0
    print("e0"+ " " + str(e0))
    #biases updating 
    b0 = r*delta0
    print("b0"+ " " + str(b0))
    b1 = r*delta1
    print("b1"+ " " + str(b1))
    b2 = r*delta2
    print("b2"+ " " + str(b2))
    b3 = r*delta3
    print("b3"+ " " + str(b3))
    b4 = r*delta4
    print("b4"+ " " + str(b4))
    b5 = r*delta5
    print("b5"+ " " + str(b5))

    ws[0] = ws[0] + e0
    ws[1] = ws[1] + e1
    ws[2] = ws[2] + e2
    ws[3] = ws[3] + e3
    ws[4] = ws[4] + e4
    ws[5] = ws[5] + e5
    ws[6] = ws[6] + e6
    ws[7] = ws[7] + e7
    ws[8] = ws[8] + e8
    ws[9] = ws[9] + e9
    ws[10] = ws[10] + e10

    bi[0] = bi[0] + b0
    bi[1] = bi[1] + b1
    bi[2] = bi[2] + b2
    bi[3] = bi[3] + b3
    bi[4] = bi[4] + b4
    bi[5] = bi[5] + b5


    return ws,bi

def forward(ws,x,bi,target,epochs,r):
      target = target
      h1 = hid1(ws,x,bi)
      print("h1"+ " " + str(h1))
      h2 = hid2(ws,h1,bi)
      print("h2"+ " " + str(h2))
      y  = out(ws,h2,bi)
      print("output ********" + str(y))
      print("loss "+ " " + str(target - y))
      ws,bi = back(x,h1,h2,y,ws,r,bi,target)
      print("ws and bi" + " "+ str(ws) + str(bi))

      return ws,bi

def forwardonly(ws,x,bi):

      h1 = hid1(ws,x,bi)
      h2 = hid2(ws,h1,bi)
      y  = out(ws,h2,bi)
      return y

def train(ws,x,bi,target,epochs,r):

   i=0
   while i <= epochs:
      print("*********************************************")
      ws,bi = forward(ws,x,bi,target,epochs,r)
      print("loop"+ " " + str(i))
      print("**********************************************")
      i=i+1
   return ws,bi

class network:
   def __init__(self,x,target,epochs,status,r):
       ws,bi = weightsb()
       if status == 'ran':
           f=forwardonly(ws,x,bi)
           print(f)
       if status == 'train':
           ws,bi= train(ws,x,bi,target,epochs,r)
           print("weights and biases ")
           print(ws)
           print(bi)
       if status == 'forward':
            ws = [0.61109861, 0.91113477, 0.49052329, 0.71050572, 0.27245442, 0.31537416, 0.85522599, 0.09773472, 0.44896363, 0.76933274, 0.15683539]
            bi = [0.31602596, 0.85556363, 0.96621334, 0.48494335, 0.92890805, 0.83522096,]
            print("default weights")
            print(ws)
            c = input("default weghts say yes/no if not :")
            if c == 'yes':
              f=forwardonly(ws,x,bi)
              print(f)

print("input to network x")
x = int(input())
target = int(input("Enter target: "))
print("status: train for training forward for predicting ran with random weights")
status = input("")
if status == 'train':
  print("no of epochs for training")
  epochs = int(input())
  alpha=float(input("learning rate : "))
if status == 'ran':
   epochs = 0
   alpha = 0
if status == 'forward':
  epochs = 0
  alpha = 0
net = network(x,target,epochs,status,alpha)
