import numpy as np
import pandas as pd
import torch
import torchvision as torchv
import matplotlib.pyplot as plt

#add any transofrms needed with using commas
transforms= torchv.transforms.ToTensor()

#data downloading and loading
traindata = torchv.datasets.CIFAR10(root="./captiondata",train=True,download = True,transform=transforms)
validationdata = torchv.datasets.CIFAR10(root="./captiondata",train = False,download=True,transform=transforms)

#number of different classes(targets/labels)
nptraintargets = np.array(traindata.targets)
numofclasses = np.unique(nptraintargets)
print(numofclasses)

#classes
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


#plot data change traindata.data[:] to plot number of images
plt.figure()
for index,img in enumerate(traindata.data[:8]):
    plt.subplot(4,4,index+1)
    plt.imshow(img)

#import dataloader and dataset
from torch.utils.data import DataLoader,Dataset

#load the data with dataloader (dataloader wrappes iterator around data to access the data)
trainload = DataLoader(traindata,batch_size=10,shuffle=True)
validationload = DataLoader(validationdata,batch_size=10,shuffle=True)

#model architecture
class classification(torch.nn.Module):
  def __init__(self):
    super(classification,self).__init__()
    self.conv1 = torch.nn.Conv2d(in_channels=3,out_channels=8,kernel_size=5,stride=1)
    self.conv2 = torch.nn.Conv2d(in_channels=8,out_channels=18,kernel_size=5,stride=1)
    self.pool = torch.nn.MaxPool2d(kernel_size=2,stride=2)
    self.flatten = torch.nn.Flatten()
    self.linear1 = torch.nn.Linear(450,128)
    self.linear2 = torch.nn.Linear(128,64)
    self.linear3 = torch.nn.Linear(64,10)
    self.batchnorm = torch.nn.BatchNorm2d(18)
    self.relu = torch.nn.ReLU()
    self.dropout = torch.nn.Dropout(0.2)


  def forward(self,img):

    conv = self.pool(self.relu(self.conv1(img)))
    conv1 =self.pool(self.relu((self.conv2(conv))))
    norm = self.flatten(self.batchnorm(conv1))
    l1 = self.linear1(norm)
    l2 = self.linear2(l1)
    l2 = self.dropout(l2)
    pred = self.linear3(l2)

    return pred

#install if not availabele if already installed skip
pip install torcheval


#model initialization(loss function crossentropy/optimizer sgd model.parameters() gives all the parameters to be trained)
model = classification()
lossfunction = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(),lr=0.001,momentum=0.9)

#for accuracy calculation using torcheval.metrics and multiclassaccuracy/metricvaalidation for validation accuracy/metrictrain for train accuracy calculation
from torcheval.metrics import MulticlassAccuracy
metricvalidation = MulticlassAccuracy()
metrictrain = MulticlassAccuracy()

#trianing
best =2000
numepochs=list()  #to store epochs for plotting
validationlosses =list()  #to store validation lossess for each epoch for plotting
trainlosses=list()  #to store training losses for each epoch for plotting

epochs = int(input("no of epochs for trainig")) #taking input from user for numbe of epochs
for epoch in range(epochs):
   print(str(epoch)+" epoch")
   metrictrain.reset() #resetting the metric for new epoch starts new
   trainerror=0
   validationerror = 0
   for index,batch in enumerate(trainload):
      data,label=batch
      model.train()  #to specify we are training
      optimizer.zero_grad()  #torch zero grad for resetting the gradients so they dont accumulate from last epoch
      prediction = model(data)
      loss = lossfunction(prediction,label)
      loss.backward()  #back ward pass with loss
      optimizer.step() #updating the gradients
      trainerror += loss.item()
      metrictrain.update(prediction,label) #updating the metric with values for calculating the accuracy

   model.eval() #specifying we are not training just evaluating with gradients from last epoch
   with torch.no_grad(): #no gradients updating
    for datapre,labelpre in validationload:
       preval =  model(datapre)
       error = lossfunction(preval,labelpre)
       validationerror +=error.item()
       metricvalidation.update(preval,labelpre)

   #patience for traing with (10) can changable
   if best > validationerror:
     best = validationerror
     patience = 10
   else :
     patience -=1
     if patience == 0:
       break


   validationlosses.append(validationerror/len(validationload)) #average validaton loss
   trainlosses.append(trainerror/len(trainload)) #average train loss
   numepochs.append(epoch)
   print(str(metrictrain.compute())+"train")
   print(str(metricvalidation.compute)+"validation")


print(validationlosses)
print(trainlosses)

#plot the validation and training losses
plt.figure(figsize=(7,7))
plt.plot(numepochs,trainlosses,"blue",label="training loss")
plt.plot(numepochs,validationlosses,"red",label="validation Loss")
plt.title("valdation training losses pytorch")
plt.show()



#calculating number of correct(prediction) and total number of labels for each class
correct =[0]*10
#or np.zeros(9).tolist()
total = [0]*10
model.eval()
with torch.no_grad():
  for batch in validationload:
      data,targets = batch
      outputs = model(data)
      _,predictions =torch.max(outputs,1) #return the index of maximum number in outputs in dim=1
      for target,prediction in zip(targets,predictions):
           if target == prediction:
               correct[prediction.item()] +=1
           total[prediction.item()] +=1


z = zip(correct,total)
for index,values in enumerate(z):
    a,b = values
    accuracy = (a/b)*100
    print(str(classes[index])+" "+str(accuracy))


#sample accuracys for each class after training the model with 20 epochs

#plane 66.66666666666666
#car 72.83609576427256
#bird 47.956403269754766
#cat 42.75023386342376
#deer 63.53887399463807
#dog 48.97959183673469
#frog 72.29166666666667
#horse 67.8396871945259
#ship 69.30409914204004
#truck 76.91428571428571
