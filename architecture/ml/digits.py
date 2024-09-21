import torch
import numpy as np

import torchvision as torchv
transforms = torchv.transforms.ToTensor()

train=torchv.datasets.MNIST(root=".",train=True , download =True ,transform=transforms)
val = torchv.datasets.MNIST(root=".",train=False,download=True, transform =transforms)

traini = torch.utils.data.DataLoader(train,batch_size=1,shuffle=True)
valdi =  torch.utils.data.DataLoader(val,batch_size=1,shuffle=True)

print(train.data.shape)
print(val.data.shape)

import seaborn as sn
import matplotlib.pyplot as plt

plt.figure(figsize=(2,2))
plt.imshow(train.data[0],cmap="gray")

import torch.nn.functional as f

#model architecture

class digits(torch.nn.Module):
  def __init__(self,filters,kernel,stride,mkernel,mstride):
    super(digits,self).__init__()

    self.conv1 = torch.nn.Conv2d(in_channels=1,out_channels=filters,kernel_size=kernel,stride=stride)
    self.conv2 = torch.nn.Conv2d(in_channels=filters,out_channels=filters*2,kernel_size=kernel,stride=stride)
    self.maxpool = torch.nn.MaxPool2d(kernel_size=mkernel,stride=mstride)
    self.linear1 = torch.nn.Linear(320,64)
    self.linear2 = torch.nn.Linear(64,10)
    self.relu = torch.nn.ReLU()
    self.dropout = torch.nn.Dropout(0.2)
    self.flatten = torch.nn.Flatten()
    self.softmax = torch.nn.LogSoftmax(dim=1)

  def forward(self,x):

       conv1out = self.conv1(x)
       relu1 = self.relu(conv1out)
       maxout1 = self.maxpool(relu1)
       #print(maxout1.shape)

       conv2out = self.conv2(maxout1)
       relu2 = self.relu(conv2out)
       maxout2 = self.maxpool(relu2)
       #print(maxout2.shape)

       flat = self.flatten(maxout2)
       #print(flat.shape)
       l1 = self.linear1(flat)
       relu3 = self.relu(l1)
       relu3 = self.dropout(relu3)
       l2 = self.linear2(relu3)

       #print(relu4.shape)
       finalvalues = self.softmax(l2)
       #print(finalvalues.shape)

       return finalvalues

#input kernels/stride
filters,kernel,stride,mkernel,mstride = map(int,input("filters(out_channels)/kernel/stride/mkernel/mstride").split())
print(kernel)


# pip install torcheval #
#initalize the model
model = digits(filters,kernel,stride,mkernel,mstride)
lossfunction = torch.nn.NLLLoss()
optimizer = torch.optim.SGD(model.parameters(),lr=0.01,momentum=0.5)

from torcheval.metrics import MulticlassAccuracy
metrict = MulticlassAccuracy()
metricv = MulticlassAccuracy()



#train the model
epochs = int(input("give number of epochs"))
best =20000
tra=list()
vall=list()
ep=list()

for i in range(epochs) :
  print(str(i)+"  epoch")
  for index,(image,label) in enumerate(traini):

    model.train()
    metrict.reset()
    metricv.reset()

    #ba = ba.clone().detach().type(torch.float32)
    trainingloss =0
    optimizer.zero_grad()
    prediction = model(image)
    error = lossfunction(prediction,label)
    error.backward()
    optimizer.step()
    metrict.update(prediction,label)
    trainingloss +=error.item()




  model.eval()
  validationloss=0
  with torch.no_grad():
    for indexv,(imagev,labelv) in enumerate(valdi):
         #bat = bat.clone().detach().type(torch.float32)
         pre = model(imagev)
         err =  lossfunction(pre,labelv)
         validationloss += err.item()
         metricv.update(pre ,labelv)

  if best > validationloss:
     best = validationloss
     patience = 10
  else :
     patience -= 1
     if patience ==0:
        break

  print(str(validationloss/len(valdi))+" validationloss")
  print(str(trainingloss/len(traini))+"  trainingloss")

  print(metrict.compute())
  print(metricv.compute())
  print(end="")

  tra.append(trainingloss)
  vall.append(validationloss)
  ep.append(i)



#appending for plotting
tra = torch.tensor(tra).clone().detach().type(torch.float32)
vall = torch.tensor(vall).clone().detach().type(torch.float32)

#plotting the validation and training loss
import time
plt.figure(figsize=(7,7))
plt.plot(ep,tra,"blue",label="training loss")
plt.plot(ep,vall,"red",label="validation Loss")
plt.title("valdation training losses pytorch")
plt.show()


#testing the model
#testing purpose
testing = val.data[0].reshape(1,1,28,28).clone().detach().type(torch.float32)
result=model(testing)
print(result)

#print the label
target = val.targets[0]
print(target)


#save the model
torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            ...
            }, PATH)
