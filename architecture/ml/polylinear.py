import torch
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
import numpy as np
import pandas as pd
import seaborn as sn
#from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

n=int(input("degree of the polynomial"))
epochs=int(input("no of epochs"))
fe=int(input("number of features "))
r=float(input("give learning rate"))
#for outside data uncommentbelow
#df = pd.read_csv("polylinear.csv")
#print(df.head())
#df.drop("sno",axis=1,inplace=True)

#plt.figure(figsize=(12,12))
#sn.heatmap(df.corr(),linewidths=0.1,vmax=1.0,square=True,linecolor="white",annot=True)


#dy = df["Pressure"]
#dx = df["Temperature"]

#x=np.array(dx).reshape(-1,1)
#y=np.array(dy).reshape(-1,1)

#normalization
#x_scaler =MinMaxScaler()
#y_scaler =MinMaxScaler()


#x=x_scaler.fit_transform(x)
#y=y_scaler.fit_transform(y)

x,y = make_regression(n_samples=100,n_features=1,noise=1,random_state=10)
x_c=x
x_c=x_c.flatten()
y_c=y
y=y.reshape(-1,1)

#correlation heatmap
#for data
#x_scaler =MinMaxScaler()
#y_scaler =MinMaxScaler()


#x=x_scaler.fit_transform(x)
#y=y_scaler.fit_transform(y)


#correlation coefficient using numpy()
plt.figure(figsize=(8,8))
df=pd.DataFrame({"1":x_c,"2":y_c})
sn.heatmap(df.corr(),linecolor="white",annot=True,vmax=1.0,linewidths=0.1,square=True)



print(str(x.shape) + str(y.shape))
x_tensor = torch.tensor(x,dtype=torch.float32)
y_tensor = torch.tensor(y,dtype=torch.float32)


print(x_tensor.shape)
print(y_tensor.shape)


class polylinearRegression(torch.nn.Module):

      def __init__(self):
         super(polylinearRegression,self).__init__()
         #self.poly = torch.nn.ModuleList(torch.nn.Linear(1,1) for i in range(1,n+1))
         self.Linear = torch.nn.Linear(n*fe,1)


      def forward(self,x):
           output=0

           poly_features = torch.cat([x**i for i in range(1,n+1)],dim=1)
           #for i,layer in enumerate(self.poly):
           #print(poly_features)

           output = self.Linear(poly_features)

           return output


model = polylinearRegression()

error = torch.nn.MSELoss()
optimizer =torch.optim.SGD(model.parameters(),lr=r,weight_decay=0.01)
#xavier initializtion for custom weights
#torch.nn.init.xavier_uniform_(self.Linear.weight)
optimizer.zero_grad()

for a in range (epochs):

        y_pred = model(x_tensor)
        loss = error(y_pred,y_tensor)
        loss.backward()
        #gradients clipping before
        #torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm=5.0)
        optimizer.step()
        print('a{}, loss{}'.format(a, loss.item()))

y_m = model(x_tensor)
#y_m=y_scaler.inverse_transform(y_m.detach().numpy())
plt.figure(figsize=(5,5))
plt.scatter(x,y,color="green")
plt.plot(x,y_m.detach().numpy(),color="red")
plt.show()


