import torch
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
import numpy as np
import pandas as pd

n=int(input("degree of the polynomial"))
epochs=int(input("no of epochs"))

#for outside data uncommentbelow
#df = pd.read_csv("polylinear.csv")
#print(df.head())

#dy = df["Pressure"]
#dx = df["Temperature"]

#x=np.array(dx).reshape(-1,1)
#y=np.array(dy).reshape(-1,1)
x,y = make_regression(n_samples=500,n_features=1,noise=1,random_state=2)
y=y.reshape(-1,1)
print(str(x.shape) + str(y.shape))
x_tensor = torch.tensor(x,dtype=torch.float32)
y_tensor = torch.tensor(y,dtype=torch.float32)

print(x_tensor.shape)
print(y_tensor.shape)


class polylinearRegression(torch.nn.Module):

      def __init__(self):
         super(polylinearRegression,self).__init__()
         self.poly = torch.nn.ModuleList(torch.nn.Linear(1,1) for i in range(0,n))

      def forward(self,x):
           output=0
           z=x
           for i,layer in enumerate(self.poly):
                   output = layer(z)
                   z=output
           return output


model = polylinearRegression()

error = torch.nn.MSELoss()
optimizer =torch.optim.SGD(model.parameters(),lr=0.001,weight_decay=0.01)
optimizer.zero_grad()

for a in range (epochs):

        y_pred = model(x_tensor)
        loss = error(y_pred,y_tensor)
        loss.backward()
        optimizer.step()
        print('a{}, loss{}'.format(a, loss.item()))

y_m = model(x_tensor)
plt.scatter(x,y,color="green")
plt.plot(x,y_m.detach().numpy(),color="red")
plt.show()


