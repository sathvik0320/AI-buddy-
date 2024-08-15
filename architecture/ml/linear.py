###########use 441 epochs############for better line
#torch will allow us to access most of the pytorch libraries
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression

epochs = int(input("give input for number of epochs"))
#data
#(x,y)
x, y = make_regression(n_samples=100, n_features=1, noise=1, random_state=1)
y=y.reshape(-1,1)
print(str(x.shape)+str(y.shape))

x_tensor = torch.tensor(x, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)
print(x_tensor.shape)
#definig the class for lineargresson

class LinearRegression(torch.nn.Module):

    #def __init__ is a constructor for class
    def __init__(self):
          super(LinearRegression,self).__init__()
          #torch.nn.Linear(input_dimensions,output_dimensions(1))
          self.linear = torch.nn.Linear(1,1)
          #initialized linear 1 input 1 output layer

    def forward(self,x):

         y_pred = self.linear(x)
         return y_pred


model= LinearRegression()
print(model.parameters())
loss = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(),lr=0.001,weight_decay=0.01)
optimizer.zero_grad()

for a in range (epochs):

    y_p = model(x_tensor)
    error=loss(y_p,y_tensor)
    #back propogation and finds gradients
    #optimizer.zero_grad()
    error.backward()
    #updating the parameters
    optimizer.step()
    print('a{}, error {}'.format(a, error.item()))



y_pl = model(x_tensor)
#plotting data
plt.scatter(x,y,color="red")
#plotting prediction
#tensor is converted to numpy in plotting by default  so we use tensor.detach().numpy() to avoid pytoech error
#m,b=np.polyfit(x,y_p.detach().numpy(),1)
plt.plot(x,y_pl.detach().numpy())
plt.show()






