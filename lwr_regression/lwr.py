import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import  LinearRegression 


a = pd.read_csv("tips.csv")
x = a.["tip"]
y = a.["total_bill"] 

x_train,x_test,y_train,y_test =train_test_split(x,y,random_state=0)

model = LinearRegression()

def weights(x_train,point,tau):

  M = x_train.shape[0]
  w = np.mat(np.eye(M))
   


model.fit(x_train,y_train,weights)
plt.scatter(x_test,predict())
