from sklearn.linear_model  import  LinearRegression
import pandas as pd 
import numpy as np 

import matplotlib.pyplot as plt  

a=pd.read_csv('stock.csv')
a['Date'] = pd.to_datetime(a['Date'])


#x will have all variables 
x = a[['Open','High','Low','Volume',]]
#y have output 
y = a['Close']
print(x)
print(y)

#initialization of model 
model = LinearRegression()
# trining the model 
train = model.fit(x,y) 
print(train)
#model R^2 value for x and y relation 
r_2 = train.score(x,y)
print(r_2)
#intercept bo for f(x)
int = train.intercept_
print(int)
#all coffecient b1 ,b2...bn
co = train.coef_
print(co) 

#need to calculate output we can do with 
# f(x) == y = b0 + b1(x1) + b2(x2) + b3(x3)....bn(xn)
# y = train.intercept_ + (train.coef_*x)

#use predict instead 

test = model.predict(x)
print(test)


#plotting graph for linear regression 


plt.scatter(y, test, color='green', alpha=0.5, label='Actual Values')
plt.plot([min(y), max(y)], [min(test), max(test)], linestyle='--', color='black', label='Equality Line')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Linear Regression: Actual vs. Predicted Values')
plt.legend()
plt.show()
