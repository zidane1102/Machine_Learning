#!/usr/bin/env python
# coding: utf-8

# In[30]:


#Simple Linear Regression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv(r'C:\Users\Admin\data\data_linear.csv')
a = pd.DataFrame(data)
X = a['Diện tích']
Y = a['Giá']
numerator = np.sum((X-np.mean(X)) * (Y - np.mean(Y)))
denominator = np.sum((X - np.mean(X)) ** 2)
b1 = numerator / denominator
b0 = np.mean(Y) - b1 * np.mean(X)

def predictions(x):
    return b0 + b1*x


plt.plot(X,Y,'bo')
m, b = np.polyfit(X, Y, 1)
plt.plot(X, m*X + b, 'r')
plt.ylabel('Giá')
plt.xlabel('Diện tích')
plt.legend(['Giá'])
plt.title('Dự đoán giá')
plt.grid()
plt.show()

pred_1 = predictions(50)
pred_2 = predictions(100)
pred_3 = predictions(150)

print(pred_1)
print(pred_2)
print(pred_3)


# In[40]:


#Multiple Linear Regression
import numpy as np
import pandas as pd
from pandas import read_csv
column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
data = pd.read_csv(r'C:\Users\Admin\data\housing.csv', header=None, delimiter=r"\s+", names=column_names)
a = pd.DataFrame(data)
X = data.drop('MEDV', axis = 1).values
Y = data['MEDV'].values

class OLS(object):
    
    def __init__(self):
        self.coefficients = []
    
    def reshape(self, X):
        return X.reshape(-1, 1)
    
    def concatenate(self, X):
        ones = np.ones(shape = X.shape[0]).reshape(-1,1)
        return np.concatenate((ones, X), 1)
    
    def fit(self,X,Y):
        if len(X.shape) == 1: 
            X = self.reshape(X)
        
        X = self.concatenate(X)
        self.coefficients = np.linalg.inv(X.transpose().dot(X)).dot(X.transpose()).dot(Y)
    
    def predict(self, t):
        b0 = self.coefficients[:0]
        bn = self.coefficients[1:]
        prediction = b0
        
        for xi, bi in zip(t, bn):
            prediction += (bi*xi)
        return prediction
model = OLS()
model.fit(X,Y)
model.coefficients


# In[ ]:




