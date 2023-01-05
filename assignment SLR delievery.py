# -*- coding: utf-8 -*-
"""
Created on Thu Dec 29 19:49:50 2022

@author: Vaishnav
"""

#importing the data

import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np

df = pd.read_csv(r"C:\anaconda\delivery_time.csv")
df

#===================================================================================================================================================================================================================================================================================
#EDA

df.isnull().any()

df.info()

df.describe()

df.shape

#===================================================================================================================================================================================================================================================================================
#Visualization
import matplotlib.pyplot as plt

df["Delivery Time"].hist()

df["Sorting Time"].hist()

df.boxplot(column="Delivery Time",vert=False)

df.boxplot(column="Sorting Time",vert=False)

plt.scatter (df["Sorting Time"],df["Delivery Time"],color = 'blue')
plt.xlabel("Sorting Time")
plt.ylabel("Delivery Time")
plt.show()

#=======================================================================================================================================================================================
#spliting the data into x and y

x = df[["Sorting Time"]]
y = df[["Delivery Time"]]

#=======================================================================================================================================================================================
#Without any trasnformations
#model1

LR = LinearRegression()
LR.fit(x,y)

y_pred = LR.predict(x)

# Then Finding Error
from sklearn.metrics import mean_squared_error,r2_score
mse = mean_squared_error(y, y_pred)
RMSE = np.sqrt(mse)
print("Root mean square error: ", RMSE.round(3))
print("Rsquare: ", r2_score(y, y_pred).round(3)*100)


# Scatter Plot with Plot
plt.scatter (x.iloc[:,0],y,color = 'red')
plt.plot (x.iloc[:,0],y_pred,color = 'blue')
plt.xlabel("Sorting Time")
plt.ylabel("Delivery Time")
plt.show()

#RMSE is 2.792 and R2 is 68.2
#=======================================================================================================================================================================================
#transformations-sqrt
#model 2

LR = LinearRegression()
LR.fit(np.sqrt(x),y)

y_pred_sqrt = LR.predict(np.sqrt(x))

# Then Finding Error
from sklearn.metrics import mean_squared_error,r2_score
mse = mean_squared_error(y, y_pred_sqrt)
RMSE = np.sqrt(mse)
print("Root mean square error: ", RMSE.round(3))
print("Rsquare: ", r2_score(y, y_pred_sqrt).round(3)*100)


# Scatter Plot with Plot
plt.scatter (x.iloc[:,0],y,color = 'red')
plt.plot (x.iloc[:,0],y_pred_sqrt,color = 'blue')
plt.xlabel("Sorting Time")
plt.ylabel("Delivery Time")
plt.show()

#RMSE is  2.732 and R2 is 69.6

#=======================================================================================================================================================================================
#transformation - log
#model 3

LR = LinearRegression()
LR.fit(np.log(x),y)

y_pred_log = LR.predict(np.log(x))

# Then Finding Error
from sklearn.metrics import mean_squared_error,r2_score
mse = mean_squared_error(y, y_pred_log)
RMSE = np.sqrt(mse)
print("Root mean square error: ", RMSE.round(3))
print("Rsquare: ", r2_score(y, y_pred_log).round(3)*100)


# Scatter Plot with Plot
plt.scatter (x.iloc[:,0],y,color = 'black')
plt.plot (x.iloc[:,0],y_pred_log,color = 'purple')
plt.xlabel("Sorting Time")
plt.ylabel("Delivery Time")
plt.show()

#RMSE is 2.733 and R2 is 69.5

#=======================================================================================================================================================================================
#transformation - x^2
#model 4

LR = LinearRegression()
LR.fit(x**2,y)

y_pred_2 = LR.predict(x**2)

# Then Finding Error
from sklearn.metrics import mean_squared_error,r2_score
mse = mean_squared_error(y, y_pred_2)
RMSE = np.sqrt(mse)
print("Root mean square error: ", RMSE.round(3))
print("Rsquare: ", r2_score(y, y_pred_2).round(3)*100)


# Scatter Plot with Plot
plt.scatter (x.iloc[:,0],y,color = 'black')
plt.plot (x.iloc[:,0],y_pred_2,color = 'purple')
plt.xlabel("Sorting Time")
plt.ylabel("Delivery Time")
plt.show()

#RMSE is 3.011 and R2 is 63.0

#=======================================================================================================================================================================================
#transformation - x^3
#model 5

LR = LinearRegression()
LR.fit(x**3,y)

y_pred_3 = LR.predict(x**3)

# Then Finding Error
from sklearn.metrics import mean_squared_error,r2_score
mse = mean_squared_error(y, y_pred_3)
RMSE = np.sqrt(mse)
print("Root mean square error: ", RMSE.round(3))
print("Rsquare: ", r2_score(y, y_pred_3).round(3)*100)


# Scatter Plot with Plot
plt.scatter (x.iloc[:,0],y,color = 'black')
plt.plot (x.iloc[:,0],y_pred_3,color = 'purple')
plt.xlabel("Sorting Time")
plt.ylabel("Delivery Time")
plt.show()

#RMSE is 3.253 and R2 is 56.8

#Inference :  A prediction model is built and the best model selected is model 2
#since its r2score is 69.6%


