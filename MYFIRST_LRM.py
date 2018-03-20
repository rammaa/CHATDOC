# -*- coding: utf-8 -*-
"""
Created on Sun Mar 11 18:41:32 2018

@author: User
"""
#Template
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset=pd.read_csv('Salary_Data.csv')
X=dataset.iloc[: ,:-1].values
Y=dataset.iloc[: ,1].values


from sklearn.cross_validation import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)

#Fitting of SLR to training  set
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)

#predict the test set
y_pred=regressor.predict(X_test)

#plot the graph for train
plt.scatter(X_train,Y_train,color='red')
plt.plot(X_train,regressor.predict(X_train),color='black')
plt.title('Salary vs exp(training set)')
plt.xlabel('expe')
plt.ylabel('salary')
plt.show()


#plot the graph for test
plt.scatter(X_test,Y_test,color='red')
plt.plot(X_train,regressor.predict(X_train),color='black')
plt.title('Salary vs exp(training set)')
plt.xlabel('expe')
plt.ylabel('salary')
plt.show()

