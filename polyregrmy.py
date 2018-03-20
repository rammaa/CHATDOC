# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 18:39:15 2018

@author: User
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset=pd.read_csv('Position_Salaries.csv')
X=dataset.iloc[: ,1:2].values
Y=dataset.iloc[: ,2].values

#fitting the lienar regression
from sklearn.linear_model import LinearRegression
lin_reg=LinearRegression()
lin_reg.fit(X,Y)
#fitting the polynomial
from sklearn.preprocessing  import PolynomialFeatures
polyreg=PolynomialFeatures(degree=2)
X_poly=polyreg.fit_transform(X)
linreg2=LinearRegression()
linreg2.fit(X_poly,Y)

#plotting

plt.scatter(X,Y,color='red')
plt.plot(X,lin_reg.predict(X),color='black')
plt.show() 



plt.scatter(X,Y,color='red')
plt.plot(X,linreg2.predict(polyreg.fit_transform(X)),color='black')
plt.show() 