# -
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset=pd.read_csv('50_startups.csv')
X=dataset.iloc[: ,:-1].values
Y=dataset.iloc[: ,4].values

#encode categerical variable
#encode independent variable
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder=LabelEncoder()
X[:,3]=labelencoder.fit_transform(X[:,3])

onehotencoder=OneHotEncoder(categorical_features=[3])
X=onehotencoder.fit_transform(X).toarray()

# avoiding the dummy variable
X=X[:,1:]

#splitiing tin trainer and test set
from sklearn.cross_validation import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=.2,random_state=0)

#Fitting of SLR to training  set
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)

#prediction
Y_pred=regressor.predict(X_test)


#BW elemination
import statsmodels.formula.api as sm
def backwardElimination(x,sl):
    numVars= len(x[0])
    for i in range (0,numVars):
        regressor_OLS=sm.OLS(Y,x).fit()
        maxVar=max(regressor_OLS.pvalues).astype(float)
        if maxVar>sl:
            for j in range(0,numVars-i):
                if (regressor_OLS.pvalues[j].astype(float)==maxVar):
                    x=np.delete(x,j,1)
    regressor_OLS.summary()
    return x 
SL=0.05
X_opt=X[:,[0,1,2,3,4]]
X_modeled=backwardElimination(X_opt,SL)          