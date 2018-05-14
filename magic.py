# -*- coding: utf-8 -*-
"""
Created on Fri May 11 13:11:19 2018

@author: dhananjays
"""
#importing required files
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
 #Setting seed for reproducability
#reading the file

df=pd.read_excel('JFM.xlsx')
df.head(10)
#sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='viridis')
df.corr()


from sklearn.model_selection import train_test_split
X = df.drop('consumption',axis=1)
y = df['consumption']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01,random_state=4 )

from sklearn.linear_model import Lasso
lam = Lasso()
lam.fit(X_train,y_train)
predictions = lam.predict(X_test)

from sklearn import metrics
from sklearn.metrics import r2_score
ddj = r2_score(y_test, predictions)
print(ddj)
print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))


dj=pd.read_excel('pred.xlsx')
dj
predictions = lm.predict(dj)
predictions = pd.DataFrame(predictions)
predictions
writer = pd.ExcelWriter('predtemp.xlsx', engine='xlsxwriter')
predictions.to_excel(writer, sheet_name='Sheet1')
writer.save()

#with month-
#total kwh      total money(kw*8)
#122236.4239	977891.391
#121825.0496	974600.3967
#119754.1245	958032.9958
#127235.3354	1017882.684

#with lasso
#129251.2449	1034009.959



#without month-
#170925.6465	1367405.172
#170030.5426	1360244.341


