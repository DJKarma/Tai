# -*- coding: utf-8 -*-
"""
Created on Fri May 11 13:11:19 2018

@author: dhananjays
"""
#importing required files
import numpy as np
import pandas as pd
import seaborn as sns
#reading the file
df=pd.read_excel('cmarf.xlsx')
df.head
from sklearn.model_selection import train_test_split
X = df.drop('MachineIMM',axis=1)
y = df['MachineIMM']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.11)

from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
logmodel = XGBClassifier()
#logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)
predictions = logmodel.predict(X_test)



"""from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=150)
rfc.fit(X_train, y_train)
predictions = rfc.predict(X_test)"""


from sklearn.metrics import classification_report,confusion_matrix
from sklearn import metrics
cf=confusion_matrix(y_test,predictions)
print(classification_report(y_test,predictions))
print(cf)
print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))




dj=pd.read_excel('test.xlsx')
predictions = rfc.predict(dj)
predictions = pd.DataFrame(predictions)
writer = pd.ExcelWriter('ntest.xlsx', engine='xlsxwriter')
predictions.to_excel(writer, sheet_name='Sheet1')
writer.save()




"""
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)
lm.fit(X_train,y_train)
predictions = lm.predict(X_test)



from sklearn import metrics
print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))
    
from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test,predictions))
cf=confusion_matrix(y_test,predictions)
print(cf)
sns.heatmap(cf, annot=True,annot_kws={"size": 26},xticklabels=['Pred No','Pred Yes'], yticklabels=['Actual No','Actual Yes',])
print 'Wrong percentage-',float(4*100/26),'%'
"""