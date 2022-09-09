# -*- coding: utf-8 -*-
"""
Created on Tue Sep  6 17:29:00 2022

@author: ramya
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

engg = pd.read_csv('population.csv')

data = engg.copy()


print('Dataset')
data.head()

print('Column descriptions')
data.info() 

#Checks for missing values    
print("Columnwise missing values")
print(data.isnull().sum())
""" There are no missing values """

print("Summary of numerical variables")
data.describe()

print("Summary of categorical variables")
data.describe(include='O')

data2 = pd.read_csv('population.csv',na_values=['?'])
# no missing values


# Dropping unwanted columns

col1=['Density_100']
data2 = data2.drop(columns=col1, axis=1)

# Realtionship between independent variables
correlation = data2.corr()

fig,ax = plt.subplots(3,3,figsize=(18,15))
fig.suptitle('DATA VISUALIZATIONS')

sns.distplot(data2.Population,ax=ax[0,0])
ax[0,0].set_title('Plot of Population')

sns.distplot(data2.Area,ax=ax[0,1])
ax[0,1].set_title('Plot of Area')

sns.regplot(ax=ax[0,2],x='Population',y='Area',data=data2,color='blue',fit_reg=False)
ax[0,2].set_title('Population vs Area')
"""More the area more the population"""


sns.countplot(ax=ax[1,0],x='Density',data=data2)
ax[1,0].set_title('Density')

sns.countplot(ax=ax[1,1],x='Density_1',data=data2)
ax[1,1].set_title('Density_1')

sns.countplot(ax=ax[1,2],x='Density_10',data=data2)
ax[1,2].set_title('Density_10')


sns.regplot(ax=ax[2,0],x='Country',y='Prediction_1',data=data2,color='green',fit_reg=False)
ax[2,0].set_title('Country vs Prediction of 1 year')

sns.regplot(ax=ax[2,1],x='Country',y='Prediction_10',data=data2,color='green',fit_reg=False)
ax[2,1].set_title('Country vs Prediction of 10 years')

sns.regplot(ax=ax[2,2],x='Country',y='Prediction_100',data=data2,color='green',fit_reg=False)
ax[2,2].set_title('Country vs Prediction of 100 years')

plt.show()

sns.pairplot(data2,hue='Area')
plt.show()



from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

data2 = data2.dropna(axis=0)
data2 =pd.get_dummies(data2,drop_first=True) 

# Separating input and output features
x1 = data2.drop(['Prediction_10'], axis='columns', inplace=False)
y1 = data2['Prediction_10']

# Plotting the variable Prediction_10
r = pd.DataFrame({"1. Before":y1, "2. After":np.log(y1)})
r.hist()
plt.show()

# Transforming Prediction_10 as a logarithmic value
y1 = np.log(y1)

# Splitting data into test and train
X_train, X_test, y_train, y_test = train_test_split(x1, y1, test_size=0.3, random_state = 3)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

lgr=LinearRegression(fit_intercept=True)

model_lin1=lgr.fit(X_train,y_train)

rent_predictions_lin1 = lgr.predict(X_test)

lin_mse1 = mean_squared_error(y_test, rent_predictions_lin1)
lin_rmse1 = np.sqrt(lin_mse1)
print(lin_rmse1)

r2_lin_test1=model_lin1.score(X_test,y_test)
r2_lin_train1=model_lin1.score(X_train,y_train)

residuals1=y_test-rent_predictions_lin1
plt.title('Residuals plot')
sns.regplot(x=rent_predictions_lin1, y=residuals1, scatter=True, 
            fit_reg=False,color='cyan')
plt.show()
print(residuals1.describe())

print("R squared value for train from Linear Regression=  %s"% r2_lin_train1)
print("R squared value for test from Linear Regression=  %s"% r2_lin_test1)
print("RMSE value for test from Linear Regression=  %s"% lin_rmse1)




