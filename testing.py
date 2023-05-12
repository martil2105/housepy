import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from scipy.stats import norm    

df_train = pd.read_csv('train.csv')
df_trainlog = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')
df_train["SalePrice"] = np.log(df_train['SalePrice'])

#missing data
whole_data = pd.concat((df_train, df_test)).reset_index(drop=True)
whole_data.drop(['SalePrice'], axis=1, inplace=True)

missing_values_train =  df_train.isnull().sum().sort_values(ascending=False)
missing_values_train = missing_values_train[missing_values_train  > 0]
missing_values_whole = whole_data.isnull().sum().sort_values(ascending=False)
missing_values_whole = missing_values_whole[missing_values_whole  > 0]

#dealing with missing data
var_cat = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType2', 'BsmtFinType1', 'MasVnrType']
for i in var_cat:
    whole_data[i] = whole_data[i].fillna('None')
var_num = ['LotFrontage','GarageYrBlt' ,'GarageArea', 'GarageCars', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath', 'MasVnrArea']
for i in var_num:
    whole_data[i] = whole_data[i].fillna(0)
var_catmod = ['MSZoning', 'Utilities', 'Functional', 'Exterior1st', 'Electrical', 'Exterior2nd', 'KitchenQual', 'SaleType']
for i in var_catmod:
    whole_data[i] = whole_data[i].fillna(whole_data[i].mode()[0])
whole_data["MSZonig"] = whole_data["MSZoning"].fillna(whole_data["MSZoning"].mode()[0])
missing_values_whole = whole_data.isnull().sum().sort_values(ascending=False)
missing_values_whole = missing_values_whole[missing_values_whole  > 0]

whole_data = whole_data.drop(['GarageYrBlt', 'PoolQC', 'Utilities'], axis=1)
#transforming some numerical variables that are really categorical
whole_data_dummy = pd.get_dummies(whole_data)


#skewed features
numeric_cols = whole_data_dummy.select_dtypes(include=[np.number]).columns

threshold = 0.5
for col in numeric_cols:
    skewness = whole_data_dummy[col].skew()
    if skewness > threshold:
        whole_data_dummy[col] = np.log1p(whole_data_dummy[col])  # using log1p instead of log to handle zero values

#modeling

X_train, X_test, y_train, y_test = train_test_split(whole_data_dummy[:df_train.shape[0]], df_train['SalePrice'], test_size=0.3, random_state=22)
#Linear Regression
model = LinearRegression()
fit = model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))   
print('R2 Score:', metrics.r2_score(y_test, y_pred))

#Cross Validation
from sklearn.model_selection import cross_val_score

scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
rmse_scores = np.sqrt(-scores)
print('Mean RMSE:', rmse_scores.mean())
print('Standard deviation of RMSE:', rmse_scores.std())
print('RMSE:', rmse_scores)

#Elasctic Net
from sklearn.linear_model import ElasticNetCV, ElasticNet
alphas = np.logspace(-4, 4.5, 10)
l1ratio = [.1, .5, 1]

model = ElasticNetCV(alphas=alphas, l1_ratio=l1ratio, cv=6, max_iter=10000)
model.fit(X_train, y_train)

print('Optimal alpha: %.8f'%model.alpha_)
print('Optimal l1_ratio: %.3f'%model.l1_ratio_) 

model = ElasticNet(alpha=0.001, l1_ratio=0.5)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print('R2 Score:', metrics.r2_score(y_test, y_pred))

