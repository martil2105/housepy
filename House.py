import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

df_train = pd.read_csv('train.csv')
df_trainlog = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')
df_trainlog["SalePrice"] = np.log(df_train['SalePrice'])
print(df_train.columns)
print(df_train['SalePrice'].describe())
"""
#histogram
plt.figure(figsize=(10,6))
plt.hist(df_train['SalePrice'], bins=100, edgecolor='black')
plt.title('Histogram of saleprice')
plt.xlabel('SalePrice')
plt.ylabel('Frequency')

#log histogram
plt.figure(figsize=(10,6))
plt.hist(df_trainlog['SalePrice'], bins=100, edgecolor='black')
plt.title('Log Histogram of saleprice')
plt.xlabel('SalePrice')
plt.ylabel('Frequency')
plt.show()
#correlation matrix
corrmat = df_train.corr()
k = 10 #number of variables for heatmap
cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(df_train[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()

#scatterplot overallqual/saleprice
var = ['GrLivArea', 'GarageArea', "TotalBsmtSF"]
for var in var:
    data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
    data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000), edgecolor='black')
plt.show()

#Boxplot
var = ["OverallQual", "GarageCars"]
for var in var:
    data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
    f, ax = plt.subplots(figsize=(9,6))
    fig = sns.boxplot(x=var, y='SalePrice', data=data)
    fig.axis(ymin=0, ymax=800000)

"""
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
var_num = ['LotFrontage','GarageYrBlt', 'GarageArea', 'GarageCars', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath', 'MasVnrArea']
for i in var_num:
    whole_data[i] = whole_data[i].fillna(0)
var_catmod = ['MSZoning', 'Utilities', 'Functional', 'Exterior1st', 'Electrical', 'Exterior2nd', 'KitchenQual', 'SaleType']
for i in var_catmod:
    whole_data[i] = whole_data[i].fillna(whole_data[i].mode()[0])
whole_data["MSZonig"] = whole_data["MSZoning"].fillna(whole_data["MSZoning"].mode()[0])
missing_values_whole = whole_data.isnull().sum().sort_values(ascending=False)
missing_values_whole = missing_values_whole[missing_values_whole  > 0]
#print(missing_values_whole)

#transforming some numerical variables that are really categorical
whole_data_dummy = pd.get_dummies(whole_data)
print(whole_data_dummy.shape)
#modeling

X_train, X_test, y_train, y_test = train_test_split(whole_data_dummy[:df_train.shape[0]], df_train['SalePrice'], test_size=0.3, random_state=442)
#Linear Regression
model = LinearRegression()
fit = model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))   
print('R2 Score:', metrics.r2_score(y_test, y_pred))
print(X_test.shape, y_test.shape, y_pred.shape)

selected_feature_index = 1  # Choose index of feature you want to visualize
plt.scatter(X_test.iloc[:, selected_feature_index], y_test,  color='gray')
plt.plot(X_test.iloc[:, selected_feature_index], y_pred, color='red', linewidth=2)
plt.show()
