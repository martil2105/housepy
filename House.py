import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from scipy.stats import norm   
import time 

df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')
df_train["SalePrice"] = np.log(df_train['SalePrice'])
print(df_train.columns)
print(df_train['SalePrice'].describe())

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
print(missing_values_whole)
whole_data = whole_data.drop(['GarageYrBlt', 'PoolQC', 'Utilities'], axis=1)
#transforming some numerical variables that are really categorical
whole_data_dummy = pd.get_dummies(whole_data)
print(whole_data_dummy.shape)

#skewed features
numeric_cols = whole_data_dummy.select_dtypes(include=[np.number]).columns

threshold = 0.5
for col in numeric_cols:
    skewness = whole_data_dummy[col].skew()
    if skewness > threshold:
        whole_data_dummy[col] = np.log1p(whole_data_dummy[col])  # using log1p instead of log to handle zero values
X_train, X_test, y_train, y_test = train_test_split(whole_data_dummy[:df_train.shape[0]], df_train['SalePrice'], test_size=0.3, random_state=22)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

#backward and forward selection
start_time = time.time()
from mlxtend.feature_selection import SequentialFeatureSelector as SFS

forward_feature_selector = SFS(LinearRegression(), k_features="best", forward=True, scoring='r2', cv=2)
forward_feature_selector = forward_feature_selector.fit(X_train, y_train)
fwd_selected_features = X_train.columns[list(forward_feature_selector.k_feature_idx_)]
print(fwd_selected_features)

X_train_fwd, X_test_fwd, y_train, y_test = train_test_split(X_train[fwd_selected_features], y_train, test_size = 0.2, random_state = 0)
regressor = LinearRegression()
regressor.fit(X_train_fwd, y_train)
y_pred_fwd = regressor.predict(X_test_fwd)

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred_fwd))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred_fwd))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred_fwd)))
print('R2 Score:', metrics.r2_score(y_test, y_pred_fwd))
print("coefficients:", regressor.coef_) 
end_time = time.time()
print("Time taken:", end_time - start_time)
#modeling


#Linear Regression
lm_model = LinearRegression()
fit = lm_model.fit(X_train, y_train)
y_pred = lm_model.predict(X_test)
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))   
print('R2 Score:', metrics.r2_score(y_test, y_pred))

#Cross Validation
from sklearn.model_selection import cross_val_score

scores = cross_val_score(lm_model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
rmse_scores = np.sqrt(-scores)
#print('Mean RMSE:', rmse_scores.mean())
#print('Standard deviation of RMSE:', rmse_scores.std())
#print('RMSE:', rmse_scores)

#Ridge Regression
from sklearn.linear_model import Ridge, RidgeCV

alphas = np.logspace(-4, 5.5, 20)
ridge_model = RidgeCV(alphas=alphas, cv= 5)
ridge_model.fit(X_train, y_train)
print('Optimal Alpha:', ridge_model.alpha_)

ridge_model = Ridge(alpha=3.2)
ridge_model.fit(X_train, y_train)
y_pred = ridge_model.predict(X_test)
ypre = ridge_model.predict(X_train)
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print('R2 Score:', metrics.r2_score(y_test, y_pred))

#Lasso Regression
from sklearn.linear_model import Lasso, LassoCV
alphas = np.logspace(-4, 5.5, 60)
lasso_model = LassoCV(alphas = alphas, cv=5)
lasso_model.fit(X_train, y_train)
print('Best alpha:', lasso_model.alpha_)

lasso_model = Lasso(alpha = 0.0004)
lasso_model.fit(X_train, y_train)
y_pred = lasso_model.predict(X_test)
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print('R2 Score:', metrics.r2_score(y_test, y_pred))

#Elasctic Net
from sklearn.linear_model import ElasticNetCV, ElasticNet
alphas = np.logspace(-4, 4.5, 10)
l1ratio = [.1, .5, 1]

en_model = ElasticNetCV(alphas=alphas, l1_ratio=l1ratio, cv=6, max_iter=10000)
en_model.fit(X_train, y_train)

print('Optimal alpha: %.8f'%en_model.alpha_)
print('Optimal l1_ratio: %.3f'%en_model.l1_ratio_) 

en_model = ElasticNet(alpha=0.001, l1_ratio=0.5)
en_model.fit(X_train, y_train)
y_pred = en_model.predict(X_test)
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print('R2 Score:', metrics.r2_score(y_test, y_pred))

#Cross Validation
lm_scores = cross_val_score(lm_model, X_train, y_train, cv=7,scoring="r2")
ridge_scores = cross_val_score(ridge_model, X_train, y_train, cv=7, scoring="r2")
lasso_scores = cross_val_score(lasso_model, X_train, y_train, cv=7, scoring="r2")
en_model_scores = cross_val_score(en_model, X_train, y_train, cv=7, scoring="r2")

print('Linear Regression CV scores:', lm_scores.mean())
print('Ridge Regression CV scores:', ridge_scores.mean())
print('Lasso Regression CV scores:', lasso_scores.mean())
print('Elastic Net CV scores:', en_model_scores.mean())