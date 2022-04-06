#!/usr/bin/env python
# coding: utf-8

# In[678]:


# importing all the important
import numpy as np
import pandas as pd
pd.options.display.max_rows = 100
pd.options.display.max_columns = 100
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn import metrics
import os
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

# hide warnings
import warnings
warnings.filterwarnings('ignore')


# In[679]:


#Read the data
train = pd.read_csv("train.csv")
train.head()


# In[680]:


#check number of rows and columns
train.shape


# In[681]:


#check for data type of columns
train.info()


# In[682]:


pd.options.display.max_rows = None
train.isnull().sum()


# In[683]:


#remove columns with null values> 1400 as they anyway reduce variance
train =train.drop(train.columns[train.apply(lambda col: col.isnull().sum() > 1100)], axis=1)


# In[684]:




# now we will divide our columns into categorical and numerical category
types = train.dtypes
numeric_type = types[(types == 'int64') | (types == float)] 
categorical_type = types[types == object]

categorical_columns = list(categorical_type.index)
print("Categorical")
print(categorical_columns)
print("         ")
print("Numerical")
numerical_columns = list(numeric_type.index)
print(numerical_columns)


# In[685]:



# We will now replace null values with "none" for categorical variables
NA_facility_missing = ["MasVnrType", "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2", "FireplaceQu", "GarageType", "GarageFinish", "GarageQual"]
for i in NA_facility_missing:
    train[i].fillna("none", inplace=True)


# In[686]:


#We will now replace the missing values with respective medians
train['LotFrontage'].fillna(train["LotFrontage"].median(), inplace=True)
train["GarageYrBlt"].fillna(train["GarageYrBlt"].median(), inplace=True)
train["MasVnrArea"].fillna(train["MasVnrArea"].median(), inplace=True)


# In[687]:


# The missing value in electrical is only we can remove it
train["Electrical"].dropna(inplace=True)


# In[688]:


train.shape


# In[689]:


# plotting correlations on a heatmap
plt.figure(figsize=(40, 30))
sns.heatmap(train.corr(), annot = True, cmap="YlGnBu")
plt.show()


# In[690]:


#We will now remove the highly corelated variables and also ID since it holds no imkportance in analysis
train = train.drop(['GarageCars','TotRmsAbvGrd','GarageYrBlt','Id'], axis = 1)
train.head()


# In[691]:


train['PropAge'] = (train['YrSold'] - train['YearBuilt'])
train.head()


# In[692]:


sns.jointplot(x = train['PropAge'], y = train['SalePrice'])
plt.show()


# In[693]:


#we can now drop the related columns since we found a good corelation.
train = train.drop(['MoSold','YrSold','YearBuilt','YearRemodAdd'], axis = 1)
train.head()


# In[694]:


types = train.dtypes
numeric_type = types[(types == 'int64') | (types == float)] 
categorical_type = types[types == object]

categorical_columns = list(categorical_type.index)
print("Categorical")
print(categorical_columns)
print("         ")
print("Numerical")
numerical_columns = list(numeric_type.index)
print(numerical_columns)


# In[695]:


for i in categorical_columns:
    print(train[i].value_counts())


# In[696]:


#We will now remove columns with low variance
train = train.drop(['Street','Utilities','Condition2','RoofMatl','Heating'], axis = 1)


# In[697]:


types = train.dtypes
numeric_type = types[(types == 'int64') | (types == float)] 
categorical_type = types[types == object]

categorical_columns = list(categorical_type.index)
print("Categorical")
print(categorical_columns)


# In[698]:


# creating dummy columns for all categorical columns
train = pd.get_dummies(train, drop_first=True )
train.head()


# In[699]:


#Defining x variavles
X = train.drop(['SalePrice'], axis=1)
X.head()


# In[700]:


#Defining y variable - target
y = train['SalePrice']
y.head()


# In[701]:


#Splitiing the data into  traina nd test set
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, test_size=0.3, random_state=50)


# In[702]:


X_train.head()


# In[703]:


from sklearn.preprocessing import StandardScaler


# In[704]:


# standardization
scaler = StandardScaler()
#Providing numeric variables
num_vars=['MSSubClass', 'LotFrontage', 'LotArea', 'OverallQual', 'OverallCond', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'Fireplaces', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal', 'PropAge']
#Fit on data
X_train[num_vars] = scaler.fit_transform(X_train[num_vars])
X_test[num_vars] = scaler.fit_transform(X_test[num_vars])
X_train.head()

X_train.head()


# In[705]:


X_test.head()


# In[706]:


# Running RFE 
# Since number of variavles is very high we will run RFE to chec predictive power
lm = LinearRegression()
lm.fit(X_train, y_train)
 # running RFE for top 100 variables
rfe = RFE(lm, 100)            
rfe = rfe.fit(X_train, y_train)
list(zip(X_train.columns,rfe.support_,rfe.ranking_))


# In[707]:


#selecting top 100 variables for analysis
top= X_train.columns[rfe.support_]
top


# In[708]:


# Creating X_test with selected variables
X_train_rfe = X_train[top]
X_train_rfe.head()


# In[709]:


y_train_pred = lm.predict(X_train)
metrics.r2_score(y_true=y_train, y_pred=y_train_pred)


# In[710]:



y_test_pred = lm.predict(X_test)
metrics.r2_score(y_true=y_test, y_pred=y_test_pred)


# In[711]:


list(zip(X_test.columns,rfe.support_,rfe.ranking_))


# In[712]:


top1 = X_test.columns[rfe.support_]
X_test_rfe = X_test[top1]
X_test_rfe.head()


# In[713]:


# The above result is clear example of overfitting
# Now we will use lasso and Ridge regression


# In[714]:



# Checking the dimension of X_train & y_train
print("X_train", X_train.shape)
print("y_train", y_train.shape)


# In[726]:


from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import PolynomialFeatures


# In[727]:


#Apllu L1 regularisation ie Lasso regression

lasso_reg = linear_model.Lasso(alpha =50, max_iter=100, tol = 0.1)
lasso_reg.fit(X_train, y_train)


# In[728]:


lasso_reg.score( X_test, y_test)


# In[729]:


lasso_reg.score( X_train, y_train)


# In[523]:


ridge_reg = Ridge(alpha =50, max_iter=100, tol = 0.1)
ridge_reg.fit(X_train, y_train)


# In[524]:


ridge.score( X_test, y_test)


# In[515]:


ridge_reg.score( X_train, y_train)


# In[731]:


para = pd.DataFrame(mod)
para.columns = ['Variable', 'Coeff']
para.head()


# In[ ]:





# In[ ]:


The regularisation made the r square values better.
Coefficients have also been derived


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[452]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




