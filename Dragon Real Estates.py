#!/usr/bin/env python
# coding: utf-8

# # Dragon Real Estate - Price Predictor

# In[1]:


import pandas as pd


# In[2]:


housing = pd.read_csv("data.csv")


# In[3]:


housing.head()


# In[4]:


housing.info()


# In[5]:


housing['CHAS'].value_counts()


# In[6]:


housing.describe()


# In[7]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[8]:


# For plotting Histogram
# import matplotlib.pyplot as plt
# housing.hist(bins=50, figsize=(20,15))


# # Train-Test Splitting

# In[9]:


# for learning purpose

import numpy as np

def split_train_test(data,test_ratio):
    np.random.seed(42)
    shuffled=np.random.permutation(len(data))
    print(shuffled)
    test_set_size=int(len(data)*test_ratio)
    test_indices=shuffled[:test_set_size]
    train_indices=shuffled[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]


# In[10]:


# train_set, test_set = split_train_test(housing, 0.2)


# In[11]:


# print(f"Rows in train set: {len(train_set)}\n Rows in test set:{len(test_set)}\n")


# In[12]:


from sklearn.model_selection import train_test_split
train_set, test_set =train_test_split(housing, test_size=0.2, random_state=42)
print(f"Rows in train set: {len(train_set)}\n Rows in test set:{len(test_set)}\n")


# In[13]:


from sklearn.model_selection import StratifiedShuffleSplit
split=StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing['CHAS']):
    strat_train_set=housing.loc[train_index]
    strat_test_set=housing.loc[test_index]


# In[14]:


strat_test_set.info()


# In[15]:


strat_test_set['CHAS'].value_counts()


# In[16]:


strat_train_set['CHAS'].value_counts()


# In[17]:


# 95/7


# In[18]:


# 376/28


# In[19]:


housing = strat_train_set.copy()


# # Looking for Correlations

# In[20]:


from pandas.plotting import scatter_matrix
attributes = ["MEDV", "RM", "ZN", "LSTAT"]
scatter_matrix(housing[attributes], figsize = (12,8))


# In[21]:


housing.plot(kind="scatter", x="RM", y="MEDV", alpha=1)


# # Attribute combinations

# In[22]:


housing.head()


# In[23]:


corr_matrix = housing.corr()
corr_matrix['MEDV'].sort_values(ascending=False)


# In[24]:


housing = strat_train_set.drop("MEDV", axis=1)
housing_labels = strat_train_set["MEDV"].copy()


# # Missing Attributes

# In[25]:


a = housing.dropna(subset=["RM"]) # option 1
a.shape
# Note that the original housing dataframe will remain unchanged


# In[26]:


housing.drop("RM", axis=1).shape # option 2
# Note that there is no RM column and also note that the original housing dataframe will remain unchanged


# In[27]:


median = housing["RM"].median() # Compute median for option 3
median


# In[28]:


housing["RM"].fillna(median)
# Note that the original housing dataframe will remain unchanged


# In[29]:


housing.shape


# In[30]:


from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy = "median")
imputer.fit(housing)


# In[31]:


imputer.statistics_


# In[32]:


X= imputer.transform(housing)


# In[33]:


housing_tr=pd.DataFrame(X, columns=housing.columns)


# In[34]:


housing_tr.describe()


# # Scikit-learn Design

# # Creating a Pipeline

# In[35]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
my_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
    
    ('std_scaler', StandardScaler()),
])


# In[36]:


housing_num_tr = my_pipeline.fit_transform(housing)


# In[37]:


housing_num_tr.shape


# # Selecting a desired model for Dragon Real Estate

# In[38]:


from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
# model = LinearRegression()
# model = DecisionTreeRegressor()
model = RandomForestRegressor()
model.fit(housing_num_tr, housing_labels)


# In[39]:


some_data = housing.iloc[:5]


# In[40]:


some_labels = housing_labels.iloc[:5]


# In[41]:


prepared_data = my_pipeline.transform(some_data)


# In[42]:


model.predict(prepared_data)


# In[43]:


list(some_labels)


# # Evaluating the model

# In[44]:


from sklearn.metrics import mean_squared_error
housing_predictions = model.predict(housing_num_tr)
mse = mean_squared_error(housing_labels, housing_predictions)
rmse = np.sqrt(mse)

rmse


# # Using better evaluation technique - Cross Validation

# In[45]:


from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, housing_num_tr, housing_labels, scoring="neg_mean_squared_error", cv=10)
rmse_scores = np.sqrt(-scores)


# In[46]:


rmse_scores


# In[47]:


def print_scores(scores):
    print("Scores:", scores)
    print("Mean:",scores.mean())
    print("Standard deviation:", scores.std())


# In[48]:


print_scores(rmse_scores)


# # Saving the model

# In[49]:


from joblib import dump, load
dump(model, 'Dragon.joblib')


# # Testing the model on test data

# In[50]:


X_test = strat_test_set.drop("MEDV", axis=1)
Y_test = strat_test_set["MEDV"].copy()
X_test_prepared = my_pipeline.transform(X_test)
final_predictions = model.predict(X_test_prepared)
final_mse = mean_squared_error(Y_test, final_predictions)
final_rmse = np.sqrt(final_mse)



print(final_predictions, list(Y_test))


# In[51]:


final_rmse


# # Using the model

# In[53]:


prepared_data[0]


# In[54]:


from joblib import dump, load
import numpy as np
model = load('Dragon.joblib')
features = np.array([[-0.43942006,  3.12628155, -1.12165014, -0.27288841, -1.42262747,
       -0.24141041, -1.31238772,  2.61111401, -1.0016859 , -0.5778192 ,
       -0.97491834,  0.41164221, -0.86091034]])
model.predict(features)


# In[ ]:




