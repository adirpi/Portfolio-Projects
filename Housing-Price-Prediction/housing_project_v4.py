import os
import tarfile
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from pandas.plotting import scatter_matrix
housing=pd.read_csv('https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.csv')
housing['income_cat']=pd.cut(housing['median_income'],
bins=[0.,1.5,3.0,4.5,6.,np.inf],labels=[1,2,3,4,5])


#use sklearn to do Stratified Sampling
from sklearn.model_selection import StratifiedShuffleSplit
split= StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42)
for train_index,test_index in split.split(housing,housing['income_cat']):
    strat_train_set=housing.loc[train_index]
    strat_test_set=housing.loc[test_index]

housing= strat_train_set.drop('median_house_value',axis=1)
housing_labels = strat_train_set['median_house_value'].copy()

#DataCleaning
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='median')

#remove all the attributes that have object category as median is number
housing_num=housing.drop('ocean_proximity',axis=1)
housing_cat=housing[['ocean_proximity']]
imputer.fit(housing_num)

housing_tr=pd.DataFrame(imputer.transform(housing_num),columns=housing_num.columns)

#Handling Text and Categorical Attributes
from sklearn.preprocessing import OrdinalEncoder
ordinal_encoder= OrdinalEncoder()
housing_cat_encoded= ordinal_encoder.fit_transform(housing_cat)
#print(housing_cat_encoded[0:10],ordinal_encoder.categories_)

from sklearn.preprocessing import OneHotEncoder
#Custom Transformer
from sklearn.base import BaseEstimator,TransformerMixin
rooms_ix,bedrooms_ix,population_ix,households_ix=3,4,5,6

class CombinedAttributesAdder(BaseEstimator,TransformerMixin):
    def __init__(self,add_bedrooms_per_room=True):
        self.add_bedrooms_per_room=add_bedrooms_per_room
    def fit(self,X,y=None):
        return self
    def transform(self,X,y=None):
        rooms_per_household = X[:,rooms_ix]/X[:,households_ix]
        population_per_household= X[:,population_ix]/X[:,households_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room=X[:,bedrooms_ix]/X[:,rooms_ix]
            return np.c_[X,rooms_per_household,population_per_household,bedrooms_per_room]
        else:
            return np.c_[X,rooms_per_household,population_per_household]

attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
housing_extra_attribs =attr_adder.transform(housing.values)

#Transformation Pipelines

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

num_pipeline = Pipeline([('imputer',SimpleImputer(strategy="median")),('attribs_adder',
CombinedAttributesAdder()),('std_scalar',StandardScaler())])

housing_num_tr = num_pipeline.fit_transform(housing_num)

from sklearn.compose import ColumnTransformer
num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]

full_pipeline = ColumnTransformer([("num",num_pipeline,num_attribs),
("cat",OneHotEncoder(),cat_attribs)])

print(full_pipeline)

housing_prepared =full_pipeline.fit_transform(housing)

from sklearn.linear_model import LinearRegression

lin_reg= LinearRegression()
lin_reg.fit(housing_prepared,housing_labels)


some_data = housing.iloc[:5]
some_labels = housing_labels.iloc[:5]
some_data_prepared = full_pipeline.transform(some_data)

# print ("Predictions", lin_reg.predict(some_data_prepared))
# print  ("Labels",list(some_labels))

from sklearn.metrics import mean_squared_error
housing_predictions = lin_reg.predict(housing_prepared)
lin_mse = mean_squared_error(housing_labels,housing_predictions)
lin_rmse = np.sqrt(lin_mse)
lin_rmse


from sklearn.tree import DecisionTreeRegressor
tree_reg = DecisionTreeRegressor()
tree_reg.fit(housing_prepared,housing_labels)
housing_predictions = tree_reg.predict(housing_prepared)
tree_mse = mean_squared_error(housing_labels,housing_predictions)
tree_rmse=np.sqrt(tree_mse)
tree_rmse


from sklearn.model_selection import cross_val_score
scores = cross_val_score(tree_reg,housing_prepared,housing_labels,scoring='neg_mean_squared_error',cv=10)
tree_rmse_scores = np.sqrt(-scores)
tree_rmse_scores


def display_scores(scores):
    print("Scores:",scores)
    print("Mean:",scores.mean())
    print("STD:",scores.std())


lin_scores = cross_val_score(lin_reg,housing_prepared,housing_labels,
scoring='neg_mean_squared_error',cv=10)

lin_rmse_scores = np.sqrt(-lin_scores)

#How to save model using Job lib

import joblib
joblib.dump(tree_reg,'decisiontree.sav')

loaded_model = joblib.load('decisiontree.sav')
loaded_scores = cross_val_score(loaded_model,housing_prepared,
housing_labels,scoring='neg_mean_squared_error',cv=10)
loaded_tree_rmse_scores = np.sqrt(-loaded_scores)
