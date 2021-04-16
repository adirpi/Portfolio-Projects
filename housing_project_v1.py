import os
import tarfile
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from pandas.plotting import scatter_matrix
housing=pd.read_csv('https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.csv')
# print(housing.info())
# print (housing['ocean_proximity'].value_counts())
# print (housing.describe())

# housing.hist(bins=50,figsize=(20,15))
# plt.show()

def split_train_test(data,test_ratio):
    shuffled_indices=np.random.permutation(len(data))
    test_set_size = int(len(data)*test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices= shuffled_indices[test_set_size:]
    return data.iloc[train_indices],data.iloc[test_indices]

train_set,test_set = train_test_split(housing,test_size=0.2,random_state=42)

#Stratified sampling based on Income Category
housing['income_cat']=pd.cut(housing['median_income'],bins=[0.,1.5,3.0,4.5,6.,np.inf],labels=[1,2,3,4,5])
# housing['income_cat'].hist()
#plt.show()

#use sklearn to do Stratified Sampling
from sklearn.model_selection import StratifiedShuffleSplit
split= StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42)
for train_index,test_index in split.split(housing,housing['income_cat']):
    strat_train_set=housing.loc[train_index]
    strat_test_set=housing.loc[test_index]

for set in (strat_test_set,strat_train_set):
    set.drop('income_cat',axis=1,inplace=True)

# print(strat_test_set.head())

#Keep aside the
housing_train=strat_train_set.copy()

strat_train_set.plot(kind='scatter',x='longitude',y='latitude',alpha=0.4,s=strat_train_set['population']/100,label='population'
,figsize=(10,7),c='median_house_value',cmap=plt.get_cmap("jet"),colorbar=True)
# plt.legend()
# plt.show()

corr_matrix= strat_train_set.corr()
corr_matrix['median_house_value'].sort_values(ascending=False)
attributes=['median_house_value','median_income','total_rooms','housing_median_age']

scatter_matrix(housing[attributes],figsize=(12,8))
housing.plot(kind='scatter',x='median_income',
y='median_house_value',alpha=0.1)
plt.show()
