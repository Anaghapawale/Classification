#Importing Liabraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from datetime import datetime
from math import sin, cos, pi
from sklearn import preprocessing

#Data preprocessing and result validation liabraries
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn import datasets
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn import preprocessing
from sklearn.model_selection import TimeSeriesSplit

#Reading the dataset
def load_data(file_name):
    dataset = pd.read_csv(file_name).fillna('NONE')
    print("Data loaded:"+str(dataset.shape))
    print(dataset.head())
    return dataset
  
#Filling the missing values

def missing_val(dataset):
    dataset['Bmrk']=dataset['Bmrk'].fillna('NONE')
    return dataset
  
 # Converting CSV date field to  python timestamp format
def get_timestamp(dataset):
    dataset_d = pd.DataFrame()
    dataset_d['Year']=dataset['Date'].apply(lambda x: int(x.split(' ')[0].split('/')[2]))
    dataset_d['Month']=dataset['Date'].apply(lambda x: int(x.split(' ')[0].split('/')[0]))
    dataset_d['Day']=dataset['Date'].apply(lambda x: int(x.split(' ')[0].split('/')[1]))
    dataset_d['Hr']=dataset['Date'].apply(lambda x: int(x.split(' ')[1].split(':')[0]))
    dataset_d['Min']=dataset['Date'].apply(lambda x: int(x.split(' ')[1].split(':')[1]))
    
    dataset_time = []
    for index, row in dataset_d.iterrows():
        
        time = datetime(dataset_d['Year'][index],
                       dataset_d['Month'][index],
                       dataset_d['Day'][index],
                       dataset_d['Hr'][index],
                       dataset_d['Min'][index])
        dataset_time.append(time)
    
    dataset_time = pd.DataFrame(dataset_time,columns=['TimeStamp'])
    dataset = dataset.join(dataset_time).drop(['Date'],axis=1)
    
    print("Dates converted to timestamp")

    del dataset_d,dataset_time
    return dataset
  
  ##############################################################################################
# Encoding timeStamp
# Time stamps are encoded into fourier serieses. After encoding timestamp is converted into
# unique set of 10 numbers which indicates the cycles. For example, there are 12 months in 
# a year. So my encoded value for the month of June will be sin(2*pi*6/12) & cos(2*pi*6/12)
##############################################################################################

days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

def sin_cos(n):
    theta = 2 * pi * n
    return (sin(theta), cos(theta))

def get_cycles(d):
    month = d.month - 1
    day = d.day - 1
    
    m_sin, m_cos = sin_cos(month/12)
    d_sin, d_cos = sin_cos(day / days_in_month[month])
    wd_sin, wd_cos = sin_cos(d.weekday() / 7)
    h_sin, h_cos = sin_cos(d.hour / 24)
    min_sin,min_cos = sin_cos(d.minute / 60)
    s_sin,s_cos = sin_cos(d.second / 60)
    
    del month, day
    return [m_sin, m_cos,d_sin, d_cos,wd_sin, wd_cos,
           h_sin, h_cos,min_sin,min_cos,s_sin,s_cos]

# Convert timestamp to fourier serieses to capture seasonality
def timestamp_encode(dataset):
    dataset_timestamp = pd.DataFrame(dataset['TimeStamp'])
    dataset_cycles = []
    for i, row in dataset_timestamp.iterrows():
        cycles = get_cycles(dataset_timestamp.iloc[i,0])
        dataset_cycles.append(cycles)

    dataset_cycles= pd.DataFrame(dataset_cycles,columns=['Month_sine','Month_cos','Day_sine','Day_cos','wd_sine','wd_cos',
                               'Hr_sine','Hr_cos','Min_sine','Min_cos','Sec_sine','Sec_cos'])

    dataset = pd.concat([dataset,dataset_cycles],axis=1).drop(['TimeStamp'],axis=1)
    print("Timeseries features extracted from timestamp encoding:"+str(dataset.shape))
    del dataset_cycles, cycles
    
    return dataset, dataset_timestamp
  
  
  #Applying One hot encoding for Quote column for special character
#From Quote trying to identify the Price, Yield, Spread


def special_character(dataset):
    dataset['hyphen']=dataset['Quote'].str.contains('-')
    dataset['percentage']=dataset['Quote'].str.contains('%')
    dataset['plus']=dataset['Quote'].str.contains('\+')
    dataset['decimal']=dataset['Quote'].str.contains('\.')
    dataset['dollar']=dataset['Quote'].str.contains('\$')
    return dataset
  
  def feature_dp(dataset):
    dataset.drop(['Quote'], axis=1, inplace=True)
    return dataset
  
  def res_in(dataset):
    dataset.reset_index(drop=True, inplace=True)
    return dataset
  
  # Encoding categorical variable
def cat_encode(dataset, for_test = False, le=None, le_type=None, le_provider=None, le_hyphen=None, le_percentage=None, le_plus=None, le_decimal=None, le_dollar=None, le_identifier=None, le_assest_class=None, le_region=None):
    
    if for_test == False:
        # Encoding categorical data - Type and Bmrk
        le = LabelEncoder()
        dataset['Bmrk'] = le.fit_transform(dataset['Bmrk'])
        
        # Label encoding of Quote Type
        le_type = LabelEncoder()
        dataset['Type'] = le_type.fit_transform(dataset['Type'])
        
        le_provider = LabelEncoder()
        dataset['Provider'] = le_provider.fit_transform(dataset['Provider'])
        
        le_hyphen = LabelEncoder()
        dataset['hyphen'] = le_hyphen.fit_transform(dataset['hyphen'])

        le_percentage = LabelEncoder()
        dataset['percentage'] = le_percentage.fit_transform(dataset['percentage'])

        le_plus = LabelEncoder()
        dataset['plus'] = le_plus.fit_transform(dataset['plus'])
        
        le_decimal = LabelEncoder()
        dataset['decimal'] = le_decimal.fit_transform(dataset['decimal'])
  
        le_dollar = LabelEncoder()
        dataset['dollar'] = le_dollar.fit_transform(dataset['dollar'])
        
        le_identifier = LabelEncoder()
        dataset['Identifier'] = le_identifier.fit_transform(dataset['Identifier'])
        
        le_assetclass = LabelEncoder()
        dataset['Asset Class'] = le_assetclass.fit_transform(dataset['Asset Class'])
        
        le_region = LabelEncoder()
        dataset['Region'] = le_region.fit_transform(dataset['Region'])
        
        print("Categorical variables encoded")    
        
    else:
        dataset['Bmrk'] = pd.DataFrame(le.transform(dataset['Bmrk']))
        dataset['Type'] = pd.DataFrame(le_type.transform(dataset['Type']))
        dataset['Provider'] = pd.DataFrame(le_provider.transform(dataset['Provider']))
        dataset['hyphen'] = pd.DataFrame(le_hyphen.transform(dataset['hyphen']))
        dataset['percentage'] = pd.DataFrame(le_percentage.transform(dataset['percentage']))
        dataset['plus'] = pd.DataFrame(le_plus.transform(dataset['plus']))
        dataset['decimal'] = pd.DataFrame(le_decimal.transform(dataset['decimal']))
        dataset['dollar'] = pd.DataFrame(le_dollar.transform(dataset['dollar']))
        dataset['identifier'] = pd.DataFrame(le_identifier.transform(dataset['identifier']))
        dataset['assestclass'] = pd.DataFrame(le_assetclass.transform(dataset['assetclass']))
        dataset['region'] = pd.DataFrame(le_region.transform(dataset['region']))
        print("Categorical variables encoded")
 
    return dataset, le, le_type, le_provider, le_hyphen, le_percentage, le_plus, le_decimal, le_dollar, le_identifier,le_assetclass, le_region

def one_hot(dataset):

    cols = ['Px\Spr\Yld']
    for each in cols:
        dummies = pd.get_dummies(dataset[each], drop_first=False)
        dataset = pd.concat([dataset, dummies], axis=1)
        print(dataset.head())
    return dataset
  
  def feature_px(dataset):
    dataset.drop(['Px\Spr\Yld'],axis=1, inplace=True)
    return dataset
  
  def res_ind(dataset):
    dataset.reset_index(drop=True, inplace=True)
    return dataset
  
  #Feature Scaling

def scaleColumns(dataset):
    scaler = preprocessing.MinMaxScaler()
    min_max_scaler = preprocessing.MinMaxScaler()
    cols_to_scale = ['Value','Quotes']
    for col in cols_to_scale:
        dataset[col] = pd.DataFrame(min_max_scaler.fit_transform(pd.DataFrame(dataset[col])),columns=[col])
    return dataset
  
  # Create X, y for classification where X is a matrix of independent variable and y is a matrix of dependent variable
def verified_xy(dataset):
    
    X = dataset.drop(['PAYUP','PRICE','SPREAD','YIELD'],axis=1)
    y = dataset[['PAYUP','PRICE','SPREAD','YIELD']]
    return X,y
  
  from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.70, test_size=0.30, random_state=0)

X_train = X[:int(X.shape[0]*0.7)]
X_test = X[int(X.shape[0]*0.7):]
y_train = y[:int(X.shape[0]*0.7)]
y_test = y[int(X.shape[0]*0.7):]

import numpy as np
from sklearn.model_selection import TimeSeriesSplit

def timeseries_data(dataset, X, y):
    dataset = TimeSeriesSplit()
    print(dataset)
    TimeSeriesSplit(max_train_size=7, n_splits=3)
    for train_index, test_index in dataset.split(X):
         print("TRAIN:", train_index, "TEST:", test_index)
    
         X_train, X_test = X.iloc[train_index], X.iloc[test_index]
         y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
         return X_train, X_test, y_train, y_test
    
  #Main
# Train model, validate the results and then retrain the model using entire training sample for final predictions
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from math import ceil

file_name = 'D:/Users/apawale/OneDrive - Solve Advisors Inc/Documents/My Training/ML Project/Training_4classes.csv'

dataset = load_data(file_name)
dataset = missing_val(dataset)
dataset = get_timestamp(dataset)
dataset, dataset_timestamp = timestamp_encode(dataset)
dataset = special_character(dataset)
dataset = feature_dp(dataset)
dataset = res_in(dataset)
dataset, le, le_type, le_provider, le_hyphen, le_percentage, le_plus, le_decimal, le_dollar, le_identifier, le_assetclass, le_region = cat_encode(dataset)
dataset = one_hot(dataset)
dataset = feature_px(dataset)
dataset = res_ind(dataset)
dataset = scaleColumns(dataset)
X,y = verified_xy(dataset)
X_train, X_test, y_train, y_test = timeseries_data(dataset, X, y)
print(dataset.head())

from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor()
regressor.fit(X_train,y_train)
print("Coefficient of determination R^2 <-- train set: {}".format(regressor.score(X_train,y_train)))
print("Coefficient of determination R^2 <-- train set: {}".format(regressor.score(X_test,y_test)))
from sklearn.model_selection import cross_val_score
score=cross_val_score(regressor,X,y,cv=5)
score.mean()
prediction=regressor.predict(X_test)
sns.distplot(y_test-prediction)
plt.scatter(y_test, prediction)
# Hyperparameter tuning
RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
n_estimators = [int(x) for x in np.linspace(start=100, stop=1200, num=12)]
print(n_estimators)
n_estimators =[int(x) for x in np.linspace(start=100, stop=1200, num=12)]
max_features = ["auto","sqrt"]
max_depth = [int(x) for x in np.linspace(5, 30, num = 6)]
min_samples_split = [2, 5, 10, 15, 100]
min_samples_leaf = [1, 2, 5, 10]
random_grid = {"n_estimators": n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf':min_samples_leaf}
               
print(random_grid)
rf = RandomForestRegressor()
rf_random = RandomizedSearchCV(estimator = rf, param_distributions=random_grid, scoring='neg_mean_squared_error', n_iter = 100, cv = 5, verbose = 2, random_state = 42, n_jobs= 1)
rf_random.fit(X_train,y_train)
rf_random.best_params_
rf_random.best_score_
predictions = rf_random.predict(X_test)
sns.distplot(y_test-predictions)
plt.scatter(y_test,predictions)
from sklearn import metrics
print('MAE:',metrics.mean_absolute_error(y_test,prediction))
print('MSE:',metrics.mean_squared_error(y_test,prediction))
print("RMSE:",np.sqrt(metrics.mean_squared_error(y_test,prediction)))
import pickle
file = open('random_forest_regression_model.pkl','wb')
pickle.dump(rf_random, file)
