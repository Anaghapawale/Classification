#Importing Liabraries
import numpy as np
import pandas as pd
#Importing Liabraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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

from sklearn import model_selection
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from math import sqrt
import time
from sklearn.metrics import accuracy_score
import pickle


#Reading the dataset
def load_data(file_name):
    dataset = pd.read_csv(file_name).fillna('NONE')
    print("Data loaded:"+str(dataset.shape))
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
    dataset['handle']=dataset['Quote'].str.contains('H')
    dataset['negative_sign']=dataset['Value'].apply( lambda x: str(x)[0]=='-' and len(str(x))>1)
    return dataset

def feature_dp(dataset):
    dataset.drop(['Quote'], axis=1, inplace=True)
    return dataset

def res_in(dataset):
    dataset.reset_index(drop=True, inplace=True)
    return dataset

def catg_encode(dataset,data_cols):
    # This function will encode categorical varibles using label encoder
    le={}
    for col in data_cols:
        le[str(col)] = LabelEncoder()
        dataset[col] = le[str(col)].fit_transform(dataset[col])  
    print("Data after encoding categorical variables: ")
    return dataset, le

#Feature Scaling

def scaleColumns(dataset):
    scaler = preprocessing.MinMaxScaler()
    min_max_scaler = preprocessing.MinMaxScaler()
    cols_to_scale = ['Quotes','Value']
    for col in cols_to_scale:
        dataset[col] = pd.DataFrame(min_max_scaler.fit_transform(pd.DataFrame(dataset[col])),columns=[col])
    return dataset

# Create X, y for classification where X is a matrix of independent variable and y is a matrix of dependent variable
def verified_xy(dataset):
    
    X = dataset.drop(['Px\Spr\Yld'],axis=1)
    y = dataset[['Px\Spr\Yld']]
    return X,y

def timeseries_data(dataset, X, y):

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.70, test_size=0.30, random_state=0)

    X_train = X[:int(X.shape[0]*0.7)]
    X_test = X[int(X.shape[0]*0.7):]
    y_train = y[:int(X.shape[0]*0.7)]
    y_test = y[int(X.shape[0]*0.7):]


    dataset = TimeSeriesSplit()
    print(dataset)
    TimeSeriesSplit(max_train_size=7, n_splits=3)
    for train_index, test_index in dataset.split(X):
         print("TRAIN:", train_index, "TEST:", test_index)
    
         X_train, X_test = X.iloc[train_index], X.iloc[test_index]
         y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
         return X_train, X_test, y_train, y_test
     


#RF model
def randomforestmodel1(X_train,y_train,X_test,y_test):
    t0 = time.time()
    model_rf = RandomForestRegressor(n_estimators=5000, oob_score=True, random_state=100)
    model_rf.fit(X_train, y_train) 
    pred_train_rf= model_rf.predict(X_train)
    print("Random Forest Model Accuracy")
    print(np.sqrt(mean_squared_error(y_train,pred_train_rf)))
    print(r2_score(y_train, pred_train_rf))
    tF = time.time()

    pred_test_rf = model_rf.predict(X_test)
    print(np.sqrt(mean_squared_error(y_test,pred_test_rf)))
    print(r2_score(y_test, pred_test_rf))
    print('Time to train = %.2f seconds' % (tF - t0))
    
    return np.sqrt,r2_score,np.sqrt,r2_score

#Main
# Train model, validate the results and then retrain the model using entire training sample for final predictions
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from math import ceil

file_name = 'C:/Users/Aumni/Documents/ML Project/Training_4classes (1).csv'

dataset = load_data(file_name)
dataset = missing_val(dataset)
dataset = get_timestamp(dataset)
dataset, dataset_timestamp = timestamp_encode(dataset)
dataset = special_character(dataset)
dataset = feature_dp(dataset)
dataset = res_in(dataset)
data_cols = ['Identifier','Asset Class','Region','Provider','Type','Bmrk','Px\Spr\Yld','hyphen','percentage','plus','decimal','dollar','handle','negative_sign']
dataset, le = catg_encode(dataset,data_cols)
dataset = scaleColumns(dataset)
X,y = verified_xy(dataset)
X_train, X_test, y_train, y_test = timeseries_data(dataset, X, y) 
#np.sqrt, r2_score, np.sqrt, r2_score = decisiontree1(X_train, y_train, X_test, y_test)
np.sqrt, r2_score, np.sqrt, r2_score = randomforestmodel1(X_train, y_train, X_test, y_test)
filename = 'randomforest_model.pkl'
pickle.dump(randomforestmodel1, open(filename, 'wb'))