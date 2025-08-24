import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def read_dataframe(filename, categorical = ''):
    df = pd.read_parquet(filename, engine='fastparquet')
    df['duration'] = ((df['tpep_dropoff_datetime'] - df['tpep_pickup_datetime']).dt.total_seconds()/60).round(2)
    mask = (df['duration'] >= 1) & (df['duration'] <= 60)
    df = df[mask].copy()
    df[categorical] = df[categorical].astype(str)
    return df

def train_pipe(df_train, categorical, target):
    cat_train = df_train[categorical].copy()
    train_dicts = cat_train.to_dict('records')
    
    vectorizer = DictVectorizer(sparse=True)
    X_train = vectorizer.fit_transform(train_dicts)
        
    y_train = df_train[target].values
    return X_train, y_train

def val_pipe(df_val, categorical, vectorizer):
    cat_val = df_val[categorical].copy()
    val_dicts = cat_val.to_dict('records')
    vectorizer = DictVectorizer(sparse=True)
    X_val = vectorizer.transform(val_dicts)
    return X_val

filename_train = './data/yellow_tripdata_2025-01.parquet'

categorical = ['PULocationID', 'DOLocationID']

df_train = read_dataframe(filename_train, categorical)

#filename_val = './data/yellow_tripdata_2025-02.parquet'
#df_val = read_dataframe(filename_val, categorical)

X_train, y_train = train_pipe(df_train, categorical, 'duration')

lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_train)
mse = mean_squared_error(y_train, y_pred)
rmse = np.sqrt(mse)