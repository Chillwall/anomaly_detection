import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'   
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import keras
from keras import layers
from tensorflow.keras import layers, losses
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
from tensorflow.keras.models import Model
from sklearn.preprocessing import StandardScaler

def load_dataset(path):
    """
    load dataset from local or external url
    :param path: the location of dataset
    :return: dataset in Dataframe format
    """
    df = pd.read_csv(path, header=None)
    return df

def preprocessing(df_train,df_pred):
    """
    preprcessing including drop missing value, normlizationt
    :param df: input dataset
    :return:    prediction set
    """
    #delete missing value in dataset
    df_train=df_train.dropna()
    df_pred = df_pred.dropna()
    data = df_train.iloc[:,:-1].values
    data_pred = df_pred.values
    #Normalize the data
    #Calculate the mean and standard deviation value from the training set 
    mean = tf.math.reduce_mean(data)
    std = tf.math.reduce_std(data)
    
    #Normalization formula (data - mean)/std
    df_pred = (data_pred - mean)/std  
    #Convert the data into float
    df_pred = tf.cast(df_pred, dtype=tf.float32)
    #convert format ready for model training
    x_pred = np.expand_dims(df_pred,axis=2)
    return x_pred

#load both training set and predicting set
df_train = load_dataset('../data/training.csv')
df_pred = load_dataset('../data/predicting.csv')
#preprocessing on predicting set
x_pred = preprocessing(df_train,df_pred)
#load the model
autoencoder = keras.models.load_model('../models/autoencoder_model.keras')
#predicting
x_pred_pred = autoencoder.predict(x_pred)
#compute MAE
test_mae_loss = np.mean(np.abs(x_pred_pred - x_pred), axis=1)
## this threshold is decided by mae graph, you may need to change this value everytime train the model
threshold = 0.3
#decide anomalies
result = test_mae_loss < threshold
result = [i for row in result for i in row]
#save result
df_result = pd.DataFrame(result,columns=['result'])
df_result.to_csv('../result/predicting.csv')
print('Results saved at /result/predicting.csv')
