# -*- coding: utf-8 -*-

#for adding several features to the data
def add_feature(data):
    data["x_y"]=(data.Axis1*data.Axis2) 
    data["y_z"]=(data.Axis2*data.Axis3)
    data["x_z"]=(data.Axis1*data.Axis3)
    data["Multiply"]=((data.Axis1*data.Axis1)+(data.Axis2*data.Axis2)+(data.Axis3*data.Axis3))
    data["Mag_Vec_Feature"] = np.sqrt(data.Multiply) 
    return data 


#for plotting the confusion matrixS    
def show_confusion_matrix(validations, predictions ,LABELS):

    matrix = metrics.confusion_matrix(validations, predictions)
    plt.figure(figsize=(6, 4))
    sns.heatmap(matrix,
                cmap='coolwarm',
                linecolor='white',
                linewidths=1,
                xticklabels=LABELS,
                yticklabels=LABELS,
                annot=True,
                fmt='d')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()   

#for preprocessing the test data    
def test_preprocess(time_steps,step):
    # Number of steps to advance in each iteration (for me, it should always 
    # be equal to the time_steps in order to have no overlap between segments)
    # step = time_steps
    segments = []
    labels = []
    for i in range(0, len(data) - time_steps, step):
        xs = data['Axis1'].values[i: i + time_steps]
        ys = data['Axis2'].values[i: i + time_steps]
        zs = data['Axis3'].values[i: i + time_steps]
        x_y = data['x_y'].values[i: i + time_steps]
        y_z = data['y_z'].values[i: i + time_steps]
        x_z = data['x_z'].values[i: i + time_steps]
        Mag= data['Mag_Vec_Feature'].values[i: i + time_steps]
        # Retrieve the most often used label in this segment
        label = sc.stats.mode(data["label_name"][i: i + time_steps])[0][0]
        segments.append([xs, ys, zs, Mag, x_y, y_z, x_z])
        labels.append(label)
    #one hot encoding    
    labels = np.asarray(labels)
    labels = to_categorical(labels)    
    return segments,labels
        
    

    
     
#%%
import numpy as np   
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.metrics import classification_report 
import pickle    
import scipy as sc

from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, LSTM, Flatten, core,Reshape
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.callbacks import ModelCheckpoint
from keras.models import model_from_json
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Reshape
from keras.layers import Conv2D, MaxPooling2D,GlobalAveragePooling1D
from keras.utils import np_utils,to_categorical
from keras.models import load_model

#%%Reading the Raw Test data
data = pd.read_csv("real_accelerometer_data.csv")
#%%Reading the label encoder file written in Train file
pkl_file = open('encoder.pkl', 'rb')
le_new = pickle.load(pkl_file) 

#%%Adding the fetures
data=add_feature(data)

#%%Encoding the class labels
# Add a new column to the existing DataFrame with the encoded values
data["label_name"] = le_new.transform(data["label_name"].values.ravel())
LABELS=list(le_new.classes_)
print(list(le_new.classes_)) 

#%%preprocessing the test data
#test_preprocess(time_steps,step),advisable to go with value 30 for both step and time_steps
test,labels=test_preprocess(30,30)

#%%reshaping the test data
test=np.array(test)

test=np.transpose(test,(0,2,1))

#%%testing the test data with the model
model = load_model('ML_Model.h5')
y_pred_test = model.predict(test)
max_y_pred_test = np.argmax(y_pred_test, axis=1)
max_y_test = np.argmax(labels, axis=1)
show_confusion_matrix(max_y_test, max_y_pred_test,LABELS)
print(classification_report(max_y_test, max_y_pred_test))
#%%