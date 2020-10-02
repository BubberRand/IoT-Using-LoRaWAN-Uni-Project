# -*- coding: utf-8 -*-

#%%Functions
####Data segmenting

def seperating_features(data,activity,roll_size):  
    #rolling each of the columns in the data and loading them as different arays
    x=roll_ts(data[data["label_name"]==activity].Axis1,roll_size)
    y=roll_ts(data[data["label_name"]==activity].Axis2,roll_size)
    z=roll_ts(data[data["label_name"]==activity].Axis3,roll_size)
    Mg=roll_ts(data[data["label_name"]==activity].Mag_Vec_Feature,roll_size)
    x_y=roll_ts(data[data["label_name"]==activity].x_y,roll_size)
    y_z=roll_ts(data[data["label_name"]==activity].y_z,roll_size)
    x_z=roll_ts(data[data["label_name"]==activity].x_z,roll_size)
    return x, y, z, Mg, x_y, y_z, x_z
    

#for rolling different columns of the data 
def roll_ts(series,window):
    out=np.tile(series,(len(series)-window+1,1))
    index=np.empty([len(series)-window+1,window],dtype=int)
    for i in range(0,len(series)-window+1):
        index[i]=np.arange(i,i+window)
    return out[0,index]    


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
 
#for concatenating the output obtained from seperate_feature() function     
def concat(Sit,Walk,Run):
    concat=np.concatenate((Sit,Walk,Run),axis=0)
    concat=np.transpose(concat)
    return concat
#for creating the model 
def Convolution_Model(time_steps,features):
    model = Sequential()   
    model.add(Conv1D(filters=50, kernel_size=10, activation='relu', input_shape=(time_steps,features)))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(filters=25, kernel_size=1, activation='relu'))
    model.add(MaxPooling1D(pool_size=7))
    model.add(Flatten())
    model.add(Dense(3, activation='softmax'))
    model.compile(loss='categorical_crossentropy',optimizer='adam', metrics=['accuracy'])
    return model,print(model.input_shape),print(model.output_shape)

#for adding several features to the data
def add_feature(data):
    data["x_y"]=(data.Axis1*data.Axis2) 
    data["y_z"]=(data.Axis2*data.Axis3)
    data["x_z"]=(data.Axis1*data.Axis3)
    data["Multiply"]=((data.Axis1*data.Axis1)+(data.Axis2*data.Axis2)+(data.Axis3*data.Axis3))
    data["Mag_Vec_Feature"] = np.sqrt(data.Multiply) 
    return data 
#%% Libraries
#pandas and numpy
import numpy as np   
import pandas as pd
#sklearn libraries
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_predict, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.externals import joblib
from sklearn.metrics import classification_report 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn import linear_model
from sklearn import svm
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR

#%%keras packages
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


#%%scipy packages
from scipy import stats
import scipy as sc

#%%tensorflow
import tensorflow as tf
from tensorflow import keras

#%%other libraries which may be required
import pickle
import sys
print(sys.executable)

#%%Reading the whole data
data = pd.read_csv("train_Huge.csv")

#%%Adding more features to the data
data=add_feature(data)

#%%Encoding the class labels
# Define column name of the label vector
LABEL = 'ActivityEncoded'
# Transform the labels from String to Integer via LabelEncoder
le = preprocessing.LabelEncoder()
# Add a new column to the existing DataFrame with the encoded values
data["label_name"] = le.fit_transform(data["label_name"].values.ravel())
LABELS=list(le.classes_)   

#%%Saving the encoder for encoding the test data similarly 
output = open('encoder.pkl', 'wb')
pickle.dump(le, output)
output.close()

#%%Seperating all features for each activity.
#the value for the activity variable in seperation_feature() function should be given based upon the encoding output optained i.e LABELS 
#seperating_features(data,activity,roll_size)
#Sit
Sit_x,Sit_y,Sit_z,Sit_Mg,Sit_x_y,Sit_y_z,Sit_x_z=seperating_features(data,1,30)
#Walk
Walk_x,Walk_y,Walk_z,Walk_Mg,Walk_x_y,Walk_y_z,Walk_x_z=seperating_features(data,2,30)
#Run
Run_x,Run_y,Run_z,Run_Mg,Run_x_y,Run_y_z,Run_x_z=seperating_features(data,0,30)


#%%concating the results obtained from seperating_features() functions
#concat(Sit,Walk,Run), 
x1=concat(Sit_x,Walk_x,Run_x)
x2=concat(Sit_y,Walk_y,Run_y)
x3=concat(Sit_z,Walk_z,Run_z)
x4=concat(Sit_Mg,Walk_Mg,Run_Mg)
x5=concat(Sit_x_y,Walk_x_y,Run_x_y)
x6=concat(Sit_y_z,Walk_y_z,Run_y_z)
x7=concat(Sit_x_z,Walk_x_z,Run_x_z)

#%%creating reshaping the training dataset 
final=[x1,x2,x3,x4,x5,x6,x7]
final=np.array(final)
final=np.transpose(final)

#%%processing the dependent variable

S,W,R=pd.DataFrame(Sit_x),pd.DataFrame(Walk_x),pd.DataFrame(Run_x)


S["label_name"]=1
W["label_name"]=2
R["label_name"]=0

y1,y2,y3=S["label_name"],W["label_name"],R["label_name"]

label=pd.concat([y1,y2,y3],axis=0)

label = np.asarray(label)        
label = to_categorical(label)  

del S,W,R

#%%Traing and Testing split
#from sklearn.model_selection import train_test_split
#X_Train,X_Test,Y_Train,Y_Test= train_test_split(final,label,test_size=0.2,random_state=0)
#%%creating the 1D convolutional neural network model
#Convolution_Model(time_steps,features)
model,inpu,output=Convolution_Model(30,7)

#%%Fitting the model with the training data    
epochs,batch_size=30,50
model.fit(final, label, epochs=epochs, batch_size=batch_size)
  
#%%validating the model with test dataset 
#y_pred_test = model.predict(final)
    
    
#max_y_pred_test = np.argmax(y_pred_test, axis=1)
#max_y_test = np.argmax(label, axis=1)
#show_confusion_matrix(max_y_test, max_y_pred_test,LABELS)
#print(classification_report(max_y_test, max_y_pred_test))

#%% Save Model
model.save('ML_Model.h5')  