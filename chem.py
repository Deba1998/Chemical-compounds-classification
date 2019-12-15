# -*- coding: utf-8 -*-
"""
Created on Sun Dec 15 00:07:13 2019

@author: DELL
"""
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Importing the dataset
dataset = pd.read_csv('musk_csv.csv')
X = dataset.iloc[:, 3:-1].values
y = dataset.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(output_dim = 166, init = 'uniform', activation = 'relu', input_dim = 166))
# Adding the second hidden layer
classifier.add(Dense(output_dim = 166, init = 'uniform', activation = 'relu'))

# Adding the output layer
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
hist1=classifier.fit(X_train, y_train, batch_size = 50, nb_epoch = 50)

#predicting results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)
from sklearn.metrics import confusion_matrix #creating confusion matrix
cm = confusion_matrix(y_test, y_pred)
cm

from sklearn.metrics import precision_score,recall_score # calculating precision score and recall score
precision_score(y_test,y_pred)
recall_score(y_test,y_pred)

from sklearn.metrics import f1_score# calculating f1 score
f1_score(y_test,y_pred)
hist2=classifier.fit(X_test, y_test, batch_size = 50, nb_epoch =50)
from statistics import mean #calcutaing average loss over test set
val_loss=hist2.history['loss']
k=mean(val_loss)
k
from statistics import mean # calculating average accuracy over test set
val_acc=hist2.history['acc']
r=mean(val_acc)
r
# plotting results
train_loss=hist1.history['loss']
val_loss=hist2.history['loss']
xc=range(9)
plt.plot(xc,train_loss[0:9],color='blue')
plt.plot(xc,val_loss[0:9],color='orange')
plt.title('model loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['train','test'])
plt.show()

train_acc=hist1.history['acc']
val_acc=hist2.history['acc']
xc=range(9)
plt.plot(xc,train_acc[0:9],color='blue')
plt.plot(xc,val_acc[0:9],color='orange')
plt.title('model accuracy')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend(['train','test'])
plt.show()

classifier.save("modelchem5.h5")