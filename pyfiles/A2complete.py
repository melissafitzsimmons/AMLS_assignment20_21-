
import numpy as np
from tqdm import tqdm
import pandas as pd
import cv2
import os

from sklearn import datasets
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report,accuracy_score

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import to_categorical
from keras.preprocessing import image
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPool2D
from keras.utils import np_utils
from keras.layers.normalization import BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping

import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

print('Libraries imported')

#Join file directories for img and csv file

p = os.getcwd()
print(p)

dirname = os.path.dirname(p)
csvfile = os.path.join(p, 'datasets/celeba/labels.csv')
dirname = os.path.dirname(p)
imgfile = os.path.join(p, 'datasets/celeba/img/')

print('csv and img files joined to current directory')

dataset = pd.read_csv(csvfile, sep='\\t', engine='python')

dataset.head()

#Load images 50% of original size and preprocess
dataset_image = []
for i in tqdm(range(dataset.shape[0])):
    img = image.load_img(imgfile+dataset['img_name'][i], target_size=(109,89))
    #convert to greyscale
    img = img.convert('L')
    img = image.img_to_array(img)
    #normalise
    img = img/255

    dataset_image.append(img)
    
#Label the data
# Load the data
x = np.array(dataset_image)
y = np.array(dataset['smiling']+1)/2

# Split the data into training and testing(75% training and 25% testing data)
x_train, x_test, y_train, y_test = train_test_split(x, y,test_size=0.25,shuffle=False,random_state=0)
print('Data has been preprocessed and split into 75% training and 25% validation')

#Change data type to 'float32'
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

print('CNN initially made with one convolutional layer, one hidden layer and an output layer with Adam optimizer')
print('Early stopping is used so that the epochs stop after val_loss has not reached a minimum for 10 epochs')
print
print('Firstly, the batch size is optimised')

#Evaluate batch size = 128
# building a linear stack of layers with the sequential model
model = Sequential()
# convolutional layer
model.add(Conv2D(25, kernel_size=(3,3), strides=(1,1), padding='valid', activation='relu', input_shape=(109,89,1)))
model.add(MaxPool2D(pool_size=(1,1)))
# flatten output of conv
model.add(Flatten())
# hidden layer
model.add(Dense(100, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.1))
# output layer
model.add(Dense(1, activation='sigmoid'))

# compiling the sequential model
model.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer='adam')
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)

# training the model for 10 epochs
model.fit(x_train, y_train, batch_size=128, epochs=50, validation_data=(x_test, y_test), callbacks=[es])
print('Batch size=128 accuracy is taken from final epoch')



#Evaluate batch size = 64
# building a linear stack of layers with the sequential model
model = Sequential()
# convolutional layer
model.add(Conv2D(25, kernel_size=(3,3), strides=(1,1), padding='valid', activation='relu', input_shape=(109,89,1)))
model.add(MaxPool2D(pool_size=(1,1)))
# flatten output of conv
model.add(Flatten())
# hidden layer
model.add(Dense(100, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.1))
# output layer
model.add(Dense(1, activation='sigmoid'))

# compiling the sequential model
model.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer='adam')
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)

# training the model for 10 epochs
model.fit(x_train, y_train, batch_size=64, epochs=50, validation_data=(x_test, y_test), callbacks=[es])
print('Batch size=64 accuracy is taken from final epoch')



#Evaluate batch size = 256
# building a linear stack of layers with the sequential model
model = Sequential()
# convolutional layer
model.add(Conv2D(25, kernel_size=(3,3), strides=(1,1), padding='valid', activation='relu', input_shape=(109,89,1)))
model.add(MaxPool2D(pool_size=(1,1)))
# flatten output of conv
model.add(Flatten())
# hidden layer
model.add(Dense(100, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.1))
# output layer
model.add(Dense(1, activation='sigmoid'))

# compiling the sequential model
model.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer='adam')
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)

# training the model for 10 epochs
model.fit(x_train, y_train, batch_size=256, epochs=100, validation_data=(x_test, y_test), callbacks=[es])
print('Batch size=256 accuracy was taken from final epoch')
print('Batch size 64 was selected')
print
print('Next, the optimiser was evluated')

#Evaluate using SGD optimser
# building a linear stack of layers with the sequential model
model = Sequential()
# convolutional layer
model.add(Conv2D(25, kernel_size=(3,3), strides=(1,1), padding='valid', activation='relu', input_shape=(109,89,1)))
model.add(MaxPool2D(pool_size=(1,1)))
# flatten output of conv
model.add(Flatten())
# hidden layer
model.add(Dense(100, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.1))
# output layer
model.add(Dense(1, activation='sigmoid'))

# compiling the sequential model
model.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer='SGD')
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)

# training the model for 10 epochs
model.fit(x_train, y_train, batch_size=64, epochs=100, validation_data=(x_test, y_test), callbacks=[es])
print('Accuracy using SGD optimizer was taken from final epoch')
      
      
      
#Evaluate using 'RMSprop' optimser

# building a linear stack of layers with the sequential model
model = Sequential()
# convolutional layer
model.add(Conv2D(25, kernel_size=(3,3), strides=(1,1), padding='valid', activation='relu', input_shape=(109,89,1)))
model.add(MaxPool2D(pool_size=(1,1)))
# flatten output of conv
model.add(Flatten())
# hidden layer
model.add(Dense(100, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.1))
# output layer
model.add(Dense(1, activation='sigmoid'))

# compiling the sequential model
model.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer='RMSprop')
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)

# training the model for 10 epochs
model.fit(x_train, y_train, batch_size=64, epochs=100, validation_data=(x_test, y_test), callbacks=[es])

print('Accuracy using SGD optimizer was taken from final epoch')
print('The initial optimizer Adam has achieved the highest accuracy, so is selected')
print
print('The number of convolutional layers was investigated')

#Add aditional convolutional layer
# building a linear stack of layers with the sequential model
model = Sequential()
# convolutional layer
model.add(Conv2D(25, kernel_size=(3,3), strides=(1,1), padding='valid', activation='relu', input_shape=(109,89,1)))
model.add(MaxPool2D(pool_size=(1,1)))
# convolutional layer
model.add(Conv2D(25, kernel_size=(3,3), strides=(1,1), padding='valid', activation='relu', input_shape=(109,89,1)))
model.add(MaxPool2D(pool_size=(1,1)))
# flatten output of conv
model.add(Flatten())
# hidden layer
model.add(Dense(100, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.1))
# output layer
model.add(Dense(1, activation='sigmoid'))

# compiling the sequential model
model.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer='Adam')
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)

# training the model for 10 epochs
history=model.fit(x_train, y_train, batch_size=64, epochs=100, validation_data=(x_test, y_test), callbacks=[es])
print('The accuracy with 2 convolutional layers was taken from final epoch')

#Add additional convolutional layer
# building a linear stack of layers with the sequential model
model = Sequential()
# convolutional layer
model.add(Conv2D(25, kernel_size=(3,3), strides=(1,1), padding='valid', activation='relu', input_shape=(109,89,1)))
model.add(MaxPool2D(pool_size=(1,1)))
# convolutional layer
model.add(Conv2D(25, kernel_size=(3,3), strides=(1,1), padding='valid', activation='relu', input_shape=(109,89,1)))
model.add(MaxPool2D(pool_size=(1,1)))
# convolutional layer
model.add(Conv2D(25, kernel_size=(3,3), strides=(1,1), padding='valid', activation='relu', input_shape=(109,89,1)))
model.add(MaxPool2D(pool_size=(1,1)))
# flatten output of conv
model.add(Flatten())
# hidden layer
model.add(Dense(100, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.1))
# output layer
model.add(Dense(1, activation='sigmoid'))

# compiling the sequential model
model.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer='Adam')
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)

# training the model for 10 epochs
model.fit(x_train, y_train, batch_size=64, epochs=100, validation_data=(x_test, y_test), callbacks=[es])
print('The accuracy with 3 convolutional layers was taken from final epoch')
print('3 convolutional layers were selected')
print('Due to memory issues, the CNN is unable to run with any further convolutional layers')
print
print('Next, the number of hidden layers was optimised')

# #Evaluating 2 hidden layers

# building a linear stack of layers with the sequential model
model = Sequential()
# convolutional layer
model.add(Conv2D(25, kernel_size=(3,3), strides=(1,1), padding='valid', activation='relu', input_shape=(109,89,1)))
model.add(MaxPool2D(pool_size=(1,1)))
# convolutional layer
model.add(Conv2D(25, kernel_size=(3,3), strides=(1,1), padding='valid', activation='relu', input_shape=(109,89,1)))
model.add(MaxPool2D(pool_size=(1,1)))
# convolutional layer
model.add(Conv2D(25, kernel_size=(3,3), strides=(1,1), padding='valid', activation='relu', input_shape=(109,89,1)))
model.add(MaxPool2D(pool_size=(1,1)))
# flatten output of conv
model.add(Flatten())
# hidden layer
model.add(Dense(100, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.1))
# hidden layer
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.1))
# output layer
model.add(Dense(1, activation='sigmoid'))

# compiling the sequential model
model.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer='Adam')
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)

# training the model for 10 epochs
history=model.fit(x_train, y_train, batch_size=64, epochs=100, validation_data=(x_test, y_test), callbacks=[es])
print('The accuracy with 2 hidden layers was taken from final epoch')
      

#Evaluating 3 hidden layers

# building a linear stack of layers with the sequential model
model = Sequential()
# convolutional layer
model.add(Conv2D(25, kernel_size=(3,3), strides=(1,1), padding='valid', activation='relu', input_shape=(109,89,1)))
model.add(MaxPool2D(pool_size=(1,1)))
# convolutional layer
model.add(Conv2D(25, kernel_size=(3,3), strides=(1,1), padding='valid', activation='relu', input_shape=(109,89,1)))
model.add(MaxPool2D(pool_size=(1,1)))
# convolutional layer
model.add(Conv2D(25, kernel_size=(3,3), strides=(1,1), padding='valid', activation='relu', input_shape=(109,89,1)))
model.add(MaxPool2D(pool_size=(1,1)))
# flatten output of conv
model.add(Flatten())
# hidden layer
model.add(Dense(100, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.1))
# hidden layer
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.1))
# hidden layer
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.1))
# output layer
model.add(Dense(1, activation='sigmoid'))

# compiling the sequential model
model.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer='Adam')
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)

# training the model for 10 epochs
history=model.fit(x_train, y_train, batch_size=64, epochs=100, validation_data=(x_test, y_test), callbacks=[es])
print('The accuracy with 3 hidden layers was taken from final epoch') 
print('2 hidden layers were selected')

print('Optimised model selected with 3 convolutional layers, 2 hidden layers, optimizer=Adam, batch size=64')
print
print('Training convergence on optimised model')
      # list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
      
print('Accuracy on test set was evaluated using optimized model')
#Accuracy on test set evaluated
dirname = os.path.dirname(p)
csvtest = os.path.join(p, 'test/celeba_test/labels.csv')
dirname = os.path.dirname(p)
imgtest = os.path.join(p, 'test/celeba_test/img/')

#Use Pandas to read csv file
test_dataset = pd.read_csv(csvtest, sep='\\t', engine='python')

test_dataset_image = []
for i in tqdm(range(test_dataset.shape[0])):
    img = image.load_img(imgtest+test_dataset['img_name'][i], target_size=(109,89))
    img = img.convert('L')
    img = image.img_to_array(img)
    img = img/255

    test_dataset_image.append(img)

      # Split the data
X_TEST = np.array(test_dataset_image)
Y_TEST = np.array(test_dataset['smiling']+1)/2
print('Test data has been preprocessed')
      
# building a linear stack of layers with the sequential model
model = Sequential()
# convolutional layer
model.add(Conv2D(25, kernel_size=(3,3), strides=(1,1), padding='valid', activation='relu', input_shape=(109,89,1)))
model.add(MaxPool2D(pool_size=(1,1)))
# convolutional layer
model.add(Conv2D(25, kernel_size=(3,3), strides=(1,1), padding='valid', activation='relu', input_shape=(109,89,1)))
model.add(MaxPool2D(pool_size=(1,1)))
# convolutional layer
model.add(Conv2D(25, kernel_size=(3,3), strides=(1,1), padding='valid', activation='relu', input_shape=(109,89,1)))
model.add(MaxPool2D(pool_size=(1,1)))
# flatten output of conv
model.add(Flatten())
# hidden layer
model.add(Dense(100, activation='relu'))
model.add(BatchNormalization())
# hidden layer
model.add(Dense(100, activation='relu'))
# output layer
model.add(Dense(1, activation='sigmoid'))

# compiling the sequential model
model.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer='Adam')
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)

# training the model for 10 epochs
history=model.fit(x_train, y_train, batch_size=64, epochs=100, validation_data=(X_TEST, Y_TEST), callbacks=[es])
