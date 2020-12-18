import numpy as np
from tqdm import tqdm
import pandas as pd
import os

#from sklearn import datasets
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
#from sklearn.preprocessing import MinMaxScaler
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

print('Libraries have been imported')

p = os.getcwd()
print(p)

dirname = os.path.dirname(p)
csvfile = os.path.join(p, 'datasets/cartoon_set/labels.csv')
dirname = os.path.dirname(p)
imgfile = os.path.join(p, 'datasets/cartoon_set/img/')


dataset = pd.read_csv(csvfile, sep='\\t', engine='python')
dataset.head

print('csv and image path files have been joined to current directory')

dataset_image = []
for i in tqdm(range(dataset.shape[0])):
    img = image.load_img(imgfile+dataset['file_name'][i], target_size=(100,100))
    img = image.img_to_array(img)
    img = img/255

    dataset_image.append(img)

# Label the data
x = np.array(dataset_image)
y = dataset['eye_color']

# Split the data into training and testing(75% training and 25% testing data)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=100)

#Convert data type to 'float32'
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')



#One-hot encoding of data
n_classes = 5
print("Shape before one-hot encoding: ", y_train.shape)
y_train = np_utils.to_categorical(y_train, n_classes)
y_test = np_utils.to_categorical(y_test, n_classes)
print("Shape after one-hot encoding: ",y_train.shape)

print('Final model selected has 1 convolutional layer, 1 hidden layer, optimizer=SGD, batch size=128')

# building a linear stack of layers with the sequential model
model = Sequential()
# convolutional layer
model.add(Conv2D(25, kernel_size=(3,3), strides=(1,1), padding='valid', activation='relu', input_shape=(100,100,3)))
model.add(MaxPool2D(pool_size=(1,1)))
# flatten output of conv
model.add(Flatten())
# hidden layer
model.add(Dense(100, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.1))
# output layer
model.add(Dense(5, activation='softmax'))

# compiling the sequential model
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='SGD')
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)

# training the model for 100 epochs
history=model.fit(x_train, y_train, batch_size=128, epochs=100, validation_data=(x_test, y_test), callbacks=[es])

print('Optimised model selected with 1 convolutional layers, 1 hidden layer, optimizer=SGD, batch size=128')
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
#Use Pandas to read csv file

p = os.getcwd()
print(p)

dirname = os.path.dirname(p)
csvtest = os.path.join(p, 'test/cartoon_set_test/labels.csv')
dirname = os.path.dirname(p)
imgtest = os.path.join(p, 'test/cartoon_set_test/img/')

dataset = pd.read_csv(csvfile, sep='\\t', engine='python')

dataset.head()

test_dataset = pd.read_csv(csvtest, sep='\\t', engine='python')

test_dataset.head()

test_dataset_image = []
for i in tqdm(range(test_dataset.shape[0])):
    img = image.load_img(imgtest+test_dataset['file_name'][i], target_size=(100,100))
    img = image.img_to_array(img)
    img = img/255

    test_dataset_image.append(img)

    # Label the data and one-hot encoding
X_TEST = np.array(test_dataset_image)
Y_TEST = test_dataset['eye_color']

n_classes = 5
Y_TEST = np_utils.to_categorical(Y_TEST, n_classes)

#model run on test set, dropout has been removed
# building a linear stack of layers with the sequential model
model = Sequential()
# convolutional layer
model.add(Conv2D(25, kernel_size=(3,3), strides=(1,1), padding='valid', activation='relu', input_shape=(100,100,3)))
model.add(MaxPool2D(pool_size=(1,1)))
# flatten output of conv
model.add(Flatten())
# hidden layer
model.add(Dense(100, activation='relu'))
model.add(BatchNormalization())
# output layer
model.add(Dense(5, activation='softmax'))

# compiling the sequential model
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='SGD')
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)

# training the model for 100 epochs
model.fit(x_train, y_train, batch_size=128, epochs=100, validation_data=(x_test, y_test), callbacks=[es])

