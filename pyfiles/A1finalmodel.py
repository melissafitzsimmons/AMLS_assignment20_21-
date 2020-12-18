from tqdm import tqdm
import numpy as np
import pandas as pd
import os
 
# Importing sklearn libraries
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import preprocessing
 
# Importing Keras libraries
import keras
from keras.utils import np_utils
from keras.models import Sequential
from keras.applications import VGG16
from keras.applications import imagenet_utils
from keras.callbacks import ModelCheckpoint
from keras.preprocessing import image
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.layers import Dense, Conv2D, MaxPooling2D
from keras.layers import Dropout, Flatten, GlobalAveragePooling2D
from keras.preprocessing import image
from keras.layers import merge, Input
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.inception_resnet_v2 import preprocess_input

print('libraries imported')


p = os.getcwd()
print('Current working directory:' + p)
p = os.getcwd()
print(p)

os.chdir("..")
d = os.getcwd()
print(d)

dirname = os.path.dirname(d)
csvfile = os.path.join(d, 'datasets/celeba/labels.csv')

dirname = os.path.dirname(d)
imgfile = os.path.join(d, 'datasets/celeba/img/')

print('csv and img path files joined to new directory')


#Implementing InceptionResNetV2 model
image_input = Input(shape=(218,178,3))

model = InceptionResNetV2(weights='imagenet', include_top=False, input_tensor=image_input)

#print InceptionResnet V2 model summary below
model.summary()

#use pandas to read CSV file
dataset = pd.read_csv(csvfile, sep='\\t', engine='python')

#preprocess data to input into the VGG16 model
resnet_feature_list = []
for i in tqdm(range(dataset.shape[0])):
    img = image.load_img(imgfile + dataset['img_name'][i])
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)

#conver feature list to NumPy array    
    resnet_feature = model.predict(img)
    resnet_feature_np = np.array(resnet_feature)
    resnet_feature_list.append(resnet_feature_np.flatten())


resnet_feature_list_np = np.array(resnet_feature_list)

resnet_feature_list_np.shape

#Splitting the data into 75% train and 25% test
# Label X and Y data
X = np.array(resnet_feature_list)
y = np.array(dataset['gender']+1)/2
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.25, random_state=100)

#KFold CV using 5 splits
kfold = KFold(n_splits=5, random_state=0, shuffle=True)
model_kfold = LogisticRegression(solver = 'lbfgs', max_iter=30000, C=0.01)
results_kfold = cross_val_score(model_kfold, X, y, cv=kfold)
print("Overall validation accuracy (using 5 split KFold cross validation): %.2f%%" % (results_kfold.mean()*100.0))

dirname = os.path.dirname(d)
csvtest = os.path.join(d, 'test/celeba_test/labels.csv')
dirname = os.path.dirname(d)
imgtest = os.path.join(d, 'test/celeba_test/img/')

print('csv and img tets files path files joined to new directory')

image_input = Input(shape=(218,178,3))

model = InceptionResNetV2(weights='imagenet', include_top=False, input_tensor=image_input)

model.summary()

test_dataset = pd.read_csv(csvtest, sep='\\t', engine='python')

resnet_test_feature_list = []
for i in tqdm(range(test_dataset.shape[0])):
    img = image.load_img(imgtest + test_dataset['img_name'][i])
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)

    resnet_feature = model.predict(img)
    resnet_feature_np = np.array(resnet_feature)
    resnet_test_feature_list.append(resnet_feature_np.flatten())


resnet_test_feature_list_np = np.array(resnet_test_feature_list)

resnet_test_feature_list_np.shape


# Labelling the X and Y test data
X_test = np.array(resnet_test_feature_list)
Y_test = np.array(test_dataset['gender']+1)/2

#Using the model with the optimised paramaters
model = LogisticRegression(solver = 'lbfgs', max_iter=10000, C=0.01)
model.fit(X_train, Y_train)
result = model.score(X_test, Y_test)
print("Accuracy on test set: %.2f%%" % (result*100.0))