import numpy
import pandas as pd

import keras
from keras.models import Sequential
from keras.layers import Dense,Activation,Conv2D,MaxPool2D,Flatten,Dropout
from keras.utils import np_utils
from keras.models import load_model

from sklearn.cross_validation import train_test_split

df_train = pd.read_csv('input/train.csv')

features = ["%s%s" %("pixel",pixel_no) for pixel_no in range(0,784)]
df_train_features = df_train[features]

df_train_labels = df_train["label"]
df_train_labels_categorical = np_utils.to_categorical(df_train_labels)

X_train,X_test,y_train,y_test = train_test_split(df_train_features, df_train_labels_categorical, test_size=0.10,random_state=32)

model = Sequential()
model.add(Dense(32,activation='relu',input_dim=784))
model.add(Dense(64,activation='relu'))
model.add(Dropout(0.03))
model.add(Dense(128,activation='relu'))
model.add(Dropout(0.03))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(10, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])

model.fit(X_train.values, y_train, batch_size=128, epochs=50, verbose=1)

model.save('my_model.h5')
