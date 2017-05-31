import numpy
import pandas as pd

import keras
from keras.models import Sequential
from keras.layers import Dense,Activation,Conv2D,MaxPool2D,Flatten,Dropout
from keras.utils import np_utils
from keras.models import load_model

import numpy as np

import os
import requests

from flask import Flask
app = Flask(__name__)

import tensorflow as tf

model = load_model("my_model.h5")

in_data = np.zeros(784)
prediction = model.predict(np.array([in_data]))

del model

@app.route("/")
def hello():
    return "Hello  World!"


if __name__ == "__main__":
    app.run(debug=True,host='0.0.0.0')

