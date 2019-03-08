from __future__ import absolute_import, division, print_function

import pathlib

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import tensorflow as tf
from tensorflow import keras
print(tf.__version__)
#load dataset from uci
dataset_path = keras.utils.get_file("iris.data", "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data")
##set column names
column_names = ['SL','SW','PL','PW',"Class"]
raw_dataset = pd.read_csv(dataset_path, names=column_names,
                      na_values = "?", comment='\t',
                      sep=",", skipinitialspace=True)
print(raw_dataset)
#set classes as integers
number={}
c=0
for a in raw_dataset.values:
    print(a[4])
    if a[4] not in number.keys():
        number[a[4]]=c
        c+=1
#separating tags from data
tags=raw_dataset['Class'].copy()
del raw_dataset['Class']
for i in tags.index:
    tags[i]=number.get(tags[i])
dataset = raw_dataset.copy()
dataset.tail()
#finally we get
print(dataset)
print(tags)
# declare model
model = keras.Sequential([
    keras.layers.Dense(6, input_shape=(4,)),
    keras.layers.Dense(4, activation=tf.nn.relu),
    keras.layers.Dense(3, activation=tf.nn.softmax)
])
#because i wanted softmax
model.compile(loss='sparse_categorical_crossentropy',
                optimizer="adam",
               metrics=['accuracy'])
model.summary()
model.fit(
dataset, tags,
epochs=1000, validation_split = 0.2, verbose=1)
##test prediction
test=np.array([[5.3,  3.8,  1.3,  0.4]])
print(test.shape)
re=model.predict(test,batch_size=None, verbose=1)
print(100*re)
print(number)
#savin model
model.save('./thismodel')
ca=keras.Sequential()
#reloading model
ca=keras.models.load_model('./thismodel')
ca.summary()

