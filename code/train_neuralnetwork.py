import pandas as pd
import numpy
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split

import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"


#### UNBALANCED DATASET ### 

# Set seed
numpy.random.seed(7)

# Loading data into a pandas dataframe
dataset = pd.read_csv("data/train.csv")
dataset = dataset.drop("ID_code", axis = 1)

# Split train / test
train, test = train_test_split(dataset, test_size=0.2)

# Splitting between features and target
X_train = train.drop("target", axis = 1).values
y_train = train["target"].values

X_test = test.drop("target", axis = 1).values
y_test = test["target"].values



# Neural Network
n_cols = len(train.columns) - 1 #removes the target column
model = Sequential()

# NN Structure
model.add(Dense(100, activation = "relu", input_shape = (n_cols,), name = "dense1"))
model.add(Dense(100, activation = "relu", name = "dense2"))
model.add(Dense(100, activation = "relu", name = "dense3"))
model.add(Dense(1, activation = "sigmoid", name = "dense4"))

# NN Compiler
model.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])

# NN Fit
history = model.fit(X_train, y_train, epochs = 20, validation_data = (X_test, y_test))
