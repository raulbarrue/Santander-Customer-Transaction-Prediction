import pandas as pd
import numpy
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

# User outputs
plot_results = False
predict_kaggle = False
save_model = False

main_question = input("Personalise execution? Y/N")
    if main_question == "Y".lower():

    question1 = input("Plot results? Y/N")
    if question1 == "Y".lower():
        plot_results = True

    question2 = input("Predict Kaggle test? Y/N")
    if question2 == "Y".lower():
        predict_kaggle = True

    question3 = input("Save model? Y/N"):
        if question3 == "Y".lower():
            save_model = True



#### UNBALANCED DATASET ### 

# Set seed
numpy.random.seed(7)

# Loading data into a pandas dataframe
train_dataset = pd.read_csv("data/train.csv")
train_dataset = train_dataset.drop("ID_code", axis = 1)

# Split train / test
train, test = train_test_split(train_dataset, test_size=0.2)

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
history = model.fit(X_train, y_train, epochs = 20, batch_size = 128, validation_data = (X_test, y_test))

if save_model:
    model.save(model_name) 

if plot_results:
    # Plot results
    # Test model accuracy
    score, acc = model.evaluate(X_test, y_test, verbose = 2)
    print("Test accuracy:", acc)


    # Plot training & validation accuracy values
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()


# Predicting Kaggle
if predict_kaggle:
    test_dataset = pd.read_csv("data/test.csv")
    test_id = test_dataset["ID_code"]
    test_data = test_data.drop("ID_code", axis = 1)
    predictions = model.predict_classes(test_data)
    kaggle_results = pd.DataFrame(data = predictions, index = testdf["ID_code"], columns = ["ID_code", "target"])
    kaggle_results.to_csv("/results/NN_submission.csv") 


