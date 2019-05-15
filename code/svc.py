import pandas as pd
import numpy as np
from sklearn import svm
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix

#### BALANCED DATASET ### 

# Set seed
numpy.random.seed(7)

# Loading data into a pandas dataframe
train_dataset = pd.read_csv("data/train.csv")
train_dataset = train_dataset.drop("ID_code", axis = 1)

# Split train / test
train, test = train_test_split(train_dataset, test_size=0.2)

# Separate majority and minority classes
traindf_majority = train[train.target==0]
traindf_minority = train[train.target==1]

# Upsample minority class
df_minority_upsampled = resample(traindf_minority, 
                                 replace=True,     # sample with replacement
                                 n_samples=len(traindf_majority),    # to match majority class
                                 random_state=123) # reproducible results

# Combine majority class with upsampled minority class
df_upsampled = pd.concat([traindf_majority, df_minority_upsampled])




# Splitting between features and target
X_train = df_upsampled.drop("target", axis = 1).values
y_train = df_upsampled["target"].values

X_test = test.drop("target", axis = 1).values
y_test = test["target"].values

# Logistic Regression
print("Computing SVC")