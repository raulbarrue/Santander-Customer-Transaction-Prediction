import pandas as pd
import numpy
from sklearn.utils import resample
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
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
print("Computing Logistic Regression")

lr = LogisticRegression(solver = "saga", n_jobs=12)
lr.fit(X_train, y_train)

## MAKE KAGGLE PREDICTIONS & SAVE RESULTS
load_df = pd.read_csv("data/test.csv")
test_data = load_df.drop("ID_code", axis = 1)
predictions = lr.predict(test_data)
kaggle_results = pd.DataFrame(data = predictions, index = load_df["ID_code"], columns = ["target"])
kaggle_results.to_csv("results/logistic_regression_submission.csv") 



#score = cross_val_score(lr, X_train, y_train, cv = 10, scoring = "accuracy")
#conf_matrix = confusion_matrix(y_train, y_test)


