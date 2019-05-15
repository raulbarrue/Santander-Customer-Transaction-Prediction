from xgboost import XGBClassifier
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
import pandas as pd
import numpy

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

#data_dmatrix = xgb.DMatrix(data=X_train,label=y_train)

xg_reg = XGBClassifier(objective ='binary:logistic', colsample_bytree = 0.3, learning_rate = 0.1, max_depth = 5, alpha = 10, n_estimators = 10)

xg_reg.fit(X_train,y_train)

#preds = xg_reg.predict(X_test)

#rmse = numpy.sqrt(mean_squared_error(y_test, preds))

#print("RMSE: {}".format(rmse))

## MAKE KAGGLE PREDICTIONS & SAVE RESULTS
load_df = pd.read_csv("data/test.csv")
test_data = load_df.drop("ID_code", axis = 1).values
predictions = xg_reg.predict(test_data)
kaggle_results = pd.DataFrame(data = predictions, index = load_df["ID_code"], columns = ["target"])
kaggle_results.to_csv("results/xgboost_tree_submission.csv") 