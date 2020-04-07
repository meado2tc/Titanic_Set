#titanic- feature selection

import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score
import lightgbm as lgb
from sklearn import metrics
import category_encoders as ce
import itertools
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel

def train_model(train, valid):
	print(train.columns)
	feature_cols = train.columns.drop(['Survived'])

	dtrain = lgb.Dataset(train[feature_cols], label=train['Survived'])
	dvalid = lgb.Dataset(valid[feature_cols], label=valid['Survived'])

	param = {'num_leaves': 64, 'objective': 'binary', 'metric': 'auc', 'seed': 7}
	bst = lgb.train(param, dtrain, num_boost_round=1000, valid_sets=[dvalid], 
					early_stopping_rounds=10, verbose_eval=False)
	valid_pred = bst.predict(valid[feature_cols])
	valid_score = metrics.roc_auc_score(valid['Survived'], valid_pred)
	return valid_score

def get_data_splits(dataframe, valid_fraction=0.2):
	valid_size = int(len(dataframe) * valid_fraction)
	train = dataframe[:-valid_size * 2]
	# valid size == test size, last two sections of the data
	valid = dataframe[-valid_size * 2:]
	return train, valid

basePth = r"C:\users\meado\pythCode\compts"
testFile = "test.csv"
trainFile = "train.csv"

train_data = pd.read_csv(basePth + "\\" + trainFile, index_col = "PassengerId")
test_data = pd.read_csv(basePth + "\\" + testFile, index_col = "PassengerId")

#print(train_data.head())
#collect column types
del train_data["Name"]
del test_data["Name"]

#combine data features
cat_combs = [col for col in train_data.columns if train_data[col].dtype == 'object' and train_data[col].nunique() < 10]

for col1, col2 in itertools.combinations(cat_combs, 2):
	train_data[col1 + '_' + col2] = train_data[col1] + "_" + train_data[col2]

cat_features = [col for col in train_data.columns if train_data[col].dtype == 'object']
num_features = [col for col in train_data.columns if train_data[col].dtype in ['float64', 'int64']]
missCols = [col for col in train_data.columns if train_data[col].isnull().any()]

#call split data
train, valid = get_data_splits(train_data)

# mix Pclass and sex
#missing columns- easy way to fill
for col in missCols:
	if col in num_features:
		train[col].fillna(train[col].mean(), inplace=True)
		valid[col].fillna(valid[col].mean(), inplace=True)
	elif col in cat_features:
		train[col].fillna("ZEMPTY", inplace=True)
		valid[col].fillna("ZEMPTY", inplace=True)

#start with simple lable encoder- move to more progressive
#encoder = LabelEncoder()-83 score- vs 87 score using CatBoostEncoder

target_enc = ce.CatBoostEncoder(cols=cat_features)
target_enc.fit(train[cat_features], train['Survived'])

train = train.join(target_enc.transform(train[cat_features]).add_suffix('_label'))
valid = valid.join(target_enc.transform(valid[cat_features]).add_suffix('_label'))

for col in cat_features:
	del train[col]
	del valid[col]

#check for featrues to delete-- CURRENTLY MAKES EST WORSE
X, y = train[train.columns.drop("Survived")], train['Survived']

logistic = LogisticRegression(C=.05, penalty="l2", random_state=12).fit(X, y)
model = SelectFromModel(logistic, prefit=True)

X_new = model.transform(X)

# Get back the kept features as a DataFrame with dropped columns as all 0s
selected_features = pd.DataFrame(model.inverse_transform(X_new), 
								index=X.index,
								columns=X.columns)

# Dropped columns have values of all 0s, keep other columns 
drop_columns = selected_features.columns[selected_features.var() == 0]
train = train.drop(drop_columns, axis=1)
valid = valid.drop(drop_columns, axis=1)

#train = train[selected_columns, "Survived"]


'''base fillna encoding
for col in cat_features:
	nextTrain[col + '_label'] = encoder.fit_transform(train[col])
	nextValid[col + '_label'] = encoder.fit_transform(valid[col])
	nextTrain = nextTrain.drop(col, axis=1)
	nextValid = nextValid.drop(col, axis=1)
'''

#score Model
print(train_model(train, valid))


