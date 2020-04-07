#TM- Titanic Kaggle Competition

import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score

'''Testing Methods
Methods Tests:
	1. Missing Data- SciKit Learn
		a. Categorical- similar results after OH_encoder
			i. fillNA with ffill- then add boolean
			ii. fillNA with cusom value Other
		b. Num- fillNA with mean
			i. new column for OH encoder
	2. Test/ Train data
		a. Orig tested for 
		b. Tested for Cross Val
	3. Model- using XGVR- no other models considered
	4. High Freq categorical data
		Name- Not considered- although w/ public data could be used for perfect score
		Cabin- Added to model 4/2
			FILL_NA = ZEMPTY
			OH- based off first letter
		Ticket- Check to add to model 4/2
'''
#class not used
'''
class fill_Data
	def __init__(self, attrib, estim, X_train, X_valid, y_train, y_valid):
		self.attrib = attrib
		self.estim = estim
		self.X_Train = X_Train
		self.X_valid = X_valid
		self.y_train = y_train
		self.y_valid = y_valid
'''		

#read train and test files- Change PATH PRIOR OT UPLOAD"
myPath = r"C:\Users\meado\pythCode\compts\t"

testFile = "est.csv"
trainFile = "rain.csv"

test_data = pd.read_csv(myPath + testFile, index_col = "PassengerId")
train_data = pd.read_csv(myPath + trainFile, index_col = "PassengerId")

cat_features = [col for col in test_data.columns if test_data[col].dtype == 'object' and test_data[col].nunique() < 10]
num_features = [col for col in test_data.columns if test_data[col].dtype in ['float64', 'int64']]
#potential Cabin and Ticket
xtra_featurs = ['Cabin']

#print(cat_features)
#print(num_features)

X = train_data[cat_features + num_features + xtra_featurs].copy()
X_Test = test_data[cat_features + num_features + xtra_featurs].copy()
y = train_data.Survived

#This is where I split the data- not using cross validation
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)

#Functions
def score_model(X_t=X_train, X_v=X_valid, y_t=y_train, y_v=y_valid):
	model = XGBRegressor(objective ='reg:squarederror')
	model.fit(X_t, y_t)
	preds = model.predict(X_v)
	i = 0
	for pred in preds:
		if pred > .5:
			preds[i] = 1
		else:
			preds[i] = 0
		i += 1
#will return Err terms to calc model
#	return mean_absolute_error(y_v, preds)
	
	for k in range(0, len(preds)):
		print(y_v.index[k], preds[k])


missCols = [col for col in X_train.columns if X[col].isnull().any()]

#print(missCols)

fill_X_train = X_train.copy()
fill_X_valid = X_valid.copy()

for missCol in missCols:
#	if missCol in cat_features:
#		myS = "strategy = 'fil'
#	fill_X_train[missCol], fill_X_valid[missCol] = simple_fill_Missing(fill_X_train, fill_X_valid, missCol)
	fill_X_train[missCol + '_wm'] = fill_X_train[missCol].isnull()
	fill_X_valid[missCol + '_wm'] = fill_X_valid[missCol].isnull()
	if missCol in cat_features:
		fill_X_train[missCol].fillna(method='ffill', inplace=True)
		fill_X_valid[missCol].fillna(method='ffill', inplace=True)
	elif missCol in num_features:
		#print(missCol)
		fill_X_train[missCol].fillna(fill_X_train[missCol].mean(), inplace=True)
		fill_X_valid[missCol].fillna(fill_X_valid[missCol].mean(), inplace=True)
	elif missCol in xtra_featurs:
		fill_X_train[missCol].fillna('ZEMPTY', inplace=True)
		fill_X_valid[missCol].fillna('ZEMPTY', inplace=True)

#change cabin data

fill_X_train['Cabin Adj'] = [x[0] for x in fill_X_train['Cabin']]
del fill_X_train['Cabin']

#print(fill_X_train)

fill_X_valid['Cabin Adj'] = [x[0] for x in fill_X_valid['Cabin']]
del fill_X_valid['Cabin']
cat_features = cat_features + ["Cabin Adj"]

#print(cat_features)

#print(fill_X_train['Cabin Adj'])

cat_X_train = fill_X_train.copy()
cat_X_valid = fill_X_valid.copy()

#print(fill_X_train)
# Create the CatBoost encoder
'''
cat_features = ['app', 'device', 'os', 'channel']
cb_enc = ce.CatBoostEncoder(cols = cat_features, random_state = 7)
'''
OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
cat_X_train = pd.DataFrame(OH_encoder.fit_transform(fill_X_train[cat_features]))
cat_X_valid = pd.DataFrame(OH_encoder.transform(fill_X_valid[cat_features]))

# One-hot encoding removed index; put it back
cat_X_train.index = fill_X_train.index
cat_X_valid.index = fill_X_valid.index

# Remove categorical columns (will replace with one-hot encoding)
num_X_train = fill_X_train.drop(cat_features, axis=1)
num_X_valid = fill_X_valid.drop(cat_features, axis=1)

# Add one-hot encoded columns to numerical features
cat_X_trains = pd.concat([num_X_train, cat_X_train], axis=1)
cat_X_valids = pd.concat([num_X_valid, cat_X_valid], axis=1)

#print(cat_X_trains.columns)
#test below removed each attribute and print score- no major differentiation
'''
for Att in cat_X_trains.columns:
	newXtrain = cat_X_trains.drop(Att, axis=1)
	newXvalid = cat_X_valids.drop(Att, axis=1)
	myPred = score_model(newXtrain, newXvalid, y_train, y_valid)
	print(str(Att) + " Dropped: " + str(myPred))
'''
#print(cat_X_trains)
#parch returns less than ideal results
print(score_model(cat_X_trains, cat_X_valids, y_train, y_valid))