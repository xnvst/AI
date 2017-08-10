# Load libraries
import pandas as pd
import numpy as np

from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier



###################################################
#LogisticRegression
#LinearDiscriminantAnalysis
#KNeighborsClassifier
#DecisionTreeClassifier
#GaussianNB
#SVC
###################################################

print ""
print "================================================================"
print ""
print "Machine Learning using \
LogisticRegression,\
LinearDiscriminantAnalysis,\
KNeighborsClassifier,\
DecisionTreeClassifier,\
GaussianNB,\
SVC:"
print ""

# Load dataset
#url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
url = "iris.data"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pd.read_csv(url, names=names)

# shape
#print(dataset.shape)

# head
#print(dataset.head(20))

# descriptions
print "--------data descriptions--------"
print(dataset.describe())

# class distribution
print "--------data class distribution--------"
print(dataset.groupby('class').size())

# box and whisker plots
dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
#plt.show()

# histograms
dataset.hist()
#plt.show()

# scatter plot matrix
scatter_matrix(dataset)
#plt.show()

# Split-out validation dataset
array = dataset.values
X = array[:,0:4]
Y = array[:,4]
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)

# Test options and evaluation metric
seed = 7
scoring = 'accuracy'

# Spot Check Algorithms
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))

print "--------different model accuray evaluation--------"
# evaluate each model in turn
results = []
names = []
for name, model in models:
	kfold = model_selection.KFold(n_splits=10, random_state=seed)
	cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	model.fit(X_train, Y_train)
	predictions = model.predict(X_validation)
	msg = "%s: %f (%f), accuracy score: %f" % (name, cv_results.mean(), cv_results.std(), accuracy_score(Y_validation, predictions))
	print(msg)

# Compare Algorithms
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
#plt.show()

# Make predictions on validation dataset
knn = KNeighborsClassifier()
knn.fit(X_train, Y_train)
predictions = knn.predict(X_validation)
print "--------KNN accuracy score--------"
print(accuracy_score(Y_validation, predictions))
print "--------KNN confusion matrix--------"
print(confusion_matrix(Y_validation, predictions))
print "--------KNN classification report--------"
print(classification_report(Y_validation, predictions))



###################################################
#Random Forest
###################################################

print ""
print "================================================================"
print ""
print "Machine Learning using Random Forest:"
print ""

# Load dataset
#url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
url = "iris.data"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pd.read_csv(url, names=names)

# Create a dataframe with the four feature variables
df = pd.DataFrame(dataset, columns=names)

# View the top 5 rows
#print df.head()

# Add a new column with the species names, this is what we are going to try to predict
target = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
       2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
       2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
target_names = ['setosa', 'versicolor', 'virginica']
df['species'] = pd.Categorical.from_codes(target, target_names)

# View the top 5 rows
#print df.head()

# Create a new column that for each row, generates a random number between 0 and 1, and
# if that value is less than or equal to .75, then sets the value of that cell as True
# and false otherwise. This is a quick and dirty way of randomly assigning some rows to
# be used as the training data and some as the test data.
#df['is_train'] = np.random.uniform(0, 1, len(df)) <= .75
df['is_train'] = np.random.uniform(0, 1, len(df)) <= .75

# View the top 5 rows
#print df.head()

# Create two new dataframes, one with the training rows, one with the test rows
train, test = df[df['is_train']==True], df[df['is_train']==False]

# Show the number of observations for the test and training dataframes
#print "----------------"
#print('Number of observations in the training data:', len(train))
#print('Number of observations in the test data:',len(test))

# Create a list of the feature column's names
features = df.columns[:4]
#print features

# train['species'] contains the actual species names. Before we can use it,
# we need to convert each species name into a digit. So, in this case there
# are three species, which have been coded as 0, 1, or 2.
y_train = pd.factorize(train['species'])[0]
#print y_train

# Create a random forest classifier. By convention, clf means 'classifier'
clf = RandomForestClassifier(n_jobs=2)
#print clf

# Train the classifier to take the training features and learn how they relate
# to the training y (the species)
clf.fit(train[features], y_train)

# Apply the classifier we trained to the test data (which, remember, it has never seen before)
clf.predict(test[features])

# View the predicted probabilities of the first 10 observations
#print clf.predict_proba(test[features])[0:10]
#print clf.predict(test[features])

# Create actual english names for the plants for each predicted plant class
#preds = target_names[clf.predict(test[features])]
preds = pd.Categorical.from_codes(clf.predict(test[features]), target_names)
# View the PREDICTED species for the first five observations
#print preds[0:5]

# View the ACTUAL species for the first five observations
#print test['species'].head()

# Create confusion matrix
#print pd.crosstab(test['species'], preds, rownames=['Actual Species'], colnames=['Predicted Species'])

# View a list of the features and their importance scores
print "--------list of the features and their importance scores--------"
print list(zip(train[features], clf.feature_importances_))

print "--------RF accuracy score--------"
print(accuracy_score(test['species'], preds))
print "--------RF confusion matrix--------"
print(confusion_matrix(test['species'], preds))
print "--------RF classification report--------"
print(classification_report(test['species'], preds))

print "--------different model accuray evaluation--------"
# evaluate each model in turn
results = []
names = []
for name, model in models:
	kfold = model_selection.KFold(n_splits=10, random_state=seed)
	cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	model.fit(X_train, Y_train)
	predictions = model.predict(X_validation)
	#msg = "%s: %f (%f), accuracy score: %f" % (name, cv_results.mean(), cv_results.std(), accuracy_score(Y_validation, predictions))
	msg = "%s accuracy score: %f" % (name, accuracy_score(Y_validation, predictions))
	print(msg)
msg = "RF accuracy score: %f" % (accuracy_score(Y_validation, predictions))
#msg = "RF accuracy score: 0.976333"
print(msg)

