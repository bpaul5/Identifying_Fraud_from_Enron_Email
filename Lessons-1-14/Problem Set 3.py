### Lesson 3 Decision Trees 

from sklearn import tree 
from sklearn.metrics import accuracy_score

makeTerrainData() # import from problem set 1 
prettyPicture() # import from problem set 1
output_image() # import from problem set 1

def classify(features_train, labels_train):   
	clf = tree.DecisionTreeClassifier()
	clf = clf.fit(features_train, labels_train)
	return clf 

features_train, labels_train, features_test, labels_test = makeTerrainData()
grade_fast = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==0]
bumpy_fast = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==0]
grade_slow = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==1]
bumpy_slow = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==1]
clf = classify(features_train, labels_train)
prettyPicture(clf, features_test, labels_test)
output_image("test.png", "png", open("test.png", "rb").read())

pred = clf.predict(features_test)
accuracy_score(pred, labels_test)

# making changes to the classifier 
clf = tree.DecisionTreeClassifier(min_samples_split = 2) # will split every tree if more than 2 (more complex)
clf = clf.fit(features_train, labels_train)
pred = clf.predict(features_test)
accuracy_score(pred, labels_test)

# entropy 
(sum) -Pi * log2 * Pi

import scipy.stats 
scipy.stats.entropy([2,1], base = 2)

# information gain 
info gain = entropy(parent) - [weighted avg] * entropy(children)

### Problem Set 3 

cd /Users/bindupaul/Documents/Classes/Udacity/Udacity Data/ud120-projects/tools

import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess

features_train, features_test, labels_train, labels_test = preprocess()

clf = tree.DecisionTreeClassifier(min_samples_split = 40)
clf = clf.fit(features_train, labels_train)
pred = clf.predict(features_test)
accuracy_score(pred, labels_test)

