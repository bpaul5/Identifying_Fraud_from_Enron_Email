### CROSS VALIDATION 


from sklearn import datasets
from sklearn.svm import SVC
import numpy as np 
from sklearn import cross_validation
from sklearn import datasets
from sklearn import svm 

iris = datasets.load_iris()
iris.data.shape, iris.target.shape

features = iris.data
labels = iris.target

X_train, X_test, y_train, y_test = cross_validation.train_test_split(
	features, labels, test_size=0.4, random_state=0)

clf = svm.SVC(kernel='linear', C=1).fit(X_train, y_train)
clf.score(X_test, y_test)

from sklearn import svm, grid_search, datasets 

parameters = {'kernal':('linear', 'rbf'), 'C':[1, 10]}
svr = svm.SVC()
clf = grid_search.GridSearchCV(svr, parameters)
clf.fit(iris.data, iris.target)
clf.best_params_

#########################################################################################
# MINI PROJECT


"""
    Starter code for the validation mini-project.
    The first step toward building your POI identifier!

    Start by loading/formatting the data

    After that, it's not our code anymore--it's yours!
"""

import pickle
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import cross_validation

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r") )

### first element is our labels, any added elements are predictor
### features. Keep this the same for the mini-project, but you'll
### have a different feature list when you do the final project.
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)

clf = DecisionTreeClassifier(random_state=0)
clf = clf.fit(features, labels)
pred = clf.predict(features)
acc = accuracy_score(pred, labels)

# test_size=0.3 means I'm using 30% of data 
X_train, X_test, y_train, y_test = cross_validation.train_test_split(
	features, labels, test_size=0.3, random_state=42)


clf = DecisionTreeClassifier(random_state=0)
clf = clf.fit(X_train, y_train)
pred = clf.predict(X_test)
acc = accuracy_score(pred, y_test)

























































