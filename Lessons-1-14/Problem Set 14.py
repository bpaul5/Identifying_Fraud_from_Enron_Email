### EVALUATION METRICS

### Recall is the probablility to actually identify the predicted value
# true positive / (true positive + false NEGATIVE) 

### Precision is the probability the predicted value is correct
# true positive / (true positive + false POSITIVE) 

### You ideally want a good F1 score with low false pos/neg


"""
    Starter code for the evaluation mini-project.
    Start by copying your trained/tested POI identifier from
    that which you built in the validation mini-project.

    This is the second step toward building your POI identifier!

    Start by loading/formatting the data...
"""

import pickle
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit
from sklearn.tree import DecisionTreeClassifier
from sklearn import cross_validation
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score


data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r") )

### add more features to features_list!
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)

# test_size=0.3 means I'm using 30% of data 
X_train, X_test, y_train, y_test = cross_validation.train_test_split(
	features, labels, test_size=0.3, random_state=42)


clf = DecisionTreeClassifier(random_state=0)
clf = clf.fit(X_train, y_train)
pred = clf.predict(X_test)
acc = accuracy_score(pred, y_test)

r = recall_score(y_test, pred)

p = precision_score(y_test, pred)

