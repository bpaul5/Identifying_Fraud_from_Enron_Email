### feature selection

# pick the number of features with large r^2 and low SSE (sum of squared errors) 

from sklearn.feature_selection import SelectPercentile, f_classif 
import sklearn.linear_model.Lasso

# feature slection tool 
selector = SelectPercentile(f_classif, percentile=10)
selector.fit(features_train_transformed, labels_train)
features_train_transformed = selector.transform(features_train_transformed).toarray()
features_test_transformed = selector.transform(features_train_transformed).toarray()

# also try if you know how many features you want to keep or are important
SelectKBest 

# import Lasso method 
features, labels = GetMyData()
regression = Lasso()
regression.fit(features, labels)
pred = regression.predict(features)
print regression.coef_ # if regression is 0 then discard it 

######################################################################################### 
# MINI PROJECT 

import pickle
import numpy
numpy.random.seed(42)


### The words (features) and authors (labels), already largely processed.
### These files should have been created from the previous (Lesson 10)
### mini-project.
words_file = "../text_learning/your_word_data.pkl" 
authors_file = "../text_learning/your_email_authors.pkl"
word_data = pickle.load( open(words_file, "r"))
authors = pickle.load( open(authors_file, "r") )



### test_size is the percentage of events assigned to the test set (the
### remainder go into training)
### feature matrices changed to dense representations for compatibility with
### classifier functions in versions 0.15.2 and earlier
from sklearn import cross_validation
features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(word_data, authors, test_size=0.1, random_state=42)

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,
                             stop_words='english')
features_train = vectorizer.fit_transform(features_train)
features_test  = vectorizer.transform(features_test).toarray()


### a classic way to overfit is to use a small number
### of data points and a large number of features;
### train on only 150 events to put ourselves in this regime
features_train = features_train[:150].toarray()
labels_train   = labels_train[:150]



### your code goes here
from sklearn.metrics import accuracy_score     
from sklearn.tree import DecisionTreeClassifier                    

clf = DecisionTreeClassifier()
clf = clf.fit(features_train, labels_train)
pred = clf.predict(features_test)
accuracy = accuracy_score(pred, labels_test)
print round(accuracy, 4)

# what is the importance of most important word 
most_important = max(clf.feature_importances_) 
important = clf.feature_importances_

for i in important:
	if i > .2:
		print i
		# print location of most important word 
		print numpy.where(important == i)

names = vectorizer.get_feature_names()

# most important word 1-3 
names[33614]
names[14343]
names[21323]



















































