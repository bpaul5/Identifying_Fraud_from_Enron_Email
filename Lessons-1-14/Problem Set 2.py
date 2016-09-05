### Lesson 2 SVM 

import sys
import matplotlib.pyplot as plt
import copy
import numpy as np
import pylab as pl
import random
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# import from Problem Set 1 
prettyPicture()

makeTerrainData()

features_train, labels_train, features_test, labels_test = makeTerrainData()

clf = SVC(kernel="linear")
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)

acc = accuracy_score(pred, labels_test)

# change the kernal gamma and C values to change margin line. 
def classify(features_train, labels_train):   
	clf = SVC(kernel="rbf", gamma = 1, C = 1000)
	return clf.fit(features_train, labels_train)

clf = classify(features_train, labels_train)

### draw the decision boundary with the text points overlaid
prettyPicture(clf, features_train, labels_train)
output_image("test.png", "png", open("test.png", "rb").read())

#####################################################################################Problem Set 2

cd /Users/bindupaul/Documents/Classes/Udacity/Udacity Data/ud120-projects/tools

import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess

features_train, features_test, labels_train, labels_test = preprocess()

# to change the training time I reduce the amount of data, this will decrease the accuracy
features_train = features_train[:len(features_train)/100] 
labels_train = labels_train[:len(labels_train)/100] 

# create classifier, increase "C" to increase complexity 
clf = SVC(kernel="rbf", C = 10000)
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)
accuracy_score(pred, labels_test)

# tells the time it takes for training 
t0 = time()
clf.fit(features_train, labels_train)
print "training time:", round(time()-t0,3), "s"

# tells time it takes for prediction 
t0 = time()
clf.predict(features_test)
print "training time:", round(time()-t0,3), "s"

def prediction(x):
	data = []
	for i in x:
		if i == 1:
			data.append(i)
	return len(data)


























































