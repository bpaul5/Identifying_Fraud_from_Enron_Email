### Enron Project

import os
import sys
import pickle
import pandas as pd
import numpy as np
from statistics import mean
sys.path.append("/Users/bindupaul/Documents/Classes/Udacity/Udacity Data/ud120-projects/tools/")
sys.path.append("/Users/bindupaul/Documents/Classes/Udacity/Udacity Data/ud120-projects/final_project/")
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score 
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier  
from sklearn.cluster import KMeans
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import KFold
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn import cross_validation
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn import grid_search


######################## Task 1: Select what features you'll use.

cd Documents/Classes/Udacity/Udacity Data/ud120-projects/final_project

### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
# removed: 'email_address'
features_list = ['poi','to_messages', 'deferral_payments', 'expenses', 
                 'deferred_income', 'long_term_incentive', 
                 'restricted_stock_deferred', 'shared_receipt_with_poi', 
                 'from_messages', 'other', 
                 'bonus', 'total_stock_value', 'from_poi_to_this_person', 
                 'from_this_person_to_poi', 'restricted_stock', 'salary', 
                 'total_payments', 'exercised_stock_options', 'loan_advances',
                 'director_fees']

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

######################## Task 2: Remove outliers

# found TOTAL as an outlier and removed Key:Value 
data_dict.pop("TOTAL", 0)

# creating columns for dataframe 
name_list = []
for k, v in data_dict.iteritems():
    name_list.append(k)
    
# this person has all values as 'NaN' and is removed 
name_list.remove('LOCKHART EUGENE E')

# creating data for dataframe 
df = featureFormat(data_dict, features_list, sort_keys=True)
features_pd = pd.DataFrame(data = df, index = name_list, 
                           columns = features_list)

# used to find all NaN values for each feature
def getMissingFeatures():
    data = {}
    for a in features_list:
        counter = 0
        for k, v in data_dict.iteritems():
            for i, j in v.iteritems():
                if i == a:
                    if j == 'NaN':
                        counter += 1
        data[a] = counter
        
    return data 

# obtain total missing values for each person
def getMissingPerson():
    data = {}
    for k, v in data_dict.iteritems():
        counter = 0
        for i, j in v.iteritems():
            if j == 'NaN':
                counter += 1
                data[k] = counter
    return data
    
# change feature_1 to graph that specific feature
feature_1 = 'other'
poi  = "poi"
all_features = [poi, feature_1]
data = featureFormat(data_dict, all_features )
poi, finance_features = targetFeatureSplit( data )

# compare a specific feature and obtain the position/len of the value
# find person using name_list[len]
def plotByRange():
    for f1, f2 in zip(range(0, len(finance_features)), finance_features):
        plt.scatter( f1, f2 )
    plt.show()

# find person with corresponding len above, use same feature_1 
def get_person(len):
    value = finance_features[len]
    for k, v in data_dict.iteritems():
        if v[feature_1] == value:
            return k

# compare features by poi 
def plotByPOI():
    for f1, f2 in zip(finance_features, poi):
        plt.scatter( f1, f2 )
    plt.show()

######################## Task 3: Create new feature(s)

# simple function to obtain percent 
def percent(a, b):
    num = float(a) / float(b)
    percent = num * 100 
    return percent 

# function that creates a new feature 
def newFunction():
    for k, v in data_dict.iteritems():
        if v['from_this_person_to_poi'] == 'NaN' or v['to_messages'] == 'NaN':
            v['percent_messages_to_poi'] = 'NaN'
        else: 
            # calculate the percent of messages sent to poi
            # from all to_messages
            num = percent(v['from_this_person_to_poi'], v['to_messages'])
            v['percent_messages_to_poi'] = num
    return data_dict

### Store to my_dataset for easy export below.
my_dataset = newFunction()

# this list contains newly created feature 
features_list = ['poi','to_messages', 'deferral_payments', 'expenses', 
                 'deferred_income', 'long_term_incentive', 
                 'restricted_stock_deferred', 'shared_receipt_with_poi', 
                 'from_messages', 'other', 
                 'bonus', 'total_stock_value', 'from_poi_to_this_person', 
                 'from_this_person_to_poi', 'restricted_stock', 'salary', 
                 'total_payments', 'exercised_stock_options', 'loan_advances',
                 'director_fees', 'percent_messages_to_poi']

# splitting the data into features and labels data, including new feature
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

# creating a function that will output recall and precision for all 
# values of k for SelectKBest method
def multiStep():
    recall = []
    precision = []

    # creating for loop to iterate through all values consistent with number  
    # of features in dataset 
    for i in range(1,21):

        selection = SelectKBest(k=i)
        
        # need to reimport data for each iteration since transformation was made
        data = featureFormat(my_dataset, features_list, sort_keys = True)
        labels, features = targetFeatureSplit(data)

        # transform imported data with KBest data 
        new_data = selection.fit_transform(features, labels)

        # split the data into new training and testing data 
        features_train, features_test, labels_train, labels_test = train_test_split(new_data, labels, stratify=labels, random_state=0)

        # using pipeline to scale the data and perform NiaveBayes classification 
        pipe = Pipeline(steps=[('scaler', MinMaxScaler()),
            ('NaiveBayes', GaussianNB())])

        # fit the training data 
        clf = pipe.fit(features_train, labels_train)
        pred = clf.predict(features_test)

        # finding the recall and precision and setting them as variables
        rec = round(recall_score(labels_test, pred), 3)
        prc = round(precision_score(labels_test, pred), 3)

        # appending to empty list above 
        recall.append(rec)
        precision.append(prc)
    return recall, precision

# storing the results from above as variables 
rec, prc = multiStep()

# creating a graph to show all results from multStep function 
def graph():
    # selecting x values that corresponds to the number of features
    k = range(1,21)

    plt.scatter(k, rec, color='red')
    plt.plot(k, rec, color='red', label='recall')

    plt.scatter(k, prc, color='green')
    plt.plot(k, prc, color='green', label='precision')
    
    plt.xlabel('K Best Features')
    plt.ylabel('Score')
    plt.title('Recall and Precision vs Number of Features')
    plt.legend(loc= 'upper center', fontsize='medium')
    plt.xticks(k)
    plt.show()

graph()

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

# obtaining all feature scores 
selection = SelectKBest(f_classif, k='all')
selection.fit_transform(features, labels)
scores = selection.scores_

# creating a new list to add the scores in their original order
# since the order corressponds to features_list 
original_scores = []
for i in scores:
    original_scores.append(i)

# sorting the scores for better asthetics 
scores.sort()

# this function takes the features_list and reorders it 
# respectively to the sorted order 
def feature_reorder():
    new_feat = []

    for i in scores:
        for j, k in zip(original_scores, features_list[1:]):
            if i == j:
                new_feat.append(k)
            else:
                continue
            
    return new_feat

feature_reorder()

def graph2():
    features = feature_reorder()
    plt.bar(range(1,21), scores, align='center')
    plt.xticks(range(1,21), features, rotation='vertical')
    plt.xlabel('Features')
    plt.ylabel('Feature Score')
    plt.title('Feature Score in Ascending Order') 
    plt.show()
 
graph2()

# reimport original data to select 4 best features 
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

selection = SelectKBest(k=4)
selection.fit_transform(features, labels)

my_feature_list = ['poi', 'exercised_stock_options', 'total_stock_value', 'bonus', 'salary']

features = selection.fit_transform(features, labels)

######################## Task 4: Try a varity of classifiers

# create training and testing data with SelectKBest features 
feature_train, feature_test, label_train, label_test = train_test_split( 
features, labels, stratify=labels, random_state=0)

# function to test multiple classifiers 
def pipe():
    p1 = Pipeline(steps=[('scaler', MinMaxScaler()), 
                ('NaiveBayes', GaussianNB())])

    p2 = Pipeline(steps=[('scaler', StandardScaler()),
                    ('svm', SVC(kernel='rbf', gamma=1000, C=1000))])

    p3 = Pipeline(steps=[('scaler', MinMaxScaler()),
                    ('DTree', DecisionTreeClassifier(random_state=0))]) 

    p4 = Pipeline(steps=[('scaler', MinMaxScaler()),
                    ('KNeigh', KNeighborsClassifier(n_neighbors=4))])         

    classes = [p1,p2,p3,p4]

    for i in classes:
        clf = i.fit(feature_train, label_train)
        pred = clf.predict(feature_test)
        acc = round(accuracy_score(pred, label_test), 3)
        rec = round(recall_score(label_test, pred), 3)
        prc = round(precision_score(label_test, pred), 3)
        print i.steps[1][0]
        print 'accuracy -->', acc,  
        print 'recall -->', rec,  
        print 'precision -->', prc

pipe()  


######################## Task 5: Tune your classifier to achieve better than .3 precision and recall 

### Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# using GridSearchCV() method to obtain best parameters for SVM 
# to see if it will perform better 
def GridSearch():
    labels, features = targetFeatureSplit(data)

    # creating variable to store all posssible values for SVC parameters 
    parameters = {'C': [1,10,100,1000,10000], 'gamma': [1,10,100,1000,10000]}
    #parameters = {'C': range(1,10), 'gamma': range(1,10)}  
    
    folds = 1000
    cv = StratifiedShuffleSplit(labels, folds, random_state=0) 
    # storing grid search results as classifier 
    grid = grid_search.GridSearchCV(SVC(), parameters, cv = cv, scoring='f1')

    grid.fit(features, labels)
    score = grid.best_score_
    best = grid.best_params_
    print("The best parameters are %s with a score of %0.2f"
          % (grid.best_params_, grid.best_score_))

GridSearch()

def validation():
        
    # reloading the data to avoid any errors 
    data = featureFormat(my_dataset, features_list, sort_keys = True)
    labels, features = targetFeatureSplit(data)
    selection = SelectKBest(k=4)
    selection.fit_transform(features, labels)
    features = selection.fit_transform(features, labels)
    
    # creating lists to store all values for the validation step 
    score_all = []
    precision_all = []
    recall_all = []
    
    # using SSS for validation since sample size is small for POI 
    sss = StratifiedShuffleSplit(y=labels, n_iter=100, random_state=0) 
    #kf = KFold(len(labels), 10)
    for train_indices, test_indices in sss:
       #make training and testing sets
       features_train = [features[ii] for ii in train_indices]
       features_test = [features[ii] for ii in test_indices]
       labels_train = [labels[ii] for ii in train_indices]
       labels_test = [labels[ii] for ii in test_indices]
       
       # scaling and transforming the data 
       MinMaxScaler().fit_transform(features_train, labels_train)
       clf = GaussianNB()
       #clf = SVC(kernel='rbf', gamma=1000, C=1000)
       #clf = DecisionTreeClassifier(random_state=0)
       clf.fit(features_train,labels_train)
       pred = clf.predict(features_test)
       
       # adding all the values to their respective lists 
       score_all.append(clf.score(features_test,labels_test))
       precision_all.append(precision_score(labels_test, pred))
       recall_all.append(recall_score(labels_test,pred))
       
       # calculating the average of each list 
       precision = np.average(precision_all)
       recall = np.average(recall_all)
       score = np.average(score_all)
       
    print "Precision:", precision
    print "Recall:", recall
    print "Score:", score

validation()

# storing the classifer to dump data
my_classifier = GaussianNB()      

# dumping classifier to check against tester.py 
dump_classifier_and_data(my_classifier, my_dataset, my_feature_list)

# other self created functions I used 
# used to find poi with specific feature
def getperson():
    data = []
    poi = {}
    for k, v in data_dict.iteritems():
        for i, j in v.iteritems():
            if i == 'poi':
                if j == 'NaN':
                    continue
                if j == True:
                    poi[k] = v
    
    for k, v in data_dict.iteritems():
        for i, j in v.iteritems():
            if i == 'exercised_stock_options':
                if j == 'NaN':
                    continue
                if j == 34348384:
                    data.append(k)

    return data 

# use max, min, or mean to obtain stats
def getStats(max):
    for col in features_pd:
        if col in features_list:
            value = float(max(features_pd[col]))
            
            for k,v in data_dict.iteritems():
                for i,j in v.iteritems():
                    if i == col:
                        if j == value:
                            print k, '-->', col, '-->', value 
                        if j == 'NaN' and value == 0:
                            continue

# SelectKBest method
splits = # Input an sklearn.cross_validation object here, e.g. StratifiedShuffleSplit
k = 5 # Change this to number of features to use.

# We will include all the features into variable best_features, then group by their
# occurrences.
best_features = []
for i_train, i_test in splits:
    features_train, features_test = [features[i] for i in i_train], [features[i] for i in i_test]
    labels_train, labels_test = [labels[i] for i in i_train], [labels[i] for i in i_test]

    # fit selector to training set
    selector = SelectKBest(f_classif, k = k)
    selector.fit(features_train, labels_train)

    for i in selector.get_support(indices = True):
        best_features.append(features_list[i+1]) 

# This is the part where we group by occurrences.
# At the end of this step, features_list should have k of the most occurring
# features. In other words they are features that are highly likely to have
# high scores from SelectKBest.
from collections import defaultdict
d = defaultdict(int)
for i in best_features:
    d[i] += 1
import operator
sorted_d = sorted(d.items(), key=operator.itemgetter(1))
features_list = [x[0] for x in sorted_d[-k:]]











