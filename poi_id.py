
# this will create the pickle files with the least amout of code needed. 

import os
import sys
sys.path.append(".../tools/")
sys.path.append(".../final_project/")
import pickle
from feature_format import featureFormat, targetFeatureSplit
from sklearn import cross_validation
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import MinMaxScaler
from tester import dump_classifier_and_data

with open("final_project_dataset.pkl", "r") as data_file:
	data_dict = pickle.load(data_file)

data_dict.pop("TOTAL", 0)

my_feature_list = ['poi', 'exercised_stock_options', 'total_stock_value', 'bonus', 'salary']

my_dataset = data_dict

my_classifier = GaussianNB()

dump_classifier_and_data(my_classifier, my_dataset, my_feature_list)
