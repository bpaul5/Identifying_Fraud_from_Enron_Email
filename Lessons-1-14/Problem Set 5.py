### Lesson 5 Datasets and Questions

import pickle

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))

def enron():
	data = [] 
	for k, v in enron_data.iteritems():
		for key, value in v.iteritems():
			if key == 'poi':
				if value == 1:
					print k 
					data.append(value)
	print len(data)

enron_data['LAY KENNETH L']
enron_data['SKILLING JEFFREY K']
enron_data['FASTOW ANDREW S']

# import the following functions from feature_format.py 
feature_Format()
targetFeatureSplit()

def enron():
	data = [] 
	for k, v in enron_data.iteritems():
		for key, value in v.iteritems():
			if key == 'poi':
				if value == True:
					data.append(v)
	return data

def enron2():
	data = []
	variable = enron()
	for i in variable: 
		for k,v in i.iteritems():
			if k == 'total_payments':
				if v == 'NaN':
					data.append(v)
	return len(data)
