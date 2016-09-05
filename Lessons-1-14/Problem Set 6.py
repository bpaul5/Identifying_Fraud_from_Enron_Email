### Regressions 

import matplotlib.pyplot as plt
import numpy
import random
from sklearn.linear_model import LinearRegression

from sklearn import linear_model 
reg = LinearRegression()
reg.fit(ages_train, net_worths_train)


# can predict values but it needs to be in a list even if 1 value 
reg.predict(some number)
reg.coef_ # slope 
reg.intercept_ # y-intercept 
reg.score(feature_test, label_test) # r-squared score


def ageNetWorthData():

    random.seed(42)
    numpy.random.seed(42)

    ages = []
    for ii in range(100):
        ages.append( random.randint(20,65) )
    net_worths = [ii * 6.25 + numpy.random.normal(scale=40.) for ii in ages]
### need massage list into a 2d numpy array to get it to work in LinearRegression
    ages       = numpy.reshape( numpy.array(ages), (len(ages), 1))
    net_worths = numpy.reshape( numpy.array(net_worths), (len(net_worths), 1))

    from sklearn.cross_validation import train_test_split
    ages_train, ages_test, net_worths_train, net_worths_test = train_test_split(ages, net_worths)

    return ages_train, ages_test, net_worths_train, net_worths_test


ages_train, ages_test, net_worths_train, net_worths_test = ageNetWorthData()

reg = LinearRegression()
reg.fit(ages_train, net_worths_train)
reg.predict([[27]])[0][0]
reg.coef_[0][0]
reg.intercept_[0]
reg.score(ages_train, net_worths_train)
reg.score(ages_test, net_worths_test)

#### THE BEST REGRESSION IS THE ONE THAT MINIMIZES THE SUM (ACTUAL - PREDICTED)^2
#### ACTUAL = TRAINING POINTS AND PREDICTED = I HAVE TO CALCULATE 
#### More data = more SSE (sum of squared error) but not neccesarily worse fit 
