### feature scaling 

xprime = (x - xmin) / (xmax - xmin)

# min max scaler 
from sklearn.preprocessing import MinMaxScaler
import numpy 

# numpy array needs floats 
# array hold min, suggested value, max 
weights = numpy.array([[115.], [140.], [175.]])
scaler = MinMaxScaler()
rescaled_weight = scaler.fit_transform(weights)

salary = numpy.array([[477.], [200000.], [1111258.]])
scaler = MinMaxScaler()
salary2 = scaler.fit_transform(salary)

stock = numpy.array([[3285.], [1000000.], [34348384.]])
scaler = MinMaxScaler()
stock2 = scaler.fit_transform(stock)