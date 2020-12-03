import linearReggression as lr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

multi = pd.read_csv('multivariateData1.csv', names=['a', 'b','c'])

x = multi.iloc[:, :1].values
y = multi.iloc[:, 2].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.2)
model = lr.LinearRegression(number_of_iterations=1000, learning_rate=.000000001)
model.fit(x_train, y_train)
y_predicted = model.predict(x_test)

print(model.EvaluatePerformance(y_test,y_predicted))
plt.scatter(x_test, y_test, color='Red')
plt.plot(x_test, y_predicted, color='orange')
plt.show()