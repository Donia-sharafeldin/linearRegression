import linearReggression as lr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.model_selection import train_test_split

data = pd.read_csv('uni.csv', names=['a', 'b'])

x = data.iloc[:, :-1].values
y = data.iloc[:, 1].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.2)
model = lr.LinearRegression(number_of_iterations=1000, learning_rate=.01)
model.fit(x_train, y_train)
y_predicted = model.predict(x_test)

print(model.EvaluatePerformance(y_test,y_predicted))
plt.scatter(x_test, y_test, color='Red')
plt.plot(x_test, y_predicted, color='orange')
plt.show()

