import numpy as np


class LinearRegression:
    def __init__(self, number_of_iterations, learning_rate):
        self.number_of_iterations = number_of_iterations
        self.learning_rate = learning_rate
        self.y_predicted = 0
        self.error = 0
        self.totalfunc = 0

    def fit(self, x, y):
        self.m, self.n = x.shape
        self.theta_0 = 0
        self.thetav = np.zeros(self.n)
        self.x = x
        self.y = y
        for i in range(self.number_of_iterations):
            self.computeGradient()
        return self

    def computeGradient(self):
        self.y_predicted = self.predict(self.x)
        self.computeCost()

        d1 = - ((self.x.T).dot(self.y - self.y_predicted)) / self.m

        d0 = - np.sum(self.y - self.y_predicted) / self.m

        self.thetav = self.thetav - self.learning_rate * d1

        self.theta_0 = self.theta_0 - self.learning_rate * d0

        return self

    def computeCost(self):
        self.error = self.error + (self.y - self.y_predicted) ** 2
        self.totalfunc = self.totalfunc + (self.error * (1 / (2 * self.m)))
        return self.error

    def EvaluatePerformance(self, act, pre):
        ssr = np.sum((pre - act) ** 2)
        sst = np.sum((act - np.mean(act)) ** 2)
        r2_score = 1 - (ssr / sst)
        return r2_score

    def predict(self, x):
        return x.dot(self.thetav) + self.theta_0

