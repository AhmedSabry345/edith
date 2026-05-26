import numpy as np

class LinearRegression():
    def __init__(self, learning_rate = 0.01, ephocs = 1000):
        self.learning_rate = learning_rate
        self.ephocs = ephocs
        self.w = 0
        self.b = 0

    def predict(self, x):

        # next line is to fix the problem of not having a prediction
        x = np.array(x)
        prediction = self.w * x + self.b
        return prediction
    

    def fit(self, x, y):
        len_x = len(x)
        x = np.array(x)
        y = np.array(y)
        for blah in range(self.ephocs):
            y_predict = self.predict(x)

            error = y_predict - y

            dw = (1 / len_x) * np.dot(x, error)
            db = (1 / len_x) * np.sum(error)

            self.w = self.w - self.learning_rate * dw
            self.b = self.b - self.learning_rate * db



class evaluation_matricies():
    def __init__(self):    
        pass


    def MSE (y_pred, y):
        
        Mse = np.mean((y_pred - y)**2)

        return Mse
    
    def MAE(y_pred, y):
        Mae = np.mean(abs(y_pred - y))

        return Mae

    def r2score(y_pred, y):

        nom = np.sum((y_pred - y)**2)
        dom = np.sum((y - (np.mean(y)))**2)

        r2score = 1 - (nom / dom)

        return r2score


class MultiLinearRegression():
    def __init__ (self, learning_rate = 0.01, ephocs = 1000):
        self.w = None
        self.learning_rate = learning_rate
        self.ephocs = ephocs
        self.b = 0


    def predict(self, x):

        prediction = np.dot(x, self.w) + self.b
        return prediction



    def fit(self, x, y):

        rows, cols = x.shape
        self.w = np.zeros(cols)

        for _ in range(self.ephocs):
            y_pred = self.predict(x)

            error = y_pred - y

            dw = (1 / rows) * np.dot(x.T, error)
            db = (1 / rows) * np.sum(error)

            self.w -= self.learning_rate * dw
            self.b -= self.learning_rate * db

