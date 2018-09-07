import pandas as pd
import numpy as np;
import matplotlib.pyplot as plot;
import os.path;
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

my_path = os.path.abspath(os.path.dirname(__file__))
path = os.path.join(my_path, "battery-data.csv")

data = pd.read_csv(path, header=None)
X = np.array(data[data.columns[0]]).reshape(data.shape[0], 1)
Y = np.array(data[data.columns[1]]).reshape(data.shape[0], 1)

xTrain = X[20:30];
yTrain = Y[20:30];
xTest = X[:-30];
yTest= Y[:-30];

regression = linear_model.LinearRegression();
regression.fit(xTrain, yTrain);

yPred = regression.predict(xTest);

print("Coefficients: ", regression.coef_);
print("Mean Sq. Error:  %f.02" % mean_squared_error(yTest, yPred))
print("Variance: ", r2_score(yTest, yPred));

plot.scatter(X, Y, c="green");
plot.scatter(xTest, yPred, c="pink");
plot.show();