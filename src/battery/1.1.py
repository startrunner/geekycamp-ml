import pandas as pd
import numpy as np;
import matplotlib.pyplot as plot;
import os.path;

my_path = os.path.abspath(os.path.dirname(__file__))
path = os.path.join(my_path, "battery-data.csv")

data = pd.read_csv(path, header=None)
X = np.array(data[data.columns[0]]).reshape(data.shape[0], 1)
Y = np.array(data[data.columns[1]]).reshape(data.shape[0], 1)

plot.scatter(X, Y);
plot.show();