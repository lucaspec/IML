import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# import handler
training_data = np.genfromtxt('train2.csv', delimiter=',')
y = training_data[1:6, 1]
X = training_data[1:6, 2:3]
print(X)
X_init = X

# build new features from given
X = np.append(X, X_init ** 2, axis=1)
X = np.append(X, np.sin(X_init), axis=1)
X = np.append(X, np.ones((X_init.shape[0], 1)), axis=1)
print(X)

# train the Regression model (maybe replace with Lasso?)
reg = LinearRegression(fit_intercept=False)
reg.fit(X, y)

# output weights
weights = reg.coef_
print(weights)

# sava weights to csv
np.savetxt('task1b_weights', weights, delimiter='\n')
