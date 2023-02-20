import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

training_data = np.genfromtxt('train.csv', delimiter=',')
y = training_data[1:,0]
X = training_data[1:,1:14]

sum = 0

for i in range(10):
    # Cross Validation
    y_train = np.delete(y, np.s_[15*i:15*(i+1)],0)
    y_test = y[15*i:15*(i+1)]
    X_train = np.delete(X, np.s_[15*i:15*(i+1)],0)
    X_test = X[15*i:15*(i+1)]

    # training
    clf = Ridge(alpha=200) # use wanted lambda
    clf.fit(X_train, y_train)

    # prediction 
    y_pred = clf.predict(X_test)

    # RMSE
    sum += mean_squared_error(y_test, y_pred)**0.5

RMSE_avg = sum/10

# copying values from terminal is faster than coding a writer
print(RMSE_avg)