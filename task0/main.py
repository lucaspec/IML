from unicodedata import decimal
import numpy as np
from numpy.linalg import inv
import csv

training_data = np.genfromtxt('train.csv', delimiter=',')
testing_data = np.genfromtxt('test.csv', delimiter=',')


y = training_data[1:10001,1]
X = training_data[1:10001,2:12]
X_t = X.transpose()

w_hat = (inv(X_t.dot(X)).dot(X_t)).dot(y)

X_test = testing_data[1:2001,1:11]
y_test = X_test.dot(w_hat)

# writing to csv file
header = ['Id', 'y']

with open('submission.csv', 'w', encoding='UTF8', newline='') as f:
    writer = csv.writer(f)

    # write the header
    writer.writerow(header)

    # write the data
    for i in range(len(y_test)):
        row = [int(testing_data[i+1,0]), np.float32(y_test[i])]
        writer.writerow(row)
