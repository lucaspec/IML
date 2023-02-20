import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize
from sklearn.linear_model import LinearRegression





def train_subtask3_MLP(X, y):

  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=40)
  regr = MLPRegressor(random_state=2, max_iter=500, activation='relu', hidden_layer_sizes=(1000,), early_stopping=True)
  regr.fit(X_train, y_train)

  y_predict = regr.predict(X_test)
  print(r2_score(y_test, y_predict))



def train_subtask3_SVM(X, y):

  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=40)
  regr = svm.SVR(C=1.0, epsilon=0.01 )
  regr.fit(X_train, y_train)

  y_predict = regr.predict(X_test)
  print(r2_score(y_test, y_predict))


def train_subtask3_LR(X, y):

  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=68)
  regr = LinearRegression()
  regr.fit(X_train, y_train)

  y_predict = regr.predict(X_test)
  print(r2_score(y_test, y_predict))




# access data
y_data = np.genfromtxt('train_labels.csv', delimiter=',')

y_r_1 = y_data[1:, 12]
y_r_2 = y_data[1:, 13]
y_r_3 = y_data[1:, 14]
y_r_4 = y_data[1:, 15]


#only use data which is also an output of the regression
X_data = pd.read_csv('train_features.csv', delimiter=',')
X = X_data[['RRate', 'ABPm', 'SpO2', 'Heartrate']]
age = X_data[['Age']]
age = age.to_numpy()
X = X.to_numpy()

age = np.reshape(age, (np.shape(age)[0] // 12, np.shape(age)[1] * 12))[:,1]

X = np.reshape(X, (np.shape(X)[0] // 12, np.shape(X)[1] * 12))
Xlog = np.log(X)
X2 = np.square(X)
Xsqrt = np.sqrt(X)




# number of nans, means and age as features
feature1 = np.array([])
feature2 = np.array([])
feature3 = np.array([])
feature4 = np.array([])
feature5 = np.array([])
feature6 = np.array([])
feature7 = np.array([])
feature8 = np.array([])

# array with indices of patients wished to delete
delete_patients = np.array([])

for i in range(np.shape(X)[0]):
    feature5 = np.append(feature5, sum(np.isnan(X[i, 0::4])))
    feature6 = np.append(feature6, sum(np.isnan(X[i, 1::4])))
    feature7 = np.append(feature7, sum(np.isnan(X[i, 2::4])))
    feature8 = np.append(feature8, sum(np.isnan(X[i, 3::4])))
    if sum(np.isnan(X[i, 0::4])) > 11 or sum(np.isnan(X[i, 1::4])) > 11 or sum(np.isnan(X[i, 2::4])) > 11 or sum(np.isnan(X[i, 3::4])) > 11: # delete condition
      delete_patients = np.append(delete_patients, i) 


# replace all nans by median
col_median = np.nanmedian(X, axis = 0)
inds = np.where(np.isnan(X))
X[inds] = np.take(col_median, inds[1])

for i in range(np.shape(X)[0]):
    feature1 = np.append(feature1, np.nanmean(X[i, 0::4]))
    feature2 = np.append(feature2, np.nanmean(X[i, 1::4]))
    feature3 = np.append(feature3, np.nanmean(X[i, 2::4]))
    feature4 = np.append(feature4, np.nanmean(X[i, 3::4]))

X = np.c_[X, feature1]
X = np.c_[X, feature2]
X = np.c_[X, feature3]
X = np.c_[X, feature4]
X = np.c_[X, feature5]
X = np.c_[X, feature6]
X = np.c_[X, feature7]
X = np.c_[X, feature8]
X = np.c_[X, age]

print(X.shape)




# standardise data
X = (X - np.mean(X, axis = 0))/np.std(X, axis = 0)
y_r_1 = (y_r_1 - np.mean(y_r_1, axis = 0))/np.std(y_r_1, axis = 0)
y_r_2 = (y_r_2 - np.mean(y_r_2, axis = 0))/np.std(y_r_2, axis = 0)
y_r_3 = (y_r_3 - np.mean(y_r_3, axis = 0))/np.std(y_r_3, axis = 0)
y_r_4 = (y_r_4 - np.mean(y_r_4, axis = 0))/np.std(y_r_4, axis = 0)

print(delete_patients)

# delete patients
X = np.delete(X, np.int_(delete_patients), axis = 0)
y_r_1 = np.delete(y_r_1, np.int_(delete_patients), axis = 0)
y_r_2 = np.delete(y_r_2, np.int_(delete_patients), axis = 0)
y_r_3 = np.delete(y_r_3, np.int_(delete_patients), axis = 0)
y_r_4 = np.delete(y_r_4, np.int_(delete_patients), axis = 0)
print(X.shape)

print('start training')

#train_subtask3_SVM(X, y_r_1)
#train_subtask3_SVM(X, y_r_2)
#train_subtask3_SVM(X, y_r_3)
#train_subtask3_SVM(X, y_r_4)


#train_subtask3_MLP(X, y_r_1)
#train_subtask3_MLP(X, y_r_2)
#train_subtask3_MLP(X, y_r_3)
#train_subtask3_MLP(X, y_r_4)

train_subtask3_LR(X, y_r_1)
train_subtask3_LR(X, y_r_2)
train_subtask3_LR(X, y_r_3)
train_subtask3_LR(X, y_r_4)