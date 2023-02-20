import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

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




def train_subtask3_LR(X, y):

  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=40)
  regr = LinearRegression()
  regr.fit(X_train, y_train)


  y_predict = regr.predict(X_test)
  print(r2_score(y_test, y_predict))

  #return y_predict

y_data = np.genfromtxt('train_labels.csv', delimiter=',')

y_r_1 = y_data[1:, 12]
y_r_2 = y_data[1:, 13]
y_r_3 = y_data[1:, 14]
y_r_4 = y_data[1:, 15]

y = y_r_3


X_data = pd.read_csv('train_features.csv', delimiter=',')
X = X_data[['SpO2']]
X = X.to_numpy()
X2 = X_data.to_numpy()

X_reg = np.empty(shape=(0, 37))

for i in range(0, X2.shape[0], 12):
  X_reg = np.r_[X_reg, np.reshape(np.nanmean(X2[i:i+12,:], axis=0), (1,37))]

col_median = np.nanmedian(X_reg, axis = 0)
inds = np.where(np.isnan(X_reg))
X_reg[inds] = np.take(col_median, inds[1])

print(X_reg.shape)
print(y_r_3.shape)




X = np.reshape(X, (np.shape(X)[0] // 12, np.shape(X)[1] * 12))


for i in range(np.shape(X)[0]):
    row_median = np.nanmedian(X[i,:])
    inds = np.where(np.isnan(X[i,:]))
    X[i,inds[0]] = row_median

delete = np.array([])
for i in range(np.shape(X)[0]):
    delete = np.append(delete,np.where(np.nanvar(X[i,:])<0.1))
print(delete.size)


mean = np.nanmean(X, axis=1)
variance = np.nanvar(X, axis=1)

inds = np.where(np.isnan(mean))
mean[inds] = np.nanmean(mean)

inds = np.where(np.isnan(variance))
variance[inds] = np.nanmean(variance)

tol = np.nanvar(mean)

mean = np.reshape(mean, (mean.size,1))
variance = np.reshape(variance, (variance.size,1))
y_train = np.reshape(y,(y.size,1))

X_reg = np.c_[X_reg, mean, variance**0.5, X_reg**0.5, X_reg**2, y_r_4, y_r_2, y_r_1]

col_median = np.nanmedian(X_reg, axis = 0)
inds = np.where(np.isnan(X_reg))
X_reg[inds] = np.take(col_median, inds[1])

corr = np.corrcoef(X_reg.T, y_r_3)[:-1,-1]
keep = np.where(np.abs(corr) > 0.6)

print(keep)



out = train_subtask3_LR(X_reg[:,keep[0]],y)




#np.savetxt("reg_Heartrate.csv", np.reshape(out, (out.size,1)), delimiter=',')


