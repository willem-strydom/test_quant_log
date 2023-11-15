
from sklearn.preprocessing import MinMaxScaler
from experiment import experiment
import numpy as np
scaler = MinMaxScaler(feature_range=(-1, 1))
"""
#loading and sorting the data
diabetes_data = pd.read_csv("diabetes.csv").to_numpy()
diabetes_x = diabetes_data[:,:-1]
diabetes_y = diabetes_data[:,-1]
diabetes_y = np.where(diabetes_y == 0,-1, diabetes_y)
# avoid overflow error

diabetes_x = scaler.fit_transform(diabetes_x)
bias = np.ones((diabetes_x.shape[0],1))
diabetes_x = np.hstack((bias,diabetes_x))

#loading sonar data
sonar_data = pd.read_csv("sonar.csv").to_numpy()
sonar_x = sonar_data[:,:-1]
sonar_x = scaler.fit_transform(sonar_x)
#add intercept
bias = np.ones((sonar_data.shape[0],1))
sonar_x = np.hstack((bias, sonar_x)).astype(float)
sonar_y = sonar_data[:,-1]
#convert labels to +1 -1
sonar_y = np.where(sonar_y == "M",1,-1)
"""
from gen_data import gen_data
X,y = gen_data()
X = scaler.fit_transform(X)

bias = np.ones((X.shape[1],1)).T
X = np.vstack((bias, X)).astype(float)
X = X.T
y = y.T
bins = [1,2,3,4]
normal_iters, quant_iters,  w_quant, w = experiment(X,y, [3], [11,12,13])