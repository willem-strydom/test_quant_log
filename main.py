import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from experiment import experiment


#loading and sorting the data
diabetes_data = pd.read_csv("diabetes.csv").to_numpy()
diabetes_x = diabetes_data[:,:-1]
diabetes_y = diabetes_data[:,-1]
diabetes_y = np.where(diabetes_y == 0,-1, diabetes_y)
# avoid overflow error
scaler = MinMaxScaler(feature_range=(-1, 1))
diabetes_x = scaler.fit_transform(diabetes_x)
bias = np.ones((diabetes_x.shape[0],1))
diabetes_x = np.hstack((bias,diabetes_x))


normal_iters, quant_iters, normal_loss, quant_loss = experiment(diabetes_x,diabetes_y,10)

print(f"normal iterations: {np.mean(normal_iters)}, quantized iterations: {np.mean(quant_iters)}, "
      f"normal loss {np.mean(normal_loss)}, quantized loss {np.mean(quant_loss)}")
"""
#loading sonar data
sonar_data = pd.read_csv("sonar.csv").to_numpy()
sonar_x = sonar_data[:,:-1]
#add intercept
bias = np.ones((sonar_data.shape[0],1))
sonar_x = np.hstack((bias, sonar_x)).astype(float)
sonar_y = sonar_data[:,-1]
#convert labels to +1 -1
sonar_y = np.where(sonar_y == "M",1,-1)


normal_iters, quant_iters, normal_loss, quant_loss = experiment(sonar_x, sonar_y,10)

print(f"normal iterations: {np.mean(normal_iters)}, quantized iterations: {np.mean(quant_iters)}, "
      f"normal loss {np.mean(normal_loss)}, quantized loss {np.mean(quant_loss)}")

"""