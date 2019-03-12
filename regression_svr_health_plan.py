import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler


def main():
    data = pd.read_csv('plano-saude.csv')
    x = data.iloc[:, 0:1].values
    y = data.iloc[:, 1:2].values
    
    linear_regression = SVR(kernel='linear')
    linear_regression.fit(x, y)
    linear_regression.score(x, y)
    plt.scatter(x, y)
    plt.plot(x, linear_regression.predict(x), color='red')
    
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    x = scaler_x.fit_transform(x)
    y = scaler_y.fit_transform(y)
    
    rbf_regression = SVR()
    rbf_regression.fit(x, y)
    rbf_regression.score(x, y)
    plt.scatter(x, y)
    plt.plot(x, linear_regression.predict(x), color='red')
    
    value = np.asarray(40)
    value = value.reshape(-1, 1)
    predict1 = scaler_y.inverse_transform(
            rbf_regression.predict(scaler_x.transform(value)))
    print(str(predict1))


if __name__ == '__main__':
    main()