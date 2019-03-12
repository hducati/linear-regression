import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor


def main():
    data = pd.read_csv('plano-saude.csv')
    x = data.iloc[:, 0:1].values
    y = data.iloc[:, 1].values
    
    regressor = RandomForestRegressor(n_estimators=10)
    regressor.fit(x, y)
    score = regressor.score(x, y)
    
    x_test = np.arange(min(x), max(x), 0.1)
    x_test = x_test.reshape(-1, 1)
    plt.scatter(x, y)
    plt.plot(x_test, regressor.predict(x_test), color='red')
    plt.title('Regressão com random forest')
    plt.xlabel('Idade')
    plt.ylabel('Custo')
    
    value = np.asarray(40)
    value = value.reshape(-1, 1)
    predict = regressor.predict(value)
    print(str(predict))
    