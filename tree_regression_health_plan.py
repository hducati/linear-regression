import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.tree import DecisionTreeRegressor


def main():
    data = pd.read_csv('plano-saude2.csv')
    x = data.iloc[:, 0:1].values
    y = data.iloc[:, 1].values
    
    regression = DecisionTreeRegressor()
    regression.fit(x, y)
    score = regression.score(x, y)
    print(str(score))
   
    plt.scatter(x, y)
    plt.plot(x, regression.predict(x), color='red')
    plt.title('Tree regression')
    plt.xlabel('Idade')
    plt.ylabel('Custo')
    
    x_test = np.arange(min(x), max(x), 0.1)
    x_test = x_test.reshape(-1, 1)
    plt.scatter(x, y)
    plt.plot(x_test, regression.predict(x_test), color='red')
    plt.title('Tree regression')
    plt.xlabel('Idade')
    plt.ylabel('Custo')
    