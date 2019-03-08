import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from yellowbrick.regressor import ResidualsPlot
from sklearn.linear_model import LinearRegression


def main():
    data = pd.read_csv('plano-saude.csv')
    # .values transform to a numpy array
    x = data.iloc[:, 0].values
    y = data.iloc[:, 1].values
    corr_coef = np.corrcoef(x, y)
    # algoritmos no scikit learn necessitam estar no formato de matriz
    x = x.reshape(-1, 1)
    
    regression = LinearRegression()
    # realizando o treinamento
    regression.fit(x, y)
    # b0
    regression.intercept_
    # b1
    regression.coef_
    
    plt.scatter(x, y)
    plt.plot(x, regression.predict(x), color='red')
    plt.title('Regressão linear simples')
    plt.xlabel('Idade')
    plt.ylabel('Custo')
    
    value = [40]
    value = np.asarray(value)
    value = value.reshape(-1, 1)
    prevision1 = regression.predict(value)
    # y = b0 + b1 * x1
    prevision2 = regression.intercept_ + regression.coef_ * value
    # verificando a pontuacao do algoritmo de regressão
    score = regression.score(x, y)
    # plotando um grafico para melhor visualizacao dos dados.
    visualizer = ResidualsPlot(regression)
    visualizer.fit(x, y)
    # Train R² é a mesma coisa que regression.score
    visualizer.poof()
