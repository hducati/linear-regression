import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


def main():
    data = pd.read_csv('plano-saude2.csv')
    x = data.iloc[:, 0:1].values
    y = data.iloc[:, 1].values
    
    regression1 = LinearRegression()
    regression1.fit(x, y)
    score1 = regression1.score(x, y)
    print('Score one: ' + str(score1))

    value = [40]
    value = np.asarray(value)
    value = value.reshape(-1, 1)
    regression1.predict(value)
    # Plotando gráfico para melhor visualização
    plt.scatter(x, y)
    plt.plot(x, regression1.predict(x), color='red')
    plt.title('Linear Regression')
    plt.ylabel('Custo')
    plt.xlabel('Idade')
    # Regressão polinomial
    poly = PolynomialFeatures(degree=3)
    x_poly = poly.fit_transform(x)
    
    regression2 = LinearRegression()
    regression2.fit(x_poly, y)
    score2 = regression2.score(x_poly, y)
    print('Score 2: ' + str(score2)) 
    
    regression2.predict(poly.fit_transform(value))
    # Plotando gráfico de regressão polinomial
    plt.scatter(x, y)
    plt.plot(x, regression2.predict(poly.fit_transform(x)), color='red')
    plt.title('Polynomial regression')
    plt.xlabel('Idade')
    plt.ylabel('Custo')
    

if __name__ == '__main__':
    main()
