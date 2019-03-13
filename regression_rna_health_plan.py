import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler


def main():
    data = pd.read_csv('plano-saude.csv')
    x = data.iloc[:, 0:1].values
    y = data.iloc[:, 1:2].values
    
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    x = scaler_x.fit_transform(x)
    y = scaler_y.fit_transform(y)
    # função de ativação relu tende a dar melhores resultados em regressões
    regressor = MLPRegressor()
    regressor.fit(x, y.ravel())
    print(str(regressor.score(x, y)))
    
    plt.scatter(x, y)
    plt.plot(x, regressor.predict(x), color='red')
    plt.title('Regressão com redes neurais')
    plt.xlabel('Idade')
    plt.ylabel('Custos')
    
    value = np.asarray(40)
    value = value.reshape(-1, 1)
    predictions = scaler_y.inverse_transform(regressor.predict(scaler_x.transform(value)))  
    print(str(predictions))
    

if __name__ == '__main__':
    main()
