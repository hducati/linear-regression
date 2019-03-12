import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.svm import SVR


def main():
    """
        Quando for utilizar o algoritmo SVM é necessario fazer o escalonamento
        de ambos os dados(x e y).
    """
    data = pd.read_csv('house-prices.csv')
    x = data.iloc[:, 3:19].values
    y = data.iloc[:, 2:3].values
    
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    x = scaler_x.fit_transform(x)
    y = scaler_y.fit_transform(y)
    
    x_training, x_test, y_training, y_test = train_test_split(
            x, y, test_size=0.3, random_state=0)

    regression = SVR()
    regression.fit(x_training, y_training)
    score = regression.score(x_training, y_training) 
    print(str(score))
    predict = regression.predict(x_test)
    
    y_test = scaler_y.inverse_transform(y_test)
    predict = scaler_y.inverse_transform(predict)
    
    mae = mean_absolute_error(y_test, predict)
    print(str(mae))


if __name__ == '__main__':
    main()