import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error


def main():
    data = pd.read_csv('house-prices.csv')
    x = data.iloc[:, 3:19].values
    y = data.iloc[:, 2:3].values
    
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    x = scaler_x.fit_transform(x)
    y = scaler_y.fit_transform(y)
    
    x_training, x_test, y_training, y_test = train_test_split(x, y,
                                                              test_size=0.3, random_state=0)
    # (qt.atributos + saida)/2
    regressor = MLPRegressor(hidden_layer_sizes=(9, 9))
    regressor.fit(x_training, y_training)
    print(str(regressor.score(x_training, y_training)))
    print(str(regressor.score(x_test, y_test)))
    predictions = regressor.predict(x_test)
    
    y_test = scaler_y.inverse_transform(y_test)
    predictions = scaler_y.inverse_transform(predictions)    
    
    mae = mean_absolute_error(y_test, predictions)
    print(str(mae))
    

if __name__ == '__main__':
    main()    