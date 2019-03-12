import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error


def main():
    data = pd.read_csv('house-prices.csv')
    x = data.iloc[:, 3:19].values
    y = data.iloc[:, 2].values
    score_list = []
    predict_list = []
    
    for i in range(0, 31):
        x_training, x_test, y_training, y_test = train_test_split(x, y, test_size=0.3, random_state=i)
        regression = RandomForestRegressor(n_estimators=10)
        regression.fit(x_training, y_training)
        score = regression.score(x_training, y_training)
        score_list.append(score)
        print(str(score))
        
        predict = regression.predict(x_test)
        predict_list.append(predict)
        mae = mean_absolute_error(y_test, predict)
        print(str(mae))

    predict_array = np.asarray(predict_list)
    predict_array = predict_array.reshape(-1, 1)
    print(min(predict_array))
    print(max(predict_array))
    

if __name__ == '__main__':
    main()