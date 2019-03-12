import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error


def main():
    data = pd.read_csv('house-prices.csv')
    x = data.iloc[:, 3:19].values
    y = data.iloc[:, 2].values
    
    x_training, x_test, y_training, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
    regression = DecisionTreeRegressor()
    regression.fit(x_training, y_training)
    score = regression.score(x_training, y_training)
    print(str(score))
    
    prevision = regression.predict(x_test)
    mae = mean_absolute_error(y_test, prevision)
    print(str(mae))
    