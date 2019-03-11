import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error


def main():
    data = pd.read_csv('house-prices.csv')
    x = data.iloc[:, 3:19].values
    y = data.iloc[:, 2].values
    
    x_training, x_test, y_training, y_test = train_test_split(
            x, y, test_size=0.3, random_state=0)
    poly = PolynomialFeatures(degree=4)
    x_training_poly = poly.fit_transform(x_training)
    x_test_poly = poly.transform(x_test)
    print(x_test_poly)

    regression = LinearRegression()
    regression.fit(x_training_poly, y_training)
    score = regression.score(x_training_poly, y_training)
    print('Score: ' + str(score))
    prevision = regression.predict(x_test_poly)
  
    mae = mean_absolute_error(y_test, prevision)
    print(str(mae))
    
    
if __name__ == '__main__':
    main()