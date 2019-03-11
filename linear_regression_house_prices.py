import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


def main():
    base = pd.read_csv('house-prices.csv')
    x = base.iloc[:, 5:6].values
    y = base.iloc[:, 2].values

    x_training, x_test, y_training, y_test = train_test_split(
            x, y, test_size=0.30, random_state=0)
    regression = LinearRegression()
    regression.fit(x_training, y_training)
    score = regression.score(x_training, y_training)
    print('Score: ' + str(score))
    
    plt.scatter(x_training, y_training)
    plt.plot(x_training, regression.predict(x_training), color='red')
    
    prevision = regression.predict(x_test)
    result = abs(y_test - prevision)
    print('Abs value: ' + str(result))

    plt.scatter(x_test, y_test)
    plt.plot(x_test, regression.predict(x_test), color='red')
    regression.score(x_test, y_test)
    

if __name__ == '__main__':
    main()