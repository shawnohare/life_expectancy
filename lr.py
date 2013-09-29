# Use either iptyon --pylab to open iptyon with pylab support in a terminal
# or once ipython is running use %pylab or %matplotlib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model

def main():
    data = pd.read_csv('US_life_expectany.csv')
    x = data['Year']
    y = data['US_life_expectancy']

    # sklearn's linear regressor wants data as an array of arrays
    x_train = np.array([[i] for i in x])
    y_train = np.array([[i] for i in y])
    
    # We set up a matplotlib figure object and plot the WHO data
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.scatter(x,y)
    ax.set_title('WHO US Life Expectancy Data')
    ax.set_xlabel('Year')
    ax.set_ylabel('Life Expectancy at Birth (years)')

    # Now we create a linear regression object
    regr = linear_model.LinearRegression()
    regr.fit(x_train,y_train)
    
    # plotting the regression line
    ax.plot(x, regr.predict(x_train), color='green', linewidth=2, label='Regression line')
    ax.legend(loc='best')
    fig.savefig('lr_life_expectancy.svg')
    fig.savefig('lr_life_expectancy.png')
    plt.show()

if __name__ == "__main__":
    main()






