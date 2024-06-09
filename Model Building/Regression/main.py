from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import random

style.use('fivethirtyeight')

xs = np.array([1,2,3,4,5,6], dtype= np.float64)
ys = np.array([5,4,6,5,6,7], dtype= np.float64)

# plt.scatter(xs,ys)
# plt.show()

def create_dataset(how_much, variance, step_up=2, correlation=False):
    """
    Args:
        how_much (int/float):  This is how many data points that we want in the set. 
        variance (int/float): This will dictate how much each point can vary from the previous point. The more variance, the less-tight the data will be.
        step_up (int, optional): This will be how far to step on average per point. Defaults to 2.
        correlation (bool, optional):  This will be either False, pos, or neg to indicate that we want no correlation, positive correlation, or negative correlation. Defaults to False.
    """
    value = 1
    ys = []
    for i in range(how_much):
        y = value + random.randrange(-variance, variance)
        ys.append(y)
        if correlation and correlation == 'pos':
            value += step_up
        elif correlation and correlation == 'neg':
            value -= step_up
    xs = [i for i in range(len(ys))]
    
    return np.array(xs, dtype= np.float64), np.array(ys, dtype=np.float64)

def best_fit_slope_and_intercepts(xs,ys):
    m = (((mean(xs)*mean(ys)) - mean(xs*ys)) /
         ((mean(xs)**2) - mean(xs**2)))
    b = mean(ys) - m * mean(xs)
    return m, b


def squared_error(ys_org, ys_line):
    return sum((ys_line - ys_org) **2 )  

def coeff_of_determination(ys_org, ys_line):
    y_mean_line = [mean(ys_org) for y in ys_org]
    print(f"Y mean line - {y_mean_line}")
    squared_error_regr = squared_error(ys_org=ys_org, ys_line=ys_line)
    squared_error_regr_y_mean = squared_error(ys_org=ys_org, ys_line=y_mean_line)
    return 1 - (squared_error_regr / squared_error_regr_y_mean)

xs , ys = create_dataset(40, 80, 2, correlation='pos')

m,b = best_fit_slope_and_intercepts(xs,ys)

regression_line = [(m*x)+b for x in xs] # Values for regression line y-coordinates 
print(f"mean:{m}, y-intercept(b):{b} \nregression Line: {regression_line}")

predict_x = 8
predict_y = (m*predict_x) + b

r_sq = coeff_of_determination(ys, regression_line)
print(r_sq)

plt.scatter(xs,ys)
plt.scatter(predict_x,predict_y,color='g',s=100)
plt.plot(xs,regression_line)
plt.show()


