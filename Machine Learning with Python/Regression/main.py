# importing the required packages  
import math, datetime
import pickle
import quandl as qdl  
import numpy as np  
import pandas as pd  
from sklearn import preprocessing, svm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib import style

style.use('ggplot')

qdl.ApiConfig.api_key = "X_sBTh-U1mjk783m-zQ4"   


df = qdl.get("WIKI/GOOGL")

df = df[['Adj. Open','Adj. High','Adj. Low','Adj. Close','Adj. Volume']]

df['HL_PCT'] =  (df['Adj. High'] - df['Adj. Close']) / df['Adj. Close'] * 100.0
df['PCT_Change'] =  (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0

#           yes           no          no          no      <- affects change in price
df = df[['Adj. Close', 'HL_PCT', 'PCT_Change', 'Adj. Volume']]

forecast_col = 'Adj. Close'
df.fillna(-99999, inplace= True)

forecast_out = int(math.ceil(0.1*len(df)))

df['label'] = df[forecast_col].shift(-forecast_out) # this will shift the col in upwards direction 
""" As per 1% or 0.01 learning rate
Date       Adj. Close    HL_PCT    PCT_Change  Adj. Volume  label
2004-08-19   50.322842  3.712563    0.324968   44659000.0  *69.078238
2004-10-08  *69.078238  1.415814    -0.720825   11069500.0  90.805307
"""
print(df.tail(35))

X = np.array(df.drop(['label','Adj. Close'], axis=1)) # X -> features
X = preprocessing.scale(X) # normalization of data
X_lately = X[-forecast_out:] # only consist of forecast_out elements, last 10% of the data
X = X[:-forecast_out] # will exclude forecast_out, 1st 90% of the data 


df.dropna(inplace=True)
y = np.array(df.label) # y(axis) -> label

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2)

clf = LinearRegression(n_jobs= -1)
clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)

# with open('Linear_regression.pickle','wb') as f:
#     pickle.dump(clf,f)

# pickle_in = open('Linear_regression.pickle','rb')
# clf = pickle.loads(pickle_in)

forecast_set = clf.predict(X_lately)

print(f"Simple_Linear_Regression:{accuracy}, Forecast_set:{forecast_set, forecast_out}")

df['forecast'] = np.nan
last_date = df.iloc[-1].name
# print(last_date.timestamp() )
# last_unix = last_date.timestamp()
one_day = 86400
# next_unix = last_date + one_day
next_unix = last_date.timestamp() + 86400

for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)] + [i]

print(df.tail(35))

#plot closing price for know && we predicted values for next days in forecast_set
df['Adj. Close'].plot()
df['forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()