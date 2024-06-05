# importing the required packages  
import math
import quandl as qdl  
import numpy as np  
import pandas as pd  
from sklearn import preprocessing, svm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

qdl.ApiConfig.api_key = "X_sBTh-U1mjk783m-zQ4"   


df = qdl.get("WIKI/GOOGL")

df = df[['Adj. Open','Adj. High','Adj. Low','Adj. Close','Adj. Volume']]

df['HL_PCT'] =  (df['Adj. High'] - df['Adj. Close']) / df['Adj. Close'] * 100.0
df['PCT_Change'] =  (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0

df = df[['Adj. Close', 'HL_PCT', 'PCT_Change', 'Adj. Volume']]

forecast_col = 'Adj. Close'
df.fillna(-99999, inplace= True)

forecast_out = int(math.ceil(0.01*len(df)))

df['label'] = df[forecast_col].shift(-forecast_out)
df.dropna(inplace=True)


X = np.array(df.drop(['label'], axis=1)) # X -> features
y = np.array(df.label)                   # y -> label

X = preprocessing.scale(X) # normalization of data
# X = X[:-forecast_out+1]  # not required as dropped rows which do not have 'label' on line 28

y = np.array(df.label) 

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2)

clf = LinearRegression(n_jobs= -1)
clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)

svm_lr = svm.SVR()
svm_lr.fit(X_train, y_train)
accuracy_svm = svm_lr.score(X_test, y_test)

print(f"Simple_Linear_Regression:{accuracy}, Support_Vector_Regression: {accuracy_svm}")

