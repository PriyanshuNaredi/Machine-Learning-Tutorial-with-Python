from ucimlrepo import fetch_ucirepo 
import numpy as np
from sklearn import preprocessing, neighbors
from sklearn.model_selection import train_test_split
import pandas as pd

# Headers - Clump_thickness  Uniformity_of_cell_size  Uniformity_of_cell_shape  Marginal_adhesion  Single_epithelial_cell_size  Bare_nuclei  Bland_chromatin  Normal_nucleoli  Mitoses
df = pd.read_csv('Machine Learning with Python/KNN/breast-cancer-wisconsin.data')

df.replace('?', -99999, inplace=True)
df.drop(['id'], axis=1, inplace=True)

X = np.array(df.drop(['Class'], axis=1))
y = np.array(df['Class'])


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.21)

clf = neighbors.KNeighborsClassifier()
clf.fit(X_train, y_train)

accuracy = clf.score(X_test, y_test)
print(accuracy)


example_measures = np.array([[4,2,1,1,1,2,3,2,1],[4,1,2,2,2,1,3,1,2]])
# example_measures = example_measures.reshape(2,-1)
example_measures = example_measures.reshape(len(example_measures),-1)

prediction = clf.predict(example_measures)
print(prediction)









