import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
from sklearn.cluster import MeanShift
from sklearn import preprocessing
# from sklearn.model_selection import train_test_split
import pandas as pd
# Warnings ignore
import warnings
warnings.filterwarnings('ignore')

style.use('ggplot')

'''
Pclass Passenger Class (1 = 1st; 2 = 2nd; 3 = 3rd)
survival Survival (0 = No; 1 = Yes)
name Name
sex Sex
age Age
sibsp Number of Siblings/Spouses Aboard
parch Number of Parents/Children Aboard
ticket Ticket Number
fare Passenger Fare (British pound)
cabin Cabin
embarked Port of Embarkation (C = Cherbourg; Q = Queenstown; S = Southampton)
boat Lifeboat
body Body Identification Number
home.dest Home/Destination
'''

df = pd.read_excel("Machine Learning with Python/K-Means/titanic.xls")
orig_df = pd.DataFrame.copy(df)
# print(df.head())
df.fillna(0, inplace=True)
df.drop(['body','name','boat','ticket'], axis=1, inplace=True)
# print(df.head())

def handle_non_numerical_data(df):
    """
Used to convert non-integer values to integer 
text_digit_vals dictionary stores the the values of text to a corresponding int value
Example: text_digit_vals = {0:"female", 1:"male"}
    """
    columns = df.columns.values
    for column in columns:
        text_digit_vals = {}
        def convert_to_int(val):
            return text_digit_vals[val]

        if df[column].dtype != np.int64 and df[column].dtype != np.float64:
            column_contents = df[column].values.tolist()
            unique_elements = set(column_contents)
            x = 0
            for unique in unique_elements:
                if unique not in text_digit_vals:
                    text_digit_vals[unique] = x
                    x+=1

            df[column] = list(map(convert_to_int, df[column]))

    return df

df = handle_non_numerical_data(df)
# print(df.head())


X = np.array(df.drop(['survived'], axis=1).astype(float))
X = preprocessing.scale(X)
y = np.array(df['survived'])

clf = MeanShift()

clf.fit(X)


labels = clf.labels_
cluster_centers = clf.cluster_centers_

orig_df['cluster_group'] = np.nan


for i in range(len(X)):
    orig_df['cluster_group'].iloc[i] = labels[i]

n_clusters = len(np.unique(labels))

survivor_rates = {}
for i in range(n_clusters):
    temp_df = orig_df[(orig_df['cluster_group']==float(i))]
    survivor_cluster = temp_df[ (temp_df['survived'] == 1) ]
    survivor_rate = len(survivor_cluster)/len(temp_df)
    survivor_rates[i] = survivor_rate
print(survivor_rates)



for i in range(len(survivor_rates)):
    cluster = orig_df[(orig_df['cluster_group']==i)]
    cluster_fc = cluster[ cluster['pclass']==1]
    print(cluster.describe())

    



