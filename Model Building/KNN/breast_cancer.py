import numpy as np
from collections import Counter
import warnings
import pandas as pd
import random

def k_nearest_neighbor(data, predict, k=3):
    if len(data) >= k:
        warnings.warn('k is set to a value less than total voting groups ')
        
    distances = []
    for group in data:
        for features in data[group]:
            # euclidean_distances = np.sqrt(np.sum((np.array(features)- np.array(predict))**2))
            euclidean_distances = np.linalg.norm(np.array(features) - np.array(predict))
            distances.append([euclidean_distances,group])
            
    votes = [i[1] for i in sorted(distances)[:k]]
    # print(Counter(votes).most_common(1))
    vote_result = Counter(votes).most_common(1)[0][0]
    confidence = Counter(votes).most_common(1)[0][1] / k
    
    # print(vote_result,confidence)
    return vote_result, confidence


accuracies = []
    
for i in range(25):
    df = pd.read_csv('Machine Learning with Python/KNN/breast-cancer-wisconsin.data')
    df.replace('?', -99999, inplace=True)
    df.drop(['id'], axis=1, inplace=True)

    full_data = df.astype(float).values.tolist()

    print(full_data[:5])
    print(20*'*')
    random.shuffle(full_data)
    print(full_data[:5])

    test_size = 0.4
    train_set = {2:[], 4:[]}
    test_set = {2:[],4:[]}

    train_data = full_data[:-int(test_size * len(full_data))]
    test_data = full_data[-int(test_size * len(full_data)) :] # last 20% of full data

    for i in train_data:
        """train_set[i[-1]] find last col in the data and fills train set in either 2 or 4 """
        train_set[i[-1]].append(i[:-1])

    for i in test_data:
        """train_set[i[-1]] find last col in the data and fills train set in either 2 or 4 """
        test_set[i[-1]].append(i[:-1])


    correct = 0
    total = 0

    for group in test_set:
        for data in test_set[group]:
            vote,confidence = k_nearest_neighbor(data=train_set, predict=data, k=5)
            if group == vote:
                correct += 1
            else:
                print(confidence)
            total += 1
            
            


    # print("accuracy:",correct/total)
    accuracies.append(correct/total)

print(sum(accuracies)/len(accuracies))

