import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
from collections import Counter
import warnings


style.use('fivethirtyeight')

dataset = {
    """ features """
    'k':[[1,2],[2,3],[3,1]],
    'r':[[6,5],[7,7],[8,6]]
}
new_feature = [5,7]


# [[plt.scatter(ii[0],ii[1],s=100) for ii in dataset[i]] for i in dataset]
# plt.show()


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
    print(Counter(votes).most_common(1))
    vote_result = Counter(votes).most_common(1)[0][0]
    
    return vote_result


result = k_nearest_neighbor(dataset, new_feature)
print(result)

















