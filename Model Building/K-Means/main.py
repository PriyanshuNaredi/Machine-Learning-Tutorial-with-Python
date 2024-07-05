import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np

style.use('ggplot')


X = np.array([[1, 2],
              [1.5, 1.8],
              [5, 8],
              [8, 8],
              [1, 0.6],
              [9, 11],[1, 3],
                     [8, 9],
                     [0, 3],
                     [5, 4],
                     [6, 4]])


# plt.scatter(X[:, 0], X[:, 1], s=150)
# plt.show()


colors = 10*["g", "r", "c", "y"]


class K_Means():
    def __init__(self, k=2, tol=0.001, max_iterations=300):
        self.k = k
        self.tol = tol
        self.max_iterations = max_iterations

    def fit(self, data):
        self.centroid = {}
        for i in range(self.k):
            self.centroid[i] = data[i]
        for i in range(self.max_iterations):
            self.classifications = {}
            for i in range(self.k):
                self.classifications[i] = []
            for featureset in data:
                distances = [np.linalg.norm(
                    featureset-self.centroid[centroid]) for centroid in self.centroid]
                classification = distances.index(min(distances))
                self.classifications[classification].append(featureset)
            prev_centroids = dict(self.centroid)
            for classification in self.classifications:
                self.centroid[classification] = np.average(
                    self.classifications[classification], axis=0)
            optimized = True

            for c in self.centroid:
                org_centroid = prev_centroids[c]
                curr_centroid = self.centroid[c]
                if np.sum((curr_centroid - org_centroid) / org_centroid * 100.0) > self.tol:
                    optimized = False
            if optimized == True:
                break

    def predict(self, data):
        distances = [np.linalg.norm(
            data-self.centroid[centroid]) for centroid in self.centroid]
        classification = distances.index(min(distances))
        return classification


clf = K_Means()

clf.fit(X)

for centroid in clf.centroid:
    plt.scatter(clf.centroid[centroid][0], clf.centroid[centroid]
                [1], marker="o", color="k", s=150, linewidths=5)
for classification in clf.classifications:
    color = colors[classification]
    for featureset in clf.classifications[classification]:
        plt.scatter(featureset[0], featureset[1],
                    marker="x", color=color, s=150, linewidths=5)


# unknowns = np.array([[1, 3],
#                      [8, 9],
#                      [0, 3],
#                      [5, 4],
#                      [6, 4],])

# for unknown in unknowns:
#     classification = clf.predict(unknown)
#     plt.scatter(unknown[0], unknown[1], marker="*",
#                 color=colors[classification], s=150, linewidths=5)


plt.show()
